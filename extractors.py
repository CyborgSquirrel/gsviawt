"""Feature extractors for wt_features.py.

An extractor turns one view of a render_objaverse.py h5 into one or more
`[L, P, D]` token volumes (L feature layers, P = (image_size / patch_size)^2
patches in row-major HxW order, D channels). wt_features.py instantiates one
from the `extractor` config node (`_target_: extractors.<Class>`), so a new
extractor is a new subclass of `FeatureExtractor` plus a `conf/extractor/*.yaml`.

Implemented:
    WTExtractor    World Tracing decoder tokens from a noised GT layered point
                   cloud (the tensor that feeds the SplitTransformerProjection head)
    MoGeExtractor  World Tracing's own image encoder (frozen MoGe-1 DINOv2 + final_project) on the RGB image
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from util import intrinsics_name, resize_depth_nearest
from wt_infer_layers import rgba_from_render

log = logging.getLogger(__name__)


@dataclass
class Row:
    """One output row: `tokens` is [L, P, D]; `columns` are extra per-row
    scalar datasets (e.g. noise_level, seed); `note` is appended to the log line."""

    tokens: torch.Tensor
    columns: dict = field(default_factory=dict)
    note: str = ""


class FeatureExtractor:
    """Base class. `name` names the default output file (`<input>.<name>_features.h5`)."""

    name = "features"

    def load(self, device: torch.device) -> None:
        """Load weights onto `device`. Called once before the first view."""
        raise NotImplementedError

    def extract_view(self, hf, index: int):
        """Yield the `Row`s for view `index` of the open render h5 `hf`. Row order
        within a view is the extractor's own and is what the output keeps."""
        raise NotImplementedError

    def attrs(self) -> dict:
        """Extra h5 attrs describing the model (image_size, patch_size, ...)."""
        return {}


def forward_fill_depth(depth):
    """depth: (H, W, L) float32, NaN or <= 0 = no hit. Returns a copy where a
    layer without a hit takes the layer above's depth (layer 0 is left as-is),
    plus the layer-0 hit mask (H, W)."""
    depth = np.where(depth > 0, depth, np.nan).astype(np.float32)
    valid = np.isfinite(depth[..., 0])
    for k in range(1, depth.shape[-1]):
        depth[..., k] = np.where(np.isfinite(depth[..., k]), depth[..., k], depth[..., k - 1])
    return depth, valid


def unproject_stack(depth, K):
    """depth: (H, W, L) with NaN where invalid; K: (3, 3). Returns (H, W, L, 3)
    camera-space XYZ (OpenCV axes), NaN where depth is NaN. Pixel (u, v) are the
    integer indices, same convention as debug_pointcloud.unproject_depth_peel."""
    height, width, _ = depth.shape
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float32)  # (H, W, 3)
    rays = pixels @ np.linalg.inv(K).T.astype(np.float32)  # (H, W, 3), unit depth
    return rays[:, :, None, :] * depth[..., None]


def noised_input(xyz, valid, t, norm_mean, norm_std):
    """xyz: (H, W, L, 3) camera-space, valid: (H, W). Returns
    (x_t [1, S, 3], valid_mask [1, S, 1]) with S = L*H*W, token order (l, h, w)
    matching denoise_geometry's `b c t h w -> b (t h w) c`."""
    num_layers = xyz.shape[2]
    x0 = torch.from_numpy(np.nan_to_num(xyz)).permute(3, 2, 0, 1)  # (3, L, H, W)
    x0 = (x0 - norm_mean[:, None, None, None]) / norm_std[:, None, None, None]
    eps = torch.randn_like(x0)
    x_t = (1.0 - t) * x0 + t * eps
    x_t = x_t.permute(1, 2, 3, 0).reshape(1, -1, 3)  # (1, L*H*W, 3)
    mask = torch.from_numpy(valid)[None].expand(num_layers, -1, -1)
    return x_t, mask.reshape(1, -1, 1).float()


class _StopForward(Exception):
    """Raised by the tap hook once the wanted block's output is captured."""


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


class WTExtractor(FeatureExtractor):
    """World Tracing decoder tokens, one forward pass per (view, seed, noise level) on
    a noised GT layered point cloud.

    For each view the render's `depth_peel` (H, W, L) is forward-filled across layers
    (WT trains on / emits a forward-filled stack, so a layer with no hit copies the
    layer above), unprojected with the depth intrinsics to camera-space XYZ (OpenCV:
    X right, Y down, Z forward), and z-score normalized. Flow-matching noise is then
    applied, x_t = (1 - t) * x0 + t * eps with t = `noise_level` (0 clean .. 1 pure
    noise). Pixels with no layer-0 hit are treated as invalid and replaced by pure
    noise inside the model (`invalid_fill_mode="noise"`), exactly as in
    wt_infer_layers.py. The model is called once at timestep t with `_context_only`,
    which makes `_forward_denoising` return `decoder_tokens` before the head runs.

    Rows go view, then seed, then noise level; all noise levels of a (view, seed)
    share the same sub-seed, i.e. the same eps. Columns: `noise_level`, `seed`.

    Only xyz_norm_mode="zscore" configs (r75b, r76) are supported: the release ships
    the inverse normalization only, and the forward z-score is trivial.
    """

    name = "wt"

    def __init__(
        self,
        config="r75b",
        ckpt="r75b",
        bf16_weights_hack=True,
        noise_level=0.5,
        seeds_per_view=1,
        seed=42,
        bg_color=(0, 0, 0),
        hard_alpha=False,
        block=None,
        shared_seed=False,
    ):
        self.config = config
        self.ckpt = ckpt
        if block is None:
            self.blocks = None
        else:
            self.blocks = [int(b) for b in block] if hasattr(block, "__iter__") else [int(block)]
            if not self.blocks or len(set(self.blocks)) != len(self.blocks):
                raise ValueError(f"block must be null, an int, or a non-empty list of distinct ints, got {block!r}")
        self.bf16_weights_hack = bool(bf16_weights_hack)
        levels = list(noise_level) if hasattr(noise_level, "__iter__") else [noise_level]
        self.noise_levels = [float(t) for t in levels]
        if not self.noise_levels:
            raise ValueError("noise_level must not be an empty list")
        for t in self.noise_levels:
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"noise_level must be in [0, 1], got {t}")
        if seeds_per_view < 1:
            raise ValueError(f"seeds_per_view must be >= 1, got {seeds_per_view}")
        self.seeds_per_view = int(seeds_per_view)
        self.seed = int(seed)
        self.shared_seed = bool(shared_seed)
        if self.shared_seed and self.seeds_per_view != 1:
            raise ValueError("shared_seed needs seeds_per_view=1 (every row would repeat the same noise)")
        self._shared = None
        self.bg_color = tuple(bg_color)
        self.hard_alpha = bool(hard_alpha)
        # One SeedSequence off the main seed; every (view, k) pair spawns the next
        # child of it, shared by all noise levels of that pair.
        self.seed_sequence = np.random.SeedSequence(self.seed)

    def load(self, device):
        from wt.checkpoint import build_model_and_load_ckpt
        from wt.inference import XYZ_MEAN, XYZ_STD, _bypass_activation_checkpointing

        self.device = device
        self.model, self.wt_cfg = build_model_and_load_ckpt(self.config, self.ckpt, device)
        mode = self.wt_cfg["inference_kwargs"]["xyz_norm_mode"]
        if mode != "zscore":
            raise NotImplementedError(
                f"{self.config}: xyz_norm_mode={mode!r}; only 'zscore' configs are supported "
                "(no forward transform in the release)."
            )
        if self.bf16_weights_hack:
            log.info("bf16_weights_hack: casting stored weights to bf16")
            self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.image_size = self.wt_cfg["image_size"]
        self.num_layers = self.wt_cfg["model_kwargs"]["num_layers"]
        self.norm_mean = torch.from_numpy(XYZ_MEAN)
        self.norm_std = torch.from_numpy(XYZ_STD)
        self._bypass = _bypass_activation_checkpointing
        self.num_blocks = len(self.model.net.decoder_blocks)
        self.attention_pattern = self.wt_cfg["model_kwargs"]["attention_pattern"]
        if self.blocks is not None:
            bad = [b for b in self.blocks if not 0 <= b < self.num_blocks]
            if bad:
                raise ValueError(f"block must be in [0, {self.num_blocks}) or null, got {bad}")
            if self.attention_pattern != "frame_ray_global":
                raise NotImplementedError(f"block taps need attention_pattern='frame_ray_global', got {self.attention_pattern!r}")
            log.info("tapping the output of decoder blocks %s / %d", self.blocks, self.num_blocks - 1)

    def extract_view(self, hf, index):
        from wt.data import preprocess_rgba_for_model

        image_size, num_layers, device = self.image_size, self.num_layers, self.device
        depth_peel = hf["depth_peel"][index]  # (H, W, L)
        K = hf[intrinsics_name(hf, "depth")][index].astype(np.float32)
        if depth_peel.shape[:2] != (image_size, image_size):
            # The model runs at a fixed square size. Layered depth has NaN holes and sharp
            # layer edges, so it is resampled by nearest neighbour (no blending of depths);
            # the RGB goes through WT's own preprocessing below, which resizes it.
            depth_peel, K = resize_depth_nearest(depth_peel, K, image_size)
        if depth_peel.shape[2] != num_layers:
            raise ValueError(
                f"view {index}: depth_peel has {depth_peel.shape[2]} layers, "
                f"{self.config} expects {num_layers}."
            )

        depth, valid = forward_fill_depth(depth_peel)
        xyz = unproject_stack(depth, K)

        rgb_t, _, _ = preprocess_rgba_for_model(
            rgba_from_render(hf, index, hard_alpha=self.hard_alpha),
            image_size=image_size,
            num_layers=num_layers,
            center_crop=False,
            bg_color=self.bg_color,
        )
        rgb_t = rgb_t.to(device)

        for _ in range(self.seeds_per_view):
            seed = self._next_seed()
            for noise_level in self.noise_levels:
                # re-seeded per noise level so every level sees the same eps
                # (and the same internal invalid-pixel fill noise)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                x_t, valid_mask = noised_input(xyz, valid, noise_level, self.norm_mean, self.norm_std)
                x_t, valid_mask = x_t.to(device), valid_mask.to(device)

                conditioning = {
                    "rgb": rgb_t[:, None].repeat(1, num_layers, 1, 1, 1),
                    "noise_height": image_size,
                    "noise_width": image_size,
                    "noise_nview": num_layers,
                    "batch_size": 1,
                    "valid_mask": valid_mask,
                    "invalid_fill_mode": "noise",
                    "_context_only": True,  # return decoder tokens, skip the head
                }
                t = torch.full((1,), noise_level, device=device)
                with torch.no_grad(), _autocast(device), self._bypass(self.model):
                    if self.blocks is None:
                        outputs = [(None, self.model(x_t, t, conditioning)["decoder_tokens"][0])]  # (L, P, D)
                    else:
                        taps = self._block_outputs(x_t, t, conditioning)
                        outputs = [(b, taps[b]) for b in self.blocks]
                for block, tokens in outputs:
                    columns = {"noise_level": noise_level, "seed": seed}
                    if block is not None:
                        columns["block"] = block
                    yield Row(
                        tokens,
                        columns,
                        f"noise_level={noise_level:g} seed={seed}, {int(valid.sum())}/{valid.size} valid px"
                        + ("" if block is None else f", block {block}"),
                    )

    def _next_seed(self):
        """Sub-seed of the next (view, draw) pair: the next child of the main SeedSequence, or,
        with shared_seed, always the first child (the same noise for every row)."""
        if self.shared_seed and self._shared is not None:
            return self._shared
        # (>> 1 keeps it inside torch.manual_seed's / int64's range.)
        seed = int(self.seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
        if self.shared_seed:
            self._shared = seed
        return seed

    def _block_outputs(self, x_t, t, conditioning):
        """{block: (L, P, D)} outputs of the decoder blocks in `self.blocks` from ONE forward pass:
        forward hooks capture them and the last one aborts the pass, so later blocks never run.
        Blocks see frame-wise [B*L, P, D], ray-wise [B*P, L, D] or global [B, L*P, D] tokens
        (frame_ray_global cycle); each capture is put back into [B, L, P, D] the way
        decode_to_output_tokens does."""
        captured = {}
        last = max(self.blocks)

        def make_hook(index):
            def hook(module, args, output):
                captured[index] = output
                if index == last:
                    raise _StopForward
            return hook

        handles = [self.model.net.decoder_blocks[b].register_forward_hook(make_hook(b)) for b in self.blocks]
        try:
            self.model(x_t, t, conditioning)
        except _StopForward:
            pass
        finally:
            for handle in handles:
                handle.remove()
        missing = [b for b in self.blocks if b not in captured]
        if missing:
            raise RuntimeError(f"decoder blocks {missing} never ran")
        result = {}
        for b, out in captured.items():
            num_layers, num_patches = self.num_layers, out.numel() // (self.num_layers * out.shape[-1])
            if ("frame", "ray", "global")[b % 3] == "ray":
                out = out.reshape(1, num_patches, num_layers, -1).permute(0, 2, 1, 3)
            else:
                out = out.reshape(1, num_layers, num_patches, -1)
            result[b] = out[0]
        return result

    def attrs(self):
        return {
            "wt_config": self.config,
            "ckpt": self.ckpt,
            "block": -1 if self.blocks is None else np.array(self.blocks, dtype=np.int64),
            "shared_seed": self.shared_seed,
            "main_seed": self.seed,
            "image_size": self.image_size,
            "patch_size": self.wt_cfg["model_kwargs"]["patch_size"],
        }


class MoGeExtractor(FeatureExtractor):
    """World Tracing's image encoder on its own: the frozen MoGe-1 DINOv2 ViT-L backbone
    plus WT's trained `final_project`, with the weights taken from the WT checkpoint
    (`net.encoder.*`; EMA state if present, like `load_checkpoint`), so the features are
    exactly what WT's decoder receives as image tokens, before its fusion step
    (`fuse_context_tokens`: layer-norm, x0.3, concat with the geometry tokens).

    Mirrors `MultilayerBackbone.encode_image_moge`: the render composited onto `bg_color`
    at the WT config's `image_size` (same preprocessing as WTExtractor), normalized with
    the encoder's image mean/std, `get_intermediate_layers` of the last 4 blocks
    concatenated to 4096 channels, then `final_project` -> decoder_embed_dim (1536 for
    r75b), under fp16 autocast as WT does for a frozen encoder. MoGe sees RGB only: no
    depth, no noise, one deterministic row of shape [1, P, D] per view (a single feature
    layer -- replicating it across Gaussian layers is the training head's job).
    """

    name = "moge"

    def __init__(self, config="r75b", ckpt="r75b", weights=None, project=True, num_blocks=4, bf16_weights_hack=True, bg_color=(0, 0, 0), hard_alpha=False):
        from wt.checkpoint import CONFIGS

        if config not in CONFIGS:
            raise ValueError(f"Unknown config {config!r}; available: {sorted(CONFIGS)}")
        self.config = config
        self.ckpt = ckpt
        self.weights = weights
        self.project = bool(project)
        if weights is not None and project:
            raise ValueError("`weights` (the original MoGe) has no final_project; set project=false")
        self.bf16_weights_hack = bool(bf16_weights_hack)
        self.bg_color = tuple(bg_color)
        self.hard_alpha = bool(hard_alpha)
        self.image_size = CONFIGS[config]["image_size"]
        self.patch_size = 14  # model_config.MOGE_PATCH_SIZE
        self.num_blocks = int(num_blocks)  # WT's `encoder.intermediate_layers` = 4: the last 4 DINOv2 blocks
        if project and self.num_blocks != 4:
            raise ValueError("WT's final_project takes the 4-block concat (4096 channels); set project=false for num_blocks != 4")

    def load(self, device):
        from wt._core.arch.models import config as model_config
        from wt._core.vendor.moge.moge_model import MoGeModel
        from wt.checkpoint import resolve_ckpt_path

        self.device = device
        if self.weights == "dinov2_vitl14":
            # Meta's original DINOv2 ViT-L/14 (no MoGe fine-tuning), same last-4-block tap;
            # ImageNet mean/std like the MoGe encoders
            from wt._core.vendor.moge.dinov2 import hub

            self.backbone, self.final_project = hub.dinov2_vitl14(pretrained=True), None
            self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        elif self.weights is not None:
            # the original pretrained MoGe-1 (a local model.pt or HF repo id): backbone only,
            # no WT final_project, so the output is the raw 4-block concat (4096 channels)
            import os

            path = self.weights
            if not os.path.exists(path):
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(repo_id=self.weights, filename="model.pt")
            moge_ckpt = torch.load(path, map_location="cpu", weights_only=True)
            moge = MoGeModel(**moge_ckpt["model_config"])
            moge.load_state_dict(moge_ckpt["model"])
            self.backbone, self.final_project = moge.backbone, None
            self.image_mean, self.image_std = moge.image_mean, moge.image_std
            del moge, moge_ckpt
        else:
            self._load_wt_encoder(MoGeModel, model_config, resolve_ckpt_path)
            if not self.project:
                self.final_project = None
        modules = [m for m in (self.backbone, self.final_project) if m is not None]
        if self.bf16_weights_hack:
            log.info("bf16_weights_hack: casting stored weights to bf16")
            modules = [m.to(torch.bfloat16) for m in modules]
            self.image_mean, self.image_std = self.image_mean.to(torch.bfloat16), self.image_std.to(torch.bfloat16)
        modules = [m.to(device).eval() for m in modules]
        self.backbone = modules[0]
        self.final_project = modules[1] if len(modules) > 1 else None
        log.info(
            "MoGe encoder (%s): last %d blocks -> %d channels",
            self.weights or f"WT {self.config}/{self.ckpt}", self.num_blocks, 1024 * self.num_blocks if self.final_project is None else self.final_project.out_features,
        )

    def _load_wt_encoder(self, MoGeModel, model_config, resolve_ckpt_path):
        checkpoint = torch.load(resolve_ckpt_path(self.ckpt), map_location="cpu", weights_only=False, mmap=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            ema = checkpoint.get("ema_state_dict")
            state = ema if ema is not None else checkpoint["model_state_dict"]
        else:
            state = checkpoint
        (marker,) = [k for k in state if k.endswith("encoder.final_project.weight")]
        prefix = marker[: -len("final_project.weight")]
        enc = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

        # the same construction as MultilayerBackbone.build_encoder (inference_mode)
        self.backbone = MoGeModel(**model_config.MOGE_CONFIG_FA3).backbone
        self.backbone.load_state_dict(
            {k: v for k, v in enc.items() if not k.startswith("final_project.") and k not in ("image_mean", "image_std")}
        )
        self.final_project = torch.nn.Linear(4096, enc["final_project.weight"].shape[0])
        self.final_project.load_state_dict({"weight": enc["final_project.weight"], "bias": enc["final_project.bias"]})
        self.image_mean, self.image_std = enc["image_mean"], enc["image_std"]
        del checkpoint, state, enc

    def extract_view(self, hf, index):
        from wt.data import preprocess_rgba_for_model

        rgb_t, _, _ = preprocess_rgba_for_model(
            rgba_from_render(hf, index, hard_alpha=self.hard_alpha),
            image_size=self.image_size,
            num_layers=1,
            center_crop=False,
            bg_color=self.bg_color,
        )  # [1, 3, S, S] in [0, 1]
        img = rgb_t.to(self.device)
        mean = self.image_mean.to(dtype=img.dtype, device=img.device)
        std = self.image_std.to(dtype=img.dtype, device=img.device)
        grid = self.image_size // self.patch_size
        x = F.interpolate((img - mean) / std, (grid * self.patch_size,) * 2, mode="bilinear")
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda")
        with torch.no_grad(), autocast:
            blocks = self.backbone.get_intermediate_layers(x, self.num_blocks)  # 4 x [1, P, 1024]
            tokens = torch.cat(blocks, -1)  # [1, P, 1024 * num_blocks]
            if self.final_project is not None:
                tokens = self.final_project(tokens)  # [1, P, D]
        yield Row(tokens.float(), {}, f"last {self.num_blocks} blocks" + (" -> final_project" if self.final_project is not None else ""))

    def attrs(self):
        return {
            "wt_config": self.config,
            "ckpt": self.ckpt,
            "moge_weights": str(self.weights),
            "projected": self.final_project is not None,
            "num_blocks": self.num_blocks,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        }
