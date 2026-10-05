#!/usr/bin/env python3
"""Run World Tracing ONE forward pass per view on a noised GT layered point
cloud and save the decoder token volume -- the tensor that would be handed to
the prediction head (`SplitTransformerProjection`), with the head itself
skipped.

For each explicit view in `views`:
  * the render's `depth_peel` (H, W, L) is forward-filled across layers (WT
    trains on / emits a forward-filled stack, so a layer with no hit copies
    the layer above), unprojected with the depth intrinsics to camera-space
    XYZ (OpenCV: X right, Y down, Z forward), and z-score normalized;
  * flow-matching noise is applied, x_t = (1 - t) * x0 + t * eps with
    t = `noise_level` (0 clean .. 1 pure noise), eps ~ N(0, 1);
  * pixels with no layer-0 hit are treated as invalid and replaced by pure
    noise inside the model (`invalid_fill_mode="noise"`), exactly as in
    wt_infer_layers.py;
  * the model is called once at timestep t with `_context_only`, which makes
    `_forward_denoising` return `decoder_tokens` before the head runs.

Output h5:
    features     (N, L, P, D)  decoder tokens, in the dtype the model returned
                               (bfloat16 is stored as its uint16 bit pattern,
                               since h5py/numpy have no bf16 -- see the
                               `torch_dtype` attr). gzip, one chunk per view.
    source_h5    attr          absolute path of the render h5 the views come from
    mesh_index   (N,) int64    the source h5's `mesh_index` of each row's view
                               (which mesh it is; `mesh_paths` in the source h5)
    view_indices (N,) int64    row index into the input h5 (each view repeated
                               `seeds_per_view` times; N = len(views) * seeds_per_view,
                               view-major)
    noise_level  (N,) float32  t used for that view
    seed         (N,) int64    sub-seed used to noise that view: the n-th value
                               generated from SeedSequence(`seed`), n = view position *
                               seeds_per_view + k; the main seed is in the `main_seed` attr
    attrs: config_json, input, config, ckpt, features layout/dtype.

Only xyz_norm_mode="zscore" configs (r75b, r76) are supported: the release
ships the inverse normalization only, and the forward z-score is trivial.

Config is Hydra (`conf/wt_features.yaml`); run in the container venv:

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python \\
        wt_features.py input=/app/bla/lite.h5 'views=[0,5,12]' noise_level=0.5
"""

import json
import logging
import os
import sys

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import LazyDataset, intrinsics_name  # noqa: E402
from wt_infer_layers import rgba_from_render  # noqa: E402

log = logging.getLogger(__name__)


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


def to_h5_array(tokens):
    """-> (numpy array, torch dtype string). bfloat16 has no numpy dtype, so it
    is stored as its raw uint16 bit pattern."""
    dtype = str(tokens.dtype)
    if tokens.dtype == torch.bfloat16:
        return tokens.contiguous().view(torch.int16).cpu().numpy().view(np.uint16), dtype
    return tokens.cpu().numpy(), dtype


@hydra.main(version_base=None, config_path="conf", config_name="wt_features")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from wt.checkpoint import build_model_and_load_ckpt
    from wt.data import preprocess_rgba_for_model
    from wt.inference import XYZ_MEAN, XYZ_STD, _bypass_activation_checkpointing

    if not 0.0 <= cfg.noise_level <= 1.0:
        raise ValueError(f"noise_level must be in [0, 1], got {cfg.noise_level}")
    views = [int(v) for v in cfg.views]
    if cfg.seeds_per_view < 1:
        raise ValueError(f"seeds_per_view must be >= 1, got {cfg.seeds_per_view}")
    output = cfg.output or f"{cfg.input}.wt_features.h5"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, wt_cfg = build_model_and_load_ckpt(cfg.config, cfg.ckpt, device)
    if wt_cfg["inference_kwargs"]["xyz_norm_mode"] != "zscore":
        raise NotImplementedError(
            f"{cfg.config}: xyz_norm_mode={wt_cfg['inference_kwargs']['xyz_norm_mode']!r}; "
            "only 'zscore' configs are supported (no forward transform in the release)."
        )
    if cfg.bf16_weights_hack:
        log.info("--bf16_weights_hack: casting stored weights to bf16")
        model = model.to(torch.bfloat16)
    model.eval()

    image_size = wt_cfg["image_size"]
    num_layers = wt_cfg["model_kwargs"]["num_layers"]
    norm_mean = torch.from_numpy(XYZ_MEAN)
    norm_std = torch.from_numpy(XYZ_STD)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    def one_chunk_per_view(ds, *, shape, dtype):
        ds.dataset_kwargs = {"chunks": (1, *shape), **ds.dataset_kwargs}

    # One SeedSequence off the main seed; row n (= view_pos * seeds_per_view + k)
    # gets the n-th draw.
    # (>> 1 keeps it inside torch.manual_seed's / int64's range.)
    seeds_per_view = int(cfg.seeds_per_view)
    seeds = [
        int(x >> np.uint64(1))
        for x in np.random.SeedSequence(cfg.seed).generate_state(
            len(views) * seeds_per_view, dtype=np.uint64
        )
    ]
    feat_dtype = None
    with (
        h5py.File(cfg.input, "r") as hf,
        h5py.File(output, "w") as out,
        LazyDataset(
            out,
            "features",
            dataset_kwargs=dict(compression="gzip", compression_opts=int(cfg.compression_level)),
            init_hook=one_chunk_per_view,
        ) as features,
    ):
        k_name = intrinsics_name(hf, "depth")
        for i, index in enumerate(views):
            depth_peel = hf["depth_peel"][index]  # (H, W, L)
            if depth_peel.shape[:2] != (image_size, image_size):
                raise ValueError(
                    f"view {index}: depth_peel is {depth_peel.shape[:2]}, but {cfg.config} "
                    f"runs at {image_size}x{image_size}; re-render or resize first."
                )
            if depth_peel.shape[2] != num_layers:
                raise ValueError(
                    f"view {index}: depth_peel has {depth_peel.shape[2]} layers, "
                    f"{cfg.config} expects {num_layers}."
                )
            K = hf[k_name][index].astype(np.float32)

            depth, valid = forward_fill_depth(depth_peel)
            xyz = unproject_stack(depth, K)

            rgb_t, _, _ = preprocess_rgba_for_model(
                rgba_from_render(hf, index, hard_alpha=cfg.hard_alpha),
                image_size=image_size,
                num_layers=num_layers,
                center_crop=False,
                bg_color=tuple(cfg.bg_color),
            )
            rgb_t = rgb_t.to(device)

            for s in range(seeds_per_view):
                n = i * seeds_per_view + s
                seed = seeds[n]
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                x_t, valid_mask = noised_input(xyz, valid, cfg.noise_level, norm_mean, norm_std)
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
                t = torch.full((1,), cfg.noise_level, device=device)
                with (
                    torch.no_grad(),
                    autocast_ctx,
                    _bypass_activation_checkpointing(model),
                ):
                    tokens = model(x_t, t, conditioning)["decoder_tokens"][0]  # (L, P, D)

                arr, torch_dtype = to_h5_array(tokens)
                features.append(arr)
                if n == 0:
                    feat_dtype = torch_dtype
                    features.dataset.attrs["layout"] = "N, L (layers), P (patches, row-major HxW grid), D"
                    features.dataset.attrs["torch_dtype"] = torch_dtype
                    features.dataset.attrs["stored_as_uint16_bits"] = torch_dtype == "torch.bfloat16"
                log.info(
                    "view %d: tokens %s %s, noise_level=%g seed=%d, %d/%d valid px",
                    index, tuple(tokens.shape), torch_dtype, cfg.noise_level, seed,
                    int(valid.sum()), valid.size,
                )

        mesh_of_view = {v: int(hf["mesh_index"][v]) for v in views}
        out.create_dataset(
            "mesh_index", data=np.repeat(np.array([mesh_of_view[v] for v in views], dtype=np.int64), seeds_per_view)
        )
        out.create_dataset("view_indices", data=np.repeat(np.array(views, dtype=np.int64), seeds_per_view))
        out.create_dataset("noise_level", data=np.full(len(seeds), cfg.noise_level, np.float32))
        out.create_dataset("seed", data=np.array(seeds, dtype=np.int64))
        out.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
        out.attrs["main_seed"] = int(cfg.seed)
        out.attrs["input"] = str(cfg.input)
        out.attrs["source_h5"] = os.path.abspath(str(cfg.input))
        out.attrs["wt_config"] = cfg.config
        out.attrs["ckpt"] = cfg.ckpt
        out.attrs["image_size"] = image_size
        out.attrs["patch_size"] = wt_cfg["model_kwargs"]["patch_size"]
        out.attrs["torch_dtype"] = feat_dtype

    log.info("wrote %s", output)


if __name__ == "__main__":
    main()
