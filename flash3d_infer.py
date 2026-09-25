#!/usr/bin/env python3
"""Single-image inference with the pretrained RealEstate10K "v1" Flash3D model.

Flash3D ("Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a
Single Image", 3DV 2025) predicts a layered 3D Gaussian Splat scene from one
image in a single forward pass: a frozen depth network (UniDepth) gives a
metric first-layer depth, and a ResNet head predicts per-pixel Gaussian
parameters for that layer plus one or more occluded layers behind it.

This is a smoke-test wrapper, vendored from two upstream sources:
  - github.com/eldar/flash3d (the training/eval repo) for the model config
    shape, and
  - the official HF Space huggingface.co/spaces/szymanowiczs/flash3d (its
    `flash3d/` package: `networks/`, `util/`, `unidepth/`) as the actual
    single-image inference reference -- the base repo has no demo script of
    its own, only `evaluate.py`, which needs a full RealEstate10K dataloader.
The Space's `unidepth/` is a self-contained vendored copy of UniDepth v1, so
this needs no `torch.hub.load` (the base repo's `unidepth_encoder.py` does,
which would pull whatever UniDepth's `main` branch looks like today -- that
repo has since moved to v2 and would silently mismatch this checkpoint).

The checkpoint + its config are pulled from
https://huggingface.co/einsafutdinov/flash3d on first run and cached under
mounts/huggingface. Deliberately v1 (not v2): the repo only publishes a
config yaml next to model_re10k_v1.pth, and v1 is what the official demo
actually runs.

Flash3D reconstructs a forward-facing scene from one camera position, not an
object turntable -- there's no geometry for anything not visible in the
input photo. So besides the .ply (for orbit_video.py, mainly useful for
eyeballing the point cloud from odd angles), this renders a small side-to-side
look-around from the source view instead of an orbit.

Usage:
    python flash3d_infer.py image=/path/to/room.jpg
"""

import logging
import sys
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

_FLASH3D_ROOT = Path(__file__).resolve().parent / "flash3d"
if str(_FLASH3D_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLASH3D_ROOT))

from networks.gaussian_predictor import GaussianPredictor  # noqa: E402
from networks.gauss_util import focal2fov, getProjectionMatrix, render_predicted  # noqa: E402

log = logging.getLogger(__name__)

_HF_REPO = "einsafutdinov/flash3d"
_CKPT = "model_re10k_v1.pth"
_CFG = "config_re10k_v1.yaml"

# Maps flash3d's native per-pixel frame (predicted by backprojecting UniDepth
# depth: x right, y down, z forward -- OpenCV/COLMAP convention) so that +Z
# becomes the world-up axis orbit_video.py assumes. Only used for the .ply
# export -- the preview render loop below stays in the native frame, since
# its own camera path is expressed there too.
_ZUP_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def _load_model(device: torch.device):
    cfg_path = hf_hub_download(repo_id=_HF_REPO, filename=_CFG)
    ckpt_path = hf_hub_download(repo_id=_HF_REPO, filename=_CKPT)
    model_cfg = OmegaConf.load(cfg_path)

    model = GaussianPredictor(model_cfg)
    model.load_model(ckpt_path, device=str(device))
    model.to(device).eval()
    return model, model_cfg


def _preprocess(image_path: Path, model_cfg: DictConfig) -> torch.Tensor:
    """Resize to the model's native (256, 384) and zero-pad by
    dataset.pad_border_aug on every side -- the fixed-size path of the
    official demo's own preprocess() (app.py, dynamic_size=False). The
    padding isn't cosmetic: UniDepth's intrinsics estimate and gauss_means's
    backprojection both run on this padded grid.
    """
    image = Image.open(image_path).convert("RGB")
    h, w = model_cfg.dataset.height, model_cfg.dataset.width
    image = image.resize((w, h), Image.BICUBIC)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    pad = model_cfg.dataset.pad_border_aug
    tensor = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), mode="constant", value=0.0)
    return tensor.unsqueeze(0)


def _extract_gaussians(outputs: dict, model_cfg: DictConfig, crop: bool):
    """Flatten GaussianPredictor's per-pixel, per-layer (H, W) grids into one
    flat per-Gaussian dict, in the network's native camera frame (raw,
    already-activated opacity/scaling/rotation -- see gaussian_decoder.py).

    `crop=True` drops the pad_border_aug border (what the .ply export wants,
    matching the official demo's own export path). `crop=False` keeps the
    full padded grid, which is what the K_src intrinsics were predicted for
    -- needed so the preview render loop's camera math lines up with the
    Gaussians it's rendering. Returns (reconstruction, (h, w)).
    """
    num_gauss = model_cfg.model.gaussians_per_pixel
    h, w, pad = model_cfg.dataset.height, model_cfg.dataset.width, model_cfg.dataset.pad_border_aug
    H, W = h + 2 * pad, w + 2 * pad
    if not crop:
        h, w, pad = H, W, 0

    def crop_flat(t):
        t = rearrange(t, "b c (H W) -> b c H W", H=H, W=W)
        return rearrange(t[..., pad : pad + h, pad : pad + w], "b c h w -> b c (h w)")

    def crop_hw(t):
        return t[..., pad : pad + h, pad : pad + w]

    means = rearrange(crop_flat(outputs[("gauss_means", 0, 0)]), "(b v) c n -> b (v n) c", v=num_gauss)[0, :, :3]
    scaling = rearrange(crop_hw(outputs[("gauss_scaling", 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    rotation = rearrange(crop_hw(outputs[("gauss_rotation", 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    opacity = rearrange(crop_hw(outputs[("gauss_opacity", 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    features_dc = rearrange(
        crop_hw(outputs[("gauss_features_dc", 0, 0)]), "(b v) c h w -> b (v h w) 1 c", v=num_gauss
    )[0]
    reconstruction = {"xyz": means, "scaling": scaling, "rotation": rotation, "opacity": opacity,
                       "features_dc": features_dc}
    if model_cfg.model.max_sh_degree > 0:
        reconstruction["features_rest"] = rearrange(
            crop_hw(outputs[("gauss_features_rest", 0, 0)]), "(b v) (sh c) h w -> b (v h w) sh c", c=3, v=num_gauss
        )[0]
    return reconstruction, (h, w)


def _construct_list_of_attributes(num_rest: int) -> list:
    """Standard INRIA-format 3DGS .ply field order -- matches orbit_video.py's
    load_ply and splatter_image_infer.py's _export_ply."""
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    attributes += [f"f_dc_{i}" for i in range(3)]
    attributes += [f"f_rest_{i}" for i in range(num_rest)]
    attributes.append("opacity")
    attributes += [f"scale_{i}" for i in range(3)]
    attributes += [f"rot_{i}" for i in range(4)]
    return attributes


def _export_ply(reconstruction: dict, path: str, min_opacity: float) -> None:
    """Write flash3d's Gaussians as a standard INRIA-format 3DGS .ply,
    remapped into orbit_video.py's Z-up world convention.

    The official demo's own exporter (util/vis3d.py, vendored here too)
    writes opacity post-sigmoid, which is right for its own Model3D viewer
    but wrong for orbit_video.py's load_ply, which always re-applies
    sigmoid on read -- so opacity must be written as a logit here instead
    (same fix as splatter_image_infer.py's _export_ply, for the analogous
    upstream mismatch on that model). Rotation channel order (xyzw, not the
    wxyz the .ply schema wants) is confirmed against that same official
    export code, which feeds gauss_rotation straight into scipy's
    `from_quat` (xyzw by convention) with no reordering first.
    """
    opacity_raw = reconstruction["opacity"][:, 0]
    valid = opacity_raw > min_opacity
    xyz = reconstruction["xyz"][valid].detach().cpu().numpy()
    scaling = reconstruction["scaling"][valid].detach().cpu().numpy()
    rotation_xyzw = reconstruction["rotation"][valid].detach().cpu().numpy()
    opacity = opacity_raw[valid].detach().cpu().numpy()
    f_dc = reconstruction["features_dc"][valid, 0].detach().cpu().numpy()
    f_rest = reconstruction.get("features_rest")
    f_rest = (
        f_rest[valid].detach().cpu().reshape(xyz.shape[0], -1).numpy()
        if f_rest is not None
        else np.zeros((xyz.shape[0], 0), dtype=np.float32)
    )

    xyz = xyz @ _ZUP_ROTATION.T
    rot_mats = _ZUP_ROTATION[None] @ Rotation.from_quat(rotation_xyzw).as_matrix()
    rotation_xyzw = Rotation.from_matrix(rot_mats).as_quat()
    rotation_wxyz = rotation_xyzw[:, [3, 0, 1, 2]]

    log_scale = np.log(np.clip(scaling, 1e-8, None))
    p = np.clip(opacity, 1e-6, 1 - 1e-6)
    opacity_logit = np.log(p / (1 - p))[:, None]
    normals = np.zeros_like(xyz)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dtype_full = [(a, "f4") for a in _construct_list_of_attributes(f_rest.shape[1])]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacity_logit, log_scale, rotation_wxyz), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def _wiggle_cameras(num_views: int, max_yaw_deg: float) -> np.ndarray:
    """Camera-to-world poses for a small side-to-side pan around the
    identity source-camera pose. Flash3D only reconstructs the layer(s) of
    geometry behind what the input photo actually shows, so an object-style
    360-degree orbit would sweep past into empty space; a look-around
    wiggle is the standard preview motion for this kind of forward-facing,
    few-layer reconstruction."""
    yaw = np.radians(max_yaw_deg) * np.sin(np.linspace(0.0, 2 * np.pi, num_views, endpoint=False))
    c2w = np.zeros((num_views, 4, 4), dtype=np.float32)
    c2w[:, 3, 3] = 1.0
    cos, sin = np.cos(yaw), np.sin(yaw)
    c2w[:, 0, 0], c2w[:, 0, 2] = cos, sin
    c2w[:, 1, 1] = 1.0
    c2w[:, 2, 0], c2w[:, 2, 2] = -sin, cos
    return c2w


def _render_loop(reconstruction: dict, model_cfg: DictConfig, K: torch.Tensor, image_hw: tuple,
                  cfg: DictConfig, device: torch.device):
    H, W = image_hw
    fovX = focal2fov(K[0, 0].item(), W)
    fovY = focal2fov(K[1, 1].item(), H)
    proj_mtrx = getProjectionMatrix(cfg.znear, cfg.zfar, fovX, fovY).transpose(0, 1).to(device)
    background = torch.tensor(list(model_cfg.model.bg_colour), dtype=torch.float32, device=device)
    pc = {k: v.contiguous().float().to(device) for k, v in reconstruction.items()}

    frames = []
    for c2w in _wiggle_cameras(cfg.num_loop_views, cfg.max_yaw_deg):
        world_view_transform = torch.from_numpy(c2w).inverse().transpose(0, 1).float().to(device)
        camera_center = (-world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)).float()
        full_proj_transform = (world_view_transform @ proj_mtrx).float()
        render = render_predicted(
            model_cfg, pc, world_view_transform, full_proj_transform, proj_mtrx,
            camera_center, (fovX, fovY), (H, W), background, model_cfg.model.max_sh_degree,
        )["render"]
        frames.append(torch.clamp(render * 255, 0, 255).byte().permute(1, 2, 0).cpu().numpy())
    return frames


@hydra.main(version_base=None, config_path="conf", config_name="flash3d_infer")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)

    model, model_cfg = _load_model(device)
    log.info("Loaded flash3d re10k_v1 checkpoint (%d params)", sum(p.numel() for p in model.parameters()))

    image = _preprocess(Path(cfg.image), model_cfg).to(device)
    outputs = model({("color_aug", 0, 0): image})

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.export_ply:
        reconstruction_export, _ = _extract_gaussians(outputs, model_cfg, crop=True)
        _export_ply(reconstruction_export, str(out_dir / "reconstruction.ply"), cfg.min_opacity)
        log.info("Wrote %s", out_dir / "reconstruction.ply")

    if cfg.export_loop:
        reconstruction_render, image_hw = _extract_gaussians(outputs, model_cfg, crop=False)
        K = outputs[("K_src", 0, 0)][0]
        frames = _render_loop(reconstruction_render, model_cfg, K, image_hw, cfg, device)
        loop_path = out_dir / "loop.mp4"
        imageio.mimsave(loop_path, frames, fps=cfg.fps)
        log.info("Wrote %s", loop_path)


if __name__ == "__main__":
    main()
