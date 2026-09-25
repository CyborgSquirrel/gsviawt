#!/usr/bin/env python3
"""Single-image inference with the pretrained "cars" Splatter Image model.

Splatter Image ("Splatter Image: Ultra-Fast Single-View 3D Reconstruction",
CVPR 2024) predicts a 3D Gaussian Splat from one image in a single forward
pass. This is a smoke-test wrapper around the upstream repo, vendored
verbatim under splatter-image/ (github.com/szymanowiczs/splatter-image
@ 78a6ad0) -- see splatter-image/README.md for the original project.

The checkpoint + its config are pulled from
https://huggingface.co/szymanowiczs/splatter-image-v1 on first run and
cached under mounts/huggingface (same HF cache the rest of this repo uses).

Usage:
    python splatter_image_infer.py image=/path/to/car.jpg
"""

import logging
import sys
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
import torchvision
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from PIL import Image

_SPLATTER_IMAGE_ROOT = Path(__file__).resolve().parent / "splatter-image"
if str(_SPLATTER_IMAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPLATTER_IMAGE_ROOT))

from plyfile import PlyData, PlyElement  # noqa: E402

from gaussian_renderer import render_predicted  # noqa: E402
from scene.gaussian_predictor import GaussianSplatPredictor  # noqa: E402
from utils.app_utils import (  # noqa: E402
    construct_list_of_attributes,
    remove_background,
    resize_foreground,
    resize_to_128,
    set_white_background,
    to_tensor,
)
from utils.camera_utils import get_loop_cameras  # noqa: E402
from utils.general_utils import matrix_to_quaternion  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402

log = logging.getLogger(__name__)

_HF_REPO = "szymanowiczs/splatter-image-v1"
_DATASET = "cars"  # the only checkpoint this script has been set up for so far


def _load_model(device: torch.device):
    cfg_path = hf_hub_download(repo_id=_HF_REPO, filename=f"config_{_DATASET}.yaml")
    ckpt_path = hf_hub_download(repo_id=_HF_REPO, filename=f"model_{_DATASET}.pth")
    model_cfg = OmegaConf.load(cfg_path)

    model = GaussianSplatPredictor(model_cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, model_cfg


def _preprocess(image_path: Path, remove_bg: bool, foreground_ratio: float) -> torch.Tensor:
    image = Image.open(image_path).convert("RGBA")
    if remove_bg:
        import rembg

        image = remove_background(image, rembg.new_session())
        image = resize_foreground(image, foreground_ratio)
    if image.mode == "RGBA":
        image = set_white_background(image)
    image = resize_to_128(image)
    return to_tensor(np.array(image))


def _source_camera(radius: float):
    """Canonical single input view: front-on (zero elevation), `radius` from the origin."""
    c2w = get_loop_cameras(num_imgs_in_loop=1, radius=radius, max_elevation=0.0)[0]
    c2w = torch.from_numpy(c2w).transpose(0, 1).unsqueeze(0)  # (1, 4, 4)
    quat = matrix_to_quaternion(c2w[0, :3, :3].transpose(0, 1))
    # view_to_world_transforms: (1 example, 1 input view, 4, 4); quats: (1, 1, 4)
    return c2w.unsqueeze(0), quat.unsqueeze(0).unsqueeze(0)


def _target_cameras(model_cfg: DictConfig, radius: float, num_views: int):
    """Turntable loop around the object, using the model's own znear/zfar/fov
    (unlike upstream gradio_app.py, which hardcodes the multi-category model's
    camera range -- that mismatches the cars checkpoint's znear=0.8/zfar=1.8)."""
    projection_matrix = getProjectionMatrix(
        znear=model_cfg.data.znear,
        zfar=model_cfg.data.zfar,
        fovX=model_cfg.data.fov * np.pi / 180,
        fovY=model_cfg.data.fov * np.pi / 180,
    ).transpose(0, 1)

    loop = get_loop_cameras(num_imgs_in_loop=num_views, radius=radius, max_elevation=np.pi / 4, elevation_freq=1.5)

    world_view_transforms, camera_centers = [], []
    for c2w in loop:
        world_view_transform = torch.from_numpy(c2w).inverse().transpose(0, 1)
        camera_center = world_view_transform.inverse()[3, :3].clone()
        world_view_transforms.append(world_view_transform)
        camera_centers.append(camera_center)

    world_view_transforms = torch.stack(world_view_transforms)
    camera_centers = torch.stack(camera_centers)
    full_proj_transforms = world_view_transforms.bmm(
        projection_matrix.unsqueeze(0).expand(world_view_transforms.shape[0], 4, 4)
    )
    return world_view_transforms, full_proj_transforms, camera_centers


def _export_ply(reconstruction: dict, path: str) -> None:
    """Write a batch-of-1, un-activated reconstruction (model(..., activate_output=False)) as a
    standard INRIA-format 3DGS .ply, in the network's own native coordinate frame.

    Upstream's own export_to_obj (utils/app_utils.py) additionally rotates everything by a fixed
    matrix tuned for their Gradio 3D viewer -- that breaks orbit_video.py's Z-up world convention
    (confirmed empirically: with that rotation applied, the object spins around its nose-tail axis
    instead of yawing level). The network's raw frame already renders upright there, so this skips
    that rotation rather than trying to undo it downstream every time.
    """
    r = {k: v[0] for k, v in reconstruction.items()}
    valid = torch.where(r["opacity"] > -2.5)[0]

    xyz = r["xyz"][valid].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = r["features_dc"][valid].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = r["features_rest"][valid].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = r["opacity"][valid].detach().cpu().numpy()
    # enlarge Gaussians slightly, like upstream -- otherwise the .ply has artefacts
    scale = (r["scaling"][valid] + torch.abs(r["scaling"][valid] * 0.1)).detach().cpu().numpy()
    rotation = r["rotation"][valid].detach().cpu().numpy()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


@hydra.main(version_base=None, config_path="conf", config_name="splatter_image_infer")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)

    model, model_cfg = _load_model(device)
    log.info("Loaded %s checkpoint (%d params)", _DATASET, sum(p.numel() for p in model.parameters()))

    image = _preprocess(Path(cfg.image), cfg.remove_background, cfg.foreground_ratio).to(device)
    view_to_world_source, rot_transform_quats = _source_camera(cfg.source_radius)
    view_to_world_source = view_to_world_source.to(device)
    rot_transform_quats = rot_transform_quats.to(device)

    reconstruction_unactivated = model(
        image.unsqueeze(0).unsqueeze(0),
        view_to_world_source,
        rot_transform_quats,
        None,
        activate_output=False,
    )
    reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
    reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
    reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.export_ply:
        _export_ply(reconstruction_unactivated, str(out_dir / "reconstruction.ply"))
        log.info("Wrote %s", out_dir / "reconstruction.ply")

    if cfg.export_loop:
        world_view_transforms, full_proj_transforms, camera_centers = _target_cameras(
            model_cfg, cfg.source_radius, cfg.num_loop_views
        )
        background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        frames = []
        for r_idx in range(world_view_transforms.shape[0]):
            render = render_predicted(
                reconstruction,
                world_view_transforms[r_idx].to(device),
                full_proj_transforms[r_idx].to(device),
                camera_centers[r_idx].to(device),
                background,
                model_cfg,
                focals_pixels=None,
            )["render"]
            render = to_512(render)
            frames.append(torch.clamp(render * 255, 0, 255).byte().permute(1, 2, 0).cpu().numpy())

        loop_path = out_dir / "loop.mp4"
        imageio.mimsave(loop_path, frames, fps=cfg.fps)
        log.info("Wrote %s", loop_path)


if __name__ == "__main__":
    main()
