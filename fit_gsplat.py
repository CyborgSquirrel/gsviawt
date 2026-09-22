#!/usr/bin/env python3
"""Optimize a layered 3D Gaussian Splatting model from one render HDF5.

Input is an `.h5` (or glob/list of them) in the `render_objaverse.py` schema
(`images`, `depth_peel`, `depth_intrinsics`, `image_intrinsics`, `camera_pose`,
`mesh_index`; legacy `camera_intrinsics` is also accepted), read via
`gs_dataset.H5Catalog` and split into train/val view sets with `data.split_fn`
(a Hydra config group at `conf/data/split_fn/*.yaml`, the exact same one
train_gs.py uses -- see `conf/gsplat.yaml`'s `defaults:`). No mesh grouping:
each split's catalog ROW ORDER is taken directly as the view order (see
load_catalog_views) -- fit_gsplat.py fits ONE scene, from ONE object, at a
time; a corpus spanning multiple meshes isn't handled (produces garbage or
a clean mesh_index-mismatch error, deliberately not engineered around yet).

Within that scene's TRAIN split, the **primary** view -- whichever one
sorts first in the (already-split) catalog -- seeds the Gaussians; the
rest are **secondary**, supervision-only views. With
`data.split_fn=gs_dataset.split_by_indices`, that's exactly the first
entry of `train_idx` -- put the view you want to seed from there for
explicit control (e.g. `data/split_fn=by_indices
data.split_fn.train_idx='[0,1,2]' data.split_fn.val_idx='[3,4]'`).

The Gaussians are seeded entirely from the primary view's 6-layer depth peel --
every pixel-hit in every peel layer becomes one Gaussian, back-projected to world
space with that view's camera (reusing `debug_pointcloud.unproject_depth_peel`).
Each Gaussian therefore has a well-defined origin `(v, u, layer)` in the primary
depth map, and the optimizer never adds or removes Gaussians (no densification),
so that 1:1 correspondence survives to the output.

They are then optimized with `gsplat` against the RGB (L1 + D-SSIM) and alpha
(L1) of the primary + secondary (i.e. every train-split) views. Secondary views
contribute supervision only, never new Gaussians. Fully-occluded deeper-layer
Gaussians (seen in no supplied view) keep their initial front-pixel colour. The
val-split views (if any) never contribute Gaussians OR gradient -- they exist
purely for the val/loss* photometric metrics logged alongside training,
comparable to train_gs.py's own train/val split.

`images` may be a higher resolution than `depth_peel`: the seed grid and the
output grids stay at the depth-peel resolution (`depth_intrinsics`), while the
photometric loss renders at the image resolution (`image_intrinsics`).

By default (`optimize_means: false`) the Gaussian centers are locked to those
back-projected seed positions and only scale / rotation / opacity / colour are
optimized, so `gaussian_means` in the output equals the unprojected depth peel
exactly. Set `optimize_means: true` to let the centers move too.

Output is written next to the input as:
  <h5>.gsplat.view<primary>.h5  -- Gaussian attributes as (H, W, 6, .) grids
      (NaN marks an empty slot, same layout as `depth_peel` / wt_infer_layers'
      `points`), a `layer_valid` mask, and the views used (cameras + GT).
  <h5>.gsplat.view<primary>.ply  -- standard 3DGS point cloud (if write_ply)
  <h5>.gsplat.view<primary>.val/ -- gt|render comparison PNGs (if val_every)

Trained as a `pl.LightningModule` (GSFitLightningModule) under a plain
`pl.Trainer`, sharing its loss/schedule/OOM-handling building blocks with
train_gs.py via module.py -- see that module's docstring for what's
actually shared vs. deliberately not. There's no dataset in the usual sense
(one fixed scene is optimized every step, no batching/epochs over examples),
so GSFitDataModule's loader is a length-1 dummy: one Lightning "epoch" is
exactly one optimizer step, the same trick train_gs.py's GSFixedViewsDataset
fixed-view mode uses (see bla/train_gs_lightning_plan.md).

Config is Hydra (`conf/gsplat.yaml`); run in the container venv, e.g.

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python fit_gsplat.py \\
        hdf5_path=/app/bla/obj_lite.h5 iters=1500

gsplat JIT-compiles CUDA kernels on first import; this module points it at the
pip `nvidia-cuda-nvcc` toolchain (the base image has no system `nvcc`).
"""

import json
import logging
import os
import shutil
import sys
import tempfile

import h5py
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_pointcloud import unproject_depth_peel  # noqa: E402
from gs_dataset import H5Catalog  # noqa: E402
from module import DSSIMLoss  # noqa: E402
from util import intrinsics_name, timed  # noqa: E402

log = logging.getLogger(__name__)

# camera-local axis flip: Blender/OpenGL (X right, Y up, Z back) <-> OpenCV
# (X right, Y down, Z forward). Same flip `debug_pointcloud` applies inline.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
SH_C0 = 0.28209479177387814  # SH band-0 constant, for RGB <-> sh0 in the .ply


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_view_set(hdf5_path, ds, order):
  """Read exactly the given view indices (order matters -- index 0 is treated
  as "primary" for the mesh_index consistency check). Returns a dict of
  stacked arrays; called once per load_catalog_views call (once for the
  train split, once for the val split)."""
  with h5py.File(hdf5_path, "r") as f:
    n = f[ds.pose].shape[0]
    for i in order:
      if not 0 <= i < n:
        raise SystemExit(f"view index {i} out of range [0, {n}) in {hdf5_path}")

    if "mesh_index" in f:
      mi = f["mesh_index"][:]
      picked = {int(mi[i]) for i in order}
      if len(picked) > 1:
        raise SystemExit(
          f"selected views span mesh_index {sorted(picked)} -- views "
          f"must all be the same object")
      mesh_idx = int(mi[order[0]])
    else:
      mesh_idx = -1

    mesh_path = ""
    if "mesh_paths" in f and 0 <= mesh_idx < f["mesh_paths"].shape[0]:
      mp = f["mesh_paths"][mesh_idx]
      mesh_path = mp.decode() if isinstance(mp, bytes) else str(mp)

    depth_intr = ds.intrinsics if ds.intrinsics in f else intrinsics_name(f, "depth")
    image_intr = ds.image_intrinsics if ds.image_intrinsics in f else intrinsics_name(f, "image", None)

    images = np.stack([np.asarray(f[ds.images][i]) for i in order])          # (V,IH,IW,3|4) u8
    depth = np.stack([np.asarray(f[ds.depth][i]) for i in order]).astype(np.float32)  # (V,DH,DW,L)
    K = np.stack([np.asarray(f[depth_intr][i]) for i in order]).astype(np.float32)   # (V,3,3), matches depth
    pose = np.stack([np.asarray(f[ds.pose][i]) for i in order]).astype(np.float32)     # (V,4,4) c2w
    # image_intrinsics matches the RGB pass; a legacy file with only
    # camera_intrinsics (always equal-res) -> the depth K applies to both.
    if image_intr is not None:
      image_K = np.stack([np.asarray(f[image_intr][i]) for i in order]).astype(np.float32)
    else:
      image_K = K

  return {
    "path": hdf5_path, "order": order, "mesh_index": mesh_idx, "mesh_path": mesh_path,
    "images": images, "depth": depth, "K": K, "image_K": image_K, "pose": pose,
  }


def load_catalog_views(catalog: H5Catalog, ds_names):
  """catalog row order IS the view order -- row 0 ("primary") seeds the
  Gaussians, the rest ("secondary") are supervision-only. No mesh grouping,
  no multi-file handling: reads every row against the FIRST row's path,
  assuming (not checking) the whole catalog is one file/one object. With
  split_fn=gs_dataset.split_by_indices, "primary" is exactly the first
  entry of train_idx/val_idx -- put the view you want to seed from there
  for explicit control.

  Multi-mesh corpora are NOT handled -- deliberately out of scope for now.
  A catalog spanning more than one mesh either produces garbage (if it
  happens to share depth-peel layer counts etc.) or a clean SystemExit from
  _load_view_set's own mesh_index check below; both are acceptable until
  multi-mesh fitting is actually built.

  Returns None if `catalog` is empty (e.g. an empty val split)."""
  rows = list(catalog.df.iter_rows(named=True))
  if not rows:
    return None
  order = [int(r["view_idx"]) for r in rows]
  out = _load_view_set(rows[0]["path"], ds_names, order)
  out["primary"] = order[0]
  out["secondary"] = order[1:]
  return out


# ---------------------------------------------------------------------------
# gaussian init  (primary view, all 6 depth-peel layers)
# ---------------------------------------------------------------------------

def _logit(x, eps=1e-4):
  x = np.clip(x, eps, 1.0 - eps)
  return np.log(x / (1.0 - x))


def init_gaussians(depth_primary, K_primary, pose_primary, image_primary,
                   knn_k, init_opacity):
  """Seed one Gaussian per depth-peel hit in the primary view. Returns numpy
  arrays; `u/v/layer` record each Gaussian's pixel + peel-layer of origin (in
  depth-peel pixels). `image_primary` may be a different resolution than the
  depth peel -- the seed colour is nearest-sampled at the scaled location."""
  pts, u, v, layer = unproject_depth_peel(
    depth_primary, K_primary, pose_primary, space="world")          # (P,3), (P,), (P,), (P,)
  if len(pts) == 0:
    raise SystemExit("primary view has no depth-peel hits -- nothing to seed")

  dh, dw = depth_primary.shape[:2]
  ih, iw = image_primary.shape[:2]
  iv = np.clip(np.round(v * (ih / dh)), 0, ih - 1).astype(np.int64)
  iu = np.clip(np.round(u * (iw / dw)), 0, iw - 1).astype(np.int64)
  colors = image_primary[iv, iu, :3].astype(np.float32) / 255.0      # front-pixel colour

  # isotropic initial scale = mean distance to the knn_k nearest neighbours
  from scipy.spatial import cKDTree
  k = min(knn_k + 1, len(pts))
  dist, _ = cKDTree(pts).query(pts, k=k)                    # (P, k), or (P,) when k == 1
  dist = rearrange(dist, "p -> p 1") if dist.ndim == 1 else dist
  neighbours = dist[:, 1:] if k > 1 else dist               # column 0 is the point itself
  nn = reduce(neighbours, "p k -> p", "mean")
  nn = np.clip(nn, 1e-6, None).astype(np.float32)

  return {
    "means": pts.astype(np.float32),
    "scales_log": repeat(np.log(nn), "p -> p xyz", xyz=3).astype(np.float32),
    "quats": np.tile([1.0, 0.0, 0.0, 0.0], (len(pts), 1)).astype(np.float32),
    "opac_logit": np.full(len(pts), _logit(np.float32(init_opacity)), np.float32),
    "colors_logit": _logit(colors).astype(np.float32),
    "u": u.astype(np.int64), "v": v.astype(np.int64), "layer": layer.astype(np.int64),
  }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def make_viewmats(poses):
  """poses: (V,4,4) camera-to-world, OpenGL cam axes. Returns (V,4,4)
  world-to-camera in OpenCV convention -- what gsplat wants."""
  c2w_cv = poses @ OPENGL_TO_OPENCV
  return np.linalg.inv(c2w_cv).astype(np.float32)


def activate_gaussians(params):
  """params: the raw optimizer dict (means/scales_log/quats/opac_logit/
  colors_logit). Returns gsplat.rasterization-ready values (means passed
  through as-is; everything else activated): {"means","quats","scales",
  "opacities","colors"}. Split out of render() so PreviewSourceCallback can
  reuse just the activation, without also rasterizing."""
  return {
    "means": params["means"],
    "quats": F.normalize(params["quats"], dim=-1),
    "scales": torch.exp(params["scales_log"]),
    "opacities": torch.sigmoid(params["opac_logit"]),
    "colors": torch.sigmoid(params["colors_logit"]),
  }


def render(params, viewmats, Ks, width, height):
  import gsplat
  rgb, alpha, _ = gsplat.rasterization(
    **activate_gaussians(params),
    viewmats=viewmats, Ks=Ks, width=width, height=height,
    sh_degree=None, render_mode="RGB", packed=True,
  )
  # no `backgrounds`: colours come back un-composited (== premultiplied by the
  # rendered alpha), which is exactly what the premultiplied GT is compared to.
  return rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _scatter_grid(flat, v, u, layer, H, W, L):
  """flat: (G,) or (G,C). Returns (H,W,L) or (H,W,L,C) float32 with NaN in
  every slot that has no Gaussian."""
  shape = (H, W, L) if flat.ndim == 1 else (H, W, L, flat.shape[1])
  grid = np.full(shape, np.nan, np.float32)
  grid[v, u, layer] = flat
  return grid


def _scene_dataset(f, name, data):
  """One scene's grid, gzipped, chunked as the whole array. A future batched
  .h5 (many scenes in one file) should chunk one scene at a time the same way."""
  f.create_dataset(name, data=data, chunks=data.shape,
                   compression="gzip", compression_opts=4)


def save_output(path, cfg, views, params_np, uvl, H, W, L, final_loss, scene_scale):
  v, u, layer = uvl["v"], uvl["u"], uvl["layer"]
  means = params_np["means"]
  scales = np.exp(params_np["scales_log"])
  quats = params_np["quats"] / np.linalg.norm(params_np["quats"], axis=-1, keepdims=True)
  opac = 1.0 / (1.0 + np.exp(-params_np["opac_logit"]))
  colors = 1.0 / (1.0 + np.exp(-params_np["colors_logit"]))

  with h5py.File(path, "w") as f:
    # the full run config (resolved), as JSON -- everything the run was given.
    f.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    # things that aren't in the config: derived from the source file or the fit.
    f.attrs["mesh_index"] = views["mesh_index"]
    f.attrs["mesh_path"] = views["mesh_path"]
    f.attrs["final_loss"] = float(final_loss)
    f.attrs["num_gaussians"] = int(len(means))
    f.attrs["scene_scale"] = float(scene_scale)
    f.attrs["layer_layout"] = "(H, W, 6) like depth_peel; NaN = empty slot"

    # per-scene Gaussian grids (H, W, 6, .) -- NaN where the primary depth peel
    # had no hit. Whole-array chunk + gzip (mostly NaN, compresses hard).
    _scene_dataset(f, "gaussian_means", _scatter_grid(means, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_scales", _scatter_grid(scales, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_quats", _scatter_grid(quats, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_opacities", _scatter_grid(opac, v, u, layer, H, W, L))
    _scene_dataset(f, "gaussian_colors", _scatter_grid(colors, v, u, layer, H, W, L))
    valid = np.zeros((H, W, L), bool)
    valid[v, u, layer] = True
    _scene_dataset(f, "layer_valid", valid)

    # cameras / GT for the views actually used (primary first). Both
    # intrinsics always written (equal when RGB/depth res match), mirroring
    # render_objaverse's h5.
    f.create_dataset("camera_pose_used", data=views["pose"])
    f.create_dataset("depth_intrinsics_used", data=views["K"])            # matches depth / grid
    f.create_dataset("image_intrinsics_used", data=views["image_K"])      # matches images_used
    _scene_dataset(f, "depth_peel_primary", views["depth"][0])
    _scene_dataset(f, "images_used", views["images"])
    f.create_dataset("view_index_used", data=np.asarray(views["order"], np.int64))


def write_ply(path, params_np, scene_scale, opacity_threshold=None,
              max_scale_ratio=None, max_anisotropy=None):
  """Standard INRIA-format 3DGS .ply. `gsplat.export_splats` writes every
  field raw, and viewers apply the activations themselves: exp(scale),
  sigmoid(opacity), SH_C0*f_dc + 0.5. So pass the *unactivated* optimizer
  params -- log-scales, logit-opacities, SH-DC colours -- not the activated
  values stored in the .h5.

  All pruning is opt-in (None = off, the default), by:
    - low opacity:  sigmoid(opacity) <= opacity_threshold
    - huge: largest scale axis  >  max_scale_ratio * scene_scale
    - sliver: scale max/min ratio  >  max_anisotropy
  export_splats' own opacity threshold only applies to the compressed format,
  not plain "ply". Returns a {reason: count} dict plus "kept"/"total"."""
  import gsplat
  opac = 1.0 / (1.0 + np.exp(-params_np["opac_logit"]))
  scales = np.exp(params_np["scales_log"])                 # (N,3) world units
  ax_max, ax_min = scales.max(1), np.maximum(scales.min(1), 1e-12)

  drop_opacity = (np.zeros(len(opac), bool) if opacity_threshold is None
                  else opac <= float(opacity_threshold))
  drop_huge = (np.zeros_like(drop_opacity) if max_scale_ratio is None
               else ax_max > float(max_scale_ratio) * scene_scale)
  drop_sliver = (np.zeros_like(drop_opacity) if max_anisotropy is None
                 else ax_max / ax_min > float(max_anisotropy))
  keep = ~(drop_opacity | drop_huge | drop_sliver)

  colors = (1.0 / (1.0 + np.exp(-params_np["colors_logit"])))[keep]
  sh0 = ((colors - 0.5) / SH_C0)[:, None, :]           # (N,1,3) SH band-0 coeff
  quats = params_np["quats"][keep]
  quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
  gsplat.export_splats(
    means=torch.from_numpy(params_np["means"][keep]),
    scales=torch.from_numpy(params_np["scales_log"][keep]),  # log-space; viewer exp()s
    quats=torch.from_numpy(quats.astype(np.float32)),
    opacities=torch.from_numpy(params_np["opac_logit"][keep]),  # logit; viewer sigmoid()s
    sh0=torch.from_numpy(sh0.astype(np.float32)),
    shN=torch.zeros(len(colors), 0, 3),
    format="ply", save_to=path,
  )
  return {
    "total": int(keep.size), "kept": int(keep.sum()),
    "low_opacity": int(drop_opacity.sum()),
    "huge": int(drop_huge.sum()), "sliver": int(drop_sliver.sum()),
  }


def _build_val_panel(gt_rgb, gt_alpha, render_rgb, render_alpha):
  """Returns the [gt | render | gt_alpha | render_alpha] panel as one
  (H_total, W_total, 3) uint8 array, one row per view."""
  V = gt_rgb.shape[0]
  rows = []
  for i in range(V):
    g = (gt_rgb[i].detach().cpu().numpy() * 255).astype(np.uint8)
    r = (render_rgb[i].detach().cpu().numpy() * 255).astype(np.uint8)
    ga = (gt_alpha[i, ..., 0].detach().cpu().numpy() * 255).astype(np.uint8)
    ra = (render_alpha[i, ..., 0].detach().cpu().numpy() * 255).astype(np.uint8)
    ga3 = np.repeat(ga[..., None], 3, axis=2)
    ra3 = np.repeat(ra[..., None], 3, axis=2)
    rows.append(np.concatenate([g, r, ga3, ra3], axis=1))
  return np.concatenate(rows, axis=0)


def dump_val(val_dir, it, panel):
  from PIL import Image
  os.makedirs(val_dir, exist_ok=True)
  Image.fromarray(panel).save(os.path.join(val_dir, f"iter{it:05d}.png"))


# ---------------------------------------------------------------------------
# orbit previews (wandb.Video, periodic during optimization)
#
# Unlike train_gs.py's Gaussians (anchored in an arbitrary source camera's own
# frame -- see gs_dataset.py), these are seeded via unproject_depth_peel's
# "world" branch, i.e. already in true Blender world coordinates (object
# recentred near the origin by render_objaverse.py's normalize_object). So the
# orbit can use the same look_at_c2w construction orbit_video.py uses,
# directly, no per-scene "up" derivation needed. Deliberately anchored to the
# PRIMARY view's own actual capture distance (`scene_scale`), not an
# auto-fit/bounding-sphere distance like orbit_video.py's default: this
# optimization has no floor forcing Gaussian scale to stay above ~1 pixel of
# spacing (see train_gs.py's GSModel.forward for why that matters), so a
# tighter-than-capture orbit distance can expose the same false "gaps"
# artifact that turned out to be a real bug there -- keeping the same
# distance the model was actually supervised at avoids manufacturing that
# confusion here.
#
# NOTE: kept as plain functions here, not moved to module.py or shared
# with orbit_video.py/train_gs.py's own orbit callbacks -- deliberately out
# of scope for this pass, see the PR description.
# ---------------------------------------------------------------------------

WORLD_UP = np.array([0.0, 0.0, 1.0], np.float32)  # Blender / render_objaverse is Z-up


def look_at_c2w(eye, target, up=WORLD_UP):
  """OpenGL camera-to-world (X right, Y up, -Z forward) looking from `eye` at
  `target`. Same construction as orbit_video.py's look_at_c2w."""
  z = eye - target
  z = z / (np.linalg.norm(z) + 1e-8)
  if abs(np.dot(z, up)) > 0.999:
    up = np.array([0.0, 1.0, 0.0], np.float32) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
  x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-8)
  y = np.cross(z, x)
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return c2w


def write_mp4(frames, path, fps, crf):
  """H.264 .mp4 via imageio's ffmpeg backend (imageio-ffmpeg ships a static
  binary -- nothing needed on the system PATH). Same as orbit_video.py's."""
  import imageio.v2 as imageio
  writer = imageio.get_writer(
    path, format="FFMPEG", mode="I", fps=float(fps),
    codec="libx264", macro_block_size=1, pixelformat="yuv420p",
    ffmpeg_params=["-crf", str(int(crf))],
  )
  try:
    for fr in frames:
      writer.append_data(np.ascontiguousarray(fr))
  finally:
    writer.close()
  return path


def render_orbit_frames(params, K_ref, dist, width, height, device, num_frames=24, elevation_deg=20.0):
  """Renders a turntable orbit around the world origin at radius `dist`
  (the primary view's own capture distance), reusing this module's own
  `render()`. Yields (H,W,3) uint8 frames one at a time (one gsplat call per
  frame, rather than batching all num_frames cameras into one call) to keep
  peak memory bounded regardless of num_frames -- write_mp4 consumes frames
  one at a time anyway, so nothing needs the full orbit in memory at once."""
  elev = np.radians(float(elevation_deg))
  azimuths = np.linspace(0.0, 2 * np.pi, int(num_frames), endpoint=False)
  K_t = torch.from_numpy(K_ref[None]).to(device)
  for az in azimuths:
    d = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)], np.float32)
    pose = look_at_c2w(d * dist, np.zeros(3, np.float32))
    viewmat = torch.from_numpy(make_viewmats(pose[None])).to(device)
    with torch.no_grad():
      rgb, _ = render(params, viewmat, K_t, width, height)
    yield (rgb[0].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)


def log_orbit_video(wandb_run, step, frames, fps, crf, workdir):
  path = os.path.join(workdir, f"orbit_{step}.mp4")
  write_mp4(frames, path, fps, crf)
  wandb_run.log({"orbit": wandb.Video(path, caption=f"iter {step}", format="mp4")}, step=step)


# ---------------------------------------------------------------------------
# lightning module
# ---------------------------------------------------------------------------

class _SingleBatchDataset(torch.utils.data.Dataset):
  """Length-1 dataset: fit_gsplat has no dataset in the usual sense (one
  fixed set of views is fitted every step, not sampled from a corpus), so a
  length-1 loader makes one Lightning "epoch" equal exactly one optimizer
  step -- the same trick train_gs.py's GSFixedViewsDataset fixed-view mode
  uses (see bla/train_gs_lightning_plan.md's epoch==step discussion). The
  yielded value is never used; training_step/validation_step read straight
  off the module's own registered buffers instead."""
  def __len__(self):
    return 1

  def __getitem__(self, idx):
    return 0


class GSFitDataModule(pl.LightningDataModule):
  def train_dataloader(self):
    return DataLoader(_SingleBatchDataset(), batch_size=1)

  def val_dataloader(self):
    # Always returns a loader (even with no val views); GSFitLightningModule
    # only actually renders when it has val views, and main() sets
    # limit_val_batches=0 otherwise so this is a no-op in that case rather
    # than needing a None-returning special case here.
    return DataLoader(_SingleBatchDataset(), batch_size=1)


class GSFitLightningModule(pl.LightningModule):
  """One scene's Gaussian parameters, optimized directly (no encoder/decoder
  network -- contrast train_gs.py's GSLightningModule, which predicts
  Gaussians from an image). `views` is the primary+secondary supervision
  set (contributes gradient); `val_views` (optional) is a disjoint held-out
  set used only for the val/loss* metrics below."""

  def __init__(self, cfg, g, uvl, views, val_views, scene_scale, K_orbit):
    super().__init__()
    self.cfg = cfg
    self.uvl = uvl
    self.scene_scale = scene_scale
    self.optimize_means = bool(cfg.optimize_means)
    self.K_orbit = K_orbit
    self.orbit_workdir = None
    self.views_seen = 0
    self.final_loss = float("nan")
    self.iters = int(cfg.iters)
    self.log_every = max(1, self.iters // 10)
    self.preview_source = None  # stashed by PreviewSourceCallback, read by module.PanelCallback

    self.params = nn.ParameterDict({
      k: nn.Parameter(torch.from_numpy(v), requires_grad=(k != "means" or self.optimize_means))
      for k, v in g.items()
    })
    self.dssim = DSSIMLoss()
    self._register_view_buffers("", views)

    self.has_val = val_views is not None
    if self.has_val:
      self._register_view_buffers("val_", val_views)

    V = views["images"].shape[0]
    views_per_iter = OmegaConf.select(cfg, "views_per_iter")
    self.views_per_iter = V if views_per_iter is None else min(int(views_per_iter), V)
    self.view_rng = np.random.default_rng(int(cfg.seed))

    # stashed by training_step, reused by the val-panel dump when it already
    # covers the full view set (views_per_iter >= V) -- avoids a redundant
    # render, same optimization the original hand-rolled loop made.
    self._last_rgb_c = None
    self._last_alpha = None

  def _register_view_buffers(self, prefix, views):
    self.register_buffer(f"{prefix}viewmats", torch.from_numpy(make_viewmats(views["pose"])), persistent=False)
    self.register_buffer(f"{prefix}Ks", torch.from_numpy(views["image_K"]), persistent=False)
    gt_rgb = torch.from_numpy(views["images"][..., :3].astype(np.float32) / 255.0)
    if views["images"].shape[-1] == 4:
      gt_alpha = torch.from_numpy(views["images"][..., 3:4].astype(np.float32) / 255.0)
    else:  # equal-resolution RGB-only file (guarded in main()): mask from the surface layer
      gt_alpha = torch.from_numpy((views["depth"][:, :, :, 0] > 0).astype(np.float32))[..., None]
    gt_rgb = gt_rgb * gt_alpha  # composite GT over black, matching a black-bg render
    self.register_buffer(f"{prefix}gt_rgb", gt_rgb, persistent=False)
    self.register_buffer(f"{prefix}gt_alpha", gt_alpha, persistent=False)

  def configure_optimizers(self):
    cfg = self.cfg
    groups = [
      {"params": [self.params["scales_log"]], "lr": float(cfg.lr.scales)},
      {"params": [self.params["quats"]], "lr": float(cfg.lr.quats)},
      {"params": [self.params["opac_logit"]], "lr": float(cfg.lr.opacities)},
      {"params": [self.params["colors_logit"]], "lr": float(cfg.lr.colors)},
    ]
    if self.optimize_means:
      groups.insert(0, {"params": [self.params["means"]], "lr": float(cfg.lr.means) * self.scene_scale})
    return torch.optim.Adam(groups)

  def _photom_loss(self, rgb, alpha, gt_rgb, gt_alpha):
    rgb_c = rgb * alpha  # premultiply so bg stays black on both sides
    l1 = (rgb_c - gt_rgb).abs().mean()
    dssim = self.dssim(rgb_c.permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2))
    mask = (alpha - gt_alpha).abs().mean()
    ls, lm = float(self.cfg.lambda_ssim), float(self.cfg.lambda_mask)
    loss = (1 - ls) * l1 + ls * dssim + lm * mask
    return loss, rgb_c, {"l1": l1, "dssim": dssim, "mask": mask}

  def training_step(self, batch, batch_idx):
    return self._step("train", batch, batch_idx)

  def validation_step(self, batch, batch_idx):
    return self._step("val", batch, batch_idx)

  def _step(self, stage, batch, batch_idx):
    if stage == "val":
      if not self.has_val:
        return None
      vm, ks, gt_rgb, gt_alpha = self.val_viewmats, self.val_Ks, self.val_gt_rgb, self.val_gt_alpha
    else:
      V = self.viewmats.shape[0]
      if self.views_per_iter >= V:
        vm, ks, gt_rgb, gt_alpha = self.viewmats, self.Ks, self.gt_rgb, self.gt_alpha
      else:
        idx = torch.from_numpy(
          self.view_rng.choice(V, size=self.views_per_iter, replace=False)).to(self.device)
        vm, ks, gt_rgb, gt_alpha = self.viewmats[idx], self.Ks[idx], self.gt_rgb[idx], self.gt_alpha[idx]

    IH, IW = gt_rgb.shape[1:3]
    with torch.set_grad_enabled(stage == "train"):
      rgb, alpha = render(self.params, vm, ks, IW, IH)
      loss, rgb_c, parts = self._photom_loss(rgb, alpha, gt_rgb, gt_alpha)

    def _log(key, *args, **kwargs):
      self.log(f"{stage}/{key}", *args, **kwargs, batch_size=1)

    _log("loss", loss, prog_bar=(stage == "train"))
    for k, v in parts.items():
      _log(f"loss/photom_{k}", v.detach())

    if stage == "train":
      self._last_rgb_c, self._last_alpha = rgb_c.detach(), alpha.detach()
      self.views_seen += self.views_per_iter
      self.final_loss = loss.item()  # plain float: fed to log.info/%f, save_output, wandb.summary
      self.log("views_seen", self.views_seen, reduce_fx="max", batch_size=1)

      it = self.current_epoch
      last = it == self.iters - 1
      if it % self.log_every == 0 or last:
        log.info("iter %d/%d  loss %.5f  (l1 %.5f  dssim %.5f  mask %.5f)",
                 it, self.iters, loss.item(), parts["l1"].item(), parts["dssim"].item(),
                 parts["mask"].item())
    return loss

  def params_np(self):
    return {k: v.detach().cpu().numpy() for k, v in self.params.items()}

  def _wandb_run(self):
    return self.logger.experiment if self.logger is not None else None

  def on_fit_start(self):
    if bool(self.cfg.orbit.enabled):
      self.orbit_workdir = tempfile.mkdtemp(prefix="fit_gsplat_orbit_")

  def on_fit_end(self):
    if self.orbit_workdir is not None:
      shutil.rmtree(self.orbit_workdir, ignore_errors=True)
      self.orbit_workdir = None

  def on_train_epoch_end(self):
    # `self.current_epoch` here is still the just-finished 0-indexed
    # iteration (Lightning bumps it after this hook), matching the original
    # hand-rolled loop's post-step `it` exactly -- no +1 correction needed
    # (contrast train_gs.py's OrbitCallback, a separate pl.Callback where the
    # bump has already happened by the time it runs).
    it = self.current_epoch
    last = it == self.iters - 1
    cfg = self.cfg

    val_every = int(cfg.val_every)
    if val_every and (it % val_every == 0 or last):
      self._dump_val_panel(it)

    orbit_every = int(cfg.orbit.every) if cfg.orbit.enabled else 0
    wandb_run = self._wandb_run()
    if orbit_every and wandb_run is not None and (it % orbit_every == 0 or last):
      IH, IW = self.gt_rgb.shape[1:3]
      frames = render_orbit_frames(
        self.params, self.K_orbit, self.scene_scale, IW, IH, self.device,
        num_frames=cfg.orbit.num_frames, elevation_deg=cfg.orbit.elevation_deg,
      )
      log_orbit_video(wandb_run, it, frames, cfg.orbit.fps, cfg.orbit.crf, self.orbit_workdir)

  def _dump_val_panel(self, it):
    # Always panels against the FULL supervision view set, independent of
    # what this step happened to sample -- otherwise the panel's row
    # count/pairing would silently depend on views_per_iter. NOTE: this is
    # the primary+secondary supervision set's own fit quality, NOT the
    # val/loss* held-out metrics above -- same "val" name as the original
    # script, now sitting a bit awkwardly next to the new held-out split;
    # flagged in the PR description rather than renamed here.
    V = self.viewmats.shape[0]
    if self.views_per_iter >= V:
      full_rgb_c, full_alpha = self._last_rgb_c, self._last_alpha
    else:
      IH, IW = self.gt_rgb.shape[1:3]
      with torch.no_grad():
        full_rgb, full_alpha = render(self.params, self.viewmats, self.Ks, IW, IH)
        full_rgb_c = full_rgb * full_alpha
    panel = _build_val_panel(self.gt_rgb, self.gt_alpha, full_rgb_c, full_alpha)
    dump_val(f"{self.cfg.output_stem}.val", it, panel)
    wandb_run = self._wandb_run()
    if wandb_run is not None:
      wandb_run.log({"val/panel": wandb.Image(panel, caption=f"iter {it}")}, step=it)


class PreviewSourceCallback(pl.Callback):
  """Producer half of module.PanelCallback's preview_source hand-off (see
  that module's own comment block for the full contract). Unlike
  train_gs.py's version, there's no model and no dataset batch to run here
  -- the 3DGS IS pl_module.params, this optimization's own live state --
  so this just activates it (activate_gaussians, the same helper render()
  uses) and packages it with the fixed supervision view set.

  Train-stage only for now -- see module.PanelCallback."""

  def on_train_epoch_end(self, trainer, pl_module):
    gauss = activate_gaussians(pl_module.params)
    gauss["sh_degree"] = None
    pl_module.preview_source = {
      "gauss": gauss,
      "views": {
        "viewmat": pl_module.viewmats,
        "K": pl_module.Ks,
        "width": pl_module.gt_rgb.shape[2],
        "height": pl_module.gt_rgb.shape[1],
        "gt_rgb": rearrange(pl_module.gt_rgb, "v h w c -> v c h w"),
      },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="gsplat")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.seed))

  with timed("load"):
    catalog = H5Catalog(
      cfg.hdf5_path,
      H5Catalog.path().alias("path"),
      H5Catalog.index().alias("view_idx"),
      H5Catalog.dataset("mesh_index").alias("mesh_id"),
    )
    split_fn = hydra.utils.instantiate(cfg.data.split_fn)
    train_catalog, val_catalog = split_fn(catalog, seed=cfg.seed)
    views = load_catalog_views(train_catalog, cfg.datasets)
    if views is None:
      raise SystemExit("train split is empty -- check hdf5_path / data.split_fn")
    val_views = load_catalog_views(val_catalog, cfg.datasets)

  out_h5 = cfg.output_path or f"{views['path']}.gsplat.view{views['primary']}.h5"
  stem = out_h5[:-3] if out_h5.endswith(".h5") else out_h5

  with open_dict(cfg):
    cfg.output_stem = stem              # so GSFitLightningModule can find the .val/ dir
    cfg.has_val = val_views is not None  # read by conf/gsplat.yaml's trainer.* interpolations

  # Stashed above BEFORE this: OmegaConf.to_container(cfg, resolve=True)
  # below (for wandb's own config snapshot) resolves the whole tree,
  # including trainer.num_sanity_val_steps's ${has_val} interpolation --
  # has_val must already exist on cfg by the time that call runs.
  wandb_run, logger = None, False
  if cfg.wandb.mode != "disabled":
    wandb_run = wandb.init(
      project=cfg.wandb.project,
      mode=cfg.wandb.mode,
      tags=list(cfg.wandb.tags),
      name=cfg.wandb.name,
      config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger = WandbLogger(experiment=wandb_run)

  V = views["images"].shape[0]
  IH, IW = views["images"].shape[1:3]      # RGB render / supervision resolution
  DH, DW, L = views["depth"].shape[1:]     # depth-peel = seed + output-grid resolution
  if views["images"].shape[-1] != 4 and (IH, IW) != (DH, DW):
    raise SystemExit("RGB-only file (no alpha) with images and depth_peel at "
                     "different resolutions is not supported -- the mask can't be "
                     "derived. Use an RGBA render or equal resolutions.")

  views_per_iter = OmegaConf.select(cfg, "views_per_iter")
  views_per_iter = V if views_per_iter is None else min(int(views_per_iter), V)
  log.info("primary=%d secondary=%s val=%s  %d views  RGB %dx%d  depth/grid %dx%d  "
           "%d peel layers  views_per_iter=%d  mesh=%s",
           views["primary"], views["secondary"], val_views["order"] if val_views else [],
           V, IW, IH, DW, DH, L, views_per_iter, views["mesh_path"] or views["mesh_index"])

  with timed("init"):
    g = init_gaussians(views["depth"][0], views["K"][0], views["pose"][0],
                       views["images"][0], int(cfg.knn_k), float(cfg.init_opacity))
  uvl = {"u": g.pop("u"), "v": g.pop("v"), "layer": g.pop("layer")}
  n_gauss = len(g["means"])
  per_layer = np.bincount(uvl["layer"], minlength=L)
  log.info("seeded %d Gaussians (from the %dx%d depth peel)  per-layer counts %s",
           n_gauss, DW, DH, per_layer.tolist())

  scene_scale = float(np.linalg.norm(views["pose"][0][:3, 3])) or 1.0
  log.info("Gaussian centers: %s",
           "trainable" if bool(cfg.optimize_means) else "FROZEN at depth-peel seed positions")

  model = GSFitLightningModule(cfg, g, uvl, views, val_views, scene_scale, views["image_K"][0])

  callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
  trainer = pl.Trainer(
    logger=logger, callbacks=callbacks, **OmegaConf.to_container(cfg.trainer, resolve=True))
  with timed("optimize"):
    trainer.fit(model, datamodule=GSFitDataModule())

  params_np = model.params_np()
  with timed("write"):
    save_output(out_h5, cfg, views, params_np, uvl, DH, DW, L, model.final_loss, scene_scale)
    if bool(cfg.write_ply):
      st = write_ply(f"{stem}.ply", params_np, scene_scale,
                     cfg.ply_opacity_threshold,
                     cfg.get("ply_max_scale_ratio"), cfg.get("ply_max_anisotropy"))
      log.info(".ply: kept %d/%d Gaussians  (pruned %d low-opacity, %d huge, %d sliver)",
               st["kept"], st["total"], st["low_opacity"], st["huge"], st["sliver"])

  log.info("final loss %.5f  ->  %s%s%s", model.final_loss, out_h5,
           f"  {stem}.ply" if cfg.write_ply else "",
           f"  {stem}.val/" if int(cfg.val_every) else "")
  if wandb_run is not None:
    wandb_run.summary["final_loss"] = model.final_loss
    wandb_run.summary["num_gaussians"] = n_gauss
    wandb_run.finish()


if __name__ == "__main__":
  main()
