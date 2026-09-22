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

Trained as a `pl.LightningModule` (GSFitLightningModule) under a plain
`pl.Trainer`, sharing its loss/schedule/OOM-handling building blocks with
train_gs.py via module.py -- see that module's docstring for what's
actually shared vs. deliberately not. Supervision views flow through a
real `GSFitViewDataset`/`GSFitDataModule` (one item = one view;
`loader.batch_size`, default 1, picks how many are rendered+compared per
step, replacing what used to be manual per-iteration subsampling). `iters`
(== `trainer.max_steps`) is a raw optimizer-step count, independent of
train split size / batch_size -- a PyTorch "epoch" (one full pass over
the train split) still exists and drives `check_val_every_n_epoch` /
per-epoch-cadence callbacks, but doesn't bound training length itself.

Config is Hydra (`conf/gsplat.yaml`); run in the container venv, e.g.

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python fit_gsplat.py \\
        hdf5_path=/app/bla/obj_lite.h5 iters=1500

gsplat JIT-compiles CUDA kernels on first import; this module points it at the
pip `nvidia-cuda-nvcc` toolchain (the base image has no system `nvcc`).
"""

import json
import logging
import os
import sys

import h5py
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_pointcloud import unproject_depth_peel  # noqa: E402
from gs_dataset import H5Catalog, _EmptyDataset  # noqa: E402
from module import DSSIMLoss, OPENGL_TO_OPENCV  # noqa: E402
from util import intrinsics_name, timed  # noqa: E402

log = logging.getLogger(__name__)

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
  "opacities","colors"}. Split out of render() so get_preview_source() can
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


# ---------------------------------------------------------------------------
# dataset / lightning module
#
# Orbit-preview pose generation used to live here (world-frame look_at_c2w,
# anchored to scene_scale) -- now fully centralized in module.OrbitCallback,
# since get_preview_source() below hands it world-space Gaussians same as
# train_gs.py does, so one shared orbit formula covers both scripts. See
# module.py's own comment block.
# ---------------------------------------------------------------------------

def _composite_gt(views):
  """views["images"] -> (gt_rgb, gt_alpha) numpy float32 arrays, alpha-
  premultiplied and composited over black -- matches what render() itself
  returns (un-composited, i.e. already premultiplied by its own rendered
  alpha), so the two are directly comparable. Shared by GSFitViewDataset
  (per-batch training/val targets) and GSFitLightningModule._register_view_
  buffers (the fixed full-view-set preview buffers) -- same GT, two
  different purposes for it."""
  images = views["images"].astype(np.float32) / 255.0
  if images.shape[-1] == 4:
    rgb, alpha = images[..., :3], images[..., 3:4]
  else:  # equal-resolution RGB-only file (guarded in main()): mask from the surface layer
    rgb = images
    alpha = (views["depth"][:, :, :, 0] > 0).astype(np.float32)[..., None]
  gt_rgb = np.ascontiguousarray(rgb * alpha)  # composited over black, matches render()
  gt_alpha = np.ascontiguousarray(alpha)
  return gt_rgb, gt_alpha


class GSFitViewDataset(Dataset):
  """One item = one view's photometric supervision target: GT rgb
  (premultiplied by alpha, composited over black -- matches what render()
  itself returns, so _photom_loss can compare them directly), alpha,
  intrinsics, and a gsplat-ready viewmat (already converted via
  make_viewmats -- nothing downstream needs the raw camera-to-world pose
  per item). Thin wrapper over `views`'s already-in-memory stacked arrays
  (load_catalog_views already read the whole split into memory upfront) --
  no I/O per __getitem__. `len(dataset)` is the number of views in that
  split; one instance for the TRAIN split, a separate one for the held-out
  VAL split."""

  def __init__(self, views):
    self.gt_rgb, self.gt_alpha = _composite_gt(views)
    self.K = views["image_K"]
    self.viewmat = make_viewmats(views["pose"])

  def __len__(self):
    return len(self.viewmat)

  def __getitem__(self, i):
    return {
      "gt_rgb": torch.from_numpy(self.gt_rgb[i]),
      "gt_alpha": torch.from_numpy(self.gt_alpha[i]),
      "K": torch.from_numpy(self.K[i]),
      "viewmat": torch.from_numpy(self.viewmat[i]),
    }


class GSFitDataModule(pl.LightningDataModule):
  """Real Dataset+DataLoader (GSFitViewDataset) over the TRAIN/VAL view
  splits main() already loaded into memory -- each training step's view
  selection is now just DataLoader(shuffle=True, batch_size=cfg.loader.
  batch_size): standard PyTorch batching/shuffling, instead of a
  hand-rolled per-iteration np.random.Generator.choice. val_ds is a
  gs_dataset._EmptyDataset (same convention train_gs.py uses) when there's
  no held-out split -- Lightning then skips validation on its own, no
  extra "has_val" bookkeeping needed anywhere."""

  def __init__(self, cfg, views, val_views):
    super().__init__()
    self.cfg = cfg
    self.train_ds = GSFitViewDataset(views)
    self.val_ds = GSFitViewDataset(val_views) if val_views is not None else _EmptyDataset()

  def train_dataloader(self):
    return DataLoader(
      self.train_ds, shuffle=True,
      **OmegaConf.to_container(self.cfg.loader, resolve=True),
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_ds, shuffle=False,
      **OmegaConf.to_container(self.cfg.loader, resolve=True),
    )


class GSFitLightningModule(pl.LightningModule):
  """One scene's Gaussian parameters, optimized directly (no encoder/decoder
  network -- contrast train_gs.py's GSLightningModule, which predicts
  Gaussians from an image). `views` is the primary+secondary supervision
  set (contributes gradient); `val_views` (optional) is a disjoint held-out
  set used only for the val/loss* metrics below.

  Besides the real per-view training/val batches (see GSFitDataModule),
  this also keeps the FULL view set as registered buffers, used only by
  get_preview_source() below for the wandb panel/orbit preview -- a
  comparison against every supervision view, independent of whatever a
  given training batch happened to sample."""

  def __init__(self, cfg, g, uvl, views, val_views, scene_scale):
    super().__init__()
    self.cfg = cfg
    self.uvl = uvl
    self.scene_scale = scene_scale
    self.optimize_means = bool(cfg.optimize_means)
    self.views_seen = 0
    self.final_loss = float("nan")

    self.params = nn.ParameterDict({
      k: nn.Parameter(torch.from_numpy(v), requires_grad=(k != "means" or self.optimize_means))
      for k, v in g.items()
    })
    self.dssim = DSSIMLoss()
    self._register_view_buffers("", views)
    if val_views is not None:
      self._register_view_buffers("val_", val_views)

  def _register_view_buffers(self, prefix, views):
    self.register_buffer(f"{prefix}viewmats", torch.from_numpy(make_viewmats(views["pose"])), persistent=False)
    self.register_buffer(f"{prefix}Ks", torch.from_numpy(views["image_K"]), persistent=False)
    gt_rgb, gt_alpha = _composite_gt(views)
    self.register_buffer(f"{prefix}gt_rgb", torch.from_numpy(gt_rgb), persistent=False)
    self.register_buffer(f"{prefix}gt_alpha", torch.from_numpy(gt_alpha), persistent=False)

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
    def _log(key, *args, **kwargs):
      self.log(f"{stage}/{key}", *args, **kwargs, batch_size=B)

    B = batch["gt_rgb"].shape[0]
    gt_rgb, gt_alpha = batch["gt_rgb"], batch["gt_alpha"]
    vm, ks = batch["viewmat"], batch["K"]
    IH, IW = gt_rgb.shape[1:3]

    with torch.set_grad_enabled(stage == "train"):
      rgb, alpha = render(self.params, vm, ks, IW, IH)
      loss, _, parts = self._photom_loss(rgb, alpha, gt_rgb, gt_alpha)

    _log("loss", loss, prog_bar=(stage == "train"))
    for k, v in parts.items():
      _log(f"loss/photom_{k}", v.detach())

    if stage == "train":
      self.views_seen += B
      self.final_loss = loss.item()  # plain float: fed to log.info/%f, save_output, wandb.summary
      self.log("views_seen", self.views_seen, reduce_fx="max", batch_size=B)

    return loss

  def params_np(self):
    return {k: v.detach().cpu().numpy() for k, v in self.params.items()}

  def get_preview_source(self, mode):
    """{"train": entry, "val": entry-or-None} for module.PanelCallback/
    OrbitCallback -- see module.py's own comment block for the full
    contract. No model, no dataset batch -- the 3DGS IS self.params, this
    optimization's own live state, shared by every stage (unlike
    train_gs.py's model, which predicts a DIFFERENT Gaussian set per
    forward-passed item) -- so this just activates it once
    (activate_gaussians, the same helper render() uses) and pairs it with
    each stage's own fixed view set (self.viewmats/self.val_viewmats etc,
    from _register_view_buffers -- independent of GSFitDataModule's
    per-step batches).

    mode picks both the cache granularity (epoch: once per
    self.current_epoch; step: once per self.trainer.global_step) and
    whether "val" is computed at all -- skipped (left None) in epoch mode
    even when a val split exists, since nothing pulls it there (see
    module.PanelCallback._step)."""
    key = (mode, self.current_epoch if mode == "epoch" else self.trainer.global_step)
    if getattr(self, "_preview_cache_key", None) == key:
      return self._preview_cache

    with torch.no_grad():
      gauss = activate_gaussians(self.params)
    gauss["sh_degree"] = None

    def _entry(prefix):
      gt_rgb = getattr(self, f"{prefix}gt_rgb", None)
      if gt_rgb is None:
        return None
      return {
        "gauss": gauss,
        # scene_scale: the primary view's own real capture distance from the
        # (world) origin -- module.OrbitCallback's orbit radius. Already true
        # world space here (see this module's docstring), unlike
        # train_gs.py's version of this field, which has to derive an
        # equivalent quantity from a camera-relative frame.
        "scene_scale": self.scene_scale,
        "views": {
          "viewmat": getattr(self, f"{prefix}viewmats"),
          "K": getattr(self, f"{prefix}Ks"),
          "width": gt_rgb.shape[2],
          "height": gt_rgb.shape[1],
          "gt_rgb": rearrange(gt_rgb, "v h w c -> v c h w"),
        },
      }

    source = {"train": _entry(""), "val": _entry("val_") if mode == "step" else None}
    self._preview_cache_key, self._preview_cache = key, source
    return source

  def _wandb_run(self):
    return self.logger.experiment if self.logger is not None else None


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

  IH, IW = views["images"].shape[1:3]      # RGB render / supervision resolution
  DH, DW, L = views["depth"].shape[1:]     # depth-peel = seed + output-grid resolution
  if views["images"].shape[-1] != 4 and (IH, IW) != (DH, DW):
    raise SystemExit("RGB-only file (no alpha) with images and depth_peel at "
                     "different resolutions is not supported -- the mask can't be "
                     "derived. Use an RGBA render or equal resolutions.")

  log.info("primary=%d secondary=%s val=%s  %d views  RGB %dx%d  depth/grid %dx%d  "
           "%d peel layers  loader.batch_size=%s  mesh=%s",
           views["primary"], views["secondary"], val_views["order"] if val_views else [],
           views["images"].shape[0], IW, IH, DW, DH, L, cfg.loader.batch_size,
           views["mesh_path"] or views["mesh_index"])

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

  model = GSFitLightningModule(cfg, g, uvl, views, val_views, scene_scale)
  datamodule = GSFitDataModule(cfg, views, val_views)

  callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
  trainer = pl.Trainer(
    logger=logger, callbacks=callbacks, **OmegaConf.to_container(cfg.trainer, resolve=True))
  with timed("optimize"):
    trainer.fit(model, datamodule=datamodule)

  params_np = model.params_np()
  with timed("write"):
    save_output(out_h5, cfg, views, params_np, uvl, DH, DW, L, model.final_loss, scene_scale)
    if bool(cfg.write_ply):
      st = write_ply(f"{stem}.ply", params_np, scene_scale,
                     cfg.ply_opacity_threshold,
                     cfg.get("ply_max_scale_ratio"), cfg.get("ply_max_anisotropy"))
      log.info(".ply: kept %d/%d Gaussians  (pruned %d low-opacity, %d huge, %d sliver)",
               st["kept"], st["total"], st["low_opacity"], st["huge"], st["sliver"])

  log.info("final loss %.5f  ->  %s%s", model.final_loss, out_h5,
           f"  {stem}.ply" if cfg.write_ply else "")
  if wandb_run is not None:
    wandb_run.summary["final_loss"] = model.final_loss
    wandb_run.summary["num_gaussians"] = n_gauss
    wandb_run.finish()


if __name__ == "__main__":
  main()
