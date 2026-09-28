#!/usr/bin/env python3
"""Faithful 3D Gaussian Splatting optimization on a COLMAP scene (same format
as gaussian-splatting/ -- images/ + sparse/0/, e.g. data/tandt/truck), ported
onto this repo's Hydra + Lightning + wandb-logging + shared-callback shape
(module.py's, originally fit_gsplat.py's) instead of the reference's
argparse + raw training loop.

Unlike fit_gsplat.py (which seeds one Gaussian per depth-peel pixel, from a
synthetic render, and never adds to that set), `gaussian_layout: set`
(the default) ports the REAL algorithm: free `xyz`, the COLMAP sparse point
cloud as the seed, and full growing/shrinking densification via
gsplat.strategy.DefaultStrategy -- gsplat's own reimplementation of
gaussian-splatting/scene/gaussian_model.py's densify_and_prune/
reset_opacity, verified line-by-line against it (matching grad/scale
thresholds, split/duplicate/prune conditions, opacity-reset value).
Empirically cross-checked too: an early-run Gaussian-count trajectory
(iters 600-1600) landed within ~1% of an instrumented copy of the reference
train.py on this exact scene at every checkpoint.

`gaussian_layout: layered` is the other option, for an h5 scene only (see
load_scene_h5/init_gaussians_layered): seeds one Gaussian per depth-peel
hit in the primary train view instead, same as fit_gsplat.py's own
seeding (reused verbatim), and disables growing/splitting/pruning
specifically (refine_start_iter pushed past cfg.iters, so DefaultStrategy's
own grow/prune gate never trips) -- the Gaussian count stays fixed all
run, its (u, v, layer) origin in that view's depth grid tracked
throughout, so the final params can be scattered back into a
fit_gsplat.py-schema grid .h5 (save_output_layered) alongside the usual
flat output. Periodic OPACITY RESET is a separate mechanism (gated by
reset_every, not refine_start_iter) and is deliberately left running --
it mutates existing Gaussians' opacity in place, never adds/removes/
reorders any, so it doesn't disturb the fixed layer stack at all.
Everything else (loss, optimizers, training loop, SH color) is unchanged
from `set` mode -- the params stay a plain flat
ParameterDict either way; "layered" only changes how they're seeded, that
densify/prune never runs, and one extra file gets written at the end.

One more layered-only difference: means is means_original (frozen, the
depth-peel-seeded position) + an optimizable offset (zero-initialized --
training starts exactly at the seed), not a single free tensor like
`set` mode's. self.params["means"] holds the offset itself (still
trainable, still what gsplat.strategy's optimizer/check_sanity plumbing
expects at that key); Fit3DGSLightningModule._active_means() is the one
place that resolves means_original + offset into the actual position
every render/preview/eval/output call needs. cfg.mean_offset_reg_weight
adds an L2 penalty on the offset (not the resolved position) to the
train loss, pulling Gaussians back toward their seeded origin instead of
letting the offset drift unconstrained.

cfg.background (none/white/random) picks what's composited under the
render and, wherever real alpha exists (hdf5_path; COLMAP photos have
none), recomposited into gt_rgb to match -- see conf/fit_3dgs.yaml's own
comment for the none/white/random split and why "random" only applies to
the train stage.

DefaultStrategy ships with one real bug in the exact pinned version this
repo vendors (v1.5.3 + one commit, see gsplat-src/'s own comments): its
opacity-reset scheduling condition was dead code from a `&`/`and`
operator-precedence mistake, fixed in our vendored copy. As shipped
(unpatched), DO NOT trust it blindly -- see that dir's comments for the
exact diff against the original PyPI release before relying on it elsewhere.

Scene loading (COLMAP parsing, camera math, cameras_extent) reuses
gaussian-splatting/'s own scene.dataset_readers / utils.graphics_utils
directly (pure file-IO/math, no architectural coupling) rather than
reimplementing COLMAP binary parsing; everything else (the Dataset/
DataModule, the training loop, losses, logging) is this repo's own.
gaussian-splatting/ here is trimmed to just those pure-Python files (plus
lpipsPyTorch/, for the final eval) -- not the full reference repo, and
none of its CUDA extensions (diff-gaussian-rasterization/simple-knn/
fused-ssim) are vendored or built; this file renders with gsplat only.

Rendering is gsplat (packed=True, full SH -- sh0/shN, active degree ramped
by oneupSHdegree() every 1000 iters same as the reference), NOT
diff-gaussian-rasterization. COLMAP's R/T are already world-to-camera in
OpenCV convention (unlike fit_gsplat.py's Blender/OpenGL renders), so
getWorld2View2()'s output is used directly as gsplat's viewmat -- no
OPENGL_TO_OPENCV flip needed here.

Trained as a `pl.LightningModule` (Fit3DGSLightningModule) with MANUAL
optimization (`automatic_optimization = False`): gsplat's ops need to run
between `backward()` and `optimizer.step()` (to grow/shrink the per-Gaussian
Adam state before the step that would otherwise mismatch shapes), which
Lightning's automatic-optimization hook ordering doesn't expose. Shares its
loss/OOM-handling/PanelCallback/OrbitCallback building blocks with
fit_gsplat.py/train_gs.py via module.py -- see that module's docstring.

Held-out test views (llffhold=8, same as the reference's --eval) are used
both as periodic val/loss* metrics during training AND for a final
PSNR/SSIM/LPIPS(vgg) pass at the end (cfg.final_eval), logged to wandb and
written as results.json next to the output -- the same methodology
gaussian-splatting/metrics.py uses, so a run is self-checking against the
numbers we already reproduced manually.

Config is Hydra (`conf/fit_3dgs.yaml`); run in the container venv, e.g.

    docker exec -w /app gsviawt-3dgs-repro-app-gpu-1 python fit_3dgs.py \\
        source_path=/app/data/tandt/truck

gsplat's CUDA kernels are prebuilt into the image (see Dockerfile);
CUDA_HOME/PATH/TORCH_CUDA_ARCH_LIST are already set at the image level,
unlike fit_gsplat.py's docstring (stale) suggests.
"""

import json
import logging
import os
import sys
import warnings

import h5py
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import wandb

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
from module import DSSIMLoss, OPENGL_TO_OPENCV  # noqa: E402
from gs_dataset import H5Catalog, _EmptyDataset, split_by_view  # noqa: E402
from util import timed  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "gaussian-splatting"))
from scene.dataset_readers import readColmapSceneInfo  # noqa: E402
from utils.graphics_utils import BasicPointCloud, getWorld2View2, fov2focal  # noqa: E402
from utils.sh_utils import RGB2SH, SH2RGB  # noqa: E402
from utils.general_utils import get_expon_lr_func  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# scene loading -- COLMAP parsing reused from gaussian-splatting/ verbatim
# (scene.dataset_readers.readColmapSceneInfo); everything downstream (image
# loading + resize, viewmat/K, into our own per-camera dicts) is ours,
# replacing the reference's Camera/Scene classes with something that fits a
# plain Dataset.
# ---------------------------------------------------------------------------

def _resolution_for(orig_w, orig_h, resolution):
  """Mirrors gaussian-splatting/utils/camera_utils.py's loadCam exactly for
  resolution in {"auto", 1, 2, 4, 8}: "auto" (the reference's -1) auto-halves
  down to <=1600px width (warns once per call site -- Python's default
  warnings filter already does that, no manual bookkeeping needed),
  {1,2,4,8} divide directly. No resolution_scale (this repo never uses
  multi-resolution-scale training) and no arbitrary-float branch (the
  reference's "else: global_down = orig_w / args.resolution", for a literal
  target width -- not used by any of our configs)."""
  if resolution in (1, 2, 4, 8):
    return round(orig_w / resolution), round(orig_h / resolution)
  assert resolution == "auto", f"unsupported resolution {resolution!r} (use \"auto\", 1, 2, 4, or 8)"
  if orig_w > 1600:
    warnings.warn("Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.")
    global_down = orig_w / 1600
  else:
    global_down = 1
  return int(orig_w / global_down), int(orig_h / global_down)


def _coerce_resolution(x):
  """Hydra hands this through as either the string "auto" or a number
  (1/2/4/8) -- only the latter should be int()-cast."""
  return x if str(x) == "auto" else int(x)


def _load_camera(cam_info, resolution):
  """cam_info: a gaussian-splatting CameraInfo (R, T, FovX, FovY, image_path,
  width/height at COLMAP's original resolution). Returns a dict: 'image'
  (H,W,3) float32 in [0,1] at the resolution actually used, 'alpha' (H,W,1)
  float32, always 1 (real photos have no matte -- see cfg.background), 'viewmat'
  (4,4) world-to-camera (OpenCV convention, straight from getWorld2View2 --
  COLMAP's R/T already ARE this, unlike fit_gsplat.py's OpenGL renders), 'K'
  (3,3) (assumes centered principal point, same as the reference's FoV-only
  camera model), 'name' (for results.json / logging)."""
  w, h = _resolution_for(cam_info.width, cam_info.height, resolution)
  # Plain .resize(), no explicit resample filter -- matches
  # utils/general_utils.py's PILtoTorch exactly (PIL's own default), so
  # training targets and final-eval GT are the identical pixels the
  # reference trains/evaluates against.
  image = np.asarray(Image.open(cam_info.image_path).convert("RGB").resize((w, h)))
  image = image.astype(np.float32) / 255.0
  alpha = np.ones_like(image[..., :1])  # COLMAP captures have no matte -- always fully opaque

  viewmat = getWorld2View2(cam_info.R, cam_info.T)
  fx, fy = fov2focal(cam_info.FovX, w), fov2focal(cam_info.FovY, h)
  K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], np.float32)

  return {"image": image, "alpha": alpha, "viewmat": viewmat, "K": K,
          "name": cam_info.image_name, "width": w, "height": h}


def load_scene(source_path, images, eval_split, resolution):
  """Returns (pcd, scene_extent, train_cams, test_cams). pcd: gaussian-
  splatting's BasicPointCloud (points/colors/normals, from sparse/0/
  points3D). scene_extent: the NerfppNorm radius (== the reference's
  scene.cameras_extent), from the TRAIN cameras only (matches
  readColmapSceneInfo). train_cams/test_cams: lists of _load_camera() dicts,
  sorted by image_name (matches the reference's own sort, so an llffhold=8
  split lands on the identical images)."""
  info = readColmapSceneInfo(source_path, images, "", eval_split, False)
  train_cams = [_load_camera(c, resolution) for c in info.train_cameras]
  test_cams = [_load_camera(c, resolution) for c in info.test_cameras]
  return info.point_cloud, float(info.nerf_normalization["radius"]) or 1.0, train_cams, test_cams


def load_scene_h5(hdf5_path, val_fraction, seed):
  """Alternative to load_scene() for a render_objaverse.py-schema HDF5
  (`images`, `image_intrinsics`, `camera_pose` -- one object, no COLMAP/SfM
  point cloud) instead of a real COLMAP capture. Returns the exact same
  (pcd, scene_extent, train_cams, test_cams) shape load_scene() does, so
  nothing downstream (init_gaussians, Fit3DGSViewDataset, training loop,
  run_final_eval) needs to know which loader ran.

  Split: H5Catalog + gs_dataset.split_by_view (this project's own train/val
  convention, same one fit_gsplat.py/train_gs.py use) in place of COLMAP's
  llffhold=8 -- there's no `images` folder / sorted filenames to hold out
  every 8th of here, just a flat view axis.

  Camera convention: `camera_pose` is camera-to-world in Blender/OpenGL
  axes (X right, Y up, Z back), same as gs_dataset.py's/fit_gsplat.py's
  renders -- flip via OPENGL_TO_OPENCV then invert (NOT getWorld2View2,
  which assumes COLMAP's own already-OpenCV R/T). Matches
  gaussian-splatting's own readCamerasFromTransforms (`c2w[:3,1:3] *= -1`)
  applied to exactly this kind of Blender-synthetic camera.

  Background: 'image' is alpha-premultiplied over BLACK regardless of
  cfg.background ('alpha' is kept alongside it, raw, unpremultiplied) --
  cfg.background != "none" recomposites gt_rgb onto the chosen colour, and
  passes the same colour as `backgrounds=` to gsplat.rasterization, inside
  Fit3DGSLightningModule._step/run_final_eval, not here. Default matches
  ModelParams' `white_background` DEFAULT (False) == cfg.background=none.

  Point cloud seed: no SfM points exist for a synthetic render, so this
  mirrors gaussian-splatting/scene/dataset_readers.readNerfSyntheticInfo's
  OWN fallback for exactly this situation (Blender synthetic scenes with no
  points3d.ply) -- 100_000 uniform-random points in a box, SH0 colour from
  near-zero random values (SH2RGB of shs in [0, 1/255) -- a quirk of the
  reference's own code, reproduced verbatim, not something to "fix" here).
  Box half-extent (0.75) isn't the reference's literal 1.3 -- that's sized
  for standard NeRF-synthetic scenes' own (larger) camera/object scale.
  0.75 instead comes from blender_script.py's own normalize_scene(), which
  guarantees every rendered object's bounding box max dimension is exactly
  1.0 (the object itself sits inside [-0.5, 0.5]^3): 0.75 gives the same
  kind of generous margin around the actual object that 1.3 gives around
  the reference's own scenes, for the optimizer to grow/prune into."""
  catalog = H5Catalog(
    hdf5_path, H5Catalog.path().alias("path"), H5Catalog.index().alias("view_idx"),
    H5Catalog.dataset("mesh_index").alias("mesh_id"))
  train_catalog, val_catalog = split_by_view(catalog, val_fraction=val_fraction, seed=seed)
  train_idx = sorted(int(r["view_idx"]) for r in train_catalog.df.iter_rows(named=True))
  val_idx = sorted(int(r["view_idx"]) for r in val_catalog.df.iter_rows(named=True))

  with h5py.File(hdf5_path, "r") as f:
    images = np.asarray(f["images"])                          # (V,H,W,3|4) u8
    K = np.asarray(f["image_intrinsics"]).astype(np.float32)  # (V,3,3)
    pose = np.asarray(f["camera_pose"]).astype(np.float32)    # (V,4,4) c2w, OpenGL axes

  def _cam(i):
    img = images[i].astype(np.float32) / 255.0
    if img.shape[-1] == 4:
      rgb, alpha = img[..., :3], img[..., 3:4]
    else:
      rgb, alpha = img, np.ones_like(img[..., :1])
    image = np.ascontiguousarray(rgb * alpha)  # premultiplied over black
    viewmat = np.linalg.inv(pose[i] @ OPENGL_TO_OPENCV).astype(np.float32)
    h, w = image.shape[:2]
    # view_idx: this view's row in the h5 (not just its position within the
    # split) -- gaussian_layout="layered" (init_gaussians_layered) needs it
    # to re-read this same view's raw depth_peel, which load_scene_h5 itself
    # never loads (COLMAP cameras have no equivalent field).
    return {"image": image, "alpha": alpha, "viewmat": viewmat, "K": K[i], "name": f"view{i:03d}",
            "width": w, "height": h, "view_idx": i}

  train_cams = [_cam(i) for i in train_idx]
  test_cams = [_cam(i) for i in val_idx]

  centers = pose[train_idx][:, :3, 3]
  center = centers.mean(axis=0)
  scene_extent = float(np.max(np.linalg.norm(centers - center, axis=1)) * 1.1) or 1.0

  rng = np.random.default_rng(seed)
  num_pts, half_extent = 100_000, 0.75
  xyz = (rng.random((num_pts, 3), dtype=np.float32) * (2 * half_extent) - half_extent)
  shs = rng.random((num_pts, 3), dtype=np.float32) / 255.0
  pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3), np.float32))

  return pcd, scene_extent, train_cams, test_cams


# ---------------------------------------------------------------------------
# gaussian init -- create_from_pcd, ported: SH0 from point colour, SH-rest
# zeroed, isotropic initial scale from same-order neighbour distance (cKDTree
# in place of simple_knn's distCUDA2 CUDA kernel -- same quantity: distCUDA2
# returns the MEAN SQUARED distance to the knn_k nearest neighbours, and the
# reference takes scale = sqrt(that) -- i.e. the RMS distance, not the
# arithmetic mean of the (unsquared) distances cKDTree.query returns
# directly. Already this repo's convention, see fit_gsplat.py's own
# init_gaussians, just squared/rooted correctly here to match distCUDA2).
# ---------------------------------------------------------------------------

def _logit(x, eps=1e-4):
  x = np.clip(x, eps, 1.0 - eps)
  return np.log(x / (1.0 - x))


def init_gaussians(pcd, max_sh_degree, init_opacity, knn_k):
  pts = np.asarray(pcd.points, np.float32)
  colors = np.asarray(pcd.colors, np.float32)
  n = len(pts)
  log.info("Number of points at initialisation: %d", n)

  from scipy.spatial import cKDTree
  k = min(knn_k + 1, n)
  dist, _ = cKDTree(pts).query(pts, k=k)
  dist = rearrange(dist, "p -> p 1") if dist.ndim == 1 else dist
  neighbours = dist[:, 1:] if k > 1 else dist  # column 0 is the point itself
  nn = np.sqrt(np.clip((neighbours ** 2).mean(axis=1), 1e-7, None)).astype(np.float32)

  num_sh = (int(max_sh_degree) + 1) ** 2
  sh0 = RGB2SH(torch.from_numpy(colors)).numpy()[:, None, :].astype(np.float32)  # (P,1,3)
  shN = np.zeros((n, num_sh - 1, 3), np.float32)

  return {
    "means": pts,
    "scales": repeat(np.log(nn), "p -> p xyz", xyz=3).astype(np.float32),
    "quats": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
    "opacities": np.full(n, _logit(np.float32(init_opacity)), np.float32),
    "sh0": sh0, "shN": shN,
  }


# ---------------------------------------------------------------------------
# gaussian init -- LAYERED alternative to init_gaussians() above, for
# gaussian_layout="layered" (h5 scenes with depth_peel only -- see
# load_scene_h5's own guard in main()). One Gaussian per depth-peel hit in
# the PRIMARY train view's 6-layer peel (fit_gsplat.py's own convention:
# "whichever [view] sorts first in the (already-split) catalog" --
# train_cams[0]["view_idx"], since load_scene_h5 already sorts train_idx).
# Every Gaussian keeps a fixed (u, v, layer) origin in that view's depth
# grid, so the final flat param set can be scattered back into a
# fit_gsplat.py-schema (H, W, L, .) grid (see save_output_layered) --
# that 1:1 correspondence is also exactly why densification (which adds/
# removes Gaussians) is incompatible with this mode and must stay off.
#
# Reuses module.init_gaussians() verbatim for the actual unprojection/
# KNN-scale/front-pixel-colour work (same quantity, same seeding scene) --
# only the colour representation differs downstream: fit_gsplat's was flat
# sigmoid (colors_logit), this file trains full SH (sh0/shN, degree
# ramped), so the seed colour is converted logit->sigmoid->RGB2SH into sh0
# with shN left at zero, exactly like init_gaussians()'s own pcd-colour
# conversion above.
# ---------------------------------------------------------------------------

def init_gaussians_layered(hdf5_path, primary_view_idx, max_sh_degree, init_opacity, knn_k):
  with h5py.File(hdf5_path, "r") as f:
    if "depth_peel" not in f or "depth_intrinsics" not in f:
      raise SystemExit(
        f"{hdf5_path}: no depth_peel/depth_intrinsics -- gaussian_layout=layered "
        "needs a render_objaverse.py h5 with depth data, not just images")
    depth = np.asarray(f["depth_peel"][primary_view_idx]).astype(np.float32)   # (H,W,L)
    K = np.asarray(f["depth_intrinsics"][primary_view_idx]).astype(np.float32)  # (3,3)
    pose = np.asarray(f["camera_pose"][primary_view_idx]).astype(np.float32)    # (4,4) c2w
    image = np.asarray(f["images"][primary_view_idx])                          # (IH,IW,3|4) u8
    mesh_index = int(f["mesh_index"][primary_view_idx]) if "mesh_index" in f else -1
    mesh_path = ""
    if "mesh_paths" in f and 0 <= mesh_index < f["mesh_paths"].shape[0]:
      mp = f["mesh_paths"][mesh_index]
      mesh_path = mp.decode() if isinstance(mp, bytes) else str(mp)

  from module import init_gaussians as _seed_from_depth_peel
  seed = _seed_from_depth_peel(depth, K, pose, image, knn_k, init_opacity)
  n = len(seed["means"])
  log.info("layered seed: %d Gaussians from the primary view's %dx%d, %d-layer depth peel",
           n, depth.shape[1], depth.shape[0], depth.shape[2])

  num_sh = (int(max_sh_degree) + 1) ** 2
  colors = 1.0 / (1.0 + np.exp(-seed["colors_logit"]))  # logit -> flat RGB in [0,1]
  sh0 = RGB2SH(torch.from_numpy(colors)).numpy()[:, None, :].astype(np.float32)
  shN = np.zeros((n, num_sh - 1, 3), np.float32)

  g = {
    "means": seed["means"], "scales": seed["scales_log"], "quats": seed["quats"],
    "opacities": seed["opac_logit"], "sh0": sh0, "shN": shN,
  }
  meta = {
    "u": seed["u"], "v": seed["v"], "layer": seed["layer"],
    "H": depth.shape[0], "W": depth.shape[1], "L": depth.shape[2],
    "mesh_index": mesh_index, "mesh_path": mesh_path,
  }
  return g, meta


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(params, active_sh_degree, viewmats, Ks, width, height, packed=True, backgrounds=None):
  """backgrounds: None (default) -- un-composited, i.e. premultiplied by the
  rendered alpha (== rendered over black), matching gt_rgb under
  cfg.background="none". (3,) otherwise -- ONE colour shared by every
  camera in this call (gsplat's packed=True constraint, see
  Fit3DGSLightningModule._step's own comment) -- gsplat composites it in
  directly; the caller is responsible for recompositing gt_rgb onto the
  SAME colour first (see Fit3DGSLightningModule._step/run_final_eval)."""
  import gsplat
  colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (P,K,3)
  rgb, alpha, info = gsplat.rasterization(
    means=params["means"], quats=F.normalize(params["quats"], dim=-1),
    scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"]),
    colors=colors, sh_degree=active_sh_degree,
    viewmats=viewmats, Ks=Ks, width=width, height=height,
    render_mode="RGB", packed=packed, backgrounds=backgrounds,
  )
  return rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0), info


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def write_ply(path, params_np):
  """Standard INRIA-format 3DGS .ply with full SH -- gsplat.export_splats
  writes every field raw (viewers apply exp(scale)/sigmoid(opacity)
  themselves), so pass the unactivated optimizer params directly."""
  import gsplat
  quats = params_np["quats"] / np.linalg.norm(params_np["quats"], axis=-1, keepdims=True)
  gsplat.export_splats(
    means=torch.from_numpy(params_np["means"]),
    scales=torch.from_numpy(params_np["scales"]),
    quats=torch.from_numpy(quats.astype(np.float32)),
    opacities=torch.from_numpy(params_np["opacities"]),
    sh0=torch.from_numpy(params_np["sh0"]),
    shN=torch.from_numpy(params_np["shN"]),
    format="ply", save_to=path,
  )


def save_output(path, cfg, params_np, scene_extent, final_loss, n_seeded):
  with h5py.File(path, "w") as f:
    f.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    f.attrs["final_loss"] = float(final_loss)
    f.attrs["num_gaussians"] = int(len(params_np["means"]))
    f.attrs["num_gaussians_seeded"] = int(n_seeded)
    f.attrs["scene_extent"] = float(scene_extent)
    for k, v in params_np.items():
      f.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def save_output_layered(path, params_np, meta):
  """gaussian_layout="layered" only: an ADDITIONAL fit_gsplat.py-schema
  (H, W, L, .) grid .h5 (gaussian_means/scales/quats/opacities/colors +
  layer_valid), scattered from the same flat params save_output() also
  wrote flat -- written alongside, never instead of, that flat .h5/.ply
  (which stay format-agnostic and are what the Chamfer/disparity tooling
  already consumes). Lets orbit_video.py and anything else built for
  fit_gsplat.py's own output consume a layered fit_3dgs.py run unmodified.

  colors here is DC-only (SH_C0*sh0 + 0.5) -- fit_gsplat.py's grid schema
  has no SH slot at all (flat colour), so any learned view-dependent shN
  this file trained is dropped for this specific output; the full-SH
  colour survives in the flat .h5/.ply from save_output()/write_ply()."""
  from module import SH_C0, _scatter_grid
  means = params_np["means"]
  scales = np.exp(params_np["scales"])
  quats = params_np["quats"] / np.linalg.norm(params_np["quats"], axis=-1, keepdims=True)
  opac = 1.0 / (1.0 + np.exp(-params_np["opacities"]))
  colors = np.clip(SH_C0 * params_np["sh0"][:, 0, :] + 0.5, 0.0, 1.0)

  v, u, layer = meta["v"], meta["u"], meta["layer"]
  H, W, L = meta["H"], meta["W"], meta["L"]
  with h5py.File(path, "w") as f:
    f.attrs["mesh_index"] = meta["mesh_index"]
    f.attrs["mesh_path"] = meta["mesh_path"]
    f.attrs["layer_layout"] = "(H, W, 6) like depth_peel; NaN = empty slot"

    def _scene_dataset(name, data):
      f.create_dataset(name, data=data, chunks=data.shape, compression="gzip", compression_opts=4)

    _scene_dataset("gaussian_means", _scatter_grid(means, v, u, layer, H, W, L))
    _scene_dataset("gaussian_scales", _scatter_grid(scales, v, u, layer, H, W, L))
    _scene_dataset("gaussian_quats", _scatter_grid(quats, v, u, layer, H, W, L))
    _scene_dataset("gaussian_opacities", _scatter_grid(opac, v, u, layer, H, W, L))
    _scene_dataset("gaussian_colors", _scatter_grid(colors, v, u, layer, H, W, L))
    valid = np.zeros((H, W, L), bool)
    valid[v, u, layer] = True
    f.create_dataset("layer_valid", data=valid)


# ---------------------------------------------------------------------------
# dataset / lightning module
# ---------------------------------------------------------------------------

class Fit3DGSViewDataset(Dataset):
  """One item = one camera's photometric target -- image/alpha already
  loaded/resized by load_scene()/load_scene_h5(), viewmat/K precomputed.
  gt_alpha is always 1 for COLMAP captures (no matte); cfg.background !=
  "none" recomposites gt_rgb onto the configured colour using it (see
  Fit3DGSLightningModule._step) -- a no-op wherever alpha is all-1."""

  def __init__(self, cams):
    self.cams = cams

  def __len__(self):
    return len(self.cams)

  def __getitem__(self, i):
    c = self.cams[i]
    return {
      "gt_rgb": torch.from_numpy(c["image"]),
      "gt_alpha": torch.from_numpy(c["alpha"]),
      "K": torch.from_numpy(c["K"]),
      "viewmat": torch.from_numpy(c["viewmat"]),
    }


class Fit3DGSDataModule(pl.LightningDataModule):
  def __init__(self, cfg, train_cams, test_cams):
    super().__init__()
    self.cfg = cfg
    self.train_ds = Fit3DGSViewDataset(train_cams)
    self.val_ds = Fit3DGSViewDataset(test_cams) if test_cams else _EmptyDataset()

  def train_dataloader(self):
    return DataLoader(self.train_ds, shuffle=True, **OmegaConf.to_container(self.cfg.loader, resolve=True))

  def val_dataloader(self):
    return DataLoader(self.val_ds, shuffle=False, **OmegaConf.to_container(self.cfg.loader, resolve=True))


class Fit3DGSLightningModule(pl.LightningModule):
  automatic_optimization = False

  def __init__(self, cfg, g, scene_extent, preview_cams):
    super().__init__()
    self.cfg = cfg
    self.scene_extent = scene_extent
    self.max_sh_degree = int(cfg.sh_degree)
    self.active_sh_degree = 0
    self.views_seen = 0
    self.final_loss = float("nan")
    self.step_count = 0
    self.gaussian_layout = str(cfg.gaussian_layout)

    # gaussian_layout="layered": means = means_original (frozen, the
    # depth-peel-seeded position -- see init_gaussians_layered) + an
    # optimizable offset (means_original.zeros_like init, so training
    # starts exactly at the seed). means_original is a buffer, not a
    # Parameter -- never touched by any optimizer, no .grad, moved to
    # the right device automatically by Lightning's own .to() (like
    # self.params already is) since register_buffer participates in
    # that the same way parameters do. self.params["means"] keeps its
    # key name (gsplat.strategy.DefaultStrategy.check_sanity asserts
    # "means" is present in both params and optimizers_dict; densify/
    # prune never touches it in this mode anyway -- see the
    # refine_start_iter comment below) but now holds the OFFSET, not
    # the position -- _active_means() is the one place that resolves
    # the two into the actual position every render call needs; nothing
    # else should read self.params["means"] directly as if it were one.
    self.mean_offset_reg_weight = float(cfg.mean_offset_reg_weight)
    self.params = nn.ParameterDict()
    for k, v in g.items():
      if k == "means" and self.gaussian_layout == "layered":
        self.register_buffer("means_original", torch.from_numpy(v), persistent=False)
        self.params["means"] = nn.Parameter(torch.zeros_like(self.means_original))
      else:
        self.params[k] = nn.Parameter(torch.from_numpy(v))
    self.dssim = DSSIMLoss()

    # gaussian_layout="layered" keeps a fixed 1:1 Gaussian<->(u,v,layer)
    # correspondence (see init_gaussians_layered/save_output_layered) --
    # growing/splitting/pruning would break that, so those specifically
    # must never fire. DefaultStrategy's step_post_backward ALSO does
    # periodic opacity reset (gated independently by reset_every, not by
    # refine_start_iter/refine_every) -- resetting a Gaussian's opacity
    # in place doesn't add/remove/reorder anything, so it's perfectly
    # compatible with a fixed layer stack and stays on. Rather than
    # skip the strategy object entirely (which would also silently kill
    # that reset), only refine_start_iter is pushed past cfg.iters --
    # grow/prune's own "step > refine_start_iter" gate then never trips
    # for the whole run, while everything else in step_post_backward
    # (opacity reset, the per-step grad2d/count bookkeeping) runs exactly
    # as it would in "set" mode.
    import gsplat.strategy as gsstrat
    refine_start_iter = (int(cfg.iters) if self.gaussian_layout == "layered"
                        else int(cfg.densify.from_iter))
    # opacity_reset_interval: null (conf/fit_3dgs.yaml) disables periodic
    # reset entirely. DefaultStrategy itself has no boolean "off" for this
    # -- reset_every is a plain int, gated only by `step % reset_every ==
    # 0 and step > 0` (checked against gsplat-src directly: no disable
    # flag anywhere in the library or its own examples) -- so "disabled"
    # is expressed as a reset_every value that provably can't be hit by
    # any step in this run: cfg.iters + 1 exceeds every step < iters, and
    # step 0 is already excluded by the strategy's own `step > 0` guard.
    # This conversion lives here, once, so callers say what they mean
    # (null) instead of having to know/pass that sentinel themselves.
    reset_every = (int(cfg.iters) + 1 if cfg.opacity_reset_interval is None
                  else int(cfg.opacity_reset_interval))
    self.strategy = gsstrat.DefaultStrategy(
      prune_opa=float(cfg.densify.prune_opacity), grow_grad2d=float(cfg.densify.grad_threshold),
      grow_scale3d=float(cfg.percent_dense), prune_scale3d=float(cfg.densify.prune_scale3d),
      refine_start_iter=refine_start_iter, refine_stop_iter=int(cfg.densify.until_iter),
      refine_every=int(cfg.densify.interval), reset_every=reset_every,
      absgrad=False, revised_opacity=False,
    )
    # initialize_state()'s tensors are created lazily (None until the first
    # step_post_backward call, then allocated directly on whatever device
    # the render `info` is on) -- no manual device placement needed here,
    # unlike a plain dict of eagerly-created tensors would.
    self.strategy_state = self.strategy.initialize_state(scene_scale=scene_extent)

    # A fixed handful of preview cameras (module.PanelCallback/OrbitCallback,
    # via get_preview_source() below) -- not the full train/val split (that's
    # GSFitDataModule's job), just enough to render a comparison panel.
    self._preview = {
      k: {
        "viewmat": torch.from_numpy(np.stack([c["viewmat"] for c in cams])),
        "K": torch.from_numpy(np.stack([c["K"] for c in cams])),
        "gt_rgb": torch.from_numpy(np.stack([c["image"] for c in cams])),
      }
      for k, cams in preview_cams.items() if cams
    }

  def _active_means(self):
    """The actual world-space Gaussian positions -- self.params["means"]
    directly in "set" mode, means_original + self.params["means"] (now an
    offset) in "layered" mode. The only place that resolves the two;
    render()/get_preview_source()/run_final_eval()/params_np() all go
    through this instead of reading self.params["means"] as if it were
    always a position."""
    if self.gaussian_layout == "layered":
      return self.means_original + self.params["means"]
    return self.params["means"]

  def configure_optimizers(self):
    cfg = self.cfg
    # max_steps here is its OWN config value (default 30000), not cfg.iters:
    # the reference's position_lr_max_steps is an independent hyperparameter
    # that just happens to default to the same number as --iterations --
    # tying it to cfg.iters would make a shorter debug run (e.g. iters=700)
    # decay the xyz LR to near-zero within those 700 steps instead of
    # reproducing the first 700 steps of a real 30000-iter run.
    self.means_lr_fn = get_expon_lr_func(
      lr_init=float(cfg.lr.means_init) * self.scene_extent,
      lr_final=float(cfg.lr.means_final) * self.scene_extent,
      lr_delay_mult=float(cfg.lr.means_delay_mult),
      max_steps=int(cfg.lr.means_max_steps),
    )
    self.optimizers_dict = {
      "means": torch.optim.Adam([self.params["means"]], lr=self.means_lr_fn(0), eps=1e-15),
      "sh0": torch.optim.Adam([self.params["sh0"]], lr=float(cfg.lr.features_dc), eps=1e-15),
      "shN": torch.optim.Adam([self.params["shN"]], lr=float(cfg.lr.features_rest), eps=1e-15),
      "opacities": torch.optim.Adam([self.params["opacities"]], lr=float(cfg.lr.opacities), eps=1e-15),
      "scales": torch.optim.Adam([self.params["scales"]], lr=float(cfg.lr.scales), eps=1e-15),
      "quats": torch.optim.Adam([self.params["quats"]], lr=float(cfg.lr.quats), eps=1e-15),
    }
    self.strategy.check_sanity(self.params, self.optimizers_dict)
    return list(self.optimizers_dict.values())

  def _photom_loss(self, rgb, gt_rgb):
    l1 = (rgb - gt_rgb).abs().mean()
    dssim = self.dssim(rgb.permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2))
    ld = float(self.cfg.lambda_dssim)
    return (1 - ld) * l1 + ld * dssim, {"l1": l1, "dssim": dssim}

  def training_step(self, batch, batch_idx):
    return self._step("train", batch, batch_idx)

  def validation_step(self, batch, batch_idx):
    return self._step("val", batch, batch_idx)

  def _step(self, stage, batch, batch_idx):
    """Shared by training_step/validation_step (fit_gsplat.py/train_gs.py's
    own convention -- one _step, not two near-identical copies: that's how
    views_seen ended up logged for train but silently never for val, a
    literal duplicated-code bug, not a deliberate omission). Everything
    that mutates optimizer/strategy state (backward, the 6 optimizers'
    .step(), gsplat's grow/prune/reset, the means-LR/SH-degree schedule,
    should_stop) stays train-only -- gated by `stage == "train"` inline,
    same as fit_gsplat.py gates its own train-only bits (random_bg,
    final_loss) inside its shared _step."""
    def _log(key, *args, **kwargs):
      self.log(f"{stage}/{key}", *args, **kwargs, batch_size=B)

    gt_rgb, gt_alpha, K, viewmat = batch["gt_rgb"], batch["gt_alpha"], batch["K"], batch["viewmat"]
    B = gt_rgb.shape[0]
    H, W = gt_rgb.shape[1:3]
    it = self.step_count

    # cfg.background: see conf/fit_3dgs.yaml's own comment. "random" is
    # train-only (an augmentation, like train_gs.py's loss.random_bg) --
    # val/final_eval stay on bg=None so metrics are comparable run over
    # run; "white" applies to every stage, a scene convention not an
    # augmentation, like the reference's own white_background flag.
    #
    # bg is a single (3,) colour, not (B,3) -- gsplat's packed=True
    # rasterizer wants backgrounds shaped just (channels,), ONE colour
    # shared by every camera in the call (verified against train_gs.py's
    # own render(), which hits the exact same gsplat constraint); with
    # this file's default loader.batch_size=1 that's one random colour
    # per step anyway, same granularity train_gs.py's per-item loop gets.
    bg_mode = str(self.cfg.background)
    if bg_mode == "white":
      bg = torch.ones(3, device=gt_rgb.device)
    elif bg_mode == "random" and stage == "train":
      bg = torch.rand(3, device=gt_rgb.device)
    else:
      bg = None
    if bg is not None:
      # gt_rgb is already alpha-premultiplied over BLACK -- recompositing
      # onto a different background is just += bg*(1-alpha), no
      # un-premultiply needed (same identity train_gs.py's _step uses).
      gt_rgb = gt_rgb + bg[None, None, None, :] * (1.0 - gt_alpha)

    if stage == "train":
      self.step_count += 1
      iteration = it + 1

      # trainer.max_steps is NOT enforced under manual optimization
      # (Lightning ties it to the automatic-optimization step-counting
      # path, which we bypass entirely) -- with trainer.max_epochs=-1 as
      # the only other stop condition, an unguarded run here is unbounded
      # (confirmed the hard way: a 300-iters config ran ~700 epochs
      # before OOM-thrashing on a runaway Gaussian count). Signal stop
      # explicitly once our own 1-indexed `iteration` reaches cfg.iters
      # -- Lightning checks should_stop within the epoch loop, not just
      # between epochs, so this still stops promptly mid-epoch rather
      # than running to that epoch's end.
      if iteration >= int(self.cfg.iters):
        self.trainer.should_stop = True

      if iteration % 1000 == 0 and self.active_sh_degree < self.max_sh_degree:
        self.active_sh_degree += 1
      self.optimizers_dict["means"].param_groups[0]["lr"] = self.means_lr_fn(it)

    with torch.set_grad_enabled(stage == "train"):
      # render_params: self.params with "means" resolved to the actual
      # position (_active_means()) -- a shallow dict, so every other key
      # is still the exact same Parameter tensor render()/gsplat read
      # elsewhere. self.params itself (offset still at "means" in
      # layered mode) is what goes to the strategy/optimizers below --
      # they operate on the trainable leaf, not the resolved position.
      render_params = {**self.params, "means": self._active_means()}
      rgb, alpha, info = render(render_params, self.active_sh_degree, viewmat, K, W, H, backgrounds=bg)
      if stage == "train":
        # step_pre_backward calls info["means2d"].retain_grad() -- must
        # run before backward for a non-leaf tensor to keep its .grad
        # populated.
        self.strategy.step_pre_backward(self.params, self.optimizers_dict, self.strategy_state, it, info)
      # photom_loss: the L1+D-SSIM combo alone, logged as its own metric
      # (loss/photom below) so it stays readable independent of whatever
      # non-photometric terms (currently just mean_offset_reg) get folded
      # into the full `loss` that's actually optimized -- fold additional
      # terms into `loss` here, never back into photom_loss itself.
      photom_loss, parts = self._photom_loss(rgb, gt_rgb)
      loss = photom_loss
      offset_reg = None
      if stage == "train" and self.gaussian_layout == "layered" and self.mean_offset_reg_weight > 0:
        # L2 penalty on the offset itself (not the resolved position) --
        # pulls Gaussians back toward their depth-peel-seeded origin,
        # the whole point of splitting means into a frozen original +
        # an optimizable offset instead of leaving means fully free.
        # NOT folded into `parts` (that loop below logs everything in it
        # under "loss/photom_*" -- this isn't a photometric term).
        offset_reg = self.params["means"].pow(2).sum(-1).mean()
        loss = loss + self.mean_offset_reg_weight * offset_reg

    if stage == "train":
      for opt in self.optimizers_dict.values():
        opt.zero_grad(set_to_none=True)
      self.manual_backward(loss)

      for opt in self.optimizers_dict.values():
        opt.step()

      # Post-backward, post-step (matches gsplat's own reference trainer's
      # ordering, examples/simple_trainer.py -- grow/prune happens AFTER
      # this step's gradient update is applied, using the gradient info
      # accumulated THIS step, so it acts on next step's population, not
      # this one's). Always called, in both gaussian_layout modes:
      # __init__ is what actually disables grow/prune for "layered"
      # (refine_start_iter pushed past cfg.iters), not a guard here --
      # opacity reset still needs to run.
      self.strategy.step_post_backward(
        self.params, self.optimizers_dict, self.strategy_state, it, info, packed=True)

      self.views_seen += B
      self.final_loss = loss.item()
      if iteration % int(self.cfg.densify.interval) == 0:
        log.info("iter %d: loss=%.5f  %d Gaussians", iteration, loss.item(), len(self.params["means"]))

    _log("loss", loss, prog_bar=(stage == "train"))
    _log("loss/photom", photom_loss.detach())
    for k, v in parts.items():
      _log(f"loss/photom_{k}", v.detach())
    if offset_reg is not None:
      _log("loss/mean_offset_reg", offset_reg.detach())
    if stage == "train":
      self.log("train/num_gaussians", float(len(self.params["means"])), batch_size=B)
      self.log("train/active_sh_degree", float(self.active_sh_degree), batch_size=B)
    # Bare key, no stage prefix (matches fit_gsplat.py) -- logged every
    # call, both stages, so the x-axis-like counter has continuous
    # coverage across train AND val points instead of gaps during
    # validation-only logging windows. Only train increments it.
    self.log("views_seen", self.views_seen, reduce_fx="max", batch_size=B)
    return loss

  def params_np(self):
    out = {k: v.detach().cpu().numpy() for k, v in self.params.items()}
    # save_output/write_ply/save_output_layered all expect "means" to be
    # the actual position -- resolve the offset here, once, rather than
    # leak the offset/original split to every output writer.
    if self.gaussian_layout == "layered":
      out["means"] = self._active_means().detach().cpu().numpy()
    return out

  def get_preview_source(self, mode):
    """See module.py's PanelCallback/OrbitCallback contract. COLMAP means
    are natively world-space already (no per-view camera-relative frame to
    undo, unlike train_gs.py's), so this is a direct activation + the fixed
    preview cameras from __init__.

    self.step_count, not self.trainer.global_step: manual optimization
    (this file's whole reason for MANUAL optimization -- gsplat needs to
    run between backward() and each optimizer's own .step(), and there
    are several separate optimizers, one per param group) means Lightning
    never sees any of our raw torch.optim.Adam.step() calls, so
    trainer.global_step stays frozen at 0 for the entire run (see
    module._completed_steps' own comment) -- every "step"-mode preview
    would otherwise cache-hit on the FIRST call forever, logging the same
    frame every time cadence fires instead of a fresh one."""
    key = (mode, self.current_epoch if mode == "epoch" else self.step_count)
    if getattr(self, "_preview_cache_key", None) == key:
      return self._preview_cache

    with torch.no_grad():
      gauss = {
        "means": self._active_means(), "quats": F.normalize(self.params["quats"], dim=-1),
        "scales": torch.exp(self.params["scales"]), "opacities": torch.sigmoid(self.params["opacities"]),
        "colors": torch.cat([self.params["sh0"], self.params["shN"]], dim=1),
        "sh_degree": self.active_sh_degree,
      }

    def _entry(stage):
      p = self._preview.get(stage)
      if p is None:
        return None
      return {
        "gauss": gauss, "scene_scale": self.scene_extent,
        "views": {
          "viewmat": p["viewmat"], "K": p["K"],
          "width": p["gt_rgb"].shape[2], "height": p["gt_rgb"].shape[1],
          "gt_rgb": rearrange(p["gt_rgb"], "v h w c -> v c h w"),
        },
      }

    source = {"train": _entry("train"), "val": _entry("val") if mode == "step" else None}
    self._preview_cache_key, self._preview_cache = key, source
    return source

  def on_fit_start(self):
    # self._preview is a plain dict of raw tensors (not registered
    # parameters/buffers), so Lightning's automatic device placement never
    # touches it -- move it here explicitly. self.params is already on
    # self.device by this hook (real Parameters). self.strategy_state
    # doesn't need this: its tensors are created lazily, directly on
    # whatever device the render `info` is already on.
    self._preview = {
      k: {kk: vv.to(self.device) for kk, vv in v.items()}
      for k, v in self._preview.items()
    }


# ---------------------------------------------------------------------------
# final eval -- llffhold test split, same methodology as
# gaussian-splatting/metrics.py (per-image PSNR/SSIM/LPIPS(vgg), then mean).
# ---------------------------------------------------------------------------

def run_final_eval(model, test_cams, device):
  from lpipsPyTorch import lpips

  # Same rule _step uses for val: "white" applies here too (a scene
  # convention, not an augmentation), "random"/"none" don't (no single
  # "the" random colour to eval against -- stay on the comparable bg=None
  # everything else uses). (3,), not (1,3) -- see _step's own comment on
  # gsplat's packed=True backgrounds shape constraint.
  bg = torch.ones(3, device=device) if str(model.cfg.background) == "white" else None

  psnrs, ssims, lpipss = [], [], []
  model.eval()
  with torch.no_grad():
    for c in test_cams:
      gt = torch.from_numpy(c["image"])[None].to(device)
      K = torch.from_numpy(c["K"])[None].to(device)
      vm = torch.from_numpy(c["viewmat"])[None].to(device)
      if bg is not None:
        alpha_gt = torch.from_numpy(c["alpha"])[None].to(device)
        gt = gt + bg[None, None, None, :] * (1.0 - alpha_gt)
      render_params = {**model.params, "means": model._active_means()}
      rgb, _, _ = render(render_params, model.active_sh_degree, vm, K, c["width"], c["height"], backgrounds=bg)
      mse = (rgb - gt).pow(2).mean()
      psnrs.append(float(-10 * torch.log10(mse)))
      ssims.append(float(1.0 - model.dssim(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))))
      lpipss.append(float(lpips(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), net_type="vgg")))
  return {
    "PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssims)), "LPIPS": float(np.mean(lpipss)),
    "num_test_views": len(test_cams),
  }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="fit_3dgs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.seed))

  use_h5 = bool(cfg.get("hdf5_path"))
  default_stem = cfg.hdf5_path if use_h5 else cfg.source_path.rstrip("/")
  out_h5 = cfg.output_path or f"{default_stem}.fit3dgs.h5"
  stem = out_h5[:-3] if out_h5.endswith(".h5") else out_h5

  gaussian_layout = str(cfg.gaussian_layout)
  if gaussian_layout not in ("set", "layered"):
    raise SystemExit(f"gaussian_layout={gaussian_layout!r} -- must be \"set\" or \"layered\"")
  if gaussian_layout == "layered" and not use_h5:
    raise SystemExit("gaussian_layout=layered needs hdf5_path (a render_objaverse.py h5 with "
                     "depth_peel) -- COLMAP scenes have no depth peel to seed a fixed layer stack from")

  background = str(cfg.background)
  if background not in ("none", "white", "random"):
    raise SystemExit(f"background={background!r} -- must be \"none\", \"white\" or \"random\"")

  wandb_run, logger = None, False
  if cfg.wandb.mode != "disabled":
    wandb_run = wandb.init(
      project=cfg.wandb.project, mode=cfg.wandb.mode, tags=list(cfg.wandb.tags),
      name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger = WandbLogger(experiment=wandb_run)

  with timed("load"):
    if use_h5:
      pcd, scene_extent, train_cams, test_cams = load_scene_h5(
        cfg.hdf5_path, float(cfg.data.split_fn.val_fraction), int(cfg.seed))
    else:
      pcd, scene_extent, train_cams, test_cams = load_scene(
        cfg.source_path, cfg.images, bool(cfg.eval), _coerce_resolution(cfg.resolution))
  log.info("%d train views, %d test views (llffhold), scene_extent=%.4f",
           len(train_cams), len(test_cams), scene_extent)

  layered_meta = None
  with timed("init"):
    if gaussian_layout == "layered":
      primary_view_idx = train_cams[0]["view_idx"]
      g, layered_meta = init_gaussians_layered(
        cfg.hdf5_path, primary_view_idx, int(cfg.sh_degree), float(cfg.init_opacity), int(cfg.knn_k))
    else:
      g = init_gaussians(pcd, int(cfg.sh_degree), float(cfg.init_opacity), int(cfg.knn_k))
  n_seeded = len(g["means"])

  preview_cams = {
    "train": train_cams[:4],
    "val": test_cams[:4] if test_cams else [],
  }
  model = Fit3DGSLightningModule(cfg, g, scene_extent, preview_cams)
  datamodule = Fit3DGSDataModule(cfg, train_cams, test_cams)

  callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
  trainer = pl.Trainer(logger=logger, callbacks=callbacks, **OmegaConf.to_container(cfg.trainer, resolve=True))
  with timed("optimize"):
    trainer.fit(model, datamodule=datamodule)

  log.info("%d Gaussians (seeded from %d)", len(model.params["means"]), n_seeded)

  params_np = model.params_np()
  with timed("write"):
    save_output(out_h5, cfg, params_np, scene_extent, model.final_loss, n_seeded)
    write_ply(f"{stem}.ply", params_np)
    if layered_meta is not None:
      grid_h5 = f"{stem}.grid.h5"
      save_output_layered(grid_h5, params_np, layered_meta)
      log.info("layered grid: %s", grid_h5)

  results = None
  if bool(cfg.final_eval) and test_cams:
    with timed("final_eval"):
      # trainer.fit() returning moves the LightningModule back off the
      # accelerator (Lightning's own teardown) -- gsplat's CUDA kernels
      # don't accept CPU tensors at all, so move it back explicitly rather
      # than trust wherever the model happens to sit post-fit.
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = model.to(device)
      results = run_final_eval(model, test_cams, device)
    log.info("final eval (test split, llffhold=8): PSNR %.4f  SSIM %.4f  LPIPS %.4f  (%d views)",
             results["PSNR"], results["SSIM"], results["LPIPS"], results["num_test_views"])
    with open(f"{stem}.results.json", "w") as f:
      json.dump({"ours_final": results}, f, indent=1)

  log.info("final loss %.5f  ->  %s  %s.ply", model.final_loss, out_h5, stem)
  if wandb_run is not None:
    wandb_run.summary["final_loss"] = model.final_loss
    wandb_run.summary["num_gaussians"] = len(params_np["means"])
    wandb_run.summary["num_gaussians_seeded"] = n_seeded
    if results is not None:
      for k, v in results.items():
        wandb_run.summary[f"final_eval/{k}"] = v
    wandb_run.finish()


if __name__ == "__main__":
  main()
