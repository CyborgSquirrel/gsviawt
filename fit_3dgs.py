#!/usr/bin/env python3
"""Faithful 3D Gaussian Splatting optimization on a COLMAP scene (same format
as gaussian-splatting/ -- images/ + sparse/0/, e.g. data/tandt/truck), ported
onto this repo's Hydra + Lightning + wandb-logging + shared-callback shape
(fit_gsplat.py's, via module.py) instead of the reference's argparse + raw
training loop.

Unlike fit_gsplat.py (which seeds one Gaussian per depth-peel pixel, from a
synthetic render, and never adds to that set), this ports the REAL algorithm:
free `xyz`, the COLMAP sparse point cloud as the seed, and full growing/
shrinking densification via gsplat.strategy.DefaultStrategy -- gsplat's own
reimplementation of gaussian-splatting/scene/gaussian_model.py's
densify_and_prune/reset_opacity, verified line-by-line against it (matching
grad/scale thresholds, split/duplicate/prune conditions, opacity-reset
value). Empirically cross-checked too: an early-run Gaussian-count
trajectory (iters 600-1600) landed within ~1% of an instrumented copy of the
reference train.py on this exact scene at every checkpoint.

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
  (H,W,3) float32 in [0,1] at the resolution actually used, 'viewmat' (4,4)
  world-to-camera (OpenCV convention, straight from getWorld2View2 -- COLMAP's
  R/T already ARE this, unlike fit_gsplat.py's OpenGL renders), 'K' (3,3)
  (assumes centered principal point, same as the reference's FoV-only camera
  model), 'name' (for results.json / logging)."""
  w, h = _resolution_for(cam_info.width, cam_info.height, resolution)
  # Plain .resize(), no explicit resample filter -- matches
  # utils/general_utils.py's PILtoTorch exactly (PIL's own default), so
  # training targets and final-eval GT are the identical pixels the
  # reference trains/evaluates against.
  image = np.asarray(Image.open(cam_info.image_path).convert("RGB").resize((w, h)))
  image = image.astype(np.float32) / 255.0

  viewmat = getWorld2View2(cam_info.R, cam_info.T)
  fx, fy = fov2focal(cam_info.FovX, w), fov2focal(cam_info.FovY, h)
  K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], np.float32)

  return {"image": image, "viewmat": viewmat, "K": K, "name": cam_info.image_name, "width": w, "height": h}


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

  Background: black, alpha-premultiplied -- matches ModelParams'
  `white_background` DEFAULT (False), and this file's own render(), which
  never passes `backgrounds=` to gsplat.rasterization (implicitly black).

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
    return {"image": image, "viewmat": viewmat, "K": K[i], "name": f"view{i:03d}", "width": w, "height": h}

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
# rendering
# ---------------------------------------------------------------------------

def render(params, active_sh_degree, viewmats, Ks, width, height, packed=True):
  import gsplat
  colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (P,K,3)
  rgb, alpha, info = gsplat.rasterization(
    means=params["means"], quats=F.normalize(params["quats"], dim=-1),
    scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"]),
    colors=colors, sh_degree=active_sh_degree,
    viewmats=viewmats, Ks=Ks, width=width, height=height,
    render_mode="RGB", packed=packed,
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


# ---------------------------------------------------------------------------
# dataset / lightning module
# ---------------------------------------------------------------------------

class Fit3DGSViewDataset(Dataset):
  """One item = one COLMAP camera's photometric target -- image already
  loaded/resized by load_scene(), viewmat/K precomputed. No alpha channel
  (COLMAP captures have no matte): gt_alpha is always 1, matching the
  reference's rasterizer, which has no background compositing of its own
  either (renders straight onto whatever background colour the config
  picks -- see Fit3DGSLightningModule._step)."""

  def __init__(self, cams):
    self.cams = cams

  def __len__(self):
    return len(self.cams)

  def __getitem__(self, i):
    c = self.cams[i]
    return {
      "gt_rgb": torch.from_numpy(c["image"]),
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

    self.params = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v)) for k, v in g.items()})
    self.dssim = DSSIMLoss()

    import gsplat.strategy as gsstrat
    self.strategy = gsstrat.DefaultStrategy(
      prune_opa=float(cfg.densify.prune_opacity), grow_grad2d=float(cfg.densify.grad_threshold),
      grow_scale3d=float(cfg.percent_dense), prune_scale3d=float(cfg.densify.prune_scale3d),
      refine_start_iter=int(cfg.densify.from_iter), refine_stop_iter=int(cfg.densify.until_iter),
      refine_every=int(cfg.densify.interval), reset_every=int(cfg.opacity_reset_interval),
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
    it = self.step_count
    self.step_count += 1
    iteration = it + 1

    # trainer.max_steps is NOT enforced under manual optimization (Lightning
    # ties it to the automatic-optimization step-counting path, which we
    # bypass entirely) -- with trainer.max_epochs=-1 as the only other stop
    # condition, an unguarded run here is unbounded (confirmed the hard way:
    # a 300-iters config ran ~700 epochs before OOM-thrashing on a runaway
    # Gaussian count). Signal stop explicitly once our own 1-indexed
    # `iteration` reaches cfg.iters -- Lightning checks should_stop within
    # the epoch loop, not just between epochs, so this still stops promptly
    # mid-epoch rather than running to that epoch's end.
    if iteration >= int(self.cfg.iters):
      self.trainer.should_stop = True

    if iteration % 1000 == 0 and self.active_sh_degree < self.max_sh_degree:
      self.active_sh_degree += 1
    self.optimizers_dict["means"].param_groups[0]["lr"] = self.means_lr_fn(it)

    gt_rgb, K, viewmat = batch["gt_rgb"], batch["K"], batch["viewmat"]
    H, W = gt_rgb.shape[1:3]

    rgb, alpha, info = render(self.params, self.active_sh_degree, viewmat, K, W, H)
    # step_pre_backward calls info["means2d"].retain_grad() -- must run
    # before backward for a non-leaf tensor to keep its .grad populated.
    self.strategy.step_pre_backward(self.params, self.optimizers_dict, self.strategy_state, it, info)
    loss, parts = self._photom_loss(rgb, gt_rgb)

    for opt in self.optimizers_dict.values():
      opt.zero_grad(set_to_none=True)
    self.manual_backward(loss)

    for opt in self.optimizers_dict.values():
      opt.step()

    # Post-backward, post-step (matches gsplat's own reference trainer's
    # ordering, examples/simple_trainer.py -- grow/prune happens AFTER this
    # step's gradient update is applied, using the gradient info accumulated
    # THIS step, so it acts on next step's population, not this one's).
    self.strategy.step_post_backward(
      self.params, self.optimizers_dict, self.strategy_state, it, info, packed=True)

    self.views_seen += gt_rgb.shape[0]
    self.final_loss = loss.item()
    if iteration % int(self.cfg.densify.interval) == 0:
      log.info("iter %d: loss=%.5f  %d Gaussians", iteration, loss.item(), len(self.params["means"]))
    self.log("train/loss", loss, prog_bar=True, batch_size=gt_rgb.shape[0])
    for k, v in parts.items():
      self.log(f"train/loss/photom_{k}", v.detach(), batch_size=gt_rgb.shape[0])
    self.log("train/num_gaussians", float(len(self.params["means"])), batch_size=gt_rgb.shape[0])
    self.log("train/active_sh_degree", float(self.active_sh_degree), batch_size=gt_rgb.shape[0])
    self.log("views_seen", self.views_seen, reduce_fx="max", batch_size=gt_rgb.shape[0])
    return loss

  def validation_step(self, batch, batch_idx):
    gt_rgb, K, viewmat = batch["gt_rgb"], batch["K"], batch["viewmat"]
    H, W = gt_rgb.shape[1:3]
    with torch.no_grad():
      rgb, _, _ = render(self.params, self.active_sh_degree, viewmat, K, W, H)
      loss, parts = self._photom_loss(rgb, gt_rgb)
    self.log("val/loss", loss, batch_size=gt_rgb.shape[0])
    for k, v in parts.items():
      self.log(f"val/loss/photom_{k}", v.detach(), batch_size=gt_rgb.shape[0])
    return loss

  def params_np(self):
    return {k: v.detach().cpu().numpy() for k, v in self.params.items()}

  def get_preview_source(self, mode):
    """See module.py's PanelCallback/OrbitCallback contract. COLMAP means
    are natively world-space already (no per-view camera-relative frame to
    undo, unlike train_gs.py's), so this is a direct activation + the fixed
    preview cameras from __init__."""
    key = (mode, self.current_epoch if mode == "epoch" else self.trainer.global_step)
    if getattr(self, "_preview_cache_key", None) == key:
      return self._preview_cache

    with torch.no_grad():
      gauss = {
        "means": self.params["means"], "quats": F.normalize(self.params["quats"], dim=-1),
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

  psnrs, ssims, lpipss = [], [], []
  model.eval()
  with torch.no_grad():
    for c in test_cams:
      gt = torch.from_numpy(c["image"])[None].to(device)
      K = torch.from_numpy(c["K"])[None].to(device)
      vm = torch.from_numpy(c["viewmat"])[None].to(device)
      rgb, _, _ = render(model.params, model.active_sh_degree, vm, K, c["width"], c["height"])
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

  with timed("init"):
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
