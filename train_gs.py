#!/usr/bin/env python3
"""Train a point-cloud-anchored 3D Gaussian Splatting prediction model.

Flash3D-style encoder/decoder (gs_encoder.py / gs_decoder.py): a ResNet-50
trunk over RGB concatenated with a pixel-aligned multi-layer point cloud
(gs_dataset.py), decoding per-pixel per-layer Gaussian appearance/shape
parameters (opacity, scale, rotation, SH-degree-1 color). Gaussian positions
are NOT predicted -- they're fixed exactly to the input point cloud (see
gs_dataset.py's dense_unproject_camera), so there's no depth-prediction
network and no learned offset, unlike Flash3D itself.

Training objective (Flash3D's cross-view photometric setup): encode one
source view, render the predicted Gaussians into the source view itself
(self-reconstruction) plus several other views of the same object, and
supervise with L1 + D-SSIM (+ optional LPIPS) against the real renders.
Rendering uses gsplat, with all geometry kept in the source camera's own
frame (see gs_dataset.relative_viewmats) -- no world coordinates involved.

Usage:
    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python train_gs.py \\
        data.photom_h5=/app/bla/lite_blackbg.h5 train.max_epochs=1000

gsplat JIT-compiles CUDA kernels on first import; see _setup_cuda_toolchain
(copied from fit_gsplat.py, which needs the same env setup).
"""

import contextlib as ctl
import functools as ft
import logging
import os
import shutil
import sys
import tempfile

import gsplat
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from fit_gsplat import SH_C0  # noqa: E402
from gs_dataset import (OPENGL_TO_OPENCV, GaussH5ValDataset,  # noqa: E402
                        GSFixedViewsDataset, GSPairDataset, H5Catalog,
                        _EmptyDataset)
from gs_decoder import GaussianResnetDecoder, GSDecoderStack  # noqa: E402
from gs_encoder import GSResnetEncoder  # noqa: E402
from module import DSSIMLoss, WarmupCosineAnnealingLR, guarded_render  # noqa: E402
from orbit_video import look_at_c2w, write_mp4  # noqa: E402
from util import collate_with_batch_size, pipe, set_mode, timed  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class GSModel(nn.Module):
  """Encoder + decoder wrapper: builds the concatenated [RGB, per-layer xyz,
  per-layer validity] input tensor, runs it through the ResNet trunk, and
  returns per-pixel per-layer Gaussian appearance/shape parameters. Positions
  are not part of this module's output -- see gs_dataset.dense_unproject_camera
  and flatten_gaussians below."""

  def __init__(self, cfg):
    super().__init__()
    self.num_layers = int(cfg.data.num_layers)
    self.max_sh_degree = int(cfg.model.max_sh_degree)
    self.min_scale_mult = float(cfg.model.min_scale_mult)
    in_channels = 3 + 3 * self.num_layers

    # TODO: Option to configure which layers we train decoder heads for

    self.encoder = GSResnetEncoder(
      num_layers=cfg.model.backbone.num_layers, pretrained=cfg.model.backbone.pretrained,
      bn_order=cfg.model.backbone.bn_order, in_channels=in_channels,
    )
    decoder_kwargs = dict(
      num_ch_dec=tuple(cfg.model.backbone.num_ch_dec),
      upsample_mode=cfg.model.backbone.upsample_mode,
      opacity_scale=cfg.model.opacity_scale, opacity_bias=cfg.model.opacity_bias,
      scale_scale=cfg.model.scale_scale, scale_bias=cfg.model.scale_bias,
      sh_scale=cfg.model.sh_scale, scale_lambda=cfg.model.scale_lambda,
    )
    self.decoder = GSDecoderStack(
      self.encoder.num_ch_enc,
      num_layers=self.num_layers,
      max_sh_degree=self.max_sh_degree,
      **decoder_kwargs,
    )

    self.register_buffer("xyz_mean", torch.tensor(list(cfg.model.xyz_mean), dtype=torch.float32))
    self.register_buffer("xyz_std", torch.tensor(list(cfg.model.xyz_std), dtype=torch.float32))

  def build_input(self, rgb, xyz_cam):
    """
    rgb: (B,3,IH,IW) in [0,1]
    xyz_cam: (B,L,DH,DW,3) NaN-invalid
    Returns (B,3+4L,DH,DW), resizing rgb to the depth-peel resolution if they differ.
    """
    _, _, dh, dw, _ = xyz_cam.shape
    if rgb.shape[-2:] != (dh, dw):
      rgb = F.interpolate(rgb, size=(dh, dw), mode="bilinear", align_corners=False)
    rgb_norm = (rgb - 0.45) / 0.225

    xyz_filled = torch.nan_to_num(xyz_cam, nan=0.0)
    xyz_norm = (xyz_filled - self.xyz_mean) / self.xyz_std
    xyz_norm = rearrange(xyz_norm, "b l h w c -> b (l c) h w")

    inp, _ = pack([rgb_norm, xyz_norm], "b * h w")
    return inp

  def forward(self, rgb, xyz_cam):
    x = self.build_input(rgb, xyz_cam)
    _, _, dh, dw, _ = xyz_cam.shape
    # NOTE: The 5-level U-Net halves spatial dims 4x (conv1 + maxpool + 2 more
    # strided stages) then doubles back up 5x via nearest-neighbor upsample;
    # that only round-trips exactly when H/W are multiples of 32 (Flash3D's own
    # inputs are; render_objaverse.py's 504x504 renders aren't), so pad up to
    # the next multiple of 32 before the encoder and crop the decoder's output
    # back to (dh, dw) -- same fix Flash3D applies via its pad_border_aug, just
    # replicate-padded to the exact multiple instead of a fixed border.

    # Pad if necessary
    pad_h, pad_w = (-dh) % 32, (-dw) % 32
    if pad_h or pad_w:
      x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    feats = self.encoder(x)
    out = self.decoder(feats)

    # Ensure padding is stripped out
    out = {
      k: v[..., :dh, :dw] # (B,L,C,H,W)
      for k, v in out.items()
    }

    return out


def model_forward(
  model,
  batch,
  *,
  need_flat_gauss: bool=False,
  need_render: bool=False,
  device,
):
  B = batch["batch_size"]

  out = {}
  out["gauss"] = model(
    batch["views"]["rgb"][:, 0].to(device),
    batch["source"]["xyz_cam"].to(device),
  )

  # render 3DGS if necessary
  if need_render:
    # TODO: overrides
    # gauss_render = gauss_pred
    # gt = item.get("ground_truth")
    # if gt is not None:
    #   gauss_render = apply_ground_truth_overrides(gauss, gt, hit, predict_params, device)
    # src = item["source"]
    # hit = src["hit"].to(device)

    pred_rgb = []
    pred_alpha = []

    for (
      x_viewmats,
      x_Ks,
      x_gauss_flat,
    ) in zip(
      batch["views"]["viewmat"].to(device), # (B,V,N,M)
      batch["views"]["K_image"].to(device), # (B,V,N,M)
      flatten_gaussians(
        B,
        out["gauss"],
        batch["source"]["xyz_cam"].to(device),
        batch["source"]["hit"].to(device),
      ),
    ):
      _b, _v, _c, ih, iw = batch["views"]["rgb"].shape

      x_pred_rgb, x_pred_alpha, _ = gsplat.rasterization(
        means=x_gauss_flat["means"],
        quats=x_gauss_flat["quats"],
        scales=x_gauss_flat["scales"],
        opacities=x_gauss_flat["opacities"],
        colors=x_gauss_flat["colors"],
        viewmats=x_viewmats,
        Ks=x_Ks,
        width=int(iw),
        height=int(ih),
        sh_degree=model.max_sh_degree,
        render_mode="RGB",
        packed=True,
      )

      x_pred_rgb   = rearrange(x_pred_rgb, "v h w c -> v c h w")
      x_pred_alpha = rearrange(x_pred_rgb, "v h w c -> v c h w")

      pred_rgb.append(x_pred_rgb)
      pred_alpha.append(x_pred_alpha)

    pred_rgb = torch.stack(pred_rgb).clamp(0.0, 1.0)
    pred_alpha = torch.stack(pred_alpha).clamp(0.0, 1.0)

    out["rgb"] = pred_rgb
    out["alpha"] = pred_alpha

  return out


def flatten_gaussians(b, gauss, xyz_cam, hit):
  """
  xyz_cam: (B,L,H,W,3) f32 camera-space positions (NaN where invalid)
  same layer/pixel order as `hit` and every entry of `gauss` (each (L,C,H,W)).
  Filters to valid entries and returns gsplat.rasterization-ready tensors.
  """

  # NOTE(andrei): There are different ways one could go about flattening the
  # Gaussians with our setup. We could put the entire batch of Gaussians into
  # arrays, since gsplat.rasterize() does support this kind of batched
  # rendering.
  #
  # However in this case it would be necessary to set the opacity of the invalid
  # Gaussians to zero to prevent them from showing up.
  #
  # The alternative is to just call gsplat.rasterize() in a for loop which is
  # what I went with. We have a lot of invalid Gaussians so I think this makes
  # more sense.

  for idx in range(b):
    means_flat = rearrange(xyz_cam[idx], "l h w c -> (l h w) c")
    valid_flat = rearrange(hit[idx], "l h w -> (l h w)")

    opacity_flat  = rearrange(gauss["opacity"] [idx], "l c h w -> (l h w c)") * valid_flat.to(gauss["opacity"].dtype)
    scale_flat    = rearrange(gauss["scale"]   [idx], "l c h w -> (l h w) c")
    rotation_flat = rearrange(gauss["rotation"][idx], "l c h w -> (l h w) c")
    colors_flat   = rearrange(gauss["sh_dc"]   [idx], "l c h w -> (l h w) 1 c")
    if "sh_rest" in gauss:
      colors_flat = pack(
        [
          colors_flat[idx],
          rearrange(gauss["sh_rest"], "l (k c) h w -> (l h w) k c", c=3)
        ],
        "n * c",
      )

    keep = valid_flat
    yield {
      "means": means_flat[keep],
      "quats": rotation_flat[keep],
      "scales": scale_flat[keep],
      "opacities": opacity_flat[keep],
      "colors": colors_flat[keep],
    }


PREDICTABLE_PARAMS = ("opacity", "scale", "rotation", "color")


def apply_ground_truth_overrides(gauss, gt, hit, predict_params, device):
  """Overwrites gauss's fields NOT listed in `predict_params` with values
  taken directly from `gt` (gs_dataset._load_ground_truth's output),
  converted to gauss's own units/shape. Only overrides where hit &
  gt["valid"] agree -- elsewhere keeps the model's own prediction."""
  out = dict(gauss)
  mask = (hit & gt["valid"].to(device)).unsqueeze(1)   # (L,1,H,W)

  if "opacity" not in predict_params:
    gt_opacity = gt["opacity"].to(device).unsqueeze(1)   # (L,1,H,W)
    out["opacity"] = torch.where(mask, gt_opacity, gauss["opacity"])

  if "scale" not in predict_params:
    gt_scale = rearrange(gt["scale"].to(device), "l h w c -> l c h w")
    m3 = repeat(mask, "l 1 h w -> l c h w", c=3)
    out["scale"] = torch.where(m3, gt_scale, gauss["scale"])
    if "raw_scale" in gauss:
      out["raw_scale"] = torch.where(m3, torch.log(gt_scale.clamp(min=1e-8)), gauss["raw_scale"])

  if "rotation" not in predict_params:
    gt_rotation = rearrange(gt["rotation"].to(device), "l h w c -> l c h w")
    m4 = repeat(mask, "l 1 h w -> l c h w", c=4)
    out["rotation"] = torch.where(m4, gt_rotation, gauss["rotation"])

  if "color" not in predict_params:
    gt_color = rearrange(gt["color"].to(device), "l h w c -> l c h w")
    gt_sh_dc = (gt_color - 0.5) / SH_C0
    m3 = repeat(mask, "l 1 h w -> l c h w", c=3)
    out["sh_dc"] = torch.where(m3, gt_sh_dc, gauss["sh_dc"])
    if "sh_rest" in gauss:
      # No ground-truth signal exists for view-dependent shading at all
      # (fit_gsplat.py has no SH>0) -- zero it rather than leave an
      # untrained network output riding on top of a now-fixed flat color.
      m_rest = repeat(mask, "l 1 h w -> l c h w", c=gauss["sh_rest"].shape[1])
      out["sh_rest"] = torch.where(m_rest, torch.zeros_like(gauss["sh_rest"]), gauss["sh_rest"])

  return out


# ---------------------------------------------------------------------------
# losses (DSSIMLoss now lives in module.py, shared with fit_gsplat.py)
# ---------------------------------------------------------------------------


def compute_direct_loss(gauss, gt, hit, cfg_loss, device):
  """Direct per-pixel supervision of the model's own predicted Gaussian
  params against a fit_gsplat.py ground-truth grid (gs_dataset's
  _load_ground_truth output -- rotation already converted into the source
  camera's own frame, positions never compared since they're fixed to the
  same point cloud by construction). `gauss`: raw decoder output, each
  (L,C,H,W). `hit`: (L,H,W) bool. Returns (total, parts) -- `parts` only has
  keys for fields whose weight is actually nonzero (omit-if-skipped, same
  convention as the rest of this file's metrics dicts). cfg_loss.supervised_layers
  (None or a list of layer indices), when set, restricts which depth-peel
  layers actually contribute to this loss -- everything else about the
  model (input point cloud, predicted/rendered layers) is unchanged, only
  which layers get gradient from THIS loss."""
  mask = hit & gt["valid"].to(device)
  if cfg_loss.supervised_layers is not None:
    layer_mask = torch.zeros(hit.shape[0], dtype=torch.bool, device=device)
    layer_mask[list(cfg_loss.supervised_layers)] = True
    mask = mask & layer_mask[:, None, None]
  parts = {}
  total = torch.zeros((), device=device)
  if not mask.any():
    return total, parts

  eps = 1e-6
  if cfg_loss.direct_opacity_weight > 0:
    opacity_pred = rearrange(gauss["opacity"], "l c h w -> l h w c")[..., 0]
    # nan_to_num: outside `mask` (invalid/padding grid cells) gt_opacity is
    # NaN-filled at the source, same as gt_scale -- those cells are excluded
    # from the loss value via [mask] below, but pow(2)'s backward (2*x) still
    # propagates NaN through the unselected positions' local Jacobian even
    # though their incoming gradient is zero (0*NaN=NaN). abs()'s backward
    # (sign(x)) never had this problem; L2 does, so the guard is required now.
    gt_opacity = torch.nan_to_num(gt["opacity"].to(device), nan=0.0)
    parts["opacity"] = (opacity_pred - gt_opacity).pow(2)[mask].mean()
    total = total + cfg_loss.direct_opacity_weight * parts["opacity"]

  if cfg_loss.direct_scale_weight > 0:
    # Log-space L2 against gauss["raw_scale"] (pre-floor/pixel_scale logit,
    # not the activated "scale" -- units must match gt after log()). L2 over
    # L1 concentrates gradient on the worst-residual pixels instead of
    # applying equal pressure everywhere. gt_scale is NaN-padded outside
    # `mask`; nan_to_num guards pow(2)'s backward (2*x) from propagating
    # that NaN even though the incoming gradient there is zero.
    scale_pred = rearrange(gauss["raw_scale"], "l c h w -> l h w c")
    gt_scale = torch.nan_to_num(gt["scale"].to(device), nan=eps).clamp(min=eps)
    log_diff = scale_pred - torch.log(gt_scale)
    parts["scale"] = log_diff.pow(2)[mask].mean()
    total = total + cfg_loss.direct_scale_weight * parts["scale"]

  if cfg_loss.direct_rotation_weight > 0:
    # Sign-invariant: q and -q represent the same rotation (quaternion
    # double-cover), a naive L1/L2 would be wrong on that half of cases.
    rot_pred = rearrange(gauss["rotation"], "l c h w -> l h w c")
    dot = (rot_pred * gt["rotation"].to(device)).sum(-1).abs().clamp(max=1.0)
    parts["rotation"] = (1.0 - dot)[mask].mean()
    total = total + cfg_loss.direct_rotation_weight * parts["rotation"]

  if cfg_loss.direct_color_weight > 0:
    # fit_gsplat.py has no SH>0 -- its ground truth is flat RGB, comparable
    # to our sh_dc via SH degree-0 evaluation. Only sh_dc gets a gradient
    # from this; sh_rest has no ground-truth signal in this source at all.
    # L1 (not L2, unlike opacity/scale above): abs()'s backward (sign(x))
    # never propagates NaN through masked-out positions the way pow(2)'s
    # backward (2*x) does, so no nan_to_num guard is needed here.
    shdc_pred = rearrange(gauss["sh_dc"], "l c h w -> l h w c")
    color_pred = SH_C0 * shdc_pred + 0.5
    parts["color"] = (color_pred - gt["color"].to(device)).abs()[mask].mean()
    total = total + cfg_loss.direct_color_weight * parts["color"]

  return total, parts


# ---------------------------------------------------------------------------
# orbit previews
#
# The model's Gaussians live entirely in the source view's own camera frame
# (see gs_dataset.py's module docstring) -- there's no Blender world frame
# available to orbit around. To still get a non-tumbling turntable, we derive
# a stable "up" direction by taking Blender's world +Z and expressing it in
# that same source-camera frame via the view's own (otherwise-unused) raw
# pose, then build the orbit entirely within that frame using the same
# look_at_c2w/OPENGL_TO_OPENCV convention as the rest of the render path.
# ---------------------------------------------------------------------------

def _up_in_source_frame(pose_gl):
  """pose_gl: (4,4) camera-to-world, Blender/OpenGL axes (a view's raw,
  un-transformed pose). Returns Blender's world +Z direction expressed in
  this camera's own OpenCV frame (the frame flatten_gaussians' `means` live
  in), via the same axis flip debug_pointcloud.py/gs_dataset.py use
  elsewhere (self-inverse, so it applies the same in either direction)."""
  r_gl = pose_gl[:3, :3]
  up_gl = r_gl.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
  up_cv = up_gl * np.array([1.0, -1.0, -1.0], dtype=np.float32)
  return up_cv / (np.linalg.norm(up_cv) + 1e-8)


def _orbit_c2w_gl(means, up, num_frames, elevation_deg):
  """Closed circular orbit (OpenGL local axes, matching look_at_c2w) around
  `means`'s centroid, entirely within `means`'s own frame, at the SAME
  distance the real source camera was from the object.

  This deliberately does not fit a "nicely framed" distance to the means'
  bounding sphere (orbit_video.py's approach, meant for arbitrary/unknown-
  scale 3DGS models): our Gaussians are anchored one-per-source-pixel, so
  their scale only covers the point spacing produced at that original
  capture distance/resolution. A tighter-fit orbit camera zooms in past that
  native density and exposes the gaps between neighboring Gaussians as a
  fine grid/moire artifact -- confirmed by comparing against a render at the
  actual capture distance, which shows no such pattern. The real camera sat
  at this frame's own origin (see gs_dataset.py), so the distance is just
  the centroid's norm -- no separate bookkeeping needed."""
  center = means.mean(axis=0)
  up = up / (np.linalg.norm(up) + 1e-8)

  # azimuth=0 reference: from the object back toward where the real camera
  # was (its own frame's origin), projected off the up axis -- an arbitrary
  # but recognizable, non-degenerate starting angle.
  ref = -center
  ref = ref - np.dot(ref, up) * up
  if np.linalg.norm(ref) < 1e-6:
    arbitrary = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(np.dot(arbitrary, up)) > 0.99:
      arbitrary = np.array([0.0, 1.0, 0.0], np.float32)
    ref = arbitrary - np.dot(arbitrary, up) * up
  ref = ref / np.linalg.norm(ref)
  right = np.cross(up, ref)

  dist = max(float(np.linalg.norm(center)), 1e-3)
  elev = np.radians(float(elevation_deg))
  azimuths = np.linspace(0.0, 2 * np.pi, int(num_frames), endpoint=False)

  poses = []
  for az in azimuths:
    eq_dir = ref * np.cos(az) + right * np.sin(az)
    direction = eq_dir * np.cos(elev) + up * np.sin(elev)
    eye = center + dist * direction
    poses.append(look_at_c2w(eye, center, up=up))
  return np.stack(poses).astype(np.float32)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def _external_val_dataset(cfg):
  """Builds val_ds as a standalone GSPairDataset over its OWN H5Catalog
  (data.photom_h5_val), completely independent of whatever train_ds is doing
  -- source AND target views are both drawn from this corpus itself
  (deterministic_targets=True, exactly like today's multi-scene val split),
  not from data.photom_h5 / data.fixed_source_view. See conf/train_gs.yaml's
  data.photom_h5_val comment for why this is the deliberate design."""
  catalog = H5Catalog(
    cfg.data.photom_h5_val,
    H5Catalog.path().alias("path"),
    H5Catalog.index().alias("view_idx"),
    H5Catalog.dataset("mesh_index").alias("mesh_id"),
  )
  return GSPairDataset(
    catalog, num_layers=cfg.data.num_layers, num_target_views=cfg.data.num_target_views,
    seed=cfg.seed, deterministic_targets=True,
  )


class GSDataModule(pl.LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.train_ds = None
    self.val_ds = None

  def setup(self, stage=None):
    if self.train_ds is not None:
      return
    cfg = self.cfg
    if cfg.data.fixed_source_view is not None:
      # Single-batch overfit mode (fit_gsplat.py's primary/secondary terms
      # applied here): the exact same source+target views every step, no
      # resampling at all -- nothing to hold out, so val is empty. photom_h5
      # is optional here (mutually exclusive with gauss_h5, see main()'s
      # config-validation block) -- when unset, GSFixedViewsDataset builds
      # the source view straight from gauss_h5's own embedded primary view.
      photom_h5 = None
      if cfg.data.photom_h5 is not None:
        photom_h5 = cfg.data.photom_h5 if isinstance(cfg.data.photom_h5, str) else cfg.data.photom_h5[0]
      self.train_ds = GSFixedViewsDataset(
        source_view=cfg.data.fixed_source_view,
        target_views=list(cfg.data.fixed_target_views) if cfg.data.fixed_target_views is not None else [],
        num_layers=cfg.data.num_layers,
        photom_h5=photom_h5, gauss_h5=cfg.data.gauss_h5,
      )
      # data.photom_h5_val is orthogonal to this mode's own view selection --
      # if set, val still comes from that wholly separate corpus (source AND
      # targets both drawn from it), not from fixed_source_view/targets.
      # Otherwise, when data.gauss_h5 is set, fall back to ITS OWN embedded
      # views (GaussH5ValDataset) -- gives real photometric val numbers for
      # a direct-supervision-only run without needing any separate render
      # corpus on disk.
      if cfg.data.photom_h5_val is not None:
        self.val_ds = _external_val_dataset(cfg)
      elif cfg.data.gauss_h5 is not None:
        self.val_ds = GaussH5ValDataset(cfg.data.gauss_h5, num_layers=cfg.data.num_layers)
      else:
        self.val_ds = _EmptyDataset()
    elif cfg.data.photom_h5_val is not None:
      # Multi-scene training with an external validation corpus: bypass
      # data.split_fn entirely and use the WHOLE photom_h5 catalog for
      # train_ds -- data.split_fn becomes a no-op here (logged in main()'s
      # config-validation block, not silently swallowed).
      catalog = H5Catalog(
        cfg.data.photom_h5,
        H5Catalog.path().alias("path"),
        H5Catalog.index().alias("view_idx"),
        H5Catalog.dataset("mesh_index").alias("mesh_id"),
      )
      self.train_ds = GSPairDataset(
        catalog, num_layers=cfg.data.num_layers, num_target_views=cfg.data.num_target_views,
        seed=cfg.seed, deterministic_targets=False,
      )
      self.val_ds = _external_val_dataset(cfg)
    else:
      catalog = H5Catalog(
        cfg.data.photom_h5,
        H5Catalog.path().alias("path"),
        H5Catalog.index().alias("view_idx"),
        H5Catalog.dataset("mesh_index").alias("mesh_id"),
      )

      split_fn = hydra.utils.instantiate(cfg.data.split_fn)
      train_catalog, val_catalog = split_fn(catalog, seed=cfg.seed)

      self.train_ds = GSPairDataset(
        train_catalog, num_layers=cfg.data.num_layers, num_target_views=cfg.data.num_target_views,
        seed=cfg.seed, deterministic_targets=False,
      )
      self.val_ds = GSPairDataset(
        val_catalog, num_layers=cfg.data.num_layers, num_target_views=cfg.data.num_target_views,
        seed=cfg.seed, deterministic_targets=True,
      )
    if len(self.train_ds) == 0:
      raise SystemExit("train split is empty -- check data.photom_h5 / data.split_fn")

  def train_dataloader(self):
    return DataLoader(
      self.train_ds,
      shuffle=True,
      collate_fn=collate_with_batch_size,
      **OmegaConf.to_container(self.cfg.loader, resolve=True),
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_ds,
      shuffle=False,
      collate_fn=collate_with_batch_size,
      num_workers=0,
      batch_size=self.cfg.loader.batch_size,
    )


class GSLightningModule(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.views_seen = 0

    with timed("build_model"):
      self.model = GSModel(cfg)
    self.dssim = DSSIMLoss(reduction="none")

    # One directory for everything this run produces (currently just
    # checkpoints, but named generically for whatever else lands here later
    # -- orbit videos/panels are wandb-only right now).
    # Defaults to the Unix timestamp at startup, so back-to-back runs never
    # share a directory unless cfg.output_dir is set explicitly.
    self.output_dir = cfg.output_dir
    os.makedirs(self.output_dir, exist_ok=True)

  def configure_optimizers(self):
    optim = hydra.utils.instantiate(self.cfg.optim)(self.model.parameters())
    sched = hydra.utils.instantiate(self.cfg.sched)(optim)
    return {
      "optimizer": optim,
      "lr_scheduler": {
        "scheduler": sched,
        "interval": "step",
      }
    }

  def training_step(self, batch, batch_idx):
    return self._step("train", batch, batch_idx)

  def validation_step(self, batch, batch_idx):
    return self._step("val", batch, batch_idx)

  def _step(self, stage, batch, batch_idx):
    B = batch["batch_size"]
    _, V, *_ = batch["views"]["rgb"].shape

    @ft.wraps(self.log)
    def _log(key, *args, **kwargs):
      self.log(f"{stage}/{key}", *args, **kwargs, batch_size=B)

    cfg = self.cfg
    device = self.device

    need_render = (
      cfg.loss.photom.l1_weight > 0
      or cfg.loss.photom.ssim_weight > 0
      or cfg.loss.photom.mask_weight > 0
    )

    if stage == "train":
      self.views_seen += B * V
    self.log("views_seen", self.views_seen, reduce_fx="max")

    # forward pass
    model_pred = model_forward(self.model, batch, need_render=need_render, device=device)
    pred_rgb = model_pred["rgb"]
    pred_alpha = model_pred["alpha"]

    # loss
    loss = torch.zeros((), device=device)
    out = {}

    # photometric losses
    gt_rgb = batch["views"]["rgb"]
    gt_alpha = batch["views"]["alpha"]

    def _splatter_metric(metric_name, metric_full):
      """
      metric_full: (B,V,...)
      """
      _log(f"{metric_name}", metric_full.mean().detach())
      _log(f"{metric_name}/source", metric_full[:,0].mean().detach())
      _log(f"{metric_name}/targets", metric_full[:,1:].mean().detach())

    def f(a):
      return rearrange(a, "b v ... -> (b v) ...")
    def g(a):
      return rearrange(a, "(b v) ... -> b v ...", b=B, v=V)

    # NOMERGE: Composite GT RGB onto white background

    # TODO: Maybe integrate alpha into the loss function somehow? Maybe not?

    # TODO: LPIPS loss.

    ## L1 loss
    if cfg.loss.photom.l1_weight > 0:
      _photom_l1 = g((f(pred_rgb) - f(gt_rgb)).abs()) # (B,V,C,H,W)
      _splatter_metric("loss/photom_l1", _photom_l1)
      loss = loss + cfg.loss.photom.l1_weight * _photom_l1.mean()

    ## D-SSIM Loss
    if cfg.loss.photom.dssim_weight > 0:
      _photom_dssim = g(self.dssim(f(pred_rgb), f(gt_rgb))) # (B,V,C,H,W)
      _splatter_metric("loss/photom_dssim", _photom_dssim)
      loss = loss + cfg.loss.photom.dssim_weight * _photom_dssim.mean()

    # regularization

    ## scale regularization
    scale_reg = torch.zeros((), device=device)
    if cfg.loss.scale_reg_weight > 0:
      scale_reg = pipe(
        model_pred["gauss"]["scale"],
        lambda a: rearrange(a, "b ... -> b (...)"),
        lambda a: a[a > cfg.loss.scale_reg_thresh],
        lambda a: a.mean() if a.numel() > 0 else torch.zeros((), device=device),
      )

      loss = loss + cfg.loss.scale_reg_weight * scale_reg
      _log("loss/scale_reg", scale_reg.detach())

    # TODO: skim through this code, fix it up
    # if need_direct:
    #   raise NotImplementedError()
    # if need_direct:
    #   direct_total, direct_parts = compute_direct_loss(gauss, gt, hit, cfg.loss, device)
    #   loss = loss + direct_total
    #   for k, v in direct_parts.items():
    #     metrics[f"direct_{k}"] = v.detach()

    _log("loss", loss.detach())
    out["loss"] = loss

    # TODO: Are any of these metrics actually needed?
    # metrics["mean_opacity"] = flat["opacities"].mean().detach() if flat["opacities"].numel() else torch.zeros((), device=device)
    # metrics["mean_scale"] = flat["scales"].mean().detach() if flat["scales"].numel() else torch.zeros((), device=device)
    # metrics["frac_kept"] = torch.tensor(
    #   flat["means"].shape[0] / max(1, item["source"]["hit"].numel()), device=device)

    return out

  def get_preview_source(self):
    """Gaussians + views for module.PanelCallback's [GT | render | |diff|]
    panel, computed from a fixed train item (self.trainer.datamodule's
    train_ds[0]) via this model's own forward pass. Cached per-epoch
    (lazily -- no __init__ changes needed for this, it's this method's own
    bookkeeping) so a second call in the same epoch doesn't redundantly
    redo the forward pass; called on-demand by PanelCallback, only on
    epochs it's actually about to log, not every epoch.

    NOTE(andrei): Not sure if setting model to eval is the right move here,
    but gonna do it for now."""
    epoch = self.current_epoch
    if getattr(self, "_preview_cache_epoch", None) == epoch:
      return self._preview_cache

    batch = collate_with_batch_size([self.trainer.datamodule.train_ds[0]])
    device = self.device
    with set_mode(self.model, "eval"), torch.no_grad():
      gauss = self.model(
        batch["views"]["rgb"][:, 0].to(device),
        batch["source"]["xyz_cam"].to(device),
      )
      flat = next(flatten_gaussians(
        batch["batch_size"], gauss,
        batch["source"]["xyz_cam"].to(device), batch["source"]["hit"].to(device),
      ))
    flat["sh_degree"] = self.model.max_sh_degree

    v = batch["views"]
    source = {
      "gauss": flat,
      "views": {
        "viewmat": v["viewmat"][0].to(device),
        "K": v["K_image"][0].to(device),
        "width": v["rgb"].shape[-1],
        "height": v["rgb"].shape[-2],
        "gt_rgb": v["rgb"][0].to(device),
      },
    }
    self._preview_cache_epoch, self._preview_cache = epoch, source
    return source


class OrbitCallback(pl.Callback):
  def __init__(
    self,
    *,
    every_n_epochs: int,
    num_frames: int,
    fps: int,
    crf: int,
    elevation_deg: float,
  ):
    self.every_n_epochs = every_n_epochs
    self.num_frames = num_frames
    self.fps = fps
    self.crf = crf
    self.elevation_deg = elevation_deg

    self.train_batch = None
    self.val_batch = None
    self.workdir = None

  def on_fit_start(self, trainer, pl_module):
    dm = trainer.datamodule
    self.train_batch = collate_with_batch_size([dm.train_ds[0]])
    self.val_batch = collate_with_batch_size([dm.val_ds[0]]) if len(dm.val_ds) > 0 else None
    self.workdir = tempfile.mkdtemp(prefix="gs_orbit_")

  def on_train_epoch_end(self, trainer, pl_module):
    return self._on_epoch_end("train", trainer, pl_module)

  def on_validation_epoch_end(self, trainer, pl_module):
    return self._on_epoch_end("val", trainer, pl_module)

  def _render_orbit(self, model, batch, device):
    """Renders a turntable orbit of `item`'s source-view Gaussians. Returns a
    generator of (H,W,3) uint8 frames, or None if the source view seeded zero
    Gaussians (nothing to show) or the render hit a CUDA OOM (see
    module.guarded_render)."""

    B = batch["batch_size"]

    model_pred = model_forward(model, batch, device=device)
    _b, _v, _c, ih, iw = batch["views"]["rgb"].shape

    # TODO: Ground truth override
    # gauss = apply_ground_truth_overrides(gauss, gt, hit, predict_params, device)

    frames = None
    with guarded_render("orbit render"), torch.no_grad():
      # TODO: Actually properly support batch size > 1
      for (
        x_gauss_flat,
        x_pose_gl,
        x_K_image,
      ) in zip(
        flatten_gaussians(
          B,
          model_pred["gauss"],
          batch["source"]["xyz_cam"].to(device),
          batch["source"]["hit"].to(device),
        ),
        batch["source"]["pose_gl"],
        batch["views"]["K_image"][:,0],
      ):
        means_np = x_gauss_flat["means"].detach().cpu().numpy()
        up = _up_in_source_frame(x_pose_gl.numpy())
        poses_gl = _orbit_c2w_gl(means_np, up, self.num_frames, self.elevation_deg)
        c2w_cv = poses_gl @ OPENGL_TO_OPENCV
        viewmats = torch.from_numpy(np.linalg.inv(c2w_cv).astype(np.float32)).to(device)
        # ks = src["K_image"].to(device)[None].expand(len(poses_gl), -1, -1)
        ks = repeat(x_K_image.to(device), "... -> b ...", b=len(poses_gl))

        rgb_out, _, _ = gsplat.rasterization(
          means=x_gauss_flat["means"],
          quats=x_gauss_flat["quats"],
          scales=x_gauss_flat["scales"],
          opacities=x_gauss_flat["opacities"],
          colors=x_gauss_flat["colors"],
          viewmats=viewmats,
          Ks=ks,
          width=int(iw),
          height=int(ih),
          sh_degree=model.max_sh_degree,
          render_mode="RGB",
          packed=True,
        )
        rgb_np = (rgb_out.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        frames = [rgb_np[i] for i in range(rgb_np.shape[0])]
    return frames

  def _on_epoch_end(self, stage, trainer, pl_module):
    # Count of COMPLETED epochs (Lightning hasn't bumped trainer.current_epoch
    # yet at this point in the hook) -- "every_n_epochs=1" means "every
    # epoch", matching conf/train_gs.yaml's comment on that field.
    epoch = trainer.current_epoch + 1
    if epoch % self.every_n_epochs != 0:
      return
    model = pl_module.model
    device = pl_module.device

    wandb_run = pl_module.logger.experiment if pl_module.logger is not None else None
    if wandb_run is None:
      return

    # NOTE(andrei): Not sure if setting model to eval is the right move here,
    # but gonna do it for now.
    with set_mode(model, "eval"), timed(f"orbit@epoch{epoch}"):
      match stage:
        case "train":
          batch = self.train_batch
        case "val":
          batch = self.val_batch
        case _:
          raise RuntimeError(f"Unexpected {stage=}")

      if batch is None:
        return

      frames = self._render_orbit(model, batch, device)

      if frames is None:
        return

      # NOMERGE: ideally the rendering would be the same as orbit_video.py
      tag = f"{stage}/orbit"
      path = os.path.join(self.workdir, f"{tag.replace('/', '_')}.mp4")
      write_mp4(frames, path, self.fps, self.crf)
      wandb_run.log({tag: wandb.Video(path, caption=tag, format="mp4")})

  def on_fit_end(self, trainer, pl_module):
    if self.workdir is not None:
      shutil.rmtree(self.workdir, ignore_errors=True)
      self.workdir = None


@hydra.main(version_base=None, config_path="conf", config_name="train_gs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.seed))
  torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)

  # ---- config validation (unchanged SystemExit checks, plus the new
  # grad_accum_steps==1 assertion for GSFixedViewsDataset -- see
  # train_gs_lightning_plan.md) ----
  if cfg.data.photom_h5 is not None and cfg.data.gauss_h5 is not None:
    raise SystemExit(
      "data.photom_h5 and data.gauss_h5 can't both be set (for now) -- gauss_h5 "
      "is self-sufficient as a training source (it embeds its own primary "
      "view's image/depth/pose, see gs_dataset._read_primary_from_gauss_h5), "
      "so set data.photom_h5=null for direct-supervision-only training, or "
      "data.gauss_h5=null for ordinary photometric training")
  if cfg.data.gauss_h5 is not None and cfg.data.fixed_source_view is None:
    raise SystemExit("data.gauss_h5 requires data.fixed_source_view (single-fixed-view scope only)")
  if (cfg.data.fixed_source_view is not None and cfg.data.photom_h5 is None
      and cfg.data.gauss_h5 is None):
    raise SystemExit("data.fixed_source_view requires at least one of data.photom_h5/data.gauss_h5")
  direct_enabled = any(cfg.loss[k] > 0 for k in (
    "direct_opacity_weight", "direct_scale_weight", "direct_rotation_weight", "direct_color_weight"))
  if direct_enabled and cfg.data.gauss_h5 is None:
    raise SystemExit("loss.direct_*_weight > 0 requires data.gauss_h5 to be set")

  if cfg.data.photom_h5_val is not None and cfg.data.fixed_source_view is None:
    log.info(
      "data.photom_h5_val set -- data.split_fn is ignored, the full "
      "training corpus is used for train_ds.")
  # A real (nonempty, force_render=True) val_ds gets built either from
  # data.photom_h5_val directly, or -- when that's unset -- as a fallback
  # from data.gauss_h5's own embedded views (GaussH5ValDataset, see
  # GSDataModule.setup). val/rec_loss is 0-by-construction when all
  # photometric weights are 0 UNLESS it's the GaussH5ValDataset fallback AND
  # a direct_*_weight is nonzero -- that val_ds's items carry "ground_truth"
  # (unlike a photom_h5_val corpus, which never does), so compute_loss's
  # direct-supervision terms fire during validation too and val/rec_loss
  # picks those up instead of reading 0.
  has_real_val_ds = cfg.data.photom_h5_val is not None or (
    cfg.data.fixed_source_view is not None and cfg.data.gauss_h5 is not None)
  uses_gauss_h5_val_fallback = (
    cfg.data.photom_h5_val is None and cfg.data.fixed_source_view is not None
    and cfg.data.gauss_h5 is not None)

  predict_params = set(cfg.model.predict_params)
  unknown = predict_params - set(PREDICTABLE_PARAMS)
  if unknown:
    raise SystemExit(
      f"model.predict_params has unknown entries {sorted(unknown)} -- "
      f"must be a subset of {PREDICTABLE_PARAMS}")
  if predict_params != set(PREDICTABLE_PARAMS) and cfg.data.gauss_h5 is None:
    raise SystemExit(
      "model.predict_params (restricting which fields the network predicts) "
      "requires data.gauss_h5 to supply the rest")
  for field in PREDICTABLE_PARAMS:
    if field not in predict_params and cfg.loss[f"direct_{field}_weight"] > 0:
      log.warning(
        "model.predict_params excludes '%s' but loss.direct_%s_weight > 0 -- "
        "that field is forced to ground truth everywhere it's used, so this "
        "loss term trains a head with no effect on rendering/output", field, field)

  if cfg.data.fixed_source_view is not None:
    # Single-batch overfit mode (fit_gsplat.py's primary/secondary terms
    # applied here): the exact same source+target views every step, no
    # resampling at all -- nothing to hold out, so val is empty.
    # fixed_target_views only matters when there's a photom_h5 to draw them
    # from -- gauss_h5-only mode has no photom corpus, targets are forced
    # empty (GSDataModule.setup / GSFixedViewsDataset).
    if cfg.data.photom_h5 is not None and cfg.data.fixed_target_views is None:
      raise SystemExit(
        "data.fixed_target_views must be set when data.fixed_source_view + "
        "data.photom_h5 are set")
    if int(cfg.train.grad_accum_steps) != 1:
      # Under Lightning, "epoch" == "optimizer step" for this length-1
      # dataset only when grad_accum_steps == 1 (otherwise it takes
      # grad_accum_steps Lightning-epochs to complete one optimizer step,
      # since the length-1 loader yields exactly one batch per epoch) -- see
      # train_gs_lightning_plan.md. Every fixed-view run this session already
      # used grad_accum_steps=1 anyway (accumulating over the same repeated
      # example is pure wasted compute), so this should never actually fire.
      raise SystemExit(
        "data.fixed_source_view (GSFixedViewsDataset) requires "
        "train.grad_accum_steps == 1 -- see train_gs_lightning_plan.md")

  datamodule = GSDataModule(cfg)
  datamodule.setup()

  with open_dict(cfg):
    # TODO: Grad accum logic
    cfg.total_steps = len(datamodule.train_dataloader()) * cfg.trainer.max_epochs

  model = GSLightningModule(cfg)

  # Stashed into cfg (not just logged) so it lands in wandb's persisted run
  # config below (OmegaConf.to_container(cfg, ...)) -- unlike the console-only
  # "Total params" line from Lightning's own ModelSummary table, this is
  # queryable after the fact. open_dict since Hydra's cfg is struct-locked by
  # default (adding a key not in the yaml schema would otherwise raise
  # ConfigAttributeError).
  with open_dict(cfg):
    cfg.model.num_params = sum(p.numel() for p in model.parameters())

  logger = False
  if cfg.wandb.mode != "disabled":
    tags: list[str] = OmegaConf.to_container(cfg.wandb.tags, resolve=True)

    # NOTE(andrei): Ensure that there's a tag to easily distinguish these runs
    # from others, and that it's the first tag that shows up.
    tag = "3dgs-model"
    with ctl.suppress(ValueError):
      tags.remove(tag)
    tags = [tag, *tags]

    wandb_run = wandb.init(
      project=cfg.wandb.project,
      mode=cfg.wandb.mode,
      tags=tags,
      name=cfg.wandb.name,
      config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger = WandbLogger(experiment=wandb_run)

  callbacks = []
  callbacks += list(hydra.utils.instantiate(cfg.callbacks).values())

  trainer = pl.Trainer(
    **OmegaConf.to_container(cfg.trainer, resolve=True),
    logger=logger,
    callbacks=callbacks,
  )
  with timed("train"):
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
  main()
