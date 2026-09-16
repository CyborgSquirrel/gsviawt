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
        data.h5_paths=/app/bla/lite_blackbg.h5 train.max_epochs=1000

gsplat JIT-compiles CUDA kernels on first import; see _setup_cuda_toolchain
(copied from fit_gsplat.py, which needs the same env setup).
"""

import logging
import math
import os
import shutil
import sys
import sysconfig
import tempfile


def _setup_cuda_toolchain() -> None:
  """Make the pip-installed CUDA toolkit (`nvidia-cuda-nvcc` etc.) usable by
  gsplat's `torch.utils.cpp_extension` JIT build: export CUDA_HOME / PATH and
  add the `libcudart.so` dev symlink the linker's `-lcudart` needs (the wheel
  ships only `libcudart.so.13`). No-ops cleanly if the layout isn't there or a
  system CUDA is already configured."""
  if os.environ.get("CUDA_HOME") and os.path.exists(
      os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")):
    return
  for libdir in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
    cuda_home = os.path.join(libdir, "nvidia", "cu13")
    if not os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
      continue
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    lib = os.path.join(cuda_home, "lib")
    link, real = os.path.join(lib, "libcudart.so"), os.path.join(lib, "libcudart.so.13")
    if os.path.exists(real) and not os.path.exists(link):
      try:
        os.symlink("libcudart.so.13", link)
      except OSError:
        pass
    return


_setup_cuda_toolchain()

import hydra  # noqa: E402
import lightning.pytorch as pl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange, repeat  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from gs_dataset import (  # noqa: E402
  OPENGL_TO_OPENCV, GSFixedViewsDataset, GSPairDataset, H5Catalog, _EmptyDataset, split_by_mesh,
)
from gs_decoder import GaussianResnetDecoder, GSDecoderStack  # noqa: E402
from gs_encoder import GSResnetEncoder  # noqa: E402
from fit_gsplat import SH_C0  # noqa: E402
from orbit_video import look_at_c2w, write_mp4  # noqa: E402
from util import timed  # noqa: E402

log = logging.getLogger("train_gs")


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
    in_channels = 3 + 4 * self.num_layers

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
    if cfg.model.one_gauss_decoder:
      self.decoder = GaussianResnetDecoder(
        self.encoder.num_ch_enc, num_layers=self.num_layers,
        max_sh_degree=self.max_sh_degree, **decoder_kwargs)
    else:
      self.decoder = GSDecoderStack(
        self.encoder.num_ch_enc, num_layers=self.num_layers,
        max_sh_degree=self.max_sh_degree, **decoder_kwargs)

    self.register_buffer("xyz_mean", torch.tensor(list(cfg.model.xyz_mean), dtype=torch.float32))
    self.register_buffer("xyz_std", torch.tensor(list(cfg.model.xyz_std), dtype=torch.float32))

  def build_input(self, rgb, xyz_cam, hit):
    """rgb: (3,IH,IW) in [0,1]. xyz_cam: (L,DH,DW,3) NaN-invalid, hit:
    (L,DH,DW) bool, both at depth-peel resolution. Returns (3+4L,DH,DW),
    resizing rgb to the depth-peel resolution if they differ."""
    dh, dw = hit.shape[1], hit.shape[2]
    if rgb.shape[-2:] != (dh, dw):
      rgb = F.interpolate(rgb[None], size=(dh, dw), mode="bilinear", align_corners=False)[0]
    rgb_norm = (rgb - 0.45) / 0.225

    xyz_filled = torch.nan_to_num(xyz_cam, nan=0.0)
    xyz_norm = (xyz_filled - self.xyz_mean) / self.xyz_std   # (L,DH,DW,3)
    xyz_norm = rearrange(xyz_norm, "l h w c -> l c h w")
    valid = hit.to(xyz_norm.dtype).unsqueeze(1)               # (L,1,DH,DW)
    per_layer = torch.cat([xyz_norm, valid], dim=1).reshape(-1, dh, dw)  # (4L,DH,DW)
    return torch.cat([rgb_norm, per_layer], dim=0)            # (3+4L,DH,DW)

  def forward(self, rgb, xyz_cam, hit, fx):
    x = self.build_input(rgb, xyz_cam, hit)
    dh, dw = x.shape[-2:]
    # The 5-level U-Net halves spatial dims 4x (conv1 + maxpool + 2 more
    # strided stages) then doubles back up 5x via nearest-neighbor upsample;
    # that only round-trips exactly when H/W are multiples of 32 (Flash3D's
    # own inputs are; render_objaverse.py's 504x504 renders aren't), so pad
    # up to the next multiple of 32 before the encoder and crop the decoder's
    # output back to (dh, dw) -- same fix Flash3D applies via its
    # pad_border_aug, just replicate-padded to the exact multiple instead of
    # a fixed border.
    pad_h, pad_w = (-dh) % 32, (-dw) % 32
    if pad_h or pad_w:
      x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    feats = self.encoder(x.unsqueeze(0))
    out = self.decoder(feats)
    out = {k: v[0, :, :, :dh, :dw] for k, v in out.items()}   # (L,C,H,W)

    # Anchor "scale" to each pixel's own real geometric footprint (its
    # inter-pixel spacing at its own depth, depth/fx for a fronto-parallel
    # approximation) instead of an unconstrained absolute value, WITH A HARD
    # FLOOR (min_scale_mult), not just a good initial value: gsplat's
    # antialiasing eps2d floor (hard-coded minimum ~3px projected size, see
    # gsplat.rendering docs) makes the photometric loss's gradient w.r.t.
    # scale vanish once a Gaussian is already sub-floor -- so nothing during
    # training actually stops `exp(raw)` from drifting back down there over
    # enough steps even after starting at a sane value (confirmed: a 15-min
    # run looked fine, a 65-min run on the same data drifted back into the
    # grid artifact, with a few neighbors ballooning outward to compensate
    # for the ones that collapsed). Adding a constant floor to the
    # dimensionless multiplier -- not just to its initial value -- closes the
    # loophole structurally: scale can never render below

    # min_scale_mult * pixel_scale, regardless of what raw drifts to.
    # pixel_scale = torch.nan_to_num(xyz_cam[..., 2], nan=1.0) / fx   # (L,H,W)
    # multiplier = self.min_scale_mult + out["scale"]                 # (L,3,H,W), floor + exp(raw)*scale_lambda
    # out["scale"] = multiplier * pixel_scale.unsqueeze(1)            # (L,3,H,W)

    return out


def flatten_gaussians(xyz_cam, hit, gauss):
  """xyz_cam: (L,H,W,3) f32 camera-space positions (NaN where invalid), same
  layer/pixel order as `hit` and every entry of `gauss` (each (L,C,H,W)).
  Filters to valid entries and returns gsplat.rasterization-ready tensors."""
  means_flat = xyz_cam.reshape(-1, 3)
  valid_flat = hit.reshape(-1)

  opacity_flat = rearrange(gauss["opacity"], "l c h w -> (l h w c)") * valid_flat.to(gauss["opacity"].dtype)
  scale_flat = rearrange(gauss["scale"], "l c h w -> (l h w) c")
  rotation_flat = rearrange(gauss["rotation"], "l c h w -> (l h w) c")
  sh_dc_flat = rearrange(gauss["sh_dc"], "l c h w -> (l h w) 1 c")
  if "sh_rest" in gauss:
    sh_rest_flat = rearrange(gauss["sh_rest"], "l (k c) h w -> (l h w) k c", c=3)
    colors_flat = torch.cat([sh_dc_flat, sh_rest_flat], dim=1)
  else:
    colors_flat = sh_dc_flat

  keep = valid_flat
  return {
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


def _predict_params(cfg):
  """The set of PREDICTABLE_PARAMS entries in cfg.model.predict_params."""
  return set(cfg.model.predict_params)


def run_model_source(model, item, device, predict_params):
  """Runs the model on `item`'s source view once. Returns (gauss, gauss_render,
  hit, flat): `gauss` is the raw (L,C,H,W) per-layer decoder output dict --
  always the network's own unmodified prediction, so compute_direct_loss's
  metrics stay meaningful even for fields not being trained. `gauss_render` is
  the same dict with any non-predicted fields overridden by ground truth (see
  apply_ground_truth_overrides) -- what rendering and visualization should
  actually use (identical to `gauss` when predict_params covers every field).
  `hit` is (L,H,W) bool. `flat` is flatten_gaussians' output built from
  `gauss_render` -- shared by both the photometric render path and the
  direct-parameter-loss path below (see compute_loss), computed exactly once
  regardless of which (or both) are active this step."""
  src = item["source"]
  rgb = src["rgb"].to(device)
  xyz_cam = src["xyz_cam"].to(device)
  hit = src["hit"].to(device)
  fx = src["K_depth"][0, 0].to(device)

  gauss = model(rgb, xyz_cam, hit, fx)
  gauss_render = gauss
  gt = item.get("ground_truth")
  if gt is not None:
    gauss_render = apply_ground_truth_overrides(gauss, gt, hit, predict_params, device)
  flat = flatten_gaussians(xyz_cam, hit, gauss_render)
  return gauss, gauss_render, hit, flat


def render_photometric(item, flat, sh_degree, device):
  """Renders source+targets in one gsplat call -- the actually-expensive
  part of what used to be render_scene, split out so it can be skipped
  entirely when no photometric loss term is active this step (see
  compute_loss's `need_photo` gate). Returns (pred_rgb, pred_alpha, gt_rgb,
  gt_alpha), all (V,H,W,3)/(V,H,W,1)."""
  import gsplat

  views = [item["source"]] + item["targets"]
  viewmats = rearrange([v["viewmat"] for v in views], "v a b -> v a b").to(device)
  Ks = rearrange([v["K_image"] for v in views], "v a b -> v a b").to(device)
  gt_rgb = rearrange([v["rgb"] for v in views], "v c h w -> v h w c").to(device)
  gt_alpha = rearrange([v["alpha"] for v in views], "v c h w -> v h w c").to(device)
  ih, iw = gt_rgb.shape[1:3]

  pred_rgb, pred_alpha, _ = gsplat.rasterization(
    means=flat["means"], quats=flat["quats"], scales=flat["scales"],
    opacities=flat["opacities"], colors=flat["colors"],
    viewmats=viewmats, Ks=Ks, width=int(iw), height=int(ih),
    sh_degree=sh_degree, render_mode="RGB", packed=True,
  )
  return pred_rgb.clamp(0.0, 1.0), pred_alpha.clamp(0.0, 1.0), gt_rgb, gt_alpha


# ---------------------------------------------------------------------------
# losses (SSIM ported from fit_gsplat.py's hand-rolled version, kept
# unreduced over the view axis so source/target losses can be split out)
# ---------------------------------------------------------------------------

def _gaussian_window(size=11, sigma=1.5, device="cpu"):
  coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
  g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
  g = g / g.sum()
  return (g[:, None] * g[None, :])[None, None]


def ssim_map(x, y, window):
  """x, y: (V,C,H,W) in [0,1]. Standard 11x11 Gaussian-window SSIM, returned
  unreduced (per-pixel) so callers can average per-view instead of globally."""
  c = x.shape[1]
  w = window.expand(c, 1, -1, -1)
  pad = w.shape[-1] // 2
  mu_x = F.conv2d(x, w, padding=pad, groups=c)
  mu_y = F.conv2d(y, w, padding=pad, groups=c)
  mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
  sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
  sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
  sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
  c1, c2 = 0.01 ** 2, 0.03 ** 2
  return ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))


def per_view_losses(pred_rgb, pred_alpha, gt_rgb, gt_alpha, window, cfg_loss):
  """pred/gt _rgb: (V,H,W,3), _alpha: (V,H,W,1). Returns (per_view (V,) total
  weighted loss, parts dict of (V,) component losses), all premultiplied by
  alpha (matches fit_gsplat.py's convention: bg stays black on both sides)."""
  pred_c = rearrange(pred_rgb * pred_alpha, "v h w c -> v c h w")
  gt_c = rearrange(gt_rgb * gt_alpha, "v h w c -> v c h w")
  l1 = (pred_c - gt_c).abs().mean(dim=(1, 2, 3))
  dssim = 1.0 - ssim_map(pred_c, gt_c, window).mean(dim=(1, 2, 3))
  mask_l1 = (pred_alpha - gt_alpha).abs().mean(dim=(1, 2, 3))
  per_view = (
    cfg_loss.l1_weight * l1 + cfg_loss.ssim_weight * dssim + cfg_loss.mask_weight * mask_l1
  )
  return per_view, {"l1": l1, "ssim": dssim, "mask": mask_l1}


def compute_direct_loss(gauss, gt, hit, cfg_loss, device):
  """Direct per-pixel supervision of the model's own predicted Gaussian
  params against a fit_gsplat.py ground-truth grid (gs_dataset's
  _load_ground_truth output -- rotation already converted into the source
  camera's own frame, positions never compared since they're fixed to the
  same point cloud by construction). `gauss`: raw decoder output, each
  (L,C,H,W). `hit`: (L,H,W) bool. Returns (total, parts) -- `parts` only has
  keys for fields whose weight is actually nonzero (omit-if-skipped, same
  convention as the rest of this file's metrics dicts)."""
  mask = hit & gt["valid"].to(device)
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
    # nan_to_num guard: same reasoning as direct_scale_weight/direct_opacity_weight
    # above -- gt["color"] is NaN-padded outside `mask`, and pow(2)'s backward
    # (2*x) propagates that NaN through the unselected positions' local Jacobian
    # even though their incoming gradient is zero (0*NaN=NaN).
    shdc_pred = rearrange(gauss["sh_dc"], "l c h w -> l h w c")
    color_pred = SH_C0 * shdc_pred + 0.5
    gt_color = torch.nan_to_num(gt["color"].to(device), nan=0.0)
    parts["color"] = (color_pred - gt_color).pow(2)[mask].mean()
    total = total + cfg_loss.direct_color_weight * parts["color"]

  return total, parts


def compute_loss(model, item, cfg, device, window, force_render=False):
  gauss, gauss_render, hit, flat = run_model_source(model, item, device, _predict_params(cfg))

  need_photo = (force_render or cfg.loss.l1_weight > 0
                or cfg.loss.ssim_weight > 0 or cfg.loss.mask_weight > 0)
  need_direct = (cfg.loss.direct_opacity_weight > 0 or cfg.loss.direct_scale_weight > 0
                 or cfg.loss.direct_rotation_weight > 0 or cfg.loss.direct_color_weight > 0)

  loss = torch.zeros((), device=device)
  pred_rgb = pred_alpha = gt_rgb = gt_alpha = None
  metrics = {}

  if need_photo:
    pred_rgb, pred_alpha, gt_rgb, gt_alpha = render_photometric(item, flat, cfg.model.max_sh_degree, device)
    per_view, parts = per_view_losses(pred_rgb, pred_alpha, gt_rgb, gt_alpha, window, cfg.loss)
    loss = loss + per_view.mean()
    metrics["l1"] = parts["l1"].mean().detach()
    metrics["ssim"] = parts["ssim"].mean().detach()
    metrics["mask"] = parts["mask"].mean().detach()
    metrics["loss_source"] = per_view[0].detach()
    n_targets = per_view.shape[0] - 1
    if n_targets > 0:
      metrics["loss_targets_mean"] = per_view[1:].mean().detach()

  scale_reg = torch.zeros((), device=device)
  if cfg.loss.scale_reg_weight > 0:
    big = flat["scales"][flat["scales"] > cfg.loss.scale_reg_thresh]
    if big.numel() > 0:
      scale_reg = big.mean()
      loss = loss + cfg.loss.scale_reg_weight * scale_reg

  # SH color coefficients are unconstrained (Flash3D's own parameterization --
  # see gs_decoder.py), and empirically prone to runaway blowup for
  # off-training-direction views: since SH evaluation is view-direction
  # dependent, an unregularized network can drive coefficients to extreme
  # magnitudes that happen to cancel out for whichever views were sampled
  # this step, producing large view-dependent color error (visible as
  # rainbow/static artifacts) on any other view. Softly penalize magnitude
  # above a sane bound (dc=+-2 already saturates degree-0 color via
  # SH_C0*dc+0.5) rather than constraining the activation itself, so the
  # parameterization stays identical to Flash3D's when nothing has gone wrong.
  color_reg = torch.zeros((), device=device)
  if cfg.loss.color_reg_weight > 0:
    big_color = flat["colors"].abs()
    big_color = big_color[big_color > cfg.loss.color_reg_thresh]
    if big_color.numel() > 0:
      color_reg = big_color.mean()
      loss = loss + cfg.loss.color_reg_weight * color_reg

  if need_direct:
    gt = item.get("ground_truth")
    if gt is None:
      raise ValueError(
        "loss.direct_*_weight > 0 but item has no 'ground_truth' -- set data.ground_truth_h5")
    direct_total, direct_parts = compute_direct_loss(gauss, gt, hit, cfg.loss, device)
    loss = loss + direct_total
    for k, v in direct_parts.items():
      metrics[f"direct_{k}"] = v.detach()

  metrics["loss"] = loss.detach()
  metrics["scale_reg"] = scale_reg.detach()
  metrics["color_reg"] = color_reg.detach()
  metrics["mean_opacity"] = flat["opacities"].mean().detach() if flat["opacities"].numel() else torch.zeros((), device=device)
  metrics["mean_scale"] = flat["scales"].mean().detach() if flat["scales"].numel() else torch.zeros((), device=device)
  metrics["frac_kept"] = torch.tensor(
    flat["means"].shape[0] / max(1, item["source"]["hit"].numel()), device=device)

  return loss, metrics, (pred_rgb, pred_alpha, gt_rgb, gt_alpha, gauss_render)


# ---------------------------------------------------------------------------
# wandb image panels
# ---------------------------------------------------------------------------

def _val_panel(gt_rgb, pred_rgb):
  """gt_rgb/pred_rgb: (V,H,W,3) in [0,1]. Returns one uint8 (H, V*3*W, 3)
  image: for each view, [GT | render | |diff|] side by side."""
  v = gt_rgb.shape[0]
  rows = []
  for i in range(v):
    g = (gt_rgb[i].detach().cpu().numpy() * 255).astype(np.uint8)
    r = (pred_rgb[i].detach().cpu().numpy() * 255).astype(np.uint8)
    d = ((gt_rgb[i] - pred_rgb[i]).abs().detach().cpu().numpy() * 255).astype(np.uint8)
    rows.append(np.concatenate([g, r, d], axis=1))
  return np.concatenate(rows, axis=0)


def _layer_opacity_panel(gauss):
  """gauss["opacity"]: (L,1,H,W). Returns one uint8 (H, L*W) grayscale image,
  one column per layer's mean opacity."""
  op = gauss["opacity"][:, 0].detach().cpu().numpy()   # (L,H,W)
  cols = [(op[l] * 255).astype(np.uint8) for l in range(op.shape[0])]
  return np.concatenate(cols, axis=1)


def log_render_panel(wandb_run, step, tag, pred_rgb, gt_rgb, gauss):
  """tag: e.g. "train" or "val" -- one row per rendered view (source/primary
  first, then each target/secondary), [GT | render | |diff|] per row."""
  import wandb
  wandb_run.log({
    f"{tag}/panel": wandb.Image(_val_panel(gt_rgb, pred_rgb), caption="GT | render | |diff|, one row per view (source first, then targets)"),
    f"{tag}/layer_opacity": wandb.Image(_layer_opacity_panel(gauss), caption="mean opacity per layer, front layer first"),
  }, step=step)


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


def render_orbit(model, item, cfg, device):
  """Renders a turntable orbit of `item`'s source-view Gaussians. Returns a
  generator of (H,W,3) uint8 frames, or None if the source view seeded zero
  Gaussians (nothing to show)."""
  src = item["source"]
  rgb, xyz_cam, hit = src["rgb"].to(device), src["xyz_cam"].to(device), src["hit"].to(device)
  fx = src["K_depth"][0, 0].to(device)
  ih, iw = src["rgb"].shape[-2:]

  predict_params = _predict_params(cfg)
  with torch.no_grad():
    gauss = model(rgb, xyz_cam, hit, fx)
    gt = item.get("ground_truth")
    if gt is not None:
      gauss = apply_ground_truth_overrides(gauss, gt, hit, predict_params, device)
    flat = flatten_gaussians(xyz_cam, hit, gauss)
  if flat["means"].shape[0] == 0:
    return None

  means_np = flat["means"].detach().cpu().numpy()
  up = _up_in_source_frame(src["pose_gl"].numpy())
  poses_gl = _orbit_c2w_gl(means_np, up, cfg.orbit.num_frames, cfg.orbit.elevation_deg)
  c2w_cv = poses_gl @ OPENGL_TO_OPENCV
  viewmats = torch.from_numpy(np.linalg.inv(c2w_cv).astype(np.float32)).to(device)
  ks = src["K_image"].to(device)[None].expand(len(poses_gl), -1, -1)

  import gsplat
  with torch.no_grad():
    rgb_out, _, _ = gsplat.rasterization(
      means=flat["means"], quats=flat["quats"], scales=flat["scales"],
      opacities=flat["opacities"], colors=flat["colors"],
      viewmats=viewmats, Ks=ks, width=int(iw), height=int(ih),
      sh_degree=cfg.model.max_sh_degree, render_mode="RGB", packed=True,
    )
  frames = (rgb_out.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
  return (frames[i] for i in range(frames.shape[0]))


def log_orbit_video(wandb_run, step, tag, frames, fps, crf, workdir):
  import wandb
  path = os.path.join(workdir, f"{tag.replace('/', '_')}.mp4")
  write_mp4(frames, path, fps, crf)
  wandb_run.log({tag: wandb.Video(path, caption=tag, format="mp4")}, step=step)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

class GSDataModule(pl.LightningDataModule):
  """Mechanical extraction of the dataset-construction branch that used to
  live at the top of main(): single-batch overfit mode (GSFixedViewsDataset)
  vs. multi-scene mode (H5Catalog + split_by_mesh + GSPairDataset). See this
  module's docstring / gs_dataset.py for what each dataset type means. The
  data.fixed_target_views / train.grad_accum_steps==1 SystemExit checks live
  in main()'s config-validation block, not here -- see
  train_gs_lightning_plan.md.

  setup() is idempotent (guarded on self.train_ds): main() calls it once
  explicitly, before GSLightningModule exists, to get len(train_ds) for the
  derived LR-schedule step count (see main()); Lightning's own fit() flow
  calls it again regardless, so it must be a no-op the second time."""

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
      # resampling at all -- nothing to hold out, so val is empty.
      h5_path = cfg.data.h5_paths if isinstance(cfg.data.h5_paths, str) else cfg.data.h5_paths[0]
      self.train_ds = GSFixedViewsDataset(
        h5_path, source_view=cfg.data.fixed_source_view,
        target_views=list(cfg.data.fixed_target_views), num_layers=cfg.data.num_layers,
        ground_truth_h5=cfg.data.ground_truth_h5,
      )
      self.val_ds = _EmptyDataset()
    else:
      catalog = H5Catalog(
        cfg.data.h5_paths,
        H5Catalog.path().alias("path"),
        H5Catalog.index().alias("view_idx"),
        H5Catalog.dataset("mesh_index").alias("mesh_id"),
      )
      train_catalog, val_catalog = split_by_mesh(catalog, val_fraction=cfg.data.val_fraction, seed=cfg.train.seed)
      self.train_ds = GSPairDataset(
        train_catalog, num_layers=cfg.data.num_layers, num_target_views=cfg.data.num_target_views,
        seed=cfg.train.seed, deterministic_targets=False,
      )
      self.val_ds = GSPairDataset(
        val_catalog, num_layers=cfg.data.num_layers, num_target_views=cfg.data.num_target_views,
        seed=cfg.train.seed, deterministic_targets=True,
      )
    if len(self.train_ds) == 0:
      raise SystemExit("train split is empty -- check data.h5_paths / data.val_fraction")

  def train_dataloader(self):
    return DataLoader(
      self.train_ds, batch_size=1, shuffle=True, num_workers=self.cfg.train.num_workers,
      collate_fn=lambda batch: batch[0], persistent_workers=self.cfg.train.num_workers > 0,
    )

  def val_dataloader(self):
    # cfg.val.every<=0 (validation disabled entirely, independent of whether
    # there's data to validate against) is handled by main()'s
    # Trainer(limit_val_batches=...) -- this only covers "no data at all"
    # (replaces today's _EmptyDataset-guarded skip in run_validation).
    if len(self.val_ds) == 0:
      return None
    return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: batch[0])


class GSLightningModule(pl.LightningModule):
  """Wraps GSModel + the training/validation step logic that used to be the
  hand-rolled loop body in main(). Manual optimization
  (automatic_optimization=False): Lightning's own gradient accumulation
  divides by the *configured* accumulate_grad_batches regardless of how many
  micro-batches were actually accumulated -- silently wrong (a
  4x-effective-LR bug observed for GSFixedViewsDataset, whose length-1
  DataLoader doesn't have the relationship Lightning's automatic
  accumulation assumes between "epoch" and "accumulation window"). Manual
  accumulation via self._micro_step sidesteps that: we count exactly how
  many micro-batches went into every optimizer step ourselves.

  self.global_train_step is OUR OWN real-optimizer-step counter -- distinct
  from Lightning's self.global_step (which counts training_step calls, i.e.
  micro-batches, under manual optimization). It's the wandb x-axis and the
  checkpoint-cadence key, exactly matching the old hand-rolled loop's `step`
  semantics."""

  def __init__(self, cfg, total_steps):
    super().__init__()
    self.cfg = cfg
    self.total_steps = int(total_steps)
    self.automatic_optimization = False

    with timed("build_model"):
      self.model = GSModel(cfg)
    self.register_buffer("window", _gaussian_window(), persistent=False)

    self._grad_accum_steps = int(cfg.train.grad_accum_steps)
    self._micro_step = 0
    self._accum_loss = 0.0
    self._accum_metrics = {}
    self.global_train_step = 0

    ckpt_dir = cfg.checkpoint.dir or os.getcwd()
    os.makedirs(ckpt_dir, exist_ok=True)
    self.ckpt_path = os.path.join(ckpt_dir, "train_gs.ckpt.pt")

  def configure_optimizers(self):
    opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.cfg.train.lr))

    # Linear warmup -> cosine decay to min_lr, built from torch's own
    # scheduler classes (LinearLR + CosineAnnealingLR chained via
    # SequentialLR) rather than a hand-rolled LambdaLR closure.
    warmup_steps = max(1, int(self.cfg.train.warmup_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
      opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
      opt, T_max=max(1, self.total_steps - warmup_steps), eta_min=float(self.cfg.train.min_lr))
    sched = torch.optim.lr_scheduler.SequentialLR(
      opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    # "interval"/"frequency" are automatic-optimization-only metadata -- under
    # automatic_optimization=False we step `sched` ourselves (training_step),
    # but returning it this way still lets Lightning checkpoint its
    # state_dict alongside the optimizer's.
    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

  def training_step(self, batch, batch_idx):
    item = batch
    loss, metrics, _ = compute_loss(self.model, item, self.cfg, self.device, self.window)
    self.manual_backward(loss / self._grad_accum_steps)
    self._micro_step += 1
    self._accum_loss += loss.item() / self._grad_accum_steps
    for k, v in metrics.items():
      self._accum_metrics[k] = self._accum_metrics.get(k, 0.0) + v.item() / self._grad_accum_steps

    if self._micro_step < self._grad_accum_steps:
      return  # still accumulating this optimizer step's gradient

    opt = self.optimizers()
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip_norm))
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched = self.lr_schedulers()
    sched.step()

    accum_loss, accum_metrics = self._accum_loss, self._accum_metrics
    self._micro_step = 0
    self._accum_loss = 0.0
    self._accum_metrics = {}

    step = self.global_train_step
    if step % self.cfg.train.log_every == 0:
      self._log_train_console(step, accum_loss, accum_metrics, grad_norm)
      if self.logger is not None:
        self._log_train_wandb(step, accum_loss, accum_metrics, grad_norm, sched)

    if self.cfg.checkpoint.every > 0 and step > 0 and step % self.cfg.checkpoint.every == 0:
      self.trainer.save_checkpoint(self.ckpt_path)
      log.info("checkpoint -> %s", self.ckpt_path)

    self.global_train_step += 1

  def _log_train_console(self, step, accum_loss, accum_metrics, grad_norm):
    if "l1" in accum_metrics:
      photo_str = (
        f"(l1 {accum_metrics['l1']:.5f} ssim {accum_metrics['ssim']:.5f} "
        f"mask {accum_metrics['mask']:.5f})  src {accum_metrics['loss_source']:.5f}"
      )
    else:
      photo_str = "(photometric off)"
    tgt_str = f"tgt {accum_metrics['loss_targets_mean']:.5f}" if "loss_targets_mean" in accum_metrics else "tgt n/a"
    direct_str = "  ".join(
      f"direct_{k} {accum_metrics[f'direct_{k}']:.5f}"
      for k in ("opacity", "scale", "rotation", "color") if f"direct_{k}" in accum_metrics
    )
    log.info(
      "step %d/%d  loss %.5f  %s %s  %s  kept %.3f  grad_norm %.3f",
      step, self.total_steps, accum_loss, photo_str, tgt_str, direct_str,
      accum_metrics["frac_kept"], float(grad_norm),
    )

  def _log_train_wandb(self, step, accum_loss, accum_metrics, grad_norm, sched):
    log_dict = {
      "train/loss": accum_loss,
      "train/grad_norm": float(grad_norm),
      "train/lr": sched.get_last_lr()[0],
      "train/epoch": self.current_epoch,
    }
    for key, wandb_key in (
      ("l1", "train/loss_l1"),
      ("ssim", "train/loss_ssim"),
      ("mask", "train/loss_mask"),
      ("loss_source", "train/loss_source"),
      ("loss_targets_mean", "train/loss_targets_mean"),
      ("direct_opacity", "train/loss_direct_opacity"),
      ("direct_scale", "train/loss_direct_scale"),
      ("direct_rotation", "train/loss_direct_rotation"),
      ("direct_color", "train/loss_direct_color"),
      ("scale_reg", "train/loss_scale_reg"),
      ("color_reg", "train/loss_color_reg"),
      ("mean_opacity", "train/mean_opacity"),
      ("mean_scale", "train/mean_scale"),
      ("frac_kept", "train/frac_gaussians_kept"),
    ):
      if key in accum_metrics:
        log_dict[wandb_key] = accum_metrics[key]
    self.logger.experiment.log(log_dict, step=step)

  def on_validation_epoch_start(self):
    self._val_losses = []
    self._val_losses_source = []
    self._val_losses_targets = []

  def validation_step(self, batch, batch_idx):
    if self.trainer.sanity_checking:
      # Lightning's own convention: self.log(name, value) auto-reduces
      # (mean by default) over the epoch and is keyed to
      # self.trainer.global_step, which stays 0 through the whole sanity
      # check (it only advances on real optimizer steps) -- so a sanity-check
      # self.log call would log real-looking numbers under the same step key
      # training will use for its first genuine log. We bypass self.log
      # entirely and use our own counter/omit-if-absent-key convention (see
      # _log_train_wandb) instead, so this guard just needs to prove
      # validation_step doesn't crash during the sanity pass, not produce
      # numbers worth keeping.
      return
    item = batch
    _, metrics, extras = compute_loss(self.model, item, self.cfg, self.device, self.window, force_render=True)
    self._val_losses.append(metrics["loss"].item())
    self._val_losses_source.append(metrics["loss_source"].item())
    if "loss_targets_mean" in metrics:
      self._val_losses_targets.append(metrics["loss_targets_mean"].item())
    if batch_idx == 0 and self.logger is not None:
      pred_rgb, _, gt_rgb, _, gauss = extras
      log_render_panel(self.logger.experiment, self.global_train_step, "val", pred_rgb, gt_rgb, gauss)

  def on_validation_epoch_end(self):
    if self.trainer.sanity_checking or not self._val_losses:
      return
    val_metrics = {
      "val/rec_loss": float(np.mean(self._val_losses)),
      "val/loss_source": float(np.mean(self._val_losses_source)),
    }
    if self._val_losses_targets:
      val_metrics["val/loss_targets_mean"] = float(np.mean(self._val_losses_targets))
    log.info("step %d  validation: %s", self.global_train_step, val_metrics)
    if self.logger is not None:
      self.logger.experiment.log(val_metrics, step=self.global_train_step)

  def on_train_end(self):
    self.trainer.save_checkpoint(self.ckpt_path)
    log.info("final checkpoint -> %s", self.ckpt_path)


class OrbitCallback(pl.Callback):
  """Turntable orbit-video + static render-panel preview, epoch-cadenced
  (see conf/train_gs.yaml's orbit.*). Ported from the old hand-rolled loop's
  maybe_render_orbits/orbit_train_item/orbit_val_item machinery unchanged:
  train_ds[0]/val_ds[0] re-randomize their source/target views on every
  access (GSPairDataset.__getitem__), so the exact same fixed item has to be
  snapshotted ONCE (on_fit_start) and reused every render, or progress
  wouldn't be comparable over time."""

  def __init__(self, cfg):
    self.cfg = cfg
    self.orbit_train_item = None
    self.orbit_val_item = None
    self.orbit_workdir = None

  def on_fit_start(self, trainer, pl_module):
    dm = trainer.datamodule
    self.orbit_train_item = dm.train_ds[0]
    self.orbit_val_item = dm.val_ds[0] if len(dm.val_ds) > 0 else None
    self.orbit_workdir = tempfile.mkdtemp(prefix="train_gs_orbit_")

  def on_train_epoch_end(self, trainer, pl_module):
    cfg = self.cfg
    # Count of COMPLETED epochs (Lightning hasn't bumped trainer.current_epoch
    # yet at this point in the hook) -- "every_n_epochs=1" means "every
    # epoch", matching conf/train_gs.yaml's comment on that field.
    epoch = trainer.current_epoch + 1
    if epoch % max(1, int(cfg.orbit.every_n_epochs)) != 0:
      return
    model = pl_module.model
    device = pl_module.device
    step = pl_module.global_train_step
    wandb_run = pl_module.logger.experiment if pl_module.logger is not None else None
    model.eval()
    with timed(f"orbit_preview@epoch{epoch}"):
      for tag, item in (("train", self.orbit_train_item), ("val", self.orbit_val_item)):
        if item is None:
          continue
        frames = render_orbit(model, item, cfg, device)
        if frames is None:
          log.warning("orbit/%s: source view seeded zero Gaussians, skipping orbit", tag)
        elif wandb_run is not None:
          log_orbit_video(wandb_run, step, f"orbit/{tag}", frames, cfg.orbit.fps, cfg.orbit.crf, self.orbit_workdir)
        # Also log a static [GT | render | |diff|] panel for the source
        # (primary) view and every target (secondary) view of this same
        # fixed item, on the same cheap cadence -- covers the case with no
        # real val split (e.g. single-batch overfit mode), where
        # validation_step never fires.
        if wandb_run is not None:
          with torch.no_grad():
            _, _, extras = compute_loss(model, item, cfg, device, pl_module.window, force_render=True)
          pred_rgb, _, gt_rgb, _, gauss = extras
          log_render_panel(wandb_run, step, tag, pred_rgb, gt_rgb, gauss)
    model.train()

  def on_fit_end(self, trainer, pl_module):
    if self.orbit_workdir is not None:
      shutil.rmtree(self.orbit_workdir, ignore_errors=True)
      self.orbit_workdir = None


@hydra.main(version_base=None, config_path="conf", config_name="train_gs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.train.seed))
  torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)
  if cfg.train.accelerator == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
  elif cfg.train.accelerator == "cuda" and not torch.cuda.is_available():
    log.warning("cuda not available, falling back to cpu")
    device = "cpu"
  else:
    device = cfg.train.accelerator

  # ---- config validation (unchanged SystemExit checks, plus the new
  # grad_accum_steps==1 assertion for GSFixedViewsDataset -- see
  # train_gs_lightning_plan.md) ----
  if cfg.data.ground_truth_h5 is not None and cfg.data.fixed_source_view is None:
    raise SystemExit("data.ground_truth_h5 requires data.fixed_source_view (single-fixed-view scope only)")
  direct_enabled = any(cfg.loss[k] > 0 for k in (
    "direct_opacity_weight", "direct_scale_weight", "direct_rotation_weight", "direct_color_weight"))
  if direct_enabled and cfg.data.ground_truth_h5 is None:
    raise SystemExit("loss.direct_*_weight > 0 requires data.ground_truth_h5 to be set")

  predict_params = _predict_params(cfg)
  unknown = predict_params - set(PREDICTABLE_PARAMS)
  if unknown:
    raise SystemExit(
      f"model.predict_params has unknown entries {sorted(unknown)} -- "
      f"must be a subset of {PREDICTABLE_PARAMS}")
  if predict_params != set(PREDICTABLE_PARAMS) and cfg.data.ground_truth_h5 is None:
    raise SystemExit(
      "model.predict_params (restricting which fields the network predicts) "
      "requires data.ground_truth_h5 to supply the rest")
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
    if cfg.data.fixed_target_views is None:
      raise SystemExit("data.fixed_target_views must be set when data.fixed_source_view is set")
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
  steps_per_epoch = math.ceil(len(datamodule.train_dataloader()) / int(cfg.train.grad_accum_steps))
  total_steps = int(cfg.train.max_epochs) * steps_per_epoch

  model = GSLightningModule(cfg, total_steps=total_steps)

  log.info("train=%d val=%d views, device=%s, output=%s, total_steps=%d",
           len(datamodule.train_ds), len(datamodule.val_ds), device, model.ckpt_path, total_steps)

  logger = False
  if cfg.wandb.mode != "disabled":
    import wandb
    wandb_run = wandb.init(
      project=cfg.wandb.project, mode=cfg.wandb.mode, tags=list(cfg.wandb.tags),
      name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger = WandbLogger(experiment=wandb_run)

  # GSDataModule.val_dataloader() returns None when there's no val split at
  # all (fixed-view overfit mode). Lightning does NOT treat that as "skip
  # validation" -- it raises TypeError the next time it actually tries to
  # iterate the dataloader, whether that's the sanity check or (if the
  # sanity check was disabled) the first real check_val_every_n_epoch
  # boundary later in training. limit_val_batches=0 is the only setting that
  # reliably keeps Lightning from calling val_dataloader() at all.
  has_val = len(datamodule.val_ds) > 0
  trainer = pl.Trainer(
    max_epochs=int(cfg.train.max_epochs),
    max_steps=-1,
    accelerator="gpu" if device == "cuda" else "cpu",
    devices=1,
    logger=logger,
    callbacks=[OrbitCallback(cfg)] if cfg.orbit.enabled else [],
    enable_checkpointing=False,   # we save checkpoints ourselves (see
                                    # GSLightningModule.training_step /
                                    # on_train_end), driven by our own step
                                    # counter -- not Lightning's ModelCheckpoint.
    check_val_every_n_epoch=max(1, int(cfg.val.every)),
    limit_val_batches=0 if (cfg.val.every <= 0 or not has_val) else 1.0,
    num_sanity_val_steps=0 if not has_val else 2,
    log_every_n_steps=1,   # irrelevant to us -- we bypass self.log entirely
                             # and log via self.logger.experiment.log
                             # ourselves; keeps Lightning's unrelated internal
                             # metric-aggregation from warning about the
                             # default (50) exceeding a length-1 DataLoader.
    default_root_dir=os.getcwd(),
  )
  with timed("train"):
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
  main()
