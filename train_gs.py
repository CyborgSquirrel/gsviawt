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
        data.h5_paths=/app/bla/lite_blackbg.h5 train.max_steps=1000

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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from gs_dataset import (  # noqa: E402
  OPENGL_TO_OPENCV, GSFixedViewsDataset, GSPairDataset, GSViewsDataset, _EmptyDataset, split_by_mesh,
)
from gs_decoder import GaussianResnetDecoder, GSDecoderStack  # noqa: E402
from gs_encoder import GSResnetEncoder  # noqa: E402
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
    pixel_scale = torch.nan_to_num(xyz_cam[..., 2], nan=1.0) / fx   # (L,H,W)
    multiplier = self.min_scale_mult + out["scale"]                 # (L,3,H,W), floor + exp(raw)*scale_lambda
    out["scale"] = multiplier * pixel_scale.unsqueeze(1)            # (L,3,H,W)
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


def render_scene(model, item, sh_degree, device):
  """Runs the model on `item`'s source view and renders source+targets in
  one gsplat call. Returns (pred_rgb, pred_alpha, gt_rgb, gt_alpha, flat,
  gauss): the first four are (V,H,W,3)/(V,H,W,1), `flat` is
  flatten_gaussians' output, `gauss` is the raw (L,C,H,W) per-layer dict
  (for validation visualizations)."""
  import gsplat

  src = item["source"]
  rgb = src["rgb"].to(device)
  xyz_cam = src["xyz_cam"].to(device)
  hit = src["hit"].to(device)
  fx = src["K_depth"][0, 0].to(device)

  gauss = model(rgb, xyz_cam, hit, fx)
  flat = flatten_gaussians(xyz_cam, hit, gauss)

  views = [src] + item["targets"]
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
  return pred_rgb.clamp(0.0, 1.0), pred_alpha.clamp(0.0, 1.0), gt_rgb, gt_alpha, flat, gauss


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


def compute_loss(model, item, cfg, device, window):
  pred_rgb, pred_alpha, gt_rgb, gt_alpha, flat, gauss = render_scene(
    model, item, cfg.model.max_sh_degree, device)
  per_view, parts = per_view_losses(pred_rgb, pred_alpha, gt_rgb, gt_alpha, window, cfg.loss)
  loss = per_view.mean()

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

  n_targets = per_view.shape[0] - 1
  metrics = {
    "loss": loss.detach(), "l1": parts["l1"].mean().detach(), "ssim": parts["ssim"].mean().detach(),
    "mask": parts["mask"].mean().detach(), "scale_reg": scale_reg.detach(), "color_reg": color_reg.detach(),
    "loss_source": per_view[0].detach(),
    "mean_opacity": flat["opacities"].mean().detach() if flat["opacities"].numel() else torch.zeros((), device=device),
    "mean_scale": flat["scales"].mean().detach() if flat["scales"].numel() else torch.zeros((), device=device),
    "frac_kept": torch.tensor(
      flat["means"].shape[0] / max(1, item["source"]["hit"].numel()), device=device),
  }
  if n_targets > 0:
    metrics["loss_targets_mean"] = per_view[1:].mean().detach()
  return loss, metrics, (pred_rgb, pred_alpha, gt_rgb, gt_alpha, gauss)


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
  list of (H,W,3) uint8 frames, or None if the source view seeded zero
  Gaussians (nothing to show)."""
  src = item["source"]
  rgb, xyz_cam, hit = src["rgb"].to(device), src["xyz_cam"].to(device), src["hit"].to(device)
  fx = src["K_depth"][0, 0].to(device)
  ih, iw = src["rgb"].shape[-2:]

  with torch.no_grad():
    gauss = model(rgb, xyz_cam, hit, fx)
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
  return [frames[i] for i in range(frames.shape[0])]


def log_orbit_video(wandb_run, step, tag, frames, fps, crf, workdir):
  import wandb
  path = os.path.join(workdir, f"{tag.replace('/', '_')}.mp4")
  write_mp4(frames, path, fps, crf)
  wandb_run.log({tag: wandb.Video(path, caption=tag, format="mp4")}, step=step)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def run_validation(model, val_ds, cfg, device, window, wandb_run, step):
  if len(val_ds) == 0:
    return {}
  model.eval()
  losses, losses_source, losses_targets = [], [], []
  with torch.no_grad():
    n = min(int(cfg.val.num_scenes), len(val_ds))
    for i in range(n):
      item = val_ds[i]
      _, metrics, extras = compute_loss(model, item, cfg, device, window)
      losses.append(metrics["loss"].item())
      losses_source.append(metrics["loss_source"].item())
      if "loss_targets_mean" in metrics:
        losses_targets.append(metrics["loss_targets_mean"].item())
      if i == 0 and wandb_run is not None:
        pred_rgb, _, gt_rgb, _, gauss = extras
        log_render_panel(wandb_run, step, "val", pred_rgb, gt_rgb, gauss)
  model.train()
  out = {
    "val/rec_loss": float(np.mean(losses)),
    "val/loss_source": float(np.mean(losses_source)),
  }
  if losses_targets:
    out["val/loss_targets_mean"] = float(np.mean(losses_targets))
  return out


@hydra.main(version_base=None, config_path="conf", config_name="train_gs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.train.seed))
  device = cfg.train.device if (cfg.train.device != "cuda" or torch.cuda.is_available()) else "cpu"
  if device != cfg.train.device:
    log.warning("cuda not available, falling back to cpu")

  if cfg.data.fixed_source_view is not None:
    # Single-batch overfit mode (fit_gsplat.py's primary/secondary terms
    # applied here): the exact same source+target views every step, no
    # resampling at all -- nothing to hold out, so val is empty.
    if cfg.data.fixed_target_views is None:
      raise SystemExit("data.fixed_target_views must be set when data.fixed_source_view is set")
    h5_path = cfg.data.h5_paths if isinstance(cfg.data.h5_paths, str) else cfg.data.h5_paths[0]
    train_ds = GSFixedViewsDataset(
      h5_path, source_view=cfg.data.fixed_source_view,
      target_views=list(cfg.data.fixed_target_views), num_layers=cfg.data.num_layers,
    )
    val_ds = _EmptyDataset()
  else:
    views_ds = GSViewsDataset(cfg.data.h5_paths, num_layers=cfg.data.num_layers)
    train_subset, val_subset = split_by_mesh(views_ds, val_fraction=cfg.data.val_fraction, seed=cfg.train.seed)
    train_ds = GSPairDataset(
      train_subset, num_target_views=cfg.data.num_target_views,
      seed=cfg.train.seed, deterministic_targets=False,
    )
    val_ds = GSPairDataset(
      val_subset, num_target_views=cfg.data.num_target_views,
      seed=cfg.train.seed, deterministic_targets=True,
    )
  if len(train_ds) == 0:
    raise SystemExit("train split is empty -- check data.h5_paths / data.val_fraction")

  from torch.utils.data import DataLoader
  train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=cfg.train.num_workers,
    collate_fn=lambda batch: batch[0], persistent_workers=cfg.train.num_workers > 0,
  )

  with timed("build_model"):
    model = GSModel(cfg).to(device)
  model.train()

  opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

  def lr_lambda(step):
    if step < cfg.train.warmup_steps:
      return (step + 1) / cfg.train.warmup_steps
    progress = (step - cfg.train.warmup_steps) / max(1, cfg.train.max_steps - cfg.train.warmup_steps)
    progress = min(progress, 1.0)
    min_ratio = cfg.train.min_lr / cfg.train.lr
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

  sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
  window = _gaussian_window(device=device)

  wandb_run = None
  if cfg.wandb.mode != "disabled":
    import wandb
    wandb_run = wandb.init(
      project=cfg.wandb.project, mode=cfg.wandb.mode, tags=list(cfg.wandb.tags),
      name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True),
    )

  ckpt_dir = cfg.checkpoint.dir or os.getcwd()
  os.makedirs(ckpt_dir, exist_ok=True)
  ckpt_path = os.path.join(ckpt_dir, "train_gs.ckpt.pt")

  log.info("train=%d val=%d views, device=%s, output=%s",
           len(train_ds), len(val_ds), device, ckpt_path)

  # Fixed scenes for orbit previews, picked once and reused every epoch so
  # progress is comparable over time (train_ds[idx] itself re-randomizes
  # which view is "source" on every access -- see GSPairDataset.__getitem__
  # -- so we deliberately snapshot one draw rather than re-indexing later).
  orbit_train_item = train_ds[0]
  orbit_val_item = val_ds[0] if len(val_ds) > 0 else None
  orbit_workdir = tempfile.mkdtemp(prefix="train_gs_orbit_")

  def maybe_render_orbits(epoch, step):
    if not cfg.orbit.enabled or epoch % max(1, int(cfg.orbit.every_n_epochs)) != 0:
      return
    model.eval()
    with timed(f"orbit_preview@epoch{epoch}"):
      for tag, item in (("train", orbit_train_item), ("val", orbit_val_item)):
        if item is None:
          continue
        frames = render_orbit(model, item, cfg, device)
        if frames is None:
          log.warning("orbit/%s: source view seeded zero Gaussians, skipping orbit", tag)
        elif wandb_run is not None:
          log_orbit_video(wandb_run, step, f"orbit/{tag}", frames, cfg.orbit.fps, cfg.orbit.crf, orbit_workdir)
        # Also log a static [GT | render | |diff|] panel for the source
        # (primary) view and every target (secondary) view of this same
        # fixed item, on the same cheap cadence -- covers the case with no
        # real val split (e.g. single-batch overfit mode), where
        # run_validation never fires.
        if wandb_run is not None:
          with torch.no_grad():
            _, _, extras = compute_loss(model, item, cfg, device, window)
          pred_rgb, _, gt_rgb, _, gauss = extras
          log_render_panel(wandb_run, step, tag, pred_rgb, gt_rgb, gauss)
    model.train()

  data_iter = iter(train_loader)
  step = 0
  epoch = 0
  with timed("train"):
    while step < int(cfg.train.max_steps):
      opt.zero_grad(set_to_none=True)
      accum_loss = 0.0
      accum_metrics = {}
      epoch_ended = False
      for _ in range(int(cfg.train.grad_accum_steps)):
        try:
          item = next(data_iter)
        except StopIteration:
          data_iter = iter(train_loader)
          item = next(data_iter)
          epoch_ended = True
        loss, metrics, _ = compute_loss(model, item, cfg, device, window)
        (loss / cfg.train.grad_accum_steps).backward()
        accum_loss += loss.item() / cfg.train.grad_accum_steps
        for k, v in metrics.items():
          accum_metrics[k] = accum_metrics.get(k, 0.0) + v.item() / cfg.train.grad_accum_steps

      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
      opt.step()
      sched.step()

      if epoch_ended:
        epoch += 1
        maybe_render_orbits(epoch, step)

      if step % cfg.train.log_every == 0:
        tgt_str = f"tgt {accum_metrics['loss_targets_mean']:.5f}" if "loss_targets_mean" in accum_metrics else "tgt n/a"
        log.info(
          "step %d/%d  loss %.5f  (l1 %.5f ssim %.5f mask %.5f)  src %.5f %s  "
          "kept %.3f  grad_norm %.3f",
          step, cfg.train.max_steps, accum_loss, accum_metrics["l1"], accum_metrics["ssim"],
          accum_metrics["mask"], accum_metrics["loss_source"], tgt_str,
          accum_metrics["frac_kept"], float(grad_norm),
        )
        if wandb_run is not None:
          log_dict = {
            "train/loss": accum_loss, "train/loss_l1": accum_metrics["l1"],
            "train/loss_ssim": accum_metrics["ssim"], "train/loss_mask": accum_metrics["mask"],
            "train/loss_scale_reg": accum_metrics["scale_reg"], "train/loss_color_reg": accum_metrics["color_reg"],
            "train/loss_source": accum_metrics["loss_source"],
            "train/mean_opacity": accum_metrics["mean_opacity"], "train/mean_scale": accum_metrics["mean_scale"],
            "train/frac_gaussians_kept": accum_metrics["frac_kept"],
            "train/grad_norm": float(grad_norm), "train/lr": sched.get_last_lr()[0],
            "train/epoch": epoch,
          }
          if "loss_targets_mean" in accum_metrics:
            log_dict["train/loss_targets_mean"] = accum_metrics["loss_targets_mean"]
          wandb_run.log(log_dict, step=step)

      if cfg.val.every > 0 and step > 0 and step % cfg.val.every == 0:
        with timed(f"validation@step{step}"):
          val_metrics = run_validation(model, val_ds, cfg, device, window, wandb_run, step)
        if val_metrics:
          log.info("step %d  validation: %s", step, val_metrics)
          if wandb_run is not None:
            wandb_run.log(val_metrics, step=step)

      if cfg.checkpoint.every > 0 and step > 0 and step % cfg.checkpoint.every == 0:
        torch.save({
          "model_state_dict": model.state_dict(), "optimizer": opt.state_dict(),
          "scheduler": sched.state_dict(), "step": step,
          "config_json": OmegaConf.to_container(cfg, resolve=True),
        }, ckpt_path)
        log.info("checkpoint -> %s", ckpt_path)

      step += 1

  torch.save({
    "model_state_dict": model.state_dict(), "optimizer": opt.state_dict(),
    "scheduler": sched.state_dict(), "step": step,
    "config_json": OmegaConf.to_container(cfg, resolve=True),
  }, ckpt_path)
  log.info("final checkpoint -> %s", ckpt_path)
  shutil.rmtree(orbit_workdir, ignore_errors=True)


if __name__ == "__main__":
  main()
