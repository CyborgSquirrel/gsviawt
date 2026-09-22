#!/usr/bin/env python3
"""Shared PyTorch Lightning building blocks for train_gs.py and fit_gsplat.py.

Both scripts optimize Gaussian-splat parameters against gsplat photometric
renders under a `pl.Trainer` (train_gs.py: a network's predicted Gaussians
over a dataset; fit_gsplat.py: one scene's own Gaussians, directly). That's
where the overlap actually is -- the loss/schedule/OOM-handling plumbing,
plus the PanelCallback/OrbitCallback pair below, which each PULL a
normalized snapshot from the LightningModule (get_preview_source(), one
implementation per script) and do the actual gsplat.rasterization()+
encode+wandb-log call once, here.

Both scripts' get_preview_source() returns Gaussians AND camera views in
true WORLD space (train_gs.py's Gaussians are natively predicted in the
source view's own camera frame -- see gs_dataset.py's module docstring --
but get_preview_source() transforms them into world space using that
view's own raw pose before returning, precisely so this module doesn't
need to know or care about that distinction). That's what makes
OrbitCallback fully shared below: one world-frame turntable formula
(WORLD_UP/look_at_c2w), no per-script pose method at all -- it only needs
a distance (each get_preview_source() includes its own "scene_scale") and
the already-world-space Gaussians.
"""

import contextlib as ctl
import logging
import os
import shutil
import tempfile

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from omegaconf import OmegaConf
from torch.optim import Optimizer

import wandb

log = logging.getLogger(__name__)

# Both scripts' conf/*.yaml lean on this for callback/trainer fields that are
# a small expression over another field rather than a literal (e.g.
# "${eval:'${iters} // 50'}") -- registered here, once, so importing either
# script (train_gs.py imports fit_gsplat.py imports this module) can't
# double-register it and raise.
if not OmegaConf.has_resolver("eval"):
  OmegaConf.register_new_resolver("eval", eval)

# Camera-local axis flip: Blender/OpenGL (X right, Y up, Z back) <-> OpenCV
# (X right, Y down, Z forward). Canonical copy -- fit_gsplat.py and
# gs_dataset.py each used to define their own identical constant.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
WORLD_UP = np.array([0.0, 0.0, 1.0], np.float32)  # Blender / render_objaverse is Z-up


def look_at_c2w(eye, target, up=WORLD_UP):
  """OpenGL camera-to-world (X right, Y up, -Z forward) looking from `eye`
  at `target`. Canonical copy -- fit_gsplat.py and orbit_video.py each used
  to define their own identical function."""
  z = eye - target
  z = z / (np.linalg.norm(z) + 1e-8)
  if abs(np.dot(z, up)) > 0.999:
    up = np.array([0.0, 1.0, 0.0], np.float32) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
  x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-8)
  y = np.cross(z, x)
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return c2w


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.SequentialLR):
  def __init__(
    self,
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
  ) -> None:
    self.total_steps = int(total_steps)
    self.warmup_steps = max(1, int(warmup_steps))
    self.min_lr = float(min_lr)

    warmup = torch.optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=1.0 / self.warmup_steps,
      end_factor=1.0,
      total_iters=self.warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=max(1, self.total_steps - self.warmup_steps),
      eta_min=self.min_lr,
    )
    super().__init__(
      optimizer,
      schedulers=[warmup, cosine],
      milestones=[self.warmup_steps],
    )


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def gaussian_window(size=11, sigma=1.5, device="cpu"):
  coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
  g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
  g = g / g.sum()
  return (g[:, None] * g[None, :])[None, None]


class DSSIMLoss(nn.Module):
  """1 - windowed SSIM, as an nn.Module loss (same shape as torch.nn.MSELoss
  etc.: `reduction` picked at construction, `forward(input, target)`
  computes it -- no separate window buffer for the caller to manage, this
  owns and device-places its own).

  x, y passed to forward: (N,C,H,W) in [0,1]. Standard 11x11 Gaussian-window
  SSIM, channels filtered independently.

  reduction:
    "mean" (default) -- scalar, averaged over every element. What
      fit_gsplat.py's single fixed-batch loss wants directly.
    "sum"  -- scalar, summed over every element.
    "none" -- per-pixel (N,C,H,W) map, unreduced -- e.g. train_gs.py's
      `_step` needs this to log/split per-source/target-view means itself
      before reducing to a scalar loss (flatten any extra leading dims into
      N first, same as any other elementwise torch loss with
      reduction="none")."""

  def __init__(self, window_size=11, sigma=1.5, reduction="mean"):
    super().__init__()
    if reduction not in ("mean", "sum", "none"):
      raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
    self.reduction = reduction
    self.register_buffer("window", gaussian_window(window_size, sigma), persistent=False)

  def forward(self, x, y):
    c = x.shape[1]
    w = self.window.expand(c, 1, -1, -1)
    pad = w.shape[-1] // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    dssim = 1.0 - ssim
    if self.reduction == "none":
      return dssim
    return dssim.mean() if self.reduction == "mean" else dssim.sum()


@ctl.contextmanager
def guarded_render(tag):
  """Context manager: runs the wrapped block, catching CUDA OOM so a single
  bad preview/validation render can't crash the whole training run. gsplat
  has no built-in memory-budget/OOM-catch mechanism of its own -- an
  outlier Gaussian scale (e.g. from an undertrained network early in
  training) can blow up isect_tiles' allocation regardless of what else is
  using the GPU (confirmed against gsplat's own GitHub issues #464/#487:
  same failure mode, same root cause -- excessive Gaussian volume -- no
  upstream fix). Only meant for preview/validation call sites (render_orbit,
  the orbit panel log, validation_step) where skipping one frame is safe
  and correct -- NOT in the main training step, where an OOM mid-step
  should still surface loudly rather than silently dropping an optimizer
  step. Callers pre-initialize their result variable to a fallback (e.g.
  None) BEFORE the `with` block and assign it from inside -- if a CUDA OOM
  fires, that assignment is simply never reached and the fallback stands."""
  try:
    yield
  except torch.OutOfMemoryError as e:
    log.warning("%s: CUDA OOM during render, skipping this frame -- %s", tag, e)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# preview_source / PanelCallback
#
# Both scripts periodically render their CURRENT Gaussians against a fixed
# handful of views and log a [GT | render | |diff|] comparison to wandb.
# What differs per script is how you get from "current state" to actual
# gsplat.rasterization()-ready Gaussian arrays: train_gs.py runs its model
# forward + flatten_gaussians on a batch item; fit_gsplat.py just activates
# its own live optimizer params (no model, no batch). What's identical once
# you have those arrays -- the render call, building the panel image, and
# logging it -- lives here.
#
# Pull, not push: PanelCallback below CALLS pl_module.get_preview_source(mode)
# itself, only when its own cadence (every_n_epochs XOR every_n_steps --
# see __init__) actually fires, instead of some other callback
# unconditionally stashing a value onto pl_module every epoch whether or
# not it's ever read. Each LightningModule (train_gs.py's GSLightningModule,
# fit_gsplat.py's GSFitLightningModule) implements that one method itself,
# taking mode ("epoch" | "step", matching which cadence fired) and
# returning
#   {"train": {"gauss": {...}, "scene_scale": float,
#              "views": {"viewmat","K","width","height","gt_rgb"}},
#    "val": same shape, or None}
# (gauss arrays already flattened/activated -- exactly what render() already
# needs in fit_gsplat.py, or one flatten_gaussians() yield in train_gs.py;
# gt_rgb: (V,C,H,W) in [0,1]; sh_degree: None or an int) -- cached keyed by
# (mode, epoch-or-step) (see either implementation) so a second caller at
# the same point doesn't redundantly redo the forward pass. No callback
# ordering to get right, no cost on ticks the panel isn't even logged.
#
# "val" is only ever populated in step mode (mode == "step"): epoch mode
# stays train-only, same as it always has been (on_train_epoch_end is the
# only epoch-cadence hook either callback implements -- there's no
# on_validation_epoch_end here, deliberately). Step mode has no equivalent
# "validation batch end" hook to split train/val across, so instead
# on_train_batch_end logs BOTH stages together in one wandb.log call
# whenever it fires and a val split exists (get_preview_source(mode)
# leaves "val" as None otherwise, or in epoch mode where nothing asks for
# it) -- see _step below on both callbacks.
# ---------------------------------------------------------------------------

def _build_panel(gt_rgb, pred_rgb):
  """gt_rgb/pred_rgb: (N,C,H,W) in [0,1]. Returns one uint8 (H, N*3*W, 3)
  image: for each view, [GT | render | |diff|] side by side."""
  n = gt_rgb.shape[0]
  rows = []
  for i in range(n):
    gc = gt_rgb[i].detach().cpu()
    pc = pred_rgb[i].detach().cpu()
    g = (gc * 255).to(torch.uint8)
    r = (pc * 255).to(torch.uint8)
    d = ((gc - pc).abs() * 255).to(torch.uint8)
    rows.append(pack([g, r, d], "c * w")[0])
  return pack(rows, "c h *")[0]


def _cadence_check(every_n_epochs, every_n_steps, cls_name):
  if (every_n_epochs is None) == (every_n_steps is None):
    raise ValueError(f"{cls_name}: pass exactly one of every_n_epochs/every_n_steps")


class PanelCallback(pl.Callback):
  """Shared consumer half of the preview_source hand-off -- see this
  module's own comment block above for the full contract. Pulls
  pl_module.get_preview_source(mode) and renders its Gaussians into its
  views in ONE gsplat.rasterization() call per stage, then logs a
  [GT | render | |diff|] panel per stage.

  every_n_epochs XOR every_n_steps picks which cadence this instance
  fires on (exactly one must be set) -- epoch cadence only ever previews
  "train" (on_train_epoch_end); step cadence previews "train" and, when a
  val split exists, "val" too, together in one wandb.log call
  (on_train_batch_end -- there's no per-step validation loop to hook a
  separate "val batch end" into)."""

  def __init__(self, *, every_n_epochs: int | None = None, every_n_steps: int | None = None):
    _cadence_check(every_n_epochs, every_n_steps, "PanelCallback")
    self.every_n_epochs = every_n_epochs
    self.every_n_steps = every_n_steps

  def on_train_epoch_end(self, trainer, pl_module):
    self._step("epoch", trainer, pl_module)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self._step("step", trainer, pl_module)

  def _step(self, mode, trainer, pl_module):
    """mode: "epoch" (from on_train_epoch_end) or "step" (from
    on_train_batch_end) -- a no-op unless this instance is actually
    configured for that mode (every_n_epochs/every_n_steps, see
    __init__)."""
    if mode == "epoch":
      if self.every_n_epochs is None:
        return
      # Count of COMPLETED epochs (Lightning hasn't bumped
      # trainer.current_epoch yet at this point in the hook) --
      # "every_n_epochs=1" means "every epoch", matching conf/*.yaml's
      # comment on that field elsewhere.
      n = trainer.current_epoch + 1
      if n % self.every_n_epochs != 0:
        return
      stages = ("train",)
    else:
      if self.every_n_steps is None:
        return
      n = trainer.global_step  # count of COMPLETED optimizer steps
      if n == 0 or n % self.every_n_steps != 0:
        return
      stages = ("train", "val")

    wandb_run = pl_module.logger.experiment if pl_module.logger is not None else None
    if wandb_run is None:
      return

    log_payload = {}
    with guarded_render(f"{mode}/panel"), torch.no_grad():
      import gsplat
      source = pl_module.get_preview_source(mode)
      for stage in stages:
        entry = source.get(stage)
        if entry is None:
          continue
        gauss, views = entry["gauss"], entry["views"]
        rgb, _, _ = gsplat.rasterization(
          means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
          opacities=gauss["opacities"], colors=gauss["colors"],
          viewmats=views["viewmat"], Ks=views["K"],
          width=int(views["width"]), height=int(views["height"]),
          sh_degree=gauss["sh_degree"], render_mode="RGB", packed=True,
        )
        pred_rgb = rearrange(rgb.clamp(0.0, 1.0), "v h w c -> v c h w")
        panel = _build_panel(views["gt_rgb"], pred_rgb)
        log_payload[f"{stage}/panel"] = wandb.Image(
          panel, caption="GT | render | |diff|, one row per view")
    if not log_payload:
      return

    log_payload["views_seen"] = pl_module.views_seen  # TODO: better way to handle this
    wandb_run.log(log_payload)


def write_mp4(frames, path, fps, crf):
  """H.264 .mp4 via imageio's ffmpeg backend (imageio-ffmpeg ships a static
  binary -- nothing needed on the system PATH). Canonical copy -- orbit_video.py
  and fit_gsplat.py both used to define their own identical version of this;
  they now import it from here."""
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


def _orbit_viewmats(dist, num_frames, elevation_deg):
  """N raw world-frame c2w poses on a circular orbit around the world
  origin at radius `dist` (a fixed capture distance, not an auto-fit
  bounding-sphere framing -- both scripts anchor to their own scene_scale
  for the same "avoid exposing per-Gaussian spacing as a moire pattern"
  reason, documented at each get_preview_source() implementation),
  converted to gsplat-ready (N,4,4) world-to-camera viewmats (OpenCV
  convention)."""
  elev = np.radians(float(elevation_deg))
  azimuths = np.linspace(0.0, 2 * np.pi, int(num_frames), endpoint=False)
  poses = np.stack([
    look_at_c2w(
      np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)], np.float32) * dist,
      np.zeros(3, np.float32),
    )
    for az in azimuths
  ])
  c2w_cv = poses @ OPENGL_TO_OPENCV
  return np.linalg.inv(c2w_cv).astype(np.float32)


class OrbitCallback(pl.Callback):
  """Shared consumer half of the same pull pattern PanelCallback uses.
  Pulls pl_module.get_preview_source(mode) -- same call PanelCallback
  makes, same cache -- and, since both scripts' Gaussians are already in
  world space there (see this module's own docstring), builds ONE
  world-frame turntable orbit per stage (_orbit_viewmats, radius = that
  stage's own "scene_scale") entirely in here. No per-script pose method
  needed at all. Renders all N orbit frames of a stage in ONE batched
  gsplat.rasterization() call, writes an mp4, and logs wandb.Video.

  every_n_epochs XOR every_n_steps picks which cadence this instance
  fires on -- see PanelCallback's own docstring, same contract exactly
  (epoch cadence == train only; step cadence == train + val together)."""

  def __init__(self, *, every_n_epochs: int | None = None, every_n_steps: int | None = None,
              num_frames: int = 24, fps: int = 12, crf: int = 28, elevation_deg: float = 20.0):
    _cadence_check(every_n_epochs, every_n_steps, "OrbitCallback")
    self.every_n_epochs = every_n_epochs
    self.every_n_steps = every_n_steps
    self.num_frames = num_frames
    self.fps = fps
    self.crf = crf
    self.elevation_deg = elevation_deg
    self.workdir = None

  def on_fit_start(self, trainer, pl_module):
    self.workdir = tempfile.mkdtemp(prefix="orbit_")

  def on_fit_end(self, trainer, pl_module):
    if self.workdir is not None:
      shutil.rmtree(self.workdir, ignore_errors=True)
      self.workdir = None

  def on_train_epoch_end(self, trainer, pl_module):
    self._step("epoch", trainer, pl_module)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self._step("step", trainer, pl_module)

  def _step(self, mode, trainer, pl_module):
    if mode == "epoch":
      if self.every_n_epochs is None:
        return
      n = trainer.current_epoch + 1  # count of COMPLETED epochs
      if n % self.every_n_epochs != 0:
        return
      stages = ("train",)
    else:
      if self.every_n_steps is None:
        return
      n = trainer.global_step  # count of COMPLETED optimizer steps
      if n == 0 or n % self.every_n_steps != 0:
        return
      stages = ("train", "val")

    wandb_run = pl_module.logger.experiment if pl_module.logger is not None else None
    if wandb_run is None:
      return

    log_payload = {}
    with guarded_render(f"{mode}/orbit"), torch.no_grad():
      import gsplat
      source = pl_module.get_preview_source(mode)
      for stage in stages:
        entry = source.get(stage)
        if entry is None:
          continue
        gauss, views = entry["gauss"], entry["views"]
        device = gauss["means"].device
        viewmats = torch.from_numpy(
          _orbit_viewmats(entry["scene_scale"], self.num_frames, self.elevation_deg)).to(device)
        Ks = repeat(views["K"][0], "... -> b ...", b=viewmats.shape[0])
        rgb, _, _ = gsplat.rasterization(
          means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
          opacities=gauss["opacities"], colors=gauss["colors"],
          viewmats=viewmats, Ks=Ks,
          width=int(views["width"]), height=int(views["height"]),
          sh_degree=gauss["sh_degree"], render_mode="RGB", packed=True,
        )
        rgb_np = (rgb.clamp(0.0, 1.0).cpu().numpy() * 255).astype("uint8")
        frames = [rgb_np[i] for i in range(rgb_np.shape[0])]
        path = os.path.join(self.workdir, f"{stage}_orbit_{mode}{n}.mp4")
        write_mp4(frames, path, self.fps, self.crf)
        log_payload[f"{stage}/orbit"] = wandb.Video(path, caption=f"{mode} {n}", format="mp4")
    if not log_payload:
      return

    log_payload["views_seen"] = pl_module.views_seen  # TODO: better way to handle this
    wandb_run.log(log_payload)
