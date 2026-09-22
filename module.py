#!/usr/bin/env python3
"""Shared PyTorch Lightning building blocks for train_gs.py and fit_gsplat.py.

Both scripts optimize Gaussian-splat parameters against gsplat photometric
renders under a `pl.Trainer` (train_gs.py: a network's predicted Gaussians
over a dataset; fit_gsplat.py: one scene's own Gaussians, directly). That's
where the overlap actually is -- the loss/schedule/OOM-handling plumbing,
plus the PanelCallback/OrbitCallback pair below, which each PULL a
normalized {"gauss", ...} snapshot from the LightningModule (via
get_preview_source()/get_orbit_source(), one implementation per script) and
do the actual gsplat.rasterization()+encode+wandb-log call once, here.
What's NOT shared -- deliberately -- is how each script gets from "current
state" to that normalized snapshot: Gaussian representation, SH vs. flat
colour, world- vs. source-camera frame, and (for orbit) camera-distance
policy all differ per script for reasons documented at each
get_preview_source()/get_orbit_source() implementation.
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
from einops import pack, rearrange
from omegaconf import OmegaConf
from torch.optim import Optimizer

import wandb

log = logging.getLogger(__name__)

# Both scripts' conf/*.yaml lean on this for trainer.* fields that are a
# small expression over another field rather than a literal (e.g.
# "${eval:'max(1, ${val_every})'}") -- registered here, once, so importing
# either script (train_gs.py imports fit_gsplat.py imports this module)
# can't double-register it and raise.
if not OmegaConf.has_resolver("eval"):
  OmegaConf.register_new_resolver("eval", eval)


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
# Pull, not push: PanelCallback below CALLS pl_module.get_preview_source()
# itself, only when its own every_n_epochs cadence actually fires, instead
# of some other callback unconditionally stashing a value onto pl_module
# every epoch whether or not it's ever read. Each LightningModule
# (train_gs.py's GSLightningModule, fit_gsplat.py's GSFitLightningModule)
# implements that one method itself, returning
#   {"gauss": {"means","quats","scales","opacities","colors","sh_degree"},
#    "views": {"viewmat","K","width","height","gt_rgb"}}
# (gauss arrays already flattened/activated -- exactly what render() already
# needs in fit_gsplat.py, or one flatten_gaussians() yield in train_gs.py;
# gt_rgb: (V,C,H,W) in [0,1]; sh_degree: None or an int) -- and caching it
# per-epoch (see either implementation) so a second caller in the same
# epoch (there isn't one yet, but nothing here assumes there won't be)
# doesn't redundantly redo the forward pass. No callback ordering to get
# right, no per-epoch cost on epochs the panel isn't even logged.
#
# Train-stage only for now: on_validation_epoch_end is commented out below
# rather than implemented, since it's not yet clear what fit_gsplat.py
# would put in preview_source for "val" that's comparable to train_gs.py's
# val split -- revisit once that's settled instead of guessing now.
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


class PanelCallback(pl.Callback):
  """Shared consumer half of the preview_source hand-off -- see this
  module's own comment block above for the full contract. Pulls
  pl_module.get_preview_source() and renders its Gaussians into its views
  in ONE gsplat.rasterization() call, then logs a [GT | render | |diff|]
  panel."""

  def __init__(self, *, every_n_epochs: int):
    self.every_n_epochs = every_n_epochs

  def on_train_epoch_end(self, trainer, pl_module):
    self._on_epoch_end("train", trainer, pl_module)

  # Deliberately not implemented -- see this module's comment block above.
  # def on_validation_epoch_end(self, trainer, pl_module):
  #   self._on_epoch_end("val", trainer, pl_module)

  def _on_epoch_end(self, stage, trainer, pl_module):
    # Count of COMPLETED epochs (Lightning hasn't bumped trainer.current_epoch
    # yet at this point in the hook) -- "every_n_epochs=1" means "every
    # epoch", matching conf/*.yaml's comment on that field elsewhere.
    epoch = trainer.current_epoch + 1
    if epoch % self.every_n_epochs != 0:
      return
    wandb_run = pl_module.logger.experiment if pl_module.logger is not None else None
    if wandb_run is None:
      return

    panel = None
    with guarded_render(f"{stage}/panel"), torch.no_grad():
      import gsplat
      source = pl_module.get_preview_source()
      gauss, views = source["gauss"], source["views"]
      rgb, _, _ = gsplat.rasterization(
        means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
        opacities=gauss["opacities"], colors=gauss["colors"],
        viewmats=views["viewmat"], Ks=views["K"],
        width=int(views["width"]), height=int(views["height"]),
        sh_degree=gauss["sh_degree"], render_mode="RGB", packed=True,
      )
      pred_rgb = rearrange(rgb.clamp(0.0, 1.0), "v h w c -> v c h w")
      panel = _build_panel(views["gt_rgb"], pred_rgb)
    if panel is None:
      return

    wandb_run.log({
      f"{stage}/panel": wandb.Image(
        panel, caption="GT | render | |diff|, one row per view"),
    })


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


class OrbitCallback(pl.Callback):
  """Shared consumer half of the same pull pattern PanelCallback uses.
  Pulls pl_module.get_orbit_source(num_frames, elevation_deg) --
  {"gauss": {means,quats,scales,opacities,colors,sh_degree}, "viewmats":
  (N,4,4), "K": (3,3), "width", "height"} -- expands K to (N,3,3), renders
  all N orbit frames in ONE batched gsplat.rasterization() call, writes an
  mp4, and logs wandb.Video.

  Train-stage only for now -- see PanelCallback's own note on why val is
  deferred; same reasoning applies here.

  Camera-pose generation (world-frame vs. source-camera-frame, fixed-at-
  capture-distance vs. auto-fit) stays entirely inside each script's own
  get_orbit_source() -- see train_gs.py's / fit_gsplat.py's own
  implementations for why those genuinely differ and aren't unified here."""

  def __init__(self, *, every_n_epochs: int, num_frames: int = 24,
              fps: int = 12, crf: int = 28, elevation_deg: float = 20.0):
    self.every_n_epochs = every_n_epochs
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
    self._on_epoch_end("train", trainer, pl_module)

  # Deliberately not implemented -- see PanelCallback.
  # def on_validation_epoch_end(self, trainer, pl_module):
  #   self._on_epoch_end("val", trainer, pl_module)

  def _on_epoch_end(self, stage, trainer, pl_module):
    epoch = trainer.current_epoch + 1
    if epoch % self.every_n_epochs != 0:
      return
    wandb_run = pl_module.logger.experiment if pl_module.logger is not None else None
    if wandb_run is None:
      return

    frames = None
    with guarded_render(f"{stage}/orbit"), torch.no_grad():
      import gsplat
      source = pl_module.get_orbit_source(self.num_frames, self.elevation_deg)
      gauss, viewmats = source["gauss"], source["viewmats"]
      Ks = source["K"][None].expand(viewmats.shape[0], -1, -1)
      rgb, _, _ = gsplat.rasterization(
        means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
        opacities=gauss["opacities"], colors=gauss["colors"],
        viewmats=viewmats, Ks=Ks,
        width=int(source["width"]), height=int(source["height"]),
        sh_degree=gauss["sh_degree"], render_mode="RGB", packed=True,
      )
      rgb_np = (rgb.clamp(0.0, 1.0).cpu().numpy() * 255).astype("uint8")
      frames = [rgb_np[i] for i in range(rgb_np.shape[0])]
    if frames is None:
      return

    path = os.path.join(self.workdir, f"{stage}_orbit_{trainer.current_epoch}.mp4")
    write_mp4(frames, path, self.fps, self.crf)
    wandb_run.log({
      f"{stage}/orbit": wandb.Video(path, caption=f"epoch {epoch}", format="mp4"),
    })
