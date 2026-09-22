#!/usr/bin/env python3
"""Shared PyTorch Lightning building blocks for train_gs.py and fit_gsplat.py.

Both scripts optimize Gaussian-splat parameters against gsplat photometric
renders under a `pl.Trainer` (train_gs.py: a network's predicted Gaussians
over a dataset; fit_gsplat.py: one scene's own Gaussians, directly). That's
where the overlap actually is -- the loss/schedule/OOM-handling plumbing
below -- not the render loop or orbit-preview code itself, which differ per
script for reasons documented at their own call sites (Gaussian
representation, SH vs. flat colour, world- vs. source-camera frame, batched
vs. per-frame rasterization).
"""

import contextlib as ctl
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import Optimizer

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
