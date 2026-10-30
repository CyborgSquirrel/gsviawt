#!/usr/bin/env python3
"""Shared PyTorch Lightning building blocks for train_gs.py and fit_3dgs.py.

Both scripts optimize Gaussian-splat parameters against gsplat photometric
renders under a `pl.Trainer` (train_gs.py: a network's predicted Gaussians
over a dataset; fit_3dgs.py: one scene's own Gaussians, directly). That's
where the overlap actually is -- the loss/schedule/OOM-handling plumbing,
plus the PanelCallback/OrbitCallback pair below, which each PULL a
normalized snapshot from the LightningModule (preview_entries(), one
implementation per script) and do the actual gsplat.rasterization()+
encode+wandb-log call once, here.

Both scripts' preview_entries() yield Gaussians AND camera views in
true WORLD space (train_gs.py's Gaussians are natively predicted in the
source view's own camera frame -- see gs_dataset.py's module docstring --
but preview_entries() transforms them into world space using that
view's own raw pose before returning, precisely so this module doesn't
need to know or care about that distinction). That's what makes
OrbitCallback fully shared below: one world-frame turntable formula
(WORLD_UP/look_at_c2w), no per-script pose method at all -- it only needs
a distance (each entry includes its own "scene_scale") and
the already-world-space Gaussians.

Also holds the depth-peel Gaussian-seeding helpers (SH_C0/init_gaussians/
make_viewmats/_scatter_grid) that used to live in fit_gsplat.py, now used
by fit_3dgs.py's layered mode, orbit_video.py and gs_decoder.py.
"""

import collections
import contextlib as ctl
import contextvars
import inspect
import logging
import math
import os
import shutil
import tempfile
import time

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import call
from omegaconf import OmegaConf
from torch.optim import Optimizer

import wandb

from debug_pointcloud import unproject_depth_peel

log = logging.getLogger(__name__)

# Both scripts' conf/*.yaml lean on this for callback/trainer fields that are
# a small expression over another field rather than a literal (e.g.
# "${eval:'${iters} // 50'}") -- registered here, once, so importing either
# script (both import this module, directly or via gs_decoder.py) can't
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


# ---------------------------------------------------------------------------
# depth-peel Gaussian seeding -- shared by fit_3dgs.py's layered mode,
# orbit_video.py and gs_decoder.py (formerly fit_gsplat.py's own, moved here
# when that script was deleted)
# ---------------------------------------------------------------------------

SH_C0 = 0.28209479177387814  # SH band-0 constant, for RGB <-> sh0 in the .ply


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

  # isotropic initial scale = RMS distance to the knn_k nearest neighbours
  # (sqrt(mean(d^2)), not mean(d) -- matches simple_knn's distCUDA2 exactly,
  # same formula fit_3dgs.py's own init_gaussians() uses for "set" mode;
  # this used to be a plain mean here, a leftover from fit_gsplat.py's own
  # less careful version, before this function moved into module.py).
  from scipy.spatial import cKDTree
  k = min(knn_k + 1, len(pts))
  dist, _ = cKDTree(pts).query(pts, k=k)                    # (P, k), or (P,) when k == 1
  dist = rearrange(dist, "p -> p 1") if dist.ndim == 1 else dist
  neighbours = dist[:, 1:] if k > 1 else dist               # column 0 is the point itself
  nn = np.sqrt(np.clip((neighbours ** 2).mean(axis=1), 1e-7, None)).astype(np.float32)

  return {
    "means": pts.astype(np.float32),
    "scales_log": repeat(np.log(nn), "p -> p xyz", xyz=3).astype(np.float32),
    "quats": np.tile([1.0, 0.0, 0.0, 0.0], (len(pts), 1)).astype(np.float32),
    "opac_logit": np.full(len(pts), _logit(np.float32(init_opacity)), np.float32),
    "colors": colors.astype(np.float32),  # [0,1] RGB, not logit -- fit_3dgs.py converts
                                           # straight to SH0 (RGB2SH), no flat-colour model
                                           # here to justify a logit representation at all
    "u": u.astype(np.int64), "v": v.astype(np.int64), "layer": layer.astype(np.int64),
  }


def make_viewmats(poses):
  """poses: (V,4,4) camera-to-world, OpenGL cam axes. Returns (V,4,4)
  world-to-camera in OpenCV convention -- what gsplat wants."""
  c2w_cv = poses @ OPENGL_TO_OPENCV
  return np.linalg.inv(c2w_cv).astype(np.float32)


def _scatter_grid(flat, v, u, layer, H, W, L):
  """flat: (G,) or (G,C). Returns (H,W,L) or (H,W,L,C) float32 with NaN in
  every slot that has no Gaussian."""
  shape = (H, W, L) if flat.ndim == 1 else (H, W, L, flat.shape[1])
  grid = np.full(shape, np.nan, np.float32)
  grid[v, u, layer] = flat
  return grid


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


def time_wsd_progress(seconds, max_seconds, warmup_fraction, decay_fraction):
  """Position in [0, 1] of a warmup-stable-decay schedule over max_seconds: linear 0 -> 1 during
  the first warmup_fraction, 1 until the last decay_fraction, then a cosine 1 -> 0."""
  warmup_end = warmup_fraction * max_seconds
  decay_start = (1.0 - decay_fraction) * max_seconds
  if seconds < warmup_end:
    return seconds / warmup_end
  if seconds < decay_start:
    return 1.0
  if seconds >= max_seconds:
    return 0.0
  return 0.5 * (1.0 + math.cos(math.pi * (seconds - decay_start) / (max_seconds - decay_start)))


class TrainClock(pl.Callback):
  """Counts the TRAINING time spent, for time-driven LR schedules (TimeWSD) and a hard stop.

  The clock counts the optimization pass of each batch: the forward pass, the backward pass and the
  optimizer step (plus the LR scheduler step). It does not count the wait for the batch (about 5% of a
  batch with the cached WT features), validation, checkpoint saves, or any callback. Lightning calls,
  per batch: [next(data) ->] on_train_batch_start -> training_step, backward, optimizer step -> LR
  scheduler step -> on_train_batch_end; validation runs after the last batch of an epoch. So this
  callback must be FIRST in the trainer's callback list (its on_train_batch_end runs before the other
  callbacks' ones and stops the clock) and its tail() LAST of ours (its on_train_batch_start runs after
  the other callbacks' ones and starts the clock); on_fit_start checks that the order is right.

  With max_seconds set, the trainer is asked to stop once that much training time is spent; Lightning
  then still runs its validation, and final.ckpt is written to the trainer's default_root_dir. The
  elapsed seconds are saved in checkpoints, so a resumed run continues the clock."""

  def __init__(self, max_seconds=None):
    self.max_seconds = None if max_seconds is None else float(max_seconds)
    self._elapsed = 0.0
    self._start = None   # when the current batch's optimization pass began (None: between batches)
    self._stopped = False

  def seconds(self):
    """Training seconds so far, including the batch in progress."""
    return self._elapsed + (0.0 if self._start is None else time.monotonic() - self._start)

  def tail(self):
    return TrainClockTail(self)

  def on_fit_start(self, trainer, pl_module):
    callbacks = trainer.callbacks
    after = callbacks[callbacks.index(self) + 1:]
    tails = [c for c in after if isinstance(c, TrainClockTail) and c.clock is self]
    if callbacks[0] is not self or len(tails) != 1:
      raise RuntimeError("TrainClock must be the first callback and its tail() must come after it, after the other callbacks")

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    # The batch's GPU work is queued asynchronously: without this the CPU only waits for it later (in the
    # progress bar / logger reading the loss), after the clock has stopped, and the clock misses most of it
    if pl_module.device.type == "cuda":
      torch.cuda.synchronize(pl_module.device)
    self._elapsed = self.seconds()
    self._start = None
    if self.max_seconds is not None and self._elapsed >= self.max_seconds:
      self._stopped = True
      trainer.should_stop = True

  def on_train_end(self, trainer, pl_module):
    if self._stopped:
      trainer.save_checkpoint(os.path.join(trainer.default_root_dir, "final.ckpt"))

  def state_dict(self):
    return {"elapsed": self._elapsed}

  def load_state_dict(self, state):
    self._elapsed = state["elapsed"]


class TrainClockTail(pl.Callback):
  """Starts the TrainClock for a batch once every other callback's on_train_batch_start has run --
  see TrainClock."""

  def __init__(self, clock):
    self.clock = clock

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    self.clock._start = time.monotonic()


class Trigger:
  """When a consumer callback acts, as one config value (`trigger_event`):
    - a str: the name of an event; the trigger is due on the steps where that event fired (see Events:
      some Events source, or a scheduler's trough_event, fires it). It registers itself as a consumer of
      the name with the current Events (Events.current), which must be current before it is built;
    - a mapping with exactly one of every_n_steps / every_n_epochs / every_n_seconds: a cadence of this
      consumer's own, no event needed. every_n_steps counts completed optimizer steps, every_n_epochs
      completed epochs (due on the last batch of one), every_n_seconds TRAINING seconds on the
      TrainClock, which must then be among the trainer's callbacks: due when the clock passes another
      multiple of it (the first call only notes where the clock is, so a resumed run doesn't fire at once);
    - a list of any of these: due when any of them is due (e.g. [trough, periodic]);
    - None: never due.
  due() is meant to be called once per batch, from on_train_batch_end."""

  CADENCES = ("every_n_steps", "every_n_epochs", "every_n_seconds")

  def __init__(self, spec):
    self.event = None
    self.events = None
    self.cadence = None
    self.any_of = ()
    self._seconds_index = None
    if spec is None:
      return
    if isinstance(spec, (list, tuple)) or OmegaConf.is_list(spec):
      self.any_of = tuple(Trigger(s) for s in spec)
      return
    if isinstance(spec, str):
      self.event = spec
      self.events = Events.current()
      self.events.consumes(spec)
      return
    given = {k: v for k, v in dict(spec).items() if v is not None}
    unknown = set(dict(spec)) - set(self.CADENCES)
    if unknown or len(given) != 1:
      raise ValueError(f"an event name, or a mapping with exactly one of {self.CADENCES} (and nothing else), got {dict(spec)}")
    (key, value), = given.items()
    if value <= 0:
      raise ValueError(f"{key} must be positive, got {value}")
    self.cadence = (key, value)

  def due(self, trainer, pl_module, batch_idx):
    if self.any_of:
      return any([t.due(trainer, pl_module, batch_idx) for t in self.any_of])   # all of them: each one keeps its own state
    if self.event is not None:
      return self.events.fired(self.event)
    if self.cadence is None:
      return False
    key, every = self.cadence
    if key == "every_n_steps":
      n = _completed_steps(trainer, pl_module)
      return n != 0 and n % every == 0
    if key == "every_n_epochs":
      return batch_idx + 1 == trainer.num_training_batches and (trainer.current_epoch + 1) % every == 0
    clock = next((c for c in trainer.callbacks if isinstance(c, TrainClock)), None)
    if clock is None:
      raise RuntimeError("every_n_seconds needs a TrainClock in the trainer's callbacks")
    index = int(clock.seconds() // every)
    previous, self._seconds_index = self._seconds_index, index
    return previous is not None and index > previous


class Events(pl.Callback):
  """The events that happened in the current training step, by name. The names are fired by `sources` (a
  mapping name -> a cadence mapping like Trigger's, e.g. {orbit: {every_n_seconds: 1800}}) in this
  callback's own on_train_batch_end, and by a scheduler (TimeWSD's trough_event) while it steps; consumers
  (Panel/ScaleLayers/OrbitCallback, EventCheckpoint, EventValidation, ... -- anything that takes a Trigger)
  read them with fired(name) in their own on_train_batch_end. The set is cleared at the start of every
  batch.

  The run's Events is made current first thing (Events(...).make_current(), a context variable), before
  anything that names an event is built: the producers (the sources, TimeWSD) call produces(name) and the
  consumers (Trigger) consumes(name) on it as they are built, and on_train_start fails when a
  consumed name has no producer. It must also be in the trainer's callbacks, right after the TrainClock
  (if there is one) and before every consumer."""

  _current = contextvars.ContextVar("events", default=None)

  @classmethod
  def current(cls):
    """The run's Events."""
    events = cls._current.get()
    if events is None:
      raise RuntimeError("no Events is current -- make one and call make_current() on it before building "
                         "the callbacks and the scheduler")
    return events

  def make_current(self):
    self._current.set(self)

  def __init__(self, sources=None):
    self.produced = set()
    self.consumed = set()
    self.sources = {name: Trigger(spec) for name, spec in (sources or {}).items()}
    for name, trigger in self.sources.items():
      if trigger.event is not None:
        raise ValueError(f"events.{name}: a source is a cadence mapping, not another event's name")
      self.produces(name)
    self._fired = set()

  def produces(self, name):
    self.produced.add(name)

  def consumes(self, name):
    self.consumed.add(name)

  def fire(self, name):
    self._fired.add(name)

  def fired(self, name):
    return name in self._fired

  def on_train_start(self, trainer, pl_module):
    missing = self.consumed - self.produced
    if missing:
      raise ValueError(f"event names nothing produces: {sorted(missing)}; produced: {sorted(self.produced)}")

  def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    self._fired.clear()

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    for name, trigger in self.sources.items():
      if trigger.due(trainer, pl_module, batch_idx):
        self.fire(name)


class EventCheckpoint(pl.Callback):
  """Saves a checkpoint in the trainer's default_root_dir whenever its trigger is due (a Trigger: an event
  name, or a cadence of its own). filename is a str.format() template over {count} (how many this
  callback has saved, this one included; kept in checkpoints, so a resumed run goes on counting) and {step}
  (the completed optimizer steps). trigger_event=None turns it off. (The TrainClock writes final.ckpt
  itself when the budget is spent.)"""

  def __init__(self, *, trigger_event, filename="checkpoint-{count}-step{step}.ckpt"):
    self.trigger = Trigger(trigger_event)
    self.filename = filename
    self.count = 0

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if not self.trigger.due(trainer, pl_module, batch_idx):
      return
    self.count += 1
    name = self.filename.format(count=self.count, step=_completed_steps(trainer, pl_module))
    trainer.save_checkpoint(os.path.join(trainer.default_root_dir, name))

  def state_dict(self):
    return {"count": self.count}

  def load_state_dict(self, state):
    self.count = state["count"]


class EventValidation(pl.Callback):
  """Runs the validation loop right after a training step whenever its trigger is due (a Trigger: an event
  name, or a cadence of its own), which Lightning itself can't do: it only validates on its own interval
  (trainer.val_check_interval / check_val_every_n_epoch) or when asked to stop. This does what the
  training epoch loop's on_advance_end does for a due validation (lightning/pytorch/loops/
  training_epoch_loop.py), with its private state, and then tells that loop to skip its own check for
  this step, so a step is never validated twice. trigger_event=None turns it off.
  The TrainClock doesn't count the validation (it counts the optimization pass of a batch only).

  Several of these can run side by side, a small one often and a big one rarely: each has a `name` (the
  datamodule's `validation` while it runs, so the metrics are logged as <set>/<name>/<metric>) and its own
  `volumes`, how many volumes of EACH validation set it scores: a whole number (200, or the string "200") is
  that many volumes, a percentage string ("50%") that proportion of the set, None the whole set (decimals
  are rejected as a count: floats and ints get mixed up in configs). It takes batches from the start of each set's dataloader
  until it has that many (the val dataloaders shuffle, so every validation scores a fresh random sample); a
  set with fewer volumes is scored whole (trainer.limit_val_batches doesn't apply to these). Lightning's own
  validations (at the end of training, ...) are sized by trainer.limit_val_batches instead.

  ema: run the validation with the LightningModule's EMA weights (`pl_module.ema`, diffusion mode); give it its own
  name, e.g. big_ema, so its metrics sit next to the raw weights' (<set>/big/..., <set>/big_ema/...)."""

  def __init__(self, *, name, trigger_event, volumes=None, at_end=False, ema=False):
    self.name = name
    self.ema = ema   # validate with the module's EMA weights (diffusion mode)
    self.trigger = Trigger(trigger_event)
    self.count, self.fraction = self._parse_volumes(volumes)
    self.at_end = at_end

  @staticmethod
  def _parse_volumes(volumes):
    """volumes -> (count, fraction), one of them None (both None: the whole set): a whole number (int or digit
    string) is a count, a "p%" string with 0 < p <= 100 a proportion; anything else raises."""
    if volumes is None:
      return None, None
    if isinstance(volumes, bool) or isinstance(volumes, float):
      raise ValueError(f"volumes={volumes!r}: use a whole number (a count) or a percentage string like \"50%\"")
    if isinstance(volumes, int):
      if volumes <= 0:
        raise ValueError(f"volumes={volumes!r} must be positive")
      return volumes, None
    if isinstance(volumes, str):
      text = volumes.strip()
      if text.isdigit() and int(text) > 0:
        return int(text), None
      if text.endswith("%"):
        try:
          percent = float(text[:-1])
        except ValueError:
          percent = None
        if percent is not None and 0 < percent <= 100:
          return None, percent / 100
    raise ValueError(f"volumes={volumes!r} must be a positive whole number, a percentage string like \"50%\" (0 < p <= 100) or None")

  def _batches(self, loader):
    """How many batches of `loader` to run for `volumes` (trainer.limit_val_batches is for Lightning's own)."""
    n = len(loader)
    if self.count is None and self.fraction is None:
      return n
    want = math.ceil(self.count / loader.batch_size) if self.count is not None else math.ceil(self.fraction * n)
    return min(want, n)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    due = self.trigger.due(trainer, pl_module, batch_idx)   # every step, for its own bookkeeping
    if trainer.should_stop:   # the last step (TrainClock's max_seconds, ...)
      due = self.at_end
    if not due or (self.ema and getattr(pl_module, "ema", None) is None):   # an EMA validation of a model without an EMA: nothing to run
      return
    loop = trainer.fit_loop.epoch_loop
    val_loop = loop.val_loop
    val_loop.setup_data()   # makes the loaders and their batch limits the first time (a no-op after)
    limits = val_loop._max_batches
    val_loop._max_batches = [self._batches(dl) for dl in val_loop._combined_loader.flattened]
    datamodule = trainer.datamodule
    datamodule.validation = self.name
    trainer.validating = True   # picks the right trainer._active_loop
    first_loop_iter = trainer._logger_connector._first_loop_iter
    try:
      if not loop._should_accumulate():
        call._call_lightning_module_hook(trainer, "on_validation_model_zero_grad")
      with (pl_module.ema.applied(pl_module.model) if self.ema else ctl.nullcontext()):
        val_loop.run()
    finally:
      val_loop._max_batches = limits   # Lightning's own validations use trainer.limit_val_batches
      datamodule.validation = "lightning"
    trainer.training = True
    trainer._logger_connector._first_loop_iter = first_loop_iter
    # A one-shot flag of Lightning's epoch loop (loops/training_epoch_loop.py): its on_advance_end, which runs
    # after every batch, computes whether Lightning's own check wants a validation (the time interval, the end
    # of an epoch, a stop request), then, if this flag is set, forces that to False and clears the flag, whether
    # or not a validation was due. We are in on_train_batch_end, inside the same step's advance(), so it is
    # consumed in this very step: it suppresses Lightning's own validation of this step only (never a later
    # one), and the validation above is the step's only one. Running the val loop resets Lightning's
    # _last_val_time (evaluation_loop.py), so a time interval doesn't fire again right after.
    loop._skip_next_val = True


class ValidationStats(pl.Callback):
  """Logs, for every validation set of every validation (EventValidation's and Lightning's own), how many
  volumes it scored (<set>/<name>/n_volumes) and the standard error of its mean loss
  (<set>/<name>/loss_se, from the batch losses, which are independent random samples of the set when
  the validation scores part of it), so a logged loss comes with its sample size and its noise."""

  def on_validation_epoch_start(self, trainer, pl_module):
    self._losses = collections.defaultdict(list)
    self._volumes = collections.Counter()

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
    self._losses[dataloader_idx].append(float(outputs["loss"]))
    self._volumes[dataloader_idx] += int(batch["batch_size"])

  def on_validation_epoch_end(self, trainer, pl_module):
    datamodule = trainer.datamodule
    for idx, name in enumerate(datamodule.val_datasets):
      losses = self._losses.get(idx)
      if not losses:
        continue
      prefix = f"{name}/{datamodule.validation}"
      pl_module.log(f"{prefix}/n_volumes", float(self._volumes[idx]), add_dataloader_idx=False)
      if len(losses) > 1:
        pl_module.log(f"{prefix}/loss_se", float(np.std(losses, ddof=1) / len(losses) ** 0.5), add_dataloader_idx=False)


class TimeWSD(torch.optim.lr_scheduler.LRScheduler):
  """Warmup-stable-decay LR driven by a TrainClock instead of a step count. Over the clock's span
  [start_seconds, max_seconds] the LR goes min_lr -> peak over the first warmup_fraction of the span,
  stays at the peak, and follows a cosine back down to min_lr over the last decay_fraction (peak = each
  param group's LR when the scheduler is made); before start_seconds it is min_lr. With cycle_seconds the
  span is cut into cycles of that length (it must divide the span), each a full warmup -> plateau -> decay,
  so the LR returns to min_lr and warms up again at every cycle boundary. start_seconds=0 is a
  schedule over the whole run; a run resumed from a decayed checkpoint can warm up again by setting it to
  the clock's value there. trough_event names an event of the current Events (Events.current) fired on the
  step where a cycle ends, when the LR is back at min_lr. Step it once per optimizer step like any scheduler; the LR it sets depends on
  the clock's time at that moment. Pair it with TrainClock(max_seconds) to stop the run at the end of
  the decay (train_gs.py does).

  The schedule's parameters are saved with the scheduler's state, and loading a state saved with
  different ones raises: resume with reset_scheduler=true (train_gs.py) to take the config's instead."""

  PARAMS = ("max_seconds", "start_seconds", "cycle_seconds", "warmup_fraction", "decay_fraction", "min_lr")

  def __init__(self, optimizer, clock, max_seconds, start_seconds=0.0, cycle_seconds=None, warmup_fraction=0.05, decay_fraction=0.1, min_lr=1e-6,
               trough_event=None):
    self.clock = clock
    self.max_seconds = float(max_seconds)
    self.start_seconds = float(start_seconds)
    if not 0.0 <= self.start_seconds < self.max_seconds:
      raise ValueError(f"start_seconds={start_seconds} must be in [0, max_seconds={max_seconds})")
    self.cycle_seconds = None if cycle_seconds is None else float(cycle_seconds)
    if self.cycle_seconds is not None:
      cycles = (self.max_seconds - self.start_seconds) / self.cycle_seconds
      if self.cycle_seconds <= 0 or abs(cycles - round(cycles)) > 1e-9:
        raise ValueError(f"cycle_seconds={cycle_seconds} must divide max_seconds - start_seconds = {max_seconds - start_seconds}")
    self.warmup_fraction = float(warmup_fraction)
    self.decay_fraction = float(decay_fraction)
    self.min_lr = float(min_lr)
    self.trough_event = trough_event
    self.events = None
    if trough_event is not None:
      self.events = Events.current()
      self.events.produces(trough_event)
    self._cycle = None   # which cycle the clock is in, to see it cross a trough
    super().__init__(optimizer)

  def step(self, epoch=None):
    super().step(epoch)
    if self.trough_event is None:
      return
    # the trough is the end of a cycle (the whole span without cycle_seconds): the first step the clock
    # reaches it, like the cadence of a Trigger -- the first call only notes the cycle, so a resumed run
    # doesn't fire at once
    length = self.cycle_seconds or (self.max_seconds - self.start_seconds)
    cycle = int(max(self.clock.seconds() - self.start_seconds, 0.0) // length)
    previous, self._cycle = self._cycle, cycle
    if previous is not None and cycle > previous:
      self.events.fire(self.trough_event)

  def get_lr(self):
    seconds = max(self.clock.seconds() - self.start_seconds, 0.0)
    span = self.max_seconds - self.start_seconds
    if self.cycle_seconds is not None and seconds < span:
      seconds, span = seconds % self.cycle_seconds, self.cycle_seconds
    progress = time_wsd_progress(seconds, span, self.warmup_fraction, self.decay_fraction)
    return [self.min_lr + (peak - self.min_lr) * progress for peak in self.base_lrs]

  def state_dict(self):
    # the clock and the events live in their own callbacks; the trough event is a config choice
    return {k: v for k, v in super().state_dict().items() if k not in ("clock", "events", "trough_event", "_cycle")}

  def load_state_dict(self, state_dict):
    differing = {k: (state_dict[k], getattr(self, k)) for k in self.PARAMS if k in state_dict and state_dict[k] != getattr(self, k)}
    if differing:
      raise ValueError(
        "the checkpoint's TimeWSD schedule differs from this run's config (saved, config): "
        f"{differing}; pass reset_scheduler=true to build the schedule from the config instead")
    super().load_state_dict(state_dict)


def needs_clock(scheduler_factory):
  """Whether a scheduler (class or functools.partial of one) takes a TrainClock."""
  return "clock" in inspect.signature(scheduler_factory).parameters


def make_scheduler(factory, optimizer, clock=None):
  """factory(optimizer), or factory(optimizer, clock=clock) for a scheduler that takes a TrainClock."""
  if not needs_clock(factory):
    return factory(optimizer)
  if clock is None:
    raise ValueError("this scheduler needs a TrainClock, but none was given")
  return factory(optimizer, clock=clock)


class WarmupLR(torch.optim.lr_scheduler.LinearLR):
  """Linear warmup only -- LR holds flat at the optimizer's base LR once
  warmup_steps is reached (LinearLR's own behavior past total_iters), no
  decay afterward. Same (optimizer, total_steps, warmup_steps, min_lr)
  signature as WarmupCosineAnnealingLR so `sched._target_=module.WarmupLR`
  alone swaps the scheduler on the CLI, no other sched.* override needed;
  total_steps/min_lr are accepted but unused (no decay phase to shape)."""

  def __init__(
    self,
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
  ) -> None:
    self.warmup_steps = max(1, int(warmup_steps))
    super().__init__(
      optimizer,
      start_factor=1.0 / self.warmup_steps,
      end_factor=1.0,
      total_iters=self.warmup_steps,
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


class ProgressWandbLogger(WandbLogger):
  """WandbLogger that stamps the progress metrics (views_seen, pixels_seen, ...
  -- the counters we want as an x-axis, whatever the row is about) onto every
  row it logs, so they sit in the same row as the learning rate, the losses,
  the panels and the videos.

  progress_fn: called at the start of every log_metrics(), returns a dict of
  progress metrics, which is overlaid onto the metrics passed in. None: no
  overlay, plain WandbLogger. If the overlay overwrites a metric that was
  passed in, that is warned about once -- the progress metrics are meant to
  come from progress_fn only.

  step_fn: called when log_metrics() is given no step, to get one (the
  trainer/global_step the row is logged with). Callers that pass a step keep
  theirs. None: a row logged without a step has no trainer/global_step."""

  def __init__(self, *args, progress_fn=None, step_fn=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.progress_fn = progress_fn
    self.step_fn = step_fn
    self._warned_overwrite = False

  def log_metrics(self, metrics, step=None):
    if step is None and self.step_fn is not None:
      step = self.step_fn()
    if self.progress_fn is not None:
      progress = self.progress_fn()
      overwritten = sorted(metrics.keys() & progress.keys())
      if overwritten and not self._warned_overwrite:
        self._warned_overwrite = True
        log.warning("progress_fn overwrote metrics passed to log_metrics(): %s "
                    "(warning once)", ", ".join(overwritten))
      metrics = {**metrics, **progress}
    super().log_metrics(metrics, step)


def _completed_steps(trainer, pl_module):
  """trainer.global_step, except for a LightningModule using MANUAL
  optimization (pl_module.automatic_optimization == False) that steps its
  own raw torch.optim.Optimizer objects directly rather than through
  Lightning's LightningOptimizer proxies (fit_3dgs.py does this
  deliberately -- gsplat's grow/prune/reset tensor surgery needs the real
  optimizers, not a proxy, and it manages several of them at once, which
  would inflate Lightning's own per-.step()-call counter anyway).
  trainer.global_step is driven ENTIRELY by hooks Lightning injects onto
  those proxies' own .step() method -- bypass the proxies and it silently
  stays 0 forever (confirmed directly: instrumented a run and watched it
  sit at 0 for 10/10 steps). A module built that way must expose its own
  accurate step_count instead (fit_3dgs.py already tracks one, for its own
  should_stop/LR-schedule/SH-degree-ramp bookkeeping) -- this reads that
  when present, trainer.global_step otherwise. fit_gsplat.py/train_gs.py
  both use plain automatic optimization and are unaffected either way."""
  if not pl_module.automatic_optimization:
    return int(pl_module.step_count)
  return trainer.global_step


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
# Pull, not push: PanelCallback below CALLS pl_module.preview_entries(stage)
# itself, only when its own trigger_event (see Trigger) actually fires,
# instead of some other callback
# unconditionally stashing a value onto pl_module every epoch whether or
# not it's ever read. Each LightningModule (train_gs.py's GSLightningModule,
# fit_gsplat.py's GSFitLightningModule) implements two methods itself:
#   preview_stages() -> the names of the stages that have preview objects:
#       "train", then each val set
#   preview_entries(stage, count=1, with_layers=False) -> a GENERATOR of up
#       to `count` entries of that stage, one at a time (distinct objects,
#       the first one always the same), each
#   {"gauss": {...}, "scene_scale": float,
#    "views": {"viewmat","K","width","height","gt_rgb"},
#    "mesh_index": int (optional), "gauss_layers": {...} (optional)}
# (gauss arrays already flattened/activated -- exactly what render() already
# needs in fit_gsplat.py, or one flatten_gaussians() yield in train_gs.py;
# gt_rgb: (V,C,H,W) in [0,1]; sh_degree: None or an int).
#
# Stateless: nothing is cached between calls. A consumer that keeps no entry
# beyond its own loop iteration never has more than one in memory, and one
# that asks again simply computes again (an entry costs about one forward
# pass) -- so there is no cache to size, key or release, and no callback
# ordering to get right.
#
# gauss_layers (optional, only with_layers=True, only on the first entry,
# only train_gs.py's GSLightningModule populates it): the PRE-flatten_gaussians
# per-layer decoder output -- {"scale": (L,3,H,W), "hit": (L,H,W) bool,
# ...the rest of the raw gauss dict} -- ScaleLayersCallback below reads this
# instead of "gauss" specifically so a single exploded pixel doesn't
# disappear into flatten_gaussians' mean/render.
#
# The callbacks act when their trigger is due on a batch end (on_train_batch_end) and preview every
# stage: there is no "validation batch end" hook to split train/val across, so ALL stages are logged
# together in one wandb.log call whenever it fires -- see _step on each callback.
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
  pl_module.preview_entries(stage) and renders its Gaussians into its
  views in ONE gsplat.rasterization() call per stage, then logs a
  [GT | render | |diff|] panel per stage.

  trigger_event (see Trigger: an event name, or a cadence of its own) says
  when it acts, on a batch end; it previews "train" and, when a val split
  exists, "val" too, together in one wandb.log call (there's no per-step
  validation loop to hook a separate "val batch end" into)."""

  def __init__(self, *, trigger_event):
    self.trigger = Trigger(trigger_event)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self._step(trainer, pl_module, batch_idx)

  def _step(self, trainer, pl_module, batch_idx):
    if not self.trigger.due(trainer, pl_module, batch_idx):
      return
    mode = "step"
    n = _completed_steps(trainer, pl_module)  # count of COMPLETED optimizer steps
    stages = None  # every stage the module has

    if pl_module.logger is None:
      return

    log_payload = {}
    with guarded_render(f"{mode}/panel"), torch.no_grad():
      import gsplat
      for stage in (stages or pl_module.preview_stages()):
        entry = next(pl_module.preview_entries(stage), None)
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

    pl_module.logger.log_metrics(log_payload)


def _colorize_scale_layers(scale, hit):
  """scale: (L,3,H,W) tensor -- the raw per-axis scale MULTIPLIER for each
  depth-peel layer, straight from the decoder (train_gs.py's GSModel
  output), before flatten_gaussians collapses layers/pixels into one point
  list. hit: (L,H,W) bool -- which grid cells hold an actual Gaussian
  (the rest is empty depth-peel padding).

  Returns one uint8 (H, L*W, 3) array: log10(max-axis scale) colorized
  (inferno) per layer side by side, invalid cells rendered flat gray so
  "no Gaussian here" reads differently from "small Gaussian here" under
  the colormap. Normalized against this call's OWN min/max (not a fixed
  range) -- the point is spotting a relative outlier, and the value can
  span many orders of magnitude (an exp() activation with no upper
  clamp -- see loss.scale_reg's own docstring in conf/train_gs.yaml)."""
  import matplotlib

  scale_np = scale.detach().float().cpu().numpy()       # (L,3,H,W)
  hit_np = hit.detach().cpu().numpy().astype(bool)       # (L,H,W)
  mag = scale_np.max(axis=1)                             # (L,H,W) -- worst axis per pixel

  log_mag = np.log10(np.clip(mag, 1e-8, None))
  valid = log_mag[hit_np]
  lo, hi = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
  if hi <= lo:
    hi = lo + 1.0
  norm = np.clip((log_mag - lo) / (hi - lo), 0.0, 1.0)

  cmap = matplotlib.colormaps["inferno"]
  rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)     # (L,H,W,3)
  rgb[~hit_np] = 60

  image = np.concatenate([rgb[l] for l in range(rgb.shape[0])], axis=1)  # (H, L*W, 3)
  return image


class ScaleLayersCallback(pl.Callback):
  """Per-depth-peel-layer visualization of the model's predicted Gaussian
  `scale`, pulled from the UNFLATTENED gauss_layers entry
  preview_entries(with_layers=True) puts alongside its usual flattened "gauss" (see
  train_gs.py's _preview_entry) -- flatten_gaussians concatenates every
  layer/pixel into one ~14k-point cloud for PanelCallback/OrbitCallback,
  which is exactly where a single exploded pixel goes invisible (buried in
  the mean/the render). This reads the pre-flatten (L,C,H,W) grid straight
  from the decoder instead, so one bad pixel in one layer stays visually
  and numerically identifiable.

  Same pull/cadence contract as PanelCallback/OrbitCallback (see this
  module's own top-of-file comment block): a trigger_event, pulls pl_module.preview_entries(stage). Stages whose entry has no "gauss_layers" key (an
  implementation that doesn't populate it) are skipped, same as a stage
  with no entry at all."""

  def __init__(self, *, trigger_event):
    self.trigger = Trigger(trigger_event)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self._step(trainer, pl_module, batch_idx)

  def _step(self, trainer, pl_module, batch_idx):
    if not self.trigger.due(trainer, pl_module, batch_idx):
      return
    mode = "step"
    n = _completed_steps(trainer, pl_module)  # count of COMPLETED optimizer steps
    stages = None  # every stage the module has

    if pl_module.logger is None:
      return

    log_payload = {}
    with guarded_render(f"{mode}/scale_layers"), torch.no_grad():
      for stage in (stages or pl_module.preview_stages()):
        entry = next(pl_module.preview_entries(stage, with_layers=True), None)
        if entry is None:
          continue
        layers = entry.get("gauss_layers")
        if layers is None:
          continue
        image = _colorize_scale_layers(layers["scale"], layers["hit"])
        log_payload[f"{stage}/scale_layers"] = wandb.Image(
          image, caption="log10(max-axis scale) per depth-peel layer, left to right")
    if not log_payload:
      return

    pl_module.logger.log_metrics(log_payload)


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
  reason, documented at each preview_entries() implementation),
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


def _orbit_frame_size(width, height, K, resolution):
  """(width, height, K) an orbit frame is rendered with. resolution None: the
  preview view's own. Otherwise the longer side becomes `resolution` pixels
  (aspect kept, both sides rounded to even numbers) and K is scaled by the
  actual per-axis ratios so the same 3D point lands at the same relative spot."""
  if resolution is None:
    return width, height, K
  scale = resolution / max(width, height)
  new_w, new_h = max(2, 2 * round(width * scale / 2)), max(2, 2 * round(height * scale / 2))
  K = K * K.new_tensor([new_w / width, new_h / height, 1.0])[:, None]
  return new_w, new_h, K


class OrbitCallback(pl.Callback):
  """Shared consumer half of the same pull pattern PanelCallback uses.
  Pulls pl_module.preview_entries(stage, count) -- one object at a time, kept
  only while its orbit is rendered -- and, since both scripts' Gaussians are already in
  world space there (see this module's own docstring), builds ONE
  world-frame turntable orbit per stage (_orbit_viewmats, radius = that
  stage's own "scene_scale") entirely in here. No per-script pose method
  needed at all. Renders all N orbit frames of a stage in ONE batched
  gsplat.rasterization() call, writes an mp4, and logs wandb.Video.

  trigger_event says when it acts -- see PanelCallback's own docstring, same contract exactly
  (train + val together).

  resolution: size in pixels of the longer side of the rendered frames, e.g.
  1024. The preview view's aspect ratio is kept (frame sides are rounded to
  even numbers for the mp4 encoder) and its K is scaled to match, so the
  framing is unchanged -- only the pixel density. None (default): the
  preview views' own size.

  grid: [width, height] objects per video (default [1, 1], one object). With more, every stage's video
  is a grid of that many DIFFERENT meshes (pl_module.preview_entries(stage, width * height); the first
  object is the one the single-object callbacks use), each orbiting in its own cell, row by row; cells
  without an object (fewer meshes than cells) stay black. `resolution` is then the longer side of the
  whole video, so each cell gets resolution // max(width, height) (rounded to even); None: each
  cell is the preview view's own size.

  source_view: with True (default) every object's cell shows its source view -- the image the model got as
  input -- to the LEFT of its orbit, the same size as an orbit frame and held still for every frame. The
  cell, and so the video, is then twice as wide; `resolution` still sizes the orbit frames (not the pair).

  path_format: where each mp4 is written, as a str.format() template over
  {stage} ("train"/"val"), {mode} (always "step") and {n} (completed
  steps; format specs like {n:06d} work). Parent directories are
  created and the files are kept. None (default): a throwaway temp dir,
  named "{stage}_orbit_{mode}{n}.mp4" and deleted when fit ends -- the
  videos then only survive as the wandb.Video uploads."""

  def __init__(self, *, trigger_event,
              num_frames: int = 24, fps: int = 12, crf: int = 28, elevation_deg: float = 20.0,
              path_format: str | None = None, resolution: int | None = None, grid: tuple[int, int] = (1, 1),
              source_view: bool = True):
    self.trigger = Trigger(trigger_event)
    self.source_view = bool(source_view)
    grid = tuple(int(g) for g in grid)
    if len(grid) != 2 or min(grid) < 1:
      raise ValueError(f"OrbitCallback: grid must be [width, height] with both >= 1, got {list(grid)}")
    self.grid = grid
    self.num_frames = num_frames
    self.fps = fps
    self.crf = crf
    self.elevation_deg = elevation_deg
    self.path_format = path_format
    self.resolution = resolution
    self.workdir = None

  def on_fit_start(self, trainer, pl_module):
    if self.path_format is None:
      self.workdir = tempfile.mkdtemp(prefix="orbit_")

  def on_fit_end(self, trainer, pl_module):
    if self.workdir is not None:
      shutil.rmtree(self.workdir, ignore_errors=True)
      self.workdir = None

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self._step(trainer, pl_module, batch_idx)

  def _render_object(self, entry, gsplat, resolution):
    """The num_frames frames, (F, H, W, 3) uint8, of one object's orbit at `resolution` (longer side); with
    source_view, (F, H, 2W, 3): the object's source view (still) to the left of the orbit."""
    gauss, views = entry["gauss"], entry["views"]
    device = gauss["means"].device
    viewmats = torch.from_numpy(
      _orbit_viewmats(entry["scene_scale"], self.num_frames, self.elevation_deg)).to(device)
    width, height, K = _orbit_frame_size(int(views["width"]), int(views["height"]), views["K"][0], resolution)
    Ks = repeat(K, "... -> b ...", b=viewmats.shape[0])
    rgb, _, _ = gsplat.rasterization(
      means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
      opacities=gauss["opacities"], colors=gauss["colors"],
      viewmats=viewmats, Ks=Ks,
      width=width, height=height,
      sh_degree=gauss["sh_degree"], render_mode="RGB", packed=True,
    )
    frames = (rgb.clamp(0.0, 1.0).cpu().numpy() * 255).astype("uint8")
    if not self.source_view:
      return frames
    # views["gt_rgb"][0] is the source view (premultiplied over black, like the orbit's background)
    source = F.interpolate(views["gt_rgb"][:1].float(), size=(height, width), mode="bilinear", antialias=True)
    source = (source.clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return np.concatenate([np.broadcast_to(source, frames.shape), frames], axis=2)

  def _render_stage(self, entries, gsplat):
    """(frames, mesh_indices) of a stage, from an iterable of its entries (a generator: one entry lives at a
    time, only while its orbit is rendered); None when there is no entry. frames: a list of (H, W, 3) uint8
    arrays. One object: its orbit as is. A grid: the objects' orbits side by side, row by row, each centred
    in a black cell of the same size."""
    cols, rows = self.grid
    cell_res = self.resolution if self.grid == (1, 1) else (
      None if self.resolution is None else max(2, 2 * (self.resolution // max(cols, rows) // 2)))
    orbits, meshes = [], []
    for entry in entries:
      orbits.append(self._render_object(entry, gsplat, cell_res))
      meshes.append(entry.get("mesh_index"))
    entry = None   # the last entry is not kept either
    if not orbits:
      return None
    if self.grid == (1, 1):
      return list(orbits[0]), meshes
    cell_h, cell_w = max(o.shape[1] for o in orbits), max(o.shape[2] for o in orbits)
    canvas = np.zeros((self.num_frames, rows * cell_h, cols * cell_w, 3), np.uint8)
    for j, orbit in enumerate(orbits):
      r, c = divmod(j, cols)
      y0, x0 = r * cell_h + (cell_h - orbit.shape[1]) // 2, c * cell_w + (cell_w - orbit.shape[2]) // 2
      canvas[:, y0:y0 + orbit.shape[1], x0:x0 + orbit.shape[2]] = orbit
    return list(canvas), meshes

  def _step(self, trainer, pl_module, batch_idx):
    if not self.trigger.due(trainer, pl_module, batch_idx):
      return
    mode = "step"
    n = _completed_steps(trainer, pl_module)  # count of COMPLETED optimizer steps
    stages = None  # every stage the module has

    if pl_module.logger is None:
      return

    log_payload = {}
    with guarded_render(f"{mode}/orbit"), torch.no_grad():
      import gsplat
      count = self.grid[0] * self.grid[1]
      for stage in (stages or pl_module.preview_stages()):
        rendered = self._render_stage(pl_module.preview_entries(stage, count), gsplat)
        if rendered is None:
          continue
        frames, meshes = rendered
        if self.path_format is None:
          path = os.path.join(self.workdir, f"{stage}_orbit_{mode}{n}.mp4")
        else:
          path = self.path_format.format(stage=stage, mode=mode, n=n)
          os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_mp4(frames, path, self.fps, self.crf)
        caption = f"{mode} {n}" + (f", meshes {meshes}" if count > 1 and None not in meshes else "")
        log_payload[f"{stage}/orbit"] = wandb.Video(path, caption=caption, format="mp4")
    if not log_payload:
      return

    pl_module.logger.log_metrics(log_payload)
