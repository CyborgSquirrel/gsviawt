"""Diffusion mode of train_gs.py: a diffusion model over the fitted layered 3DGS grids (World Tracing's recipe).

The data are fit_3dgs.py `gaussian_layout=layered` fits, one finalized grid .h5 per source view (see
finalize_fit_dataset.py: `mesh<M>_view<V>.h5`, (H, W, 6, .) arrays, NaN where a slot has no Gaussian). Gaussian POSITIONS come
from the source view's point cloud (the depth peel, as everywhere in train_gs.py); the model generates, per slot, 14 channels
(CHANNELS), normalized per channel (x0):
    opacity (1)   logit, clamped to +-7
    scale (3)     log of the absolute size
    rotation (4)  unit quaternion in the source camera frame, sign canonicalized (w >= 0)
    sh_dc (3)     (rgb - 0.5) / SH_C0 (degree-0 colour, what the head's sh_dc means)
    offset (3)    the fitted mean minus the depth-peel position (camera frame)

WT's recipe, as the paper gives it (https://arxiv.org/abs/2606.13652): rectified flow x_t = (1 - t) x0 + t eps, the network outputs x0
(x0-parameterization), the loss is the squared L2 distance to x0 on the valid slots, invalid slots' input is replaced by the noise
(x_net = A x_t + (1 - A) eps, A = the point cloud's per-layer hit mask), timestep conditioning by AdaLN (t * 1000, sinusoidal, MLP, 6
modulation vectors per block), per-layer timesteps (diffusion_schedule.py), 20 Euler steps of v = (x_t - x0_hat) / max(t, 0.05).

Architecture otherwise as the baseline's: the frozen DINOv2 mix backbone (model/backbone=moge_mix) and a 3-block frame/ray/global
trunk; the token is lossless, the 1024 feature channels next to the raw 14 * 14 * 14 noisy values of the patch (padded to a multiple
of 64 for the heads).
"""

import contextlib
import glob
import logging
import math
import os
import random
import warnings

import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch.utils.data import Dataset

from diffusion_schedule import LayerAwareSchedule
# the x0 layout, SH_C0 and quat_forms live in gaussians.py (re-exported here: gs_diffusion.CHANNELS, gs_diffusion.SCALE, ...)
from gaussians import SH_C0, X0_CHANNELS, X0_DIMS, X0_FIELDS, X0_SLICES, quat_forms  # noqa: F401
from gs_dataset import OPENGL_TO_OPENCV, rotate_quats_wxyz
from gs_decoder import AttentionBlock, apply_rope, rope_cos_sin
from grid_io import Grid, grid_shape, load_grid, read_grid  # noqa: F401  (load_grid: the dense dict, for the callers that want it)
from util import register_eval_resolver

log = logging.getLogger(__name__)

# the run configs' ${eval:...} fields (module.py registered it when this module still imported it, for SH_C0)
register_eval_resolver()

FIELDS, DIMS, CHANNELS = X0_FIELDS, X0_DIMS, X0_CHANNELS   # 14 channels
OPACITY_LOGIT_CLAMP = 7.0
T_MIN_CLAMP = 0.05     # WT's: the x0 -> velocity conversion divides by max(t, T_MIN_CLAMP)


# ---------------------------------------------------------------------------
# grids <-> x0 channels
# ---------------------------------------------------------------------------

def permute_scale_axes(scale, quat, order):
  """(N, 3) scales, (N, 4) wxyz quaternions, (N, 3) axis orders -> the same Gaussians with their axes in that order: the scales
  permuted and the rotation's columns permuted to match (a column negated where the permutation is a reflection -- a Gaussian is
  symmetric under axis negation). Covariance R diag(s^2) R^T unchanged."""
  from scipy.spatial.transform import Rotation
  R = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()                        # (N, 3, 3)
  R = np.take_along_axis(R, order[:, None, :], axis=2)
  R[np.linalg.det(R) < 0, :, 2] *= -1
  q = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
  return np.take_along_axis(scale, order, axis=1), q.astype(np.float32)


def sort_scale_axes(scale, quat):
  """The Gaussians with their axes ordered largest -> smallest (permute_scale_axes)."""
  return permute_scale_axes(scale, quat, np.argsort(-scale, axis=1))


ENCODING_DEFAULTS = dict(scale_order="keep", scale_floor_px=None, scale_floor_mode="max", rot_form="keep")   # what a config without the key means
SCALE_ORDERS = ("keep", "sorted", "random")
ROT_FORMS = ("keep", "canonical", "random")
SCALE_FLOOR_MODES = ("max", "add")
FLOOR_ADD_EPS = 0.1   # scale_floor_mode add: an axis at (or under) the floor is stored as this fraction of the floor above it


def normalize_encoding(encoding):
  """diffusion.scale_encoding (or a checkpoint's x0_encoding) -> the complete, validated encoding dict. The defaults fill missing keys;
  the legacy boolean scale_sort (configs and checkpoints before scale_order) maps to scale_order sorted / keep."""
  e = dict(encoding or {})
  if "scale_sort" in e:
    legacy = "sorted" if e.pop("scale_sort") else "keep"
    if e.setdefault("scale_order", legacy) != legacy:
      raise ValueError(f"scale_encoding: scale_sort and scale_order disagree ({legacy} vs {e['scale_order']})")
  e = {**ENCODING_DEFAULTS, **e}
  unknown = sorted(set(e) - set(ENCODING_DEFAULTS))
  if unknown:
    raise ValueError(f"scale_encoding: unknown keys {unknown}")
  if e["scale_order"] not in SCALE_ORDERS:
    raise ValueError(f"scale_order must be one of {SCALE_ORDERS}, got {e['scale_order']!r}")
  if e["rot_form"] not in ROT_FORMS:
    raise ValueError(f"rot_form must be one of {ROT_FORMS}, got {e['rot_form']!r}")
  if e["scale_floor_mode"] not in SCALE_FLOOR_MODES:
    raise ValueError(f"scale_floor_mode must be one of {SCALE_FLOOR_MODES}, got {e['scale_floor_mode']!r}")
  return e


def set_rot_form(q, rot_form, rng=None):
  """(N, 4) wxyz -> the same Gaussians' quaternions in the requested form, w >= 0. canonical: the form with the largest |w| (the
  one closest to no rotation in this frame); random: a uniformly random form per Gaussian; keep: q unchanged."""
  if rot_form == "keep":
    return q
  forms = quat_forms(q)
  pick = (np.abs(forms[:, :, 0]).argmax(1) if rot_form == "canonical"
          else (rng if rng is not None else np.random.default_rng()).integers(4, size=len(q)))
  out = forms[np.arange(len(q)), pick]
  return np.where(out[:, :1] < 0, -out, out).astype(q.dtype)


def encode_grid(grid, source, scale_order=None, scale_floor_px=None, rot_form="keep", rng=None, scale_sort=None,
                scale_floor_mode="max"):
  """The grid's Gaussians as raw (un-normalized) x0 channels, (CHANNELS, L, H, W) float32, zero at the slots without a Gaussian.
  `grid`: a grid_io.Grid (read_grid: a sparse file's rows, used as they are) or load_grid's dense dict (gathered into one first: the
  same rows, so the same x0 bit for bit).
  `source`: the dataset item's "source" dict (xyz_cam (L, H, W, 3) with NaN where there is no hit, hit (L, H, W), pose_gl (4, 4),
  K_depth (3, 3)).
  scale_floor_px: every scale axis is raised to at least this fraction of the pixel footprint at the Gaussian's depth in the source
  camera (z / fx): an axis thinner than ~1/10 px is invisible, but its log value (where the fit happened to leave it) is a wide,
  unpredictable target. scale_floor_mode: how the floor is applied -- max (every axis raised to it, the channel the log of the
  axis) | add (the channel the log of the axis's excess over the floor, at least FLOOR_ADD_EPS floors: the decode adds the floor
  back, scale = floor + exp(channel), so no prediction is ever under it; an axis at the floor renders FLOOR_ADD_EPS floors larger).
  scale_order: the order of every Gaussian's axes (the rotation permuted to match) -- keep (as fitted),
  sorted (largest -> smallest, sort_scale_axes: a scale channel means the same axis in every slot), random (a fresh random order per
  call). scale_sort: the legacy boolean (True = sorted). Both render the same as the grid (bla/_test_scale_reencode.py: floor 0.1 px >= 56 dB PSNR
  against the original render over 5000 renders). rot_form (set_rot_form, after the sort): which of a Gaussian's 4 equivalent
  quaternions is stored -- keep (as fitted, in the camera frame: after the sort an effectively random one), canonical (largest
  |w|), random (a fresh random one per call; rng: its generator, default a fresh one)."""
  # scale_order None: keep, or the legacy flag's choice; both given must agree
  e = normalize_encoding({**({"scale_order": scale_order} if scale_order is not None else {}), "scale_floor_px": scale_floor_px,
                          "scale_floor_mode": scale_floor_mode, "rot_form": rot_form, **({"scale_sort": scale_sort} if scale_sort is not None else {})})
  scale_order, rot_form = e["scale_order"], e["rot_form"]
  if "random" in (scale_order, rot_form) and rng is None:
    rng = np.random.default_rng()
  if not isinstance(grid, Grid):
    grid = Grid.from_dense(grid)
  valid = grid.valid                                                 # (L, H, W)
  hit = source["hit"].numpy()
  mismatch = int((valid != hit).sum())
  if mismatch:   # one slot is enough: a Gaussian without a point (NaN offset) would poison the loss
    raise ValueError(
      f"the grid's layer_valid disagrees with the point cloud's hit mask on {mismatch} of {valid.size} slots: "
      "is this grid from this render / this view?")
  c2w = source["pose_gl"].numpy() @ OPENGL_TO_OPENCV
  w2c = np.linalg.inv(c2w)
  # the math on the valid slots only (~6% of them), as (N, .) rows in [valid]'s order, scattered into the zeros once; every op is
  # elementwise per slot (or per row), so the values are the dense version's bit for bit, and the rng draws the same (N, .) shapes
  li, yi, xi = grid.idx                                              # np.nonzero(valid)
  take = lambda k: grid.rows[k]                                      # (N, .)
  means = take("gaussian_means")                                     # world frame
  # world -> source camera; a single row doubled: numpy's matmul takes BLAS gemv for one row, gemm (the dense version's, (W, 3)
  # per (layer, row)) for more, and the two can differ in the last bit
  means_cam = ((np.repeat(means, 2, 0) if len(means) == 1 else means) @ w2c[:3, :3].T)[:len(means)] + w2c[:3, 3]
  offset = means_cam - source["xyz_cam"].numpy()[li, yi, xi]

  quat = np.zeros((len(li), 4), np.float32)
  scales = take("gaussian_scales").astype(np.float32)
  floor = None
  if scale_floor_px is not None:
    footprint = means_cam[:, 2] / float(source["K_depth"][0, 0])                   # (N,): one pixel's size at that depth
    floor = float(scale_floor_px) * np.nan_to_num(footprint)[:, None]
    if scale_floor_mode == "max":
      scales = np.maximum(scales, floor)
  if len(li):
    q = rotate_quats_wxyz(take("gaussian_quats"), c2w[:3, :3].T)    # world -> source camera
    if scale_order == "sorted":
      scales, q = sort_scale_axes(scales, q)
    elif scale_order == "random":
      scales, q = permute_scale_axes(scales, q, np.argsort(rng.random((len(q), 3)), axis=1))
    q = np.where(q[:, :1] < 0, -q, q)                                # q and -q are one rotation: w >= 0
    quat[:] = set_rot_form(q, rot_form, rng)

  opacity = np.clip(take("gaussian_opacities"), 1e-6, 1 - 1e-6)
  opacity = np.clip(np.log(opacity / (1 - opacity)), -OPACITY_LOGIT_CLAMP, OPACITY_LOGIT_CLAMP)
  if floor is not None and scale_floor_mode == "add":   # (after the sort: the excess has the axes' order, the floor is per Gaussian)
    scales = np.maximum(scales - floor, FLOOR_ADD_EPS * floor)
  scale = np.log(np.maximum(scales, 1e-8))
  color = (take("gaussian_colors") - 0.5) / SH_C0

  x0 = np.zeros((CHANNELS,) + valid.shape, np.float32)
  x0[:, li, yi, xi] = np.concatenate([opacity[:, None], scale, quat, color, offset], axis=-1).astype(np.float32).T
  return x0                                                          # (CHANNELS, L, H, W)


SCALE, ROTATION, OFFSET = X0_SLICES["scale"], X0_SLICES["rotation"], X0_SLICES["offset"]
_SCALE_IDX = torch.arange(SCALE.start, SCALE.stop)


def augment_x0(x0, source, rng, slot_shuffle=False, slot_shuffle_block=2, slot_shuffle_max_px=3.0):
  """Render-invariant augmentation of an item's raw x0 (CHANNELS, L, H, W) (encode_grid's), returned as a new array: the same
  Gaussians, written down differently. Training items only (diffusion.augment); the statistics and the validations see encode_grid.
  (A random scale-axis order is an encoding option instead: diffusion.scale_encoding.scale_order=random.)
  slot_shuffle: every layer is tiled into n x n blocks (n = slot_shuffle_block; the tiling shifted by a random 0..n-1 slots in
  each direction per item, the partial blocks at the borders left alone); every block has its Gaussians randomly permuted among its
  slots (all or nothing: the fitted arrangement is just one of the permutations): every channel moves along, the offsets are
  re-anchored to the new slot (mean = anchor + offset kept). Only the block's valid slots whose anchors are within slot_shuffle_max_px
  pixel footprints of the block's mean anchor take part (not across depth edges); the others stay.
  (With the x0 loss on, the x0 target is shuffled too: at high t its best prediction blends a block's Gaussians.)"""
  x0 = x0.copy()
  if slot_shuffle:
    shuffle_slots(x0, source, rng, int(slot_shuffle_block), slot_shuffle_max_px)
  return x0


def shuffle_slots(x0, source, rng, n, max_px):
  """augment_x0's block shuffle, in place on x0 (CHANNELS, L, H, W)."""
  c, l, h, w = x0.shape
  sy, sx = (int(v) for v in rng.integers(n, size=2))                 # the tiling's shift, 0..n-1 slots
  hb, wb = (h - sy) // n, (w - sx) // n
  if hb == 0 or wb == 0:
    return
  rows, cols = slice(sy, sy + hb * n), slice(sx, sx + wb * n)

  anchor = source["xyz_cam"].numpy()[:, rows, cols]                  # (L, H', W', 3)
  hit = source["hit"].numpy()[:, rows, cols]                         # (L, H', W')
  A = anchor.reshape(l, hb, n, wb, n, 3).transpose(0, 1, 3, 2, 4, 5).reshape(l, hb, wb, n * n, 3)
  V = hit.reshape(l, hb, n, wb, n).transpose(0, 1, 3, 2, 4).reshape(l, hb, wb, n * n)
  X = x0[:, :, rows, cols].reshape(c, l, hb, n, wb, n).transpose(0, 1, 2, 4, 3, 5).reshape(c, l, hb, wb, n * n)

  with warnings.catch_warnings():                                    # blocks without a valid slot: "mean of empty slice"
    warnings.simplefilter("ignore", RuntimeWarning)
    centre = np.nanmean(np.where(V[..., None], A, np.nan), axis=3, keepdims=True)                   # (L, hb, wb, 1, 3)
  dist = np.linalg.norm(A - centre, axis=-1)                                                          # NaN at invalid slots
  footprint = np.nan_to_num(A[..., 2]) / float(source["K_depth"][0, 0])
  take = V & (np.nan_to_num(dist, nan=np.inf) <= max_px * footprint)                         # (L, hb, wb, n*n)

  # a uniform random permutation of each block's taking slots among themselves; the others map to themselves
  pos = np.broadcast_to(np.arange(n * n), take.shape)
  dest = np.sort(np.where(take, pos, n * n), axis=-1)                # the taking slots, in order (then n*n padding)
  src = np.take_along_axis(pos, np.argsort(np.where(take, rng.random(take.shape), 2.0), axis=-1), axis=-1)
  k = take.sum(-1, keepdims=True)
  first = np.arange(n * n) < k                                       # (L, hb, wb, n*n): the first k entries are real
  perm = np.concatenate([np.broadcast_to(np.arange(n * n), take.shape), np.zeros(take.shape[:-1] + (1,), int)], -1)
  np.put_along_axis(perm, np.where(first, dest, n * n), src, axis=-1)   # slot -> the slot its Gaussian comes from (padding -> the
  perm = perm[..., :n * n]                                           #   extra column, dropped)

  Xn = np.take_along_axis(X, np.broadcast_to(perm, X.shape), axis=-1)
  An = np.take_along_axis(A, perm[..., None].repeat(3, -1), axis=-2)                                 # the Gaussians' old anchors
  off = Xn[OFFSET] + np.moveaxis(An, -1, 0) - np.moveaxis(A, -1, 0)  # mean = old anchor + offset, re-anchored to the new slot
  Xn[OFFSET] = np.where(take, off, X[OFFSET])
  x0[:, :, rows, cols] = Xn.reshape(c, l, hb, wb, n, n).transpose(0, 1, 2, 4, 3, 5).reshape(c, l, hb * n, wb * n)


def quat_to_matrix(q):
  """(N, 4) unit wxyz quaternions -> (N, 3, 3) rotation matrices (torch; scipy's convention)."""
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def matrix_to_quat(R):
  """(N, 3, 3) rotation matrices -> (N, 4) unit wxyz quaternions with w >= 0 (torch; the largest-component branch per row)."""
  m = R
  t = torch.stack([1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                   1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1).clamp_min(1e-12)
  r = t.sqrt()                                                       # 2|w|, 2|x|, 2|y|, 2|z|
  a, b, c = m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]
  d, e, f = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]
  cand = torch.stack([
    torch.stack([r[:, 0] / 2, a / (2 * r[:, 0]), b / (2 * r[:, 0]), c / (2 * r[:, 0])], 1),
    torch.stack([a / (2 * r[:, 1]), r[:, 1] / 2, d / (2 * r[:, 1]), e / (2 * r[:, 1])], 1),
    torch.stack([b / (2 * r[:, 2]), d / (2 * r[:, 2]), r[:, 2] / 2, f / (2 * r[:, 2])], 1),
    torch.stack([c / (2 * r[:, 3]), e / (2 * r[:, 3]), f / (2 * r[:, 3]), r[:, 3] / 2], 1)], 1)   # (N, 4 branches, 4)
  q = cand[torch.arange(len(m), device=m.device), t.argmax(1)]
  q = F.normalize(q, dim=1)
  return torch.where(q[:, :1] < 0, -q, q)


def format_projection(mean, std, hit, anchor_z, fx, scale_sort=False, scale_floor_px=None, scale_order=None, rot_form="keep",
                      scale_floor_mode="max", quat="unit"):
  """A function on normalized x0 predictions (B, L, CHANNELS, H, W) that puts the valid slots' Gaussians into the training grids'
  format (encode_grid with this encoding): the scale axes ordered largest -> smallest (the rotation permuted to match, as
  sort_scale_axes), every axis at least scale_floor_px pixel footprints, the rotation a unit quaternion with w >= 0. For the sampler's
  intermediate predictions (GSDiffusionModel.sample's `project`): render-invariant apart from the floor. hit (B or 1, L, H, W);
  anchor_z (L, H, W) the anchors' depth (the floor is at the Gaussian's mean depth: anchor + offset z); fx the depth camera's
  focal length. Takes the run's encoding as keywords: scale_order (keep | sorted | random; only sorted is projected) or the legacy
  scale_sort, rot_form (canonical: the quaternion form with the largest |w|; keep / random: left as predicted). quat: unit (the
  rotation becomes a unit quaternion with w >= 0) | keep (the predicted quaternion is left exactly as it is,
  except where the sort reorders the axes: there the permuted rotation at the predicted length, its sign as matrix_to_quat gives it)."""
  if scale_floor_px is not None and scale_floor_mode != "max":
    raise NotImplementedError("format_projection: only scale_floor_mode max (with add the decode applies the floor itself)")
  if quat not in ("unit", "keep"):
    raise ValueError(f"format_projection: quat must be unit or keep, got {quat!r}")
  if quat == "keep" and rot_form == "canonical":
    raise NotImplementedError("format_projection: quat=keep with rot_form canonical")
  if scale_order is not None:
    scale_sort = scale_order == "sorted"
  sc, ro = SCALE, ROTATION   # the scale and rotation channels

  def project(x0):
    x = x0.clone()
    v = hit.expand(x.shape[0], *hit.shape[1:]).bool()                # (B, L, H, W)
    raw = x * std[None, None, :, None, None] + mean[None, None, :, None, None]
    s = raw[:, :, sc].permute(0, 1, 3, 4, 2)[v].float()               # (N, 3) log-scales
    q_raw = raw[:, :, ro].permute(0, 1, 3, 4, 2)[v].float()
    q = F.normalize(q_raw, dim=1)
    moved = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    if scale_sort:
      order = torch.argsort(-s, dim=1, stable=True)   # stable: tied axes (the floor ties many) keep their order -- as encode_grid's
                                                      # sort, so an encoded grid is a fixed point (CUDA's unstable sort reverses ties)
      R = torch.gather(quat_to_matrix(q), 2, order[:, None, :].expand(-1, 3, -1))
      R[torch.linalg.det(R) < 0, :, 2] *= -1
      q, s = matrix_to_quat(R), torch.gather(s, 1, order)
      moved = (order != torch.arange(3, device=order.device)).any(1)
    else:
      q = torch.where(q[:, :1] < 0, -q, q)
    if rot_form == "canonical":   # q times 1, i, j, k: the same Gaussian; keep the one with the largest |w|
      forms = quat_forms(q)
      q = forms[torch.arange(len(q), device=q.device), forms[:, :, 0].abs().argmax(1)]
      q = torch.where(q[:, :1] < 0, -q, q)
    if quat == "keep":   # as predicted, but where the axes were reordered: the permuted rotation at the predicted length
      q = torch.where(moved[:, None], q * q_raw.norm(dim=1, keepdim=True), q_raw)
    if scale_floor_px is not None:
      z = anchor_z.expand(*v.shape)[v].float() + raw[:, :, OFFSET.start + 2].permute(0, 1, 2, 3)[v].float()   # the mean's depth, as encode_grid
      floor = torch.log(float(scale_floor_px) * z.clamp_min(1e-3) / float(fx))   # (a wild predicted offset must not make it NaN)
      s = torch.maximum(s, floor[:, None])
    out = raw.permute(0, 1, 3, 4, 2).clone()                          # (B, L, H, W, C)
    out[..., sc][v] = s.to(out.dtype)
    out[..., ro][v] = q.to(out.dtype)
    sub = out.permute(0, 1, 4, 2, 3)
    return (sub - mean[None, None, :, None, None]) / std[None, None, :, None, None]

  return project


DECODE_CLAMPS = ("channel", "scale")


def decode_x0(x0, mean, std, lo=None, hi=None, clamp="channel", floor=None):
  """Normalized x0 (B, L, CHANNELS, H, W) -> the Gaussian dict flatten_gaussians reads (gs_decoder.activate_gaussians' layout).
  lo / hi (CHANNELS,): the range of every channel in the training grids (an unbounded exp(scale) of a half-trained sampler would
  otherwise become a scene-sized Gaussian). clamp (diffusion.decode_clamp): channel = every channel clamped into its own range;
  scale = only the 3 scale channels, all into ONE range (the smallest lo and the largest hi over the 3 axes: with sorted axes a
  per-axis range would cut an axis written into the "wrong" channel), every other channel unclamped (as gs_decoder's head).
  A clamped value has no gradient. floor (scale_floor_mode add; GSDiffusionModel.floor_context): {"px": scale_floor_px, "anchor_z"
  (B, L, H, W), "fx" (B,)}: scale = floor + exp(channel), the floor scale_floor_px pixel footprints at the Gaussian's mean depth
  (anchor + offset z, as encode_grid's), raw_scale its log."""
  raw = x0 * std[None, None, :, None, None] + mean[None, None, :, None, None]
  if lo is not None:
    if clamp == "scale":
      idx = _SCALE_IDX.to(lo.device)
      lo = torch.full_like(lo, -float("inf")).index_fill(0, idx, float(lo[SCALE].min()))
      hi = torch.full_like(hi, float("inf")).index_fill(0, idx, float(hi[SCALE].max()))
    elif clamp != "channel":
      raise ValueError(f"decode clamp must be one of {DECODE_CLAMPS}, got {clamp!r}")
    raw = torch.maximum(torch.minimum(raw, hi[None, None, :, None, None]), lo[None, None, :, None, None])
  opacity, scale, rotation, sh_dc, offset = raw.split(DIMS, dim=2)
  size = torch.exp(scale)
  if floor is not None:
    z = floor["anchor_z"][:, :, None].to(raw) + offset[:, :, 2:3]                   # (B, L, 1, H, W): the mean's depth
    size = size + float(floor["px"]) * z.clamp_min(1e-3) / floor["fx"].to(raw)[:, None, None, None, None]
    scale = torch.log(size)
  return {
    "opacity": torch.sigmoid(opacity),
    "raw_scale": scale,
    "scale": size,
    "rotation": F.normalize(rotation + 1e-6, dim=2),
    "sh_dc": sh_dc,
    "offset": offset,
  }


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_grid_index(grids_dir, source_h5=None):
  """{row of the render h5 of the grid's source view: path}, and the render h5 the grids were fit from (the grids' own attribute,
  a path on the machine that fit them; source_h5 overrides it, for another machine)."""
  index, sources = {}, set()
  for path in sorted(glob.glob(os.path.join(grids_dir, "*.h5"))):
    with h5py.File(path, "r") as f:
      index[int(f.attrs["primary_view_row"])] = path
      sources.add(str(f.attrs["source_h5"]))
  if not index:
    raise SystemExit(f"no finalized grids (*.h5 with primary_view_row) in {grids_dir!r}")
  if len(sources) != 1:
    raise SystemExit(f"the grids in {grids_dir!r} come from several render files: {sorted(sources)}")
  return index, source_h5 or sources.pop()


def check_grids(grid_index, image_size, patch_size, num_layers):
  """The grids' (H, W, L) must be the model's grid: image_size square (a multiple of the patch size, so the patches tile it) and
  num_layers layers. (data.depth_size must stay null: grids are at the depth peel's own resolution.)"""
  h, w, l = grid_shape(next(iter(grid_index.values())))
  if (h, w) != (image_size, image_size) or h % patch_size or l != num_layers:
    raise SystemExit(
      f"the grids are {h}x{w} with {l} layers, but the model wants {image_size}x{image_size} (a multiple of the patch size "
      f"{patch_size}) with {num_layers} layers: check data.image_size / data.num_layers")


def make_listing(grid_index, source_h5, out_path):
  """The 'features' file GSDataModule's datasets are built from, without features: one row per grid (its source view and mesh)."""
  rows = sorted(grid_index)
  with h5py.File(source_h5, "r") as f:
    meshes = f["mesh_index"][:][rows]
  with h5py.File(out_path, "w") as f:
    f.create_dataset("view_indices", data=np.array(rows, np.int64))
    f.create_dataset("mesh_index", data=np.asarray(meshes, np.int64))
    f.create_dataset("features", data=np.zeros((len(rows), 1, 1, 1), np.float16)).attrs["stored_as_uint16_bits"] = False
    f.attrs["source_h5"] = source_h5
    f.attrs["input"] = source_h5
  return out_path


class GridTargetDataset(Dataset):
  """Wraps a dataset of render items (gs_dataset.WTFeatureDataset) and adds each item's ground-truth Gaussians, "x0_raw"
  (CHANNELS, L, H, W) (see encode_grid, with `encoding`: its scale_order / scale_floor_px / rot_form), read from the grid file of its source view."""

  def __init__(self, inner, grid_index, encoding=None, seeded=False, augment=None):
    """seeded: the random parts of the encoding (scale_order / rot_form random) and the target views are a function of the item's
    source view, the same in every epoch and every run (validation sets); otherwise fresh per call (training). augment: augment_x0's keyword arguments
    (diffusion.augment), for the training set only; None: none."""
    self.inner = inner
    self.seeded = seeded
    self.seed_targets = seeded   # eval_diffusion turns this off: it seeds the target views itself (its own, older convention)
    augment = dict(augment or {})
    self.augment = augment if augment.get("slot_shuffle") else None   # the default: off
    self.grid_index = grid_index
    self.encoding = normalize_encoding(encoding)
    self.volumes = inner.volumes   # the preview code reads the volumes' mesh/view columns
    missing = sorted(set(int(v) for v in inner.volumes.df["view_idx"].to_list()) - set(grid_index))
    if missing:
      raise SystemExit(f"source views {missing} have no grid file")

  def __len__(self):
    return len(self.inner)

  def __getitem__(self, idx):
    if self.seed_targets:   # the volume's target views (drawn from Python's global random) a function of the volume; global state kept
      state = random.getstate()
      random.seed(f"val_targets/{int(self.volumes.df['view_idx'][idx])}")
      try:
        item = self.inner[idx]
      finally:
        random.setstate(state)
    else:
      item = self.inner[idx]
    grid = read_grid(self.grid_index[item["source_view"]])
    # half precision in transit (dense (14, 6, 504, 504) is 85 MB in float32): the log-scales (|x| < 17), logits, colours and offsets
    # lose < 2% of the channel's std; the model upcasts it
    rng = np.random.default_rng(int(item["source_view"])) if self.seeded else None
    x0 = encode_grid(grid, item["source"], rng=rng, **self.encoding)
    if self.augment:   # a fresh stream per call: torch's RNG is seeded per dataloader worker
      x0 = augment_x0(x0, item["source"], np.random.default_rng(int(torch.randint(2 ** 62, ()))), **self.augment)
    item["x0_raw"] = torch.from_numpy(x0).half()
    return item


def encoding_tag(encoding):
  """The cache subdirectory of an encoding's moments: "" for the default encoding (the caches written before encodings existed)."""
  e = normalize_encoding(encoding)
  parts = {"keep": [], "sorted": ["sort"], "random": ["randorder"]}[e["scale_order"]]
  parts = parts + ([f"floor{float(e['scale_floor_px']):g}px" + ("add" if e["scale_floor_mode"] == "add" else "")]
                   if e["scale_floor_px"] is not None else [])
  parts += [f"rot{e['rot_form']}"] if e["rot_form"] != "keep" else []
  return "_".join(parts)


def _grid_moments(dataset, i, path, cache_dir):
  """(sum, sum of squares, count, min, max) of the x0 channels over the valid slots of volume i; cached per grid file (size + mtime)
  and per encoding (dataset.encoding: its own subdirectory of cache_dir)."""
  if cache_dir and encoding_tag(dataset.encoding):
    cache_dir = os.path.join(cache_dir, encoding_tag(dataset.encoding))
  st = os.stat(path)
  stamp = np.array([st.st_size, st.st_mtime_ns], np.int64)
  cache = os.path.join(cache_dir, os.path.basename(path) + ".npz") if cache_dir else None
  if cache and os.path.exists(cache):
    z = np.load(cache)
    if (z["stamp"] == stamp).all():
      return z["s"], z["s2"], int(z["n"]), z["lo"], z["hi"]
  item = dataset.inner[i]
  grid = read_grid(path)
  x = torch.from_numpy(encode_grid(grid, item["source"], rng=np.random.default_rng(int(item["source_view"])), **dataset.encoding))[:, item["source"]["hit"]].double()   # (CHANNELS, valid slots), full precision
  if x.shape[1]:
    moments = (x.sum(1).numpy(), x.square().sum(1).numpy(), x.shape[1], x.amin(1).numpy(), x.amax(1).numpy())
  else:   # a grid without a Gaussian: no moments (compute_stats raises when no grid has any)
    moments = (np.zeros(CHANNELS), np.zeros(CHANNELS), 0, np.full(CHANNELS, np.inf), np.full(CHANNELS, -np.inf))
  if cache:
    try:
      os.makedirs(cache_dir, exist_ok=True)
      np.savez(cache, stamp=stamp, s=moments[0], s2=moments[1], n=moments[2], lo=moments[3], hi=moments[4])
    except OSError:
      pass   # a read-only grids dir: just recompute next time
  return moments


def compute_stats(dataset, grid_index, cache_dir=None):
  """Per-channel (mean, std, min, max) of the x0 channels over the VALID slots of `dataset`'s volumes (the training grids). The
  moments of each grid are cached in cache_dir, so the pass over the grids happens once; the volumes are read without their
  target views and image."""
  inner = dataset.inner
  saved = inner.num_views, inner.image_size
  inner.num_views, inner.image_size = 0, None
  total = np.zeros(CHANNELS)
  total2 = np.zeros(CHANNELS)
  lo = np.full(CHANNELS, np.inf)
  hi = np.full(CHANNELS, -np.inf)
  count = 0
  try:
    for i in range(len(dataset)):
      s, s2, n, lo_i, hi_i = _grid_moments(dataset, i, grid_index[int(dataset.volumes.df["view_idx"][i])], cache_dir)
      total += s
      total2 += s2
      count += n
      lo, hi = np.minimum(lo, lo_i), np.maximum(hi, hi_i)
  finally:
    inner.num_views, inner.image_size = saved
  if count == 0:
    raise ValueError("no valid Gaussian slot in the training grids")
  mean = total / count
  std = np.sqrt(np.maximum(total2 / count - mean ** 2, 0)).clip(min=1e-4)
  log.info("x0 statistics over %d valid slots of %d grids: mean %s std %s", count, len(dataset), mean.round(3), std.round(3))
  return tuple(torch.from_numpy(v).float() for v in (mean, std, lo, hi))


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

TOKEN_DROPS = ("none", "empty", "empty_except_layer0")
COND_NULLS = ("zeros", "learned")
GUIDANCE_DEFAULTS = dict(scale=1.0, interval=(0.0, 1.0), fields=None)   # diffusion.cfg: what a config without the key means


def normalize_cond_dropout(cd):
  """diffusion.cond_dropout (None or missing keys: off) -> {"p": float in [0, 1], "null": zeros | learned}."""
  cd = {"p": 0.0, "null": "zeros", **dict(cd or {})}
  unknown = sorted(set(cd) - {"p", "null"})
  if unknown:
    raise ValueError(f"diffusion.cond_dropout: unknown keys {unknown}")
  p = float(cd["p"])
  if not 0.0 <= p <= 1.0:
    raise ValueError(f"diffusion.cond_dropout.p must be in [0, 1], got {p}")
  if cd["null"] not in COND_NULLS:
    raise ValueError(f"diffusion.cond_dropout.null must be one of {COND_NULLS}, got {cd['null']!r}")
  return {"p": p, "null": str(cd["null"])}


def normalize_guidance(g):
  """diffusion.cfg (classifier-free guidance at sampling; None or missing keys: the defaults, off) -> {"scale": float,
  "interval": (lo, hi), "fields": tuple of FIELDS names or None (all)}."""
  g = {**GUIDANCE_DEFAULTS, **{k: v for k, v in dict(g or {}).items()}}
  unknown = sorted(set(g) - set(GUIDANCE_DEFAULTS))
  if unknown:
    raise ValueError(f"diffusion.cfg: unknown keys {unknown}")
  lo, hi = (float(v) for v in g["interval"])
  if lo > hi:
    raise ValueError(f"diffusion.cfg.interval must be [lo, hi] with lo <= hi, got {[lo, hi]}")
  fields = None
  if g["fields"] is not None:
    fields = tuple(str(f) for f in g["fields"])
    bad = sorted(set(fields) - set(FIELDS))
    if bad or not fields:
      raise ValueError(f"diffusion.cfg.fields must be a non-empty subset of {FIELDS} (or null: all), got {list(fields)}")
  return {"scale": float(g["scale"]), "interval": (lo, hi), "fields": fields}


def warn_guidance_without_dropout(guidance, cond_dropout_p):
  """Guidance (scale != 1) on a run trained without condition dropout: its unconditional branch was never trained."""
  if guidance["scale"] != 1.0 and float(cond_dropout_p) == 0.0:
    msg = (f"diffusion.cfg.scale={guidance['scale']:g} on a run trained with diffusion.cond_dropout.p=0: the unconditional "
           "branch (null image condition) was never trained, so it (and the guided mix) is not meaningful")
    log.warning(msg)
    warnings.warn(msg)


def token_keep(hit, patch_size, token_drop):
  """The tokens the trunk runs on (diffusion.token_drop), (B, L, P) bool in the tokens' (b, l, (h w)) order, or None (none: all, the
  dense path). A token is one layer's patch; it is empty when none of its patch_size^2 slots is valid (hit). empty: every empty
  token dropped; empty_except_layer0: the empty tokens of layers 1.. dropped, all of layer 0 kept (its empty patches are the
  background: image features only)."""
  if token_drop not in TOKEN_DROPS:
    raise ValueError(f"token_drop must be one of {TOKEN_DROPS}, got {token_drop!r}")
  if token_drop == "none":
    return None
  b, l, h, w = hit.shape
  keep = hit.bool().reshape(b, l, h // patch_size, patch_size, w // patch_size, patch_size).any(5).any(3).reshape(b, l, -1)
  if token_drop == "empty_except_layer0":
    keep[:, 0] = True
  return keep


class PackIndex:
  """Index tables for running the trunk on the kept tokens of a (B, L, P) keep mask only, as packed (T, D) rows in (b, l, p) order:
  a frame segment (one (sample, layer)) and a global segment (one sample) are contiguous runs of rows (cu_frame / cu_global, the
  zero-length ones left out); the ray block's columns (one (sample, patch) with >= 1 kept layer) are a (C, L) table of row
  indices, -1 where that layer's token is dropped."""

  def __init__(self, keep):
    B, L, P = keep.shape
    dev = keep.device
    self.B, self.L, self.P = B, L, P
    self.flat = keep.reshape(-1).nonzero().squeeze(1)                 # (T,) the kept tokens' flat (b, l, p) indices
    self.T = len(self.flat)
    self.b = self.flat // (L * P)
    self.l = (self.flat // P) % L
    self.p = self.flat % P
    self.frame = self.b * L + self.l                                  # (T,) the row's (sample, layer)
    self.cu_frame, self.max_frame = self._cu(self.frame, B * L)
    self.cu_global, self.max_global = self._cu(self.b, B)
    table = torch.full((B * P, L), -1, dtype=torch.long, device=dev)  # (sample, patch) x layer -> row
    col = self.b * P + self.p
    table[col, self.l] = torch.arange(self.T, device=dev)
    used = (table >= 0).any(1)
    self.ray_table = table[used]                                      # (C, L)
    self.ray_valid = self.ray_table >= 0
    self.ray_col = (used.cumsum(0) - 1)[col]                          # (T,) the row's place in the table: (ray_col, l)

  @staticmethod
  def _cu(seg, n):
    counts = torch.bincount(seg, minlength=n)
    counts = counts[counts > 0]                                       # the rows are sorted by segment
    cu = torch.zeros(len(counts) + 1, dtype=torch.int32, device=seg.device)
    cu[1:] = counts.cumsum(0)
    return cu, int(counts.max()) if len(counts) else 0

  def gather(self, x):
    """(B, L, P, ...) -> the kept rows (T, ...)."""
    return x.reshape(self.B * self.L * self.P, *x.shape[3:])[self.flat]

  def scatter(self, rows):
    """(T, ...) rows -> (B, L, P, ...), zeros at the dropped tokens."""
    out = rows.new_zeros(self.B * self.L * self.P, *rows.shape[1:]).index_copy(0, self.flat, rows)
    return out.reshape(self.B, self.L, self.P, *rows.shape[1:])


def segment_attention(q, k, v, cu, max_len):
  """q, k, v (T, H, hd) -> (T, H, hd): softmax attention within the segments cu (int32 cumulative lengths). Under CUDA autocast
  (bf16 / fp16) torch's varlen flash kernel, at the autocast dtype (as SDPA's dense path under autocast); otherwise (fp32, CPU)
  SDPA per segment."""
  if q.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16):
    from torch.nn.attention.varlen import varlen_attn
    dt = torch.get_autocast_dtype("cuda")
    return varlen_attn(q.to(dt).contiguous(), k.to(dt).contiguous(), v.to(dt).contiguous(), cu, cu, max_len, max_len)
  bounds = cu.tolist()
  return torch.cat([F.scaled_dot_product_attention(q[a:e].transpose(0, 1), k[a:e].transpose(0, 1), v[a:e].transpose(0, 1))
                    .transpose(0, 1) for a, e in zip(bounds[:-1], bounds[1:])])


def sinusoidal_embedding(t, dim=256, theta=10_000.0):
  """WT's (Wan 2.1's) timestep embedding: [cos, sin] of t * theta^(-k / (dim / 2)); t: (N,) -> (N, dim) float32."""
  half = dim // 2
  freqs = torch.outer(t.double(), torch.pow(theta, -torch.arange(half, device=t.device).double() / half))
  return torch.cat([freqs.cos(), freqs.sin()], dim=1).float()


class DiTBlock(AttentionBlock):
  """AttentionBlock with WT's DiT-style AdaLN (DecoderBlockDiT): the timestep gives, per (sample, layer), a shift / scale / gate for
  the attention and for the MLP, added to a learned per-block modulation. adaln: (B, L or 1, 6, D)."""

  def __init__(self, dim, num_heads, mode, mlp_ratio=4.0, layer_scale_init=1e-4):
    super().__init__(dim, num_heads, mode, mlp_ratio, layer_scale_init)
    self.modulation = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

  def _attend_normed(self, h, cos, sin):
    """Like AttentionBlock._attend, with the (modulated) normalization already applied."""
    n, s, d = h.shape
    q, k, v = self.qkv(h).reshape(n, s, 3, self.num_heads, d // self.num_heads).unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    if cos is not None:
      q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return self.proj(out.transpose(1, 2).reshape(n, s, d))

  def _attend_short(self, h):
    """Attention over sequences of a handful of tokens (the L layers of a pixel): an explicit softmax(q k^T) v. The fused kernels are
    built for long sequences: flash pads to 128 and its backward asks 9.3 GiB of scratch at batch 4 with 59 heads, this needs ~2 GiB
    with the gradients and is faster."""
    n, s, d = h.shape
    hd = d // self.num_heads
    q, k, v = self.qkv(h).reshape(n, s, 3, self.num_heads, hd).unbind(2)                    # (n, s, heads, hd)
    q, k = self.q_norm(q), self.k_norm(k)
    att = torch.einsum("nshd,nthd->nhst", q, k) * hd ** -0.5
    out = torch.einsum("nhst,nthd->nshd", att.softmax(-1).to(v.dtype), v)
    return self.proj(out.reshape(n, s, d))

  def _mix(self, h, cos, sin, b, l, p, d):
    if self.mode == "frame":
      return self._attend_normed(h.reshape(b * l, p, d), cos, sin).reshape(b, l, p, d)
    if self.mode == "ray":
      return self._attend_short(h.transpose(1, 2).reshape(b * p, l, d)).reshape(b, p, l, d).transpose(1, 2)
    return self._attend_normed(h.reshape(b, l * p, d), cos.repeat(l, 1), sin.repeat(l, 1)).reshape(b, l, p, d)

  def forward(self, x, cos, sin, adaln):
    b, l, p, d = x.shape
    mod = (self.modulation + adaln.float()).unbind(2)                 # six (B, L or 1, D)
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (m[:, :, None, :] for m in mod)
    h = (self.norm1(x).float() * (1 + scale_a) + shift_a).to(x.dtype)
    x = x + (self.ls1 * self._mix(h, cos, sin, b, l, p, d) * gate_a).to(x.dtype)
    h = (self.norm2(x).float() * (1 + scale_m) + shift_m).to(x.dtype)
    return x + (self.ls2 * self.mlp(h) * gate_m).to(x.dtype)

  def _mix_packed(self, h, pack, cos_r, sin_r):
    """_mix on packed rows h (T, D): the frame / global blocks attend within the rows' segments (RoPE at each row's own patch
    position), the ray block over each column's kept layers (the explicit softmax, the dropped layers' keys masked)."""
    t, d = h.shape
    hd = d // self.num_heads
    q, k, v = self.qkv(h).reshape(t, 3, self.num_heads, hd).unbind(1)                      # (T, heads, hd)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.mode == "ray":
      idx = pack.ray_table.clamp_min(0)                                                     # (C, L)
      att = torch.einsum("cshd,cthd->chst", q[idx], k[idx]) * hd ** -0.5
      att = att.masked_fill(~pack.ray_valid[:, None, None, :], float("-inf"))
      out = torch.einsum("chst,cthd->cshd", att.softmax(-1).to(v.dtype), v[idx])[pack.ray_col, pack.l]   # back to rows
    else:
      q, k = apply_rope(q, cos_r, sin_r), apply_rope(k, cos_r, sin_r)
      cu, mx = (pack.cu_frame, pack.max_frame) if self.mode == "frame" else (pack.cu_global, pack.max_global)
      out = segment_attention(q, k, v, cu, mx)
    return self.proj(out.reshape(t, d))

  def forward_packed(self, x, pack, cos_r, sin_r, adaln, seg):
    """forward on the packed rows x (T, D) of a PackIndex. adaln (N, 6, D) and seg (T,): row i is modulated by adaln[seg[i]] (the
    frame blocks: N = B * L, one per (sample, layer); ray / global: N = B, the layers' mean)."""
    mod = self.modulation + adaln.float()                                                    # (N, 6, D)
    row = lambda i: mod[:, i][seg]                                                           # (T, D), gathered when used
    h = (self.norm1(x).float() * (1 + row(1)) + row(0)).to(x.dtype)
    x = x + (self.ls1 * self._mix_packed(h, pack, cos_r, sin_r) * row(2)).to(x.dtype)
    h = (self.norm2(x).float() * (1 + row(4)) + row(3)).to(x.dtype)
    return x + (self.ls2 * self.mlp(h) * row(5)).to(x.dtype)


class DiTTrunk(nn.Module):
  """gs_decoder.TransformerTrunk with AdaLN blocks: frame -> ray -> global, the per-layer FiLM before every frame block, and WT's
  timestep rule: the frame blocks get each layer's own AdaLN input, the ray and global blocks the mean over the layers."""

  MODES = ("frame", "ray", "global")

  def __init__(self, dim, num_layers, num_blocks, num_heads, mlp_ratio=4.0, layer_scale_init=1e-4, checkpoint_activations=False):
    super().__init__()
    self.checkpoint_activations = checkpoint_activations
    self.num_heads = num_heads
    self.blocks = nn.ModuleList(
      DiTBlock(dim, num_heads, self.MODES[i % 3], mlp_ratio, layer_scale_init) for i in range(num_blocks))
    self.layer_embed = nn.Embedding(num_layers, dim)
    self.film = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
    nn.init.zeros_(self.film[-1].weight)
    nn.init.zeros_(self.film[-1].bias)
    self._rope = {}

  def _rope_tables(self, grid, device):
    key = (grid, device)
    if key not in self._rope:
      cos, sin = rope_cos_sin(grid, self.blocks[0].qkv.in_features // self.num_heads)
      self._rope[key] = (cos.to(device), sin.to(device))
    return self._rope[key]

  def forward(self, x, adaln):
    _, num_layers, num_patches, _ = x.shape
    grid = int(round(num_patches ** 0.5))
    cos, sin = self._rope_tables(grid, x.device)
    gamma, beta = self.film(self.layer_embed(torch.arange(num_layers, device=x.device))).chunk(2, dim=-1)
    gamma, beta = gamma[None, :, None], beta[None, :, None]
    adaln_mean = adaln.mean(1, keepdim=True)
    for block in self.blocks:
      if block.mode == "frame":
        x = x * (1 + gamma) + beta
      a = adaln if block.mode == "frame" else adaln_mean
      if self.checkpoint_activations and torch.is_grad_enabled():
        x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, a, use_reentrant=False)
      else:
        x = block(x, cos, sin, a)
    return x

  def forward_packed(self, x, adaln, pack):
    """forward on the kept tokens of `pack` only: x (T, D) packed rows, adaln (B, L, 6, D) -> (T, D). What forward computes for
    them when the dropped tokens are left out of every block (as keys too): the same weights, FiLM, RoPE and AdaLN per token."""
    b, num_layers = adaln.shape[:2]
    cos, sin = self._rope_tables(int(round(pack.P ** 0.5)), x.device)
    cos_r, sin_r = cos[pack.p], sin[pack.p]                           # RoPE at each row's own patch position
    gamma, beta = self.film(self.layer_embed(torch.arange(num_layers, device=x.device))).chunk(2, dim=-1)   # (L, D)
    gamma, beta = gamma[pack.l], beta[pack.l]
    adaln_frame = adaln.reshape(b * num_layers, *adaln.shape[2:])
    adaln_mean = adaln.mean(1)                                        # (B, 6, D)
    for block in self.blocks:
      if block.mode == "frame":
        x = x * (1 + gamma) + beta
        a, seg = adaln_frame, pack.frame
      else:
        a, seg = adaln_mean, pack.b
      if self.checkpoint_activations and torch.is_grad_enabled():
        x = torch.utils.checkpoint.checkpoint(block.forward_packed, x, pack, cos_r, sin_r, a, seg, use_reentrant=False)
      else:
        x = block.forward_packed(x, pack, cos_r, sin_r, a, seg)
    return x


class DiffusionHead(nn.Module):
  """(image features (B, L, P, D), the network's noisy x (B, L, C, H, W), t (B, L)) -> x0_hat (B, L, C, H, W).
  Token = [normalized features | the patch's raw noisy values | zero pad]: nothing is squeezed through a projection -- unless dim is
  set below that width (diffusion.dim): then a learned Linear maps the token to dim (DiT's patch embedding), a smaller trunk."""

  def __init__(self, feature_dim, num_layers, channels=CHANNELS, patch_size=14, dim=None, num_blocks=3, mlp_ratio=4.0,
               layer_scale_init=1e-4, t_embed_channels=256, img_token_scale=0.3, checkpoint_activations=False, cond_null="zeros"):
    """cond_null (diffusion.cond_dropout.null): what a dropped sample's image tokens become -- zeros (the z-scored features' mean;
    no parameter) | learned (a learned (feature_dim,) vector, zero-initialized: the parameter `null_cond` exists only then)."""
    super().__init__()
    if cond_null not in COND_NULLS:
      raise ValueError(f"cond_null must be one of {COND_NULLS}, got {cond_null!r}")
    if cond_null == "learned":
      self.null_cond = nn.Parameter(torch.zeros(feature_dim))
    self.patch_size, self.channels, self.img_token_scale, self.t_embed_channels = patch_size, channels, img_token_scale, t_embed_channels
    self.noise_dim = channels * patch_size ** 2
    in_dim = feature_dim + self.noise_dim
    self.dim = dim or int(math.ceil(in_dim / 64) * 64)
    if self.dim % 64:
      raise ValueError(f"diffusion.dim={self.dim} must be a multiple of 64")
    self.in_proj = nn.Linear(in_dim, self.dim) if self.dim < in_dim else None   # below the lossless width: projected
    self.pad = self.dim - in_dim if self.in_proj is None else 0
    self.time_embed = nn.Sequential(nn.Linear(t_embed_channels, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
    self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, 6 * self.dim))
    self.trunk = DiTTrunk(self.dim, num_layers, num_blocks, self.dim // 64, mlp_ratio, layer_scale_init, checkpoint_activations)
    self.out = nn.Linear(self.dim, self.noise_dim)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)

  def adaln(self, t):
    """t: (B, L) in [0, 1] -> (B, L, 6, D), in fp32 (WT computes the time embedding outside autocast)."""
    b, l = t.shape
    with torch.autocast(device_type=t.device.type, enabled=False):
      emb = self.time_embed(sinusoidal_embedding(t.reshape(-1) * 1000.0, self.t_embed_channels))
      return self.time_proj(emb).reshape(b, l, 6, self.dim)

  def forward(self, feats, x_net, t, pack=None, uncond=None):
    """pack: a PackIndex (diffusion.token_drop): the trunk and the output projection run on its kept tokens only, and the
    prediction is 0 at the dropped ones (fully empty patches: nothing reads them). None: every token (the dense path).
    uncond: (B,) bool, the samples whose image condition is replaced by the null (diffusion.cond_dropout): every token of the sample
    (all layers, all patches) gets the null instead of its z-scored, img_token_scale-scaled features; None: no sample (the call
    exactly as before)."""
    b, l, p, _ = feats.shape
    feats = (feats - feats.mean((1, 2, 3), keepdim=True)) / (feats.var((1, 2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
    if uncond is not None:   # after the per-sample z-score and the scale (a null before them would be renormalized), before packing
      null = self.null_cond.to(feats.dtype) if hasattr(self, "null_cond") else feats.new_zeros(())
      feats = torch.where(uncond.to(feats.device).bool()[:, None, None, None], null, feats * self.img_token_scale)
    patches = rearrange(x_net, "b l c (h py) (w px) -> b l (h w) (c py px)", py=self.patch_size, px=self.patch_size)
    if pack is not None and pack.T == 0:   # nothing kept (no valid slot in the batch): no output is read, the dense path keeps the
      pack = None                          # graph (a loss of 0 with zero gradients) without an empty attention
    if pack is not None:   # the feature z-scoring above stays over all tokens; everything after it is per token
      feats, patches = pack.gather(feats), pack.gather(patches)
    tokens = torch.cat([feats if uncond is not None else feats * self.img_token_scale, patches.to(feats.dtype)], dim=-1)
    if self.in_proj is not None:
      tokens = self.in_proj(tokens)
    elif self.pad:
      tokens = F.pad(tokens, (0, self.pad))
    if pack is None:
      out = self.out(self.trunk(tokens, self.adaln(t)))
    else:
      out = pack.scatter(self.out(self.trunk.forward_packed(tokens, self.adaln(t), pack)))
    grid = int(round(p ** 0.5))
    return rearrange(out, "b l (h w) (c py px) -> b l c (h py) (w px)", h=grid, w=grid, py=self.patch_size, px=self.patch_size)


class GSDiffusionModel(nn.Module):
  """The frozen image backbone + DiffusionHead, with the training loss and the sampler. Same surface as train_gs.GSModel for the code
  that renders (inputs(), forward() -> the Gaussian dict, max_sh_degree, backbone, head): forward() SAMPLES."""

  max_sh_degree = 0
  phase = 1   # diffusion.schedule.phase (set in __init__)
  decode_clamp = "channel"   # diffusion.decode_clamp (set in __init__)
  floor_add_px = None        # diffusion.scale_encoding.scale_floor_px with scale_floor_mode add (set in __init__): the decode adds it
  token_drop = "none"        # diffusion.token_drop (set in __init__)
  cond_dropout_p = 0.0       # diffusion.cond_dropout.p (set in __init__): 0 = never dropped, nothing drawn
  guidance = GUIDANCE_DEFAULTS   # diffusion.cfg, normalized (set in __init__): scale 1 = conditional only, one forward per step

  def __init__(self, cfg, stats):
    super().__init__()
    d = cfg.diffusion
    self.cfg_d = d
    if cfg.model.get("backbone") is None:
      raise SystemExit("diffusion mode needs an image backbone: model/backbone=moge_mix")
    self.num_layers = int(cfg.data.num_layers)
    self.patch_size = int(cfg.model.patch_size)
    # (configs without the keys, older runs: no dropout, no guidance)
    cond_dropout = normalize_cond_dropout(d.get("cond_dropout"))
    self.cond_dropout_p = cond_dropout["p"]
    self.guidance = normalize_guidance(d.get("cfg"))
    warn_guidance_without_dropout(self.guidance, self.cond_dropout_p)
    self.backbone = hydra.utils.instantiate(cfg.model.backbone)
    self.head = DiffusionHead(
      int(cfg.model.feature_dim), self.num_layers, patch_size=self.patch_size, dim=d.get("dim"), num_blocks=int(d.num_blocks),
      img_token_scale=float(d.img_token_scale), checkpoint_activations=bool(d.get("checkpoint_activations", False)),
      layer_scale_init=float(cfg.model.transformer.layer_scale_init), cond_null=cond_dropout["null"])
    mean, std, lo, hi = stats
    self.register_buffer("x0_mean", mean.clone().float())
    self.register_buffer("x0_std", std.clone().float())
    self.register_buffer("x0_lo", lo.clone().float())     # the training grids' range of each channel: samples are clamped into it
    self.register_buffer("x0_hi", hi.clone().float())
    self.decode_clamp = str(d.get("decode_clamp", "channel"))   # (configs without the key: every channel, as before)
    enc = normalize_encoding(dict(d.scale_encoding) if d.get("scale_encoding") is not None else None)
    self.floor_add_px = (float(enc["scale_floor_px"]) if enc["scale_floor_mode"] == "add" and enc["scale_floor_px"] is not None
                         else None)
    if self.decode_clamp not in DECODE_CLAMPS:
      raise SystemExit(f"diffusion.decode_clamp must be one of {DECODE_CLAMPS}, got {self.decode_clamp!r}")
    fixed_t = d.schedule.get("fixed_t")
    if fixed_t is not None and cfg.diffusion.get("photom") is not None and cfg.diffusion.photom.enabled:
      raise SystemExit("diffusion.schedule.fixed_t with diffusion.photom: the rendering loss's t gate never opens at t = 1, the "
                       "loss would silently be zero")
    if fixed_t is not None and (float(fixed_t) != 1.0 or int(d.steps) != 1):
      raise SystemExit("diffusion.schedule.fixed_t: only 1.0 (always pure noise: a deterministic regressor from the image, the "
                       f"same-architecture control), with diffusion.steps=1 (one step from noise = the x0 prediction); got "
                       f"fixed_t={fixed_t}, steps={d.steps}")
    self.schedule = LayerAwareSchedule(self.num_layers, standard=dict(d.schedule.standard), plateau=dict(d.schedule.plateau),
                                       fixed_t=fixed_t)
    self.phase = int(d.schedule.get("phase", 1))
    if self.phase not in (1, 2):
      raise SystemExit(f"diffusion.schedule.phase must be 1 or 2, got {self.phase}")
    self.steps = int(d.steps)
    self.eval_seed = int(d.eval_seed)
    self.token_drop = str(d.get("token_drop", "none"))   # (configs without the key: every token, as before)
    if self.token_drop not in TOKEN_DROPS:
      raise SystemExit(f"diffusion.token_drop must be one of {TOKEN_DROPS}, got {self.token_drop!r}")

  def pack(self, hit):
    """The PackIndex of diffusion.token_drop for this hit mask (B, L, H, W), or None (token_drop none: the dense trunk)."""
    if self.token_drop == "none":
      return None
    return PackIndex(token_keep(hit, self.patch_size, self.token_drop))

  def _head(self, feats, x_net, t, pack, uncond=None):
    """self.head, with the pack only when there is one (token_drop none: the head's call exactly as before), and the null-condition
    mask uncond (B,) only when there is one."""
    if uncond is not None:
      return self.head(feats, x_net, t, pack, uncond=uncond)
    return self.head(feats, x_net, t) if pack is None else self.head(feats, x_net, t, pack)

  @staticmethod
  def _uncond(condition, b, device):
    """condition (True: every sample conditioned | False: none | a (B,) bool tensor, True = that sample keeps its image) -> the
    head's uncond mask (B,) bool, or None (every sample conditioned: the head's call as before)."""
    if condition is True:
      return None
    if condition is False:
      return torch.ones(b, dtype=torch.bool, device=device)
    condition = torch.as_tensor(condition, device=device)
    if condition.shape != (b,) or condition.dtype != torch.bool:
      raise ValueError(f"condition must be True, False or a ({b},) bool tensor, got {condition.dtype} {tuple(condition.shape)}")
    return ~condition

  def predict_x0(self, image, hit, x_net, t, condition=True, feats=None):
    """The head's x0 prediction (B, L, C, H, W, normalized) for the network input x_net (B, L, C, H, W: the noisy grid, invalid
    slots already noise, as training_loss / sample build it) at timesteps t (B, L), with diffusion.token_drop's packing from hit.
    condition: True (conditional) | False (unconditional: every sample's image tokens the null) | a (B,) bool tensor (True = that
    sample conditioned). feats: the backbone's features of image, if already computed (then image is not read)."""
    feats = self._feats(image) if feats is None else feats
    a = hit[:, :, None].to(x_net.dtype)
    return self._head(feats, self._net_input(x_net, a), t, self.pack(hit), self._uncond(condition, x_net.shape[0], x_net.device))

  def _guided_x0(self, feats, x_net, t, pack, guidance):
    """The guided x0 prediction (float) of one sampler step inside the guidance interval: x0_u + scale * (x0_c - x0_u), on
    guidance["fields"] only (the other channels: the conditional prediction). Scale 0 on all fields: the unconditional forward
    alone."""
    s, fields = guidance["scale"], guidance["fields"]
    x_u = self._head(feats, x_net, t, pack, torch.ones(x_net.shape[0], dtype=torch.bool, device=x_net.device)).float()
    if s == 0.0 and fields is None:
      return x_u
    x_c = self._head(feats, x_net, t, pack).float()
    out = x_u + s * (x_c - x_u)
    if fields is not None:
      on = torch.zeros(CHANNELS, dtype=torch.bool, device=out.device)
      for f in fields:
        on[X0_SLICES[f]] = True
      out = torch.where(on[None, None, :, None, None], out, x_c)
    return out

  def inputs(self, batch, device):
    """(image, hit mask, per-volume seeds, floor context): the noise of a sample is a function of its volume (see eval_noise)."""
    seeds = [self.eval_seed + int(v) for v in batch["source_view"]]
    return batch["image"].to(device), batch["source"]["hit"].to(device), seeds, self.floor_context(batch, device)

  def floor_context(self, batch, device):
    """What decode needs of the batch for scale_floor_mode add (decode_x0's floor), else None."""
    if self.floor_add_px is None:
      return None
    src = batch["source"]
    return {"px": self.floor_add_px, "anchor_z": torch.nan_to_num(src["xyz_cam"][..., 2].to(device).float()),
            "fx": src["K_depth"][:, 0, 0].to(device).float()}

  def _feats(self, image):
    feats = self.backbone(image)
    return feats.expand(-1, self.num_layers, -1, -1) if feats.shape[1] == 1 else feats

  def normalize(self, x0_raw):
    """(B, C, L, H, W) raw -> (B, L, C, H, W) normalized."""
    x0 = x0_raw.permute(0, 2, 1, 3, 4)
    return (x0 - self.x0_mean[None, None, :, None, None]) / self.x0_std[None, None, :, None, None]

  def eval_noise(self, seeds, shape, device):
    """Per-volume reproducible randomness: the timesteps (B, L) and the noise (B, L, C, H, W) of volume i are a function of seeds[i]
    only, so a volume gets the same ones in every validation, whatever the subset or the order."""
    ts, epss = [], []
    for seed in seeds:
      ts.append(self.schedule.sample(1, phase=self.phase, generator=torch.Generator().manual_seed(int(seed))))
      gen = torch.Generator(device=device).manual_seed(int(seed))
      epss.append(torch.randn((1, *shape[1:]), device=device, generator=gen))
    return torch.cat(ts).to(device), torch.cat(epss)

  def _net_input(self, x_net, a):
    """The noisy grid the network sees. The deterministic control (schedule.fixed_t = 1) gets no noise at all, only the validity
    mask (0 at valid slots, 1 at invalid): a diffusion model can tell valid from invalid slots in its input (invalid slots are
    re-noised at every sampler step, valid ones carry the partial x0), so a regressor without the mask would be handicapped."""
    if self.schedule.fixed_t is None:
      return x_net
    return (1 - a).expand_as(x_net).to(x_net.dtype)

  def training_loss(self, image, hit, x0_raw, seeds=None, return_pred=False, t=None, condition=None):
    """The masked x0 loss. Returns (loss, parts, counts): parts = per-field and per-t-bucket detached means, counts the number of valid
    slots each is a mean over (a bucket can be empty: it is not a mean then, and must not be logged). seeds: per-volume seeds
    for the timesteps and the noise (validation); None: the global random stream. return_pred: also the prediction x0_hat
    (B, L, C, H, W, normalized, with its graph) and the timesteps t (B, L), for the rendering loss (diffusion.photom). t: these
    timesteps (B, L) instead of the schedule's (the noise is still the seeds').
    condition: None = the training's condition dropout (diffusion.cond_dropout): on the global random stream only (seeds None) with
    p > 0, each sample's image condition is replaced by the null with probability p (one draw of (B,) uniforms on x0's device, after
    the timesteps and the noise; with p = 0 nothing is drawn); seeded calls (validations) are conditional. True | False | a (B,)
    bool tensor (True = that sample conditioned): that, explicitly (e.g. the unconditional branch's validation loss). When a
    dropout was drawn, parts also holds "cond_drop" (the (B,) bool mask; not a mean: the caller pops it) and the x0 loss over
    the conditioned / dropped samples' valid slots as "cond_kept" / "cond_dropped" (counts: their slots)."""
    feats = self._feats(image)
    x0 = self.normalize(x0_raw)
    b, l = x0.shape[:2]
    if seeds is None:
      if t is not None:
        raise ValueError("training_loss: t needs seeds (the given timesteps are a validation's, with its seeded noise)")
      t = self.schedule.sample(b, phase=self.phase).to(x0.device)                     # (B, L)
      eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype)
    else:
      t_seeded, eps = self.eval_noise(seeds, x0.shape, x0.device)
      # the deterministic control (schedule.fixed_t) has no other timestep: its prediction doesn't depend on one
      t = t_seeded if t is None or self.schedule.fixed_t is not None else t.to(x0.device, t_seeded.dtype)
    t5 = t[:, :, None, None, None]
    a = hit[:, :, None].to(x0.dtype)                                                 # (B, L, 1, H, W): the validity mask
    x_net = self._net_input(a * ((1 - t5) * x0 + t5 * eps) + (1 - a) * eps, a)     # WT Eq. 7: invalid slots see the max noise
    drop = None
    if condition is None:
      if seeds is None and self.cond_dropout_p > 0:   # (p = 0: no draw, the random streams as without dropout)
        drop = torch.rand(b, device=x0.device) < self.cond_dropout_p                 # (B,) True = the null condition
      uncond = drop
    else:
      uncond = self._uncond(condition, b, x0.device)
    x0_hat = self._head(feats, x_net, t, self.pack(hit), uncond)
    err = (x0_hat.float() - x0).square()                                            # (B, L, C, H, W)
    valid = a.sum().clamp_min(1)
    loss = (err * a).sum() / (valid * CHANNELS)
    with torch.no_grad():
      parts, counts = {}, {}
      if drop is not None:
        parts["cond_drop"] = drop
        for name, sel in (("cond_kept", ~drop), ("cond_dropped", drop)):
          m = a * sel[:, None, None, None, None].to(a.dtype)
          counts[name] = m.sum()
          parts[name] = ((err * m).sum() / (counts[name].clamp_min(1) * CHANNELS)).detach()
      for name, sl in X0_SLICES.items():
        dim = sl.stop - sl.start
        parts[name] = ((err[:, :, sl] * a).sum() / (valid * dim)).detach()
        counts[name] = valid
      for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        in_bucket = a * ((t5 >= lo) & (t5 < hi)).to(a.dtype)
        name = f"t{lo:g}-{min(hi, 1):g}"
        counts[name] = in_bucket.sum()
        parts[name] = ((err * in_bucket).sum() / (counts[name].clamp_min(1) * CHANNELS)).detach()
    if return_pred:
      return loss, parts, counts, x0_hat, t
    return loss, parts, counts

  @torch.no_grad()
  def xt_probe(self, image, hit, x0_raw, seeds, ts, gain_delta=0.01):
    """The x_t reliance probe (diffusion.probe): does the x0 prediction use the noisy input, or only the image? Per volume and per
    shared timestep t in ts (every layer at t, as in the sampler), from x_t = (1 - t) x0 + t eps with the volume's own seeded noise
    (eval_noise: the validation loss's; invalid slots at the noise, through _net_input, as the model always sees them):
      err_model  the masked x0 error of the model's prediction (the head call of training_loss / sample, packing included), per
                 valid slot and channel, in normalized units
      err_copy   the same error of c(t) x_t, c = (1 - t) / ((1 - t)^2 + t^2): the best linear estimate of unit-variance data from
                 x_t alone (its error is t^2 / ((1 - t)^2 + t^2) on such data)
      gain       <x0_hat(x_t + delta) - x0_hat(x_t), delta> / |delta|^2 over the valid slots, delta = gain_delta z (z a seeded
                 standard normal on the valid slots, its own stream: not eps's): the prediction's response to its input (1 / (1 - t)
                 for "unshrink x_t", 0 for a prediction that ignores it); its two forwards in fp32 (autocast off), so a small delta
                 is not lost to bf16
    Returns {t: {"err_model", "err_copy", "gain", "valid"}}, each (B,) float32 on the device (valid: the volume's valid slots; a
    volume without any has NaN values). The volumes run one at a time (bounded memory, and each volume's values do not depend on
    the batch it came in); nothing draws from a global random stream."""
    device = image.device
    b, l = hit.shape[:2]
    x0_all = self.normalize(x0_raw.to(device).float())                               # (B, L, C, H, W)
    out = {float(t): {k: torch.full((b,), float("nan"), device=device) for k in ("err_model", "err_copy", "gain", "valid")}
           for t in ts}
    fp32 = torch.autocast(device_type=device.type, enabled=False)
    for i in range(b):
      x0, h = x0_all[i:i + 1], hit[i:i + 1]
      feats = self._feats(image[i:i + 1])
      pack = self.pack(h)
      _, eps = self.eval_noise([seeds[i]], x0.shape, device)
      a = h[:, :, None].to(x0.dtype)                                                 # (1, L, 1, H, W)
      n = a.sum() * CHANNELS
      for t in ts:
        out[float(t)]["valid"][i] = a.sum()
      if n == 0:
        continue
      for t in ts:
        t = float(t)
        res = out[t]
        # the deterministic control (schedule.fixed_t) has no other timestep (as training_loss); its input has no x_t anyway
        t_in = torch.full((1, l), float(self.schedule.fixed_t) if self.schedule.fixed_t is not None else t, device=device)
        x_t = (1 - t) * x0 + t * eps
        head = lambda x, f=feats: self._head(f, self._net_input(a * x + (1 - a) * eps, a), t_in, pack).float()
        masked_mse = lambda pred: ((pred - x0).square() * a).sum() / n
        res["err_model"][i] = masked_mse(head(x_t))
        res["err_copy"][i] = masked_mse((1 - t) / ((1 - t) ** 2 + t ** 2) * x_t)
        gen = torch.Generator(device=device).manual_seed(
          int(np.random.SeedSequence([int(seeds[i]), round(t * 1e6), 0x7874]).generate_state(1)[0]))   # 0x7874: "xt"
        delta = gain_delta * torch.randn(x0.shape, device=device, generator=gen) * a
        with fp32:
          f32 = feats.float()
          base = head(x_t.float(), f32)
          moved = head(x_t.float() + delta, f32)
        res["gain"][i] = ((moved - base) * delta).sum() / delta.square().sum()
    return out

  @torch.no_grad()
  def sample(self, image, hit, seeds=None, steps=None, project=None, floor=None, guidance=None):
    """20 Euler steps of the flow from pure noise -> the Gaussian dict. The noise of volume i is a function of seeds[i] (default:
    eval_seed + i): reproducible. Invalid slots stay at the noise. project: applied to every step's x0 prediction before it is used
    (e.g. format_projection; the last step's x0 prediction is the output).
    guidance: classifier-free guidance (diffusion.cfg's keys; None: the model's, self.guidance). scale 1: the conditional
    prediction only (one forward per step, as without guidance); otherwise, at the steps whose t is in interval (inclusive):
    x0_u + scale * (x0_c - x0_u) on the raw (normalized) predictions of fields (null: all), before project and the decode's clamp
    (scale 0 on all fields: the unconditional prediction, one forward); outside the interval: the conditional prediction."""
    if guidance is None:
      guidance = self.guidance
    else:
      guidance = normalize_guidance(guidance)
      warn_guidance_without_dropout(guidance, self.cond_dropout_p)
    steps = steps or self.steps
    feats = self._feats(image)
    b, l = hit.shape[:2]
    seeds = seeds if seeds is not None else [self.eval_seed + i for i in range(b)]
    _, eps = self.eval_noise(seeds, (b, l, CHANNELS, *hit.shape[-2:]), image.device)
    a = hit[:, :, None].to(eps.dtype)
    x = eps
    ts = torch.linspace(1.0, 0.0, steps + 1, device=image.device)
    pack = self.pack(hit)   # the same tokens at every step
    for i in range(steps):
      t = ts[i]
      x_net = self._net_input(a * x + (1 - a) * eps, a)
      lo, hi = guidance["interval"]
      if guidance["scale"] != 1.0 and lo <= float(t) <= hi:
        x0_hat = self._guided_x0(feats, x_net, t.expand(b, l), pack, guidance)
      else:
        x0_hat = self._head(feats, x_net, t.expand(b, l), pack).float()
      if project is not None:
        x0_hat = project(x0_hat)
      x = x - (t - ts[i + 1]) * (x - x0_hat) / t.clamp(min=T_MIN_CLAMP)
    return self.decode(x, floor)

  def decode(self, x0, floor=None):
    """Normalized x0 (B, L, C, H, W) -> the Gaussian dict (differentiable inside the clamp range). floor: floor_context's (needed
    with scale_floor_mode add)."""
    if self.floor_add_px is not None and floor is None:
      raise ValueError("scale_floor_mode add: decode needs the floor context (GSDiffusionModel.floor_context)")
    return decode_x0(x0.float(), self.x0_mean, self.x0_std, self.x0_lo, self.x0_hi, clamp=self.decode_clamp, floor=floor)

  def forward(self, inputs):
    image, hit, seeds, floor = inputs
    return self.sample(image, hit, seeds, floor=floor)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAWeights:
  """Exponential moving average of a module's trainable parameters (WT: decay 0.9995 after warmup). The decay ramps up as
  min(decay, (1 + n) / (10 + n)) over the first updates. Lives on the parameters' device, fp32."""

  def __init__(self, module, decay=0.9995):
    self.decay = float(decay)
    self.updates = 0
    self.shadow = {n: p.detach().clone().float() for n, p in module.named_parameters() if p.requires_grad}
    self.stash = {}   # while applied(): the module's own tensors (a checkpoint saved then must hold these, not the average)

  @torch.no_grad()
  def update(self, module):
    self.updates += 1
    d = min(self.decay, (1 + self.updates) / (10 + self.updates))
    params = dict(module.named_parameters())
    for n, s in self.shadow.items():
      if s.device != params[n].device:   # the module moved to the GPU after the average was made
        s = self.shadow[n] = s.to(params[n].device)
      s.mul_(d).add_(params[n].detach().float(), alpha=1 - d)

  @contextlib.contextmanager
  def applied(self, module):
    """Run with the averaged weights in the module; the live weights come back afterwards. The two tensors are swapped (the
    parameter's data points at the average and the average's slot keeps the live one's), no copy: nothing is allocated."""
    params = dict(module.named_parameters())
    try:
      with torch.no_grad():
        for n, s in self.shadow.items():
          p = params[n]
          if s.device != p.device or s.dtype != p.dtype:
            s = self.shadow[n] = s.to(p.device, p.dtype)
          self.stash[n] = p.data
          p.data = s
      yield
    finally:
      with torch.no_grad():
        for n, live in self.stash.items():
          params[n].data = live
      self.stash = {}

  def state_dict(self):
    return {"decay": self.decay, "updates": self.updates, "shadow": {n: s.cpu() for n, s in self.shadow.items()}}

  def load_state_dict(self, state):
    self.updates = int(state["updates"])
    missing = sorted(set(self.shadow) - set(state["shadow"]))
    unexpected = sorted(set(state["shadow"]) - set(self.shadow))
    if missing:   # runs before Lightning loads the model's weights: the copy made at __init__, not the checkpoint's weights
      log.warning(
        "EMA checkpoint has no average for %d parameters, they keep their initial copy (not the checkpoint's weights): %s",
        len(missing), missing)
    if unexpected:
      log.warning("EMA checkpoint has %d parameters the model no longer has, dropped: %s", len(unexpected), unexpected)
    for n, s in state["shadow"].items():
      if n in self.shadow:
        self.shadow[n].copy_(s.to(self.shadow[n].device))


class SpikeSkip:
  """WT's loss-spike filter: a loss above `factor` times the running average of the accepted losses is skipped (no
  optimizer step). A non-finite loss is always skipped and never enters the average. The filter only skips spikes once
  `warmup` losses have been accepted -- counted by the filter itself and checkpointed with it, so a chained resume keeps
  its average and does not restart the warmup. factor None: off, nothing is skipped (the average of the finite losses is
  still kept, for logging)."""

  def __init__(self, factor=4.0, decay=0.99, warmup=200):
    self.factor = None if factor is None else float(factor)
    self.decay = float(decay)
    self.warmup = int(warmup)
    self.average = None   # running average of the accepted losses
    self.accepted = 0     # losses folded into it
    self.nonfinite = 0    # non-finite losses seen (all skipped)

  def check(self, value):
    """True: skip this loss. False: it is accepted and folded into the average."""
    value = float(value)
    if not math.isfinite(value):
      self.nonfinite += 1
      return self.factor is not None   # off: the step goes ahead, but the average stays finite
    if self.factor is not None and self.accepted >= self.warmup and value > self.factor * self.average:
      return True
    self.average = value if self.average is None else self.decay * self.average + (1 - self.decay) * value
    self.accepted += 1
    return False

  def state_dict(self):
    return {"average": self.average, "accepted": self.accepted, "nonfinite": self.nonfinite}

  def load_state_dict(self, state):
    # factor/decay/warmup stay the config's, so a resumed run can change them
    self.average = state["average"]
    self.accepted = int(state["accepted"])
    self.nonfinite = int(state.get("nonfinite", 0))
