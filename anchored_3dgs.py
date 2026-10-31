"""Anchored layered 3DGS: one shared Gaussian set per mesh, fitted jointly over all its training views, from which every
view's layered grid (one Gaussian per depth-peel slot, fit_3dgs.py `gaussian_layout=layered`'s format) is cut out.

Per mesh (the rows of a render_objaverse.py h5 with one `mesh_index`):
  - slots: every training view's depth-peel hits (unproject_depth_peel order), with their footprint (one depth pixel's width at
    the hit, z / f), front-pixel colour and sample spacing (footprint / max(|cos(normal, view ray)|, 0.2))
  - merge (cluster_slots): collision-free centroid clustering of all slots into anchors -- an anchor never holds two slots of
    one view, so a view's slots map 1:1 onto its anchors (its "subset")
  - combined fit (fit_combined): per-anchor offset (free 3D, from the anchor = its slots' mean), quaternion, SH0 colour, and
    `levels` density levels of log-scale + opacity logit; a view renders its subset at, per anchor, the level of its own spacing
    relative to the anchor's mean spacing (view_levels). Each step: a random training view's subset, rendered into an independent
    random training camera, 0.8 L1 + 0.2 D-SSIM + offset_reg_weight x mean |offset|^2 over the subset; plain dense Adam over all
    anchors; the offset lr decays exponentially offset_lr_start -> offset_lr_end (x base) over the run.
  - per-view fine-tune (finetune): the view's subset at its levels, flattened to one Gaussian per slot, from the combined fit's
    parameters (never from another view's fine-tune: anchors are shared), fresh optimizer, a few hundred steps with the cameras of
    all training views.
  - outputs (write_view): <root>/mesh<M>_view<V>/ with gaussians.h5 / gaussians.ply / gaussians.grid.h5, written by fit_3dgs.py's
    own writers, so finalize_fit_dataset.py and gs_diffusion.encode_grid take them unchanged.

The Hydra entry point (stages, resume, config) is fit_anchored.py; this module is the per-stage library.
"""

import hashlib
import json
import logging
import os
import shutil

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from debug_pointcloud import unproject_depth_peel
from fit_3dgs import render, save_output, save_output_layered, write_ply
from module import DSSIMLoss, OPENGL_TO_OPENCV, SH_C0

log = logging.getLogger(__name__)

PARAM_KEYS = ("offset", "scales", "quats", "opacities", "sh0")
LEVELED = ("scales", "opacities")   # one entry per density level and anchor


# ---------------------------------------------------------------------------
# views / slots
# ---------------------------------------------------------------------------

def mesh_rows(hdf5_path, mesh_index):
  """The h5 rows of one mesh, sorted (its local view v is rows[v])."""
  with h5py.File(hdf5_path, "r") as f:
    rows = np.flatnonzero(np.asarray(f["mesh_index"]) == int(mesh_index))
  if not len(rows):
    raise SystemExit(f"{hdf5_path}: no rows with mesh_index {mesh_index}")
  return [int(r) for r in rows]


def split_views(n, val_fraction, held_views):
  """(train, excluded) local view indices. held_views (regression mode) replaces the validation split; otherwise the last
  round(val_fraction x n) views are excluded."""
  if held_views is not None:
    held = sorted(int(v) for v in held_views)
    if any(not 0 <= v < n for v in held):
      raise SystemExit(f"regression.held_views {held}: the mesh has {n} views")
  else:
    n_val = int(round(float(val_fraction) * n))
    held = list(range(n - n_val, n))
  train = [v for v in range(n) if v not in held]
  if not train:
    raise SystemExit("no training views left")
  return train, held


def scene_extent_of(poses):
  """fit_3dgs.load_scene_h5's scene_extent: 1.1 x the largest distance of a training camera from their mean."""
  centers = poses[:, :3, 3]
  return float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) * 1.1) or 1.0


def load_view(f, row):
  """One h5 row's slots + camera. Slots in unproject_depth_peel order: points (N,3) world, u/v/layer (depth-peel pixel / layer),
  foot (N,), the front image pixel's RGB (N,3) in [0,1] (not premultiplied), spacing (N,). Also the premultiplied GT image and
  the camera (OpenCV world-to-camera viewmat, image K) and the raw depth peel (for the grid's layer_valid check)."""
  depth = np.asarray(f["depth_peel"][row])                        # (H,W,L) f32
  Kd = np.asarray(f["depth_intrinsics"][row])
  pose = np.asarray(f["camera_pose"][row]).astype(np.float32)
  image = np.asarray(f["images"][row])                            # (IH,IW,3|4) u8
  pts, u, v, layer = unproject_depth_peel(depth, Kd, pose, "world")
  foot = depth[v, u, layer] / np.sqrt(Kd[0, 0] * Kd[1, 1])
  # the image pixel containing the depth pixel's centre (module.init_gaussians' mapping; RGB and depth may differ in size)
  dh, dw = depth.shape[:2]
  ih, iw = image.shape[:2]
  iv = np.clip(np.floor((v + 0.5) * (ih / dh)), 0, ih - 1).astype(np.int64)
  iu = np.clip(np.floor((u + 0.5) * (iw / dw)), 0, iw - 1).astype(np.int64)
  color = image[iv, iu, :3].astype(np.float32) / 255
  if "depth_peel_normals" in f:
    n = np.asarray(f["depth_peel_normals"][row])[v, u, layer].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    ray = (pts - pose[:3, 3]) / np.linalg.norm(pts - pose[:3, 3], axis=1, keepdims=True)
    cos = np.nan_to_num(np.abs((n * ray).sum(1)), nan=1.0)        # no normal: head-on
  else:
    log.warning("row %d: no depth_peel_normals -- spacing = footprint", row)
    cos = np.ones(len(pts))
  spacing = foot / np.maximum(cos, 0.2)
  img = image.astype(np.float32) / 255
  rgb, alpha = (img[..., :3], img[..., 3:4]) if img.shape[-1] == 4 else (img, np.ones_like(img[..., :1]))
  return {
    "row": int(row), "points": pts, "u": u, "v": v, "layer": layer, "foot": foot, "color": color, "spacing": spacing,
    "gt": np.ascontiguousarray(rgb * alpha), "viewmat": np.linalg.inv(pose @ OPENGL_TO_OPENCV).astype(np.float32),
    "K": np.asarray(f["image_intrinsics"][row]).astype(np.float32), "pose": pose, "valid": depth > 0,
  }


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def cluster_slots(P, F, V, r=1.0, k=8, max_rounds=200, verbose=True):
  """Collision-free merge of slots into anchors: centroid-linkage clustering with a cannot-link constraint (an anchor never holds
  two slots of the same view), in rounds of mutual-nearest-neighbour merges:
    - every slot starts as its own cluster (count 1, view set = its view)
    - each round: for every cluster, its nearest valid partner among its k nearest clusters (KD-tree on the current means): no
      shared view, means within r x min(the two clusters' mean footprints), and the merged cluster's RMS spread around its new
      mean <= r x that footprint; pairs that are each other's partner merge (sums of positions / footprints / squared norms add)
    - until a round merges nothing (or max_rounds)
  Every slot keeps weight 1, so an anchor's position is the plain mean of its slots, independent of merge order.
  P (N,3) positions, F (N,) footprints, V (N,) view ids (any non-negative ints; a cluster's view set is a bitmask of
  ceil((max+1) / 64) uint64 words, so any number of views works). Returns (labels (N,) slot -> anchor, anchor means, counts,
  mean footprints)."""
  n = len(P)
  V = np.asarray(V, np.int64)
  label = np.arange(n)                      # slot -> current cluster id (ids index the arrays below)
  S = P.astype(np.float64).copy()           # sum of positions
  Q = (P.astype(np.float64) ** 2).sum(1)    # sum of squared norms (for the RMS spread)
  Fs = F.astype(np.float64).copy()          # sum of footprints
  C = np.ones(n, np.int64)                  # counts
  M = np.zeros((n, int(V.max()) // 64 + 1 if n else 1), np.uint64)   # view bitmasks, 64 views per word
  M[np.arange(n), V // 64] = np.uint64(1) << (V % 64).astype(np.uint64)
  alive = np.arange(n)
  for rnd in range(max_rounds):
    if len(alive) < 2:
      break
    mu = S[alive] / C[alive, None]
    fo = Fs[alive] / C[alive]
    kk = min(k + 1, len(alive))
    dist, nb = cKDTree(mu).query(mu, k=kk, workers=-1)
    dist, nb = dist[:, 1:], nb[:, 1:]       # drop self
    a = np.arange(len(alive))[:, None]
    lim = r * np.minimum(fo[:, None], fo[nb])
    Ma = M[alive]
    ok = (dist <= lim) & ~(Ma[:, None, :] & Ma[nb]).any(-1)
    cn = C[alive][:, None] + C[alive][nb]
    Sn = S[alive][:, None, :] + S[alive][nb]
    Qn = Q[alive][:, None] + Q[alive][nb]
    rms2 = Qn / cn - ((Sn / cn[..., None]) ** 2).sum(-1)
    ok &= rms2 <= lim ** 2
    has = ok.any(1)
    first = np.argmax(ok, 1)                # neighbours are sorted by distance: first valid = nearest valid
    partner = np.where(has, nb[a[:, 0], first], -1)
    mutual = has & (partner >= 0) & (partner[np.clip(partner, 0, None)] == np.arange(len(alive)))
    i = np.flatnonzero(mutual & (np.arange(len(alive)) < partner))
    if i.size == 0:
      break
    gi, gj = alive[i], alive[partner[i]]   # merge cluster gj into gi
    S[gi] += S[gj]; Q[gi] += Q[gj]; Fs[gi] += Fs[gj]; C[gi] += C[gj]; M[gi] |= M[gj]
    remap = np.arange(n)
    remap[gj] = gi
    label = remap[label]
    dead = np.zeros(n, bool); dead[gj] = True
    alive = alive[~dead[alive]]
    if verbose and (rnd < 5 or rnd % 10 == 0):
      log.info("  merge round %d: %d pairs, %d clusters", rnd, i.size, len(alive))
  ids, label = np.unique(label, return_inverse=True)
  label = label.ravel()
  # cannot-link holds: no anchor has two slots of one view
  assert len(np.unique(label * (int(V.max()) + 1) + V)) == n, "an anchor holds two slots of one view"
  return label, S[ids] / C[ids, None], C[ids], Fs[ids] / C[ids]


def build_anchors(views, r, n_levels, k=8, max_rounds=200):
  """Merges the training views' slots (views: load_view dicts, in training-view order) into anchors. Returns a dict: labels
  (all slots, views concatenated), slot_counts (per view), anchor pos / foot / spacing (the means over its slots) / color (its
  first slot's, in view order), and levels (per slot: the density level its view uses at its anchor)."""
  P = np.concatenate([w["points"] for w in views])
  F = np.concatenate([w["foot"] for w in views])
  Vi = np.concatenate([np.full(len(w["points"]), j) for j, w in enumerate(views)])
  labels, _, cnt, _ = cluster_slots(P, F, Vi, r=r, k=k, max_rounds=max_rounds, verbose=True)
  n_anchor = int(labels.max()) + 1
  pos = np.stack([np.bincount(labels, weights=P[:, i], minlength=n_anchor) for i in range(3)], 1) / cnt[:, None]
  foot = np.bincount(labels, weights=F, minlength=n_anchor) / cnt
  sp_all = np.concatenate([w["spacing"] for w in views])
  spacing = np.bincount(labels, weights=sp_all, minlength=n_anchor) / cnt
  first = np.full(n_anchor, -1)
  first[labels[::-1]] = np.arange(len(labels))[::-1]           # each anchor's first slot
  color = np.concatenate([w["color"] for w in views])[first]
  # density level per slot: log2 of its view's spacing over the anchor's mean spacing, rounded, centred on the middle level
  # (float32 like the prototype's torch path, so ties round the same way)
  ratio = np.log(sp_all / spacing[labels]).astype(np.float32)
  levels = np.clip(np.round(ratio / np.float32(np.log(2))) + (n_levels - 1) // 2, 0, n_levels - 1).astype(np.int64)
  counts = np.array([len(w["points"]) for w in views], np.int64)
  log.info("%d slots of %d views -> %d anchors (r %.3g); a view uses %.0f%% of them on average; slots per anchor median %d",
           len(P), len(views), n_anchor, r, 100 * counts.mean() / n_anchor, int(np.median(cnt)))
  return {"labels": labels, "slot_counts": counts, "pos": pos, "foot": foot, "spacing": spacing, "color": color,
          "levels": levels}


def per_view(arr, slot_counts):
  """Splits a concatenated per-slot array into the views' pieces."""
  return np.split(arr, np.cumsum(slot_counts)[:-1])


def labels_digest(labels):
  return hashlib.sha1(np.ascontiguousarray(labels, np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# parameters / rendering
# ---------------------------------------------------------------------------

def init_params(anchors, n_levels, init_opacity, device):
  """Per-anchor parameters: offset 0, isotropic scale = the anchor's footprint, levels spaced one octave apart around it,
  identity rotation, opacity init_opacity, SH0 from the first slot's colour."""
  foot = torch.tensor(anchors["foot"], device=device, dtype=torch.float32)
  n = len(foot)
  lev = (torch.arange(n_levels, device=device, dtype=torch.float32) - (n_levels - 1) / 2) * float(np.log(2))
  return {
    "offset": torch.zeros(n, 3, device=device),
    "scales": torch.log(foot)[:, None, None].repeat(1, n_levels, 3) + lev[None, :, None],
    "quats": torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1),
    "opacities": torch.full((n, n_levels), float(np.log(init_opacity / (1 - init_opacity))), device=device),
    "sh0": ((torch.tensor(anchors["color"], device=device, dtype=torch.float32) - 0.5) / SH_C0)[:, None, :],
  }


def gather(params, anchor_pos, idx, lev):
  """A subset's flat render parameters (fit_3dgs.render's dict) from per-anchor params: anchor idx at level lev."""
  return {
    "means": anchor_pos[idx] + params["offset"][idx], "quats": params["quats"][idx],
    "scales": params["scales"][idx, lev], "opacities": params["opacities"][idx, lev],
    "sh0": params["sh0"][idx], "shN": params["sh0"].new_zeros(len(idx), 0, 3),
  }


def view_params(params, anchor_pos, labels_v, levels_v):
  """One view's subset at its levels, flattened to one Gaussian per slot (slot order; collision-free merge -> 1:1): a dict of
  detached copies, means_original (the anchor, frozen) + offset."""
  with torch.no_grad():
    return {
      "means_original": anchor_pos[labels_v].clone(), "offset": params["offset"][labels_v].clone(),
      "scales": params["scales"][labels_v, levels_v].clone(), "quats": params["quats"][labels_v].clone(),
      "opacities": params["opacities"][labels_v, levels_v].clone(), "sh0": params["sh0"][labels_v].clone(),
    }


def flat_render_params(flat):
  return {"means": flat["means_original"] + flat["offset"], "quats": flat["quats"], "scales": flat["scales"],
          "opacities": flat["opacities"], "sh0": flat["sh0"], "shN": flat["sh0"].new_zeros(len(flat["sh0"]), 0, 3)}


def flat_numpy(flat):
  """fit_3dgs.py's flat parameter layout (save_output / write_ply / save_output_layered): means world, raw log-scales (N,3),
  quats (N,4), opacity logits (N,), sh0 (N,1,3), shN (N,0,3)."""
  p = flat_render_params(flat)
  return {k: v.detach().float().cpu().numpy() for k, v in p.items()}


class Cameras:
  """The GPU-resident targets of a set of views: premultiplied GT (V,H,W,3), viewmats, image Ks."""

  def __init__(self, views, device):
    self.gt = torch.tensor(np.stack([w["gt"] for w in views]), device=device)
    self.viewmat = torch.tensor(np.stack([w["viewmat"] for w in views]), device=device)
    self.K = torch.tensor(np.stack([w["K"] for w in views]), device=device)
    self.H, self.W = self.gt.shape[1:3]

  def __len__(self):
    return len(self.gt)

  def render(self, p, i):
    rgb, _, _ = render(p, 0, self.viewmat[[i]], self.K[[i]], self.W, self.H)
    return rgb


def photometric(dssim, rgb, gt, lambda_dssim):
  """(1 - lambda) L1 + lambda D-SSIM, rgb (1,H,W,3), gt (H,W,3)."""
  l1 = (rgb - gt[None]).abs().mean()
  ds = dssim(rgb.permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2))
  return (1 - lambda_dssim) * l1 + lambda_dssim * ds


@torch.no_grad()
def heldout_psnr(p, cams):
  """Mean over cams of the full-image PSNR (render clamped to [0,1], GT premultiplied over black, no background) -- fit_3dgs's
  val/PSNR."""
  return float(np.mean([float(-10 * torch.log10((cams.render(p, i)[0] - cams.gt[i]).pow(2).mean())) for i in range(len(cams))]))


def exp_factor(it, steps, start, end):
  """lr factor at 1-indexed step it: exponential start -> end over steps."""
  return start * (end / start) ** ((it - 1) / max(1, steps - 1))


def make_adam(params, lrs):
  return {k: torch.optim.Adam([params[k]], lr=float(lrs[k]), eps=1e-15) for k in params}


# ---------------------------------------------------------------------------
# optimization
# ---------------------------------------------------------------------------

def fit_combined(params, anchors_t, cams, cfg, scene_extent, rng, log_step=None):
  """The combined fit, in place on `params` (per-anchor, requires_grad). anchors_t: {"pos": (A,3), "labels": [per-view slot ->
  anchor (N_v,)], "levels": [per-view (N_v,)]} on the device, views indexed like cams. Each step: subset view and camera drawn
  independently from the training views. Returns (photometric, total) per-step losses (numpy)."""
  steps = int(cfg.combined.steps)
  lrs = {k: float(cfg.lr[k]) for k in PARAM_KEYS}
  lrs["offset"] *= scene_extent
  opt = make_adam(params, lrs)
  dssim = DSSIMLoss().to(cams.gt.device)
  views = list(range(len(cams)))
  photo_log, loss_log = [], []   # detached GPU scalars, copied to the host once at the end (no per-step sync)
  for it in range(1, steps + 1):
    sv = int(rng.choice(views))
    cam = int(rng.choice(views))
    idx, lev = anchors_t["labels"][sv], anchors_t["levels"][sv]
    rgb = cams.render(gather(params, anchors_t["pos"], idx, lev), cam)
    photo = photometric(dssim, rgb, cams.gt[cam], float(cfg.loss.lambda_dssim))
    opt["offset"].param_groups[0]["lr"] = lrs["offset"] * exp_factor(
      it, steps, float(cfg.combined.offset_lr_start), float(cfg.combined.offset_lr_end))
    loss = photo + float(cfg.loss.offset_reg_weight) * params["offset"][idx].pow(2).sum(-1).mean()
    for o in opt.values():
      o.zero_grad(set_to_none=True)
    loss.backward()
    for o in opt.values():
      o.step()
    photo_log.append(photo.detach()); loss_log.append(loss.detach())
    if log_step is not None:
      log_step(it, photo.detach(), loss.detach())
  return torch.stack(photo_log).cpu().numpy(), torch.stack(loss_log).cpu().numpy()


def finetune(flat, cams, cfg, scene_extent, rng, log_step=None):
  """Per-view fine-tune of a flattened subset (view_params; offset/scales/quats/opacities/sh0 trained, means_original frozen),
  in place, fresh Adam; cameras drawn from all training views; offset lr exponential pos_lr_start -> pos_lr_end (x the combined
  base). Returns (photometric, total) per-step losses (numpy)."""
  steps = int(cfg.finetune.steps)
  params = {k: flat[k].requires_grad_(True) for k in PARAM_KEYS}
  lrs = {k: float(cfg.lr[k]) for k in PARAM_KEYS}
  lrs["offset"] *= scene_extent
  opt = make_adam(params, lrs)
  dssim = DSSIMLoss().to(cams.gt.device)
  views = list(range(len(cams)))
  photo_log, loss_log = [], []   # detached GPU scalars, copied to the host once at the end (no per-step sync)
  for it in range(1, steps + 1):
    cam = int(rng.choice(views))
    rgb = cams.render(flat_render_params(flat), cam)
    photo = photometric(dssim, rgb, cams.gt[cam], float(cfg.loss.lambda_dssim))
    opt["offset"].param_groups[0]["lr"] = lrs["offset"] * exp_factor(
      it, steps, float(cfg.finetune.pos_lr_start), float(cfg.finetune.pos_lr_end))
    loss = photo + float(cfg.loss.offset_reg_weight) * flat["offset"].pow(2).sum(-1).mean()
    for o in opt.values():
      o.zero_grad(set_to_none=True)
    loss.backward()
    for o in opt.values():
      o.step()
    photo_log.append(photo.detach()); loss_log.append(loss.detach())
    if log_step is not None:
      log_step(it, photo.detach(), loss.detach())
  for k in PARAM_KEYS:
    flat[k] = flat[k].detach()
  return torch.stack(photo_log).cpu().numpy(), torch.stack(loss_log).cpu().numpy()


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def view_done(root, name):
  """A view's grid already exists: in the run dir, or collected by finalize_fit_dataset.py into grids/. A run-dir fit
  that is not an anchored one (e.g. a fit_3dgs.py fit in the same root) is an error, not a finished view."""
  run = os.path.join(root, name)
  if os.path.exists(os.path.join(run, "gaussians.grid.h5")):
    with h5py.File(os.path.join(run, "gaussians.h5"), "r") as f:
      if "anchored" not in json.loads(f.attrs["config_json"]):
        raise SystemExit(f"{run} holds a fit that is not an anchored one -- use a separate output_dir")
    return True
  return os.path.exists(os.path.join(root, "grids", name + ".h5"))


def write_view(root, name, params_np, view, meta, config, scene_extent, final_loss, losses=None):
  """<root>/<name>/ in fit_3dgs.py's layered format: gaussians.h5 (flat raw params + attrs, config_json = `config`),
  gaussians.ply, gaussians.grid.h5 (save_output_layered; layer_valid == the view's depth_peel > 0, checked), + losses.npz.
  Written into <root>/.tmp_<name> (a name finalize_fit_dataset.py skips) and renamed into place."""
  if not all(np.isfinite(v).all() for v in params_np.values()):
    raise RuntimeError(f"{name}: non-finite parameters")
  tmp = os.path.join(root, ".tmp_" + name)
  if os.path.exists(tmp):
    shutil.rmtree(tmp)
  os.makedirs(tmp)
  grid_meta = {"u": view["u"], "v": view["v"], "layer": view["layer"], "H": view["valid"].shape[0],
               "W": view["valid"].shape[1], "L": view["valid"].shape[2], **meta}
  save_output(os.path.join(tmp, "gaussians.h5"), OmegaConf.create(config), params_np, scene_extent, final_loss,
              len(params_np["means"]))
  write_ply(os.path.join(tmp, "gaussians.ply"), params_np)
  grid_path = os.path.join(tmp, "gaussians.grid.h5")
  save_output_layered(grid_path, params_np, grid_meta)
  with h5py.File(grid_path, "r") as g:
    if not np.array_equal(np.asarray(g["layer_valid"]), view["valid"]):
      raise RuntimeError(f"{name}: layer_valid != depth_peel > 0")
  if losses is not None:
    np.savez(os.path.join(tmp, "losses.npz"), **losses)
  verify_written(tmp)
  final = os.path.join(root, name)
  if os.path.exists(final):   # an older, grid-less dir (view_done was false): replaced
    shutil.rmtree(final)
  os.replace(tmp, final)


def verify_written(d):
  """fsync every file in d (surfaces delayed write errors, e.g. a full disk, that close() did not report) and read
  every h5 dataset back, so a truncated file is never renamed into place as a finished view."""
  min_free = 2 << 30
  if shutil.disk_usage(d).free < min_free:
    raise RuntimeError(f"{d}: less than {min_free >> 30} GB free on the output disk")
  for f in sorted(os.listdir(d)):
    p = os.path.join(d, f)
    fd = os.open(p, os.O_RDONLY)
    try:
      os.fsync(fd)
    finally:
      os.close(fd)
    if f.endswith(".h5"):
      with h5py.File(p, "r") as h:
        h.visititems(lambda _, obj: obj[...] if isinstance(obj, h5py.Dataset) else None)
        dict(h.attrs)


def save_npz_atomic(path, **arrays):
  tmp = path + ".tmp.npz"
  np.savez(tmp, **arrays)
  os.replace(tmp, path)


def write_json_atomic(path, obj):
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(obj, f, indent=1)
  os.replace(tmp, path)
