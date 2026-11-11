"""Learnable x0 grids (diffusion.learn_grids): the diffusion model's training data -- one layered 3DGS grid per training volume --
as parameters, optimized only THROUGH the model's losses on its x0 prediction (the rendering loss; the x0 loss too with
x0_grad through_input).

Each training volume's grid is its valid slots' raw x0 channels (gs_diffusion.encode_grid's 14 channels, CHANNELS order, in the raw
encoding: no sort, no floor, quaternions as fitted; the scale channels log-scales or, with scale_param cumulative, (a, b, c): see
gs_diffusion.log_to_cumulative), stored sparsely: one (N, d) tensor per channel group (X0_FIELDS) over the rows
of all volumes, a volume's rows in np.nonzero(hit) order (layer, y, x: the grid files' order). A training step scatters a batch's
rows into the dense (B, CHANNELS, L, H, W) grid the model normalizes and noises; the gradient reaches the rows through the noisy
input x_t = (1 - t) x0 + t eps only (the x0 target is the detached copy), so at t = 1 it is zero.

init: pretrained (the fitted grid files, encoded as the dataset encodes them) | fresh (fresh_rows: what anchored_3dgs.init_params
gives one slot -- offset 0, isotropic scale = the slot's pixel footprint, identity rotation, opacity 0.1, the slot's colour).
The offsets are stored divided by the volume's scene extent (the anchored fit's, see scene_extent), so ONE Adam learning rate
lr.offset on them is the fitter's lr.offset x scene_extent on the offsets for every volume (Adam's step is invariant to a constant
rescaling of a parameter, up to eps).

Also the grids' parameter statistics (param_stats) logged with the validations, and the dump of the final grids to grid files
(to_grid)."""

import json
import logging

import h5py
import numpy as np
import torch
import torch.nn as nn

from gaussians import SH_C0, X0_CHANNELS, X0_FIELDS, X0_SLICES
from grid_io import Grid
from gs_diffusion import cumulative_to_log, log_to_cumulative

log = logging.getLogger(__name__)

INITS = ("pretrained", "fresh")
X0_GRADS = ("through_input", "none")   # diffusion.learn_grids.x0_grad
COLORS = ("peel_rgb", "image")
# the anchored fitter's parameter groups (conf/fit_anchored.yaml lr.*) for each x0 channel group
LR_KEYS = {"opacity": "opacities", "scale": "scales", "rotation": "quats", "sh_dc": "sh0", "offset": "offset"}


def learn_grids_config(cfg):
  """cfg.diffusion.learn_grids when it is enabled, else None (a config without the key: off)."""
  d = cfg.get("diffusion")
  lg = d.get("learn_grids") if d is not None else None
  return lg if lg is not None and bool(lg.get("enabled", False)) else None


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def read_colors(h5, row, color, num_layers):
  """The colour of every depth-peel slot of render row `row`, (L, H, W, 3) float32 in [0, 1]. peel_rgb: the layer's own surface
  colour (the render's depth_peel_rgb, its diffuse-colour pass); image: the source image's pixel over the depth pixel's centre
  (anchored_3dgs.load_view's mapping), the same for every layer."""
  if color == "peel_rgb":
    if "depth_peel_rgb" not in h5:
      raise SystemExit(f"diffusion.learn_grids.fresh.color=peel_rgb: {h5.filename} has no depth_peel_rgb (use color=image)")
    return np.ascontiguousarray(np.asarray(h5["depth_peel_rgb"][row]).transpose(2, 0, 1, 3)).astype(np.float32) / 255
  if color != "image":
    raise SystemExit(f"diffusion.learn_grids.fresh.color must be one of {COLORS}, got {color!r}")
  dh, dw = h5["depth_peel"].shape[1:3]
  image = np.asarray(h5["images"][row])
  ih, iw = image.shape[:2]
  iv = np.clip(np.floor((np.arange(dh) + 0.5) * (ih / dh)), 0, ih - 1).astype(np.int64)
  iu = np.clip(np.floor((np.arange(dw) + 0.5) * (iw / dw)), 0, iw - 1).astype(np.int64)
  rgb = image[iv][:, iu, :3].astype(np.float32) / 255                    # (H, W, 3)
  return np.broadcast_to(rgb, (num_layers, dh, dw, 3)).copy()


def log_scales(rows, scale_param="log"):
  """The (n, 3) log-scales of raw x0 rows (n, CHANNELS) (numpy or torch): the scale channels, or with scale_param cumulative their
  cumulative_to_log."""
  s = rows[:, X0_SLICES["scale"]]
  return cumulative_to_log(s) if scale_param == "cumulative" else s


def fresh_rows(source, colors, opacity=0.1, scale_param="log"):
  """anchored_3dgs.init_params for one slot each, as raw x0 rows (N, CHANNELS) float32 in np.nonzero(hit) order: opacity logit of
  `opacity`, the 3 log-scales log(footprint) (footprint = one depth pixel at the slot's depth, z / fx: an isotropic Gaussian a pixel
  wide), the identity quaternion, sh_dc (colour - 0.5) / SH_C0, offset 0 (the mean on the depth-peel hit). source: an item's
  "source" dict; colors (L, H, W, 3) (read_colors). scale_param cumulative: the isotropic scale as (a, b, c) = (log footprint,
  softplus^-1(CUM_GAP_FLOOR) twice): the axes 1e-4 (in log) apart, the encode's floor for tied axes."""
  hit = np.asarray(source["hit"])
  li, yi, xi = np.nonzero(hit)
  z = np.asarray(source["xyz_cam"])[li, yi, xi, 2]
  foot = z / float(np.asarray(source["K_depth"])[0, 0])
  rows = np.zeros((len(li), X0_CHANNELS), np.float32)
  rows[:, X0_SLICES["opacity"]] = np.log(opacity / (1 - opacity))
  rows[:, X0_SLICES["scale"]] = np.log(foot)[:, None]
  if scale_param == "cumulative":
    rows[:, X0_SLICES["scale"]] = log_to_cumulative(rows[:, X0_SLICES["scale"]].astype(np.float64))
  rows[:, X0_SLICES["rotation"].start] = 1.0
  rows[:, X0_SLICES["sh_dc"]] = (colors[li, yi, xi] - 0.5) / SH_C0
  return rows


def scene_extent(grid_path, poses_of):
  """The scene extent the anchored fit of a grid used for its offset learning rate (its config_json's anchored.scene_extent); a grid
  without it: anchored_3dgs.scene_extent_of's formula (1.1 x the largest distance of a camera from their mean) over poses_of() (the
  camera-to-world poses of the mesh's views in the training pool, read only then)."""
  with h5py.File(grid_path, "r") as f:
    cfg = json.loads(f.attrs["config_json"]) if "config_json" in f.attrs else {}
  value = (cfg.get("anchored") or {}).get("scene_extent")
  if value is not None:
    return float(value)
  centers = np.asarray(poses_of())[:, :3, 3]
  return float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) * 1.1) or 1.0


def neighbour_pairs(hit, start):
  """(P, 2) int64 row indices (offset by start) of the valid slots that are horizontal or vertical neighbours in one layer."""
  idx = np.full(hit.shape, -1, np.int64)
  idx[hit] = np.arange(int(hit.sum())) + start
  pairs = []
  for a, b in ((idx[:, :, :-1], idx[:, :, 1:]), (idx[:, :-1, :], idx[:, 1:, :])):
    ok = (a >= 0) & (b >= 0)
    pairs.append(np.stack([a[ok], b[ok]], 1))
  return np.concatenate(pairs)


def build_init(lg, inner, grid_index, encoding, encode_grid, read_grid):
  """The init of every volume of the training dataset `inner` (a WTFeatureDataset), in source-view order: {"views" [source view
  rows], "mesh" [mesh ids], "rows" [(N_i, CHANNELS) raw x0], "ref_rows" (fresh init: the fitted grids' rows, the reference;
  pretrained: None, the init is the reference), "extent" [floats], "foot" [(N_i,)], "pairs" (P, 2), "hit" [(L, H, W) bool],
  "anchor" [(N_i, 3) camera-frame depth-peel points], "pose" [(4, 4) pose_gl]}. lg: diffusion.learn_grids; encoding: the
  dataset's (the fitted rows are encode_grid's, the values the dataset gives, in float32)."""
  init = str(lg.init)
  if init not in INITS:
    raise SystemExit(f"diffusion.learn_grids.init must be one of {INITS}, got {init!r}")
  saved = inner.num_views, inner.image_size
  inner.num_views, inner.image_size = 0, None   # only the source view's point cloud
  out = {k: [] for k in ("views", "mesh", "rows", "ref_rows", "extent", "foot", "hit", "anchor", "pose")}
  pairs, start = [], 0
  try:
    order = np.argsort(np.asarray(inner.volumes.df["view_idx"].to_list()), kind="stable")
    h5 = h5py.File(inner.source_h5, "r") if init == "fresh" else None
    for i in order.tolist():
      item = inner[i]
      v, src = int(item["source_view"]), item["source"]
      hit = src["hit"].numpy()
      fitted = encode_grid(read_grid(grid_index[v]), src, rng=np.random.default_rng(v), **encoding)[:, hit].T.copy()
      if init == "pretrained":
        rows = fitted
      else:
        rows = fresh_rows({k: src[k].numpy() for k in ("hit", "xyz_cam", "K_depth")},
                          read_colors(h5, v, str(lg.fresh.color), hit.shape[0]), float(lg.fresh.opacity),
                          encoding.get("scale_param", "log"))
        out["ref_rows"].append(np.ascontiguousarray(fitted, np.float32))
      anchor = src["xyz_cam"].numpy()[hit]
      out["views"].append(v)
      out["mesh"].append(int(item["mesh_index"]))
      out["rows"].append(np.ascontiguousarray(rows, np.float32))
      out["extent"].append(scene_extent(grid_index[v], lambda m=int(item["mesh_index"]): _pool_poses(inner, m)))
      out["foot"].append((anchor[:, 2] / float(src["K_depth"][0, 0])).astype(np.float32))
      out["hit"].append(hit)
      out["anchor"].append(anchor.astype(np.float32))
      out["pose"].append(src["pose_gl"].numpy())
      pairs.append(neighbour_pairs(hit, start))
      start += len(rows)
    if h5 is not None:
      h5.close()
  finally:
    inner.num_views, inner.image_size = saved
  out["pairs"] = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
  if not out["ref_rows"]:   # pretrained: the reference (the fitted grids) is the init itself
    out["ref_rows"] = None
  log.info("learn_grids: %d volumes (%s init), %d valid slots (%.1f MB of float32 rows)", len(out["views"]), init, start,
           start * X0_CHANNELS * 4 / 2 ** 20)
  return out


def _pool_poses(inner, mesh_id):
  """The camera-to-world poses of the mesh's views in the dataset's pool (scene_extent's fallback)."""
  with h5py.File(inner.source_h5, "r") as f:
    return np.stack([np.asarray(f["camera_pose"][v]) for v in sorted(inner.pool[mesh_id])])


# ---------------------------------------------------------------------------
# the parameters
# ---------------------------------------------------------------------------

class LearnableGrids(nn.Module):
  """The grids of the training volumes (build_init's), as one (N, d) tensor per channel group over all volumes' rows: Parameters
  when learnable, else (frozen: the controls) plain buffers. The offset group holds offset / scene extent (see the module
  docstring). Non-persistent buffers: what build_init recomputes identically on a resume (the init itself, footprints, neighbour
  pairs); the checkpoint holds only the learnable rows."""

  def __init__(self, init, learnable=True, scale_param="log"):
    """scale_param: the encoding's (diffusion.scale_encoding.scale_param): what the rows' scale channels are (stats and the dump read
    the log-scales through log_scales); the rows are stored as the x0 channels either way."""
    super().__init__()
    self.learnable = bool(learnable)
    self.scale_param = scale_param
    self.views = [int(v) for v in init["views"]]
    self.mesh = [int(m) for m in init["mesh"]]
    self.index = {v: i for i, v in enumerate(self.views)}
    counts = [len(r) for r in init["rows"]]
    self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    self.hit = list(init["hit"])          # (L, H, W) bool per volume, CPU (the dump's validity masks)
    self.anchor = list(init["anchor"])    # (N_i, 3) per volume, CPU
    self.pose = list(init["pose"])
    self.paths = list(init.get("paths") or [])   # the volumes' fitted grid files (the dump copies their attributes)
    # the reference, the fitted grids' rows (CPU; None: the init is the reference); CPU, not a buffer: it is only read by the metrics
    self.ref_rows = None if init.get("ref_rows") is None else torch.from_numpy(np.concatenate(init["ref_rows"]))
    rows = torch.from_numpy(np.concatenate(init["rows"]) if counts else np.zeros((0, X0_CHANNELS), np.float32))
    ext = torch.from_numpy(np.repeat(np.asarray(init["extent"], np.float32), counts))
    self.register_buffer("row_extent", ext, persistent=False)
    self.register_buffer("foot", torch.from_numpy(np.concatenate(init["foot"]) if counts else np.zeros(0, np.float32)),
                         persistent=False)
    self.register_buffer("pairs", torch.from_numpy(np.asarray(init["pairs"], np.int64)), persistent=False)
    self.params = nn.ParameterDict() if self.learnable else None
    for name, sl in X0_SLICES.items():
      data = rows[:, sl].clone()
      if name == "offset":
        data = data / ext[:, None]
      self.register_buffer(f"init_{name}", data.clone(), persistent=False)
      if self.learnable:
        self.params[name] = nn.Parameter(data)
      else:
        self.register_buffer(f"frozen_{name}", data, persistent=False)

  def group(self, name):
    """The (N, d) stored rows of a channel group (offset: / extent)."""
    return self.params[name] if self.learnable else getattr(self, f"frozen_{name}")

  def rows(self, lo=0, hi=None, init=False):
    """Raw x0 rows [lo, hi) (all by default), (n, CHANNELS) in CHANNELS order; init: the init's (no graph)."""
    hi = len(self.row_extent) if hi is None else hi
    parts = []
    for name in X0_FIELDS:
      g = getattr(self, f"init_{name}") if init else self.group(name)
      g = g[lo:hi]
      parts.append(g * self.row_extent[lo:hi, None] if name == "offset" else g)
    return torch.cat(parts, 1)

  def volume_rows(self, i, init=False):
    return self.rows(int(self.starts[i]), int(self.starts[i + 1]), init=init)

  def has(self, view):
    return int(view) in self.index

  def dense(self, views, hit):
    """The raw grids of the volumes `views` (their source view rows), (B, CHANNELS, L, H, W) float32, zero at the invalid slots
    (the dataset's x0_raw layout), with the graph to the rows when learnable. hit (B, L, H, W): the batch's masks (checked)."""
    out = []
    for v, h in zip(views, hit):
      i = self.index.get(int(v))
      if i is None:
        raise KeyError(f"learn_grids: source view {int(v)} has no learnable grid (not a training volume)")
      r = self.volume_rows(i)
      h = h.to(r.device).bool()
      if int(h.sum()) != len(r):
        raise ValueError(f"learn_grids: volume {int(v)} has {len(r)} rows, its hit mask {int(h.sum())} valid slots")
      x = r.new_zeros((X0_CHANNELS, *h.shape))
      x[:, h] = r.T
      out.append(x)
    return torch.stack(out)

  def param_groups(self, lr):
    """The grid optimizer's groups: one per channel group, at the anchored fitter's learning rate of that group (LR_KEYS; offset:
    lr.offset on offset / extent, i.e. lr.offset x scene_extent on the offsets)."""
    return [{"params": [self.params[name]], "lr": float(lr[LR_KEYS[name]]), "name": f"grids/{name}"} for name in X0_FIELDS]

  @torch.no_grad()
  def to_grid(self, i, attrs=None):
    """Volume i's current rows as a grid_io.Grid in the grid files' format (world-frame means and quaternions -- the quaternion
    at its own raw length -- scales exp, opacities sigmoid, colours sh_dc SH_C0 + 0.5), with attrs; plus its raw rows
    (n, CHANNELS) float32."""
    from gs_dataset import OPENGL_TO_OPENCV, rotate_quats_wxyz
    raw = self.volume_rows(i).float().cpu().numpy()
    c2w = self.pose[i] @ OPENGL_TO_OPENCV
    means_cam = self.anchor[i] + raw[:, X0_SLICES["offset"]]
    q = raw[:, X0_SLICES["rotation"]]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_world = (rotate_quats_wxyz(q / np.maximum(norm, 1e-12), c2w[:3, :3]) * norm) if len(q) else q
    hit = self.hit[i]
    rows = {
      "gaussian_means": (means_cam @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32),
      "gaussian_scales": np.exp(log_scales(raw, self.scale_param)).astype(np.float32),
      "gaussian_quats": q_world.astype(np.float32),
      "gaussian_opacities": (1 / (1 + np.exp(-raw[:, X0_SLICES["opacity"].start]))).astype(np.float32),
      "gaussian_colors": (raw[:, X0_SLICES["sh_dc"]] * SH_C0 + 0.5).astype(np.float32),
    }
    return Grid(hit, np.nonzero(hit), rows, dict(attrs or {})), raw


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

STAT_QUANTILES = (0.05, 0.5, 0.95)
MAX_QUANTILE_ELEMENTS = 1 << 22


def stat_space(rows, foot, scale_param="log"):
  """The groups of raw rows (n, CHANNELS) in the units their statistics are logged in: opacity in [0, 1] (sigmoid), scale the 3
  log-scales, rotation the raw quaternion, sh_dc the rgb colour (sh_dc SH_C0 + 0.5), offset in pixel footprints (offset / foot)."""
  return {
    "opacity": torch.sigmoid(rows[:, X0_SLICES["opacity"]]),
    "scale": log_scales(rows, scale_param),
    "rotation": rows[:, X0_SLICES["rotation"]],
    "sh_dc": rows[:, X0_SLICES["sh_dc"]] * SH_C0 + 0.5,
    "offset": rows[:, X0_SLICES["offset"]] / foot[:, None],
  }


def _quantiles(x, qs):
  x = x.flatten().float()
  if x.numel() > MAX_QUANTILE_ELEMENTS:   # torch.quantile's input limit: a fixed subsample
    x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:MAX_QUANTILE_ELEMENTS].to(x.device)]
  return torch.quantile(x, torch.tensor(qs, device=x.device, dtype=x.dtype)).tolist()


@torch.no_grad()
def param_stats(rows, foot, pairs, init_rows=None, scale_param="log"):
  """The grids' parameter statistics, {name: float}: per group (stat_space's units) mean / std / p5 / p50 / p95 (offset: of the
  offset's length in footprints; rotation: of the quaternion's length; scale: also of log(largest axis / footprint) as scale_rel);
  the fraction of degenerate Gaussians (frac_transparent opacity < 0.01, frac_huge largest axis > 10 footprints, frac_tiny largest
  axis < 0.1 footprints, frac_degenerate any of them); per group the drift from init_rows (drift/<group>: the RMS over the rows of
  the per-row L2 distance, stat_space units) and the smoothness (smooth/<group>: the mean over neighbouring slot pairs, same layer,
  of the squared L2 difference, stat_space units)."""
  rows, foot = rows.float(), foot.float()
  s = stat_space(rows, foot, scale_param)
  out = {}
  log_rel = s["scale"].amax(1) - torch.log(foot)
  summary = {"opacity": s["opacity"], "scale": s["scale"], "rotation": s["rotation"].norm(dim=1), "sh_dc": s["sh_dc"],
             "offset": s["offset"].norm(dim=1), "scale_rel": log_rel}
  for name, x in summary.items():
    if x.numel() == 0:
      continue
    out[f"{name}/mean"] = float(x.mean())
    out[f"{name}/std"] = float(x.std()) if x.numel() > 1 else 0.0
    for q, v in zip(STAT_QUANTILES, _quantiles(x, STAT_QUANTILES)):
      out[f"{name}/p{round(q * 100):02d}"] = v
  if len(rows):
    transparent = s["opacity"][:, 0] < 0.01
    huge, tiny = log_rel > np.log(10.0), log_rel < np.log(0.1)
    for name, m in (("transparent", transparent), ("huge", huge), ("tiny", tiny), ("degenerate", transparent | huge | tiny)):
      out[f"frac_{name}"] = float(m.float().mean())
  if init_rows is not None and len(rows):
    s0 = stat_space(init_rows.float(), foot, scale_param)
    for name in X0_FIELDS:
      out[f"drift/{name}"] = float((s[name] - s0[name]).square().sum(1).mean().sqrt())
  if len(pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    for name in X0_FIELDS:
      out[f"smooth/{name}"] = float((s[name][a] - s[name][b]).square().sum(1).mean())
  return out
