"""Dataset for train_gs.py.

Reads render_objaverse.py-schema HDF5 files (`images`, `depth_peel`,
`depth_intrinsics`/`image_intrinsics` (or legacy `camera_intrinsics`),
`camera_pose`, `mesh_index`). Three pieces, kept deliberately separate:
  - GSViewsDataset: flat catalog of every (file_idx, mesh_id, view_idx) --
    no mesh-group index is cached, no sampling, no splitting.
  - split_by_mesh: a plain function computing a train/val split over the
    distinct mesh groups found in a GSViewsDataset (whole meshes, never
    split across), returned as a pair of torch.utils.data.Subset objects.
  - GSPairDataset: wraps a Subset of a GSViewsDataset (e.g. one half of a
    split_by_mesh result), grouping the subset's own items by mesh on the
    fly, and yielding items of one source view + up to `num_target_views`
    other views of the same mesh.

All geometry stays in the SOURCE camera's own frame -- never Blender world
space. The source view's point cloud is unprojected directly into its own
camera space (no pose applied at all); rendering into a different (target)
camera is done by transforming *cameras* via a pose relative to the source,
not by moving points through world coordinates. This mirrors Flash3D's own
design (it predicts Gaussians in the source camera's frame and applies a
relative pose to render novel views) and makes the self-reconstruction
view's transform the identity matrix by construction.
"""

import glob as _glob
import logging
import os
import random

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, Subset, random_split

from util import intrinsics_name

log = logging.getLogger(__name__)

# Camera-local axis flip: Blender/OpenGL (X right, Y up, Z back) <-> OpenCV
# (X right, Y down, Z forward). Same flip fit_gsplat.py / debug_pointcloud.py use.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

# z-score constants for camera-space xyz (WT's own training distribution,
# from the finetune-wt-lora worktree's wt_dataset.xyz_to_x0), derived from
# render_objaverse.py's own convention (unit-cube-normalized objects,
# camera_distance~=1.9 -- hence the mean Z of ~1.9). Used to bring the
# point-cloud input channels to an RGB-comparable numeric range, the
# analogue of Flash3D's "divide depth by 20" trick.
XYZ_MEAN = np.array([-0.0024, -0.0040, 1.9043], dtype=np.float32)
XYZ_STD = np.array([0.281, 0.316, 0.706], dtype=np.float32)


def dense_unproject_camera(depth_peel, intrinsics):
  """depth_peel: (H,W,L) f32, <=0/NaN = no hit. intrinsics: (3,3), this
  view's own camera. Returns (xyz_cam, hit): xyz_cam is (H,W,L,3) f32 in
  this camera's own OpenCV frame (X right, Y down, Z forward), NaN where not
  hit; hit is (H,W,L) bool. Same ray formula as
  debug_pointcloud.unproject_depth_peel(space="camera"), densified."""
  h, w, l = depth_peel.shape
  uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
  pixels_h = np.stack([uu, vv, np.ones_like(uu)], axis=-1)          # (H,W,3)
  rays_cv = pixels_h @ np.linalg.inv(intrinsics).T                  # (H,W,3), unit-depth rays
  hit = depth_peel > 0                                              # NaN and legacy -1.0 both fail
  xyz_cam = rays_cv[:, :, None, :] * depth_peel[:, :, :, None]      # (H,W,L,3)
  return np.where(hit[..., None], xyz_cam, np.nan).astype(np.float32), hit


def relative_viewmats(poses):
  """poses: (V,4,4) camera-to-world, OpenGL axes, poses[0] = the source
  view. Returns viewmats: (V,4,4) world-to-camera in OpenCV convention, but
  with "world" redefined as the source camera's own frame -- viewmats[0] is
  always the identity matrix."""
  c2w_cv = poses @ OPENGL_TO_OPENCV                # (V,4,4), still world-referenced
  rel_c2w = np.linalg.inv(c2w_cv[0]) @ c2w_cv       # (V,4,4), source-frame-referenced
  return np.linalg.inv(rel_c2w).astype(np.float32)  # (V,4,4), viewmats[0] == eye(4)


def xyz_to_x0(xyz):
  return (xyz - XYZ_MEAN) / XYZ_STD


_h5_cache = {}       # path -> h5py.File, process-global
_h5_cache_pid = None  # pid that populated _h5_cache


def _get_h5(path):
  """Process-safe global h5py.File cache, keyed by path. h5py.File handles
  aren't fork-safe -- if a DataLoader worker is forked (the default
  multiprocessing start method on Linux) after this cache already holds an
  open handle, the child would otherwise inherit that same handle baked
  into its copy of the dict. Guard against that by resetting the cache
  whenever the current pid doesn't match the pid that populated it, so
  each process (including each forked worker) always opens its own
  handles rather than reusing a parent's."""
  global _h5_cache, _h5_cache_pid
  pid = os.getpid()
  if _h5_cache_pid != pid:
    _h5_cache = {}
    _h5_cache_pid = pid
  h = _h5_cache.get(path)
  if h is None:
    h = h5py.File(path, "r")
    _h5_cache[path] = h
  return h


def _expand_h5_paths(h5_paths):
  paths = [h5_paths] if isinstance(h5_paths, str) else list(h5_paths)
  out = []
  for p in paths:
    matches = sorted(_glob.glob(p)) if any(c in p for c in "*?[") else [p]
    if not matches:
      raise FileNotFoundError(f"No h5 files matched {p!r}")
    out.extend(matches)
  return out


def _read_view(f, path, view_idx, k_depth_name, k_image_name):
  images = np.asarray(f["images"][view_idx])
  depth_peel = np.asarray(f["depth_peel"][view_idx]).astype(np.float32)
  K_depth = np.asarray(f[k_depth_name][view_idx]).astype(np.float32)
  K_image = (np.asarray(f[k_image_name][view_idx]).astype(np.float32)
             if k_image_name is not None else K_depth)
  pose = np.asarray(f["camera_pose"][view_idx]).astype(np.float32)

  if images.shape[-1] == 4:
    rgb = images[..., :3].astype(np.float32) / 255.0
    alpha = images[..., 3].astype(np.float32) / 255.0
  else:
    ih, iw = images.shape[:2]
    dh, dw = depth_peel.shape[:2]
    if (ih, iw) != (dh, dw):
      raise ValueError(
        f"{path!r} view {view_idx}: RGB-only (no alpha) file with images "
        f"{ih}x{iw} != depth_peel {dh}x{dw} -- the mask can't be derived; "
        "use an RGBA render or matching resolutions.")
    rgb = images.astype(np.float32) / 255.0
    alpha = (depth_peel[..., 0] > 0).astype(np.float32)

  return {
    "rgb": rgb, "alpha": alpha, "pose": pose,
    "K_depth": K_depth, "K_image": K_image, "depth_peel": depth_peel,
  }


def _build_item(f, path, source_view, target_views, num_layers, mesh_id, mesh_path):
  """Shared by GSPairDataset (sampled source/targets) and GSFixedViewsDataset
  (fixed source/targets, for single-batch overfit sanity checks): given a
  DECIDED source view + target view list, reads them and builds the full
  item dict train_gs.py's render_scene/render_orbit expect."""
  k_depth_name = intrinsics_name(f, "depth")
  k_image_name = intrinsics_name(f, "image", None)

  src = _read_view(f, path, source_view, k_depth_name, k_image_name)
  xyz_cam, hit = dense_unproject_camera(src["depth_peel"], src["K_depth"])
  if hit.shape[-1] != num_layers:
    raise ValueError(
      f"{path!r} view {source_view}: depth_peel has {hit.shape[-1]} "
      f"layers, expected num_layers={num_layers}")

  targets = [_read_view(f, path, v, k_depth_name, k_image_name) for v in target_views]

  poses = np.stack([src["pose"]] + [t["pose"] for t in targets], axis=0)
  viewmats = relative_viewmats(poses)   # (1+k, 4, 4), viewmats[0] == eye(4)

  def to_view_dict(v, viewmat):
    return {
      "rgb": rearrange(torch.from_numpy(v["rgb"]), "h w c -> c h w").contiguous(),  # (3,H,W)
      "alpha": torch.from_numpy(v["alpha"])[None],                       # (1,H,W)
      "K_image": torch.from_numpy(v["K_image"]),                        # (3,3)
      "viewmat": torch.from_numpy(viewmat),                             # (4,4)
    }

  return {
    "source": {
      **to_view_dict(src, viewmats[0]),
      "xyz_cam": rearrange(torch.from_numpy(xyz_cam), "h w l c -> l h w c").contiguous(),  # (L,H,W,3)
      "hit": rearrange(torch.from_numpy(hit), "h w l -> l h w").contiguous(),              # (L,H,W)
      "K_depth": torch.from_numpy(src["K_depth"]),
      "pose_gl": torch.from_numpy(src["pose"]),          # raw camera-to-world, Blender/OpenGL
                                                           # axes -- only used to derive a stable
                                                           # "up" direction for orbit previews
                                                           # (train_gs.py); not used in training.
    },
    "targets": [to_view_dict(t, viewmats[1 + i]) for i, t in enumerate(targets)],
    "mesh_index": mesh_id,
    "mesh_path": mesh_path,
    "source_view": int(source_view),
    "target_views": [int(v) for v in target_views],
  }


class GSViewsDataset(Dataset):
  """Flat catalog of every view across `h5_paths`: one item per
  (file_idx, mesh_id, view_idx) tuple -- `items` is the single source of
  truth for this dataset's contents, nothing else is cached from it. In
  particular, no view->mesh-group index is precomputed/stored here: mesh
  grouping is only ever needed transiently (to compute a split, or to find
  a source view's sibling views), so split_by_mesh and GSPairDataset below
  each derive it on the fly, locally, from `items` -- avoids keeping a
  derived index in sync with `items` for its own sake.
  """

  def __init__(self, h5_paths, num_layers=6):
    self.paths = _expand_h5_paths(h5_paths)
    self.num_layers = num_layers

    self.mesh_paths = {}  # (file_idx, mesh_id) -> str -- static per-mesh metadata,
                            # unrelated to any grouping/split, fine to cache as-is.
    self.items = []        # (file_idx, mesh_id, view_idx), flat, one per view
    for file_idx, path in enumerate(self.paths):
      with h5py.File(path, "r") as f:
        mesh_index = f["mesh_index"][:]
        paths_ds = f["mesh_paths"][:] if "mesh_paths" in f else None
        for view_idx, mi in enumerate(mesh_index):
          mesh_id = int(mi)
          self.items.append((file_idx, mesh_id, view_idx))
          key = (file_idx, mesh_id)
          if paths_ds is not None and key not in self.mesh_paths:
            mp = paths_ds[mesh_id]
            self.mesh_paths[key] = mp.decode() if isinstance(mp, bytes) else str(mp)

    log.info("GSViewsDataset: %d files, %d views total", len(self.paths), len(self.items))

  def __len__(self):
    return len(self.items)

  def _h5(self, file_idx):
    return _get_h5(self.paths[file_idx])

  def __getitem__(self, idx):
    """Returns the raw (file_idx, mesh_id, view_idx) tuple -- a single view
    has no meaningful source/target structure on its own; GSPairDataset
    does the actual h5 reading, via this dataset's _h5()."""
    return self.items[idx]


def split_by_mesh(views_ds, val_fraction=0.1, seed=42):
  """Splits `views_ds` into train/val torch.utils.data.Subset objects, at
  the MESH-GROUP level (whole meshes, never split a mesh's views across
  train/val). Collects a (file_idx, mesh_id) -> [dataset index, ...] dict
  by iterating `views_ds.items` (indices into `views_ds`, NOT view_idx --
  that's what Subset needs), splits the dict's keys via torch's
  random_split, then collects each split's dataset indices and wraps them
  in a Subset.

  A single-mesh corpus can't be split at the group level (nothing to hold
  out) -- returns (Subset(views_ds, <all indices>), Subset(views_ds, []))
  regardless of val_fraction; the view-level source/target sampling inside
  GSPairDataset already provides variety there (the expected setting is
  val_fraction=0, pure overfit)."""
  groups = {}   # (file_idx, mesh_id) -> [dataset index, ...]
  for i, (file_idx, mesh_id, _) in enumerate(views_ds.items):
    groups.setdefault((file_idx, mesh_id), []).append(i)

  keys = sorted(groups)
  if len(keys) <= 1:
    return Subset(views_ds, list(range(len(views_ds)))), Subset(views_ds, [])

  n_val = max(1, round(len(keys) * val_fraction)) if val_fraction > 0 else 0
  n_train = len(keys) - n_val
  generator = torch.Generator().manual_seed(seed)
  train_keys, val_keys = random_split(keys, [n_train, n_val], generator=generator)

  train_indices = [i for k in train_keys for i in groups[k]]
  val_indices = [i for k in val_keys for i in groups[k]]
  return Subset(views_ds, train_indices), Subset(views_ds, val_indices)


class GSPairDataset(Dataset):
  """Wraps a torch.utils.data.Subset of a GSViewsDataset (see split_by_mesh
  above -- this class doesn't split anything itself). One item = one
  (fixed) source view among the subset's views, indexed directly like
  Flash3D's own RealEstate10K dataset (index -> a specific source frame),
  plus up to `num_target_views` OTHER views of the same mesh sampled given
  that source (Flash3D samples target frames near the source frame in a
  video; we have no frame order, so target views are just drawn from the
  rest of the same mesh's views). `len(dataset)` is therefore the number of
  VIEWS in the subset, not the number of meshes -- one epoch = every view
  in the subset used as source exactly once.

  Mesh grouping (which views are siblings of a given source view) is
  derived once here at construction, from the subset's own items -- not
  read off any persistent index on GSViewsDataset (see its docstring).
  """

  def __init__(self, views_subset, num_target_views=3, seed=42, deterministic_targets=False):
    self.views_ds = views_subset.dataset
    self.num_target_views = num_target_views
    self.seed = seed
    # Whether target-view sampling is reproducible (seeded off idx) or fresh
    # every __getitem__ call -- e.g. stable val visualizations/metrics vs.
    # data variety across train epochs. No default tied to a "split" concept
    # anymore (there's no such concept on this class); the caller decides,
    # see train_gs.py's main().
    self.deterministic_targets = deterministic_targets

    self.groups = {}   # (file_idx, mesh_id) -> [view_idx, ...], this subset only
    self.items = []    # (file_idx, mesh_id, view_idx), this subset only
    for i in range(len(views_subset)):
      file_idx, mesh_id, view_idx = views_subset[i]
      key = (file_idx, mesh_id)
      self.groups.setdefault(key, []).append(view_idx)
      self.items.append((file_idx, mesh_id, view_idx))

    log.info(
      "GSPairDataset: %d mesh groups / %d views selected (of %d views total in the underlying GSViewsDataset)",
      len(self.groups), len(self.items), len(self.views_ds),
    )

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    file_idx, mesh_id, source_view = self.items[idx]
    views = self.groups[(file_idx, mesh_id)]
    path = self.views_ds.paths[file_idx]
    f = self.views_ds._h5(file_idx)

    rng = random.Random(f"{self.seed}_{idx}") if self.deterministic_targets else random.Random()
    # Sample num_target_views+1 candidates (all still <= len(views), since
    # num_target_views <= len(views)-1) and drop source_view from that small
    # sample instead of first filtering it out of the whole (possibly much
    # larger) views list -- avoids an O(len(views)) scan every call.
    k = min(self.num_target_views, len(views) - 1)
    if k > 0:
      sampled = rng.sample(views, k + 1)
      target_views = [v for v in sampled if v != source_view][:k]
    else:
      target_views = []

    return _build_item(
      f, path, source_view, target_views, self.views_ds.num_layers,
      mesh_id, self.views_ds.mesh_paths.get((file_idx, mesh_id), ""),
    )


class GSFixedViewsDataset(Dataset):
  """Always returns the exact same (source, targets) item, no resampling at
  all -- fit_gsplat.py's primary/secondary terminology applied to train_gs.py,
  for single-batch overfit sanity checks (does the whole pipeline converge
  on one fixed set of views before trusting it on anything harder). `source`
  and every entry of `targets` must share the same mesh_index, checked at
  construction (mirrors fit_gsplat.py's own primary/secondary check)."""

  def __init__(self, h5_path, source_view, target_views, num_layers=6):
    self.path = h5_path
    self.source_view = int(source_view)
    self.target_views = [int(v) for v in target_views]
    self.num_layers = num_layers

    with h5py.File(h5_path, "r") as f:
      self.mesh_paths_ds = f["mesh_paths"][:] if "mesh_paths" in f else None
      if "mesh_index" in f:
        mi = f["mesh_index"][:]
        order = [self.source_view] + self.target_views
        picked = {int(mi[i]) for i in order}
        if len(picked) > 1:
          raise ValueError(
            f"{h5_path!r}: source view {self.source_view} and target views "
            f"{self.target_views} span multiple mesh_index values {sorted(picked)} "
            "-- they must all show the same object")
        self.mesh_id = int(mi[self.source_view])
      else:
        self.mesh_id = -1

    mesh_path = ""
    if self.mesh_paths_ds is not None and 0 <= self.mesh_id < len(self.mesh_paths_ds):
      mp = self.mesh_paths_ds[self.mesh_id]
      mesh_path = mp.decode() if isinstance(mp, bytes) else str(mp)
    self.mesh_path = mesh_path

    log.info(
      "GSFixedViewsDataset: source=%d targets=%s mesh=%s",
      self.source_view, self.target_views, mesh_path or self.mesh_id,
    )

  def __len__(self):
    return 1

  def _h5(self):
    return _get_h5(self.path)

  def __getitem__(self, idx):
    return _build_item(
      self._h5(), self.path, self.source_view, self.target_views,
      self.num_layers, self.mesh_id, self.mesh_path,
    )


class _EmptyDataset(Dataset):
  """A zero-length placeholder for `val_ds` when there's nothing to validate
  against (e.g. GSFixedViewsDataset's single-batch overfit mode) -- every
  caller already guards on `len(val_ds) == 0`."""

  def __len__(self):
    return 0

  def __getitem__(self, idx):
    raise IndexError("_EmptyDataset has no items")
