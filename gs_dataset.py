"""Dataset for train_gs.py.

Reads render_objaverse.py-schema HDF5 files (`images`, `depth_peel`,
`depth_intrinsics`/`image_intrinsics` (or legacy `camera_intrinsics`),
`camera_pose`, `mesh_index`). Three pieces, kept deliberately separate:
  - GSViewsDataset: flat catalog of every (file_idx, view_idx), plus the
    mesh-group structure -- no sampling, no splitting.
  - split_by_mesh: a plain function computing a train/val split over a
    GSViewsDataset's mesh groups (whole meshes, never split across).
  - GSPairDataset: wraps a GSViewsDataset restricted to a set of mesh
    groups (e.g. one half of a split_by_mesh result), yielding items of one
    source view + up to `num_target_views` other views of the same mesh.

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
import random

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, random_split

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
  """Flat, ungrouped catalog of every view across `h5_paths`: one item per
  (file_idx, view_idx) tuple. Holds the mesh-group structure (which views
  belong to which mesh) and lazy per-process h5py.File handles, but does
  NOT do any source/target sampling or train/val splitting -- that's
  GSPairDataset (wraps this, does the sampling given a chosen set of mesh
  groups) and split_by_mesh (a plain function that computes a train/val
  split over this dataset's mesh groups) below, kept deliberately separate
  so the split logic isn't tangled up with the dataset class itself.
  """

  def __init__(self, h5_paths, num_layers=6):
    self.paths = _expand_h5_paths(h5_paths)
    self.num_layers = num_layers
    self._handles = {}

    self.groups = {}      # (file_idx, mesh_id) -> [view_idx, ...]
    self.mesh_paths = {}  # (file_idx, mesh_id) -> str
    self.items = []       # (file_idx, view_idx), flat, one per view
    self.group_of = {}    # (file_idx, view_idx) -> (file_idx, mesh_id)
    for file_idx, path in enumerate(self.paths):
      with h5py.File(path, "r") as f:
        mesh_index = f["mesh_index"][:]
        paths_ds = f["mesh_paths"][:] if "mesh_paths" in f else None
        for view_idx, mi in enumerate(mesh_index):
          key = (file_idx, int(mi))
          self.groups.setdefault(key, []).append(view_idx)
          item = (file_idx, view_idx)
          self.items.append(item)
          self.group_of[item] = key
          if paths_ds is not None and key not in self.mesh_paths:
            mp = paths_ds[int(mi)]
            self.mesh_paths[key] = mp.decode() if isinstance(mp, bytes) else str(mp)

    log.info(
      "GSViewsDataset: %d files, %d mesh groups, %d views total",
      len(self.paths), len(self.groups), len(self.items),
    )

  def mesh_keys(self):
    """Sorted list of every (file_idx, mesh_id) group present."""
    return sorted(self.groups)

  def __len__(self):
    return len(self.items)

  def _h5(self, file_idx):
    # lazy per-process open -- h5py.File handles aren't fork-safe, so each
    # DataLoader worker must open its own (this runs inside __getitem__,
    # i.e. inside the worker process).
    h = self._handles.get(file_idx)
    if h is None:
      h = h5py.File(self.paths[file_idx], "r")
      self._handles[file_idx] = h
    return h

  def __getitem__(self, idx):
    """Returns the raw (file_idx, view_idx) tuple -- a single view has no
    meaningful source/target structure on its own; GSPairDataset does the
    actual h5 reading, via this dataset's groups/_h5()."""
    return self.items[idx]


def split_by_mesh(views_ds, val_fraction=0.1, seed=42):
  """Splits `views_ds`'s MESH GROUPS (not individual views) into train/val
  key lists, via torch's random_split over the sorted group keys, so whole
  meshes never leak across the split. Pass the result to GSPairDataset's
  `mesh_keys` argument.

  A single-mesh corpus can't be split at the group level (nothing to hold
  out) -- returns (all keys, []) regardless of val_fraction; the view-level
  source/target sampling inside GSPairDataset already provides variety
  there (the expected setting is val_fraction=0, pure overfit)."""
  keys = views_ds.mesh_keys()
  if len(keys) <= 1:
    return keys, []
  n_val = max(1, round(len(keys) * val_fraction)) if val_fraction > 0 else 0
  n_train = len(keys) - n_val
  generator = torch.Generator().manual_seed(seed)
  train_subset, val_subset = random_split(keys, [n_train, n_val], generator=generator)
  return list(train_subset), list(val_subset)


class GSPairDataset(Dataset):
  """Wraps a GSViewsDataset, restricted to `mesh_keys` (compute train/val
  splits with split_by_mesh above -- this class doesn't split anything
  itself). One item = one (fixed) source view among those groups, indexed
  directly like Flash3D's own RealEstate10K dataset (index -> a specific
  source frame), plus up to `num_target_views` OTHER views of the same mesh
  sampled given that source (Flash3D samples target frames near the source
  frame in a video; we have no frame order, so target views are just drawn
  from the rest of the same mesh's views). `len(dataset)` is therefore the
  number of VIEWS in `mesh_keys`, not the number of meshes -- one epoch =
  every view in the split used as source exactly once.
  """

  def __init__(self, views_ds, mesh_keys, num_target_views=3, seed=42, deterministic_targets=False):
    self.views_ds = views_ds
    self.num_target_views = num_target_views
    self.seed = seed
    # Whether target-view sampling is reproducible (seeded off idx) or fresh
    # every __getitem__ call -- e.g. stable val visualizations/metrics vs.
    # data variety across train epochs. No default tied to a "split" concept
    # anymore (there's no such concept on this class); the caller decides,
    # see train_gs.py's main().
    self.deterministic_targets = deterministic_targets

    chosen = set(mesh_keys)
    self.items = [item for item in views_ds.items if views_ds.group_of[item] in chosen]

    log.info(
      "GSPairDataset: %d mesh groups / %d views selected (of %d views total in the underlying GSViewsDataset)",
      len(chosen), len(self.items), len(views_ds),
    )

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    file_idx, source_view = self.items[idx]
    mesh_key = self.views_ds.group_of[(file_idx, source_view)]
    views = self.views_ds.groups[mesh_key]
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
      mesh_key[1], self.views_ds.mesh_paths.get(mesh_key, ""),
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
    self._handle = None

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
    if self._handle is None:
      self._handle = h5py.File(self.path, "r")
    return self._handle

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
