"""Dataset for train_gs.py.

Reads render_objaverse.py-schema HDF5 files (`images`, `depth_peel`,
`depth_intrinsics`/`image_intrinsics` (or legacy `camera_intrinsics`),
`camera_pose`, `mesh_index`). Three pieces, kept deliberately separate:
  - H5Catalog: a declarative, polars-backed flat catalog over one or more
    h5 files -- columns are either a real per-view h5 dataset (read
    verbatim) or a synthetic per-row value (this row's file path / its
    index within that file). Not specific to this project's schema at all;
    subsetting (.filter()/.take()) returns another H5Catalog built directly
    from the filtered/selected rows, so nothing downstream ever needs to
    unwrap anything to reach a "parent" dataset -- every row already
    carries its own file path.
  - split_by_mesh: a plain function computing a train/val split over the
    distinct (path, mesh_id) groups found in an H5Catalog (whole meshes,
    never split across), returned as a pair of H5Catalogs.
  - GSPairDataset: wraps an H5Catalog (e.g. one half of a split_by_mesh
    result), grouping its rows by mesh on the fly, and yielding items of
    one source view + up to `num_target_views` other views of the same
    mesh.

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
import json
import logging
import os
import random

import h5py
import numpy as np
import polars as pl
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


def _rotate_quats_wxyz(q_wxyz, R_transform):
  """q_wxyz: (...,4) numpy wxyz unit quaternions. R_transform: (3,3) rotation
  matrix applied on the left (R_out = R_transform @ R_in). Returns (...,4)
  wxyz, same shape. Uses scipy.spatial.transform.Rotation (already a project
  dependency -- fit_gsplat.py uses it too) rather than hand-rolled quaternion
  composition, which is an easy place to get sign/order conventions wrong."""
  from scipy.spatial.transform import Rotation
  shape = q_wxyz.shape
  flat = q_wxyz.reshape(-1, 4)
  xyzw = flat[:, [1, 2, 3, 0]]
  rot_out = Rotation.from_matrix(R_transform) * Rotation.from_quat(xyzw)
  out_xyzw = rot_out.as_quat()
  out_wxyz = out_xyzw[:, [3, 0, 1, 2]]
  return out_wxyz.reshape(shape).astype(np.float32)


def _load_ground_truth(gt_h5_path, source_pose_gl):
  """Reads a fit_gsplat.py-schema .h5 (per-pixel ground-truth Gaussian
  params from ONE direct 3DGS fit, primary view first). Returns
  (tensors, attrs).

  tensors, in this file's (L,H,W,...) convention (matching xyz_cam/hit):
    opacity (L,H,W) f32 in (0,1); scale (L,H,W,3) f32, absolute world-unit
    size (same units GSModel.forward's own "scale" output already ends up
    in -- no rescaling needed); rotation (L,H,W,4) f32 wxyz, ROTATED INTO
    THE SOURCE CAMERA'S OWN FRAME (fit_gsplat.py optimizes in true Blender
    world space; this codebase's Gaussians live in the source camera's own
    frame -- see this module's docstring / relative_viewmats); color
    (L,H,W,3) f32, flat sigmoid RGB (NOT SH -- fit_gsplat.py has no SH>0 at
    all); valid (L,H,W) bool; means_world (L,H,W,3) f32, sanity-check-only
    (see _check_ground_truth_consistency), callers should drop it after.
  attrs: {"config": parsed config_json dict, "mesh_index": int, "mesh_path": str}.

  source_pose_gl: (4,4) torch tensor, the SAME source view's own raw
  camera-to-world pose (Blender/OpenGL axes) the caller already read --
  used only to build the world->source-camera rotation applied to `rotation`.
  """
  with h5py.File(gt_h5_path, "r") as f:
    opacity = np.asarray(f["gaussian_opacities"])      # (H,W,L)
    scale = np.asarray(f["gaussian_scales"])            # (H,W,L,3)
    quat_world = np.asarray(f["gaussian_quats"])        # (H,W,L,4) wxyz, WORLD frame
    color = np.asarray(f["gaussian_colors"])            # (H,W,L,3)
    valid = np.asarray(f["layer_valid"])                # (H,W,L) bool
    means_world = np.asarray(f["gaussian_means"])       # (H,W,L,3)
    attrs = {
      "config": json.loads(f.attrs["config_json"]),
      "mesh_index": int(f.attrs["mesh_index"]),
      "mesh_path": str(f.attrs.get("mesh_path", "")),
    }

  c2w_cv = source_pose_gl.numpy() @ OPENGL_TO_OPENCV
  r_w2c = c2w_cv[:3, :3].T
  quat_world_lhwc = rearrange(quat_world, "h w l c -> l h w c")
  valid_lhw = rearrange(valid, "h w l -> l h w")
  # Invalid (never-optimized) slots are zero-norm, not unit quaternions --
  # scipy's Rotation rejects those outright. They're always excluded by
  # `valid` downstream anyway, so only rotate the valid subset and leave the
  # rest as a harmless identity quaternion.
  quat_cam_lhwc = np.zeros_like(quat_world_lhwc)
  quat_cam_lhwc[..., 0] = 1.0
  if valid_lhw.any():
    quat_cam_lhwc[valid_lhw] = _rotate_quats_wxyz(quat_world_lhwc[valid_lhw], r_w2c)

  tensors = {
    "opacity": rearrange(torch.from_numpy(opacity), "h w l -> l h w").contiguous().float(),
    "scale": rearrange(torch.from_numpy(scale), "h w l c -> l h w c").contiguous().float(),
    "rotation": torch.from_numpy(quat_cam_lhwc).contiguous(),
    "color": rearrange(torch.from_numpy(color), "h w l c -> l h w c").contiguous().float(),
    "valid": rearrange(torch.from_numpy(valid), "h w l -> l h w").contiguous(),
    "means_world": rearrange(torch.from_numpy(means_world), "h w l c -> l h w c").contiguous().float(),
  }
  return tensors, attrs


def _check_ground_truth_consistency(attrs, gt, gt_h5_path, source_h5_path, source_view,
                                     mesh_id, xyz_cam, hit, pose_gl):
  """Startup-time (not per-step) cross-check between a loaded ground-truth
  grid and the render/view it's meant to supervise -- a silent mismatch here
  (wrong view/mesh pairing, optimize_means=true, wrong resolution) would
  corrupt training with no visible failure otherwise. Hard-errors on
  structural mismatches; logs+warns on softer signals."""
  cfg = attrs["config"]
  gt_primary = int(cfg.get("primary", -1))
  if gt_primary != source_view:
    raise ValueError(
      f"ground truth {gt_h5_path!r}: primary={gt_primary} != "
      f"data.fixed_source_view={source_view} -- per-pixel correspondence "
      "is keyed exactly to one view")
  if attrs["mesh_index"] != mesh_id:
    raise ValueError(
      f"ground truth {gt_h5_path!r}: mesh_index={attrs['mesh_index']} != "
      f"source view's mesh_index={mesh_id} -- wrong object pairing")
  if bool(cfg.get("optimize_means", False)):
    raise ValueError(
      f"ground truth {gt_h5_path!r} was fit with optimize_means=true -- "
      "this first pass only supports optimize_means=false (gaussian_means "
      "must equal the unprojected depth peel exactly)")
  if tuple(gt["valid"].shape) != tuple(hit.shape):
    raise ValueError(
      f"ground truth grid shape {tuple(gt['valid'].shape)} != source "
      f"view's depth-peel grid {tuple(hit.shape)}")

  gt_src_path = cfg.get("hdf5_path", "")
  if os.path.basename(gt_src_path) != os.path.basename(source_h5_path):
    log.warning(
      "ground truth %r was fit from %r, training is reading %r (comparing "
      "basenames only) -- double check these are really the same render",
      gt_h5_path, gt_src_path, source_h5_path)

  mismatch = (gt["valid"] != hit).float().mean().item()
  if mismatch > 1e-6:
    log.warning(
      "ground truth %r: layer_valid disagrees with this view's own hit "
      "mask on %.4f%% of grid cells -- expected 0", gt_h5_path, mismatch * 100)

  c2w_cv = pose_gl.numpy() @ OPENGL_TO_OPENCV
  xyz_h = np.concatenate([xyz_cam.numpy(), np.ones((*xyz_cam.shape[:-1], 1), np.float32)], axis=-1)
  xyz_world_reproj = (xyz_h @ c2w_cv.T)[..., :3]
  both_valid = (hit & gt["valid"]).numpy()
  if both_valid.any():
    err = np.abs(xyz_world_reproj[both_valid] - gt["means_world"].numpy()[both_valid]).mean()
    log.info(
      "ground truth %r position sanity check: mean |reprojected xyz_cam - "
      "gaussian_means| = %.6g over %d pixels (expect ~1e-5)",
      gt_h5_path, err, int(both_valid.sum()))
    if err > 1e-3:
      log.warning("ground truth %r position mismatch larger than expected (%.6g)", gt_h5_path, err)

  log.info(
    "ground truth OK: %r primary=%d mesh_index=%d valid_frac=%.4f",
    gt_h5_path, gt_primary, attrs["mesh_index"], float(gt["valid"].float().mean()),
  )


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


def _read_primary_from_gauss_h5(f):
  """f: open h5py.File for a fit_gsplat.py-schema ground-truth .h5. Reads
  the PRIMARY view's own image/depth/pose/intrinsics directly out of that
  file's embedded '_used'/'_primary' arrays (index 0 -- fit_gsplat.py always
  lists the primary view first, config_json["primary"] is its index in the
  SOURCE render h5, not into these arrays). Same return shape as
  _read_view. This makes a gauss_h5 fully self-sufficient as a training
  source -- no separate photom_h5 render corpus needed -- since it already
  carries everything used to fit it."""
  depth_peel = np.asarray(f["depth_peel_primary"]).astype(np.float32)   # (H,W,L)
  K_depth = np.asarray(f["depth_intrinsics_used"][0]).astype(np.float32)
  K_image = np.asarray(f["image_intrinsics_used"][0]).astype(np.float32)
  pose = np.asarray(f["camera_pose_used"][0]).astype(np.float32)
  images = np.asarray(f["images_used"][0])   # (H,W,4) RGBA

  return {
    "rgb": images[..., :3].astype(np.float32) / 255.0,
    "alpha": images[..., 3].astype(np.float32) / 255.0,
    "pose": pose, "K_depth": K_depth, "K_image": K_image, "depth_peel": depth_peel,
  }


def _assemble_item(src, targets, num_layers, mesh_id, source_view, target_views):
  """Shared by GSPairDataset (sampled source/targets) and GSFixedViewsDataset
  (fixed source/targets, for single-batch overfit sanity checks): given a
  DECIDED source view + target view dicts (from _read_view or
  _read_primary_from_gauss_h5 -- same shape either way), builds the full
  item dict train_gs.py's render_scene/render_orbit expect."""
  xyz_cam, hit = dense_unproject_camera(src["depth_peel"], src["K_depth"])
  if hit.shape[-1] != num_layers:
    raise ValueError(
      f"source view {source_view}: depth_peel has {hit.shape[-1]} "
      f"layers, expected num_layers={num_layers}")

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
    "source_view": int(source_view),
    "target_views": [int(v) for v in target_views],
  }


def _build_item(f, path, source_view, target_views, num_layers, mesh_id):
  """_assemble_item, reading src/targets from an open render_objaverse.py-
  schema h5py.File (f) at path -- see _assemble_item for the shared part."""
  k_depth_name = intrinsics_name(f, "depth")
  k_image_name = intrinsics_name(f, "image", None)
  src = _read_view(f, path, source_view, k_depth_name, k_image_name)
  targets = [_read_view(f, path, v, k_depth_name, k_image_name) for v in target_views]
  return _assemble_item(src, targets, num_layers, mesh_id, source_view, target_views)


class H5Catalog(Dataset):
  """A flat, declarative catalog over one or more h5 files, not specific to
  this project's schema at all. Each column argument is a plain `pl.Expr`
  (or a bare string, shorthand for `pl.col(name)`):
    H5Catalog.path()                            -- this row's file path
    H5Catalog.index()                            -- this row's index within its file
    H5Catalog.dataset("mesh_index").alias("mesh_id")  -- an h5 dataset, renamed
    "mesh_index"                                  -- same, unaliased
  `.alias(...)` is just `pl.Expr.alias` -- these are real expressions, not a
  custom DSL, so anything else `pl.Expr` supports works too.

  Subsetting (.filter()/.take()) returns another H5Catalog built directly
  from the filtered/selected rows. Since every row already carries its own
  file path, nothing downstream ever needs to reach back to a "parent"
  dataset to resolve anything -- unlike torch.utils.data.Subset, which
  requires unwrapping `.dataset` to reach whatever the wrapped dataset
  itself owned.
  """

  _PATH_COL = "__h5_path"    # reserved: this row's file path, broadcast per file
  _INDEX_COL = "__h5_index"  # reserved: this row's index within its file

  @classmethod
  def path(cls) -> pl.Expr:
    return pl.col(cls._PATH_COL).alias("path")

  @classmethod
  def index(cls) -> pl.Expr:
    return pl.col(cls._INDEX_COL).alias("index")

  @staticmethod
  def dataset(name: str) -> pl.Expr:
    return pl.col(name)

  def __init__(self, h5_paths, *columns):
    exprs = [pl.col(c) if isinstance(c, str) else c for c in columns]

    # Which real h5 datasets do these expressions actually need to be read
    # from each file? (root column names, minus the two synthetic ones we
    # always provide ourselves -- never read from the file.)
    needed = set()
    for e in exprs:
      needed.update(e.meta.root_names())
    needed -= {self._PATH_COL, self._INDEX_COL}
    if not needed:
      raise ValueError(
        "H5Catalog needs at least one column backed by a real per-view h5 "
        'dataset (e.g. H5Catalog.dataset("mesh_index")) to know how many '
        "rows each file has -- path()/index() alone aren't enough.")
    needed = sorted(needed)

    frames = []
    for path in _expand_h5_paths(h5_paths):
      with h5py.File(path, "r") as f:
        n = f[needed[0]].shape[0]
        raw = {
          # pl.repeat(..., eager=True): a Series filled by polars itself,
          # not a materialized n-long Python list -- matters once n (views
          # per file) gets large.
          self._PATH_COL: pl.repeat(path, n, eager=True),
          self._INDEX_COL: np.arange(n),
          **{name: np.asarray(f[name][:]) for name in needed},
        }
      frames.append(pl.DataFrame(raw).select(exprs))
    self.df = pl.concat(frames)

  @classmethod
  def _from_df(cls, df: pl.DataFrame) -> "H5Catalog":
    self = cls.__new__(cls)
    self.df = df
    return self

  def filter(self, predicate: pl.Expr) -> "H5Catalog":
    return self._from_df(self.df.filter(predicate))

  def take(self, indices) -> "H5Catalog":
    return self._from_df(self.df[np.asarray(indices)])

  def __len__(self):
    return self.df.height

  def __getitem__(self, idx):
    return self.df.row(idx, named=True)


def split_by_mesh(catalog: H5Catalog, val_fraction=0.1, seed=42):
  """Splits `catalog` into train/val H5Catalogs, at the MESH-GROUP level
  (whole meshes, never split a mesh's views across train/val). The distinct
  ("path", "mesh_id") keys are split via torch's random_split (operating
  just on their count/positions -- there are few meshes, many views), then
  each split's rows are pulled out with a vectorized semi-join, not a
  per-row Python scan.

  A single-mesh corpus can't be split at the group level (nothing to hold
  out) -- returns (catalog, <empty catalog>) regardless of val_fraction;
  the view-level source/target sampling inside GSPairDataset already
  provides variety there (the expected setting is val_fraction=0, pure
  overfit)."""
  keys_df = catalog.df.select(["path", "mesh_id"]).unique().sort(["path", "mesh_id"])
  n_keys = keys_df.height
  if n_keys <= 1:
    return catalog, H5Catalog._from_df(catalog.df.clear())

  n_val = max(1, round(n_keys * val_fraction)) if val_fraction > 0 else 0
  n_train = n_keys - n_val
  generator = torch.Generator().manual_seed(seed)
  train_pos, val_pos = random_split(range(n_keys), [n_train, n_val], generator=generator)

  train_keys_df = keys_df[np.array(train_pos.indices)]
  val_keys_df = keys_df[np.array(val_pos.indices)]
  train_df = catalog.df.join(train_keys_df, on=["path", "mesh_id"], how="semi")
  val_df = catalog.df.join(val_keys_df, on=["path", "mesh_id"], how="semi")
  return H5Catalog._from_df(train_df), H5Catalog._from_df(val_df)


class GSPairDataset(Dataset):
  """Wraps an H5Catalog with "path", "mesh_id", "view_idx" columns (e.g. one
  half of a split_by_mesh result -- this class doesn't split anything
  itself). One item = one (fixed) source view among the catalog's rows,
  indexed directly like Flash3D's own RealEstate10K dataset (index -> a
  specific source frame), plus up to `num_target_views` OTHER views of the
  same mesh sampled given that source (Flash3D samples target frames near
  the source frame in a video; we have no frame order, so target views are
  just drawn from the rest of the same mesh's views). `len(dataset)` is
  therefore the number of VIEWS in the catalog, not the number of meshes --
  one epoch = every view in the catalog used as source exactly once.

  Mesh grouping (which views are siblings of a given source view) is
  derived once here at construction, via a vectorized group_by over the
  catalog's own rows.
  """

  def __init__(self, catalog: H5Catalog, num_layers=6, num_target_views=3, seed=42,
              deterministic_targets=False):
    self.catalog = catalog
    self.num_layers = num_layers
    self.num_target_views = num_target_views
    self.seed = seed
    # Whether target-view sampling is reproducible (seeded off idx) or fresh
    # every __getitem__ call -- e.g. stable val visualizations/metrics vs.
    # data variety across train epochs. No default tied to a "split" concept
    # anymore (there's no such concept on this class); the caller decides,
    # see train_gs.py's main().
    self.deterministic_targets = deterministic_targets

    groups_df = catalog.df.group_by(["path", "mesh_id"], maintain_order=True).agg(pl.col("view_idx"))
    self.groups = {
      (row["path"], row["mesh_id"]): row["view_idx"]
      for row in groups_df.iter_rows(named=True)
    }

    log.info(
      "GSPairDataset: %d mesh groups / %d views",
      len(self.groups), len(catalog),
    )

  def __len__(self):
    return len(self.catalog)

  def __getitem__(self, idx):
    row = self.catalog[idx]
    path, mesh_id, source_view = row["path"], row["mesh_id"], row["view_idx"]
    views = self.groups[(path, mesh_id)]
    f = _get_h5(path)

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

    return _build_item(f, path, source_view, target_views, self.num_layers, mesh_id)


class GSFixedViewsDataset(Dataset):
  """Always returns the exact same (source, targets) item, no resampling at
  all -- fit_gsplat.py's primary/secondary terminology applied to train_gs.py,
  for single-batch overfit sanity checks (does the whole pipeline converge
  on one fixed set of views before trusting it on anything harder). `source`
  and every entry of `targets` must share the same mesh_index, checked at
  construction (mirrors fit_gsplat.py's own primary/secondary check).

  At least one of photom_h5/gauss_h5 must be set (never both -- see
  train_gs.py's config-validation block). photom_h5 (a render_objaverse.py
  h5) supplies the source+target views for photometric supervision, indexed
  by source_view/target_views. gauss_h5 (a fit_gsplat.py h5) supplies direct
  Gaussian-parameter ground truth for the source view -- AND, when photom_h5
  is None, is used to build the source view ITSELF too, straight from its
  own embedded primary-view arrays (see _read_primary_from_gauss_h5): it's
  fully self-sufficient as a training source since it already carries
  everything used to fit it, so no separate photom_h5 is needed. In that
  mode there's no photom corpus to draw target views from, so target_views
  is forced empty regardless of what's passed in."""

  def __init__(self, source_view, target_views, num_layers=6, photom_h5=None, gauss_h5=None):
    if photom_h5 is None and gauss_h5 is None:
      raise ValueError("GSFixedViewsDataset needs at least one of photom_h5/gauss_h5 set")
    self.photom_h5 = photom_h5
    self.gauss_h5 = gauss_h5
    self.source_view = int(source_view)
    self.target_views = [int(v) for v in target_views] if photom_h5 is not None else []
    self.num_layers = num_layers

    if photom_h5 is not None:
      with h5py.File(photom_h5, "r") as f:
        if "mesh_index" in f:
          mi = f["mesh_index"][:]
          order = [self.source_view] + self.target_views
          picked = {int(mi[i]) for i in order}
          if len(picked) > 1:
            raise ValueError(
              f"{photom_h5!r}: source view {self.source_view} and target views "
              f"{self.target_views} span multiple mesh_index values {sorted(picked)} "
              "-- they must all show the same object")
          self.mesh_id = int(mi[self.source_view])
        else:
          self.mesh_id = -1
    else:
      self.mesh_id = None   # filled in below from gauss_h5's own ground-truth attrs

    self.ground_truth = None
    if gauss_h5 is not None:
      # Direct-parameter-supervision ground truth (fit_gsplat.py output) for
      # the source view. Re-reads/unprojects that one view independently of
      # __getitem__'s own item-building below -- a duplicate I/O given
      # __len__()==1 (this item never changes), not worth sharing state for.
      if photom_h5 is not None:
        with h5py.File(photom_h5, "r") as f:
          k_depth_name = intrinsics_name(f, "depth")
          k_image_name = intrinsics_name(f, "image", None)
          src = _read_view(f, photom_h5, self.source_view, k_depth_name, k_image_name)
      else:
        with h5py.File(gauss_h5, "r") as f:
          src = _read_primary_from_gauss_h5(f)
      xyz_cam_np, hit_np = dense_unproject_camera(src["depth_peel"], src["K_depth"])
      xyz_cam = rearrange(torch.from_numpy(xyz_cam_np), "h w l c -> l h w c").contiguous()
      hit = rearrange(torch.from_numpy(hit_np), "h w l -> l h w").contiguous()
      pose_gl = torch.from_numpy(src["pose"])

      gt_tensors, gt_attrs = _load_ground_truth(gauss_h5, pose_gl)
      if self.mesh_id is None:
        self.mesh_id = gt_attrs["mesh_index"]

      if photom_h5 is not None:
        _check_ground_truth_consistency(
          gt_attrs, gt_tensors, gauss_h5, photom_h5, self.source_view,
          self.mesh_id, xyz_cam, hit, pose_gl,
        )
      else:
        # No separate corpus to cross-check against -- the source view came
        # from this same file, so it's self-consistent by construction.
        log.info(
          "ground truth OK (self-contained, no photom_h5): %r primary=%d mesh_index=%d valid_frac=%.4f",
          gauss_h5, int(gt_attrs["config"].get("primary", -1)), gt_attrs["mesh_index"],
          float(gt_tensors["valid"].float().mean()),
        )
      gt_tensors.pop("means_world", None)
      self.ground_truth = gt_tensors

    log.info(
      "GSFixedViewsDataset: source=%d targets=%s mesh_id=%s photom_h5=%s gauss_h5=%s",
      self.source_view, self.target_views, self.mesh_id, self.photom_h5, self.gauss_h5,
    )

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    if self.photom_h5 is not None:
      item = _build_item(
        _get_h5(self.photom_h5), self.photom_h5, self.source_view, self.target_views,
        self.num_layers, self.mesh_id,
      )
    else:
      src = _read_primary_from_gauss_h5(_get_h5(self.gauss_h5))
      item = _assemble_item(src, [], self.num_layers, self.mesh_id, self.source_view, [])
    if self.ground_truth is not None:
      item["ground_truth"] = self.ground_truth
    return item


class _EmptyDataset(Dataset):
  """A zero-length placeholder for `val_ds` when there's nothing to validate
  against (e.g. GSFixedViewsDataset's single-batch overfit mode) -- every
  caller already guards on `len(val_ds) == 0`."""

  def __len__(self):
    return 0

  def __getitem__(self, idx):
    raise IndexError("_EmptyDataset has no items")
