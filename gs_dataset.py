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
  - split_by_mesh / split_meshes_by_* / split_views_by_*: plain functions
    computing train/val splits of an H5Catalog, returned as a pair of
    H5Catalogs. split_by_mesh splits over the distinct (path, mesh_id)
    groups (whole meshes, never split across); the newer two-stage pair
    (split_meshes_by_*, then split_views_by_* on the train meshes) is what
    train_gs.py's data.mesh_split / data.view_split use -- see their
    docstrings and render_catalog.
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
import hashlib
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


def rotate_quats_wxyz(q_wxyz, R_transform):
  """q_wxyz: (...,4) numpy wxyz unit quaternions. R_transform: (3,3) rotation
  matrix applied on the left (R_out = R_transform @ R_in). Returns (...,4)
  wxyz, same shape. Uses scipy.spatial.transform.Rotation (already a project
  dependency -- fit_gsplat.py uses it too) rather than hand-rolled quaternion
  composition, which is an easy place to get sign/order conventions wrong.
  Public (not _-prefixed): train_gs.py's get_preview_source() also uses this,
  the opposite direction (camera-to-world instead of world-to-camera), to
  transform its preview Gaussians into world space for module.OrbitCallback."""
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
    quat_cam_lhwc[valid_lhw] = rotate_quats_wxyz(quat_world_lhwc[valid_lhw], r_w2c)

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
  """One view straight out of an open h5py.File `f` (for the datasets that
  have no H5Catalog -- see _read_catalog_view for the catalog path)."""
  return _decode_view(
    path, view_idx, f["images"][view_idx], f["depth_peel"][view_idx], f[k_depth_name][view_idx],
    f[k_image_name][view_idx] if k_image_name is not None else None, f["camera_pose"][view_idx])


def _read_catalog_view(catalog, pos):
  """(view dict, view_idx) for row `pos` of a render_catalog(), via row()."""
  r = catalog.row(pos, "path", "view_idx", "images", "depth_peel", "depth_intrinsics",
                  "image_intrinsics", "camera_pose")
  view = _decode_view(r["path"], r["view_idx"], r["images"], r["depth_peel"], r["depth_intrinsics"],
                      r["image_intrinsics"], r["camera_pose"])
  return view, int(r["view_idx"])


def _decode_view(path, view_idx, images, depth_peel, K_depth, K_image, pose):
  """The raw h5 arrays of one view -> the float32 view dict _assemble_item
  takes. K_image None: same as K_depth."""
  images = np.asarray(images)
  depth_peel = np.asarray(depth_peel).astype(np.float32)
  K_depth = np.asarray(K_depth).astype(np.float32)
  K_image = np.asarray(K_image).astype(np.float32) if K_image is not None else K_depth
  pose = np.asarray(pose).astype(np.float32)

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
    "rgb": rgb,
    "alpha": alpha,
    "pose": pose,
    "K_depth": K_depth,
    "K_image": K_image,
    "depth_peel": depth_peel,
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

  poses = np.stack([src["pose"], *(t["pose"] for t in targets)], axis=0)
  viewmats = relative_viewmats(poses)   # (1+k, 4, 4), viewmats[0] == eye(4)

  def to_view_dict(v, viewmat):
    return {
      "rgb": rearrange(torch.from_numpy(v["rgb"]), "h w c -> c h w").contiguous(),
      "alpha": rearrange(torch.from_numpy(v["alpha"]), "h w -> 1 h w"),
      "K_image": torch.from_numpy(v["K_image"]),    # (3,3)
      "viewmat": torch.from_numpy(viewmat),         # (4,4)
    }

  return {
    "views": torch.utils.data.default_collate([
      to_view_dict(v, viewmat)
      for v, viewmat in zip([src, *targets], viewmats)
    ]),
    "source": {
      "xyz_cam": rearrange(torch.from_numpy(xyz_cam), "h w l c -> l h w c").contiguous(),  # (L,H,W,3)
      "hit": rearrange(torch.from_numpy(hit), "h w l -> l h w").contiguous(),              # (L,H,W)
      "K_depth": torch.from_numpy(src["K_depth"]),
      "pose_gl": torch.from_numpy(src["pose"]),          # raw camera-to-world, Blender/OpenGL
                                                           # axes -- only used to derive a stable
                                                           # "up" direction for orbit previews
                                                           # (train_gs.py); not used in training.
    },
    "mesh_index": mesh_id,
    "source_view": int(source_view),
    "target_views": [int(v) for v in target_views],
  }


def _build_catalog_item(source_catalog, source_pos, target_catalog, target_pos, num_layers):
  """_assemble_item, reading the source view (row `source_pos` of
  source_catalog) and the target views (rows `target_pos` of target_catalog)
  through H5Catalog.row()."""
  src, source_view = _read_catalog_view(source_catalog, source_pos)
  targets, target_views = [], []
  for pos in target_pos:
    view, view_idx = _read_catalog_view(target_catalog, pos)
    targets.append(view)
    target_views.append(view_idx)
  mesh_id = source_catalog.df["mesh_id"][source_pos]
  return _assemble_item(src, targets, num_layers, mesh_id, source_view, target_views)


def _build_item(f, path, source_view, target_views, num_layers, mesh_id):
  """_assemble_item, reading src/targets from an open render_objaverse.py-
  schema h5py.File (f) at path -- for datasets without an H5Catalog
  (GSFixedViewsDataset); see _assemble_item for the shared part."""
  k_depth_name = intrinsics_name(f, "depth")
  k_image_name = intrinsics_name(f, "image", None)
  src = _read_view(f, path, source_view, k_depth_name, k_image_name)
  targets = [_read_view(f, path, v, k_depth_name, k_image_name) for v in target_views]
  return _assemble_item(src, targets, num_layers, mesh_id, source_view, target_views)


class _FileColumns:
  """What an H5Catalog build function receives (`c`) for ONE h5 file: a
  polars-flavoured view of it. Everything comes back as a pl.Series, read
  from the file the first time it's asked for and cached for the rest of the
  call:
    c.path()          -- this file's path, once per row           ("path")
    c.index()         -- each row's index within the file, 0..n-1 ("view_idx")
    c.dataset(name)   -- an h5 dataset, string datasets decoded    (name)
  Series are named by those defaults; the build function is free to
  rename/combine them."""

  def __init__(self, path, f, n):
    self._path, self._f, self.n = path, f, n
    self._cache = {}

  def path(self) -> pl.Series:
    return pl.Series("path", [self._path] * self.n)

  def index(self) -> pl.Series:
    return pl.Series("view_idx", np.arange(self.n))

  def dataset(self, name) -> pl.Series:
    if name not in self._cache:
      if name not in self._f:
        raise KeyError(f"{self._path!r}: no dataset {name!r}")
      ds = self._f[name]
      values = ds.asstr()[()] if h5py.check_string_dtype(ds.dtype) is not None else ds[()]
      self._cache[name] = pl.Series(name, values)
    return self._cache[name]


class H5Catalog(Dataset):
  """A flat, polars-backed catalog over one or more h5 files, not specific to
  this project's schema at all: one table row per row of the files, plus the
  ability to read h5 data for a row. Two things are handed in:

    datasets: names of the h5 datasets that `row()` will read (e.g.
      ["images", "depth_peel"]). Every file must have all of them, with the
      same first-dimension length n -- that is the file's row count. Checked
      at construction, so a missing dataset or a mismatch fails there, naming
      the file, not at some later read. They are NOT loaded: row() fetches one
      row at a time, on demand.
    build: `build(c) -> pl.DataFrame`, called once per file with a
      _FileColumns `c` (see its docstring), returning that file's table rows
      (exactly n of them). This is where small per-row columns come from, and
      where a lookup across datasets of different lengths is just polars, e.g.
        lambda c: pl.DataFrame({
          "path": c.path(), "view_idx": c.index(),
          "mesh_name": c.dataset("mesh_paths").gather(c.dataset("mesh_index"))})
      The per-file tables are concatenated.

  row() needs the columns "path" and "view_idx" (the file and the row in it);
  construction checks for them.

  Subsetting (.filter()/.take()) returns another H5Catalog built directly
  from the filtered/selected rows. Since every row already carries its own
  file path, nothing downstream ever needs to reach back to a "parent"
  dataset to resolve anything -- unlike torch.utils.data.Subset, which
  requires unwrapping `.dataset` to reach whatever the wrapped dataset
  itself owned.
  """

  def __init__(self, h5_paths, datasets, build):
    self.datasets = tuple(datasets)
    frames = []
    for path in _expand_h5_paths(h5_paths):
      with h5py.File(path, "r") as f:
        lengths = {}
        for name in self.datasets:
          if name not in f:
            raise KeyError(f"{path!r}: no dataset {name!r}")
          lengths[name] = f[name].shape[0]
        if len(set(lengths.values())) != 1:
          raise ValueError(f"{path!r}: datasets differ in length: {lengths}")
        n = next(iter(lengths.values()))
        df = build(_FileColumns(path, f, n))
      if df.height != n:
        raise ValueError(f"{path!r}: build() returned {df.height} rows, the file has {n}")
      frames.append(df)
    self.df = pl.concat(frames)
    missing = {"path", "view_idx"} - set(self.df.columns)
    if missing:
      raise ValueError(f"build() must return the columns 'path' and 'view_idx', missing {sorted(missing)}")

  def _from_df(self, df: pl.DataFrame) -> "H5Catalog":
    """Another catalog over the same datasets, with `df` as its table."""
    new = H5Catalog.__new__(H5Catalog)
    new.datasets = self.datasets
    new.df = df
    return new

  def row(self, pos, *columns, strict=True):
    """{column: value} for the row at position `pos`. Each column is looked up
    in the table first, then read from the row's h5 file (the row `view_idx`
    of dataset `column`), as it is, with no conversion. strict=True (default)
    only accepts table columns and the `datasets` this catalog was built with;
    strict=False reads any dataset the file has, unchecked."""
    row = self.df.row(pos, named=True)
    out = {}
    for name in columns:
      if name in row:
        out[name] = row[name]
        continue
      if strict and name not in self.datasets:
        raise KeyError(f"{name!r} is neither a table column nor one of the catalog's datasets {self.datasets}")
      out[name] = _get_h5(row["path"])[name][row["view_idx"]]
    return out

  def empty(self) -> "H5Catalog":
    """A zero-row catalog with this one's columns."""
    return self._from_df(self.df.clear())

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
    return catalog, catalog.empty()

  n_val = max(1, round(n_keys * val_fraction)) if val_fraction > 0 else 0
  n_train = n_keys - n_val
  generator = torch.Generator().manual_seed(seed)
  train_pos, val_pos = random_split(range(n_keys), [n_train, n_val], generator=generator)

  train_keys_df = keys_df[np.array(train_pos.indices)]
  val_keys_df = keys_df[np.array(val_pos.indices)]
  train_df = catalog.df.join(train_keys_df, on=["path", "mesh_id"], how="semi")
  val_df = catalog.df.join(val_keys_df, on=["path", "mesh_id"], how="semi")
  return catalog._from_df(train_df), catalog._from_df(val_df)


def split_by_view(catalog: H5Catalog, val_fraction=0.1, seed=42):
  gen = torch.Generator().manual_seed(seed)

  n_total = len(catalog)
  n_val = max(1, round(n_total * val_fraction)) if val_fraction > 0 else 0
  n_train = n_total - n_val
  idx = np.arange(n_total)
  train_idx, val_idx = random_split(idx, [n_train, n_val], generator=gen)
  train_idx, val_idx = train_idx.indices, val_idx.indices

  if n_total <= 1000:
    print("Train idx", list(train_idx))
    print("Val idx", list(val_idx))

  return catalog._from_df(catalog.df[train_idx]), catalog._from_df(catalog.df[val_idx])


def split_by_indices(catalog: H5Catalog, train_idx, val_idx, seed=42, strict=True):
  """Splits `catalog` at caller-given row positions (into catalog.df, the
  same index space split_by_view's own random split operates in) instead of
  a random val_fraction -- for reproducing one specific split (e.g. matching
  an earlier run, or hand-picking which scenes are held out) rather than
  reseeding a random one. `seed` is accepted only so GSDataModule.setup()
  can call every data.split_fn the same way (split_fn(catalog,
  seed=cfg.seed)); unused here.

  `strict` (default True) requires train_idx/val_idx to exactly partition
  catalog: (1) no index in both, and (2) together they cover every row
  0..len(catalog)-1 -- a typo'd or stale index list fails loudly instead of
  silently training/validating on the wrong rows. Set False to allow a
  deliberate partial split (rows in neither list are just dropped) or an
  intentional overlap."""
  train_idx, val_idx = list(train_idx), list(val_idx)
  if strict:
    overlap = sorted(set(train_idx) & set(val_idx))
    if overlap:
      raise ValueError(f"split_by_indices: {overlap} appear in both train_idx and val_idx")
    missing = sorted(set(range(len(catalog))) - set(train_idx) - set(val_idx))
    if missing:
      raise ValueError(
        f"split_by_indices: strict=True requires train_idx+val_idx to cover every row "
        f"in [0, {len(catalog)}) -- missing {missing}")
  return catalog._from_df(catalog.df[train_idx]), catalog._from_df(catalog.df[val_idx])


# ---------------------------------------------------------------------------
# two-stage split: meshes first, then views of the train meshes
# ---------------------------------------------------------------------------

def _name_rng(seed, name):
  """A numpy Generator seeded from (seed, name) -- same SeedSequence scheme
  as view_strategy.RandomViews' per-mesh RNG (sha256, not hash(): that is
  salted per process), so anything derived from it is a function of the
  mesh's name alone, not of dataset order or size."""
  digest = hashlib.sha256(name.encode("utf-8")).digest()
  words = np.frombuffer(digest, dtype="<u4").tolist()
  return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


# The datasets a render_objaverse.py h5 is read through (see render_catalog).
RENDER_DATASETS = ("mesh_index", "images", "depth_peel", "camera_pose",
                   "depth_intrinsics", "image_intrinsics")


def _render_columns(c: "_FileColumns") -> pl.DataFrame:
  mesh_index = c.dataset("mesh_index")
  return pl.DataFrame({
    "path": c.path(),
    "view_idx": c.index(),   # the row in the h5 file, NOT the view's number within its mesh
    "mesh_id": mesh_index,
    "mesh_name": c.dataset("mesh_paths").gather(mesh_index),
  }).with_columns(
    # the view's number within its mesh, 0..n-1 in row order
    (pl.col("view_idx").rank("ordinal").over("mesh_id").cast(pl.Int64) - 1).alias("view_num"))


def render_catalog(h5_paths) -> H5Catalog:
  """An H5Catalog over render_objaverse.py-schema h5 files (str, list or
  glob(s)). Columns: path, view_idx (row in the file), mesh_id (mesh_index),
  mesh_name (the mesh's own path, `mesh_paths[mesh_index]`), view_num (the
  view's number within its mesh). Reads images, depth_peel, camera_pose,
  depth_intrinsics and image_intrinsics via row(). Legacy files with only
  `camera_intrinsics` are not supported and fail at construction."""
  return H5Catalog(h5_paths, RENDER_DATASETS, _render_columns)


def _mesh_names(catalog: H5Catalog):
  return sorted(catalog.df["mesh_name"].unique().to_list())


def _take_meshes(catalog: H5Catalog, names) -> H5Catalog:
  return catalog.filter(pl.col("mesh_name").is_in(list(names)))


def split_meshes_by_fraction(catalog: H5Catalog, val_fraction=0.1, seed=42):
  """Mesh split, stage 1 of 2: holds out `round(val_fraction * n_meshes)`
  (at least 1) whole meshes -> (train_meshes, val_meshes). Meshes are ranked
  by a hash of (seed, mesh_name) and the lowest ranks are held out, so which
  mesh lands where depends on its name and the seed only, never on dataset
  order. A corpus with a single mesh can't be split and returns
  (catalog, <empty>); val_fraction: 0 disables the split the same way."""
  names = _mesh_names(catalog)
  if len(names) <= 1 or val_fraction <= 0:
    return catalog, catalog.empty()
  n_val = min(max(1, round(len(names) * val_fraction)), len(names) - 1)
  ranked = sorted(names, key=lambda n: _name_rng(seed, n).random())
  val = set(ranked[:n_val])
  return _take_meshes(catalog, set(names) - val), _take_meshes(catalog, val)


def split_meshes_by_names(catalog: H5Catalog, val_meshes, train_meshes=None, seed=42):
  """Mesh split, stage 1 of 2, by explicit mesh path (the `mesh_paths` entry,
  see render_catalog). `val_meshes` is held out; `train_meshes` null means
  every other mesh. Unknown names and train/val overlap are errors. `seed` is
  accepted only so every mesh_split can be called the same way; unused."""
  names = set(_mesh_names(catalog))
  val_meshes = list(val_meshes)
  train_meshes = None if train_meshes is None else list(train_meshes)
  unknown = sorted(set(val_meshes + (train_meshes or [])) - names)
  if unknown:
    raise ValueError(f"split_meshes_by_names: not in the catalog: {unknown[:5]}")
  overlap = sorted(set(val_meshes) & set(train_meshes or []))
  if overlap:
    raise ValueError(f"split_meshes_by_names: in both train_meshes and val_meshes: {overlap[:5]}")
  train = set(train_meshes) if train_meshes is not None else names - set(val_meshes)
  return _take_meshes(catalog, train), _take_meshes(catalog, val_meshes)


def split_views_by_fraction(catalog: H5Catalog, val_fraction=0.0, seed=42):
  """View split, stage 2 of 2 (run on the train meshes): inside EVERY mesh,
  holds out `round(val_fraction * n_views)` (at least 1) views ->
  (train_views, val_views). Each mesh's pick is a permutation seeded from
  (seed, mesh_name), so it is a function of that mesh alone. val_fraction: 0
  disables the split (train_views is everything, val_views empty)."""
  if val_fraction <= 0:
    return catalog, catalog.empty()
  df = catalog.df.with_row_index("__row")
  per_mesh = df.group_by("mesh_name").agg(pl.col("__row").sort_by("view_num"))
  val_rows = []
  for name, rows in per_mesh.iter_rows():
    n_val = max(1, round(len(rows) * val_fraction))
    if n_val >= len(rows):
      raise ValueError(f"split_views_by_fraction: mesh {name!r} has {len(rows)} views, "
                       f"can't hold out {n_val} and keep one for training")
    val_rows += [rows[i] for i in _name_rng(seed, name).permutation(len(rows))[:n_val]]
  is_val = df["__row"].is_in(val_rows)
  keep = [c for c in df.columns if c != "__row"]
  return (catalog._from_df(df.filter(~is_val).select(keep)),
          catalog._from_df(df.filter(is_val).select(keep)))


def split_views_by_numbers(catalog: H5Catalog, val_views, train_views=None, seed=42):
  """View split, stage 2 of 2, by explicit view numbers (`view_num`, the
  view's number within its mesh -- see render_catalog), the same list for
  every mesh. `val_views` is held out; `train_views` null means every other
  view. `seed` is accepted only so every view_split can be called the same
  way; unused."""
  val_views = [int(v) for v in val_views]
  train_views = None if train_views is None else [int(v) for v in train_views]
  overlap = sorted(set(val_views) & set(train_views or []))
  if overlap:
    raise ValueError(f"split_views_by_numbers: in both train_views and val_views: {overlap[:5]}")
  val = catalog.filter(pl.col("view_num").is_in(val_views))
  train = (catalog.filter(~pl.col("view_num").is_in(val_views)) if train_views is None
           else catalog.filter(pl.col("view_num").is_in(train_views)))
  return train, val


def sample_matching_sources(train_views: H5Catalog, val_views: H5Catalog, seed=42):
  """For each mesh, `n` views drawn without replacement from its train_views,
  where `n` is how many val_views that mesh has (all of them if it has fewer)
  -- so a validation set whose sources come from train_views is as big as the
  one whose sources come from val_views. Seeded per mesh from (seed, name)."""
  counts = dict(val_views.df.group_by("mesh_name").len().iter_rows())
  df = train_views.df.with_row_index("__row")
  keep = []
  for name, rows in df.group_by("mesh_name").agg(pl.col("__row").sort_by("view_num")).iter_rows():
    n = min(counts.get(name, 0), len(rows))
    order = _name_rng(seed, f"sources/{name}").permutation(len(rows))[:n]
    keep += [rows[i] for i in sorted(order)]
  cols = [c for c in df.columns if c != "__row"]
  return train_views._from_df(df.filter(pl.col("__row").is_in(keep)).select(cols))


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

  source_views: optional list of row positions (same index space as
  H5Catalog.take()/split_by_indices) eligible to be a SOURCE view -- i.e.
  what `__len__`/`__getitem__` index over. Target-view sampling is
  unaffected -- self.groups (below) is always built from the FULL catalog,
  so targets are still drawn from every view of the source's mesh regardless
  of this restriction. None (default): every catalog row is eligible, same
  as before this option existed.

  allow_source_as_target: whether the source view itself is eligible to also
  be sampled into its own target_views list. False (default, unchanged
  behavior): source_view is excluded, so every target is a genuinely
  different view. True: source_view is just another member of the mesh's
  view pool, same odds as any other view.

  target_catalog: optional second H5Catalog the TARGET views are drawn from
  instead of `catalog` itself (groups are built from it, matched to a source
  by (path, mesh_id)) -- e.g. sources from a mesh's train views and targets
  from its held-out views. Every mesh in `catalog` must appear in it. Since a
  source then isn't in the target pool, it is never excluded from it
  (allow_source_as_target has no effect). None (default): targets come from
  `catalog`, unchanged behavior.

  target_views: optional list/set of raw view_idx values (the same domain
  as self.groups' own values -- NOT row positions like source_views above)
  eligible to be sampled as a TARGET, applied per-mesh at construction (once,
  not a per-__getitem__ filter). Lets a caller hold out a set of views from
  training's target pool (e.g. to keep them unseen for a separate
  validation dataset over the same fixed source -- see
  GSFixedSourceDataset). None (default): every view in the mesh group is
  eligible, unchanged behavior.
  """

  def __init__(self, catalog: H5Catalog, num_layers=6, num_target_views=3, seed=42,
              deterministic_targets=False, source_views=None,
              allow_source_as_target=False, target_views=None, target_catalog=None):
    self.catalog = catalog
    self.num_layers = num_layers
    self.num_target_views = num_target_views
    self.seed = seed
    self.allow_source_as_target = allow_source_as_target or target_catalog is not None
    # Whether target-view sampling is reproducible (seeded off idx) or fresh
    # every __getitem__ call -- e.g. stable val visualizations/metrics vs.
    # data variety across train epochs. No default tied to a "split" concept
    # anymore (there's no such concept on this class); the caller decides,
    # see train_gs.py's main().
    self.deterministic_targets = deterministic_targets

    # self.groups: (path, mesh_id) -> the ROW POSITIONS (into target_catalog)
    # of that mesh's views eligible as targets.
    self.target_catalog = catalog if target_catalog is None else target_catalog
    groups_df = (
      self.target_catalog.df.with_row_index("__pos")
      .group_by(["path", "mesh_id"], maintain_order=True)
      .agg(pl.col("__pos"), pl.col("view_idx"))
    )
    allowed = None if target_views is None else set(target_views)
    self.groups = {
      (row["path"], row["mesh_id"]): [
        p for p, v in zip(row["__pos"], row["view_idx"]) if allowed is None or v in allowed]
      for row in groups_df.iter_rows(named=True)
    }

    self.source_catalog = catalog if source_views is None else catalog.take(source_views)
    # each source row's position in `catalog`, to keep it out of its own targets
    self.source_pos = np.arange(len(catalog)) if source_views is None else np.asarray(source_views, dtype=int)

    log.info(
      "GSPairDataset: %d mesh groups / %d views (%d eligible as source)",
      len(self.groups), len(catalog), len(self.source_catalog),
    )

  def __len__(self):
    return len(self.source_catalog)

  def __getitem__(self, idx):
    row = self.source_catalog[idx]
    views = self.groups[(row["path"], row["mesh_id"])]
    source_pos = int(self.source_pos[idx])

    rng = random.Random(f"{self.seed}_{idx}") if self.deterministic_targets else random.Random()
    if self.allow_source_as_target:
      k = min(self.num_target_views, len(views))
      target_views = rng.sample(views, k) if k > 0 else []
    else:
      # Sample num_target_views+1 candidates (all still <= len(views), since
      # num_target_views <= len(views)-1) and drop the source from that small
      # sample instead of first filtering it out of the whole (possibly much
      # larger) views list -- avoids an O(len(views)) scan every call.
      k = min(self.num_target_views, len(views) - 1)
      if k <= 0:
        target_views = []
      else:
        sampled = rng.sample(views, k + 1)
        target_views = [v for v in sampled if v != source_pos][:k]

    return _build_catalog_item(self.source_catalog, idx, self.target_catalog, target_views, self.num_layers)


class GSFixedSourceDataset(Dataset):
  """One fixed SOURCE view, indexed by held-out TARGET view instead of by
  source row -- for validating a fixed-source setup (e.g.
  data.source_views=[N] on GSPairDataset) where indexing by source row
  collapses len(dataset) to 1 (or, across a mesh-level split_fn, 0), giving
  no real per-epoch validation signal. This class doesn't split anything
  itself -- target_views is the caller's own held-out list (e.g. the
  complement of whatever GSPairDataset.target_views excludes from train's
  own sampling pool, so train/val target sets stay disjoint).

  targets_per_item: how many target views one row renders/supervises at
  once, all through a SINGLE encoder pass on the shared source (see
  GSModel.forward/model_forward -- only batch["views"]["rgb"][:,0], the
  source, is ever encoded). len(dataset) == ceil(len(target_views) /
  targets_per_item). 1 (default): one row per held-out view, so a metric
  averaged over the whole val set sees every held-out view exactly once
  per epoch, matching GSPairDataset's own val convention
  (deterministic_targets=True) of one full, stable pass per epoch.
  """

  def __init__(self, catalog: H5Catalog, source_view, target_views, num_layers=6, targets_per_item=1):
    df = catalog.df.with_row_index("__pos")
    rows = df.filter(pl.col("view_idx") == source_view)
    if len(rows) != 1:
      raise ValueError(
        f"GSFixedSourceDataset: expected exactly one catalog row for "
        f"source_view={source_view}, found {len(rows)}")
    row = rows.row(0, named=True)
    self.catalog = catalog
    self.source_pos = row["__pos"]
    self.path = row["path"]
    self.mesh_id = row["mesh_id"]
    # the target views are looked up in the source's own file
    pos_of = dict(zip(df.filter(pl.col("path") == self.path)["view_idx"], df.filter(pl.col("path") == self.path)["__pos"]))
    self.target_pos = [pos_of[int(v)] for v in target_views]
    self.source_view = int(source_view)
    self.target_views = [int(v) for v in target_views]
    self.num_layers = num_layers
    self.targets_per_item = targets_per_item

    log.info(
      "GSFixedSourceDataset: source_view=%d, %d held-out target views (%d rows, %d per row)",
      self.source_view, len(self.target_views), len(self), targets_per_item,
    )

  def __len__(self):
    return -(-len(self.target_views) // self.targets_per_item)  # ceil div

  def __getitem__(self, idx):
    start = idx * self.targets_per_item
    chunk = self.target_pos[start:start + self.targets_per_item]
    return _build_catalog_item(self.catalog, self.source_pos, self.catalog, chunk, self.num_layers)


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


class GaussH5ValDataset(Dataset):
  """Single-item validation dataset built entirely from a fit_gsplat.py
  ground-truth .h5's own embedded views: the primary view as source, every
  one of its embedded secondary views (images_used[1:]/camera_pose_used[1:]/
  image_intrinsics_used[1:] -- the exact views fit_gsplat.py itself fit
  against) as photometric targets. ground_truth is attached too, so
  apply_ground_truth_overrides pins whichever fields the model isn't
  predicting to the real per-pixel fit instead of exposing untrained decoder
  heads on render. Used as GSDataModule's val_ds whenever data.gauss_h5 is
  set and there's no separate data.photom_h5_val corpus -- gives real
  photometric val numbers (logged via the ordinary val/loss_l1 etc. path)
  even for a direct-supervision-only training run, with no extra render
  corpus needed on disk."""

  def __init__(self, gauss_h5, num_layers=6):
    self.gauss_h5 = gauss_h5
    self.num_layers = num_layers
    with h5py.File(gauss_h5, "r") as f:
      images_used = np.asarray(f["images_used"])             # (V,H,W,4) uint8
      poses_used = np.asarray(f["camera_pose_used"])          # (V,4,4)
      k_image_used = np.asarray(f["image_intrinsics_used"])   # (V,3,3)
      view_index_used = np.asarray(f["view_index_used"])      # (V,)
      src = _read_primary_from_gauss_h5(f)

    pose_gl = torch.from_numpy(src["pose"])
    gt_tensors, gt_attrs = _load_ground_truth(gauss_h5, pose_gl)
    gt_tensors.pop("means_world", None)

    targets = [
      {
        "rgb": images_used[i, ..., :3].astype(np.float32) / 255.0,
        "alpha": images_used[i, ..., 3].astype(np.float32) / 255.0,
        "K_image": k_image_used[i].astype(np.float32),
        "pose": poses_used[i].astype(np.float32),
      }
      for i in range(1, images_used.shape[0])
    ]
    self._item = _assemble_item(
      src, targets, num_layers=num_layers, mesh_id=gt_attrs["mesh_index"],
      source_view=int(view_index_used[0]), target_views=[int(v) for v in view_index_used[1:]],
    )
    self._item["ground_truth"] = gt_tensors

    log.info(
      "GaussH5ValDataset: %r source_view=%d, %d embedded target views (the real "
      "fit_gsplat.py corpus, ground-truth-pinned)",
      gauss_h5, self._item["source_view"], len(targets),
    )

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    return self._item


class _EmptyDataset(Dataset):
  """A zero-length placeholder for `val_ds` when there's nothing to validate
  against (e.g. GSFixedViewsDataset's single-batch overfit mode) -- every
  caller already guards on `len(val_ds) == 0`."""

  def __len__(self):
    return 0

  def __getitem__(self, idx):
    raise IndexError("_EmptyDataset has no items")
