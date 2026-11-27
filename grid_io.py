"""Finalized grid files (`<root>/grids/mesh<M>_view<V>.h5`, finalize_fit_dataset.py): one reader for both layouts, the sparse writer.

dense (v1, every file written before this module; no `layout` attr): per field an (H, W, L, C) float32 array (opacities (H, W, L)),
  NaN at the slots without a Gaussian, plus `layer_valid` (H, W, L) bool -- fit_3dgs.save_output_layered's schema.
sparse (v2, attrs layout="sparse", layout_version=2): the same Gaussians as rows,
  layer_valid_lhw   (L, H, W) bool, chunks (1, H, W), gzip 4: the validity mask (the index)
  layer_offsets     (L + 1,) int64: layer l's rows are [layer_offsets[l], layer_offsets[l + 1]) (CSR indptr over layers)
  gaussian_means / gaussian_scales / gaussian_colors (N, 3), gaussian_quats (N, 4), gaussian_opacities (N,): float32, one chunk,
                    gzip 4 + shuffle (HDF5 built-ins: no filter plugin needed to read them)
  rows in np.nonzero(layer_valid_lhw) order: layer, then y, then x (attrs row_order, index="mask"); attrs grid_shape (H, W, L),
  and every attr of the dense file unchanged (source_h5, primary_view_row, config_json, ...). Lossless: the rows are the dense
  file's values bit for bit (bla/research/21_grid_storage.md: 0.61x the bytes, ~5x faster valid-row reads).

read_grid(path) -> Grid (the valid rows + their indices; never densifies a sparse file); load_grid(path) -> the dense dict
(dense()) and attrs, for the callers that want (H, W, L, .) arrays; grid_shape(path); convert_file(dense, sparse).
"""

import dataclasses
import os

import h5py
import numpy as np

FIELDS = (("gaussian_means", 3), ("gaussian_scales", 3), ("gaussian_quats", 4), ("gaussian_opacities", 1),
          ("gaussian_colors", 3))
FIELD_NAMES = tuple(k for k, _ in FIELDS)
DENSE_DATASETS = FIELD_NAMES + ("layer_valid",)
SPARSE_DATASETS = FIELD_NAMES + ("layer_valid_lhw", "layer_offsets")
LAYOUT_VERSION = 2
LAYOUT_ATTRS = ("layout", "layout_version", "grid_shape", "row_order", "index")   # the sparse file's own; not the dense file's
ROW_ORDER = "layer, y, x (np.nonzero of layer_valid_lhw (L, H, W))"
ROW_CODEC = dict(compression="gzip", compression_opts=4, shuffle=True)
MASK_CODEC = dict(compression="gzip", compression_opts=4)


@dataclasses.dataclass
class Grid:
  """A grid's Gaussians as rows: valid (L, H, W) bool; idx (li, yi, xi) = np.nonzero(valid); rows {field: (N, C) float32,
  gaussian_opacities (N,)} in idx's order (the dense gather grid[k][yi, xi, li]); attrs (load_grid's: all but config_json, and
  none of the sparse layout's own)."""
  valid: np.ndarray
  idx: tuple
  rows: dict
  attrs: dict = dataclasses.field(default_factory=dict)

  @property
  def shape(self):
    """(H, W, L), as the dense arrays."""
    l, h, w = self.valid.shape
    return h, w, l

  @classmethod
  def from_dense(cls, grid, attrs=None):
    """load_grid's dense dict -> Grid (fields missing from the dict are left out)."""
    valid = np.ascontiguousarray(grid["layer_valid"].transpose(2, 0, 1))
    li, yi, xi = np.nonzero(valid)
    return cls(valid, (li, yi, xi), {k: grid[k][yi, xi, li] for k in FIELD_NAMES if k in grid}, dict(attrs or {}))

  def dense(self, fill=np.nan):
    """The dense dict load_grid gives for a dense file: (H, W, L, C) float32 per field (opacities (H, W, L)), `fill` at the
    invalid slots, and layer_valid (H, W, L) bool."""
    li, yi, xi = self.idx
    h, w, l = self.shape
    out = {}
    for k, c in FIELDS:
      a = np.full((h, w, l) + ((c,) if c > 1 else ()), fill, np.float32)
      a[yi, xi, li] = self.rows[k]
      out[k] = a
    out["layer_valid"] = np.ascontiguousarray(self.valid.transpose(1, 2, 0))
    return out


def is_sparse(f):
  """An open h5py file -> True for the sparse layout, False for the dense one (no layout attr)."""
  layout = f.attrs.get("layout")
  if layout is None:
    return False
  if layout != "sparse" or int(f.attrs.get("layout_version", -1)) != LAYOUT_VERSION:
    raise ValueError(f"{f.filename}: unknown grid layout {layout!r} version {f.attrs.get('layout_version')!r}")
  return True


def _attrs(f):
  """load_grid's attrs: all but config_json and the sparse layout's own."""
  skip = ("config_json",) + (LAYOUT_ATTRS if is_sparse(f) else ())
  return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items() if k not in skip}


def _read_sparse(f):
  valid = f["layer_valid_lhw"][()]
  idx = np.nonzero(valid)
  rows = {k: f[k][()] for k in FIELD_NAMES}
  n = len(idx[0])
  offsets = f["layer_offsets"][()]
  if offsets[-1] != n or any(len(r) != n for r in rows.values()):
    raise ValueError(f"{f.filename}: {n} valid slots but layer_offsets ends at {offsets[-1]} and the fields hold "
                     f"{ {k: len(r) for k, r in rows.items()} } rows")
  return Grid(valid, idx, rows, _attrs(f))


def read_grid(path):
  """A finalized grid file of either layout -> Grid. A sparse file is read as rows (no dense arrays); a dense file is read whole
  and gathered (as encode_grid always did)."""
  with h5py.File(path, "r") as f:
    if is_sparse(f):
      return _read_sparse(f)
    return Grid.from_dense({k: np.asarray(f[k]) for k in DENSE_DATASETS}, _attrs(f))


def load_grid(path):
  """A finalized grid file of either layout -> ({name: array (H, W, L, .)}, attrs): for a dense file exactly what it holds, for a
  sparse one the same (NaN at the invalid slots, the dense file's attrs)."""
  with h5py.File(path, "r") as f:
    if is_sparse(f):
      g = _read_sparse(f)
      return g.dense(), g.attrs
    return {k: np.asarray(f[k]) for k in DENSE_DATASETS}, _attrs(f)


def grid_shape(path):
  """(H, W, L) of a grid file of either layout, from its metadata only."""
  with h5py.File(path, "r") as f:
    return tuple(int(v) for v in f.attrs["grid_shape"]) if is_sparse(f) else tuple(f["layer_valid"].shape)


def _copy_attrs(src, dst, skip=()):
  """src's attrs onto dst with their HDF5 types (a fixed-length string stays one)."""
  for k in src.attrs:
    if k not in skip:
      aid = src.attrs.get_id(k)
      dst.attrs.create(k, src.attrs[k], dtype=aid.dtype)


def write_sparse(path, grid, src=None):
  """Write grid (a Grid; its rows float32) as a sparse file at path. src: an open dense grid file whose attrs (all of them,
  config_json included), field / mask dataset attrs and any other datasets are carried over unchanged."""
  for k, c in FIELDS:
    r = grid.rows[k]
    want = (len(grid.idx[0]),) + ((c,) if c > 1 else ())
    if r.dtype != np.float32 or r.shape != want:
      raise ValueError(f"{k}: rows must be float32 {want}, got {r.dtype} {r.shape}")
  l, h, w = grid.valid.shape
  with h5py.File(path, "w") as f:
    if src is not None:
      _copy_attrs(src, f)
      for k in src:
        if k not in DENSE_DATASETS:
          src.copy(src[k], f, name=k)   # anything else the file carries, as it is (filters included)
    else:
      for k, v in grid.attrs.items():
        f.attrs[k] = v
    f.attrs["layout"] = "sparse"
    f.attrs["layout_version"] = LAYOUT_VERSION
    f.attrs["grid_shape"] = np.array([h, w, l], np.int64)
    f.attrs["row_order"] = ROW_ORDER
    f.attrs["index"] = "mask"
    d = f.create_dataset("layer_valid_lhw", data=grid.valid.astype(bool), chunks=(1, h, w), **MASK_CODEC)
    if src is not None:
      _copy_attrs(src["layer_valid"], d)
    counts = grid.valid.reshape(l, -1).sum(1)
    f.create_dataset("layer_offsets", data=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    for k, _ in FIELDS:
      r = grid.rows[k]
      d = f.create_dataset(k, data=r, **(dict(chunks=r.shape, **ROW_CODEC) if len(r) else {}))   # (no chunks for 0 rows)
      if src is not None:
        _copy_attrs(src[k], d)


def _same(a, b):
  """Bitwise equality (NaN payloads and signed zeros included) of two arrays of one dtype and shape."""
  a, b = np.asarray(a), np.asarray(b)
  if a.dtype != b.dtype or a.shape != b.shape:
    return False
  if a.dtype.kind == "f":
    return np.array_equal(a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}"))
  return np.array_equal(a, b)


def _same_attrs(a, b, skip=()):
  """The attrs of two h5py objects are the same: names, HDF5 types and values."""
  ka, kb = set(a.attrs) - set(skip), set(b.attrs) - set(skip)
  if ka != kb:
    return f"attr names differ: {sorted(ka ^ kb)}"
  for k in ka:
    if a.attrs.get_id(k).get_type() != b.attrs.get_id(k).get_type():
      return f"attr {k}: HDF5 types differ"
    va, vb = a.attrs[k], b.attrs[k]
    if isinstance(va, (str, bytes)) or isinstance(vb, (str, bytes)):
      same = type(va) is type(vb) and va == vb
    else:
      same = _same(va, vb)
    if not same:
      return f"attr {k}: values differ"
  return None


def verify_sparse(dense_path, sparse_path):
  """None if sparse_path holds exactly dense_path's grid, else what differs: the mask, every valid row bitwise (the dense gather),
  the rebuilt dense arrays bitwise (so the invalid slots of the dense file were all NaN), every attr (type and value), the
  datasets' attrs and every other dataset bitwise."""
  with h5py.File(dense_path, "r") as fd, h5py.File(sparse_path, "r") as fs:
    if is_sparse(fd) or not is_sparse(fs):
      return "want a dense source and a sparse copy"
    dense = {k: np.asarray(fd[k]) for k in DENSE_DATASETS}
    g = _read_sparse(fs)
    ref = Grid.from_dense(dense)
    if not _same(g.valid, ref.valid):
      return "layer_valid differs"
    for k in FIELD_NAMES:
      if not _same(g.rows[k], ref.rows[k]):
        return f"{k}: rows differ"
    for k, v in g.dense().items():
      if not _same(v, dense[k]):
        return f"{k}: the dense arrays rebuilt from the rows differ (an invalid slot of the dense file is not NaN?)"
    if tuple(fs.attrs["grid_shape"]) != dense["layer_valid"].shape:
      return "grid_shape attr differs"
    msg = _same_attrs(fd, fs, skip=LAYOUT_ATTRS)
    if msg:
      return msg
    for kd, ks in [(k, k) for k in FIELD_NAMES] + [("layer_valid", "layer_valid_lhw")]:
      msg = _same_attrs(fd[kd], fs[ks])
      if msg:
        return f"{kd}: {msg}"
    extra_d, extra_s = set(fd) - set(DENSE_DATASETS), set(fs) - set(SPARSE_DATASETS)
    if extra_d != extra_s:
      return f"other datasets differ: {sorted(extra_d ^ extra_s)}"
    for k in extra_d:
      if type(fd[k]) is not type(fs[k]):
        return f"{k}: object types differ"
      if not isinstance(fd[k], h5py.Dataset):
        continue   # (a group: copied by h5py's own object copy, not walked)
      if not _same(fd[k][()], fs[k][()]):
        return f"{k}: values differ"
      msg = _same_attrs(fd[k], fs[k])
      if msg:
        return f"{k}: {msg}"
  return None


def convert_file(dense_path, sparse_path):
  """dense_path's grid as a sparse file at sparse_path: written to sparse_path + ".tmp", verified (verify_sparse), then renamed.
  Raises (and removes the temp file) when the copy is not exact. The dense file is left alone."""
  tmp = sparse_path + ".tmp"
  try:
    with h5py.File(dense_path, "r") as f:
      if is_sparse(f):
        raise ValueError(f"{dense_path} is already sparse")
      write_sparse(tmp, Grid.from_dense({k: np.asarray(f[k]) for k in DENSE_DATASETS}), src=f)
    msg = verify_sparse(dense_path, tmp)
    if msg:
      raise RuntimeError(f"{dense_path}: the sparse copy is not exact: {msg}")
    os.replace(tmp, sparse_path)
  finally:
    if os.path.exists(tmp):
      os.remove(tmp)
