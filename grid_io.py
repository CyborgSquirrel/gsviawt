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

packed (one HDF5 file per shard, pack_grids.py; root attrs layout="packed", layout_version=PACKED_VERSION): a directory of sparse
files as ONE file, never modified once written (temp name -> verify -> fsync -> rename; pack_dir):
  /grids/<primary_view_row>/   one group per grid: exactly the sparse file's datasets (the five fields H5Ocopy'd: their compressed
                               bytes, dtype, shape, one chunk, gzip 4 + shuffle and attrs as they are) and its root attrs (all of
                               them, layout="sparse" included, so the group reads as the file did); two storage-only changes, the
                               values bit for bit the same: layer_valid_lhw is ONE chunk (L, H, W) (same gzip 4), layer_offsets is
                               COMPACT (in the object header). Written grid by grid with HDF5's default file-space settings, so each
                               grid's data is one contiguous byte range (a 56-byte contiguous layer_offsets / the per-layer mask chunks
                               would go to HDF5's shared small-data blocks, < 2 KB, and interleave the grids: grid_extents checks it)
  /rows (n,) int64             the grids' primary_view_row, ascending (the index: no group is opened to build it)
  /names (n,) str              the source file names (mesh<M>_view<V>.h5)
  root attrs                   layout, layout_version, n_grids, source_h5 and grid_shape (the grids' own, the same for all),
                               packed_from (the source directory)
A grid of a packed file is a GridRef(path, group); read_grid / load_grid / grid_shape take either a path or a GridRef. The packed
files are opened lazily, read-only, once per process (packed_file: os.register_at_fork closes them before a fork, so a DataLoader
worker opens its own and never inherits a handle).
"""

import dataclasses
import contextlib
import glob
import logging
import os
import typing

import h5py
import numpy as np

log = logging.getLogger(__name__)

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
    raise ValueError(f"{_where(f)}: unknown grid layout {layout!r} version {f.attrs.get('layout_version')!r}")
  return True


def _attrs(f):
  """load_grid's attrs: all but config_json and the sparse layout's own."""
  skip = ("config_json",) + (LAYOUT_ATTRS if is_sparse(f) else ())
  return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items() if k not in skip}


def _read_sparse(f):
  """A sparse file or a packed file's grid group -> Grid."""
  valid = f["layer_valid_lhw"][()]
  idx = np.nonzero(valid)
  rows = {k: f[k][()] for k in FIELD_NAMES}
  n = len(idx[0])
  offsets = f["layer_offsets"][()]
  if offsets[-1] != n or any(len(r) != n for r in rows.values()):
    raise ValueError(f"{_where(f)}: {n} valid slots but layer_offsets ends at {offsets[-1]} and the fields hold "
                     f"{ {k: len(r) for k, r in rows.items()} } rows")
  return Grid(valid, idx, rows, _attrs(f))


def _where(obj):
  """A file's name, a group's file:group (for messages)."""
  return obj.file.filename if obj.name == "/" else f"{obj.file.filename}:{obj.name}"


# ---------------------------------------------------------------------------------------------------------------- packed files

PACKED_LAYOUT = "packed"
PACKED_VERSION = 1
PACKED_ROOT_ATTRS = ("layout", "layout_version", "n_grids", "source_h5", "grid_shape", "packed_from")


class GridRef(typing.NamedTuple):
  """One grid of a packed file: (the file, its group "grids/<row>"). build_grid_index's value for a packed file (a plain path for a
  per-file grid)."""
  path: str
  group: str

  @property
  def row(self):
    return int(self.group.rsplit("/", 1)[1])

  def __str__(self):
    return f"{self.path}:{self.group}"


def group_name(row):
  return f"grids/{int(row)}"


_files = {}          # path -> the open packed file of this process (packed_file)
_files_pid = [None]  # the process _files belongs to


def close_packed_files():
  """Closes this process's packed-file handles (they are reopened on the next read)."""
  for f in list(_files.values()):
    try:
      f.close()
    except Exception:   # (an already closed / invalid id)
      pass
  _files.clear()


# no handle is ever inherited by a forked process (a DataLoader worker): the parent closes its own before the fork, the child
# starts from an empty table (and would drop a stale one: _files_pid)
os.register_at_fork(before=close_packed_files, after_in_child=_files.clear)


def packed_file(path):
  """The packed file at path, opened read-only once per process (lazily: in a DataLoader worker, by the worker). File locking as
  the environment says (HDF5_USE_FILE_LOCKING=FALSE works)."""
  if _files_pid[0] != os.getpid():
    _files.clear()
    _files_pid[0] = os.getpid()
  f = _files.get(path)
  if f is None or not f.id.valid:
    f = _files[path] = h5py.File(path, "r")
  return f


def is_packed(path):
  """path is a packed grids file (an HDF5 file with root attr layout="packed"); False for a directory or anything else."""
  if not os.path.isfile(path):
    return False
  try:
    with h5py.File(path, "r") as f:
      return f.attrs.get("layout") == PACKED_LAYOUT
  except OSError:
    return False


def read_packed_index(path):
  """A packed file -> ({primary_view_row: GridRef}, source_h5 (the grids' own attribute), grid_shape (H, W, L)): from its /rows and
  root attrs only (no group is opened)."""
  with h5py.File(path, "r") as f:
    if f.attrs.get("layout") != PACKED_LAYOUT:
      raise ValueError(f"{path}: not a packed grids file (layout {f.attrs.get('layout')!r})")
    if int(f.attrs["layout_version"]) != PACKED_VERSION:
      raise ValueError(f"{path}: unknown packed layout version {f.attrs['layout_version']!r}")
    rows = f["rows"][()]
    n = int(f.attrs["n_grids"])
    if len(rows) != n or len(f["grids"]) != n:
      raise ValueError(f"{path}: {n} grids in its attrs, {len(rows)} rows, {len(f['grids'])} groups")
    path = os.path.abspath(path)
    return ({int(r): GridRef(path, group_name(r)) for r in rows}, str(f.attrs["source_h5"]),
            tuple(int(v) for v in f.attrs["grid_shape"]))


@contextlib.contextmanager
def _open(src):
  """A grid file path or a GridRef -> the h5py object holding the grid (a file, or the packed file's group; that file stays open)."""
  if isinstance(src, GridRef):
    yield packed_file(src.path)[src.group]
  else:
    with h5py.File(src, "r") as f:
      yield f


def read_grid(path):
  """A finalized grid file of either layout, or a packed file's grid (a GridRef) -> Grid. A sparse file is read as rows (no dense
  arrays); a dense file is read whole and gathered (as encode_grid always did)."""
  with _open(path) as f:
    if is_sparse(f):
      return _read_sparse(f)
    return Grid.from_dense({k: np.asarray(f[k]) for k in DENSE_DATASETS}, _attrs(f))


def load_grid(path):
  """A finalized grid file of either layout (or a GridRef) -> ({name: array (H, W, L, .)}, attrs): for a dense file exactly what it
  holds, for a sparse one the same (NaN at the invalid slots, the dense file's attrs)."""
  with _open(path) as f:
    if is_sparse(f):
      g = _read_sparse(f)
      return g.dense(), g.attrs
    return {k: np.asarray(f[k]) for k in DENSE_DATASETS}, _attrs(f)


def grid_shape(path):
  """(H, W, L) of a grid file of either layout (or a GridRef), from its metadata only."""
  with _open(path) as f:
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


def convert_file(dense_path, sparse_path, fsync=False):
  """dense_path's grid as a sparse file at sparse_path: written to sparse_path + ".tmp", verified (verify_sparse), then renamed.
  Raises (and removes the temp file) when the copy is not exact. The dense file is left alone. dense_path may also be an open
  binary file object (e.g. an in-memory io.BytesIO dense file: fit_anchored.py's, which never writes a dense file to disk).
  fsync: fsync the temp file before the rename (a delayed write error, e.g. a full disk, then raises instead of leaving a
  truncated file in place)."""
  tmp = sparse_path + ".tmp"
  try:
    with h5py.File(dense_path, "r") as f:
      if is_sparse(f):
        raise ValueError(f"{dense_path} is already sparse")
      write_sparse(tmp, Grid.from_dense({k: np.asarray(f[k]) for k in DENSE_DATASETS}), src=f)
    msg = verify_sparse(dense_path, tmp)
    if msg:
      raise RuntimeError(f"{dense_path}: the sparse copy is not exact: {msg}")
    if fsync:
      fd = os.open(tmp, os.O_RDONLY)
      try:
        os.fsync(fd)
      finally:
        os.close(fd)
    os.replace(tmp, sparse_path)
  finally:
    if os.path.exists(tmp):
      os.remove(tmp)


# ---------------------------------------------------------------------------------------------------------------- packing

DATASET_PROPS = ("dtype", "shape", "compression", "compression_opts", "shuffle", "fletcher32", "scaleoffset")


def write_packed_grid(group, src):
  """src (an open sparse grid file) into group (empty, of the packed file being written): its root attrs (with their HDF5 types),
  the five fields and any other dataset copied as they are (H5Ocopy: compressed chunks, filters, attrs), layer_valid_lhw as one
  chunk (L, H, W) with its own filters, layer_offsets COMPACT (no small raw-data allocation: see the module docstring)."""
  if not is_sparse(src):
    raise ValueError(f"{_where(src)}: not a sparse grid file (convert_grids_sparse.py it first)")
  _copy_attrs(src, group)
  for k in src:
    if k in ("layer_valid_lhw", "layer_offsets"):
      continue
    src.copy(src[k], group, name=k)
  m = src["layer_valid_lhw"]
  d = group.create_dataset("layer_valid_lhw", data=m[()], chunks=m.shape, compression=m.compression,
                           compression_opts=m.compression_opts, shuffle=m.shuffle)
  _copy_attrs(m, d)
  o = src["layer_offsets"]
  dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
  dcpl.set_layout(h5py.h5d.COMPACT)
  d = group.create_dataset("layer_offsets", data=o[()], dcpl=dcpl)
  _copy_attrs(o, d)


def verify_packed_grid(src, group):
  """None if group (a packed file's grid) holds exactly src's grid (an open sparse file), else what differs: the attrs (names, HDF5
  types, values), the dataset names, per dataset dtype / shape / filters / attrs and its decoded values bit for bit; every dataset
  one chunk (or none: an empty field)."""
  msg = _same_attrs(src, group)
  if msg:
    return msg
  if set(src) != set(group):
    return f"datasets differ: {sorted(set(src) ^ set(group))}"
  for k in src:
    a, b = src[k], group[k]
    if not isinstance(a, h5py.Dataset) or not isinstance(b, h5py.Dataset):
      return f"{k}: not a dataset"
    for p in DATASET_PROPS:
      if getattr(a, p) != getattr(b, p):
        return f"{k}: {p} differs ({getattr(a, p)!r} vs {getattr(b, p)!r})"
    if b.chunks not in (None, b.shape):
      return f"{k}: {b.chunks} chunks, not one"
    if not _same(a[()], b[()]):
      return f"{k}: values differ"
    msg = _same_attrs(a, b)
    if msg:
      return f"{k}: {msg}"
  return None


def verify_packed(grids_dir, packed, names=None):
  """packed against the sparse files of grids_dir it was made from: the index (/rows, /names, root attrs), then every grid
  (verify_packed_grid). -> list of problems ([] = every grid bit-identical). names: the files expected (default: all *.h5)."""
  names = sorted(names if names is not None else (os.path.basename(p) for p in glob.glob(os.path.join(grids_dir, "*.h5"))))
  problems = []
  with h5py.File(packed, "r") as f:
    rows = f["rows"][()]
    fnames = f["names"].asstr()[()]
    if int(f.attrs["n_grids"]) != len(names) or sorted(fnames) != names or len(f["grids"]) != len(names):
      return [f"the packed file holds {int(f.attrs['n_grids'])} grids / {len(f['grids'])} groups ({len(fnames)} names), "
              f"{grids_dir} {len(names)} files"]
    if not (np.diff(rows) > 0).all():
      problems.append("/rows not strictly ascending")
    for row, name in zip(rows.tolist(), fnames.tolist()):
      with h5py.File(os.path.join(grids_dir, name), "r") as s:
        if int(s.attrs["primary_view_row"]) != row:
          problems.append(f"{name}: primary_view_row {int(s.attrs['primary_view_row'])} but packed as row {row}")
          continue
        if str(s.attrs["source_h5"]) != str(f.attrs["source_h5"]):
          problems.append(f"{name}: source_h5 differs from the packed file's")
        msg = verify_packed_grid(s, f[group_name(row)])
        if msg:
          problems.append(f"{name} (row {row}): {msg}")
  return problems


def _extents(d):
  """[(byte offset, size)] of a dataset's raw data in the file (none: compact or empty)."""
  if d.chunks:
    return [(c.byte_offset, c.size) for c in (d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks()))]
  off = d.id.get_offset()
  return [(off, d.id.get_storage_size())] if off is not None and d.id.get_storage_size() else []


def grid_extents(packed):
  """{row: (first byte, end byte, bytes of data)} of every grid's raw data (its datasets' chunks / contiguous storage) in packed,
  and the rows whose byte range overlaps another grid's (a grid is one contiguous byte range when it overlaps none)."""
  out = {}
  with h5py.File(packed, "r") as f:
    for row in f["rows"][()].tolist():
      g = f[group_name(row)]
      ex = [e for k in g for e in _extents(g[k])]
      if ex:
        out[row] = (min(o for o, _ in ex), max(o + n for o, n in ex), sum(n for _, n in ex))
  spans = sorted((lo, hi, r) for r, (lo, hi, _) in out.items())
  overlapping = set()
  end, end_row = -1, None
  for lo, hi, r in spans:
    if lo < end:
      overlapping |= {r, end_row}
    if hi > end:
      end, end_row = hi, r
  return out, sorted(overlapping)


def _fsync(path, directory=False):
  fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if directory else 0))
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def pack_dir(grids_dir, out, progress=None):
  """The sparse grid files of grids_dir (*.h5) as one packed file at out: written grid by grid (ascending primary_view_row) to
  out + ".tmp", closed, verified against every source file (verify_packed), fsynced, then moved to out (os.link: never over an
  existing file) and the temp removed. Raises if out exists (a packed file is never modified or replaced: delete it by hand), if
  grids_dir holds a leftover *.tmp (a fit still writing), if a file is not sparse, if the files disagree on source_h5 / grid_shape
  or share a row, or if the verification finds any difference (the temp is removed then). A temp left by a killed earlier run is
  removed first. -> {n_grids, bytes, source_bytes, overlapping (rows whose bytes are not one contiguous range), seconds}.
  progress: called with (grids written, total)."""
  import time
  t0 = time.perf_counter()
  grids_dir, out = os.path.abspath(grids_dir), os.path.abspath(out)
  if not os.path.isdir(grids_dir):
    raise ValueError(f"{grids_dir}: not a directory")
  tmp = out + ".tmp"
  if os.path.exists(out):
    if os.path.exists(tmp) and os.path.samefile(out, tmp):   # killed between the link and the temp's removal: out is finished
      os.remove(tmp)
    raise FileExistsError(f"{out} exists: a packed file is never overwritten (delete it by hand to pack again)")
  if os.path.realpath(os.path.dirname(out)) == os.path.realpath(grids_dir):
    raise ValueError(f"{out}: inside the grids dir it packs (its *.h5 would be taken for a grid)")
  leftovers = sorted(glob.glob(os.path.join(grids_dir, "*.tmp")))
  if leftovers:
    raise ValueError(f"{grids_dir}: {len(leftovers)} *.tmp files (a writer still running or killed?), e.g. {leftovers[0]}")
  names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(grids_dir, "*.h5")))
  if not names:
    raise ValueError(f"no *.h5 in {grids_dir}")
  rows, sources, shapes = {}, set(), set()
  for n in names:
    with h5py.File(os.path.join(grids_dir, n), "r") as f:
      if not is_sparse(f):
        raise ValueError(f"{n}: not a sparse grid file (convert_grids_sparse.py the dir first)")
      r = int(f.attrs["primary_view_row"])
      if r in rows:
        raise ValueError(f"{n} and {rows[r]}: both row {r}")
      rows[r] = n
      sources.add(str(f.attrs["source_h5"]))
      shapes.add(tuple(int(v) for v in f.attrs["grid_shape"]))
  if len(sources) != 1 or len(shapes) != 1:
    raise ValueError(f"{grids_dir}: the grids disagree on source_h5 {sorted(sources)} / grid_shape {sorted(shapes)}")
  order = sorted(rows)
  if os.path.exists(tmp):
    log.info("removing %s (left by an interrupted run)", tmp)
    os.remove(tmp)
  try:
    with h5py.File(tmp, "x") as f:
      f.attrs["layout"] = PACKED_LAYOUT
      f.attrs["layout_version"] = PACKED_VERSION
      f.attrs["n_grids"] = np.int64(len(order))
      f.attrs["source_h5"] = sources.pop()
      f.attrs["grid_shape"] = np.array(shapes.pop(), np.int64)
      f.attrs["packed_from"] = grids_dir
      f.create_dataset("rows", data=np.array(order, np.int64))
      f.create_dataset("names", data=np.array([rows[r] for r in order], dtype=object), dtype=h5py.string_dtype())
      grids = f.create_group("grids")
      for i, r in enumerate(order, 1):
        with h5py.File(os.path.join(grids_dir, rows[r]), "r") as s:
          write_packed_grid(grids.create_group(str(r)), s)
        if progress:
          progress(i, len(order))
    problems = verify_packed(grids_dir, tmp, names)
    if problems:
      raise RuntimeError(f"{tmp}: {len(problems)} grids are not exact copies, e.g. {problems[:3]}")
    _, overlapping = grid_extents(tmp)
    _fsync(tmp)
    os.link(tmp, out)   # (fails if out appeared meanwhile: never replaced)
    os.remove(tmp)
    _fsync(os.path.dirname(out), directory=True)
  finally:
    if os.path.exists(tmp):
      os.remove(tmp)
  return dict(n_grids=len(order), bytes=os.path.getsize(out), overlapping=overlapping,
              source_bytes=sum(os.path.getsize(os.path.join(grids_dir, n)) for n in names), seconds=time.perf_counter() - t0)
