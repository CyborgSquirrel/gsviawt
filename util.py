import contextlib as ctl
import logging
import time

import h5py
import numpy as np

log = logging.getLogger(__name__)


_MISSING = object()


def intrinsics_name(f, kind="depth", default=_MISSING):
  """Name of the intrinsics dataset to read from a render_objaverse h5.

  Current files carry both `depth_intrinsics` (matches `depth_peel`) and
  `image_intrinsics` (matches `images`). Older files have a single
  `camera_intrinsics` that matched `depth_peel` (and, back then, `images`
  too since the two resolutions were always equal) -- fall back to that.
  `kind` is "depth" or "image". Raises KeyError if neither is present,
  unless `default` is given.
  """
  want = f"{kind}_intrinsics"
  if want in f:
    return want
  if "camera_intrinsics" in f:
    return "camera_intrinsics"
  if default is not _MISSING:
    return default
  raise KeyError(f"no {want} (or legacy camera_intrinsics) in {getattr(f, 'filename', f)!r}")


@ctl.contextmanager
def timed(label: str):
  """Logs the wall-clock time of the block immediately when it exits, so a
  crash mid-block still leaves prior timings on record instead of losing
  them in an end-of-loop summary.
  """
  t0 = time.perf_counter()
  try:
    yield
  finally:
    log.info("%s: %.3fs", label, time.perf_counter() - t0)


class LazyDataset:
  def __init__(
    self,
    f: h5py.File,
    name: str,
    *,
    dataset_kwargs=None,
    init_hook=None,
  ):
    self.f = f
    self.name = name
    if dataset_kwargs is None:
      dataset_kwargs = {}
    self.dataset_kwargs = dataset_kwargs
    self.init_hook = init_hook
    self.entered = False
    self.initted = False

  def __enter__(self):
    if self.entered:
      raise RuntimeError(f"Tried to enter {repr(type(self))} twice")
    self.entered = True
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.entered = False
    if not self.initted:
      log.error("Exited without initializing dataset %r", self.name)
      return
    self.dataset.resize(self.idx, axis=0)

  def _ensure_initted(self, *, shape=None, dtype=None):
    if self.initted:
      return
    if shape is None or dtype is None:
      raise ValueError()

    if self.init_hook is not None:
      self.init_hook(self, shape=shape, dtype=dtype)

    if (
      self.dataset_kwargs.get("maxshape") is not None
      or self.dataset_kwargs.get("dtype") is not None
      or self.dataset_kwargs.get("shape") is not None
    ):
      raise ValueError()

    self.dataset = self.f.create_dataset(
      self.name,
      data=np.empty((1, *shape)),
      **self.dataset_kwargs,
      maxshape=(None, *shape),
      dtype=dtype,
    )
    self.idx = 0
    self.initted = True

  def append(self, item):
    if isinstance(item, np.ndarray):
      shape = item.shape
      dtype = item.dtype
    else:
      shape = ()
      dtype = type(item)
    self._ensure_initted(shape=shape, dtype=dtype)

    if self.idx >= len(self.dataset):
      self.dataset.resize(len(self.dataset)*2, axis=0)

    self.dataset[self.idx] = item
    self.idx += 1


class TransactionalWriter:
  """Appends rows to resizable datasets of an open h5py.File, a transaction at a time.

  w = TransactionalWriter(f)
  ds = w.init("depth", dataset_kwargs=dict(compression="gzip"))   # or w.append("depth", row)
  with w.transaction():
    ds.append(row)       # only cached in memory
  # leaving the block writes every cached row to the file, then flushes it; if the block
  # raises, the cache is dropped and the file is untouched

  append() outside a transaction raises. Datasets are created on the first transaction that
  appends to them (shape/dtype come from the first row); init_hook(ds, shape=, dtype=) may edit
  ds.dataset_kwargs just before that, as with LazyDataset."""

  def __init__(self, f: h5py.File):
    self.f = f
    self.datasets = {}
    self._cache = None

  def init(self, name, *, dataset_kwargs=None, init_hook=None):
    if name in self.datasets:
      raise ValueError(f"dataset {name!r} already registered")
    ds = TransactionalDataset(self, name, dataset_kwargs, init_hook)
    self.datasets[name] = ds
    return ds

  def append(self, name, item):
    if self._cache is None:
      raise RuntimeError(f"append to {name!r} outside a transaction")
    self._cache[name].append(item)

  @ctl.contextmanager
  def transaction(self):
    if self._cache is not None:
      raise RuntimeError("transactions don't nest")
    self._cache = {name: [] for name in self.datasets}
    try:
      yield self
    except BaseException:
      self._cache = None
      raise
    cache, self._cache = self._cache, None
    for name, rows in cache.items():
      if rows:
        self.datasets[name]._write(rows)
    self.f.flush()


class TransactionalDataset:
  """A dataset handed out by TransactionalWriter.init()."""

  def __init__(self, writer, name, dataset_kwargs, init_hook):
    self.writer = writer
    self.name = name
    self.dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
    self.init_hook = init_hook
    self.dataset = None

  def append(self, item):
    self.writer.append(self.name, item)

  def _write(self, rows):
    first = rows[0]
    if isinstance(first, np.ndarray):
      shape, dtype = first.shape, first.dtype
      block = np.stack(rows)
    else:
      shape, dtype = (), type(first)
      block = np.array(rows, dtype=dtype)

    if self.dataset is None:
      if self.init_hook is not None:
        self.init_hook(self, shape=shape, dtype=dtype)
      if any(self.dataset_kwargs.get(k) is not None for k in ("maxshape", "dtype", "shape")):
        raise ValueError(f"dataset_kwargs of {self.name!r} must not set maxshape/dtype/shape")
      self.dataset = self.writer.f.create_dataset(
        self.name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, **self.dataset_kwargs)

    n = len(self.dataset)
    self.dataset.resize(n + len(block), axis=0)
    self.dataset[n:] = block


def check_shape(x, pattern):
  """Validate x's shape against `pattern` (fixed axes raise on mismatch), return x."""
  import einops
  einops.parse_shape(x, pattern)
  return x

def collate_with_batch_size(features):
  import torch
  batch = torch.utils.data.default_collate(features)
  batch["batch_size"] = len(features)
  return batch

def pipe(a, *fns):
  for fn in fns:
    a = fn(a)
  return a

@ctl.contextmanager
def set_mode(model, mode):
  """
  Temporarily set a model to "train" or "eval" mode, restoring its
  original training/eval state on exit (even if an exception occurs).

  Usage:
    with set_mode(model, "eval"):
      output = model(x)
  """
  if mode not in ("train", "eval"):
    raise ValueError(f"Mode must be 'train' or 'eval', got {mode!r}")

  was_training = model.training
  model.train(mode == "train")
  try:
    yield
  finally:
    model.train(was_training)
