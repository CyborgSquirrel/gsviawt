import contextlib as ctl
import logging
import time

import numpy as np
import h5py

log = logging.getLogger(__name__)


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

