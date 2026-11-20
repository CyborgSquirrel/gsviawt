"""The x0 moments cache (<grids_dir>/.x0_moments, gs_diffusion._grid_moments): written atomically (no partial file is ever at the
cache's name, no temporary left behind), read back, and an unreadable cache file (e.g. a partial one from older code) is recomputed
instead of crashing the run."""
import os
import types

import numpy as np
import torch

import gs_diffusion as gd


def _setup(tmp_path, monkeypatch):
  grid = tmp_path / "mesh0_view3.h5"
  grid.write_bytes(b"grid")
  hit = torch.zeros(6, 4, 4, dtype=torch.bool)
  hit[0, :2] = True
  calls = []

  def encode(g, source, rng=None, **kw):
    calls.append(1)
    x = np.zeros((gd.CHANNELS, 6, 4, 4), np.float32)
    x[:, 0, :2] = np.arange(gd.CHANNELS, dtype=np.float32)[:, None, None]
    return x

  monkeypatch.setattr(gd, "read_grid", lambda path: None)
  monkeypatch.setattr(gd, "encode_grid", encode)
  ds = types.SimpleNamespace(encoding=gd.normalize_encoding(None),
                             inner=[{"source": {"hit": hit}, "source_view": 3}])
  return ds, str(grid), str(tmp_path / ".x0_moments"), calls


def test_cache_written_atomically_and_read_back(tmp_path, monkeypatch):
  ds, grid, cache_dir, calls = _setup(tmp_path, monkeypatch)
  saved = []
  real_savez = np.savez
  monkeypatch.setattr(np, "savez", lambda path, **kw: saved.append(str(path)) or real_savez(path, **kw))
  first = gd._grid_moments(ds, 0, grid, cache_dir)
  assert len(calls) == 1 and first[2] == 8
  assert saved and not saved[0].endswith("mesh0_view3.h5.npz")                   # never written at the cache's own name
  assert os.listdir(cache_dir) == ["mesh0_view3.h5.npz"]                           # the temporary renamed, nothing left over
  again = gd._grid_moments(ds, 0, grid, cache_dir)
  assert len(calls) == 1                                                           # from the cache
  assert all(np.array_equal(a, b) for a, b in zip(first, again))


def test_unreadable_cache_is_recomputed(tmp_path, monkeypatch):
  ds, grid, cache_dir, calls = _setup(tmp_path, monkeypatch)
  os.makedirs(cache_dir)
  with open(os.path.join(cache_dir, "mesh0_view3.h5.npz"), "wb") as f:
    f.write(b"PK\x03\x04 truncated")                                              # a partial zip
  out = gd._grid_moments(ds, 0, grid, cache_dir)
  assert len(calls) == 1 and out[2] == 8
  gd._grid_moments(ds, 0, grid, cache_dir)
  assert len(calls) == 1                                                           # the rewritten cache is good
