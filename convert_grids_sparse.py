#!/usr/bin/env python3
"""Convert a directory of dense finalized grids (finalize_fit_dataset.py's grids/) into the sparse layout (grid_io.py), in a NEW
directory: every *.h5 is written to <name>.h5.tmp, verified against its dense original (grid_io.verify_sparse: mask, every row and
attr bit for bit), then renamed. The dense originals are never touched. Files already in out_dir are skipped, so it can be stopped
and run again (a leftover .tmp is rewritten).

The x0 moments cache (<grids_dir>/.x0_moments[/<encoding tag>]/<name>.h5.npz, gs_diffusion._grid_moments) is carried over:
the rows are the dense file's bit for bit, so the moments are too; each entry still valid for its dense file (stamp = its size and
mtime) is copied into <out_dir>/.x0_moments with the stamp of the new file, so compute_stats does not redo the pass over the grids.
Stale entries are left behind (they would be recomputed anyway).

    python convert_grids_sparse.py grids_dir=/scratch/vc16/t26rb00/new100/anchored/grids workers=8
    -> /scratch/vc16/t26rb00/new100/anchored/grids_sparse   (then train with diffusion.grids_dir=<that>)
"""

import glob
import logging
import multiprocessing
import os
import time

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig

import grid_io

log = logging.getLogger(__name__)

MOMENTS = ".x0_moments"


def _stamp(path):
  st = os.stat(path)
  return np.array([st.st_size, st.st_mtime_ns], np.int64)


def carry_moments(grids_dir, out_dir, name):
  """Copies the moments cache entries of grids_dir/name that are valid for it into out_dir's cache, restamped for out_dir/name.
  -> (copied, stale) counts."""
  src_root, dst_root = os.path.join(grids_dir, MOMENTS), os.path.join(out_dir, MOMENTS)
  if not os.path.isdir(src_root):
    return 0, 0
  old, new = _stamp(os.path.join(grids_dir, name)), _stamp(os.path.join(out_dir, name))
  copied = stale = 0
  for tag in [""] + sorted(d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))):
    src = os.path.join(src_root, tag, name + ".npz")
    if not os.path.exists(src):
      continue
    with np.load(src) as z:
      entry = {k: z[k] for k in z.files}
    if not (entry["stamp"] == old).all():
      stale += 1
      continue
    entry["stamp"] = new
    dst = os.path.join(dst_root, tag, name + ".npz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".tmp", "wb") as f:
      np.savez(f, **entry)
    os.replace(dst + ".tmp", dst)
    copied += 1
  return copied, stale


def convert_one(args):
  """-> (name, status, dense bytes, sparse bytes, seconds, moments copied, moments stale); status: converted | skipped | error."""
  grids_dir, out_dir, name = args
  src, dst = os.path.join(grids_dir, name), os.path.join(out_dir, name)
  t = time.perf_counter()
  try:
    if os.path.exists(dst):
      with h5py.File(dst, "r") as f:
        if not grid_io.is_sparse(f):
          raise ValueError(f"{dst} exists and is not sparse: left alone")
      status = "skipped"
    else:
      grid_io.convert_file(src, dst)
      status = "converted"
    copied, stale = carry_moments(grids_dir, out_dir, name)
  except Exception as e:   # one bad file must not stop the others; reported at the end
    return name, f"error: {type(e).__name__}: {e}", os.path.getsize(src), 0, time.perf_counter() - t, 0, 0
  return name, status, os.path.getsize(src), os.path.getsize(dst), time.perf_counter() - t, copied, stale


@hydra.main(version_base=None, config_path="conf", config_name="convert_grids_sparse")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  grids_dir = os.path.abspath(cfg.grids_dir)
  out_dir = os.path.abspath(cfg.out_dir) if cfg.out_dir else grids_dir.rstrip("/") + "_sparse"
  if os.path.realpath(out_dir) == os.path.realpath(grids_dir):
    raise SystemExit("out_dir must not be grids_dir: the conversion never writes in place")
  names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(grids_dir, "*.h5")))
  if not names:
    raise SystemExit(f"no *.h5 in {grids_dir!r}")
  os.makedirs(out_dir, exist_ok=True)
  log.info("%d grids: %s -> %s (%d workers)", len(names), grids_dir, out_dir, cfg.workers)
  jobs = [(grids_dir, out_dir, n) for n in names]
  t0 = time.perf_counter()
  results = []
  with multiprocessing.get_context("spawn").Pool(cfg.workers) if cfg.workers > 1 else _Serial() as pool:
    for i, r in enumerate(pool.imap_unordered(convert_one, jobs, chunksize=4), 1):
      results.append(r)
      if r[1].startswith("error"):
        log.info("  %s: %s", r[0], r[1])
      if i % 100 == 0 or i == len(jobs):
        log.info("  %d / %d (%.0f s)", i, len(jobs), time.perf_counter() - t0)
  by = {s: [r for r in results if r[1] == s] for s in ("converted", "skipped")}
  errors = [r for r in results if r[1].startswith("error")]
  ok = by["converted"] + by["skipped"]
  dense_b, sparse_b = sum(r[2] for r in ok), sum(r[3] for r in ok)
  log.info("converted %d, skipped %d (already in %s), %d errors; moments cache: %d entries carried over, %d stale left behind",
           len(by["converted"]), len(by["skipped"]), out_dir, len(errors), sum(r[5] for r in results), sum(r[6] for r in results))
  if ok:
    log.info("size: dense %.2f GB -> sparse %.2f GB (%.3fx) over the %d grids in %s", dense_b / 1e9, sparse_b / 1e9,
             sparse_b / dense_b, len(ok), out_dir)
  if errors:
    raise SystemExit(f"{len(errors)} grids failed (listed above); run again to retry them")


class _Serial:
  """Pool's imap_unordered in this process (workers=1)."""

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def imap_unordered(self, fn, jobs, chunksize=1):
    return map(fn, jobs)


if __name__ == "__main__":
  main()
