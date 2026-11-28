#!/usr/bin/env python3
"""Pack a directory of sparse grid files (fit_anchored.py's <root>/grids_sparse/, convert_grids_sparse.py's output) into ONE packed
file per shard (grid_io's packed layout: /grids/<primary_view_row>/ per grid, the files' datasets and attrs as they are).

Kill-safe (grid_io.pack_dir): written to <out>.tmp grid by grid, closed, then EVERY grid verified against its source file (attrs
with their HDF5 types, dataset dtypes / shapes / filters / attrs, decoded values bit for bit), fsynced, and moved to <out> (never
over an existing file). A finished <out> is never touched: the script refuses to run when it exists (delete it by hand to pack
again). A <out>.tmp left by a killed run is removed and the packing starts over. The source files are only read.

    python pack_grids.py grids_dir=/scratch/vc18/t26rb00/shard3/anchored_wd0.3_rbg/grids_sparse \
        out=/scratch/vc18/t26rb00/shard3/anchored_wd0.3_rbg/grids.h5
    -> train with diffusion.grids_dir=<out>, or data.shards=[{source_h5:...,grids_dir:<out>},...]

verify_only=true: only verify an existing <out> against grids_dir (as the packing does before its rename) and report the contiguity
(grid_io.grid_extents); exit 1 on any difference. The x0 moments cache is not carried over: the first run on a packed file computes
its moments once (<out dir>/.x0_moments/<tag>/<out name>/row<R>.npz).
"""

import logging
import os
import sys
import time

import hydra
from omegaconf import DictConfig

import grid_io

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="pack_grids")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  grids_dir, out = os.path.abspath(str(cfg.grids_dir)), os.path.abspath(str(cfg.out))
  if cfg.verify_only:
    t = time.perf_counter()
    problems = grid_io.verify_packed(grids_dir, out)
    ext, overlapping = grid_io.grid_extents(out)
    log.info("verify %s against %s: %d grids, %d problems (%.0f s); %d grids not one contiguous byte range", out, grids_dir,
             len(ext), len(problems), time.perf_counter() - t, len(overlapping))
    for p in problems[:20]:
      log.info("  %s", p)
    if problems:
      sys.exit(1)
    log.info("VERIFY_OK")
    return
  t0 = time.perf_counter()

  def progress(i, n):
    if i % 250 == 0 or i == n:
      log.info("  %d / %d grids written (%.0f s)", i, n, time.perf_counter() - t0)

  try:
    r = grid_io.pack_dir(grids_dir, out, progress=progress)
  except (FileExistsError, ValueError) as e:
    raise SystemExit(str(e))
  log.info("packed %d grids: %s -> %s, %.2f GB (the files: %.2f GB, %.4fx), verified bit-identical, %.0f s", r["n_grids"],
           grids_dir, out, r["bytes"] / 1e9, r["source_bytes"] / 1e9, r["bytes"] / r["source_bytes"], r["seconds"])
  log.info("contiguity: %d grids not one contiguous byte range%s", len(r["overlapping"]),
           f" (rows {r['overlapping'][:10]}...)" if r["overlapping"] else "")
  log.info("PACK_OK")


if __name__ == "__main__":
  main()
