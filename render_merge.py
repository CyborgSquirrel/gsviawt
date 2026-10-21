"""Merge the part files of queue-mode render_objaverse.py workers into one render h5.

    scan = scan_parts(part_paths)               # which meshes the parts hold, which parts to distrust
    merge_parts(scan, meshes, "final.h5")       # rows in mesh-index order, same layout as a single run

A part is skipped ("cooked") if its write marker (util.txn_lock_path) exists -- a writer died or
failed mid-write -- or if h5py can't open it. A mesh found in several parts (a worker died after
committing it but before marking its job done, so another worker rendered it again) is taken from
the first part in the order given. Every mesh's rows must be contiguous in its part, which they are
because a mesh is one transaction.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

from util import txn_lock_path

logger = logging.getLogger(__name__)


@dataclass
class PartScan:
  segments: dict = field(default_factory=dict)   # mesh index -> (part path, first row, end row)
  cooked: list = field(default_factory=list)     # parts that were skipped, with the reason
  failed: set = field(default_factory=set)       # mesh indices a worker recorded as failed
  duplicates: int = 0                            # meshes seen in more than one part


def scan_parts(part_paths):
  scan = PartScan()
  for path in part_paths:
    if os.path.exists(txn_lock_path(path)):
      scan.cooked.append((path, "write marker present"))
      continue
    try:
      with h5py.File(path, "r") as f:
        if "failed_mesh_indices" in f.attrs:
          scan.failed |= set(json.loads(f.attrs["failed_mesh_indices"]))
        mesh_index = f["mesh_index"][()] if "mesh_index" in f else np.zeros(0, np.int64)
    except Exception as e:
      scan.cooked.append((path, f"can't read: {e!r}"))
      continue
    for mi in np.unique(mesh_index):
      rows = np.flatnonzero(mesh_index == mi)
      if not (np.diff(rows) == 1).all():
        raise ValueError(f"{path}: rows of mesh {mi} aren't contiguous")
      if int(mi) in scan.segments:
        scan.duplicates += 1
        continue
      scan.segments[int(mi)] = (path, int(rows[0]), int(rows[-1]) + 1)
  return scan


def missing_meshes(scan, num_meshes, failed=()):
  """Mesh indices that are in no good part and weren't recorded as failed."""
  return sorted(set(range(num_meshes)) - set(scan.segments) - set(scan.failed) - set(failed))


def merge_parts(scan, meshes, out_path, failed=()):
  """Write the merged file (to out_path + '.tmp', then renamed). `failed` = mesh indices to record
  as failed in addition to the ones the parts recorded. Every mesh must be present or failed."""
  failed = sorted(set(scan.failed) | set(failed))
  missing = missing_meshes(scan, len(meshes), failed)
  if missing:
    raise ValueError(f"{len(missing)} meshes are in no part and not failed: {missing[:20]}")
  order = sorted(scan.segments)
  segs = [scan.segments[mi] for mi in order]
  total = sum(b - a for _, a, b in segs)
  first_path = segs[0][0]

  with h5py.File(first_path, "r") as first, h5py.File(out_path + ".tmp", "w") as out:
    cfg = json.loads(first.attrs["config_json"])
    cfg.update(output_path=out_path, queue_path=None, queue_worker=None, queue_max_meshes=None,
               mesh_strategy={"_target_": "mesh_strategy.ListMeshes", "paths": list(meshes)})
    out.attrs["config_json"] = json.dumps(cfg)
    if failed:
      out.attrs["failed_mesh_indices"] = json.dumps(failed)
    out.create_dataset("mesh_paths", data=list(meshes), dtype=h5py.string_dtype("utf-8"))

    names = [k for k in first if k != "mesh_paths"]
    dst = {}
    for name in names:
      s = first[name]
      chunks = None if s.chunks is None else (min(s.chunks[0], total), *s.chunks[1:])
      dst[name] = out.create_dataset(
        name, shape=(total, *s.shape[1:]), dtype=s.dtype, chunks=chunks,
        compression=s.compression, compression_opts=s.compression_opts)

    o = 0
    for path, a, b in segs:
      with h5py.File(path, "r") as f:
        if sorted(k for k in f if k != "mesh_paths") != sorted(names):
          raise ValueError(f"{path} has datasets {sorted(f)}, expected {sorted(names)}")
        for name in names:
          s, d = f[name], dst[name]
          if s.shape[1:] != d.shape[1:] or s.dtype != d.dtype:
            raise ValueError(f"{path}:{name} is {s.shape}/{s.dtype}, expected (*, {d.shape[1:]})/{d.dtype}")
          if s.chunks is not None and s.chunks == d.chunks and s.chunks[0] == 1 and s.compression:
            for r in range(b - a):   # one chunk per row: copy the compressed chunk as it is
              idx = (a + r,) + (0,) * (s.ndim - 1)
              filters, buf = s.id.read_direct_chunk(idx)
              d.id.write_direct_chunk((o + r,) + (0,) * (s.ndim - 1), buf, filters)
          else:
            d[o:o + b - a] = s[a:b]
      o += b - a
  os.replace(out_path + ".tmp", out_path)
  logger.info("wrote %s: %d rows, %d meshes, %d failed", out_path, total, len(order), len(failed))
  return {"rows": total, "meshes": len(order), "failed": failed}
