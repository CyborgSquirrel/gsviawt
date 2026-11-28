"""Merge the part files of queue-mode render_objaverse.py workers into one render h5.

    scan = scan_parts(part_paths)               # which meshes the parts hold, which parts to distrust
    merge_parts(scan, meshes, "final.h5")       # rows in mesh-index order, same layout as a single run
    finish_merge(scan, meshes, "final.h5", parts_dir)   # kill-safe: merge (unless done), then delete the parts

A queue-mode worker writes one part per mesh (part_path: parts/mesh<idx>.h5, written as a temp file, fsynced, renamed), so a
kill loses at most the mesh being written. The merge is kill-safe: written to final.h5.tmp, verified against every part
(verify_merged: row / mesh counts, every mesh's rows equal to its part's), fsynced, renamed to final.h5, the directory
fsynced -- and only then are the parts deleted. A final file therefore exists only complete and verified: rerun after a
kill at any point, finish_merge redoes a merge whose .tmp was left behind, or just deletes the leftover parts.

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

from util import fsync_path, txn_lock_path

logger = logging.getLogger(__name__)


def part_path(parts_dir, mesh_index):
  """Queue mode: the file holding one rendered mesh."""
  return os.path.join(parts_dir, f"mesh{int(mesh_index):06d}.h5")


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
  """Write the merged file: to out_path + '.tmp', verified against the parts (verify_merged), fsynced, renamed to out_path,
  the directory fsynced. `failed` = mesh indices to record as failed in addition to the ones the parts recorded. Every mesh
  must be present or failed. A leftover .tmp (a killed merge) is overwritten."""
  failed = sorted(set(scan.failed) | set(failed))
  missing = missing_meshes(scan, len(meshes), failed)
  if missing:
    raise ValueError(f"{len(missing)} meshes are in no part and not failed: {missing[:20]}")
  order = sorted(scan.segments)
  segs = [scan.segments[mi] for mi in order]
  total = sum(b - a for _, a, b in segs)
  first_path = segs[0][0]
  tmp = out_path + ".tmp"

  with h5py.File(first_path, "r") as first, h5py.File(tmp, "w") as out:
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
          if _direct_copy(s, d):
            for r in range(b - a):   # one chunk per row: copy the compressed chunk as it is
              filters, buf = s.id.read_direct_chunk(_row_chunk(s, a + r))
              d.id.write_direct_chunk(_row_chunk(d, o + r), buf, filters)
          else:
            d[o:o + b - a] = s[a:b]
      o += b - a
  problems = verify_merged(tmp, scan, meshes, failed)
  if problems:
    raise RuntimeError(f"{tmp}: {len(problems)} differences from the parts, e.g. {problems[:3]} (the parts are kept)")
  fsync_path(tmp)
  os.replace(tmp, out_path)
  fsync_path(os.path.dirname(os.path.abspath(out_path)), directory=True)
  logger.info("wrote %s: %d rows, %d meshes, %d failed (verified against the parts)", out_path, total, len(order), len(failed))
  return {"rows": total, "meshes": len(order), "failed": failed}


def _direct_copy(s, d):
  return s.chunks is not None and s.chunks == d.chunks and s.chunks[0] == 1 and bool(s.compression)


def _row_chunk(ds, row):
  return (row,) + (0,) * (ds.ndim - 1)


def verify_merged(path, scan, meshes, failed=()):
  """Differences between a merged file and the parts it was made from (empty = exact): mesh_paths, failed_mesh_indices,
  the row count, and every mesh's rows of every dataset (the stored chunks byte for byte where the merge copied them as they
  are, else the decoded values)."""
  problems = []
  order = sorted(scan.segments)
  with h5py.File(path, "r") as out:
    if out["mesh_paths"].asstr()[()].tolist() != list(meshes):
      problems.append("mesh_paths differ")
    got_failed = json.loads(out.attrs["failed_mesh_indices"]) if "failed_mesh_indices" in out.attrs else []
    if sorted(got_failed) != sorted(failed):
      problems.append(f"failed_mesh_indices {got_failed} != {sorted(failed)}")
    names = [k for k in out if k != "mesh_paths"]
    total = sum(b - a for _, a, b in scan.segments.values())
    for name in names:
      if out[name].shape[0] != total:
        problems.append(f"{name}: {out[name].shape[0]} rows, expected {total}")
    if problems:
      return problems
    o = 0
    for mi in order:
      path_p, a, b = scan.segments[mi]
      with h5py.File(path_p, "r") as f:
        for name in names:
          s, d = f[name], out[name]
          if _direct_copy(s, d):
            same = all(s.id.read_direct_chunk(_row_chunk(s, a + r)) == d.id.read_direct_chunk(_row_chunk(d, o + r))
                       for r in range(b - a))
          else:
            same = np.array_equal(s[a:b], d[o:o + b - a], equal_nan=s.dtype.kind == "f")
          if not same:
            problems.append(f"mesh {mi}: {name} differs from {path_p}")
      o += b - a
  return problems


def finish_merge(scan_fn, meshes, out_path, parts_dir, failed=()):
  """Kill-safe end of a render: merge the parts into out_path unless it exists already (a final file exists only complete and
  verified, see merge_parts), then delete parts_dir (the parts, their temp files and write markers). scan_fn() -> the PartScan
  (only called when a merge is needed). Returns merge_parts' report, or None when out_path existed already."""
  report = None
  if os.path.exists(out_path):
    logger.info("%s exists already (complete and verified): only removing the leftover parts", out_path)
  else:
    report = merge_parts(scan_fn(), meshes, out_path, failed=failed)
  if os.path.isdir(parts_dir):
    leftovers = sorted(os.listdir(parts_dir))
    for name in leftovers:
      os.remove(os.path.join(parts_dir, name))
    os.rmdir(parts_dir)
    logger.info("removed %d part files", len(leftovers))
  return report
