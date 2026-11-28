#!/usr/bin/env python3
"""Continuous 3DGS dataset builder: walks the TRELLIS CSV in row order and grows the dataset shard by shard (download ->
render -> fit + pack), until target_meshes. Config: conf/build_dataset.yaml (the renderer's config under `render`, the fitter's
under `fit`, named as in render_queue.py / fit_anchored.py, + the builder's own keys).

    python build_dataset.py                    # the builder (mode=build); rerun = continue, after a kill at any point
    python build_dataset.py mode=download      # the CPU-only downloader on its own (the builder starts one if none runs)
    python build_dataset.py mode=status        # progress summary (read-only)

Layout under out_root:
  glbs/<uid>.glb                the GLBs (kept); glbs/failed/<uid>.json = the downloader's report of a mesh it can't fetch
  bad_meshes.csv                meshes rejected for good (download / render), written only by the builder; never picked again
  manifest.csv                  the finished shards (render h5 + packed grids)
  shard<k>/meshes.csv           the shard's members in CSV order with their metadata (csv_row, uid, sha256, aesthetic_score,
                                captions, glb); append-only: a member rejected at download / render stays listed (and is in
                                bad_meshes.csv), its replacement -- the next CSV row -- is appended
  shard<k>/render.h5            the render (render_queue.py; meshes rejected at render are in its failed_mesh_indices)
  shard<k>/render_list.txt      the GLBs handed to render_queue (the members not rejected at download, in order)
  shard<k>/render_work/logs/    the Blender worker logs (queue.db lives there while the shard renders)
  shard<k>/fit/_anchored/mesh<M>/  regression.json, timing.json, DONE (fit_anchored.py shard mode)
  shard<k>/grids.h5             the packed grids (grid_io packed layout)
  shard<k>/logs/fit.log, timing.json (the builder's wall time per stage, accumulated over runs; killed segments not counted)
  logs/build.log, logs/download.log; tmp/ (TMPDIR of the builder; tmp/blender: the Blender workers', emptied on start)

Per shard (strictly sequential): pick members (the next CSV rows that are in no skip list, no shard and not in bad_meshes.csv)
-> wait for their GLBs (the downloader prefetches download_ahead shards ahead) -> render_queue on the members -> a member the
downloader or the renderer rejects goes to bad_meshes.csv and the next CSV row is appended, downloaded and rendered, until
shard_size meshes rendered -> merge (kill-safe) -> fit_anchored shard mode (fit every rendered mesh, pack into grids.h5) ->
manifest -> next shard. A fit error stops the builder (a mesh that rendered must fit: fix it, rerun; DONE meshes are skipped).
All state is derived from the config and the files above: no ledger. The target counts the meshes of the skip lists (the 300
existing ones): target_meshes - |skip-list uids| new meshes are built.

The downloader is a separate process (CPU only, never needs the GPU), single instance (a lock in glbs/). It reports a mesh it
can't fetch (not in objaverse's index, HTTP 4xx, sha256 mismatch after `retries` + 1 tries) by writing glbs/failed/<uid>.json;
the builder folds those into bad_meshes.csv, so the builder stays its only writer. Network errors are retried forever (with
backoff), never reported. The builder starts a downloader as its child when none runs (it then dies with the builder); run one
on its own (mode=download) to keep prefetching while the builder is preempted.
"""

import csv
import fcntl
import glob
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

import glb_download
import render_queue
from render_queue import _die_with_parent

log = logging.getLogger("build_dataset")

MEMBER_FIELDS = ("csv_row", "uid", "sha256", "aesthetic_score", "captions", "glb")
BAD_FIELDS = ("csv_row", "uid", "sha256", "shard", "stage", "reason", "log_path", "time")
MANIFEST_FIELDS = ("shard", "dir", "render_h5", "grids_h5", "meshes", "grids", "time")


# ------------------------------------------------------------------------------------------------------------ files

def read_csv(path):
  if not os.path.exists(path):
    return []
  with open(path, newline="") as f:
    return list(csv.DictReader(f))


def write_csv_atomic(path, fields, rows):
  """The whole file, rewritten: temp -> fsync -> rename (a kill leaves the old or the new file, never a torn line)."""
  tmp = f"{path}.tmp"
  with open(tmp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
      w.writerow({k: r.get(k, "") for k in fields})
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def write_json_atomic(path, obj):
  tmp = f"{path}.tmp"
  with open(tmp, "w") as f:
    json.dump(obj, f, indent=1)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def uid_of(path_or_url):
  """A mesh's uid: the GLB's file stem / the last component of the CSV's file_identifier."""
  return os.path.splitext(os.path.basename(str(path_or_url).rstrip("/")))[0]


class Lock:
  """An exclusive flock on a file (released when the process dies, also on SIGKILL)."""

  def __init__(self, path):
    self.path = path
    self.f = None

  def acquire(self, wait=0.0):
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    self.f = open(self.path, "a")
    t_end = time.time() + wait
    while True:
      try:
        fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
      except BlockingIOError:
        if time.time() >= t_end:
          self.f.close()
          self.f = None
          return False
        time.sleep(1)

  def release(self):
    if self.f is not None:
      self.f.close()   # (closing drops the flock)
      self.f = None

  def held_by_other(self):
    """Someone else holds it (probe: take and drop it at once)."""
    with open(self.path, "a") as f:
      try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError:
        return True
      fcntl.flock(f, fcntl.LOCK_UN)
      return False


# ------------------------------------------------------------------------------------------------------------ state

class Dataset:
  """The builder's state, derived from the config and the files under out_root (re-read on every call)."""

  def __init__(self, cfg):
    self.cfg = cfg
    self.root = os.path.abspath(str(cfg.out_root))
    self.glbs = os.path.join(self.root, "glbs")
    self.failed_dir = os.path.join(self.glbs, "failed")
    self.bad_path = os.path.join(self.root, "bad_meshes.csv")
    self.manifest_path = os.path.join(self.root, "manifest.csv")
    with open(str(cfg.csv), newline="") as f:
      self.rows = [{"csv_row": str(i), "uid": uid_of(r["file_identifier"]), "sha256": r["sha256"],
                    "aesthetic_score": r["aesthetic_score"], "captions": r["captions"]} for i, r in enumerate(csv.DictReader(f))]
    self.skip = set()
    for p in cfg.skip_meshes:
      with open(str(p)) as f:
        self.skip |= {uid_of(line.strip()) for line in f if line.strip()}
    self.new_target = max(int(cfg.target_meshes) - len(self.skip), 0)

  def shard_dir(self, k):
    return os.path.join(self.root, f"shard{k}")

  def shards(self):
    """Shard numbers with a directory (>= first_shard), sorted."""
    ks = []
    for d in glob.glob(os.path.join(self.root, "shard*")):
      tail = os.path.basename(d)[len("shard"):]
      if tail.isdigit() and int(tail) >= int(self.cfg.first_shard) and os.path.isdir(d):
        ks.append(int(tail))
    return sorted(ks)

  def members(self, k):
    return read_csv(os.path.join(self.shard_dir(k), "meshes.csv"))

  def bad(self):
    """{uid: row} of bad_meshes.csv."""
    return {r["uid"]: r for r in read_csv(self.bad_path)}

  def markers(self):
    """{uid: report} of the downloader's failure reports (glbs/failed/<uid>.json)."""
    out = {}
    for p in glob.glob(os.path.join(self.failed_dir, "*.json")):
      try:
        with open(p) as f:
          out[uid_of(p)] = json.load(f)
      except (OSError, ValueError):
        pass   # being written (the downloader writes it atomically, so only a race with a rename)
    return out

  def manifest(self):
    return {int(r["shard"]): r for r in read_csv(self.manifest_path)}

  def rendered(self, k):
    return os.path.exists(os.path.join(self.shard_dir(k), "render.h5"))

  def done_new(self):
    """New meshes in finished shards."""
    return sum(int(r["meshes"]) for r in self.manifest().values())

  def assigned(self, only_rendered=False):
    """uids that are members of a shard (only_rendered: of a shard whose render h5 exists)."""
    out = set()
    for k in self.shards():
      if not only_rendered or self.rendered(k):
        out |= {m["uid"] for m in self.members(k)}
    return out

  def candidates(self, exclude):
    """CSV rows in order whose uid is in no skip list and not in `exclude`."""
    for r in self.rows:
      if r["uid"] not in self.skip and r["uid"] not in exclude:
        yield r

  def download_window(self):
    """The CSV rows whose GLBs should be on disk: the next ceil(download_ahead x shard_size) rows beyond the rendered ones
    (rows in no skip list, not rejected, not a member of a rendered shard), fewer near the target."""
    bad = set(self.bad()) | set(self.markers())
    rendered = self.assigned(only_rendered=True)
    rendered_good = len(rendered - bad)
    needed = self.new_target - rendered_good
    if needed <= 0:
      return []
    S, ahead = int(self.cfg.shard_size), float(self.cfg.download_ahead)
    n = min(math.ceil(ahead * S), needed + math.ceil((ahead - 1) * S))
    out = []
    for r in self.candidates(bad | rendered):
      if len(out) >= n:
        break
      out.append(r)
    return out


# ------------------------------------------------------------------------------------------------------------ downloader

def run_downloader(cfg, ds, passes=None):
  """Keep the download window's GLBs on disk (see the module docstring); returns when the target is reached (or after
  `passes` passes over the window). The tries of a mesh are counted per downloader process."""
  lock = Lock(os.path.join(ds.glbs, ".downloader.lock"))
  if not lock.acquire():
    log.info("another downloader holds %s: exiting", lock.path)
    return
  try:
    _download_loop(cfg, ds, passes)
  finally:
    lock.release()


def _download_loop(cfg, ds, passes):
  for p in glob.glob(os.path.join(ds.glbs, "*.tmp")):
    os.remove(p)   # a fetch killed mid-way (we hold the lock: no other downloader writes)
  index = glb_download.local_index(cfg.local_glbs)
  hf = glb_download.hf_paths(cfg.objaverse_dir)
  log.info("downloader: %d GLBs in the local pools %s, %d uids in objaverse's index", len(index), list(cfg.local_glbs), len(hf))
  tries, backoff = {}, {}   # uid -> permanent failures so far; uid -> (next try time, delay)
  n_pass = 0
  while True:
    if ds.done_new() >= ds.new_target:
      log.info("downloader: target reached")
      return
    window = ds.download_window()
    todo = [r for r in window if not os.path.exists(glb_download.glb_path(ds.glbs, r["uid"]))]
    fetched = 0
    for r in todo:
      uid = r["uid"]
      if backoff.get(uid, (0, 0))[0] > time.time():
        continue
      t0 = time.time()
      try:
        glb_download.fetch(uid, r["sha256"], ds.glbs, index, hf, timeout=float(cfg.download_timeout))
        fetched += 1
        backoff.pop(uid, None)
        log.info("downloaded row %s %s (%.1f s)", r["csv_row"], uid, time.time() - t0)
      except glb_download.PermanentError as e:
        tries[uid] = tries.get(uid, 0) + 1
        if tries[uid] > int(cfg.retries):
          os.makedirs(ds.failed_dir, exist_ok=True)
          write_json_atomic(os.path.join(ds.failed_dir, f"{uid}.json"),
                            {"csv_row": r["csv_row"], "uid": uid, "sha256": r["sha256"], "reason": str(e), "time": time.time()})
          log.warning("row %s %s can't be fetched (%s): reported", r["csv_row"], uid, e)
        else:
          backoff[uid] = (time.time() + float(cfg.download_poll), float(cfg.download_poll))
          log.warning("row %s %s: %s (try %d of %d)", r["csv_row"], uid, e, tries[uid], int(cfg.retries) + 1)
      except Exception as e:   # transient (network, 5xx, timeout): later, with backoff, never reported
        delay = min(2 * backoff.get(uid, (0, 30))[1], 3600)
        backoff[uid] = (time.time() + delay, delay)
        log.warning("row %s %s: %r -- retrying in %d s", r["csv_row"], uid, e, delay)
    n_pass += 1
    if passes is not None and n_pass >= passes:
      return
    if not fetched:
      time.sleep(float(cfg.download_poll))


# ------------------------------------------------------------------------------------------------------------ builder

class Builder:
  def __init__(self, cfg, ds):
    self.cfg, self.ds = cfg, ds
    self.downloader = None   # our child downloader, if we started one
    self.dl_lock = Lock(os.path.join(ds.glbs, ".downloader.lock"))

  # --- bookkeeping
  def add_bad(self, rows):
    """Append rows (dicts with BAD_FIELDS) to bad_meshes.csv -- the builder is its only writer."""
    if rows:
      write_csv_atomic(self.ds.bad_path, BAD_FIELDS, read_csv(self.ds.bad_path) + rows)
      for r in rows:
        log.warning("bad mesh: row %s %s (shard %s, %s): %s", r["csv_row"], r["uid"], r["shard"], r["stage"], r["reason"])

  def fold_markers(self):
    """The downloader's failure reports -> bad_meshes.csv (stage download)."""
    bad, new = self.ds.bad(), []
    shard_of = {m["uid"]: k for k in self.ds.shards() for m in self.ds.members(k)}
    for uid, m in sorted(self.ds.markers().items()):
      if uid not in bad:
        new.append({"csv_row": m["csv_row"], "uid": uid, "sha256": m["sha256"], "shard": shard_of.get(uid, ""),
                    "stage": "download", "reason": m["reason"], "log_path": os.path.join(self.ds.root, "logs", "download.log"),
                    "time": m["time"]})
    self.add_bad(new)

  def ensure_downloader(self):
    """A downloader runs: an external one (holds the lock), or our child (started now if none)."""
    if self.downloader is not None and self.downloader.poll() is None:
      return
    if self.downloader is not None:
      log.warning("the downloader exited with %s: restarting it", self.downloader.returncode)
      self.downloader = None
    if self.dl_lock.held_by_other():
      return
    overrides = [o for o in HydraConfig.get().overrides.task if not o.startswith("mode=")]
    cmd = [sys.executable, os.path.abspath(__file__), *overrides, "mode=download"]
    self.downloader = subprocess.Popen(cmd, preexec_fn=_die_with_parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info("started the downloader (pid %d)", self.downloader.pid)
    time.sleep(2)

  def stop_downloader(self):
    if self.downloader is not None and self.downloader.poll() is None:
      self.downloader.terminate()
      try:
        self.downloader.wait(timeout=20)
      except subprocess.TimeoutExpired:
        self.downloader.kill()

  def add_time(self, k, stage, t0):
    path = os.path.join(self.ds.shard_dir(k), "timing.json")
    t = json.load(open(path)) if os.path.exists(path) else {}
    t[stage] = t.get(stage, 0.0) + time.time() - t0
    write_json_atomic(path, t)

  # --- configs of the stages
  def render_cfg(self, k):
    d = self.ds.shard_dir(k)
    rc = OmegaConf.create(OmegaConf.to_container(self.cfg.render, resolve=True))
    with open_dict(rc):
      rc.pop("hydra", None)
      rc.output_path = os.path.join(d, "render.h5")
      rc.queue_dir = os.path.join(d, "render_work")
      rc.queue_tmpdir = os.path.join(self.ds.root, "tmp", "blender")
      rc.queue_retry_failed = int(self.cfg.retries)
      rc.mesh_strategy = {"_target_": "mesh_strategy.ListMeshes", "paths": [], "file": os.path.join(d, "render_list.txt")}
    return rc

  def fit_cfg(self, k):
    d = self.ds.shard_dir(k)
    fc = OmegaConf.create(OmegaConf.to_container(self.cfg.fit, resolve=True))
    with open_dict(fc):
      fc.pop("hydra", None)
      fc.hdf5_path = os.path.join(d, "render.h5")
      fc.mesh_index = None
      fc.mesh_indices = "all"
      fc.output_dir = os.path.join(d, "fit")
      fc.packed = os.path.join(d, "grids.h5")
    return fc

  # --- one shard
  def fill_members(self, k, n):
    """Append the next CSV rows to shard k's meshes.csv until it has n members not in bad_meshes.csv. -> the members."""
    path = os.path.join(self.ds.shard_dir(k), "meshes.csv")
    self.fold_markers()
    members = self.ds.members(k)
    bad = set(self.ds.bad())
    need = n - sum(m["uid"] not in bad for m in members)
    if need > 0:
      exclude = bad | self.ds.assigned()
      new = []
      for r in self.ds.candidates(exclude):
        if len(new) >= need:
          break
        new.append({**r, "glb": glb_download.glb_path(self.ds.glbs, r["uid"])})
      if len(new) < need:
        raise SystemExit(f"the CSV has no rows left for shard {k} ({len(new)} of {need})")
      members = members + new
      write_csv_atomic(path, MEMBER_FIELDS, members)
      log.info("shard %d: %d member(s) added (rows %s..%s)", k, len(new), new[0]["csv_row"], new[-1]["csv_row"])
    return members

  def wait_for_glbs(self, k, members):
    """Until every member not in bad_meshes.csv has its GLB or was reported by the downloader (-> bad_meshes.csv)."""
    last = 0.0
    while True:
      self.fold_markers()
      bad = set(self.ds.bad())
      pending = [m for m in members if m["uid"] not in bad and not os.path.exists(m["glb"])]
      if not pending:
        return
      self.ensure_downloader()
      if time.time() - last > 60:
        log.info("shard %d: waiting for %d GLBs (e.g. row %s %s)", k, len(pending), pending[0]["csv_row"], pending[0]["uid"])
        last = time.time()
      time.sleep(5)

  def run_shard(self, k, n):
    d = self.ds.shard_dir(k)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    rc = self.render_cfg(k)
    if not self.ds.rendered(k):
      while True:
        members = self.fill_members(k, n)
        n_bad = len(self.ds.bad())
        t0 = time.time()
        self.wait_for_glbs(k, members)
        self.add_time(k, "wait_download", t0)
        if len(self.ds.bad()) != n_bad:
          continue   # members rejected at download: refill
        bad = self.ds.bad()
        render_list = [m for m in members if bad.get(m["uid"], {}).get("stage") != "download"]
        paths = [m["glb"] for m in render_list]
        list_path = rc.mesh_strategy.file
        if not os.path.exists(list_path) or open(list_path).read().split() != paths:
          tmp = f"{list_path}.tmp"
          with open(tmp, "w") as f:
            f.write("".join(p + "\n" for p in paths))
          os.replace(tmp, list_path)
        log.info("shard %d: rendering %d meshes (%d already rejected)", k, len(paths), sum(m["uid"] in bad for m in render_list))
        t0 = time.time()
        scan, failed = render_queue.render(rc, paths)
        self.add_time(k, "render", t0)
        new = [{"csv_row": render_list[mi]["csv_row"], "uid": render_list[mi]["uid"], "sha256": render_list[mi]["sha256"],
                "shard": k, "stage": "render", "reason": err,
                "log_path": os.path.join(rc.queue_dir, "logs"), "time": time.time()}
               for mi, err in sorted(failed.items()) if render_list[mi]["uid"] not in bad]
        self.add_bad(new)
        if new:
          continue   # members rejected at render: refill, render the replacements
        # a member rejected at render stays rejected even if a retry in a later run rendered it after all
        rejected = {mi for mi, m in enumerate(render_list) if m["uid"] in bad}
        for mi in rejected:
          scan.segments.pop(mi, None)
        failed = {**{mi: "in bad_meshes.csv" for mi in rejected}, **failed}
        good = len(scan.segments)
        if good != n:
          raise SystemExit(f"shard {k}: {good} meshes rendered, expected {n}")
        t0 = time.time()
        render_queue.finish(rc, paths, lambda: scan, failed)
        self.add_time(k, "merge", t0)
        break
    else:
      render_queue.finish(rc, open(rc.mesh_strategy.file).read().split(), None, ())   # leftover parts of a killed merge
    qdb = os.path.join(rc.queue_dir, "queue.db")
    for p in (qdb, qdb + "-wal", qdb + "-shm"):
      if os.path.exists(p):
        os.remove(p)   # render bookkeeping only lives while the shard renders (the logs stay)

    # fit + pack (fit_anchored shard mode); a fit error is a bug: it propagates and stops the builder
    import fit_anchored
    handler = logging.FileHandler(os.path.join(d, "logs", "fit.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(handler)
    t0 = time.time()
    try:
      fit_anchored.fit_shard(self.fit_cfg(k))
    except BaseException as e:
      if not isinstance(e, (KeyboardInterrupt, SystemExit)) or isinstance(e, SystemExit) and e.code not in (0, 143):
        log.exception("FIT ERROR in shard %d (%s; the mesh is the last '=== mesh' line in %s): %r. A rendered mesh must fit, "
                      "so the builder STOPS here (no mesh is skipped): fix the cause, then rerun -- DONE meshes are kept, the "
                      "failed one resumes per view", k, d, os.path.join(d, "logs", "fit.log"), e)
        raise SystemExit(3) from e
      raise
    finally:
      logging.getLogger().removeHandler(handler)
      handler.close()
    self.add_time(k, "fit", t0)

    import h5py
    with h5py.File(os.path.join(d, "grids.h5"), "r") as f:
      n_grids = int(f.attrs["n_grids"])
    manifest = self.ds.manifest()
    if k not in manifest:
      manifest[k] = {"shard": k, "dir": d, "render_h5": os.path.join(d, "render.h5"), "grids_h5": os.path.join(d, "grids.h5"),
                     "meshes": n, "grids": n_grids, "time": time.time()}
      write_csv_atomic(self.ds.manifest_path, MANIFEST_FIELDS, [manifest[j] for j in sorted(manifest)])
    log.info("shard %d finished: %d meshes, %d grids", k, n, n_grids)

  def run(self):
    while True:
      done = self.ds.done_new()
      if done >= self.ds.new_target:
        log.info("target reached: %d existing + %d new meshes >= %d", len(self.ds.skip), done, int(self.cfg.target_meshes))
        return
      manifest = self.ds.manifest()
      open_shards = [k for k in self.ds.shards() if k not in manifest]
      k = open_shards[0] if open_shards else max(self.ds.shards() + [int(self.cfg.first_shard) - 1]) + 1
      n = min(int(self.cfg.shard_size), self.ds.new_target - done)
      log.info("=== shard %d (%d meshes; %d of %d new meshes done)", k, n, done, self.ds.new_target)
      self.ensure_downloader()
      self.run_shard(k, n)


# ------------------------------------------------------------------------------------------------------------ status

def status(cfg, ds):
  manifest, bad, markers = ds.manifest(), ds.bad(), ds.markers()
  lines = [f"dataset {ds.root}: target {int(cfg.target_meshes)} = {len(ds.skip)} existing (skip lists) + {ds.new_target} new; "
           f"{ds.done_new()} new meshes in {len(manifest)} finished shards"]
  fit_s, render_s, n_fit, n_render = 0.0, 0.0, 0, 0
  for k in ds.shards():
    d = ds.shard_dir(k)
    members = ds.members(k)
    mb = [m for m in members if m["uid"] in bad]
    stages = {}
    for m in mb:
      stages[bad[m["uid"]]["stage"]] = stages.get(bad[m["uid"]]["stage"], 0) + 1
    done = glob.glob(os.path.join(d, "fit", "_anchored", "mesh*", "DONE"))
    for p in done:
      t = json.load(open(os.path.join(os.path.dirname(p), "timing.json")))
      fit_s += sum(t.values())
      n_fit += 1
    st = json.load(open(os.path.join(d, "timing.json"))) if os.path.exists(os.path.join(d, "timing.json")) else {}
    if ds.rendered(k):
      render_s += st.get("render", 0.0) + st.get("merge", 0.0)
      n_render += len(members) - len(mb)
    parts = len(glob.glob(os.path.join(d, "render_work", "parts", "mesh*.h5")))
    state = ("finished" if k in manifest else "fit+pack" if ds.rendered(k) else f"rendering ({parts} parts)"
             if os.path.isdir(os.path.join(d, "render_work")) else "downloading")
    lines.append(f"  shard{k}: {state}; {len(members)} members, {len(mb)} bad {stages or ''}; "
                 f"rendered {'yes' if ds.rendered(k) else 'no'}; fitted {len(done)} DONE; "
                 f"packed {'yes' if os.path.exists(os.path.join(d, 'grids.h5')) else 'no'}")
  by_stage = {}
  for r in bad.values():
    by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
  lines.append(f"bad meshes: {len(bad)} {by_stage or ''}; downloader reports not yet folded: {len(set(markers) - set(bad))}")
  window = ds.download_window()
  have = sum(os.path.exists(glb_download.glb_path(ds.glbs, r["uid"])) for r in window)
  lines.append(f"GLBs: {len(glob.glob(os.path.join(ds.glbs, '*.glb')))} on disk; download window {have}/{len(window)} ready")
  if n_fit and n_render:
    per = render_s / n_render + fit_s / n_fit
    remaining = ds.new_target - ds.done_new()
    lines.append(f"timing: render {render_s / n_render:.0f} s/mesh ({n_render} meshes), fit {fit_s / n_fit:.0f} s/mesh "
                 f"({n_fit} meshes) -> ETA {remaining * per / 3600:.1f} h for the {remaining} remaining new meshes")
  print("\n".join(lines))


# ------------------------------------------------------------------------------------------------------------ main

def _setup_logging(path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  fmt = logging.Formatter("%(asctime)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
  root = logging.getLogger()
  root.setLevel(logging.INFO)
  for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
    h.setFormatter(fmt)
    root.addHandler(h)


@hydra.main(version_base=None, config_path="conf", config_name="build_dataset")
def main(cfg: DictConfig) -> None:
  if cfg.mode not in ("build", "download", "status"):
    raise SystemExit(f"mode must be build, download or status, got {cfg.mode!r}")
  ds = Dataset(cfg)
  if cfg.mode == "status":
    status(cfg, ds)
    return
  os.makedirs(ds.glbs, exist_ok=True)
  tmp = os.path.join(ds.root, "tmp")
  os.environ.update(TMPDIR=tmp, TEMP=tmp, TMP=tmp)   # never the container's /tmp (the SSD)
  _setup_logging(os.path.join(ds.root, "logs", f"{cfg.mode}.log"))
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # run the finally blocks (stop the Blender workers)
  if cfg.mode == "download":
    os.makedirs(tmp, exist_ok=True)
    run_downloader(cfg, ds)
    return
  lock = Lock(os.path.join(ds.root, ".builder.lock"))
  if not lock.acquire(wait=float(cfg.lock_wait)):
    raise SystemExit(f"another builder holds {lock.path}")
  for name in os.listdir(tmp) if os.path.isdir(tmp) else []:
    if name.startswith("tmp"):   # tempfile's leftovers of killed runs (tmp/blender is render_queue's, torch's caches stay)
      p = os.path.join(tmp, name)
      shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
  os.makedirs(tmp, exist_ok=True)
  log.info("builder start: %s; target %d (%d existing + %d new), shard_size %d, SLURM_RESTART_COUNT %s", ds.root,
           int(cfg.target_meshes), len(ds.skip), ds.new_target, int(cfg.shard_size), os.environ.get("SLURM_RESTART_COUNT"))
  b = Builder(cfg, ds)
  try:
    b.run()
  except Exception:
    log.exception("the builder STOPS on this error (fix it, then rerun: finished work is kept)")
    raise SystemExit(1)
  finally:
    b.stop_downloader()


if __name__ == "__main__":
  main()
