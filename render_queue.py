#!/usr/bin/env python3
"""Render meshes with a pool of Blender workers pulling from a job queue, then merge.

Configured like render_objaverse.py (conf/objaverse.yaml, same overrides); the queue_* keys there
configure the pool:

    python render_queue.py mesh_strategy=list mesh_strategy.file=/app/bla/meshes.txt \
        view_strategy=random-views view_strategy.num_views=10 width=504 height=504 \
        render_engine=BLENDER_EEVEE_NEXT device=cpu \
        output_path=/app/bla/out.h5 queue_dir=/app/bla/out_work queue_workers=3

mesh_strategy gives the meshes; one job is one mesh with all its views. The launcher enqueues them
(largest file first) into queue_dir/queue.db together with the resolved config, keeps queue_workers
render_objaverse.py processes running in queue mode (each takes its whole config from the queue and
writes one file per mesh under queue_dir/parts -- temp name, fsync, rename -- exiting after
queue_max_meshes meshes), and replaces any that exit. When a worker dies, the jobs it held go back to
the queue (a job that keeps killing its workers is failed after queue_max_attempts); a worker holding
one job longer than queue_claim_timeout seconds is killed. A mesh whose render raised is failed; with
queue_retry_failed > 0 it is retried that many times (in later rounds). When the queue is drained the
part files are checked (render_merge.scan_parts): meshes missing from every good part are enqueued
again, up to queue_max_rounds rounds, and then everything is merged into output_path. Meshes a worker
or the queue failed for good are recorded in the output's failed_mesh_indices.

Kill-safe: rerun it with the same overrides and queue_dir after a kill at any point and it continues.
- Claims found on a restart (their workers died with the previous launcher: a kill / preemption of the
  run) go back to the queue without counting as an attempt; only workers dying while the launcher
  watches count.
- The workers get SIGKILL when the launcher dies (PR_SET_PDEATHSIG), so killing only the launcher
  leaves no orphans; it refuses to start while any worker of its queue is still alive.
- A kill while a worker writes loses only that mesh; a failed write (e.g. a full disk) stops the worker
  (its claim goes back to the queue) instead of failing the mesh.
- The merge (render_merge.finish_merge) writes output_path.tmp, verifies it against the parts, fsyncs
  and renames it, and only then deletes the parts. An existing output_path with leftover parts (a kill
  right after the merge) is finished by deleting them; an existing output_path without parts is
  refused, as before.
queue_dir keeps queue.db and the worker logs. queue_tmpdir (if set) is the Blender workers' TMPDIR
(emptied on start): their temp dirs otherwise pile up in /tmp when they are killed.

The pool is importable (open_queue / render / finish), e.g. for build_dataset.py, which grows a queue's
mesh list (a list extending the stored one enqueues only the new meshes) before merging.
"""

import ctypes
import glob
import logging
import os
import shutil
import signal
import subprocess
import sys
import time

import hydra
from omegaconf import OmegaConf

from job_queue import JobQueue
from render_merge import finish_merge, missing_meshes, scan_parts

logger = logging.getLogger("render_queue")

PR_SET_PDEATHSIG = 1


def _die_with_parent():
  """preexec_fn of a worker: the kernel SIGKILLs it when the launcher dies (no orphans after a launcher-only kill)."""
  ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


class Pool:
  def __init__(self, cfg, queue, parts_dir, logs_dir):
    self.cfg, self.queue, self.parts_dir, self.logs_dir = cfg, queue, parts_dir, logs_dir
    self.running = {}   # worker id -> Popen
    self.crashes = 0
    # worker numbers continue after those of earlier runs in this work_dir (their logs): a worker's id tells its claims
    # apart from those of a dead worker of the same number
    old = [int(os.path.basename(p)[1:-len(".log")]) for p in glob.glob(os.path.join(logs_dir, "w*.log"))]
    self.started = max(old, default=-1) + 1

  def spawn(self):
    cfg = self.cfg
    wid = f"w{self.started:05d}"
    self.started += 1
    cmd = [
      cfg.queue_blender, "--background", "--python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_objaverse.py"),
      "--", f"queue_path={self.queue.path}", f"queue_worker={wid}", f"output_path={self.parts_dir}",
      f"hydra.run.dir={os.path.join(self.logs_dir, 'hydra_' + wid)}",
    ]
    env = {**os.environ, **cfg.queue_env}
    if cfg.queue_tmpdir is not None:
      env.update(TMPDIR=str(cfg.queue_tmpdir), TEMP=str(cfg.queue_tmpdir), TMP=str(cfg.queue_tmpdir))
    log = open(os.path.join(self.logs_dir, f"{wid}.log"), "wb")
    self.running[wid] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=_die_with_parent)
    log.close()
    return wid

  def reap(self):
    """Collect exited workers, return their jobs to the queue; kill workers holding a job too long."""
    cfg = self.cfg
    for wid, proc in list(self.running.items()):
      rc = proc.poll()
      if rc is None:
        continue
      del self.running[wid]
      if rc != 0:
        self.crashes += 1
        logger.warning("worker %s exited with %d (see %s.log)", wid, rc, wid)
    requeued = self.queue.requeue_stale(self.running)
    exhausted = self.queue.fail_exhausted(cfg.queue_max_attempts)
    if requeued:
      logger.info("%d job(s) of dead workers back in the queue", requeued)
    if exhausted:
      logger.error("job(s) %s failed for good after %d attempts", exhausted, cfg.queue_max_attempts)
    if cfg.queue_claim_timeout is not None:
      for _, wid, claimed_at in self.queue.claims():
        if wid in self.running and time.time() - claimed_at > cfg.queue_claim_timeout:
          logger.warning("worker %s has held a job for over %ds -- killing it", wid, cfg.queue_claim_timeout)
          self.running[wid].kill()

  def drain(self):
    """Keep num_workers running until the queue has no ready job and no worker is left."""
    last = 0.0
    while True:
      self.reap()
      counts = self.queue.counts()
      want = min(self.cfg.queue_workers - len(self.running), counts["ready"])
      for _ in range(max(want, 0)):
        self.spawn()
      if not self.running and counts["ready"] == 0:
        return
      if time.time() - last > 30:
        logger.info("queue %s | %d workers running, %d started, %d crashed", counts, len(self.running), self.started, self.crashes)
        last = time.time()
      time.sleep(2)

  def stop(self):
    for proc in self.running.values():
      proc.terminate()
    for proc in self.running.values():
      try:
        proc.wait(timeout=20)
      except subprocess.TimeoutExpired:
        proc.kill()
    self.running.clear()


def enqueue(queue, indices, meshes):
  queue.add([(os.path.getsize(meshes[i]), {"mesh_index": i}) for i in indices])


LAUNCHER_KEYS = ("output_path", "queue_dir", "queue_workers", "queue_max_attempts", "queue_max_rounds",
                 "queue_claim_timeout", "queue_blender", "queue_env", "queue_retry_failed", "queue_tmpdir")


def work_dirs(queue_dir):
  work = os.path.abspath(queue_dir)
  return work, os.path.join(work, "parts"), os.path.join(work, "logs")


def live_workers(queue_path):
  """pids of render_objaverse.py processes working on this queue file (e.g. orphans of a launcher killed alone)."""
  r = subprocess.run(["pgrep", "-f", f"queue_path={queue_path}( |$)"], capture_output=True, text=True)
  return [int(p) for p in r.stdout.split()]


def open_queue(cfg, meshes):
  """queue_dir/queue.db for this config and mesh list: created (all meshes enqueued), resumed (same list), or grown (the
  list extends the stored one: only the new meshes are enqueued). A different config / list is an error."""
  work, parts_dir, logs_dir = work_dirs(cfg.queue_dir)
  os.makedirs(parts_dir, exist_ok=True)
  os.makedirs(logs_dir, exist_ok=True)
  # what the workers get: the config without the launcher's own keys
  render_cfg = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k not in LAUNCHER_KEYS}
  queue = JobQueue(os.path.join(work, "queue.db"))
  if sum(queue.counts().values()) == 0:
    queue.set_meta("meshes", meshes)
    queue.set_meta("config", render_cfg)
    enqueue(queue, range(len(meshes)), meshes)
    logger.info("enqueued %d meshes", len(meshes))
    return queue
  if queue.get_meta("config") != render_cfg:
    raise ValueError(f"{work}/queue.db was made with a different render config; use another queue_dir")
  stored = queue.get_meta("meshes")
  if stored != meshes[:len(stored)]:
    raise ValueError(f"{work}/queue.db was made for a different mesh list")
  if len(meshes) > len(stored):
    queue.set_meta("meshes", meshes)
    enqueue(queue, range(len(stored), len(meshes)), meshes)
    logger.info("mesh list grown: enqueued meshes %d..%d", len(stored), len(meshes) - 1)
  logger.info("resuming: %s", queue.counts())
  return queue


def render(cfg, meshes):
  """Render every mesh of the list into queue_dir/parts (resuming whatever is there). -> (scan, failed): the PartScan of the
  parts and {mesh index: error} of the meshes that failed for good. Raises SystemExit when meshes are still missing after
  queue_max_rounds rounds."""
  work, parts_dir, logs_dir = work_dirs(cfg.queue_dir)
  queue_path = os.path.join(work, "queue.db")
  alive = live_workers(queue_path)
  if alive:
    raise SystemExit(f"render workers of {queue_path} are still running (pids {alive}): stop them first "
                     "(a launcher killed alone? kill the whole process group)")
  queue = open_queue(cfg, meshes)
  # nothing of this queue runs now: every claim is a leftover of a killed run -- back to the queue, not counted
  refunded = queue.requeue_stale((), refund=True)
  if refunded:
    logger.info("%d claim(s) of a killed run back in the queue (not counted as attempts)", refunded)
  for path in glob.glob(os.path.join(parts_dir, "*.tmp")) + glob.glob(os.path.join(parts_dir, "*.txn")):
    os.remove(path)   # a worker killed while writing (its mesh is rendered again)
  if cfg.queue_tmpdir is not None:
    shutil.rmtree(str(cfg.queue_tmpdir), ignore_errors=True)
    os.makedirs(str(cfg.queue_tmpdir), exist_ok=True)
  retry_cap = min(1 + int(cfg.queue_retry_failed), int(cfg.queue_max_attempts))

  pool = Pool(cfg, queue, parts_dir, logs_dir)
  try:
    for round_ in range(1, cfg.queue_max_rounds + 1):
      pool.drain()
      retried = queue.retry_failed(retry_cap) if (cfg.queue_retry_failed and round_ < cfg.queue_max_rounds) else []
      failed = {int(p["mesh_index"]): e for _, p, e in queue.failed()}
      scan = scan_parts(sorted(glob.glob(os.path.join(parts_dir, "*.h5"))))
      for path, why in scan.cooked:
        logger.error("not using %s: %s", path, why)
      missing = [mi for mi in missing_meshes(scan, len(meshes), failed) if mi not in {int(p["mesh_index"]) for _, p in retried}]
      logger.info("round %d: %d meshes in good parts, %d failed, %d to retry, %d missing, %d duplicates dropped",
                  round_, len(scan.segments), len(scan.failed | set(failed)), len(retried), len(missing), scan.duplicates)
      if not missing and not retried:
        break
      if round_ < cfg.queue_max_rounds and missing:
        enqueue(queue, missing, meshes)
        logger.info("enqueued the %d missing meshes again", len(missing))
    else:
      raise SystemExit(f"{len(missing)} meshes are still missing after {cfg.queue_max_rounds} rounds: {missing[:20]}")
  finally:
    pool.stop()
    queue.close()
  for mi in scan.failed:
    failed.setdefault(mi, "recorded as failed in its part")
  return scan, failed


def finish(cfg, meshes, scan_fn, failed):
  """Kill-safe merge of queue_dir/parts into output_path, then the parts are deleted (render_merge.finish_merge)."""
  _, parts_dir, _ = work_dirs(cfg.queue_dir)
  return finish_merge(scan_fn, meshes, os.path.abspath(cfg.output_path), parts_dir, failed=failed)


def queue_failed(cfg):
  """{mesh index: error} of the failed jobs in queue_dir/queue.db."""
  queue = JobQueue(os.path.join(work_dirs(cfg.queue_dir)[0], "queue.db"))
  try:
    return {int(p["mesh_index"]): e for _, p, e in queue.failed()}
  finally:
    queue.close()


@hydra.main(version_base=None, config_path="conf", config_name="objaverse")
def main(cfg):
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
  if cfg.queue_dir is None:
    raise ValueError("queue_dir must be set")
  meshes = list(hydra.utils.instantiate(cfg.mesh_strategy).meshes())
  _, parts_dir, _ = work_dirs(cfg.queue_dir)
  if os.path.exists(cfg.output_path):
    if not os.path.isdir(parts_dir):
      raise FileExistsError(f"{cfg.output_path} exists already (the part files of a finished run are deleted, so it can't be resumed)")
    open_queue(cfg, meshes).close()   # the leftover parts are this run's (same config and mesh list)
    finish(cfg, meshes, None, ())
    return
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # run render's finally (stops the workers)
  scan, failed = render(cfg, meshes)
  report = finish(cfg, meshes, lambda: scan, failed)
  logger.info("done: %s", report)


if __name__ == "__main__":
  main()
