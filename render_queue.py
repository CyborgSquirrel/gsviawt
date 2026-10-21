#!/usr/bin/env python3
"""Render a list of meshes with a pool of Blender workers pulling from a job queue, then merge.

    python render_queue.py meshes=/app/bla/meshes.txt output_path=/app/bla/out.h5 \\
        work_dir=/app/bla/out_work num_workers=3 \\
        'render_args=[view_strategy=random-views,view_strategy.num_views=10,width=504,height=504,
                      render_engine=BLENDER_EEVEE_NEXT,device=cpu]'

`meshes` is a text file with one mesh path per line; one job is one mesh with all its views. The
launcher enqueues the meshes (largest file first) into work_dir/queue.db, keeps `num_workers`
render_objaverse.py processes running (each in queue mode, writing its own part file under
work_dir/parts and exiting after `max_meshes_per_worker` meshes), and replaces any that exit. When
a worker dies, the jobs it held go back to the queue (a job that keeps killing its workers is
failed after `max_attempts`); a worker holding one job longer than `claim_timeout` seconds is
killed. When the queue is drained the part files are checked (render_merge.scan_parts): meshes
missing from every good part -- e.g. those in a part whose write marker was left behind -- are
enqueued again, up to `max_rounds` rounds, and then everything is merged into output_path. Meshes a
worker or the queue failed for good are recorded in the output's failed_mesh_indices.

Resumable: run it again with the same work_dir and it continues from the queue. It stops its
workers on exit, but not if it is killed with SIGKILL -- stop those by hand before resuming.
"""

import glob
import logging
import os
import signal
import subprocess
import sys
import time

import hydra

from job_queue import JobQueue
from render_merge import merge_parts, missing_meshes, scan_parts

logger = logging.getLogger("render_queue")


class Pool:
  def __init__(self, cfg, queue, parts_dir, logs_dir):
    self.cfg, self.queue, self.parts_dir, self.logs_dir = cfg, queue, parts_dir, logs_dir
    self.running = {}   # worker id -> Popen
    self.started = 0
    self.crashes = 0

  def spawn(self):
    cfg = self.cfg
    wid = f"w{self.started:05d}"
    self.started += 1
    cmd = [
      cfg.blender, "--background", "--python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_objaverse.py"),
      "--", *cfg.render_args,
      f"queue_path={self.queue.path}", f"queue_worker={wid}", f"queue_max_meshes={cfg.max_meshes_per_worker}",
      f"output_path={os.path.join(self.parts_dir, f'part_{wid}.h5')}",
      f"hydra.run.dir={os.path.join(self.logs_dir, 'hydra_' + wid)}",
    ]
    log = open(os.path.join(self.logs_dir, f"{wid}.log"), "wb")
    self.running[wid] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **cfg.env})
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
    exhausted = self.queue.fail_exhausted(cfg.max_attempts)
    if requeued:
      logger.info("%d job(s) of dead workers back in the queue", requeued)
    if exhausted:
      logger.error("job(s) %s failed for good after %d attempts", exhausted, cfg.max_attempts)
    if cfg.claim_timeout is not None:
      for _, wid, claimed_at in self.queue.claims():
        if wid in self.running and time.time() - claimed_at > cfg.claim_timeout:
          logger.warning("worker %s has held a job for over %ds -- killing it", wid, cfg.claim_timeout)
          self.running[wid].kill()

  def drain(self):
    """Keep num_workers running until the queue has no ready job and no worker is left."""
    last = 0.0
    while True:
      self.reap()
      counts = self.queue.counts()
      want = min(self.cfg.num_workers - len(self.running), counts["ready"])
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


@hydra.main(version_base=None, config_path="conf", config_name="render_queue")
def main(cfg):
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
  meshes = [l.strip() for l in open(cfg.meshes) if l.strip()]
  work = os.path.abspath(cfg.work_dir)
  parts_dir, logs_dir = os.path.join(work, "parts"), os.path.join(work, "logs")
  os.makedirs(parts_dir, exist_ok=True)
  os.makedirs(logs_dir, exist_ok=True)

  queue = JobQueue(os.path.join(work, "queue.db"))
  if sum(queue.counts().values()) == 0:
    queue.set_meta("meshes", meshes)
    enqueue(queue, range(len(meshes)), meshes)
    logger.info("enqueued %d meshes", len(meshes))
  else:
    if queue.get_meta("meshes") != meshes:
      raise ValueError(f"{work}/queue.db was made for a different mesh list than {cfg.meshes}")
    logger.info("resuming: %s", queue.counts())

  pool = Pool(cfg, queue, parts_dir, logs_dir)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # run the finally below
  try:
    for round_ in range(1, cfg.max_rounds + 1):
      pool.drain()
      queue_failed = {p["mesh_index"] for _, p, _ in queue.failed()}
      scan = scan_parts(sorted(glob.glob(os.path.join(parts_dir, "part_*.h5"))))
      for path, why in scan.cooked:
        logger.error("not using %s: %s", path, why)
      missing = missing_meshes(scan, len(meshes), queue_failed)
      logger.info("round %d: %d meshes in good parts, %d failed, %d missing, %d duplicates dropped",
                  round_, len(scan.segments), len(scan.failed | queue_failed), len(missing), scan.duplicates)
      if not missing:
        break
      if round_ < cfg.max_rounds:
        enqueue(queue, missing, meshes)
        logger.info("enqueued the %d missing meshes again", len(missing))
    else:
      raise SystemExit(f"{len(missing)} meshes are still missing after {cfg.max_rounds} rounds: {missing[:20]}")
  finally:
    pool.stop()

  report = merge_parts(scan, meshes, os.path.abspath(cfg.output_path), failed=queue_failed)
  logger.info("done: %s", report)


if __name__ == "__main__":
  main()
