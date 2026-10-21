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
writes its own part file under queue_dir/parts, exiting after queue_max_meshes meshes), and replaces
any that exit. When a worker dies, the jobs it held go back to the queue (a job that keeps killing its
workers is failed after queue_max_attempts); a worker holding one job longer than queue_claim_timeout
seconds is killed. When the queue is drained the part files are checked (render_merge.scan_parts):
meshes missing from every good part -- e.g. those in a part whose write marker was left behind -- are
enqueued again, up to queue_max_rounds rounds, and then everything is merged into output_path. Meshes
a worker or the queue failed for good are recorded in the output's failed_mesh_indices.

Once the merged file is written the part files (and any write markers) are deleted; queue_dir keeps
queue.db and the worker logs. Resumable until then: run it again with the same overrides and queue_dir
and it continues from the queue. It stops its workers on exit, but not if it is killed with SIGKILL --
stop those by hand before resuming. It refuses to start if output_path exists.
"""

import glob
import logging
import os
import signal
import subprocess
import sys
import time

import hydra
from omegaconf import OmegaConf

from job_queue import JobQueue
from render_merge import merge_parts, missing_meshes, scan_parts

logger = logging.getLogger("render_queue")


class Pool:
  def __init__(self, cfg, queue, parts_dir, logs_dir):
    self.cfg, self.queue, self.parts_dir, self.logs_dir = cfg, queue, parts_dir, logs_dir
    self.running = {}   # worker id -> Popen
    self.crashes = 0
    # worker numbers continue after the parts of earlier runs in this work_dir: a worker's part file is
    # opened with "w", and its id tells its claims apart from those of a dead worker of the same number
    old = [int(os.path.basename(p)[len("part_w"):-len(".h5")]) for p in glob.glob(os.path.join(parts_dir, "part_w*.h5"))]
    self.started = max(old, default=-1) + 1

  def spawn(self):
    cfg = self.cfg
    wid = f"w{self.started:05d}"
    self.started += 1
    part = os.path.join(self.parts_dir, f"part_{wid}.h5")
    if os.path.exists(part):
      raise FileExistsError(f"{part} exists already; a worker would overwrite it")
    cmd = [
      cfg.queue_blender, "--background", "--python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_objaverse.py"),
      "--", f"queue_path={self.queue.path}", f"queue_worker={wid}", f"output_path={part}",
      f"hydra.run.dir={os.path.join(self.logs_dir, 'hydra_' + wid)}",
    ]
    log = open(os.path.join(self.logs_dir, f"{wid}.log"), "wb")
    self.running[wid] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **cfg.queue_env})
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
                 "queue_claim_timeout", "queue_blender", "queue_env")


@hydra.main(version_base=None, config_path="conf", config_name="objaverse")
def main(cfg):
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
  if cfg.queue_dir is None:
    raise ValueError("queue_dir must be set")
  if os.path.exists(cfg.output_path):
    raise FileExistsError(f"{cfg.output_path} exists already (the part files of a finished run are deleted, so it can't be resumed)")
  meshes = list(hydra.utils.instantiate(cfg.mesh_strategy).meshes())
  work = os.path.abspath(cfg.queue_dir)
  parts_dir, logs_dir = os.path.join(work, "parts"), os.path.join(work, "logs")
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
  else:
    if queue.get_meta("meshes") != meshes:
      raise ValueError(f"{work}/queue.db was made for a different mesh list")
    if queue.get_meta("config") != render_cfg:
      raise ValueError(f"{work}/queue.db was made with a different render config; use another queue_dir")
    logger.info("resuming: %s", queue.counts())

  pool = Pool(cfg, queue, parts_dir, logs_dir)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # run the finally below
  try:
    for round_ in range(1, cfg.queue_max_rounds + 1):
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
      if round_ < cfg.queue_max_rounds:
        enqueue(queue, missing, meshes)
        logger.info("enqueued the %d missing meshes again", len(missing))
    else:
      raise SystemExit(f"{len(missing)} meshes are still missing after {cfg.queue_max_rounds} rounds: {missing[:20]}")
  finally:
    pool.stop()

  report = merge_parts(scan, meshes, os.path.abspath(cfg.output_path), failed=queue_failed)
  leftovers = glob.glob(os.path.join(parts_dir, "part_*"))   # part files and their write markers
  for path in leftovers:
    os.remove(path)
  os.rmdir(parts_dir)
  logger.info("done: %s; removed %d part files", report, len(leftovers))


if __name__ == "__main__":
  main()
