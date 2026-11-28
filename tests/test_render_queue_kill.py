"""render_queue / render_merge / job_queue kill-resume behaviour (bla/review_kill_resume.md), without Blender: the pool runs a fake
worker (queue_blender = a small script speaking render_objaverse.py's queue-mode protocol: claim, one part file per mesh written
temp -> rename, done). Kills are simulated by raising out of the merge at chosen points (a SIGKILL leaves the same files) and
by real SIGKILLs of the launcher."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

import render_merge
import render_queue
from job_queue import JobQueue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_WORKER = textwrap.dedent('''\
  #!{python}
  """Fake Blender worker: argv = [--background --python <script> --] queue_path=.. queue_worker=.. output_path=.. hydra..."""
  import json, os, sys, time
  sys.path.insert(0, {root!r})
  import h5py, numpy as np
  from job_queue import JobQueue
  from render_merge import part_path
  args = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--") + 1:])
  q = JobQueue(args["queue_path"])
  worker, parts = args["queue_worker"], args["output_path"]
  mode = os.environ.get("FAKE_MODE", "")
  meshes = q.get_meta("meshes")
  while True:
    job = q.claim(worker)
    if job is None:
      break
    mi = int(job["payload"]["mesh_index"])
    if mode == "sleep":
      open(os.environ["FAKE_PIDS"], "a").write(f"{{os.getpid()}}\\n")
      time.sleep(600)
    if mode == "fail_write_once" and not os.path.exists(os.environ["FAKE_FLAG"]):
      open(os.environ["FAKE_FLAG"], "w").close()
      sys.exit(1)   # a failed commit: the worker exits, its claim goes back to the queue
    if os.path.basename(meshes[mi]).startswith("bad"):
      q.fail(job, worker, "RuntimeError('bad mesh')")
      continue
    tmp = part_path(parts, mi) + f".{{worker}}.tmp"
    with h5py.File(tmp, "w") as f:
      f.attrs["config_json"] = json.dumps({{"output_path": parts}})
      f.create_dataset("mesh_paths", data=meshes, dtype=h5py.string_dtype("utf-8"))
      n = 3
      img = np.full((n, 4, 4, 4), mi, np.uint8)
      f.create_dataset("images", data=img, chunks=(1, 4, 4, 4), compression="gzip")
      f.create_dataset("camera_pose", data=np.full((n, 4, 4), mi, np.float32), chunks=(1, 4, 4))
      f.create_dataset("mesh_index", data=np.full(n, mi, np.int64))
    os.replace(tmp, part_path(parts, mi))
    q.done(job, worker)
''')


@pytest.fixture
def setup(tmp_path):
  fake = tmp_path / "fake_blender"
  fake.write_text(FAKE_WORKER.format(python=sys.executable, root=ROOT))
  fake.chmod(0o755)
  meshes = []
  for i, name in enumerate(["a", "b", "bad_c", "d"]):
    p = tmp_path / "glbs" / f"{name}.glb"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"x" * (10 + i))
    meshes.append(str(p))
  cfg = OmegaConf.create({
    "output_path": str(tmp_path / "out.h5"), "queue_dir": str(tmp_path / "work"), "queue_workers": 2,
    "queue_max_attempts": 3, "queue_max_rounds": 3, "queue_retry_failed": 0, "queue_claim_timeout": None,
    "queue_blender": str(fake), "queue_env": {}, "queue_tmpdir": str(tmp_path / "blender_tmp"),
    "width": 4, "mesh_strategy": {"_target_": "mesh_strategy.ListMeshes", "paths": meshes}})
  return cfg, meshes, tmp_path


def parts_dir(cfg):
  return render_queue.work_dirs(cfg.queue_dir)[1]


def check_output(cfg, meshes, failed):
  with h5py.File(cfg.output_path, "r") as f:
    assert f["mesh_paths"].asstr()[()].tolist() == meshes
    assert json.loads(f.attrs["failed_mesh_indices"]) == failed
    good = [i for i in range(len(meshes)) if i not in failed]
    assert f["mesh_index"][()].tolist() == [i for i in good for _ in range(3)]
    assert (f["images"][()][:, 0, 0, 0] == f["mesh_index"][()]).all()
  assert not os.path.exists(parts_dir(cfg))


def test_render_and_merge(setup):
  cfg, meshes, _ = setup
  scan, failed = render_queue.render(cfg, meshes)
  assert sorted(scan.segments) == [0, 1, 3] and set(failed) == {2}
  assert os.listdir(cfg.queue_tmpdir) == []
  render_queue.finish(cfg, meshes, lambda: scan, failed)
  check_output(cfg, meshes, [2])


def test_queue_grows_with_the_mesh_list(setup):
  cfg, meshes, _ = setup
  render_queue.render(cfg, meshes[:2])
  render_queue.render(cfg, meshes)   # a list extending the stored one: only the new meshes are enqueued
  q = JobQueue(os.path.join(cfg.queue_dir, "queue.db"))
  assert q.counts() == {"ready": 0, "claimed": 0, "done": 3, "failed": 1}
  with pytest.raises(ValueError, match="different mesh list"):
    render_queue.open_queue(cfg, meshes[1:])


def test_stale_claims_on_restart_are_not_attempts(setup):
  cfg, meshes, _ = setup
  q = render_queue.open_queue(cfg, meshes)
  for _ in range(2):   # two real worker crashes on the two first jobs (counted)
    q.claim("w00090")
    q.claim("w00091")
    q.requeue_stale(())
  q.claim("w00090")
  q.claim("w00091")   # then the run is SIGKILLed while they are claimed: the 3rd attempt is not the jobs' fault
  assert sorted(a for (a,) in q.db.execute("SELECT attempts FROM jobs")) == [0, 0, 3, 3]
  q.close()
  scan, failed = render_queue.render(cfg, meshes)   # queue_max_attempts = 3: without the refund both would fail for good
  assert sorted(scan.segments) == [0, 1, 3] and set(failed) == {2}


def test_failed_write_doesnt_fail_the_mesh(setup, monkeypatch):
  cfg, meshes, tmp_path = setup
  monkeypatch.setenv("FAKE_MODE", "fail_write_once")
  monkeypatch.setenv("FAKE_FLAG", str(tmp_path / "flag"))
  scan, failed = render_queue.render(cfg, meshes)
  assert sorted(scan.segments) == [0, 1, 3] and set(failed) == {2}


def test_retry_failed(setup):
  cfg, meshes, _ = setup
  cfg.queue_retry_failed = 1
  scan, failed = render_queue.render(cfg, meshes)
  q = JobQueue(os.path.join(cfg.queue_dir, "queue.db"))
  assert [(p["mesh_index"], e) for _, p, e in q.failed()] == [(2, "RuntimeError('bad mesh')")]
  assert q.db.execute("SELECT attempts FROM jobs WHERE status = 'failed'").fetchone()[0] == 2   # tried twice


class Killed(BaseException):
  """Stands in for a SIGKILL: escapes every `except Exception`, runs no further code of the merge."""


@pytest.mark.parametrize("where", ["before_rename", "after_rename", "deleting_parts"])
def test_merge_kill_then_rerun(setup, monkeypatch, where):
  cfg, meshes, _ = setup
  scan, failed = render_queue.render(cfg, meshes)
  real_replace, real_fsync, real_remove = os.replace, render_merge.fsync_path, os.remove
  calls = {"n": 0}

  def replace(a, b):
    if where == "before_rename" and b == cfg.output_path:
      raise Killed()
    return real_replace(a, b)

  def fsync(p, directory=False):
    real_fsync(p, directory)
    if where == "after_rename" and directory:
      raise Killed()

  def remove(p):
    calls["n"] += 1
    if where == "deleting_parts" and calls["n"] == 2:
      raise Killed()
    return real_remove(p)

  monkeypatch.setattr(render_merge.os, "replace", replace)
  monkeypatch.setattr(render_merge, "fsync_path", fsync)
  monkeypatch.setattr(render_merge.os, "remove", remove)
  with pytest.raises(Killed):
    render_queue.finish(cfg, meshes, lambda: scan, failed)
  monkeypatch.undo()
  if where == "before_rename":
    assert os.path.exists(cfg.output_path + ".tmp") and not os.path.exists(cfg.output_path)
  else:
    assert os.path.exists(cfg.output_path) and os.path.isdir(parts_dir(cfg))
  # rerun as the builder / render_queue.py's main do: an existing output only needs its leftover parts deleted
  if os.path.exists(cfg.output_path):
    render_queue.finish(cfg, meshes, None, ())
  else:
    scan2, failed2 = render_queue.render(cfg, meshes)
    render_queue.finish(cfg, meshes, lambda: scan2, failed2)
  check_output(cfg, meshes, [2])
  assert not os.path.exists(cfg.output_path + ".tmp")


def test_standalone_rerun_after_merge(setup):
  """render_queue.py's main on an existing output: finishes deleting leftover parts; without parts it still refuses."""
  from hydra import compose, initialize_config_dir
  cfg0, meshes, _ = setup
  with initialize_config_dir(version_base=None, config_dir=os.path.join(ROOT, "conf")):
    cfg = compose(config_name="objaverse", overrides=[f"{k}={v}" for k, v in (
      ("output_path", cfg0.output_path), ("queue_dir", cfg0.queue_dir), ("queue_blender", cfg0.queue_blender),
      ("queue_tmpdir", cfg0.queue_tmpdir), ("mesh_strategy", "list"), ("mesh_strategy.paths", "[" + ",".join(meshes) + "]"),
      ("queue_workers", 2))])
  scan, failed = render_queue.render(cfg, meshes)
  render_merge.merge_parts(scan, meshes, cfg.output_path, failed=failed)   # killed before the parts were deleted
  render_queue.main.__wrapped__(cfg)
  check_output(cfg, meshes, [2])
  with pytest.raises(FileExistsError, match="exists already"):
    render_queue.main.__wrapped__(cfg)


def test_verify_merged_catches_a_difference(setup):
  cfg, meshes, _ = setup
  scan, failed = render_queue.render(cfg, meshes)
  render_merge.merge_parts(scan, meshes, cfg.output_path, failed=failed)
  assert render_merge.verify_merged(cfg.output_path, scan, meshes, failed) == []
  with h5py.File(cfg.output_path, "r+") as f:
    f["camera_pose"][4] = 99.0
  problems = render_merge.verify_merged(cfg.output_path, scan, meshes, failed)
  assert problems == ["mesh 1: camera_pose differs from " + scan.segments[1][0]]


def test_launcher_kill_leaves_no_orphans(setup, tmp_path):
  """SIGKILL of the launcher alone: its workers die with it (PR_SET_PDEATHSIG); a restart refuses while workers live."""
  cfg, meshes, _ = setup
  pids = tmp_path / "pids"
  script = tmp_path / "launch.py"
  script.write_text(textwrap.dedent(f'''\
    import sys; sys.path.insert(0, {ROOT!r})
    from omegaconf import OmegaConf
    import render_queue
    render_queue.render(OmegaConf.create({OmegaConf.to_container(cfg)!r}), {meshes!r})
  '''))
  env = {**os.environ, "FAKE_MODE": "sleep", "FAKE_PIDS": str(pids)}
  launcher = subprocess.Popen([sys.executable, str(script)], env=env)
  for _ in range(200):
    if pids.exists() and len(pids.read_text().split()) == 2:
      break
    time.sleep(0.05)
  workers = [int(p) for p in pids.read_text().split()]
  assert len(workers) == 2
  os.kill(launcher.pid, signal.SIGKILL)
  launcher.wait()
  for _ in range(100):
    if not any(_alive(p) for p in workers):
      break
    time.sleep(0.05)
  assert not any(_alive(p) for p in workers)


def _alive(pid):
  try:
    with open(f"/proc/{pid}/stat") as f:
      return f.read().split()[2] != "Z"
  except FileNotFoundError:
    return False


def test_refuses_while_workers_live(setup, monkeypatch):
  cfg, meshes, _ = setup
  monkeypatch.setattr(render_queue, "live_workers", lambda path: [12345])
  with pytest.raises(SystemExit, match="still running"):
    render_queue.render(cfg, meshes)


def test_job_queue_refund_and_retry(tmp_path):
  q = JobQueue(str(tmp_path / "q.db"))
  q.add([(1, {"i": 0}), (0, {"i": 1})])
  a = q.claim("w1")
  b = q.claim("w2")
  assert q.requeue_stale(["w2"], refund=True) == 1
  att = dict(q.db.execute("SELECT id, attempts FROM jobs").fetchall())
  assert att == {a["id"]: 0, b["id"]: 1}
  q.fail(b, "w2", "boom")
  assert q.retry_failed(1) == []          # claimed once already
  assert [p for _, p in q.retry_failed(2)] == [{"i": 1}]
  assert q.counts()["ready"] == 2
  for _ in range(3):
    q.claim("w3")
    q.requeue_stale(())
  assert len(q.fail_exhausted(3)) >= 1    # real crashes (not refunded) still retire a job
