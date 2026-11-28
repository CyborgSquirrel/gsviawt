"""build_dataset.py's bookkeeping (all state derived from files: members, bad_meshes.csv, the downloader's failure reports,
the download window, the manifest) and glb_download.fetch, without Blender / GPU: the render and fit stages are faked."""
import csv
import hashlib
import json
import os
import urllib.error

import h5py
import pytest
from omegaconf import OmegaConf

import build_dataset as bd
import glb_download


def glb_bytes(i):
  return f"glb {i}\n".encode() * 50


@pytest.fixture
def env(tmp_path):
  """A fake TRELLIS CSV of 12 rows (uids u00..u11, GLBs in a local pool), a skip list holding u01."""
  pool = tmp_path / "pool"
  pool.mkdir()
  rows = []
  for i in range(12):
    data = glb_bytes(i)
    (pool / f"u{i:02d}.glb").write_bytes(data)
    rows.append({"sha256": hashlib.sha256(data).hexdigest(), "file_identifier": f"https://sketchfab.com/3d-models/u{i:02d}",
                 "aesthetic_score": f"{5 + i / 10}", "captions": '["a thing, with ""quotes"" and commas"]' if i == 2 else ""})
  with open(tmp_path / "rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sha256", "file_identifier", "aesthetic_score", "captions"])
    w.writeheader()
    w.writerows(rows)
  (tmp_path / "skip.txt").write_text("/somewhere/u01.glb\n")
  cfg = OmegaConf.create({
    "mode": "build", "csv": str(tmp_path / "rows.csv"), "skip_meshes": [str(tmp_path / "skip.txt")], "target_meshes": 7,
    "shard_size": 3, "out_root": str(tmp_path / "root"), "first_shard": 4, "download_ahead": 1.25, "retries": 1,
    "local_glbs": [str(pool)], "objaverse_dir": str(tmp_path), "download_timeout": 5, "download_poll": 0, "lock_wait": 0,
    "render": {"hydra": {}, "queue_dir": None}, "fit": {"hydra": {}}})
  return cfg, tmp_path


def uids(rows):
  return [r["uid"] for r in rows]


def test_candidates_and_window(env):
  cfg, _ = env
  ds = bd.Dataset(cfg)
  assert ds.new_target == 6   # 7 incl. the skip list's one mesh
  assert uids(ds.candidates(set()))[:3] == ["u00", "u02", "u03"]
  assert uids(ds.download_window()) == ["u00", "u02", "u03", "u04"]   # ceil(1.25 x 3)
  os.makedirs(ds.shard_dir(4))
  bd.write_csv_atomic(os.path.join(ds.shard_dir(4), "meshes.csv"), bd.MEMBER_FIELDS,
                      [dict(r, glb="") for r in ds.rows if r["uid"] in ("u00", "u02", "u03")])
  assert uids(ds.download_window()) == ["u00", "u02", "u03", "u04"]   # not rendered yet: still in the window
  open(os.path.join(ds.shard_dir(4), "render.h5"), "w").close()
  assert uids(ds.download_window()) == ["u04", "u05", "u06", "u07"]
  os.makedirs(ds.failed_dir)
  bd.write_json_atomic(os.path.join(ds.failed_dir, "u05.json"), {"csv_row": "5", "uid": "u05", "sha256": "", "reason": "x",
                                                                 "time": 0})
  assert uids(ds.download_window()) == ["u04", "u06", "u07", "u08"]
  # near the target: 1 more mesh needed -> it + the 0.25 x 3 buffer (rounded up) = 2 rows
  cfg.target_meshes = 5
  assert uids(bd.Dataset(cfg).download_window()) == ["u04", "u06"]


def test_fill_members_replaces_download_rejects(env):
  cfg, _ = env
  ds = bd.Dataset(cfg)
  b = bd.Builder(cfg, ds)
  os.makedirs(ds.shard_dir(4))
  members = b.fill_members(4, 3)
  assert uids(members) == ["u00", "u02", "u03"]
  assert b.fill_members(4, 3) == members   # idempotent
  os.makedirs(ds.failed_dir)
  bd.write_json_atomic(os.path.join(ds.failed_dir, "u02.json"), {"csv_row": "2", "uid": "u02", "sha256": "s", "reason": "HTTP 404",
                                                                 "time": 1.0})
  members = b.fill_members(4, 3)
  assert uids(members) == ["u00", "u02", "u03", "u04"]   # the rejected one stays listed, its replacement is appended
  bad = ds.bad()
  assert list(bad) == ["u02"] and bad["u02"]["stage"] == "download" and bad["u02"]["shard"] == "4"
  assert members[1]["captions"] == '["a thing, with ""quotes"" and commas"]'   # metadata round-trips through meshes.csv
  b.fold_markers()
  assert len(ds.bad()) == 1   # folded once


def fake_render(fail_first=()):
  """render_queue.render / finish stand-ins: every listed mesh renders, except the indices in fail_first on the first call."""
  calls = []

  def render(rc, paths):
    calls.append(list(paths))
    failed = {mi: "RuntimeError('bad')" for mi in fail_first} if len(calls) == 1 else {}
    scan = type("Scan", (), {})()
    scan.segments = {mi: ("p", 0, 1) for mi in range(len(paths)) if mi not in failed or len(calls) > 1}
    return scan, failed

  def finish(rc, paths, scan_fn, failed):
    with h5py.File(rc.output_path, "w") as f:
      f.attrs["failed_mesh_indices"] = json.dumps(sorted(failed))
      f.create_dataset("mesh_paths", data=paths, dtype=h5py.string_dtype())
  return render, finish, calls


def fake_fit_shard(fc):
  with h5py.File(fc.packed, "w") as f:
    f.attrs["n_grids"] = 3


def test_run_shard_render_reject_and_manifest(env, monkeypatch):
  cfg, _ = env
  ds = bd.Dataset(cfg)
  b = bd.Builder(cfg, ds)
  render, finish, calls = fake_render(fail_first=(1,))
  monkeypatch.setattr(bd.render_queue, "render", render)
  monkeypatch.setattr(bd.render_queue, "finish", finish)
  import fit_anchored
  monkeypatch.setattr(fit_anchored, "fit_shard", fake_fit_shard)
  monkeypatch.setattr(b, "ensure_downloader", lambda: None)
  monkeypatch.setattr(glb_download, "hf_paths", lambda d: {})
  bd.run_downloader(cfg, ds, passes=1)   # the GLBs of the window (from the local pool)
  b.run_shard(4, 3)
  # u02 failed to render: in bad_meshes.csv, replaced by u04; rendered again with the grown list (a rejected one stays in it,
  # and the merge records it as failed even though the second fake render call "rendered" it)
  assert [os.path.basename(p) for p in calls[1]] == ["u00.glb", "u02.glb", "u03.glb", "u04.glb"]
  assert ds.bad()["u02"]["stage"] == "render"
  with h5py.File(os.path.join(ds.shard_dir(4), "render.h5"), "r") as f:
    assert json.loads(f.attrs["failed_mesh_indices"]) == [1]
  assert uids(ds.members(4)) == ["u00", "u02", "u03", "u04"]
  m = ds.manifest()
  assert list(m) == [4] and m[4]["meshes"] == "3" and m[4]["grids"] == "3"
  assert open(os.path.join(ds.shard_dir(4), "render_list.txt")).read().split() == calls[1]


def test_run_downloader_reports_permanent_failures(env, monkeypatch):
  cfg, tmp_path = env
  os.remove(tmp_path / "pool" / "u02.glb")   # not local: from the (fake) repo, which doesn't have it
  monkeypatch.setattr(glb_download, "hf_paths", lambda d: {})
  ds = bd.Dataset(cfg)
  bd.run_downloader(cfg, ds, passes=1)
  assert not os.path.exists(os.path.join(ds.failed_dir, "u02.json"))   # retries=1: not after the first failure
  bd.run_downloader(cfg, ds, passes=2)
  report = json.load(open(os.path.join(ds.failed_dir, "u02.json")))
  assert report["reason"].startswith("u02 is not in") and report["csv_row"] == "2"
  assert sorted(os.listdir(ds.glbs)) == [".downloader.lock", "failed", "u00.glb", "u03.glb", "u04.glb"]


def test_fetch(env, monkeypatch):
  cfg, tmp_path = env
  ds = bd.Dataset(cfg)
  dest = str(tmp_path / "glbs")
  index = glb_download.local_index([tmp_path / "pool"])
  sha = {r["uid"]: r["sha256"] for r in ds.rows}
  p = glb_download.fetch("u00", sha["u00"], dest, index, {})
  assert open(p, "rb").read() == glb_bytes(0)
  assert os.stat(p).st_ino == os.stat(index["u00"]).st_ino   # hard link on the same filesystem

  def download(url, tmp, timeout):
    assert url == f"{glb_download.HF_BASE}/glbs/000-001/u03.glb"
    open(tmp, "wb").write(glb_bytes(3))

  monkeypatch.setattr(glb_download, "_download", download)
  index["u03"] = str(tmp_path / "pool" / "u04.glb")   # a local file with the wrong content: the repo's is taken
  hf = {"u03": "glbs/000-001/u03.glb", "u05": "glbs/000-001/u03.glb"}
  assert open(glb_download.fetch("u03", sha["u03"], dest, index, hf), "rb").read() == glb_bytes(3)
  with pytest.raises(glb_download.PermanentError, match="sha256 mismatch"):
    glb_download.fetch("u05", sha["u05"], dest, {}, hf)
  with pytest.raises(glb_download.PermanentError, match="not in"):
    glb_download.fetch("u06", sha["u06"], dest, {}, hf)

  monkeypatch.undo()   # the real _download, over a fake urlopen
  monkeypatch.setattr(glb_download.urllib.request, "urlopen",
                      lambda url, timeout: (_ for _ in ()).throw(urllib.error.HTTPError(url, 404, "nf", {}, None)))
  with pytest.raises(glb_download.PermanentError, match="HTTP 404"):
    glb_download.fetch("u05", sha["u05"], dest, {}, hf)
  monkeypatch.setattr(glb_download.urllib.request, "urlopen",
                      lambda url, timeout: (_ for _ in ()).throw(urllib.error.HTTPError(url, 503, "busy", {}, None)))
  with pytest.raises(urllib.error.HTTPError):   # transient: raised as it is (the downloader retries it later)
    glb_download.fetch("u05", sha["u05"], dest, {}, hf)
  monkeypatch.setattr(glb_download.urllib.request, "urlopen",
                      lambda url, timeout: (_ for _ in ()).throw(urllib.error.URLError("network down")))
  with pytest.raises(urllib.error.URLError):
    glb_download.fetch("u05", sha["u05"], dest, {}, hf)
  assert sorted(os.listdir(dest)) == ["u00.glb", "u03.glb"]   # no temp files left


def test_lock(tmp_path):
  a, b = bd.Lock(str(tmp_path / "l")), bd.Lock(str(tmp_path / "l"))
  assert a.acquire() and not b.acquire() and b.held_by_other()
  a.f.close()
  assert b.acquire()
