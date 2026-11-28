"""fit_anchored.py shard mode (build_dataset.py's fitter): several meshes of one render h5 fitted in-process, DONE markers,
regression.json written per view, packing into one grids file + clean-up, a rerun on a packed shard is a no-op, a fit error
raises. Kills are simulated by raising out of the write helpers (a SIGKILL leaves the same files). Uses a small two-mesh render
h5 cut from the local test render (rows 0-7 = mesh 0, rows 8-15 = mesh 1 of car_mesh5_50views_peelrgb_normals.h5)."""
import json
import os

import h5py
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import anchored_3dgs as an
import fit_anchored

pytestmark = [pytest.mark.cuda, pytest.mark.data]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS, HELD = 8, [6, 7]


@pytest.fixture(scope="module")
def render_h5(data_dir, tmp_path_factory):
  src = data_dir("car_mesh5_50views_peelrgb_normals.h5")
  out = str(tmp_path_factory.mktemp("render") / "render.h5")
  with h5py.File(src, "r") as s, h5py.File(out, "w") as d:
    d.attrs.update(dict(s.attrs))
    for k in s:
      if k == "mesh_paths":
        d.create_dataset(k, data=["/x/mesh_a.glb", "/x/mesh_b.glb"], dtype=h5py.string_dtype("utf-8"))
      elif k == "mesh_index":
        d.create_dataset(k, data=np.repeat(np.arange(2, dtype=np.int64), VIEWS))
      else:
        d.create_dataset(k, data=s[k][:2 * VIEWS], chunks=(1, *s[k].shape[1:]) if s[k].ndim > 1 else None,
                         compression=s[k].compression)
  return out


def make_cfg(render_h5, out, **kw):
  over = {"hdf5_path": render_h5, "output_dir": str(out), "regression.held_views": HELD, "combined.steps": 60,
          "finetune.steps": 3, "loss.background": "color", "mesh_indices": "all", "packed": str(out / "grids.h5"), **kw}
  with initialize_config_dir(version_base=None, config_dir=os.path.join(ROOT, "conf")):
    return compose(config_name="fit_anchored", overrides=[f"{k}={v}" if v is not None else f"{k}=null" for k, v in over.items()])


class Killed(BaseException):
  """Stands in for a SIGKILL."""


def names():
  return sorted(f"mesh{m}_view{v}.h5" for m in (0, 1) for v in range(VIEWS) if v not in HELD)


def test_shard_fit_pack_cleanup_noop(render_h5, tmp_path, monkeypatch):
  cfg = make_cfg(render_h5, tmp_path)
  fit_anchored.fit_shard(cfg)
  with h5py.File(tmp_path / "grids.h5", "r") as f:
    assert int(f.attrs["n_grids"]) == 12 and sorted(f["names"].asstr()[()]) == names()
  assert not (tmp_path / an.SPARSE_DIR).exists()
  for m in (0, 1):
    st = tmp_path / "_anchored" / f"mesh{m}"
    assert sorted(os.listdir(st)) == ["DONE", "regression.json", "timing.json"]
    reg = json.loads((st / "regression.json").read_text())
    assert sorted(reg["finetuned"], key=int) == [str(v) for v in range(6)]
    assert reg["finetuned_mean"] == pytest.approx(np.mean(list(reg["finetuned"].values())))
  monkeypatch.setattr(fit_anchored, "fit_mesh", lambda *a: pytest.fail("a packed shard must not be fitted again"))
  fit_anchored.fit_shard(cfg)


def test_kill_mid_finetune_then_resume(render_h5, tmp_path, monkeypatch):
  cfg = make_cfg(render_h5, tmp_path, packed=None)
  real = an.write_view
  calls = {"n": 0}

  def write_view(*a, **k):
    calls["n"] += 1
    if calls["n"] == 3:
      raise Killed()
    return real(*a, **k)

  monkeypatch.setattr(an, "write_view", write_view)
  with pytest.raises(Killed):
    fit_anchored.fit_shard(cfg)
  monkeypatch.undo()
  st = tmp_path / "_anchored" / "mesh0"
  reg = json.loads((st / "regression.json").read_text())
  assert sorted(reg["finetuned"]) == ["0", "1", "2"]   # the score of the view being written is there before its grid
  assert sorted(os.listdir(tmp_path / an.SPARSE_DIR)) == ["mesh0_view0.h5", "mesh0_view1.h5"]
  assert not (st / "DONE").exists()
  fit_anchored.fit_shard(cfg)
  reg = json.loads((st / "regression.json").read_text())
  assert sorted(reg["finetuned"], key=int) == [str(v) for v in range(6)]
  assert reg["finetuned_mean"] == pytest.approx(np.mean(list(reg["finetuned"].values())))   # over all views, not the resumed ones
  assert sorted(os.listdir(tmp_path / an.SPARSE_DIR)) == names()   # packed=None: the per-view files stay
  assert (tmp_path / "_anchored" / "mesh1" / "DONE").exists()


def test_kill_before_done_and_unscored_grids(render_h5, tmp_path, monkeypatch):
  cfg = make_cfg(render_h5, tmp_path, mesh_indices=[0], packed=None)
  real = an.write_json_atomic

  def write_json(path, obj):
    if path.endswith(fit_anchored.DONE):
      raise Killed()
    return real(path, obj)

  monkeypatch.setattr(an, "write_json_atomic", write_json)
  with pytest.raises(Killed):
    fit_anchored.fit_shard(cfg)
  monkeypatch.undo()
  st = tmp_path / "_anchored" / "mesh0"
  assert not (st / "DONE").exists() and len(os.listdir(tmp_path / an.SPARSE_DIR)) == 6
  # every grid there + all scored: the rerun only writes DONE (no fit)
  monkeypatch.setattr(an, "fit_combined", lambda *a, **k: pytest.fail("nothing to fit"))
  monkeypatch.setattr(an, "finetune", lambda *a, **k: pytest.fail("nothing to fit"))
  fit_anchored.fit_shard(cfg)
  assert (st / "DONE").exists()
  monkeypatch.undo()
  # a root resumed with the old code (regression.json only at the end): views with a grid but no score are fitted again
  (st / "DONE").unlink()
  reg = json.loads((st / "regression.json").read_text())
  for k in ("finetuned", "shared"):
    reg[k].pop("4")
  (st / "regression.json").write_text(json.dumps(reg))
  done = []
  real_ft = an.finetune
  monkeypatch.setattr(an, "finetune", lambda *a, **k: done.append(1) or real_ft(*a, **k))
  fit_anchored.fit_shard(cfg)
  assert len(done) == 1 and "4" in json.loads((st / "regression.json").read_text())["finetuned"]


def test_fit_error_raises(render_h5, tmp_path, monkeypatch):
  cfg = make_cfg(render_h5, tmp_path)

  def boom(*a, **k):
    raise RuntimeError("boom")

  monkeypatch.setattr(an, "fit_combined", boom)
  with pytest.raises(RuntimeError, match="boom"):
    fit_anchored.fit_shard(cfg)
  assert not (tmp_path / "_anchored" / "mesh0" / "DONE").exists() and not (tmp_path / "grids.h5").exists()


def test_kill_in_pack_or_cleanup(render_h5, tmp_path, monkeypatch):
  import grid_io
  cfg = make_cfg(render_h5, tmp_path)
  real_fsync = grid_io._fsync
  monkeypatch.setattr(grid_io, "_fsync", lambda *a, **k: (_ for _ in ()).throw(Killed()))
  with pytest.raises(Killed):
    fit_anchored.fit_shard(cfg)   # killed with the packed temp written and verified, before its rename
  assert not (tmp_path / "grids.h5").exists()
  monkeypatch.setattr(grid_io, "_fsync", real_fsync)
  real_cleanup = fit_anchored._cleanup
  monkeypatch.setattr(fit_anchored, "_cleanup", lambda *a: (_ for _ in ()).throw(Killed()))
  with pytest.raises(Killed):
    fit_anchored.fit_shard(cfg)   # packed, killed before the per-view files were deleted
  assert (tmp_path / "grids.h5").exists() and len(os.listdir(tmp_path / an.SPARSE_DIR)) == 12
  monkeypatch.setattr(fit_anchored, "_cleanup", real_cleanup)
  fit_anchored.fit_shard(cfg)
  assert not (tmp_path / an.SPARSE_DIR).exists() and not os.path.exists(str(tmp_path / "grids.h5") + ".tmp")
  assert sorted(os.listdir(tmp_path / "_anchored" / "mesh1")) == ["DONE", "regression.json", "timing.json"]


def test_single_mesh_standalone(render_h5, tmp_path):
  cfg = make_cfg(render_h5, tmp_path, mesh_indices=None, packed=None, mesh_index=1)
  fit_anchored.main.__wrapped__(cfg)
  assert sorted(os.listdir(tmp_path / an.SPARSE_DIR)) == [f"mesh1_view{v}.h5" for v in range(6)]
  assert sorted(os.listdir(tmp_path / "_anchored" / "mesh1")) == ["DONE", "anchors.npz", "combined.npz", "regression.json",
                                                                  "timing.json"]
  with pytest.raises(SystemExit, match="exactly one"):
    fit_anchored.main.__wrapped__(make_cfg(render_h5, tmp_path, mesh_index=1))
