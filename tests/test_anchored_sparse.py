"""fit_anchored.py's sparse output (anchored_3dgs.write_view / view_done): the grids_sparse/<name>.h5 it writes directly is the
file the old path made (dense run-dir gaussians.grid.h5 -> finalize_fit_dataset.py -> convert_grids_sparse.py), byte for byte; no
dense file is written by default; save_dense=true also writes the old dense run dir; resume / old-root behaviour of view_done."""
import filecmp
import json
import os

import h5py
import numpy as np
import pytest

import anchored_3dgs as an
import finalize_fit_dataset as ffd
import grid_io
import gs_diffusion as gd

H, W, L = 12, 10, 3


def synthetic(seed=0):
  """A view (its depth-peel slots in unproject_depth_peel order) and fit_3dgs-layout flat params for it."""
  rng = np.random.default_rng(seed)
  valid = rng.random((H, W, L)) < 0.4
  valid[:, :, 2] = False   # an empty layer
  v, u, layer = np.nonzero(valid)
  n = len(v)
  params = {"means": rng.normal(size=(n, 3)).astype(np.float32), "scales": rng.normal(-5, 1, (n, 3)).astype(np.float32),
            "quats": rng.normal(size=(n, 4)).astype(np.float32), "opacities": rng.normal(size=n).astype(np.float32),
            "sh0": rng.normal(size=(n, 1, 3)).astype(np.float32), "shN": np.zeros((n, 0, 3), np.float32)}
  view = {"u": u, "v": v, "layer": layer, "valid": valid}
  config = {"gaussian_layout": "layered", "hdf5_path": "/x/render.h5",
            "data": {"primary_view": 7, "secondary_views": [5, 6, 8], "validation_views": [9]},
            "iters": 30300, "sh_degree": 0, "lr": {"means_weight_decay": 0.3}, "lambda_dssim": 0.2, "seed": 0,
            "anchored": {"stage": "finetune", "view_in_mesh": 2}}
  return params, view, {"mesh_index": 3, "mesh_path": "/x/mesh.glb"}, config


def write(root, name, save_dense=False, seed=0):
  params, view, meta, config = synthetic(seed)
  an.write_view(str(root), name, params, view, meta, config, 1.5, 0.0123, save_dense=save_dense)


def dense_files(root):
  return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
          if f.endswith(".h5") and not grid_io.is_sparse(h5py.File(os.path.join(d, f), "r"))]


def test_default_writes_sparse_only(tmp_path):
  write(tmp_path, "mesh3_view2")
  sp = tmp_path / "grids_sparse" / "mesh3_view2.h5"
  assert sp.exists()
  assert sorted(os.listdir(tmp_path)) == ["grids_sparse"]
  assert os.listdir(tmp_path / "grids_sparse") == ["mesh3_view2.h5"]   # no .tmp left
  assert dense_files(tmp_path) == []
  with h5py.File(sp, "r") as f:
    assert grid_io.is_sparse(f)
    for k in ("source_h5", "primary_view_row", "mesh_number", "primary_view_in_mesh", "config_json", "final_loss"):
      assert k in f.attrs
    assert int(f.attrs["primary_view_row"]) == 7 and f.attrs["source_h5"] == "/x/render.h5"
  assert an.view_done(str(tmp_path), "mesh3_view2")
  assert not an.view_done(str(tmp_path), "mesh3_view1")
  index, source = gd.build_grid_index(str(tmp_path / "grids_sparse"))
  assert index == {7: str(sp)} and source == "/x/render.h5"


def test_same_file_as_old_dense_path(tmp_path):
  """save_dense=true: the dense run dir as before + the sparse grid; the dense one through finalize (dense) and
  convert_grids_sparse gives the same bytes as the directly written sparse grid."""
  name = "mesh3_view2"
  write(tmp_path, name, save_dense=True)
  assert (tmp_path / name / "gaussians.grid.h5").exists()
  assert an.view_done(str(tmp_path), name, save_dense=True)
  assert ffd.finalize_one(str(tmp_path), name, False, "dense") is None
  dense = tmp_path / "grids" / (name + ".h5")
  old = tmp_path / "old_sparse"
  old.mkdir()
  grid_io.convert_file(str(dense), str(old / (name + ".h5")))
  new = tmp_path / "grids_sparse" / (name + ".h5")
  assert filecmp.cmp(new, old / (name + ".h5"), shallow=False)
  assert grid_io.verify_sparse(str(dense), str(new)) is None
  a, b = grid_io.read_grid(str(new)), grid_io.read_grid(str(old / (name + ".h5")))
  assert np.array_equal(a.valid, b.valid) and a.attrs == b.attrs
  for k in grid_io.FIELD_NAMES:
    assert np.array_equal(a.rows[k].view(np.uint32), b.rows[k].view(np.uint32))
  # still done after finalize moved the dense grid to grids/
  assert an.view_done(str(tmp_path), name, save_dense=True)


def test_view_done_old_dense_root(tmp_path):
  """A root with a dense grid but no sparse one: an error by default (no silent refit), not done with save_dense."""
  name = "mesh3_view2"
  write(tmp_path, name, save_dense=True)
  os.remove(tmp_path / "grids_sparse" / (name + ".h5"))
  with pytest.raises(SystemExit, match="dense grid but no"):
    an.view_done(str(tmp_path), name)
  assert not an.view_done(str(tmp_path), name, save_dense=True)
  write(tmp_path, name, save_dense=True)   # resume: the dense grid is kept, the sparse one written
  assert an.view_done(str(tmp_path), name, save_dense=True) and an.view_done(str(tmp_path), name)
  # sparse only (default run), then resumed with save_dense: the dense grid is missing, so not done
  write(tmp_path, "mesh3_view4")
  assert not an.view_done(str(tmp_path), "mesh3_view4", save_dense=True)


def test_view_done_rejects_foreign_sparse(tmp_path):
  name = "mesh3_view2"
  write(tmp_path, name)
  with h5py.File(tmp_path / "grids_sparse" / (name + ".h5"), "r+") as f:
    f.attrs["config_json"] = json.dumps({"gaussian_layout": "layered"})
  with pytest.raises(SystemExit, match="not an anchored"):
    an.view_done(str(tmp_path), name)

