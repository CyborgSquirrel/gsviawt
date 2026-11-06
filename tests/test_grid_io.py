"""grid_io.py: the sparse grid layout against the dense one it replaces -- round trips (rows, the dense dict, attrs with their HDF5
types), encode_grid bit for bit on both (synthetic grids with an empty, an all-valid and a NaN-filled layer, and the real local test
grids), the x0 statistics and their moments cache, check_grids, orbit_video's reader, the converter's verify / skip / resume
behaviour and finalize_fit_dataset.py grid_format=sparse."""
import json
import os
import shutil
import subprocess
import sys
import types

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import convert_grids_sparse as cgs
import grid_io
import gs_diffusion as gd
from gs_dataset import OPENGL_TO_OPENCV
from test_encode_grid_sparse import OPTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bits(a):
  a = np.asarray(a)
  return a.view(f"u{a.dtype.itemsize}") if a.dtype.kind == "f" else a


def same(a, b):
  return np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(bits(a), bits(b))


def make_grid(hit, seed=0):
  """hit (L, H, W) -> a grid as load_grid gives it (NaN at the invalid slots, (H, W, L, .)) and its item's source (as
  test_encode_grid_sparse.synthetic)."""
  rng = np.random.default_rng(seed)
  layers, h, w = hit.shape
  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = Rotation.random(random_state=seed).as_matrix()
  pose[:3, 3] = rng.normal(size=3)
  c2w = pose @ OPENGL_TO_OPENCV
  xyz_cam = (rng.normal(size=(layers, h, w, 3)) + [0, 0, 3]).astype(np.float32)
  world = (xyz_cam + rng.normal(0, 1e-3, xyz_cam.shape)) @ c2w[:3, :3].T + c2w[:3, 3]
  nan = lambda a: np.where(hit.reshape(hit.shape + (1,) * (a.ndim - 3)), a, np.nan).astype(np.float32)
  t = lambda a: np.ascontiguousarray(np.moveaxis(nan(a), 0, 2))
  grid = {"gaussian_means": t(world), "gaussian_scales": t(np.exp(rng.normal(-7, 2, (layers, h, w, 3)))),
          "gaussian_quats": t(Rotation.random(layers * h * w, random_state=seed).as_quat().reshape(layers, h, w, 4)),
          "gaussian_opacities": t(rng.uniform(0, 1, (layers, h, w))), "gaussian_colors": t(rng.uniform(0, 1, (layers, h, w, 3))),
          "layer_valid": np.ascontiguousarray(np.moveaxis(hit, 0, 2))}
  source = {"hit": torch.from_numpy(hit), "xyz_cam": torch.from_numpy(nan(xyz_cam)), "pose_gl": torch.from_numpy(pose),
            "K_depth": torch.tensor([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])}
  return grid, source


def hits():
  """Masks with the edge cases: an empty layer, an all-valid layer, a single valid slot, no valid slot at all."""
  rng = np.random.default_rng(0)
  mixed = rng.random((6, 13, 11)) < 0.3
  mixed[1] = False
  mixed[2] = True
  one = np.zeros((6, 9, 8), bool)
  one[3, 4, 5] = True
  return {"mixed": mixed, "one": one, "empty": np.zeros((6, 7, 5), bool), "full": np.ones((2, 7, 7), bool)}


def write_dense(path, grid, primary_view_row=0, extra=False):
  """A finalized dense grid file as fit_3dgs.save_output_layered + finalize_fit_dataset.py write it."""
  with h5py.File(path, "w") as f:
    f.attrs["mesh_index"] = np.int64(0)
    f.attrs["mesh_path"] = "/x/mesh.glb"
    f.attrs["layer_layout"] = "(H, W, 6) like depth_peel; NaN = empty slot"
    for k in grid_io.FIELD_NAMES:
      f.create_dataset(k, data=grid[k], chunks=grid[k].shape, compression="gzip", compression_opts=4)
    f.create_dataset("layer_valid", data=grid["layer_valid"])
    f.attrs.update({"source_h5": "/x/render.h5", "primary_view_row": primary_view_row, "supervision_views_rows": json.dumps([1, 2]),
                    "means_optimized": True, "final_loss": 0.01, "config_json": json.dumps({"iters": 5})})
    f.attrs.create("fixed_str", np.bytes_("abc"))   # a fixed-length string: must stay one
    if extra:
      f.create_dataset("notes", data=np.arange(5, dtype=np.int16), compression="gzip")
      f["notes"].attrs["unit"] = "px"


@pytest.fixture
def files(tmp_path):
  """{name: (dense path, sparse path, grid, source)} for every hits() mask."""
  out = {}
  for i, (name, hit) in enumerate(hits().items()):
    grid, source = make_grid(hit, seed=i)
    d, s = str(tmp_path / f"{name}.dense.h5"), str(tmp_path / f"{name}.sparse.h5")
    write_dense(d, grid, primary_view_row=i, extra=name == "mixed")
    grid_io.convert_file(d, s)
    out[name] = (d, s, grid, source)
  return out


def test_round_trip(files):
  for name, (d, s, grid, _) in files.items():
    gd_, ad = grid_io.load_grid(d)
    gs, as_ = grid_io.load_grid(s)
    assert gd_.keys() == gs.keys() and all(same(gd_[k], gs[k]) for k in gd_), name   # NaN payloads included
    assert all(same(grid[k], gs[k]) for k in grid)
    assert ad == as_ and "layout" not in as_ and "config_json" not in as_
    rd, rs = grid_io.read_grid(d), grid_io.read_grid(s)
    assert same(rd.valid, rs.valid) and all(same(rd.idx[i], rs.idx[i]) for i in range(3))
    assert all(same(rd.rows[k], rs.rows[k]) for k in grid_io.FIELD_NAMES)
    assert rs.rows["gaussian_opacities"].ndim == 1 and rs.rows["gaussian_means"].dtype == np.float32
    assert grid_io.grid_shape(d) == grid_io.grid_shape(s) == grid["layer_valid"].shape == rs.shape
    assert grid_io.verify_sparse(d, s) is None
    with h5py.File(d) as fd, h5py.File(s) as fs:
      assert fs.attrs["layout"] == "sparse" and fs.attrs["layout_version"] == 2
      assert fs.attrs.get_id("fixed_str").get_type() == fd.attrs.get_id("fixed_str").get_type()
      assert fs.attrs["config_json"] == fd.attrs["config_json"]
      off = fs["layer_offsets"][()]
      assert off[0] == 0 and (np.diff(off) == grid["layer_valid"].sum((0, 1))).all()
      if name == "mixed":
        assert same(fs["notes"][()], fd["notes"][()]) and fs["notes"].attrs["unit"] == "px"
        assert fs["gaussian_means"].compression == "gzip" and fs["gaussian_means"].shuffle


def test_layer_offsets_slice_a_layer(files):
  d, s, grid, _ = files["mixed"]
  g = grid_io.read_grid(s)
  with h5py.File(s) as f:
    off = f["layer_offsets"][()]
  for layer in range(g.valid.shape[0]):
    li, yi, xi = g.idx
    sl = slice(off[layer], off[layer + 1])
    assert (li[sl] == layer).all()
    assert same(g.rows["gaussian_colors"][sl], grid["gaussian_colors"][:, :, layer][g.valid[layer]])


def test_encode_grid_bit_identical(files):
  """encode_grid on a sparse file's rows == on the dense file (both via read_grid) == on load_grid's dense dict, every encoding,
  the rng streams left in the same place."""
  for name, (d, s, _, source) in files.items():
    dense_dict, rd, rs = grid_io.load_grid(d)[0], grid_io.read_grid(d), grid_io.read_grid(s)
    for enc in OPTIONS:
      rngs = [np.random.default_rng(7) for _ in range(3)]
      x = [gd.encode_grid(g, source, rng=r, **enc) for g, r in zip((dense_dict, rd, rs), rngs)]
      assert all(same(x[0], xi) for xi in x[1:]), (name, enc)
      assert len({r.random() for r in rngs}) == 1


def test_hit_mismatch_raises_on_sparse(files):
  _, s, _, source = files["mixed"]
  source = dict(source, hit=~source["hit"])
  with pytest.raises(ValueError, match="disagrees"):
    gd.encode_grid(grid_io.read_grid(s), source)


def test_verify_catches_differences(files, tmp_path):
  d, s, _, _ = files["mixed"]
  bad = str(tmp_path / "bad.h5")
  for edit in ("row", "mask", "attr", "attr_type", "extra"):
    shutil.copy(s, bad)
    with h5py.File(bad, "r+") as f:
      if edit == "row":
        r = f["gaussian_scales"][()]
        r[3, 1] = np.nextafter(r[3, 1], np.inf)   # one ulp
        f["gaussian_scales"][...] = r
      elif edit == "mask":   # a slot moved: the same count, so layer_offsets still adds up
        m = f["layer_valid_lhw"][()]
        y, x = np.argwhere(m[0])[0]
        z = np.argwhere(~m[0])[0]
        m[0, y, x], m[0, z[0], z[1]] = False, True
        f["layer_valid_lhw"][...] = m
      elif edit == "attr":
        f.attrs["primary_view_row"] = 99
      elif edit == "attr_type":
        f.attrs["fixed_str"] = "abc"   # now a variable-length string
      else:
        del f["notes"]
    assert grid_io.verify_sparse(d, bad) is not None, edit


def test_convert_refuses_a_non_nan_invalid_slot(tmp_path):
  """An invalid slot holding a value would be dropped by the sparse layout: the conversion fails and leaves nothing behind."""
  grid, _ = make_grid(hits()["mixed"])
  y, x, l = np.argwhere(~grid["layer_valid"])[0]
  grid["gaussian_opacities"][y, x, l] = 0.5
  d, s = str(tmp_path / "d.h5"), str(tmp_path / "s.h5")
  write_dense(d, grid)
  with pytest.raises(RuntimeError, match="not exact"):
    grid_io.convert_file(d, s)
  assert not os.path.exists(s) and not os.path.exists(s + ".tmp")


def test_convert_refuses_a_sparse_source(files, tmp_path):
  with pytest.raises(ValueError, match="already sparse"):
    grid_io.convert_file(files["one"][1], str(tmp_path / "again.h5"))


def test_check_grids_both_layouts(files):
  for p in files["full"][:2]:
    gd.check_grids({0: p}, image_size=7, patch_size=7, num_layers=2)
    with pytest.raises(SystemExit, match="7x7 with 2 layers"):
      gd.check_grids({0: p}, image_size=8, patch_size=8, num_layers=2)


def test_build_grid_index_both_layouts(files, tmp_path):
  for i, kind in enumerate(("dense", "sparse")):
    sub = tmp_path / kind
    sub.mkdir()
    for name, paths in files.items():
      shutil.copy(paths[i], sub / f"{name}.h5")
    index, src = gd.build_grid_index(str(sub))
    assert src == "/x/render.h5" and sorted(index) == list(range(len(files)))


def test_orbit_load_h5_same_on_both(files):
  from omegaconf import OmegaConf
  import orbit_video
  ds = OmegaConf.load(os.path.join(ROOT, "conf", "orbit.yaml")).datasets
  for name, (d, s, _, _) in files.items():
    a, b = orbit_video.load_h5(d, ds), orbit_video.load_h5(s, ds)
    assert a.keys() == b.keys() and all(same(a[k], b[k]) for k in a), name


class _Dataset:
  """The bits of GridTargetDataset _grid_moments reads: inner[i] (the item), encoding."""

  def __init__(self, sources, encoding):
    self.inner = [{"source": s, "source_view": i} for i, s in enumerate(sources)]
    self.encoding = gd.normalize_encoding(encoding)


def _dirs(files, tmp_path):
  dense = tmp_path / "grids"
  dense.mkdir()
  for name, (d, _, _, _) in files.items():
    shutil.copy(d, dense / f"{name}.h5")
  return str(dense), str(tmp_path / "grids_sparse")


def test_moments_identical_and_carried_over(files, tmp_path, monkeypatch):
  """The x0 moments of the sparse copies are the dense ones bit for bit; the converter carries the dense dir's cache over, restamped,
  so the sparse dir's _grid_moments reads it without encoding a grid; an entry stale for its dense file is left behind."""
  dense, sparse = _dirs(files, tmp_path)
  names = sorted(files)
  enc = {"scale_order": "sorted", "scale_floor_px": 0.1, "rot_form": "random"}
  ds = _Dataset([files[n][3] for n in names], enc)
  cache_d = os.path.join(dense, ".x0_moments")
  ref = [gd._grid_moments(ds, i, os.path.join(dense, f"{n}.h5"), cache_d) for i, n in enumerate(names)]
  gd._grid_moments(_Dataset([files[n][3] for n in names], {}), 0, os.path.join(dense, f"{names[0]}.h5"), cache_d)   # tag ""
  os.utime(os.path.join(dense, "one.h5"), ns=(1, 1))                 # one.h5's entries now stale
  os.makedirs(sparse)
  for n in names:
    assert cgs.convert_one((dense, sparse, f"{n}.h5"))[1] == "converted"
  assert not os.path.exists(os.path.join(sparse, ".x0_moments", "sort_floor0.1px_rotrandom", "one.h5.npz"))
  tag = gd.encoding_tag(enc)
  assert os.path.exists(os.path.join(sparse, ".x0_moments", tag, "mixed.h5.npz"))
  assert os.path.exists(os.path.join(sparse, ".x0_moments", f"{names[0]}.h5.npz"))   # the default encoding's ("" tag)
  # fresh, no cache: the same moments from the sparse files
  fresh = [gd._grid_moments(ds, i, os.path.join(sparse, f"{n}.h5"), None) for i, n in enumerate(names)]
  for a, b in zip(ref, fresh):
    assert all(same(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
  # carried over: read from the cache, nothing encoded (but one.h5, whose entry was stale)
  encoded = []
  real = gd.encode_grid
  monkeypatch.setattr(gd, "encode_grid", lambda *a, **k: encoded.append(1) or real(*a, **k))
  cached = [gd._grid_moments(ds, i, os.path.join(sparse, f"{n}.h5"), os.path.join(sparse, ".x0_moments")) for i, n in enumerate(names)]
  assert len(encoded) == 1
  for a, b in zip(ref, cached):
    assert all(same(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def test_converter_skip_and_resume(files, tmp_path):
  dense, sparse = _dirs(files, tmp_path)
  os.makedirs(sparse)
  open(os.path.join(sparse, "mixed.h5.tmp"), "w").write("a crashed run's leftover")
  first = {n: cgs.convert_one((dense, sparse, f"{n}.h5"))[1] for n in files}
  assert set(first.values()) == {"converted"} and not os.path.exists(os.path.join(sparse, "mixed.h5.tmp"))
  mtimes = {n: os.stat(os.path.join(sparse, f"{n}.h5")).st_mtime_ns for n in files}
  again = {n: cgs.convert_one((dense, sparse, f"{n}.h5"))[1] for n in files}
  assert set(again.values()) == {"skipped"}
  assert mtimes == {n: os.stat(os.path.join(sparse, f"{n}.h5")).st_mtime_ns for n in files}   # not rewritten
  shutil.copy(os.path.join(dense, "one.h5"), os.path.join(sparse, "one.h5"))   # a dense file where a sparse one should be
  assert cgs.convert_one((dense, sparse, "one.h5"))[1].startswith("error")
  for n in files:   # the originals are untouched
    assert same(grid_io.load_grid(os.path.join(dense, f"{n}.h5"))[0]["gaussian_means"], files[n][2]["gaussian_means"])


def test_converter_cli(files, tmp_path):
  """The Hydra entry point, 2 workers, the default out dir <grids_dir>_sparse, run twice."""
  dense, sparse = _dirs(files, tmp_path)
  run = lambda: subprocess.run([sys.executable, os.path.join(ROOT, "convert_grids_sparse.py"), f"grids_dir={dense}", "workers=2"],
                               capture_output=True, text=True, cwd=str(tmp_path))
  r = run()
  assert r.returncode == 0, r.stderr
  assert f"converted {len(files)}, skipped 0" in r.stdout + r.stderr
  assert sorted(os.listdir(sparse)) == sorted(f"{n}.h5" for n in files)
  r = run()
  assert r.returncode == 0 and f"converted 0, skipped {len(files)}" in r.stdout + r.stderr
  for n in files:
    assert grid_io.verify_sparse(os.path.join(dense, f"{n}.h5"), os.path.join(sparse, f"{n}.h5")) is None
  bad = subprocess.run([sys.executable, os.path.join(ROOT, "convert_grids_sparse.py"), f"grids_dir={dense}", f"out_dir={dense}"],
                       capture_output=True, text=True, cwd=str(tmp_path))
  assert bad.returncode != 0 and "never writes in place" in bad.stdout + bad.stderr


@pytest.mark.parametrize("grid_format", ["dense", "sparse"])
def test_finalize_grid_format(tmp_path, grid_format):
  """finalize_fit_dataset.py on a grid-only (anchored) fit: grids/<name>.h5 in the asked layout, the same grid either way; with
  sparse the fit's dense grid is kept in extra/<name>/."""
  import finalize_fit_dataset as ffd
  grid, _ = make_grid(hits()["mixed"])
  run = tmp_path / "mesh3_view7"
  run.mkdir()
  cfg = {"gaussian_layout": "layered", "hdf5_path": "/x/render.h5", "data": {"primary_view": 37, "secondary_views": [38]},
         "iters": 5, "sh_degree": 0, "lr": {"means_weight_decay": 0.1}}
  dense = str(run / "gaussians.grid.h5")
  write_dense(dense, grid)
  with h5py.File(dense, "r+") as f:
    f.attrs.update({"config_json": json.dumps(cfg), "final_loss": 0.5, "num_gaussians": int(grid["layer_valid"].sum())})
  assert ffd.finalize_one(str(tmp_path), "mesh3_view7", False, grid_format) is None
  target = str(tmp_path / "grids" / "mesh3_view7.h5")
  with h5py.File(target) as f:
    assert grid_io.is_sparse(f) == (grid_format == "sparse") and f.attrs["primary_view_row"] == 37
  out, attrs = grid_io.load_grid(target)
  assert all(same(grid[k], out[k]) for k in grid) and attrs["mesh_number"] == 3
  kept = tmp_path / "extra" / "mesh3_view7" / "gaussians.grid.h5"
  assert kept.exists() == (grid_format == "sparse")
  if grid_format == "sparse":
    assert grid_io.verify_sparse(str(kept), target) is None


@pytest.mark.data
def test_real_grids(data_dir, tmp_path):
  """The local test grids (bla/grids_test) with their render items: the sparse copies encode bit for bit as the dense files for
  every encoding (seeded as the validation sets), and compute_stats over them is identical."""
  from gs_dataset import WTFeatureDataset, render_catalog
  grids = data_dir("grids_test", "fit", "grids")
  idx, src = gd.build_grid_index(grids)
  data_dir(src)
  out = tmp_path / "grids_sparse"
  out.mkdir()
  sidx = {}
  for row, p in idx.items():
    sidx[row] = str(out / os.path.basename(p))
    grid_io.convert_file(p, sidx[row])
    assert grid_io.grid_shape(sidx[row]) == grid_io.grid_shape(p)
    assert os.path.getsize(sidx[row]) < 0.8 * os.path.getsize(p)
  assert gd.build_grid_index(str(out)) == (sidx, src)
  gd.check_grids(sidx, 504, 14, 6)
  cat = render_catalog(src)
  cat = cat.take([p for p, v in enumerate(cat.df["view_idx"].to_list()) if v in set(idx)])
  inner = WTFeatureDataset(gd.make_listing(idx, src, str(tmp_path / "listing.h5")), cat, num_layers=6, num_views=0,
                           read_features=False)
  for i in range(len(inner)):
    item = inner[i]
    rd, rs = grid_io.read_grid(idx[item["source_view"]]), grid_io.read_grid(sidx[item["source_view"]])
    for enc in OPTIONS:
      ra, rb = (np.random.default_rng(item["source_view"]) for _ in range(2))
      assert same(gd.encode_grid(rd, item["source"], rng=ra, **enc), gd.encode_grid(rs, item["source"], rng=rb, **enc)), enc
      assert ra.random() == rb.random()
  for enc in (None, {"scale_order": "sorted", "scale_floor_px": 0.1}):
    stats = [gd.compute_stats(gd.GridTargetDataset(inner, ix, enc), ix) for ix in (idx, sidx)]
    assert all(same(a, b) for a, b in zip(*[[t.numpy() for t in s] for s in stats]))
