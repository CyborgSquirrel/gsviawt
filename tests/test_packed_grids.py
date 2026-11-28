"""Packed grids (one HDF5 file per shard, grid_io.pack_dir / pack_grids.py): every grid round-trips bit for bit, each grid is one
contiguous byte range, a killed packing leaves nothing usable and a rerun completes, and everything that reads grids (build_grid_index,
GridTargetDataset items, the x0 moments and their cache, data.shards, extra sets, eval_diffusion, DataLoader workers) gives exactly
what the per-file grid dir gives. On the real local test data (data marker): the synthetic shards of test_multi_shard (copies of the
new10 grids, converted to sparse files here) and real new100 sparse grids for the contiguity check."""
import glob
import os
import random
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import test_multi_shard as tms  # noqa: E402  (its synthetic shards, compose / setup_dm / dm_digest)


def sparse_copy(dense_dir, out_dir):
  """dense_dir's grids as sparse files in out_dir (grid_io.convert_file: verified bit for bit)."""
  import grid_io
  os.makedirs(out_dir, exist_ok=True)
  for p in sorted(glob.glob(os.path.join(dense_dir, "*.h5"))):
    grid_io.convert_file(p, os.path.join(out_dir, os.path.basename(p)))
  return out_dir


@pytest.fixture(scope="module")
def data(data_dir):
  """test_multi_shard's shards A, B, C and the real new10 grids, each as a sparse grid dir ("grids") and its packed file
  ("packed", next to the dir)."""
  import grid_io
  d = tms.fixture_data(data_dir)
  for name in ("A", "B", "C"):
    sp = sparse_copy(d[name]["grids"], os.path.join(d["tmp"], f"grids{name}_sparse"))
    packed = os.path.join(d["tmp"], f"shard{name}_grids.h5")
    grid_io.pack_dir(sp, packed)
    d[name] = {**d[name], "grids": sp, "packed": packed}
  sp = sparse_copy(d["real_grids"], os.path.join(d["tmp"], "real_sparse"))
  d["real"] = {"grids": sp, "packed": os.path.join(d["tmp"], "real_grids.h5")}
  grid_io.pack_dir(sp, d["real"]["packed"])
  yield d
  shutil.rmtree(d["tmp"], ignore_errors=True)


def _same(a, b):
  import grid_io
  return grid_io._same(a, b)


def test_round_trip_bit_identical(data):
  """Every grid of the packed file: the source file's datasets (dtype, shape, filters, attrs, decoded values bitwise) and attrs
  (names, HDF5 types, values); read_grid / load_grid / grid_shape the same as from the file; the index from /rows."""
  import grid_io
  import gs_diffusion as gd
  for name in ("A", "B", "real"):
    src, packed = data[name]["grids"], data[name]["packed"]
    assert grid_io.verify_packed(src, packed) == []
    files, fsrc = gd.build_grid_index(src)
    refs, psrc = gd.build_grid_index(packed)
    assert fsrc == psrc and sorted(files) == sorted(refs)
    with h5py.File(packed, "r") as f:
      assert f.attrs["layout"] == "packed" and int(f.attrs["n_grids"]) == len(files) and str(f.attrs["source_h5"]) == fsrc
      for row, path in files.items():
        g = f[refs[row].group]
        assert refs[row] == grid_io.GridRef(os.path.abspath(packed), f"grids/{row}") and refs[row].row == row
        with h5py.File(path, "r") as s:
          assert sorted(s.attrs) == sorted(g.attrs) and set(s) == set(g)
          for k in s.attrs:
            assert s.attrs.get_id(k).get_type() == g.attrs.get_id(k).get_type()
            va, vb = s.attrs[k], g.attrs[k]
            assert (va == vb) if isinstance(va, (str, bytes)) else _same(va, vb), k
          for k in s:
            a, b = s[k], g[k]
            assert (a.dtype, a.shape, a.compression, a.compression_opts, a.shuffle) == \
                (b.dtype, b.shape, b.compression, b.compression_opts, b.shuffle), k
            assert b.chunks in (None, b.shape), k                                 # one chunk per dataset
            assert _same(a[()], b[()]) and dict(a.attrs) == dict(b.attrs), k
          for k in grid_io.FIELD_NAMES:                                           # the fields' compressed bytes themselves
            if s[k].chunks:
              assert s[k].id.read_direct_chunk((0,) * s[k].ndim) == g[k].id.read_direct_chunk((0,) * g[k].ndim), k
        a, b = grid_io.read_grid(path), grid_io.read_grid(refs[row])
        assert _same(a.valid, b.valid) and all(_same(x, y) for x, y in zip(a.idx, b.idx)) and a.attrs.keys() == b.attrs.keys()
        assert all(_same(a.rows[k], b.rows[k]) for k in grid_io.FIELD_NAMES)
        assert all(_same(a.attrs[k], b.attrs[k]) if not isinstance(a.attrs[k], str) else a.attrs[k] == b.attrs[k] for k in a.attrs)
        (da, aa), (db, ab) = grid_io.load_grid(path), grid_io.load_grid(refs[row])
        assert all(_same(da[k], db[k]) for k in da) and aa.keys() == ab.keys()
        assert grid_io.grid_shape(path) == grid_io.grid_shape(refs[row])


def test_verify_catches_a_difference(data, tmp_path):
  """verify_packed is not vacuous: one changed value / attr of a source file is reported."""
  import grid_io
  src = shutil.copytree(data["B"]["grids"], tmp_path / "src")
  with h5py.File(sorted(glob.glob(str(src / "*.h5")))[3], "r+") as f:
    a = f["gaussian_colors"][()]
    a[0, 0] = np.nextafter(a[0, 0], np.float32(np.inf))
    f["gaussian_colors"][...] = a
  with h5py.File(sorted(glob.glob(str(src / "*.h5")))[5], "r+") as f:
    f.attrs["final_loss"] = np.float32(f.attrs["final_loss"]) if "final_loss" in f.attrs else np.float32(0)
  problems = grid_io.verify_packed(str(src), data["B"]["packed"])
  assert len(problems) >= 1 and any("gaussian_colors: values differ" in p for p in problems)


def test_contiguous(data, data_dir, tmp_path):
  """Each grid's datasets (chunks via get_chunk_info, contiguous storage via get_offset) form one byte range no other grid's
  data falls into, in row order -- on real-size grids (20 new100 grids) and the test shards."""
  import grid_io
  real = sorted(glob.glob(os.path.join(data_dir("new100", "anchored_wd0.3", "grids_sparse"), "*.h5")))[:20]
  src = tmp_path / "new100_20"
  src.mkdir()
  for p in real:
    shutil.copy(p, src)
  out = str(tmp_path / "new100_20.h5")
  r = grid_io.pack_dir(str(src), out)
  for packed in (out, data["A"]["packed"], data["real"]["packed"]):
    ext, overlapping = grid_io.grid_extents(packed)
    assert overlapping == [] and len(ext) == int(h5py.File(packed, "r").attrs["n_grids"])
    spans = [ext[row] for row in sorted(ext)]
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))                 # in row order, none overlapping
    assert all(hi - lo >= n for lo, hi, n in spans)
  assert r["overlapping"] == [] and r["n_grids"] == 20
  with h5py.File(out, "r") as f:                                                # (the storage-only changes)
    g = f[grid_io.group_name(int(f["rows"][0]))]
    assert g["layer_offsets"].id.get_create_plist().get_layout() == h5py.h5d.COMPACT
    assert g["layer_valid_lhw"].chunks == g["layer_valid_lhw"].shape


KILLER = """
import os, signal, sys
sys.path.insert(0, {root!r})
import grid_io
real, n = grid_io.write_packed_grid, [0]
def killing(group, src):
  real(group, src)
  n[0] += 1
  if n[0] == {after}:
    group.file.flush()
    os.kill(os.getpid(), signal.SIGKILL)
grid_io.write_packed_grid = killing
grid_io.pack_dir({src!r}, {out!r})
"""


def test_kill_during_packing_then_rerun(data, tmp_path):
  """A packer SIGKILLed mid-write leaves no file at the output name (only the temp, which nothing takes for a grids path) and no
  usable index; the rerun removes the temp and completes, verified."""
  import grid_io
  import gs_diffusion as gd
  out = str(tmp_path / "grids.h5")
  for after in (1, 7):
    p = subprocess.run([sys.executable, "-c", KILLER.format(root=ROOT, after=after, src=data["A"]["grids"], out=out)])
    assert p.returncode == -9
    assert not os.path.exists(out) and os.path.exists(out + ".tmp")
    with pytest.raises((SystemExit, OSError, ValueError, KeyError)):
      gd.build_grid_index(out)
  r = grid_io.pack_dir(data["A"]["grids"], out)
  assert not os.path.exists(out + ".tmp") and r["n_grids"] == 24
  assert grid_io.verify_packed(data["A"]["grids"], out) == []
  assert sorted(gd.build_grid_index(out)[0]) == sorted(gd.build_grid_index(data["A"]["grids"])[0])


def test_pack_refusals(data, tmp_path):
  import grid_io
  with pytest.raises(FileExistsError):
    grid_io.pack_dir(data["A"]["grids"], data["A"]["packed"])                  # a finished file is never overwritten
  with pytest.raises(ValueError, match="inside"):
    grid_io.pack_dir(data["A"]["grids"], os.path.join(data["A"]["grids"], "grids.h5"))
  with pytest.raises(ValueError, match="sparse"):
    grid_io.pack_dir(data["real_grids"], str(tmp_path / "dense.h5"))           # dense grids: convert first
  src = shutil.copytree(data["C"]["grids"], tmp_path / "src")
  (src / "mesh9_view9.h5.tmp").write_bytes(b"")
  with pytest.raises(ValueError, match="tmp"):
    grid_io.pack_dir(str(src), str(tmp_path / "c.h5"))
  assert not os.path.exists(tmp_path / "c.h5") and not os.path.exists(tmp_path / "dense.h5")
  shutil.copy(data["A"]["packed"], src / "packed.h5")                          # a packed file inside a grids dir: refused
  os.remove(src / "mesh9_view9.h5.tmp")
  import gs_diffusion as gd
  with pytest.raises(SystemExit, match="packed"):
    gd.build_grid_index(str(src))


def test_pack_script(data, tmp_path):
  """The Hydra entry point: packs, refuses to run again on the finished file, verify_only passes."""
  out = str(tmp_path / "grids.h5")
  run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "pack_grids.py"), f"grids_dir={data['C']['grids']}",
                                   f"out={out}", *a], capture_output=True, text=True, cwd=str(tmp_path))
  p = run()
  assert p.returncode == 0 and "PACK_OK" in p.stderr + p.stdout, p.stderr
  mtime = os.stat(out).st_mtime_ns
  p = run()
  assert p.returncode != 0 and "never overwritten" in p.stderr + p.stdout
  assert os.stat(out).st_mtime_ns == mtime
  p = run("verify_only=true")
  assert p.returncode == 0 and "VERIFY_OK" in p.stderr + p.stdout
  assert sorted(os.listdir(tmp_path)) == ["grids.h5"]                            # no hydra outputs, no temp


def _single(d, grids, h5, extra=()):
  return tms.setup_dm(tms.compose(tempfile.mkdtemp(dir=d["tmp"]), [f"diffusion.grids_dir={grids}", f"diffusion.source_h5={h5}",
                                                                   *extra]))


def _shards(d, key, names, extra=()):
  ov = "data.shards=[" + ",".join(f"{{source_h5:{d[n]['h5']},grids_dir:{d[n][key]}}}" for n in names) + "]"
  return tms.setup_dm(tms.compose(tempfile.mkdtemp(dir=d["tmp"]), [ov, *extra]))


def _all_items(dm):
  return tms.dm_digest(dm, n_items=10 ** 6)


def test_items_and_moments_single_source(data):
  """diffusion.grids_dir = the packed file: every item of every set (x0_raw, images, ...) and the x0 moments exactly as from the
  per-file dir; the moments computed fresh on each side (empty caches) and again from the cache."""
  import torch
  for name, h5, extra in (("real", data["src"], ("data.view_split.val_fraction=0", "data.val_train=true")),
                          ("A", data["A"]["h5"], ("data.mesh_split.val_fraction=0.25", "data.view_split.val_fraction=0.34",
                                                 "data.val_train=true"))):
    files = _single(data, data[name]["grids"], h5, extra)
    packed = _single(data, data[name]["packed"], h5, extra)
    assert _all_items(files) == _all_items(packed)
    assert all(torch.equal(a, b) for a, b in zip(files.diffusion_stats, packed.diffusion_stats))
    again = _single(data, data[name]["packed"], h5, extra)                     # from the packed file's cache
    assert all(torch.equal(a, b) for a, b in zip(again.diffusion_stats, packed.diffusion_stats))


def test_items_and_moments_multi_shard(data):
  """data.shards over packed files (and one shard packed, one per-file), with a packed extra validation set: the same corpus,
  items and moments as over the per-file dirs."""
  import torch
  ref = _shards(data, "grids", ("A", "B"), (*tms.SPLIT, tms.extra_override("test_c", data["C"])))
  packed = _shards(data, "packed", ("A", "B"), (*tms.SPLIT, tms.extra_override("test_c", {**data["C"], "grids": data["C"]["packed"]})))
  assert list(ref.val_datasets) == list(packed.val_datasets) and "test_c" in packed.val_datasets
  assert _all_items(ref) == _all_items(packed)
  assert all(torch.equal(a, b) for a, b in zip(ref.diffusion_stats, packed.diffusion_stats))
  assert [s["grids_dir"] for s in packed.shards] == [data["A"]["packed"], data["B"]["packed"]]
  mixed = tms.setup_dm(tms.compose(tempfile.mkdtemp(dir=data["tmp"]), [
    f"data.shards=[{{source_h5:{data['A']['h5']},grids_dir:{data['A']['packed']}}},{{source_h5:{data['B']['h5']},"
    f"grids_dir:{data['B']['grids']}}}]", *tms.SPLIT]))
  assert _all_items(mixed) == _all_items(_shards(data, "grids", ("A", "B"), tms.SPLIT))


def test_moments_cache_packed(data, tmp_path, monkeypatch):
  """The packed file's cache: <its dir>/.x0_moments/<tag>/<its name>/row<R>.npz, one per training grid, no collision between two
  packed files in one dir; read back without a grid read; stale (recomputed) once the packed file changes."""
  import torch
  import gs_diffusion as gd
  d = tmp_path / "shard"
  d.mkdir()
  shutil.copy(data["A"]["packed"], d / "grids.h5")
  shutil.copy(data["A"]["packed"], d / "grids_copy.h5")
  first = _single(data, str(d / "grids.h5"), data["A"]["h5"], tms.SPLIT)
  tag = gd.encoding_tag(first.train_ds.encoding)
  rows = sorted(int(v) for v in first.train_ds.inner.volumes.df["view_idx"].to_list())
  assert sorted(os.listdir(d / ".x0_moments" / tag)) == ["grids.h5"]
  assert sorted(os.listdir(d / ".x0_moments" / tag / "grids.h5")) == sorted(f"row{r}.npz" for r in rows)
  copy = _single(data, str(d / "grids_copy.h5"), data["A"]["h5"], tms.SPLIT)  # its own entries
  assert sorted(os.listdir(d / ".x0_moments" / tag)) == ["grids.h5", "grids_copy.h5"]
  assert all(torch.equal(a, b) for a, b in zip(first.diffusion_stats, copy.diffusion_stats))
  calls = []
  real = gd.read_grid
  monkeypatch.setattr(gd, "read_grid", lambda g: calls.append(g) or real(g))
  ds = first.train_ds
  gd.compute_stats(ds, ds.grid_index, gd.moments_cache_dir)
  assert calls == []                                                            # every moment from the cache
  os.utime(d / "grids.h5", ns=(os.stat(d / "grids.h5").st_atime_ns, os.stat(d / "grids.h5").st_mtime_ns + 10 ** 9))
  gd.compute_stats(ds, ds.grid_index, gd.moments_cache_dir)
  assert len(calls) == len(rows)                                                # a changed file: all recomputed


def test_dataloader_workers_and_no_locking(data, tmp_path):
  """A DataLoader with forked workers after the main process has read the packed file (compute_stats, check_grids): the parent's
  handles are closed before the fork, each worker opens its own; the items are those of the main process. And a fresh process
  with HDF5_USE_FILE_LOCKING=FALSE reads the packed grids."""
  import torch
  import grid_io
  dm = _single(data, data["A"]["packed"], data["A"]["h5"], tms.SPLIT)
  ds = dm.val_datasets["val_views_sval"]
  assert grid_io._files                                                         # the main process has the file open
  want = [ds[i]["x0_raw"] for i in range(len(ds))]
  loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2, multiprocessing_context="fork")
  got = [it["x0_raw"] for it in loader]
  assert not grid_io._files                                                     # closed before the fork
  assert len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want))
  code = (f"import sys; sys.path.insert(0, {ROOT!r}); import gs_diffusion as gd, grid_io\n"
          f"i, _ = gd.build_grid_index({data['A']['packed']!r}); g = [grid_io.read_grid(r) for r in i.values()]\n"
          "import h5py; print(len(g), sum(len(x.idx[0]) for x in g))")
  p = subprocess.run([sys.executable, "-c", code], env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"}, capture_output=True,
                     text=True)
  assert p.returncode == 0 and p.stdout.split()[0] == "24", p.stderr


def test_eval_diffusion_packed_shards_run(data):
  """eval_diffusion on a data.shards run whose grids are packed files, with a packed extra set: the datasets and normalization of
  the same run over the per-file dirs."""
  import torch
  from omegaconf import OmegaConf
  import hydra
  import eval_diffusion as ev
  sets = {}
  for key in ("grids", "packed"):
    run_dir = tempfile.mkdtemp(dir=data["tmp"])
    run_cfg = tms.compose(run_dir, ["data.shards=[" + ",".join(f"{{source_h5:{data[n]['h5']},grids_dir:{data[n][key]}}}"
                                                              for n in ("A", "B")) + "]", *tms.SPLIT])
    path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(run_cfg, path)
    with hydra.initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
      cfg = hydra.compose(config_name="eval_diffusion", overrides=[
        f"run_config={path}", f"out={tempfile.mkdtemp(dir=data['tmp'])}", "checkpoints=[init]",
        f"extra_sets={{test_c:{{source_h5:{data['C']['h5']},grids_dir:{data['C'][key]},max_meshes:null,seed:0,max_source_view_num:4}}}}"])
    run = ev.load_run_config(cfg)
    dm = ev.eval_datamodule(run)
    datasets, _ = ev.eval_datasets(cfg, run, dm)
    random.seed(0)
    sets[key] = (dm.diffusion_stats, {n: [tms.item_digest(ds[i]) for i in range(len(ds))] for n, ds in datasets.items()})
  assert all(torch.equal(a, b) for a, b in zip(sets["grids"][0], sets["packed"][0]))
  assert sets["grids"][1] == sets["packed"][1] and "test_c" in sets["packed"][1]
