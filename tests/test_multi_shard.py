"""data.shards (several render files + their grids as one training corpus) and data.extra_val_sets (validation sets from other
render files during training), on the real local test data (data marker): the new10 render file and the 3 fitted grids of its
mesh 0, plus small synthetic shards made of copies of those rows (each "mesh" of a shard = mesh 0's first views under another
mesh name, each with the 3 grids re-pointed at its own rows), so every item and grid is real data.

Single-source behaviour (data.shards null) is checked against tests/golden/multi_shard_single_source.json: digests of the datasets,
splits, x0 moments and items that GSDataModule.setup produced on master 11081cc, BEFORE data.shards existed
(`python tests/test_multi_shard.py golden` on that code wrote it)."""
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest

if __name__ != "__main__":
  pytestmark = pytest.mark.data

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden", "multi_shard_single_source.json")
SRC_ROWS = (0, 1, 2, 0, 1, 2)   # the new10 rows (mesh 0's views 0-2, the ones with a grid) each synthetic mesh is made of
ROWS_PER_MESH = len(SRC_ROWS)


def _root():
  sys.path.insert(0, os.path.dirname(HERE))
  from conftest import data_path
  return data_path


def make_shard(path, src_h5, mesh_names):
  """A render file of len(mesh_names) meshes, each = rows SRC_ROWS of src_h5 (mesh 0's views 0-2, twice), in order."""
  n = len(mesh_names) * ROWS_PER_MESH
  with h5py.File(src_h5, "r") as s, h5py.File(path, "w") as d:
    for name, ds in s.items():
      if name == "mesh_paths":
        d.create_dataset(name, data=np.array(mesh_names, dtype=object), dtype=h5py.string_dtype())
      elif name == "mesh_index":
        d.create_dataset(name, data=np.repeat(np.arange(len(mesh_names), dtype=np.int64), ROWS_PER_MESH))
      else:
        rows = np.tile(np.array(SRC_ROWS), len(mesh_names))
        d.create_dataset(name, data=ds[:max(SRC_ROWS) + 1][rows], chunks=(1, *ds.shape[1:]), compression=ds.compression)
    for k, v in s.attrs.items():
      d.attrs[k] = v
  assert n == len(mesh_names) * ROWS_PER_MESH
  return path


def make_grids(out_dir, grids_dir, shard_h5, n_meshes):
  """A grid for every row of the shard: the real grid of the new10 row it is a copy of (mesh 0, views 0-2), re-pointed at it."""
  os.makedirs(out_dir, exist_ok=True)
  for m in range(n_meshes):
    for v, src_row in enumerate(SRC_ROWS):
      dst = os.path.join(out_dir, f"mesh{m}_view{v}.h5")
      shutil.copy(os.path.join(grids_dir, f"mesh0_view{src_row}.h5"), dst)
      with h5py.File(dst, "r+") as f:
        f.attrs["primary_view_row"] = np.int64(m * ROWS_PER_MESH + v)
        f.attrs["source_h5"] = shard_h5
        f.attrs["mesh_index"] = np.int64(m)
  return out_dir


@pytest.fixture(scope="module")
def data(data_dir):
  return fixture_data(data_dir)


def fixture_data(data_dir):
  src = data_dir("trellis_new10_50views_eevee_504.h5")
  grids = data_dir("grids_test", "fit", "grids")
  tmp = tempfile.mkdtemp(prefix="multi_shard_")
  out = {"src": src, "tmp": tmp}
  out["real_grids"] = os.path.join(tmp, "real_grids")   # a copy: the moment caches go there, not into the test data
  shutil.copytree(grids, out["real_grids"], ignore=shutil.ignore_patterns(".x0_moments"))
  for name, meshes in (("A", ["shardA/m0.glb", "shardA/m1.glb", "shardA/m2.glb", "shardA/m3.glb"]),
                       ("B", ["shardB/m0.glb", "shardB/m1.glb", "shardB/m2.glb"]),
                       ("C", ["shardC/m0.glb", "shardC/m1.glb"])):
    h5 = make_shard(os.path.join(tmp, f"shard{name}.h5"), src, meshes)
    out[name] = {"h5": h5, "grids": make_grids(os.path.join(tmp, f"grids{name}"), grids, h5, len(meshes)), "meshes": meshes}
  return out


def compose(output_dir, overrides=()):
  from hydra import compose as hcompose, initialize_config_dir
  with initialize_config_dir(config_dir=os.path.join(os.path.dirname(HERE), "conf"), version_base=None):
    return hcompose(config_name="train_gs", overrides=[
      f"output_dir={output_dir}", "diffusion.enabled=true", "data.num_views=2", "data.item=null", "data.input=image",
      "data.depth_size=504", "seed=42", "diffusion.scale_encoding.scale_order=sorted", "diffusion.scale_encoding.scale_floor_px=0.1",
      "diffusion.scale_encoding.scale_floor_mode=max", "diffusion.scale_encoding.rot_form=keep", *overrides])


def setup_dm(cfg):
  import train_gs
  dm = train_gs.GSDataModule(cfg)
  dm.setup()
  return dm


def _h(*arrays):
  h = hashlib.sha256()
  for a in arrays:
    a = np.ascontiguousarray(np.asarray(a))
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
  return h.hexdigest()[:16]


def item_digest(item):
  """A digest of everything an item carries (tensors / arrays / numbers / lists, nested)."""
  import torch
  parts = []

  def walk(prefix, v):
    if isinstance(v, dict):
      for k in sorted(v):
        walk(f"{prefix}/{k}", v[k])
    elif isinstance(v, torch.Tensor):
      parts.append(f"{prefix}:{_h(v.cpu().numpy())}")
    elif isinstance(v, np.ndarray):
      parts.append(f"{prefix}:{_h(v)}")
    elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (torch.Tensor, np.ndarray, dict)):
      for k, x in enumerate(v):
        walk(f"{prefix}[{k}]", x)
    else:
      parts.append(f"{prefix}:{v!r}")
  walk("", item)
  return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def dm_digest(dm, n_items=2):
  """{set: {volumes (mesh_id, view_idx), pool, items}} + the x0 moments, of a set-up GSDataModule."""
  out = {"stats": [_h(s.numpy()) for s in dm.diffusion_stats]}
  for name, ds in {"train": dm.train_ds, **dm.val_datasets}.items():
    inner = ds.inner
    vols = inner.volumes.df.select(["mesh_id", "view_idx"]).rows()
    items = []
    for i in range(min(n_items, len(ds))):
      random.seed(f"digest/{name}/{i}")
      items.append(item_digest(ds[i]))
    out[name] = {"volumes": [list(map(int, r)) for r in vols],
                 "pool": {str(m): sorted(map(int, v)) for m, v in sorted(inner.pool.items())}, "items": items}
  return out


def single_source_configs(d):
  """(name, overrides) of the single-source setups the golden file holds."""
  return [
    ("real_new10", [f"diffusion.grids_dir={d['real_grids']}", f"diffusion.source_h5={d['src']}", "data.view_split.val_fraction=0",
                    "data.val_train=true"]),
    ("shardA_meshsplit", [f"diffusion.grids_dir={d['A']['grids']}", f"diffusion.source_h5={d['A']['h5']}",
                          "data.mesh_split.val_fraction=0.25", "data.view_split.val_fraction=0.34", "data.val_train=true"]),
  ]


def golden_digests(d):
  return {name: dm_digest(setup_dm(compose(tempfile.mkdtemp(dir=d["tmp"]), ov))) for name, ov in single_source_configs(d)}


def test_single_source_bit_identical_to_master(data):
  """data.shards null: datasets, splits, pools, x0 moments and items exactly as master 11081cc built them."""
  with open(GOLDEN) as f:
    golden = json.load(f)
  assert golden_digests(data) == golden


def shards_override(*specs):
  return "data.shards=[" + ",".join(f"{{source_h5:{s['h5']},grids_dir:{s['grids']}}}" for s in specs) + "]"


def multi(d, *names, extra=()):
  return setup_dm(compose(tempfile.mkdtemp(dir=d["tmp"]), [shards_override(*(d[n] for n in names)), *extra]))


def single(d, name, extra=()):
  return setup_dm(compose(tempfile.mkdtemp(dir=d["tmp"]), [f"diffusion.grids_dir={d[name]['grids']}",
                                                            f"diffusion.source_h5={d[name]['h5']}", *extra]))


SPLIT = ("data.mesh_split.val_fraction=0", "data.view_split.val_fraction=0.34", "data.val_train=true")


def test_one_shard_equals_single_source(data):
  """data.shards with one shard: the golden single-source digests of that render file."""
  with open(GOLDEN) as f:
    golden = json.load(f)["shardA_meshsplit"]
  dm = setup_dm(compose(tempfile.mkdtemp(dir=data["tmp"]), [shards_override(data["A"]), *single_source_configs(data)[1][1][2:]]))
  assert dm_digest(dm) == golden


def test_two_shard_corpus_ids_splits_items(data):
  import torch
  dm = multi(data, "A", "B", extra=SPLIT)
  sa, sb = single(data, "A", SPLIT), single(data, "B", SPLIT)
  nA = len(data["A"]["meshes"]) * 6
  off = {"A": (0, 0), "B": (nA, len(data["A"]["meshes"]))}   # (row offset, mesh offset) of shard B in the corpus
  assert [(s["row_offset"], s["mesh_offset"]) for s in dm.shards] == [off["A"], off["B"]]
  # unique ids, every mesh of both shards
  vols = dm.train_ds.inner.volumes.df
  assert vols.select(["view_idx"]).is_unique().all() and vols["mesh_id"].n_unique() == 4 + 3
  for name, s in (("A", sa), ("B", sb)):
    ro, mo = off[name]
    for set_name in ("train", "val_views_sval", "val_views_strain", "val_train"):
      mine = (dm.train_ds if set_name == "train" else dm.val_datasets[set_name]).inner.volumes.df
      theirs = (s.train_ds if set_name == "train" else s.val_datasets[set_name]).inner.volumes.df
      mine = mine.filter((mine["view_idx"] >= ro) & (mine["view_idx"] < ro + len(data[name]["meshes"]) * 6))
      assert sorted(zip((mine["mesh_id"] - mo).to_list(), (mine["view_idx"] - ro).to_list())) == \
          sorted(zip(theirs["mesh_id"].to_list(), theirs["view_idx"].to_list())), (name, set_name)
    # the per-mesh split, by mesh name, as split.json records it: the single-shard run's
    for mesh, m in s.split["meshes"].items():
      mm = dm.split["meshes"][mesh]
      assert (mm["train_views"], mm["val_views"], mm["shard"], mm["mesh_id"]) == (m["train_views"], m["val_views"], int(name == "B"),
                                                                                  m["mesh_id"] + mo)
    # training items: byte-equal to the single-shard ones (ids shifted by the offsets)
    for i in range(len(s.train_ds)):
      src = int(s.train_ds.inner.volumes.df["view_idx"][i]) + ro
      j = dm.train_ds.inner.volumes.df["view_idx"].to_list().index(src)
      random.seed(f"cmp/{i}")
      a = s.train_ds[i]
      random.seed(f"cmp/{i}")
      b = dm.train_ds[j]
      assert b["mesh_index"] == a["mesh_index"] + mo and b["source_view"] == a["source_view"] + ro
      assert b["target_views"] == [t + ro for t in a["target_views"]]
      for k in ("image", "x0_raw"):
        assert torch.equal(a[k], b[k])
      for grp in ("source", "views"):
        for k in a[grp]:
          assert torch.equal(a[grp][k], b[grp][k]) or torch.allclose(a[grp][k], b[grp][k], rtol=0, atol=0, equal_nan=True), (grp, k)
  with open(os.path.join(dm.cfg.output_dir, "split.json")) as f:
    rec = json.load(f)
  assert rec["held_out_meshes"] == [] and sorted(rec["held_out_views"]) == sorted(data["A"]["meshes"] + data["B"]["meshes"])
  assert [s["source_h5"] for s in rec["shards"]] == [data["A"]["h5"], data["B"]["h5"]]


def test_two_shard_moments_are_the_whole_corpus(data):
  """The x0 normalization: over every training grid of both shards (direct, uncached), each grid's moments cached in its own
  shard's grids dir."""
  import torch
  import gs_diffusion as gd
  from grid_io import read_grid
  dm = multi(data, "A", "B", extra=SPLIT)
  ds = dm.train_ds
  xs = []
  inner = ds.inner
  for i in range(len(ds)):
    item = inner[i]
    grid = read_grid(ds.grid_index[int(item["source_view"])])
    x = gd.encode_grid(grid, item["source"], rng=np.random.default_rng(int(item["source_view"])), **ds.encoding)
    xs.append(torch.from_numpy(x)[:, item["source"]["hit"]].double())
  x = torch.cat(xs, 1)
  mean, std = x.mean(1), x.var(1, unbiased=False).sqrt().clamp(min=1e-4)
  got = dm.diffusion_stats
  assert torch.allclose(got[0].double(), mean, rtol=1e-5, atol=1e-6) and torch.allclose(got[1].double(), std, rtol=1e-4, atol=1e-6)
  assert torch.equal(got[2], x.amin(1).float()) and torch.equal(got[3], x.amax(1).float())
  tag = gd.encoding_tag(ds.encoding)
  train_grids = {ds.grid_index[int(v)] for v in ds.inner.volumes.df["view_idx"].to_list()}
  for n in ("A", "B"):   # each training grid's moments in its own shard's cache, nothing else there
    cached = set(os.listdir(os.path.join(data[n]["grids"], ".x0_moments", tag)))
    assert cached == {os.path.basename(p) + ".npz" for p in train_grids if os.path.dirname(p) == data[n]["grids"]} and cached
  # the combined cache read again (second set-up, from the caches): the same numbers
  again = multi(data, "A", "B", extra=SPLIT).diffusion_stats
  assert all(torch.equal(a, b) for a, b in zip(got, again))
  # and != one shard alone where the shards differ (here the same grids: equal) -- a corpus mean is the count-weighted mean
  sa, sb = single(data, "A", SPLIT), single(data, "B", SPLIT)
  na, nb = (sum(int(sx.inner[i]["source"]["hit"].sum()) for i in range(len(sx))) for sx in (sa.train_ds, sb.train_ds))
  combo = (sa.diffusion_stats[0].double() * na + sb.diffusion_stats[0].double() * nb) / (na + nb)
  assert torch.allclose(combo, got[0].double(), rtol=1e-5, atol=1e-6)


def test_shards_guards(data):
  with pytest.raises(SystemExit, match="null"):
    setup_dm(compose(tempfile.mkdtemp(dir=data["tmp"]), [shards_override(data["A"], data["B"]),
                                                          f"diffusion.grids_dir={data['A']['grids']}"]))
  dup = dict(data["B"])
  dup_h5 = os.path.join(data["tmp"], "shardB_dup.h5")
  make_shard(dup_h5, data["src"], [data["A"]["meshes"][0], "other.glb"])
  dup["h5"], dup["grids"] = dup_h5, make_grids(os.path.join(data["tmp"], "grids_dup"), os.path.join(data["real_grids"]), dup_h5, 2)
  with pytest.raises(SystemExit, match="distinct"):
    setup_dm(compose(tempfile.mkdtemp(dir=data["tmp"]), [shards_override(data["A"], dup)]))


def extra_override(spec_name, d, max_meshes="null", max_views=4):
  return (f"+data.extra_val_sets={{{spec_name}:{{source_h5:{d['h5']},grids_dir:{d['grids']},max_meshes:{max_meshes},seed:0,"
          f"max_source_view_num:{max_views}}}}}").replace("+data", "data")


def test_extra_val_set(data):
  import torch
  import gs_diffusion as gd
  import train_gs
  base = multi(data, "A", "B", extra=SPLIT)
  dm = multi(data, "A", "B", extra=(*SPLIT, extra_override("test_c", data["C"])))
  assert list(dm.val_datasets) == [*base.val_datasets, "test_c"]                 # after the run's own sets
  assert all(torch.equal(a, b) for a, b in zip(base.diffusion_stats, dm.diffusion_stats))   # the run's normalization
  ds = dm.val_datasets["test_c"]
  assert ds.seed_targets and ds.seeded                                           # seeded like the other validation sets
  assert len(ds) == len(data["C"]["meshes"]) * 4                                 # view numbers < max_source_view_num
  assert ds.inner.volumes.df["view_idx"].to_list() == [m * 6 + v for m in range(2) for v in range(4)]
  ref = gd.extra_dataset("test_c", dm.cfg.data.extra_val_sets.test_c, dm.train_ds, train_gs.x0_encoding(dm.cfg), 14,
                         tempfile.mkdtemp(dir=data["tmp"]))
  cols = ["row", "mesh_id", "view_idx"]
  assert ref.inner.volumes.df.select(cols).equals(ds.inner.volumes.df.select(cols)) and ref.inner.pool == ds.inner.pool
  a, b = ds[3], ds[3]                                                            # paired: the same targets every time
  assert a["target_views"] == b["target_views"] and torch.equal(a["x0_raw"], b["x0_raw"])
  random.seed(1)
  c = ds[3]
  assert c["target_views"] == a["target_views"]
  one = gd.extra_dataset("test_c", dm.cfg.data.extra_val_sets.test_c, dm.train_ds, train_gs.x0_encoding(dm.cfg), 14,
                         tempfile.mkdtemp(dir=data["tmp"]))
  one.seed_targets = True
  assert one[3]["target_views"] == a["target_views"]
  with pytest.raises(SystemExit, match="grids_dir"):
    multi(data, "A", extra=(*SPLIT, extra_override("test_c", {**data["C"], "grids": "null"})))
  with pytest.raises(SystemExit, match="name"):
    multi(data, "A", extra=(*SPLIT, extra_override("val_train", data["C"])))


def test_eval_diffusion_loads_a_multi_shard_run(data):
  """eval_diffusion rebuilds a data.shards run's datasets from its saved config (with data.extra_val_sets, and an eval extra set of
  the same name): the same volumes and normalization as the run had."""
  import torch
  from omegaconf import OmegaConf
  import eval_diffusion as ev
  run_dir = tempfile.mkdtemp(dir=data["tmp"])
  run_cfg = compose(run_dir, [shards_override(data["A"], data["B"]), *SPLIT, extra_override("test_c", data["C"])])
  trained = setup_dm(run_cfg)
  path = os.path.join(run_dir, "config.yaml")
  OmegaConf.save(run_cfg, path)
  out = tempfile.mkdtemp(dir=data["tmp"])
  with __import__("hydra").initialize_config_dir(config_dir=os.path.join(os.path.dirname(HERE), "conf"), version_base=None):
    cfg = __import__("hydra").compose(config_name="eval_diffusion", overrides=[
      f"run_config={path}", f"out={out}", "checkpoints=[init]",
      f"extra_sets={{test_c:{{source_h5:{data['C']['h5']},grids_dir:{data['C']['grids']},max_meshes:null,seed:0,max_source_view_num:4}}}}"])
  run = ev.load_run_config(cfg)
  assert run.data.extra_val_sets is None and run.data.shards == run_cfg.data.shards
  dm = ev.eval_datamodule(run)
  datasets, _ = ev.eval_datasets(cfg, run, dm)
  assert list(datasets) == list(trained.val_datasets)                            # the run's sets, then the eval's test_c
  for n, ds in datasets.items():
    assert ds.inner.volumes.df.select(["mesh_id", "view_idx"]).equals(trained.val_datasets[n].inner.volumes.df.select(["mesh_id", "view_idx"]))
  assert all(torch.equal(a, b) for a, b in zip(dm.diffusion_stats, trained.diffusion_stats))
  assert os.path.exists(os.path.join(out, "split.json")) and any(f.startswith("corpus_") for f in os.listdir(out))
  with __import__("hydra").initialize_config_dir(config_dir=os.path.join(os.path.dirname(HERE), "conf"), version_base=None):
    bad = __import__("hydra").compose(config_name="eval_diffusion", overrides=[
      f"run_config={path}", f"out={out}", "checkpoints=[init]", f"grids_dir={data['A']['grids']}"])
  with pytest.raises(SystemExit, match="shards"):
    ev.load_run_config(bad)


if __name__ == "__main__" and sys.argv[1:] == ["golden"]:
  data_path = _root()
  os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
  digests = golden_digests(fixture_data(data_path))
  with open(GOLDEN, "w") as f:
    json.dump(digests, f, indent=1, sort_keys=True)
  print("wrote", GOLDEN, {k: {s: len(v["volumes"]) for s, v in d.items() if s != "stats"} for k, d in digests.items()})
