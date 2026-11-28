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


if __name__ == "__main__" and sys.argv[1:] == ["golden"]:
  data_path = _root()
  os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
  digests = golden_digests(fixture_data(data_path))
  with open(GOLDEN, "w") as f:
    json.dump(digests, f, indent=1, sort_keys=True)
  print("wrote", GOLDEN, {k: {s: len(v["volumes"]) for s, v in d.items() if s != "stats"} for k, d in digests.items()})
