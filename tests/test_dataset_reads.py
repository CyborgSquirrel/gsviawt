"""WTFeatureDataset reads each source view once and skips the target views' depth peels (not in the item): its items equal the
ones the old reads built (a frozen copy of __getitem__ as of 154874a), tensor by tensor, on a synthetic render h5."""
import random

import h5py
import numpy as np
import pytest
import torch

import gs_dataset
import gs_diffusion as gd
from gs_dataset import WTFeatureDataset, render_catalog


def old_getitem(self, idx):
  """WTFeatureDataset.__getitem__ as of 154874a: every view read whole (images + depth_peel), the source image read again."""
  vol = self.volumes[idx]
  pool = self.pool[vol["mesh_id"]]
  source_view = vol["view_idx"]
  if self.allow_source_as_target:
    k = min(self.num_views, len(pool))
    targets = random.sample(pool, k) if k > 0 else []
  else:
    in_pool = source_view in self.pool_sets[vol["mesh_id"]]
    k = min(self.num_views, len(pool) - in_pool)
    if k <= 0:
      targets = []
    else:
      sampled = random.sample(pool, min(k + 1, len(pool)))
      targets = [v for v in sampled if v != source_view][:k]
  f = gs_dataset._get_h5(self.source_h5)
  read = lambda v: gs_dataset._decode_view(
    self.source_h5, v, f["images"][v], f["depth_peel"][v], f[gs_dataset.intrinsics_name(f, "depth")][v],
    f[gs_dataset.intrinsics_name(f, "image")][v], f["camera_pose"][v])
  src = read(source_view)
  if self.depth_size is not None and src["depth_peel"].shape[:2] != (self.depth_size, self.depth_size):
    src["depth_peel"], src["K_depth"] = gs_dataset.resize_depth_nearest(src["depth_peel"], src["K_depth"], self.depth_size)
  out = gs_dataset._assemble_item(src, [read(v) for v in targets], self.num_layers, vol["mesh_id"], source_view, targets)
  if self.image_size is not None:
    from wt.data import preprocess_rgba_for_model
    from wt_infer_layers import rgba_from_render
    rgb, _, _ = preprocess_rgba_for_model(
      rgba_from_render(gs_dataset._get_h5(self.source_h5), source_view), image_size=self.image_size,
      num_layers=1, center_crop=False, bg_color=(0, 0, 0))
    out["image"] = rgb[0]
    if not self.read_features:
      return out
  feats = gs_dataset._get_h5(self.features_h5)["features"][vol["row"]]
  out["features"] = torch.from_numpy(feats).float().contiguous()
  return out


def render_h5(path, channels, meshes=2, views=5, size=16, layers=6, seed=0):
  """A render_objaverse.py-schema h5: RGBA (channels 4) or RGB-only images, depth peels NaN where there is no hit."""
  rng = np.random.default_rng(seed)
  n = meshes * views
  depth = np.sort(rng.uniform(1, 3, (n, size, size, layers)), -1).astype(np.float32)
  depth[rng.random(depth.shape) < 0.4] = np.nan
  depth = np.sort(depth, -1)                                          # the hits first, NaN after (as rendered)
  K = np.tile(np.array([[20.0, 0, size / 2], [0, 20.0, size / 2], [0, 0, 1]], np.float32), (n, 1, 1))
  pose = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
  pose[:, :3, 3] = rng.normal(size=(n, 3))
  with h5py.File(path, "w") as f:
    f["images"] = rng.integers(0, 256, (n, size, size, channels), dtype=np.uint8)
    f["depth_peel"] = depth
    f["camera_pose"] = pose
    f["depth_intrinsics"] = K
    f["image_intrinsics"] = K
    f["mesh_index"] = np.repeat(np.arange(meshes), views)
    f["mesh_paths"] = [f"mesh{m}.glb".encode() for m in range(meshes)]
  return str(path)


def same(a, b, key=""):
  assert type(a) is type(b), key
  if isinstance(a, dict):
    assert a.keys() == b.keys(), key
    for k in a:
      same(a[k], b[k], f"{key}/{k}")
  elif isinstance(a, torch.Tensor):
    assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.nan_to_num(7.0), b.nan_to_num(7.0)), key
    assert torch.equal(a.isnan(), b.isnan()), key
  elif isinstance(a, (list, tuple)):
    assert len(a) == len(b), key
    for i, (x, y) in enumerate(zip(a, b)):
      same(x, y, f"{key}[{i}]")
  else:
    assert a == b, key


@pytest.fixture
def count_reads(monkeypatch):
  """{dataset name: number of h5 reads} while the test runs."""
  counts = {}
  orig = h5py.Dataset.__getitem__

  def getitem(self, key):
    name = self.name.split("/")[-1]
    counts[name] = counts.get(name, 0) + 1
    return orig(self, key)
  monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
  return counts


@pytest.mark.parametrize("channels", [4, 3])
@pytest.mark.parametrize("kw", [dict(image_size=16, read_features=False), dict(image_size=16), dict(),
                                dict(image_size=16, depth_size=8)], ids=["image", "image+features", "features", "resized"])
def test_items_unchanged(tmp_path, channels, kw, count_reads):
  src = render_h5(tmp_path / "render.h5", channels)
  views = [0, 3, 6, 9]                                                # source views of both meshes
  listing = gd.make_listing({v: None for v in views}, src, str(tmp_path / "listing.h5"))
  ds = WTFeatureDataset(listing, render_catalog(src), num_layers=6, num_views=2, **kw)
  for i in range(len(ds)):
    random.seed(i)
    a = old_getitem(ds, i)
    random.seed(i)
    count_reads.clear()
    b = ds[i]
    same(a, b)
    assert len(b["target_views"]) == 2
    again = channels == 3 and "image_size" in kw                      # RGB-only: rgba_from_render reads the view again
    assert count_reads["images"] == 3 + again                         # (RGBA) the source once, two targets
    assert count_reads["depth_peel"] == (1 if channels == 4 else 3) + again   # RGB-only: the targets' alpha is their hit mask
