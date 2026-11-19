"""KID: the estimators (kid_compare: brute-force agreement, symmetry, same distribution ~ 0, the object exclusion), the volume /
view selection (eval_diffusion.kid_selection, extra_dataset, fixed_targets_dataset) and the images (kid_features: white with
each image's own alpha, dgm-eval's PIL resize)."""
import itertools
import os

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image

import eval_diffusion as ev
import kid_compare as kc
import kid_features as kf

NEW100 = "/mnt/hdd/gsviawt_3dgs_archive/new100/trellis_new100_50views_eevee_504.h5"
NEW100_GRIDS = "/mnt/hdd/gsviawt_3dgs_archive/new100/anchored/grids_sparse"
SHARD2 = "/mnt/hdd/gsviawt_3dgs_archive/shard2/trellis_shard2_50views_eevee_504.h5"


def brute(gen, gt, mesh, objex):
  """The estimators as O(n^2) loops over image pairs."""
  k = lambda x, y: (x @ y / len(x) + 1) ** 3
  n = len(gen)
  keep = lambda i, j: (i != j) and not (objex and mesh[i] == mesh[j])
  keep_xy = lambda i, j: not (objex and mesh[i] == mesh[j])
  xx = [k(gen[i], gen[j]) for i, j in itertools.product(range(n), range(n)) if keep(i, j)]
  yy = [k(gt[i], gt[j]) for i, j in itertools.product(range(n), range(n)) if keep(i, j)]
  xy = [k(gen[i], gt[j]) for i, j in itertools.product(range(n), range(n)) if keep_xy(i, j)]
  return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)


def clustered(rng, n_mesh, views, d, shift=0.0, paired=True, mesh_sd=1.0):
  """Features with a per-mesh latent: gen and gt views of mesh o = mu_o + view noise (gen's mu shared with gt if paired)."""
  mesh = np.repeat(np.arange(n_mesh), views)
  mu = rng.normal(0, mesh_sd, (n_mesh, d))
  mu_gen = mu if paired else rng.normal(0, mesh_sd, (n_mesh, d))
  gt = mu[mesh] + rng.normal(0, 1, (len(mesh), d))
  gen = mu_gen[mesh] + rng.normal(0, 1, (len(mesh), d)) + shift
  return gen, gt, mesh


def test_estimators_match_brute_force_and_jackknife():
  rng = np.random.default_rng(0)
  views = [3, 1, 4, 2, 3]                                       # uneven views per mesh
  mesh = np.repeat(np.arange(5), views)
  gen, gt = rng.normal(size=(len(mesh), 6)), rng.normal(size=(len(mesh), 6)) + 0.3
  est = kc.set_estimates(kc.blocks(gen, gt, mesh))
  for name, objex in (("kid_plain", False), ("kid_objex", True)):
    assert np.isclose(est[name][0], brute(gen, gt, mesh, objex), rtol=1e-10)
    for o in range(5):                                            # leave-one-mesh-out values = the estimator without that mesh
      m = mesh != o
      assert np.isclose(est[name][1][o], brute(gen[m], gt[m], mesh[m], objex), rtol=1e-10)
  full, loo = est["kid_objex"]
  _, se = kc.jackknife(full, loo)
  assert np.isclose(se, np.sqrt(4 / 5 * ((loo - loo.mean()) ** 2).sum()))


def test_symmetric_in_gen_and_gt():
  rng = np.random.default_rng(1)
  gen, gt, mesh = clustered(rng, 6, 3, 8, shift=0.5)
  a, b = kc.set_estimates(kc.blocks(gen, gt, mesh)), kc.set_estimates(kc.blocks(gt, gen, mesh))
  for k in kc.ESTIMATORS:
    assert np.isclose(a[k][0], b[k][0]) and np.allclose(a[k][1], b[k][1])


def test_identical_sets():
  """gen == gt (a perfect reconstruction): no same-mesh pair left, so the object-excluded KID is exactly 0; the plain one counts the
  matched k(x, x) in XY but not in XX / YY and goes negative."""
  rng = np.random.default_rng(2)
  _, gt, mesh = clustered(rng, 8, 4, 8)
  est = kc.set_estimates(kc.blocks(gt, gt, mesh))
  assert abs(est["kid_objex"][0]) < 1e-9 and np.abs(est["kid_objex"][1]).max() < 1e-9
  assert est["kid_plain"][0] < -1e-3


def test_same_distribution_is_zero_in_expectation():
  """gen and gt from the same view-marginal, gen conditioned on gt's mesh (the paired, clustered case): the object-excluded KID
  averages to 0, the plain one is biased low (the matched same-mesh XY pairs); an unrelated shift is detected."""
  rng = np.random.default_rng(3)
  vals = {k: [] for k in kc.ESTIMATORS}
  for _ in range(300):
    gen, gt, mesh = clustered(rng, 12, 4, 8)
    for k, (f, _) in kc.set_estimates(kc.blocks(gen, gt, mesh)).items():
      vals[k].append(f)
  ob, pl_ = np.array(vals["kid_objex"]), np.array(vals["kid_plain"])
  assert abs(ob.mean()) < 3 * ob.std() / np.sqrt(len(ob))
  assert pl_.mean() < -5 * pl_.std() / np.sqrt(len(pl_))
  gen, gt, mesh = clustered(rng, 12, 4, 8, shift=1.0)
  assert kc.set_estimates(kc.blocks(gen, gt, mesh))["kid_objex"][0] > 0.1


def test_object_exclusion_ignores_same_mesh_pairs():
  """The object-excluded KID reads no kernel value of two images of the same mesh: garbage there changes the plain one only."""
  rng = np.random.default_rng(4)
  gen, gt, mesh = clustered(rng, 5, 3, 8, shift=0.2)
  b = kc.blocks(gen, gt, mesh)
  same = mesh[:, None] == mesh[None, :]
  junk = lambda K: np.where(same, K + rng.normal(0, 5, K.shape), K)
  kxx, kyy, kxy = kc.kernel(gen, gen), kc.kernel(gt, gt), kc.kernel(gen, gt)
  kxx, kyy = (lambda m: (m + m.T) / 2)(junk(kxx)), (lambda m: (m + m.T) / 2)(junk(kyy))
  kxy = junk(kxy)
  onehot = np.eye(5)[mesh]
  bj = {"A": onehot.T @ kxx @ onehot, "B": onehot.T @ kyy @ onehot, "C": onehot.T @ kxy @ onehot,
        "trA": onehot.T @ np.diag(kxx), "trB": onehot.T @ np.diag(kyy), "n": onehot.sum(0)}
  e, ej = kc.set_estimates(b), kc.set_estimates(bj)
  assert np.isclose(e["kid_objex"][0], ej["kid_objex"][0]) and np.allclose(e["kid_objex"][1], ej["kid_objex"][1])
  assert not np.isclose(e["kid_plain"][0], ej["kid_plain"][0])


def feature_file(gen, gt, mesh, view, sets=None):
  n = len(mesh)
  return {"gen": gen, "gt": gt, "mesh": mesh, "view": view, "set": np.array(sets if sets is not None else ["s"] * n),
          "checkpoint": np.array(["c"] * n), "weights": np.array(["ema"] * n)}


def test_paired_delta():
  rng = np.random.default_rng(5)
  gen, gt, mesh = clustered(rng, 10, 4, 8, shift=0.3)
  view = np.tile(np.arange(4), 10)
  a = feature_file(gen, gt, mesh, view)
  gen_b = gen + rng.normal(0, 0.5, gen.shape)
  perm = rng.permutation(len(mesh))                              # b in another image order: aligned by (set, mesh, view)
  b = feature_file(gen_b[perm], gt[perm], mesh[perm], view[perm])
  r = kc.compare(a, b)[0]
  assert r["n_meshes"] == 10 and r["n_images"] == 40 and r["gt_max_abs_diff"] == 0
  for k in kc.ESTIMATORS:
    assert np.isclose(r[f"delta_{k}"], r[k] - r[f"{k}_b"])
    assert r[f"delta_{k}_se"] > 0
  same = kc.compare(a, a)[0]
  assert all(same[f"delta_{k}"] == 0 and same[f"delta_{k}_se"] == 0 for k in kc.ESTIMATORS)
  with pytest.raises(SystemExit):
    kc.compare(a, feature_file(gen[:-4], gt[:-4], mesh[:-4], view[:-4]))


def test_sets_are_separate_units():
  """Mesh ids repeat across sets (another render file): each set is scored on its own."""
  rng = np.random.default_rng(6)
  g1, t1, m1 = clustered(rng, 4, 2, 8)
  g2, t2, m2 = clustered(rng, 4, 2, 8, shift=1.0)
  f = feature_file(np.concatenate([g1, g2]), np.concatenate([t1, t2]), np.concatenate([m1, m2]), np.tile([0, 1], 8),
                   ["a"] * 8 + ["b"] * 8)
  r = {x["set"]: x for x in kc.compare(f)}
  assert r["a"]["n_meshes"] == r["b"]["n_meshes"] == 4
  assert np.isclose(r["b"]["kid_objex"], 1e3 * kc.set_estimates(kc.blocks(g2, t2, m2))["kid_objex"][0])


# ---- selection ----

class Vols:
  def __init__(self, mesh, view):
    self.df = pl.DataFrame({"mesh_id": mesh, "view_idx": view})


def test_kid_selection():
  rng = np.random.default_rng(7)
  mesh, view, pool = [], [], {}
  for m in range(6):
    vs = (m * 100 + rng.permutation(50)[:9]).tolist()
    pool[m] = vs[:5] if m % 2 else vs[:4]                        # the held-out views (4 or 5)
    srcs = vs[:5] if m % 2 else vs[5:9]                          # sources among them (sval-like) or not (strain-like)
    mesh += [m] * len(srcs)
    view += srcs
  vols = Vols(mesh, view)
  sel = ev.kid_selection(vols, pool, "val_views_sval", 0, 4)
  assert len(sel) == 6 and sorted(int(vols.df["mesh_id"][i]) for i, _ in sel) == list(range(6))   # one volume per mesh
  for i, ts in sel:
    m, src = int(vols.df["mesh_id"][i]), int(vols.df["view_idx"][i])
    assert src not in ts and set(ts) <= set(pool[m]) and ts == sorted(ts)
    assert len(ts) == 4                                          # sval-like 5 - source, or strain-like all 4
  assert sel == ev.kid_selection(vols, pool, "val_views_sval", 0, 4)                 # reproducible
  rev = Vols(mesh[::-1], view[::-1])                             # and a function of the volumes, not their order
  assert {(int(rev.df["view_idx"][i]), tuple(t)) for i, t in ev.kid_selection(rev, pool, "val_views_sval", 0, 4)} == \
         {(int(vols.df["view_idx"][i]), tuple(t)) for i, t in sel}
  assert sel != ev.kid_selection(vols, pool, "val_views_sval", 1, 4)
  assert sel != ev.kid_selection(vols, pool, "val_views_strain", 0, 4)
  three = Vols([0, 0], pool[0][:2])                              # source in a pool of 4: 3 targets left
  assert [len(t) for _, t in ev.kid_selection(three, {0: pool[0]}, "x", 0, 4)] == [3]


def run_split():
  import gs_dataset as g
  import gs_diffusion as gd
  idx, _ = gd.build_grid_index(NEW100_GRIDS, NEW100)
  cat = g.render_catalog(NEW100)
  cat = cat.take([p for p, v in enumerate(cat.df["view_idx"].to_list()) if v in set(idx)])
  tm, _ = g.split_meshes_by_fraction(cat, 0.1, seed=42)
  tv, vv = g.split_views_by_fraction(tm, 0.1, seed=42)
  return tv, vv, g.sample_matching_sources(tv, vv, seed=42)


@pytest.mark.skipif(not (os.path.exists(NEW100) and os.path.isdir(NEW100_GRIDS)), reason="new100 data not here")
def test_kid_selection_on_the_run_split():
  """The run's split (r14: 10% of the 45 fitted views held out = 4 per mesh): sval = 1 held-out source + its 3 other held-out views,
  strain = a train-view source + all 4 held-out views; 90 meshes each, targets always held-out views."""
  tv, vv, strain_src = run_split()
  pool = {m: list(g["view_idx"]) for m, g in vv.df.group_by("mesh_id")}
  pool = {int(m[0] if isinstance(m, tuple) else m): v for m, v in pool.items()}
  held = set(vv.df["view_idx"].to_list())
  sval = ev.kid_selection(vv, pool, "val_views_sval", 0, 4)
  strain = ev.kid_selection(strain_src, pool, "val_views_strain", 0, 4)
  assert len(sval) == len(strain) == 90
  assert all(len(t) == 3 and set(t) <= held for _, t in sval)
  assert all(len(t) == 4 and set(t) <= held for _, t in strain)
  assert all(int(vv.df["view_idx"][i]) in held for i, _ in sval)
  assert not any(int(strain_src.df["view_idx"][i]) in held for i, _ in strain)


class Like:   # the item settings of a run's validation dataset
  num_layers, num_views, allow_source_as_target, depth_size, image_size, read_features = 6, 2, False, 504, 504, False


@pytest.mark.skipif(not os.path.exists(SHARD2), reason="shard2 renders not here")
def test_extra_set_and_fixed_targets(tmp_path):
  from omegaconf import OmegaConf
  spec = OmegaConf.create({"source_h5": SHARD2, "grids_dir": None, "max_meshes": 90, "seed": 0, "max_source_view_num": 45})
  ds = ev.extra_dataset("test_shard2", spec, Like(), None, 14, str(tmp_path))
  vols = ds.volumes.df
  assert vols["mesh_id"].n_unique() == 90 and len(vols) == 90 * 45
  (tmp_path / "again").mkdir()
  ds2 = ev.extra_dataset("test_shard2", spec, Like(), None, 14, str(tmp_path / "again"))
  assert ds2.volumes.df["mesh_id"].unique().sort().to_list() == vols["mesh_id"].unique().sort().to_list()   # reproducible
  import gs_dataset as g
  cat = g.render_catalog(SHARD2)
  num = dict(zip(cat.df["view_idx"].to_list(), cat.df["view_num"].to_list()))
  assert max(num[v] for v in vols["view_idx"].to_list()) == 44          # sources: the fitted views 0-44 only
  sel = ev.kid_selection(ds.volumes, ds.pool, "test_shard2", 0, 4)
  assert len(sel) == 90 and all(len(t) == 4 for _, t in sel)
  assert all(int(vols["view_idx"][i]) not in t and len(set(cat.df["mesh_id"].to_list()[v] for v in [int(vols["view_idx"][i]), *t])) == 1
             for i, t in sel)                                           # targets: other views of the source's mesh
  fixed = ev.fixed_targets_dataset(ds, sel[:3])
  assert len(fixed) == 3
  for j, (i, t) in enumerate(sel[:3]):
    item = fixed[j]
    assert item["source_view"] == int(vols["view_idx"][i]) and sorted(item["target_views"]) == t
    assert item["views"]["rgb"].shape[0] == 1 + len(t)


# ---- images ----

def test_white_composite_own_alpha():
  rgb = torch.tensor([0.5, 0.2, 0.9, 0.0]).view(1, 1, 2, 2).expand(1, 3, 2, 2)
  alpha = torch.tensor([0.5, 1.0, 0.0, 0.0]).view(1, 1, 2, 2)
  straight = kf.white_uint8(rgb, alpha, premultiplied=False)[0, ..., 0]   # the dataset's straight RGBA: rgb a + (1 - a)
  assert straight.tolist() == [[191, 51], [255, 255]]                     # 0.75, 0.2, and white wherever a = 0 (any rgb)
  pre = kf.white_uint8(rgb * alpha, alpha, premultiplied=True)[0, ..., 0]  # a render (premultiplied): rgb + (1 - a), the same picture
  assert pre.tolist() == straight.tolist()
  assert kf.white_uint8(torch.zeros(1, 3, 2, 2), torch.zeros(1, 1, 2, 2), True).min() == 255   # empty render: white


def test_pil_resize_is_dgm_evals():
  rng = np.random.default_rng(8)
  u8 = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
  ref = []   # dgm_eval.resizer.pil_resize
  for c in range(3):
    img = Image.fromarray(u8.astype(np.float32)[:, :, c]).resize((224, 224), resample=Image.BICUBIC)
    ref.append(np.asarray(img).clip(0, 255))
  assert np.array_equal(kf.pil_resize(u8), np.stack(ref).astype(np.float32) / 255)
  assert np.allclose(kf.pil_resize(np.full((50, 50, 3), 255, np.uint8)), 1)
  x = kf.preprocess(u8[None])
  assert x.shape == (1, 3, 224, 224) and x.dtype == torch.float32


def test_gt_cache(tmp_path):
  src = tmp_path / "r.h5"
  src.write_bytes(b"x" * 10)
  c = kf.GTCache(str(tmp_path / "cache"))
  assert c.get(str(src), [3, 4]) is None
  f = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
  c.put(str(src), [3, 4], f)
  assert np.array_equal(c.get(str(src), [3, 4]), f) and c.get(str(src), [3, 5]) is None
  assert kf.GTCache(None).get(str(src), [3]) is None
