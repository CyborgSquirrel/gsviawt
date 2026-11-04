"""encode_grid (valid slots only, scattered once) against a frozen copy of the dense implementation it replaced: bit-exact for
every encoding option, the same rng draws (seeded validation items unchanged, PR #117), empty and fully valid grids."""
import itertools

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import gs_diffusion as gd
from gs_dataset import OPENGL_TO_OPENCV


def encode_grid_dense(grid, source, scale_order=None, scale_floor_px=None, rot_form="keep", rng=None, scale_sort=None,
                scale_floor_mode="max"):
  """gs_diffusion.encode_grid as of 154874a (dense over all L*H*W slots), frozen as the reference."""
  # scale_order None: keep, or the legacy flag's choice; both given must agree
  e = gd.normalize_encoding({**({"scale_order": scale_order} if scale_order is not None else {}), "scale_floor_px": scale_floor_px,
                             "scale_floor_mode": scale_floor_mode, "rot_form": rot_form, **({"scale_sort": scale_sort} if scale_sort is not None else {})})
  scale_order, rot_form = e["scale_order"], e["rot_form"]
  if "random" in (scale_order, rot_form) and rng is None:
    rng = np.random.default_rng()
  valid = grid["layer_valid"].transpose(2, 0, 1)                      # (L, H, W)
  hit = source["hit"].numpy()
  mismatch = int((valid != hit).sum())
  if mismatch:   # one slot is enough: a Gaussian without a point (NaN offset) would poison the loss
    raise ValueError(
      f"the grid's layer_valid disagrees with the point cloud's hit mask on {mismatch} of {valid.size} slots: "
      "is this grid from this render / this view?")
  c2w = source["pose_gl"].numpy() @ OPENGL_TO_OPENCV
  w2c = np.linalg.inv(c2w)
  means = grid["gaussian_means"].transpose(2, 0, 1, 3)               # (L, H, W, 3), world frame
  means_cam = means @ w2c[:3, :3].T + w2c[:3, 3]
  offset = means_cam - source["xyz_cam"].numpy()

  quat = np.zeros(valid.shape + (4,), np.float32)
  scales = grid["gaussian_scales"].transpose(2, 0, 1, 3).astype(np.float32)
  floor = None
  if scale_floor_px is not None:
    footprint = means_cam[..., 2] / float(source["K_depth"][0, 0])                 # (L, H, W): one pixel's size at that depth
    floor = float(scale_floor_px) * np.nan_to_num(footprint)[..., None]
    if scale_floor_mode == "max":
      scales = np.maximum(scales, floor)
  if valid.any():
    q = gd.rotate_quats_wxyz(grid["gaussian_quats"].transpose(2, 0, 1, 3)[valid], c2w[:3, :3].T)   # world -> source camera
    if scale_order == "sorted":
      scales[valid], q = gd.sort_scale_axes(scales[valid], q)
    elif scale_order == "random":
      scales[valid], q = gd.permute_scale_axes(scales[valid], q, np.argsort(rng.random((len(q), 3)), axis=1))
    q = np.where(q[:, :1] < 0, -q, q)                                # q and -q are one rotation: w >= 0
    quat[valid] = gd.set_rot_form(q, rot_form, rng)

  opacity = np.clip(grid["gaussian_opacities"].transpose(2, 0, 1), 1e-6, 1 - 1e-6)
  opacity = np.clip(np.log(opacity / (1 - opacity)), -gd.OPACITY_LOGIT_CLAMP, gd.OPACITY_LOGIT_CLAMP)
  if floor is not None and scale_floor_mode == "add":   # (after the sort: the excess has the axes' order, the floor is per Gaussian)
    scales = np.maximum(scales - floor, gd.FLOOR_ADD_EPS * floor)
  scale = np.log(np.maximum(scales, 1e-8))
  color = (grid["gaussian_colors"].transpose(2, 0, 1, 3) - 0.5) / gd.SH_C0

  x0 = np.concatenate([opacity[..., None], scale, quat, color, offset], axis=-1)   # (L, H, W, CHANNELS)
  x0 = np.where(valid[..., None], x0, 0.0).astype(np.float32)
  return np.ascontiguousarray(x0.transpose(3, 0, 1, 2))              # (CHANNELS, L, H, W)


def synthetic(h, w, layers=6, frac=0.3, n=None, seed=0):
  """A grid + its item's source like the real ones: NaN at the slots without a Gaussian, a random camera pose. n: exactly n valid
  slots instead of a fraction frac."""
  rng = np.random.default_rng(seed)
  hit = rng.random((layers, h, w)) < frac
  if n is not None:
    hit = (np.arange(hit.size) < n)[rng.permutation(hit.size)].reshape(hit.shape)
  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = Rotation.random(random_state=seed).as_matrix()
  pose[:3, 3] = rng.normal(size=3)
  c2w = pose @ OPENGL_TO_OPENCV
  xyz_cam = (rng.normal(size=(layers, h, w, 3)) + [0, 0, 3]).astype(np.float32)
  means_cam = xyz_cam + rng.normal(0, 1e-3, xyz_cam.shape).astype(np.float32)
  world = means_cam @ c2w[:3, :3].T + c2w[:3, 3]
  nan = lambda a: np.where(hit.reshape(hit.shape + (1,) * (a.ndim - 3)), a, np.nan).astype(np.float32)
  t = lambda a: np.ascontiguousarray(np.moveaxis(nan(a), 0, 2))   # (L, H, W, .) -> the grid file's (H, W, L, .)
  grid = {"layer_valid": np.ascontiguousarray(np.moveaxis(hit, 0, 2)), "gaussian_means": t(world),
          "gaussian_quats": t(Rotation.random(layers * h * w, random_state=seed).as_quat().reshape(layers, h, w, 4)),
          "gaussian_scales": t(np.exp(rng.normal(-7, 2, (layers, h, w, 3)))),
          "gaussian_opacities": t(rng.uniform(0, 1, (layers, h, w))),
          "gaussian_colors": t(rng.uniform(0, 1, (layers, h, w, 3)))}
  source = {"hit": torch.from_numpy(hit), "xyz_cam": torch.from_numpy(nan(xyz_cam)), "pose_gl": torch.from_numpy(pose),
            "K_depth": torch.tensor([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])}
  return grid, source


OPTIONS = [dict(scale_order=o, rot_form=r, scale_floor_px=f, scale_floor_mode=m)
           for o, r, f, m in itertools.product(("keep", "sorted", "random"), ("keep", "canonical", "random"), (None, 0.1),
                                               ("max", "add"))
           if not (f is None and m == "add")]
GRIDS = {"empty": dict(h=7, w=5, frac=0.0), "full": dict(h=7, w=5, frac=1.0), "n1": dict(h=9, w=8, n=1),
         "n2": dict(h=9, w=8, n=2), "n3": dict(h=9, w=8, n=3), "sparse": dict(h=13, w=11, frac=0.3),
         "wide": dict(h=64, w=48, frac=0.06), "dense": dict(h=40, w=56, frac=0.9)}   # (W >= 2: a real grid's width)


@pytest.mark.parametrize("grid_name", GRIDS)
@pytest.mark.parametrize("enc", OPTIONS, ids=lambda e: "-".join(str(v) for v in e.values()))
def test_bit_exact_vs_dense(enc, grid_name):
  k = list(GRIDS).index(grid_name)
  grid, source = synthetic(**GRIDS[grid_name], seed=k)
  ra, rb = np.random.default_rng(k), np.random.default_rng(k)
  a = encode_grid_dense(grid, source, rng=ra, **enc)
  b = gd.encode_grid(grid, source, rng=rb, **enc)
  assert b.dtype == np.float32 and b.shape == a.shape and b.flags.c_contiguous
  assert np.array_equal(a.view(np.uint32), b.view(np.uint32))   # bit for bit, NaNs and signed zeros included
  assert ra.random() == rb.random()                             # the same draws: the stream is where it was


def test_bit_exact_legacy_and_nan():
  grid, source = synthetic(9, 8, frac=0.5, seed=3)
  valid = grid["layer_valid"]
  y, x, l = np.argwhere(valid)[0]
  grid["gaussian_colors"][y, x, l, 1] = np.nan                       # a NaN at a valid slot passes through
  grid["gaussian_scales"][y, x, l, 0] = 0.0                          # an axis under the 1e-8 log clamp
  for kw in (dict(), dict(scale_sort=True), dict(scale_sort=False, scale_floor_px=0.1)):
    a, b = encode_grid_dense(grid, source, **kw), gd.encode_grid(grid, source, **kw)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
  assert np.isnan(b[9, l, y, x])


def test_hit_mismatch_raises():
  grid, source = synthetic(6, 6, frac=0.5, seed=4)
  grid["layer_valid"][0, 0, 0] = ~grid["layer_valid"][0, 0, 0]
  with pytest.raises(ValueError, match="disagrees"):
    gd.encode_grid(grid, source)


@pytest.mark.data
def test_bit_exact_on_real_grids(data_dir, tmp_path):
  """The local test grids (bla/grids_test) with their render items, every encoding, seeded as the validation sets are."""
  from gs_dataset import WTFeatureDataset, render_catalog
  idx, src = gd.build_grid_index(data_dir("grids_test", "fit", "grids"))
  data_dir(src)
  cat = render_catalog(src)
  cat = cat.take([p for p, v in enumerate(cat.df["view_idx"].to_list()) if v in set(idx)])
  ds = WTFeatureDataset(gd.make_listing(idx, src, str(tmp_path / "listing.h5")), cat, num_layers=6, num_views=0,
                        read_features=False)
  for i in range(len(ds)):
    item = ds[i]
    grid, _ = gd.load_grid(idx[item["source_view"]])
    for enc in OPTIONS:
      ra, rb = (np.random.default_rng(item["source_view"]) for _ in range(2))
      a = encode_grid_dense(grid, item["source"], rng=ra, **enc)
      b = gd.encode_grid(grid, item["source"], rng=rb, **enc)
      assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), enc
      assert ra.random() == rb.random()
