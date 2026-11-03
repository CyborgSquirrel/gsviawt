"""diffusion.augment (gs_diffusion.augment_x0): render-invariant -- every Gaussian (world mean = anchor + offset, covariance, opacity,
colour) is preserved, only how it is written down changes. The block shuffle moves Gaussians within shifted
n x n blocks of a layer, re-anchoring offsets, not across depth edges."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import gs_diffusion as gd

L, H, W, FX = 2, 12, 12, 100.0


def scene(seed=0, hole=True):
  rng = np.random.default_rng(seed)
  hit = np.ones((L, H, W), bool)
  if hole:
    hit[0, 3:5, 6:9] = False
  ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
  z = np.stack([np.full((H, W), 2.0), np.full((H, W), 3.0)])         # flat layers: every block passes the depth guard
  xyz = np.stack([(xs - W / 2)[None] * z / FX, (ys - H / 2)[None] * z / FX, z], -1).astype(np.float32)   # (L, H, W, 3)
  xyz[~hit] = np.nan
  x0 = np.zeros((gd.CHANNELS, L, H, W), np.float32)
  n = int(hit.sum())
  q = Rotation.random(n, random_state=seed).as_quat()[:, [3, 0, 1, 2]]
  vals = np.concatenate([rng.normal(size=(n, 1)), rng.normal(-5, 1, (n, 3)), np.where(q[:, :1] < 0, -q, q),
                         rng.normal(size=(n, 3)), rng.normal(0, 0.01, (n, 3))], 1).astype(np.float32)
  x0[:, hit] = vals.T
  source = {"hit": torch.from_numpy(hit), "xyz_cam": torch.from_numpy(xyz), "K_depth": np.diag([FX, FX, 1.0])}
  return x0, source


def gaussians(x0, source):
  """(N, 1 + 9 + 3 + 3) rows per valid slot: opacity, covariance, colour, world mean -- sorted, so slot order doesn't matter."""
  hit = source["hit"].numpy()
  v = x0[:, hit].T
  s, q = np.exp(v[:, gd.SCALE]), v[:, gd.ROTATION]
  R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
  cov = (R * s[:, None, :] ** 2) @ R.transpose(0, 2, 1)
  mean = source["xyz_cam"].numpy()[hit] + v[:, gd.OFFSET]
  rows = np.concatenate([v[:, :1], cov.reshape(-1, 9), v[:, 8:11], mean], 1).astype(np.float64)
  return rows[np.lexsort(rows.T[::-1])]


def test_off_is_identity():
  x0, src = scene()
  np.testing.assert_array_equal(gd.augment_x0(x0, src, np.random.default_rng(0)), x0)


def test_block_shuffle_preserves_gaussians_and_moves_within_blocks():
  x0, src = scene()
  for seed in range(5):
    for n in (2, 3):
      out = gd.augment_x0(x0, src, np.random.default_rng(seed), slot_shuffle=True, slot_shuffle_block=n)
      np.testing.assert_allclose(gaussians(out, src), gaussians(x0, src), atol=1e-5)
      hit = src["hit"].numpy()
      assert (out[:, ~hit] == 0).all()                               # nothing moved into an empty slot
      moved = ~np.isclose(out[0], x0[0]) & hit                       # opacity identifies a Gaussian
      assert moved.mean() > 0.3
      for l, y, x in zip(*np.nonzero(moved)):                        # each one came from within n - 1 slots
        ly, lx = np.nonzero(np.isclose(x0[0, l], out[0, l, y, x]))
        assert (np.abs(ly - y) <= n - 1).any() and (np.abs(lx - x) <= n - 1).any()


def test_block_shift_covers_all_offsets():
  """With n = 2, slots 1 and 2 of a row are in one block only when the tiling is shifted by 1: over many items they do trade."""
  x0, src = scene(hole=False)
  traded = 0
  for seed in range(40):
    out = gd.augment_x0(x0, src, np.random.default_rng(seed), slot_shuffle=True, slot_shuffle_block=2)
    traded += np.isclose(out[0, 0, 0, 2], x0[0, 0, 0, 1])
  assert 0 < traded < 40


def test_depth_guard_and_probability():
  x0, src = scene(hole=False)
  xyz = src["xyz_cam"].numpy().copy()
  xyz[0, :, :6, 2] = 20.0                                            # a depth edge down the middle of layer 0
  xyz[0, :, :6, :2] *= 10.0
  src = {**src, "xyz_cam": torch.from_numpy(xyz)}
  for seed in range(10):
    out = gd.augment_x0(x0, src, np.random.default_rng(seed), slot_shuffle=True, slot_shuffle_block=2)
    np.testing.assert_allclose(gaussians(out, src), gaussians(x0, src), atol=1e-5)
    for y in range(H):                                               # nothing crosses the edge between columns 5 and 6
      left = set(np.round(out[0, 0, y, :6], 5)) - set(np.round(x0[0, 0, :, :6].ravel(), 5))
      assert not left


def test_dataset_augments_only_when_on():
  class Inner:
    volumes = type("V", (), {"df": {"view_idx": type("C", (), {"to_list": staticmethod(lambda: [3])})}})()
  defaults = {"slot_shuffle": False, "slot_shuffle_block": 2, "slot_shuffle_max_px": 3.0}
  assert gd.GridTargetDataset(Inner(), {3: "x.h5"}, augment=defaults).augment is None
  assert gd.GridTargetDataset(Inner(), {3: "x.h5"}).augment is None
  assert gd.GridTargetDataset(Inner(), {3: "x.h5"}, augment={**defaults, "slot_shuffle": True}).augment["slot_shuffle_block"] == 2
