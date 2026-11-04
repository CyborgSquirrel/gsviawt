"""gs_diffusion core: encode_grid / decode_x0 on a synthetic grid, EMAWeights, SpikeSkip."""
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import gs_diffusion as gd
from gs_dataset import OPENGL_TO_OPENCV


def synthetic_grid(h=4, w=5, layers=6, seed=0):
  """A grid file's dict + the matching dataset `source` dict: random valid slots, means = anchors + small offsets (world frame)."""
  rng = np.random.default_rng(seed)
  hit = rng.random((layers, h, w)) < 0.6
  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = Rotation.from_euler("xyz", rng.normal(size=3)).as_matrix()
  pose[:3, 3] = rng.normal(size=3)
  c2w = pose @ OPENGL_TO_OPENCV
  xyz_cam = rng.normal(size=(layers, h, w, 3)).astype(np.float32) + np.array([0, 0, 3], np.float32)
  offset = rng.normal(scale=0.01, size=(layers, h, w, 3)).astype(np.float32)
  cam = xyz_cam + offset
  world = cam @ c2w[:3, :3].T + c2w[:3, 3]
  q = Rotation.random(layers * h * w, random_state=seed).as_quat()[:, [3, 0, 1, 2]].reshape(layers, h, w, 4)
  t = lambda a: np.moveaxis(a, 0, 2)   # (L, H, W, ...) -> the file's (H, W, L, ...)
  grid = {"layer_valid": t(hit), "gaussian_means": t(world.astype(np.float32)), "gaussian_quats": t(q.astype(np.float32)),
          "gaussian_scales": t(np.exp(rng.normal(-5, 1, (layers, h, w, 3))).astype(np.float32)),
          "gaussian_opacities": t(rng.uniform(0.05, 0.95, (layers, h, w)).astype(np.float32)),
          "gaussian_colors": t(rng.uniform(0, 1, (layers, h, w, 3)).astype(np.float32))}
  source = {"hit": torch.from_numpy(hit), "xyz_cam": torch.from_numpy(np.where(hit[..., None], xyz_cam, np.nan)),
            "pose_gl": torch.from_numpy(pose), "K_depth": torch.tensor([[100.0, 0, 2], [0, 100.0, 2], [0, 0, 1]])}
  return grid, source, offset, c2w


def test_encode_decode_roundtrip():
  grid, source, offset, c2w = synthetic_grid()
  x0 = gd.encode_grid(grid, source)                                  # (C, L, H, W) raw
  hit = source["hit"].numpy()
  assert x0.shape == (gd.CHANNELS,) + hit.shape and np.isfinite(x0).all()
  assert (x0[:, ~hit] == 0).all()
  np.testing.assert_allclose(x0[11:14].transpose(1, 2, 3, 0)[hit], offset[hit], atol=1e-4)   # offset = fit mean - anchor
  q = x0[4:8].transpose(1, 2, 3, 0)[hit]
  assert (q[:, 0] >= 0).all() and np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
  # decode with mean 0 / std 1: recovers the grid's opacity / scale / color, and the rotation in the camera frame
  out = gd.decode_x0(torch.from_numpy(x0)[None].permute(0, 2, 1, 3, 4), torch.zeros(gd.CHANNELS), torch.ones(gd.CHANNELS))
  L_H_W = lambda a: np.moveaxis(a, 2, 0)
  np.testing.assert_allclose(out["opacity"][0, :, 0].numpy()[hit], L_H_W(grid["gaussian_opacities"])[hit], rtol=1e-4)
  np.testing.assert_allclose(out["scale"][0].permute(0, 2, 3, 1).numpy()[hit], L_H_W(grid["gaussian_scales"])[hit], rtol=1e-4)
  rgb = out["sh_dc"][0].permute(0, 2, 3, 1).numpy()[hit] * gd.SH_C0 + 0.5
  np.testing.assert_allclose(rgb, L_H_W(grid["gaussian_colors"])[hit], atol=1e-5)
  r_world = Rotation.from_quat(L_H_W(grid["gaussian_quats"])[hit][:, [1, 2, 3, 0]]).as_matrix()
  r_cam = Rotation.from_quat(out["rotation"][0].permute(0, 2, 3, 1).numpy()[hit][:, [1, 2, 3, 0]]).as_matrix()
  np.testing.assert_allclose(c2w[:3, :3] @ r_cam, r_world, atol=1e-4)


def test_encode_rejects_mismatched_hit_mask():
  grid, source, _, _ = synthetic_grid()
  source["hit"] = source["hit"].clone()
  source["hit"][0, 0, 0] = ~source["hit"][0, 0, 0]
  with pytest.raises(ValueError):
    gd.encode_grid(grid, source)


def test_ema_update_apply_restore_and_state():
  m = torch.nn.Linear(3, 2)
  torch.nn.init.zeros_(m.weight)
  torch.nn.init.zeros_(m.bias)
  ema = gd.EMAWeights(m, decay=0.5)
  with torch.no_grad():
    m.weight.fill_(1.0)
  ema.update(m)                                                      # d = min(0.5, 2/11) = 2/11
  d = 2 / 11
  assert torch.allclose(ema.shadow["weight"], torch.full((2, 3), 1 - d))
  live = m.weight.detach().clone()
  with ema.applied(m):
    assert torch.allclose(m.weight, torch.full((2, 3), 1 - d))
  assert torch.equal(m.weight, live) and ema.stash == {}
  ema2 = gd.EMAWeights(torch.nn.Linear(3, 2))
  ema2.load_state_dict(ema.state_dict())
  assert ema2.updates == 1 and torch.allclose(ema2.shadow["weight"], ema.shadow["weight"])


def test_spike_skip():
  s = gd.SpikeSkip(factor=4.0, decay=0.5, warmup=2)
  assert s.check(1.0) is False and s.check(1.0) is False             # warmup: accepted
  assert s.check(10.0) is True                                        # > 4 x average: skipped, not averaged
  assert s.average == 1.0 and s.accepted == 2
  assert s.check(float("nan")) is True and s.nonfinite == 1
  assert s.check(2.0) is False and math.isclose(s.average, 1.5)
  off = gd.SpikeSkip(factor=None)
  assert off.check(1e9) is False and off.check(float("inf")) is False and off.nonfinite == 1
