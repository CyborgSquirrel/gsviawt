"""gs_diffusion.format_projection: puts x0 predictions into the training grids' format (sorted axes + rotation, floor, unit w >= 0
quaternion) -- the same Gaussians as sort_scale_axes gives (covariance unchanged), a fixed point on encoded grids."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import gs_diffusion as gd

SC, RO = slice(1, 4), slice(4, 8)   # the scale and rotation channels


def cov(s, q):
  R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
  return (R * np.exp(s)[:, None, :] ** 2) @ R.transpose(0, 2, 1)


def test_quaternion_roundtrip():
  q = Rotation.random(500, random_state=0).as_quat()[:, [3, 0, 1, 2]]
  q = np.where(q[:, :1] < 0, -q, q)
  R = gd.quat_to_matrix(torch.tensor(q))
  np.testing.assert_allclose(R.numpy(), Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(), atol=1e-6)
  np.testing.assert_allclose(gd.matrix_to_quat(R).numpy(), q, atol=1e-6)


def test_projection_sorts_floors_and_keeps_gaussians():
  rng = np.random.default_rng(0)
  B, L, H, W, C = 2, 2, 3, 4, gd.CHANNELS
  mean, std = torch.tensor(rng.normal(size=C)), torch.tensor(rng.uniform(0.5, 2, C))
  hit = torch.tensor(rng.random((1, L, H, W)) > 0.3)
  anchor_z = torch.full((L, H, W), 2.0, dtype=torch.float64)
  x = torch.tensor(rng.normal(size=(B, L, C, H, W)))
  fx = 100.0
  proj = gd.format_projection(mean, std, hit, anchor_z, fx, scale_sort=True, scale_floor_px=None)
  out = proj(x)
  raw = lambda z: (z * std[None, None, :, None, None] + mean[None, None, :, None, None]).permute(0, 1, 3, 4, 2)
  v = hit.expand(B, -1, -1, -1)
  r0, r1 = raw(x)[v].numpy(), raw(out)[v].numpy()
  s0, q0 = r0[:, SC], r0[:, RO] / np.linalg.norm(r0[:, RO], axis=1, keepdims=True)
  s1, q1 = r1[:, SC], r1[:, RO]
  assert (np.diff(s1, axis=1) <= 1e-9).all() and (q1[:, 0] >= 0).all()
  np.testing.assert_allclose(cov(s1, q1), cov(s0, q0), rtol=1e-5, atol=1e-9)        # the same Gaussians
  s_ref, q_ref = gd.sort_scale_axes(np.exp(s0), q0)
  np.testing.assert_allclose(np.exp(s1), s_ref, rtol=1e-6)
  np.testing.assert_allclose(raw(out)[~v].numpy(), raw(x)[~v].numpy())                # invalid slots untouched
  np.testing.assert_allclose(proj(out).numpy(), out.numpy(), atol=1e-5)              # a fixed point (float32 inside)
  floored = gd.format_projection(mean, std, hit, anchor_z, fx, scale_sort=True, scale_floor_px=0.1)(x)
  depth = 2.0 + raw(x)[v].numpy()[:, 13]                              # the floor is at the mean's depth: anchor + offset z
  assert np.isfinite(raw(floored).numpy()).all()
  assert (raw(floored)[v].numpy()[:, SC] >= np.log(0.1 * np.maximum(depth, 1e-3) / fx)[:, None] - 1e-5).all()


def test_sampler_hook_is_called_every_step():
  calls = []
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.head = lambda feats, x, t: x * 0.5
  m._feats = lambda image: None
  m.decode = lambda x: x
  m.eval_seed, m.steps = 0, 4
  m.schedule = type("S", (), {"sample": staticmethod(lambda b, phase, generator=None: torch.rand(b, 2, generator=generator)), "fixed_t": None})()
  hit = torch.ones(1, 2, 2, 2, dtype=torch.bool)
  out = m.sample(torch.zeros(1, 3, 4, 4), hit, seeds=[1], project=lambda z: calls.append(1) or z * 0)
  assert len(calls) == 4 and torch.all(out == 0)                     # the last step's x0 prediction is the output


def test_projection_takes_the_new_encoding_keys():
  rng = np.random.default_rng(1)
  B, L, H, W, C = 1, 2, 3, 3, gd.CHANNELS
  mean, std = torch.zeros(C, dtype=torch.float64), torch.ones(C, dtype=torch.float64)
  hit = torch.ones(1, L, H, W, dtype=torch.bool)
  x = torch.tensor(rng.normal(size=(B, L, C, H, W)))
  az = torch.full((L, H, W), 2.0, dtype=torch.float64)
  old = gd.format_projection(mean, std, hit, az, 100.0, scale_sort=True, scale_floor_px=0.1)(x)
  new = gd.format_projection(mean, std, hit, az, 100.0, scale_order="sorted", scale_floor_px=0.1, rot_form="keep")(x)
  np.testing.assert_allclose(new.numpy(), old.numpy())
  can = gd.format_projection(mean, std, hit, az, 100.0, scale_order="sorted", scale_floor_px=0.1, rot_form="canonical")(x)
  q = can[0].permute(0, 2, 3, 1)[hit[0]][:, RO].numpy()
  assert (np.abs(q).argmax(1) == 0).all()                            # w the largest component
  s = can[0].permute(0, 2, 3, 1)[hit[0]][:, SC].numpy()
  np.testing.assert_allclose(cov(s, q), cov(old[0].permute(0, 2, 3, 1)[hit[0]][:, SC].numpy(),
                                            old[0].permute(0, 2, 3, 1)[hit[0]][:, RO].numpy()), rtol=1e-4, atol=1e-8)


@pytest.mark.cuda
def test_projection_is_a_fixed_point_with_tied_axes_on_cuda():
  """The floor ties axes; an unstable sort would swap them (another quaternion form) on every projection."""
  rng = np.random.default_rng(2)
  B, L, H, W, C = 1, 2, 8, 8, gd.CHANNELS
  dev = torch.device("cuda")
  mean, std = torch.zeros(C, dtype=torch.float64, device=dev), torch.ones(C, dtype=torch.float64, device=dev)
  hit = torch.ones(1, L, H, W, dtype=torch.bool, device=dev)
  x = torch.tensor(rng.normal(size=(B, L, C, H, W)), device=dev)
  x[:, :, 2] = x[:, :, 3] = -5.0                                     # two tied (small) axes everywhere
  x[:, :, 1] = -1.0
  az = torch.full((L, H, W), 2.0, dtype=torch.float64, device=dev)
  proj = gd.format_projection(mean, std, hit, az, 100.0, scale_sort=True)
  once = proj(x)
  torch.testing.assert_close(proj(once), once, atol=1e-5, rtol=1e-5)   # float32 round trip; a tie swap would move rotations by ~1
