"""gs_diffusion.format_projection: puts x0 predictions into the training grids' format (sorted axes + rotation, floor, unit w >= 0
quaternion) -- the same Gaussians as sort_scale_axes gives (covariance unchanged), a fixed point on encoded grids."""
import numpy as np
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
