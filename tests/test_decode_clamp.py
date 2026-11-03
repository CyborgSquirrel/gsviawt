"""gs_diffusion.decode_x0's clamp (diffusion.decode_clamp): channel = every channel into its own training range; scale = only the
scales, all 3 axes into one shared range (so an axis in the "wrong" channel of a sorted encoding is not cut), the rest unclamped."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import gs_diffusion as gd


def make(seed=0):
  g = torch.Generator().manual_seed(seed)
  C = gd.CHANNELS
  mean, std = torch.randn(C, generator=g), torch.rand(C, generator=g) + 0.5
  lo, hi = mean - 1.0, mean + 1.0
  lo[gd.SCALE], hi[gd.SCALE] = torch.tensor([-2.0, -4.0, -8.0]), torch.tensor([3.0, 0.0, -2.0])   # sorted axes: narrow, lopsided
  x0 = 5 * torch.randn(2, 3, C, 4, 4, generator=g)   # far outside every range
  return x0, mean, std, lo, hi


def test_channel_clamps_every_channel():
  x0, mean, std, lo, hi = make()
  raw = x0 * std[None, None, :, None, None] + mean[None, None, :, None, None]
  ref = torch.maximum(torch.minimum(raw, hi[None, None, :, None, None]), lo[None, None, :, None, None])
  out = gd.decode_x0(x0, mean, std, lo, hi)   # the default: as before
  assert torch.equal(out["raw_scale"], ref[:, :, gd.SCALE])
  assert torch.equal(out["sh_dc"], ref[:, :, 8:11])
  assert torch.equal(out["offset"], ref[:, :, 11:14])


def test_scale_clamps_scales_to_shared_range_only():
  x0, mean, std, lo, hi = make()
  x0.requires_grad_(True)
  raw = x0 * std[None, None, :, None, None] + mean[None, None, :, None, None]
  out = gd.decode_x0(x0, mean, std, lo, hi, clamp="scale")
  assert torch.equal(out["raw_scale"], raw[:, :, gd.SCALE].clamp(-8.0, 3.0))   # every axis: [min lo, max hi] over the axes
  assert torch.equal(out["sh_dc"], raw[:, :, 8:11])
  assert torch.equal(out["offset"], raw[:, :, 11:14])
  assert torch.allclose(out["opacity"], torch.sigmoid(raw[:, :, :1]))
  assert torch.allclose(out["rotation"], torch.nn.functional.normalize(raw[:, :, 4:8] + 1e-6, dim=2))
  sum(v.sum() for k, v in out.items() if k != "raw_scale").backward()
  inside = (raw[:, :, gd.SCALE] > -8.0) & (raw[:, :, gd.SCALE] < 3.0)
  g = x0.grad
  assert (g[:, :, gd.SCALE][~inside] == 0).all() and (g[:, :, gd.SCALE][inside] != 0).all()
  assert (g[:, :, 8:] != 0).all()   # colour, offset: never clamped


def test_unknown_clamp_raises():
  x0, mean, std, lo, hi = make()
  with pytest.raises(ValueError):
    gd.decode_x0(x0, mean, std, lo, hi, clamp="nope")


def _grid_and_source(seed=0):
  rng = np.random.default_rng(seed)
  L, H, W = 2, 3, 4
  hit = rng.random((L, H, W)) > 0.2
  xyz = np.where(hit[..., None], np.stack([rng.normal(size=(L, H, W)), rng.normal(size=(L, H, W)), rng.uniform(1, 3, (L, H, W))], -1), np.nan)
  q = Rotation.random(L * H * W, random_state=seed).as_quat()[:, [3, 0, 1, 2]].reshape(H, W, L, 4)
  scales = np.exp(rng.uniform(-9, -3, (H, W, L, 3)))
  scales[..., 2] = 1e-7                                    # one axis far under the floor
  grid = {"layer_valid": hit.transpose(1, 2, 0), "gaussian_means": xyz.transpose(1, 2, 0, 3) + 0.01 * rng.normal(size=(H, W, L, 3)),
          "gaussian_scales": scales, "gaussian_quats": q, "gaussian_opacities": rng.uniform(0.1, 0.9, (H, W, L)),
          "gaussian_colors": rng.uniform(0, 1, (H, W, L, 3))}
  source = {"xyz_cam": torch.tensor(xyz, dtype=torch.float32), "hit": torch.tensor(hit), "pose_gl": torch.tensor(gd.OPENGL_TO_OPENCV),
            "K_depth": torch.tensor([[500.0, 0, 2], [0, 500.0, 2], [0, 0, 1]])}
  return grid, source


def test_floor_add_decodes_to_the_max_floor_encoding():
  """scale_floor_mode add: decode(encode) = the fitted axes over the floor (as max's, which decodes them unchanged), an axis at or
  under (1 + FLOOR_ADD_EPS) floors at that; and any channel value decodes over the floor."""
  grid, source = _grid_and_source()
  hit = source["hit"]
  kw = dict(scale_order="sorted", scale_floor_px=0.1, rot_form="keep")
  x_max = torch.tensor(gd.encode_grid(grid, source, **kw))
  x_add = torch.tensor(gd.encode_grid(grid, source, scale_floor_mode="add", **kw))
  C = gd.CHANNELS
  zero, one = torch.zeros(C), torch.ones(C)
  to = lambda x: x.permute(1, 0, 2, 3)[None]   # (C, L, H, W) -> (1, L, C, H, W)
  z = source["xyz_cam"][..., 2]
  ctx = {"px": 0.1, "anchor_z": torch.nan_to_num(z)[None], "fx": torch.tensor([500.0])}
  s_max = gd.decode_x0(to(x_max), zero, one)["scale"][0].permute(0, 2, 3, 1)[hit]                  # (N, 3)
  s_add = gd.decode_x0(to(x_add), zero, one, floor=ctx)["scale"][0].permute(0, 2, 3, 1)[hit]
  mz = z[hit] + x_add[11 + 2].permute(0, 1, 2)[hit]        # the mean's depth
  floor = (0.1 * mz / 500.0)[:, None].expand(-1, 3)
  at = s_max < (1 + gd.FLOOR_ADD_EPS) * floor                # at the floor, or within FLOOR_ADD_EPS floors over it
  assert torch.isclose(s_max, floor, rtol=1e-4).any() and (~at).any()
  torch.testing.assert_close(s_add[~at], s_max[~at], rtol=1e-4, atol=0)
  torch.testing.assert_close(s_add[at], (1 + gd.FLOOR_ADD_EPS) * floor[at], rtol=1e-4, atol=0)
  wild = gd.decode_x0(to(x_add) - 50.0 * torch.eye(C)[1][None, None, :, None, None], zero, one, floor=ctx)["scale"][0]
  assert (wild[:, 0].permute(0, 1, 2)[hit] >= 0.1 * mz / 500.0 * (1 - 1e-5)).all()


def test_floor_add_tag_and_model_wiring():
  assert gd.encoding_tag({"scale_order": "sorted", "scale_floor_px": 0.1, "scale_floor_mode": "add"}) == "sort_floor0.1pxadd"
  assert gd.encoding_tag({"scale_order": "sorted", "scale_floor_px": 0.1}) == "sort_floor0.1px"
  with pytest.raises(ValueError):
    gd.normalize_encoding({"scale_floor_mode": "nope"})
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.floor_add_px = 0.1
  with pytest.raises(ValueError):   # the floor context must be given
    m.decode(torch.zeros(1, 2, gd.CHANNELS, 3, 4))
  batch = {"source": {"xyz_cam": torch.ones(2, 2, 3, 4, 3), "K_depth": torch.eye(3).expand(2, 3, 3) * 500}}
  ctx = m.floor_context(batch, torch.device("cpu"))
  assert ctx["px"] == 0.1 and ctx["anchor_z"].shape == (2, 2, 3, 4) and torch.equal(ctx["fx"], torch.tensor([500.0, 500.0]))
