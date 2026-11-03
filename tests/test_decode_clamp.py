"""gs_diffusion.decode_x0's clamp (diffusion.decode_clamp): channel = every channel into its own training range; scale = only the
scales, all 3 axes into one shared range (so an axis in the "wrong" channel of a sorted encoding is not cut), the rest unclamped."""
import pytest
import torch

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
