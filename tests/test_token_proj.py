"""DiffusionHead's token width (diffusion.dim): null = the lossless token (features | raw noisy patch | zero pad), no projection;
below that width a learned Linear projects the token to dim (a smaller trunk), output shape unchanged."""
import pytest
import torch

import gs_diffusion as gd


def run(head, b=1, l=2, grid=2, feature_dim=64):
  p, ps = grid * grid, head.patch_size
  feats = torch.randn(b, l, p, feature_dim)
  x = torch.randn(b, l, gd.CHANNELS, grid * ps, grid * ps)
  return head(feats, x, torch.rand(b, l)), x


def test_default_is_lossless_and_dim_below_projects():
  kw = dict(feature_dim=64, num_layers=2, patch_size=2, num_blocks=3)
  in_dim = 64 + gd.CHANNELS * 4                                  # 120
  lossless = gd.DiffusionHead(**kw)
  assert lossless.in_proj is None and lossless.dim == 128 and lossless.pad == 128 - in_dim
  small = gd.DiffusionHead(dim=64, **kw)
  assert small.dim == 64 and small.pad == 0 and small.in_proj.in_features == in_dim and small.in_proj.out_features == 64
  assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in lossless.parameters()) / 2
  for head in (lossless, small):
    torch.nn.init.normal_(head.out.weight)                       # (zero-initialized: make the output depend on the input)
    out, x = run(head)
    assert out.shape == x.shape
    out.sum().backward()
  assert small.in_proj.weight.grad is not None and small.in_proj.weight.grad.abs().sum() > 0


def test_dim_must_be_multiple_of_64():
  with pytest.raises(ValueError):
    gd.DiffusionHead(64, 2, patch_size=2, dim=100)
