"""diffusion.in_proj: shared (one Linear for the whole projected token, as before) | split (the image features and the noisy values
projected separately, to in_proj_image_dim and dim - in_proj_image_dim, then concatenated)."""
import pytest
import torch

import gs_diffusion as gd

L, GRID, PS, FEAT = 6, 4, 2, 64
P = GRID * GRID
NOISE = gd.CHANNELS * PS * PS      # 56; lossless width 64 + 56 = 120 -> 128


def inputs(b=2, seed=1):
  g = torch.Generator().manual_seed(seed)
  return (torch.randn(b, L, P, FEAT, generator=g), torch.randn(b, L, gd.CHANNELS, GRID * PS, GRID * PS, generator=g),
          torch.rand(b, L, generator=g))


def head(**kw):
  torch.manual_seed(0)
  h = gd.DiffusionHead(FEAT, L, patch_size=PS, **kw)
  if hasattr(h, "out"):
    torch.nn.init.normal_(h.out.weight, std=0.02)
  return h


def test_shared_is_the_default_and_unchanged():
  h = head(dim=64)
  assert h.in_proj is not None and h.in_proj.in_features == FEAT + NOISE and not hasattr(h, "in_proj_image")
  assert head().in_proj is None and head().pad == 128 - (FEAT + NOISE)   # lossless: no projection at all


def test_split_shapes_and_parameters():
  h = head(dim=64, in_proj="split", in_proj_image_dim=16)
  assert h.in_proj is None
  assert (h.in_proj_image.in_features, h.in_proj_image.out_features) == (FEAT, 16)
  assert (h.in_proj_noise.in_features, h.in_proj_noise.out_features) == (NOISE, 48)
  feats, x_net, t = inputs()
  out = h(feats, x_net, t)
  assert out.shape == x_net.shape and torch.isfinite(out).all()


def test_split_keeps_the_parts_in_separate_channels():
  """The token's first in_proj_image_dim channels depend only on the image features, the rest only on the noisy values."""
  h = head(dim=64, in_proj="split", in_proj_image_dim=16)
  captured = []
  h.trunk.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach().clone()))
  feats, x_net, t = inputs()
  with torch.no_grad():
    h(feats, x_net, t)
    h(feats, x_net + 1.0, t)                 # only the noisy values change
    h(feats * 2.0 + 1.0, x_net, t)           # only the image features change (the z-score makes *2 a no-op; +1 too: use noise)
    feats2 = torch.randn_like(feats)
    h(feats2, x_net, t)
  base, noise_changed, _, image_changed = captured
  torch.testing.assert_close(noise_changed[..., :16], base[..., :16], rtol=0, atol=0)
  assert not torch.allclose(noise_changed[..., 16:], base[..., 16:])
  torch.testing.assert_close(image_changed[..., 16:], base[..., 16:], rtol=0, atol=0)
  assert not torch.allclose(image_changed[..., :16], base[..., :16])


def test_split_dense_and_packed_agree():
  h = head(dim=64, in_proj="split", in_proj_image_dim=16)
  for blk in h.trunk.blocks:
    blk.ls1.data.fill_(1.0)
    blk.ls2.data.fill_(1.0)
  feats, x_net, t = inputs()
  with torch.no_grad():
    dense = h(feats, x_net, t)
    packed = h(feats, x_net, t, gd.PackIndex(torch.ones(2, L, P, dtype=torch.bool)))
  torch.testing.assert_close(packed, dense, rtol=1e-5, atol=1e-6)


def test_split_with_the_unconditional_branch():
  h = head(dim=64, in_proj="split", in_proj_image_dim=16)
  feats, x_net, t = inputs()
  with torch.no_grad():
    out = h(feats, x_net, t, uncond=torch.tensor([True, False]))
  assert torch.isfinite(out).all()


def test_split_gets_gradients():
  h = head(dim=64, in_proj="split", in_proj_image_dim=16)
  feats, x_net, t = inputs()
  h(feats, x_net, t).square().mean().backward()
  assert h.in_proj_image.weight.grad.abs().sum() > 0 and h.in_proj_noise.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kw, match", [
  (dict(in_proj="split", in_proj_image_dim=16), "projected token"),                       # lossless width: nothing to split
  (dict(dim=64, in_proj="split"), "in_proj_image_dim"),
  (dict(dim=64, in_proj="split", in_proj_image_dim=64), "in_proj_image_dim"),
  (dict(dim=64, in_proj="nope"), "in_proj"),
  (dict(dim=64, in_proj="split", in_proj_image_dim=16, readout="slice"), "lossless"),
])
def test_invalid(kw, match):
  with pytest.raises(ValueError, match=match):
    head(**kw)
