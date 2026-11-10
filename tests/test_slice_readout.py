"""diffusion.readout: linear (the zero-initialized output projection, as before) | slice (no output layer: x0_hat is the trunk's
final residual stream at the token's noisy-value channels). Slice has no `out` parameter, starts as a copy of its input (LayerScale
1e-4, zero-initialized FiLM), reads exactly that slice of the stream, agrees between the dense and the packed (token_drop) paths,
and refuses a projected (narrower than lossless) token."""
import pytest
import torch

import gs_diffusion as gd

L, GRID, PS, FEAT = 6, 4, 2, 64
P = GRID * GRID


def make_head(readout, seed=0, **kw):
  torch.manual_seed(seed)
  return gd.DiffusionHead(FEAT, L, patch_size=PS, readout=readout, **kw)


def inputs(b=2, seed=1):
  g = torch.Generator().manual_seed(seed)
  feats = torch.randn(b, L, P, FEAT, generator=g)
  x_net = torch.randn(b, L, gd.CHANNELS, GRID * PS, GRID * PS, generator=g)
  t = torch.rand(b, L, generator=g)
  return feats, x_net, t


def test_linear_is_the_default_and_keeps_out():
  assert gd.DiffusionHead(FEAT, L, patch_size=PS).readout == "linear"
  assert hasattr(make_head("linear"), "out")


def test_slice_has_no_output_layer():
  head = make_head("slice")
  assert not hasattr(head, "out")
  assert not any(k.startswith("out.") for k in head.state_dict())


def test_fresh_slice_head_copies_its_input():
  head = make_head("slice")
  feats, x_net, t = inputs()
  with torch.no_grad():
    out = head(feats, x_net, t)
  assert out.shape == x_net.shape
  # LayerScale 1e-4 and the zero-initialized FiLM: the stream barely moves, so the noisy-value slice is ~x_net
  assert (out - x_net).abs().max() < 1e-2 * x_net.abs().max()


def test_slice_reads_the_stream_at_the_noisy_value_channels():
  head = make_head("slice")
  for blk in head.trunk.blocks:   # a non-trivial trunk, so the readout is not just the input
    blk.ls1.data.fill_(1.0)
    blk.ls2.data.fill_(1.0)
  feats, x_net, t = inputs()
  captured = {}
  head.trunk.register_forward_hook(lambda m, i, o: captured.setdefault("h", o))
  with torch.no_grad():
    out = head(feats, x_net, t)
  h = captured["h"]                                            # (B, L, P, dim)
  ref = h[..., FEAT:FEAT + head.noise_dim]
  ref = gd.rearrange(ref, "b l (h w) (c py px) -> b l c (h py) (w px)", h=GRID, w=GRID, py=PS, px=PS)
  torch.testing.assert_close(out, ref, rtol=0, atol=0)
  assert (out - x_net).abs().max() > 1e-2                     # and the trunk did change it


def test_slice_dense_and_packed_agree():
  head = make_head("slice")
  for blk in head.trunk.blocks:
    blk.ls1.data.fill_(1.0)
    blk.ls2.data.fill_(1.0)
  feats, x_net, t = inputs()
  with torch.no_grad():
    dense = head(feats, x_net, t)
    packed = head(feats, x_net, t, gd.PackIndex(torch.ones(2, L, P, dtype=torch.bool)))
  torch.testing.assert_close(packed, dense, rtol=1e-5, atol=1e-6)


def test_slice_gets_gradients_into_the_trunk():
  head = make_head("slice")
  feats, x_net, t = inputs()
  head(feats, x_net, t).square().mean().backward()
  assert head.trunk.blocks[0].ls1.grad is not None and head.trunk.blocks[0].ls1.grad.abs().sum() > 0


def test_slice_refuses_a_projected_token():
  with pytest.raises(ValueError, match="lossless"):
    make_head("slice", dim=64)


def test_unknown_readout():
  with pytest.raises(ValueError, match="readout"):
    make_head("nope")
