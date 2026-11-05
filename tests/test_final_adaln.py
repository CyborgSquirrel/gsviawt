"""diffusion.final_adaln: DiT's FinalLayer before DiffusionHead.out. Off: no new parameters and the plain readout out(trunk(...));
on: zero-initialized (LayerNorm + out at init), the modulation per (sample, layer) from the time embedding, and the packed rows
(diffusion.token_drop) modulated like the dense tokens they came from."""
import torch

import gs_diffusion as gd

L, GRID, PS, FEAT = 6, 4, 2, 64
P = GRID * GRID


def make_head(final_adaln, seed=0):
  torch.manual_seed(seed)
  head = gd.DiffusionHead(FEAT, L, patch_size=PS, final_adaln=final_adaln)
  torch.nn.init.normal_(head.out.weight, std=0.02)
  return head


def inputs(b=2, seed=1):
  g = torch.Generator().manual_seed(seed)
  feats = torch.randn(b, L, P, FEAT, generator=g)
  x_net = torch.randn(b, L, gd.CHANNELS, GRID * PS, GRID * PS, generator=g)
  t = torch.rand(b, L, generator=g)
  return feats, x_net, t


def trunk_out(head, feats, x_net, t):
  """The trunk's output (B, L, P, D) and the time embedding, as DiffusionHead.forward computes them."""
  feats = (feats - feats.mean((1, 2, 3), keepdim=True)) / (feats.var((1, 2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
  patches = gd.rearrange(x_net, "b l c (h py) (w px) -> b l (h w) (c py px)", py=PS, px=PS)
  tokens = torch.nn.functional.pad(torch.cat([feats * head.img_token_scale, patches], -1), (0, head.pad))
  emb = head.time_embedding(t)
  return head.trunk(tokens, head.adaln(t, emb)), emb


def to_grid(out):
  return gd.rearrange(out, "b l (h w) (c py px) -> b l c (h py) (w px)", h=GRID, w=GRID, py=PS, px=PS)


def test_off_adds_no_parameters_and_is_the_plain_readout():
  head = make_head(False)
  assert head.final_ada is None and head.final_norm is None
  assert not any(k.startswith("final_") for k in head.state_dict())
  feats, x_net, t = inputs()
  with torch.no_grad():
    h, _ = trunk_out(head, feats, x_net, t)
    torch.testing.assert_close(head(feats, x_net, t), to_grid(head.out(h)), rtol=0, atol=0)


def test_on_starts_as_layernorm_then_out():
  head = make_head(True)
  assert torch.count_nonzero(head.final_ada[-1].weight) == 0 and torch.count_nonzero(head.final_ada[-1].bias) == 0
  feats, x_net, t = inputs()
  with torch.no_grad():
    h, _ = trunk_out(head, feats, x_net, t)
    ref = head.out(torch.nn.functional.layer_norm(h, (head.dim,), eps=1e-6))
    torch.testing.assert_close(head(feats, x_net, t), to_grid(ref), rtol=1e-5, atol=1e-6)


def test_on_modulates_per_sample_and_layer_from_t():
  head = make_head(True)
  torch.nn.init.normal_(head.final_ada[-1].weight, std=0.05)
  torch.nn.init.normal_(head.final_ada[-1].bias, std=0.05)
  feats, x_net, t = inputs()
  with torch.no_grad():
    h, emb = trunk_out(head, feats, x_net, t)
    shift, scale = head.final_ada(emb).chunk(2, -1)                      # (B, L, D)
    ref = head.out(torch.nn.functional.layer_norm(h, (head.dim,), eps=1e-6) * (1 + scale[:, :, None]) + shift[:, :, None])
    torch.testing.assert_close(head(feats, x_net, t), to_grid(ref), rtol=1e-5, atol=1e-6)
    t2 = t.clone()
    t2[0, 3] = 1 - t2[0, 3]   # another t for one (sample, layer): its readout changes; the other layers of that sample only through
    diff = (head(feats, x_net, t2) - head(feats, x_net, t)).abs().amax((2, 3, 4))   # the trunk's ray / global blocks (the layers'
    others = torch.cat([diff[0, :3], diff[0, 4:]])                                   # mean AdaLN, LayerScale 1e-4): tiny
    assert diff[1].sum() == 0 and diff[0, 3] > 1e3 * others.max()


def test_packed_rows_get_their_own_sample_and_layer_modulation():
  head = make_head(True)
  torch.nn.init.normal_(head.final_ada[-1].weight, std=0.05)
  feats, x_net, t = inputs()
  g = torch.Generator().manual_seed(3)
  keep = torch.rand(2, L, P, generator=g) < 0.4
  pi = gd.PackIndex(keep)
  with torch.no_grad():
    h, emb = trunk_out(head, feats, x_net, t)
    dense = head.readout(h, emb)                                         # (B, L, P, out)
    packed = head.readout(pi.gather(h), emb, pi.frame)                   # (T, out)
    torch.testing.assert_close(packed, pi.gather(dense), rtol=1e-5, atol=1e-6)
    all_kept = gd.PackIndex(torch.ones(2, L, P, dtype=torch.bool))
    torch.testing.assert_close(head(feats, x_net, t, all_kept), head(feats, x_net, t), rtol=1e-5, atol=1e-6)
