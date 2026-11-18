"""diffusion.film: frame (the default: one FiLM before the frame-position blocks, parameters and outputs as before) | every (a FiLM
before every block, each block its own zero-initialized output projection on the shared layer_embed + hidden layer: a fresh model
computes exactly what frame's does; trained, every block gets its own gamma, beta; dense and packed agree)."""
import pytest
import torch

import gs_diffusion as gd
from tests.test_token_drop import FEAT, L, P, PS, inputs, rel


def test_default_parameters_unchanged():
  t = gd.DiTTrunk(64, L, 9, 2)
  assert t.film_mode == "frame" and not hasattr(t, "film_heads")
  assert {k for k in t.state_dict() if "film" in k} == {"film.0.weight", "film.0.bias", "film.2.weight", "film.2.bias"}
  e = gd.DiTTrunk(64, L, 9, 2, film="every")
  assert len(e.film_heads) == 9 and all((h.weight == 0).all() and (h.bias == 0).all() for h in e.film_heads)
  assert {k for k in e.state_dict() if k.startswith("film.")} == {"film.0.weight", "film.0.bias"}
  with pytest.raises(ValueError):
    gd.DiTTrunk(64, L, 9, 2, film="some")


def _pair(attention="global", seed=0):
  torch.manual_seed(seed)
  frame = gd.DiffusionHead(FEAT, L, patch_size=PS, attention=attention)
  every = gd.DiffusionHead(FEAT, L, patch_size=PS, attention=attention, film="every")
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for p in frame.parameters():
      p.copy_(0.05 * torch.randn(p.shape, generator=g))
    frame.trunk.film[-1].weight.zero_()   # a fresh (zero-init) FiLM output in frame too: both trunks' FiLM is then the identity
    frame.trunk.film[-1].bias.zero_()
  missing, unexpected = every.load_state_dict(frame.state_dict(), strict=False)
  assert set(unexpected) == {"trunk.film.2.weight", "trunk.film.2.bias"} and all("film_heads" in k for k in missing)
  return frame.eval(), every.eval()


@pytest.mark.parametrize("attention", ["global", "frame_ray_global"])
def test_zero_init_every_equals_frame(attention):
  frame, every = _pair(attention)
  feats, x_net, t = inputs(frame, 2)
  keep = torch.rand(2, L, P, generator=torch.Generator().manual_seed(0)) < 0.5
  keep[:, 0, 0] = True
  with torch.no_grad():
    assert rel(every(feats, x_net, t), frame(feats, x_net, t)) < 1e-6
    assert rel(every(feats, x_net, t, gd.PackIndex(keep)), frame(feats, x_net, t, gd.PackIndex(keep))) < 1e-6


def test_every_block_has_its_own_film():
  """Trained FiLM heads: block k's own (gamma_k, beta_k) before it (a hand-built reference), the FiLM before a non-frame block
  matters, the packed path equals the dense one."""
  _, every = _pair()
  g = torch.Generator().manual_seed(5)
  with torch.no_grad():
    for h in every.trunk.film_heads:
      h.weight.copy_(0.05 * torch.randn(h.weight.shape, generator=g))
      h.bias.copy_(0.1 * torch.randn(h.bias.shape, generator=g))
  tr = every.trunk
  feats, x_net, t = inputs(every, 2)
  x = torch.randn(2, L, P, every.dim, generator=g)
  adaln = every.adaln(t)
  with torch.no_grad():
    out = tr(x, adaln)
    cos, sin = tr._rope_tables(int(P ** 0.5), x.device)
    e = tr.film(tr.layer_embed(torch.arange(L)))
    ref, am = x, adaln.mean(1, keepdim=True)
    for k, (block, cond) in enumerate(zip(tr.blocks, tr.cond)):
      gamma, beta = tr.film_heads[k](e).chunk(2, dim=-1)
      ref = ref * (1 + gamma[None, :, None]) + beta[None, :, None]
      ref = block(ref, cos, sin, adaln if cond == "frame" else am)
    assert rel(out, ref) < 1e-6
    tr.film_heads[1].weight.zero_()   # block 1 (a ray position): its FiLM changes the output
    tr.film_heads[1].bias.zero_()
    assert rel(tr(x, adaln), out) > 1e-4
    full = gd.PackIndex(torch.ones(2, L, P, dtype=torch.bool))
    assert rel(every(feats, x_net, t, full), every(feats, x_net, t)) < 1e-6


# ---------------------------------------------------------------- the defaults bit-identical to the code before the options


def _old_forward(tr, x, adaln):
  """DiTTrunk.forward as it was before diffusion.film (one FiLM call, gamma / beta indexed once, applied at the frame blocks)."""
  _, num_layers, num_patches, _ = x.shape
  cos, sin = tr._rope_tables(int(round(num_patches ** 0.5)), x.device)
  gamma, beta = tr.film(tr.layer_embed(torch.arange(num_layers, device=x.device))).chunk(2, dim=-1)
  gamma, beta = gamma[None, :, None], beta[None, :, None]
  adaln_mean = adaln.mean(1, keepdim=True)
  for block, cond in zip(tr.blocks, tr.cond):
    if cond == "frame":
      x = x * (1 + gamma) + beta
    x = block(x, cos, sin, adaln if cond == "frame" else adaln_mean)
  return x


def _old_forward_packed(tr, x, adaln, pack):
  b, num_layers = adaln.shape[:2]
  cos, sin = tr._rope_tables(int(round(pack.P ** 0.5)), x.device)
  cos_r, sin_r = cos[pack.p], sin[pack.p]
  gamma, beta = tr.film(tr.layer_embed(torch.arange(num_layers, device=x.device))).chunk(2, dim=-1)
  gamma, beta = gamma[pack.l], beta[pack.l]
  adaln_frame = adaln.reshape(b * num_layers, *adaln.shape[2:])
  adaln_mean = adaln.mean(1)
  for block, cond in zip(tr.blocks, tr.cond):
    if cond == "frame":
      x = x * (1 + gamma) + beta
      a, seg = adaln_frame, pack.frame
    else:
      a, seg = adaln_mean, pack.b
    x = block.forward_packed(x, pack, cos_r, sin_r, a, seg)
  return x


def _old_adaln(head, t):
  b, l = t.shape
  emb = head.time_embed(gd.sinusoidal_embedding(t.reshape(-1) * 1000.0, head.t_embed_channels))
  return head.time_proj(emb).reshape(b, l, 6, head.dim)


@pytest.mark.parametrize("attention", ["global", "frame_ray_global"])
def test_defaults_bit_identical_to_before(attention):
  """film frame + colour_noise joint (the defaults): the trunk's dense and packed forward and the AdaLN input are torch.equal to the
  code before the options, and so are the dense gradients. (The packed backward is not deterministic run-to-run even in the old
  code -- its index scatters accumulate in varying order, ~1e-12 -- so its gradients are compared with a tolerance.)"""
  torch.manual_seed(0)
  head = gd.DiffusionHead(FEAT, L, patch_size=PS, attention=attention)
  g = torch.Generator().manual_seed(3)
  with torch.no_grad():
    for p in head.parameters():
      p.copy_(0.05 * torch.randn(p.shape, generator=g))
  tr = head.trunk
  x = torch.randn(2, L, P, head.dim, generator=g)
  t = torch.rand(2, L, generator=g)
  keep = torch.rand(2, L, P, generator=g) < 0.5
  keep[:, 0, 0] = True
  pack = gd.PackIndex(keep)
  with torch.no_grad():
    adaln = head.adaln(t)
    assert torch.equal(adaln, _old_adaln(head, t))
    assert torch.equal(tr(x, adaln), _old_forward(tr, x, adaln))
    xp = pack.gather(x)
    assert torch.equal(tr.forward_packed(xp, adaln, pack), _old_forward_packed(tr, xp, adaln, pack))
  params = list(head.parameters())
  new_d, old_d = (torch.autograd.grad(f().square().sum(), params, allow_unused=True)
                  for f in (lambda: tr(x, head.adaln(t)), lambda: _old_forward(tr, x, _old_adaln(head, t))))
  assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(new_d, old_d))
  new_p, old_p = (torch.autograd.grad(f().square().sum(), params, allow_unused=True)
                  for f in (lambda: tr.forward_packed(xp, head.adaln(t), pack),
                            lambda: _old_forward_packed(tr, xp, _old_adaln(head, t), pack)))
  assert all((a is None and b is None) or torch.allclose(a, b, rtol=1e-9, atol=1e-9) for a, b in zip(new_p, old_p))
