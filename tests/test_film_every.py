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
