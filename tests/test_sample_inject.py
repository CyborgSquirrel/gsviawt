"""GSDiffusionModel.sample's inject (eval_diffusion.inject_fields): before every model call the injected fields' channels of the
state are the given x0 noised to the step's t along the sampler's own noise; the other channels and a 1-step sample are unchanged."""
import torch

import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule

B, L, H, W = 2, 6, 8, 8
COL = gd.X0_SLICES["sh_dc"]


def shell():
  m = object.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.schedule, m.num_layers, m.steps, m.eval_seed = LayerAwareSchedule(L), L, 1, 0
  m.register_buffer("x0_mean", torch.zeros(gd.CHANNELS))
  m.register_buffer("x0_std", torch.ones(gd.CHANNELS))
  m.register_buffer("x0_lo", torch.full((gd.CHANNELS,), -10.0))
  m.register_buffer("x0_hi", torch.full((gd.CHANNELS,), 10.0))
  calls = []

  class Head(torch.nn.Module):
    def forward(self, feats, x_net, t):
      calls.append((x_net.clone(), float(t.flatten()[0])))
      return torch.full_like(x_net, 0.25)
  m.head = Head()
  m._feats = lambda image: None
  return m, calls


def test_inject_noises_the_given_fields_along_the_sampler_noise():
  hit = torch.ones(B, L, H, W, dtype=torch.bool)
  x0 = torch.randn(B, L, gd.CHANNELS, H, W)
  m, plain = shell()
  m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=4)
  m, inj = shell()
  m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=4, inject=(x0, ["sh_dc"]))
  eps = plain[0][0]                                     # the first call (t = 1) sees the pure noise
  for (xp, t), (xi, ti) in zip(plain, inj):
    assert t == ti
    torch.testing.assert_close(xi[:, :, COL], ((1 - t) * x0 + t * eps)[:, :, COL])
  torch.testing.assert_close(inj[0][0], plain[0][0])   # t = 1: the injection is pure noise
  rest = torch.ones(gd.CHANNELS, dtype=torch.bool)
  rest[COL] = False
  for (xp, _), (xi, _) in zip(plain, inj):              # a constant head: the other channels evolve identically
    torch.testing.assert_close(xi[:, :, rest], xp[:, :, rest])


def test_one_step_sample_unchanged_by_inject():
  hit = torch.rand(B, L, H, W) > 0.3
  x0 = torch.randn(B, L, gd.CHANNELS, H, W)
  m, _ = shell()
  a = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=1)
  b = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=1, inject=(x0, ["sh_dc"]))
  for k in a:
    torch.testing.assert_close(a[k], b[k])


def test_inject_onestep_uses_the_first_prediction():
  """inject x0 "onestep": from the second call on, the fields' channels are the first (t = 1) prediction noised to t."""
  hit = torch.ones(B, L, H, W, dtype=torch.bool)
  m, plain = shell()
  m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=4)
  m, inj = shell()
  m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=4, inject=("onestep", ["sh_dc"]))
  eps = plain[0][0]
  first = torch.full_like(eps, 0.25)                    # the shell head's prediction
  torch.testing.assert_close(inj[0][0], plain[0][0])
  for xi, t in inj[1:]:
    torch.testing.assert_close(xi[:, :, COL], ((1 - t) * first + t * eps)[:, :, COL])


def test_t_min_zero_lands_exactly_on_the_prediction():
  """t_min 0: every Euler step divides by t, so with a constant x0 prediction the sample equals it exactly at any step count; with the
  default clamp (0.05) and 50 steps the steps below t = 0.05 leave a residual of the noise."""
  hit = torch.ones(B, L, H, W, dtype=torch.bool)
  m, _ = shell()
  exact = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=50, t_min=0.0)
  torch.testing.assert_close(exact["raw_scale"], torch.full_like(exact["raw_scale"], 0.25))
  clamped = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8], steps=50)
  assert (clamped["raw_scale"] - 0.25).abs().max() > 1e-3
