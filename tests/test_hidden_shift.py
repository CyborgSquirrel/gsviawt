"""GSDiffusionModel.sample's hidden_shift (eval_diffusion.hidden_shift): Modality Forcing's sampling schedule with the visible layer 0
leading, t_0 = linspace(1, 0) and t_h = alpha t / (1 + (alpha - 1) t) for layers 1..; alpha 1 is the lockstep sampler exactly, every
layer's state lands on the model's last prediction, and the model is given the per-layer t."""
import torch

import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule

B, L, H, W = 2, 6, 8, 8


def shell(pred=None):
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
      calls.append(t.clone())
      return torch.full_like(x_net, 0.25) if pred is None else pred(x_net, t)
  m.head = Head()
  m._feats = lambda image: None
  return m, calls


def f(t, a):
  return a * t / (1 + (a - 1) * t)


def test_alpha_one_is_the_lockstep_sampler():
  hit = torch.rand(B, L, H, W) > 0.3
  pred = lambda x, t: 0.5 * x[:, :, :gd.CHANNELS] + 0.1          # a state-dependent head, so the trajectory matters
  a = shell(pred)[0].sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=5)
  b = shell(pred)[0].sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=5, hidden_shift=1.0)
  for k in a:
    torch.testing.assert_close(a[k], b[k])


def test_per_layer_t_given_to_the_model():
  hit = torch.ones(B, L, H, W, dtype=torch.bool)
  m, calls = shell()
  m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=4, hidden_shift=32.0)
  ts = torch.linspace(1, 0, 5)
  assert len(calls) == 4
  for i, t in enumerate(calls):
    assert t.shape == (B, L)
    torch.testing.assert_close(t[:, 0], ts[i].expand(B))
    torch.testing.assert_close(t[:, 1:], f(ts[i], 32.0).expand(B, L - 1))
  assert float(calls[2][0, 1]) > 0.9 > float(calls[2][0, 0])        # t_v = 0.5: the occluded layers are still nearly pure noise


def test_every_layer_lands_on_the_last_prediction():
  hit = torch.ones(B, L, H, W, dtype=torch.bool)
  for alpha in (2.0, 8.0, 32.0):
    m, _ = shell()
    g = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=6, hidden_shift=alpha)
    ref = shell()[0].decode(torch.full((B, L, gd.CHANNELS, H, W), 0.25))
    for k in g:
      torch.testing.assert_close(g[k], ref[k])
