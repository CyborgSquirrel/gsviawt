"""diffusion.schedule.fixed_t (the same-architecture deterministic control): all timesteps are 1, the network input is only the
validity mask (no noise, no x0), and one sampler step returns the decoded x0 prediction. Uses a GSDiffusionModel shell with a
recording head (no backbone)."""
import torch

import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule

B, L, H, W = 2, 6, 8, 8


def shell(fixed_t=1.0):
  m = object.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.schedule, m.num_layers, m.steps, m.eval_seed = LayerAwareSchedule(L, fixed_t=fixed_t), L, 1, 0
  m.register_buffer("x0_mean", torch.zeros(gd.CHANNELS))
  m.register_buffer("x0_std", torch.ones(gd.CHANNELS))
  m.register_buffer("x0_lo", torch.full((gd.CHANNELS,), -10.0))
  m.register_buffer("x0_hi", torch.full((gd.CHANNELS,), 10.0))
  seen = {}

  class Head(torch.nn.Module):
    def forward(self, feats, x_net, t):
      seen["x_net"], seen["t"] = x_net.clone(), t.clone()
      return torch.full_like(x_net, 0.25)
  m.head = Head()
  m._feats = lambda image: None
  return m, seen


def test_schedule_all_ones():
  s = LayerAwareSchedule(L, fixed_t=1.0)
  assert (s.sample(5, 1) == 1).all() and (s.sample(5, 2) == 1).all() and s.sample(5, 1).shape == (5, L)
  assert (s.sample(1, 1, generator=torch.Generator().manual_seed(3)) == 1).all()
  assert not (LayerAwareSchedule(L).sample(5, 1) == 1).all()


def test_input_is_the_validity_mask_and_independent_of_x0():
  m, seen = shell()
  hit = torch.rand(B, L, H, W) > 0.3
  x0_raw = torch.randn(B, gd.CHANNELS, L, H, W)
  image = torch.zeros(B, 3, 4, 4)
  _, _, _, _, t = m.training_loss(image, hit, x0_raw, return_pred=True)
  assert (t == 1).all()
  m.training_loss(image, hit, x0_raw, seeds=[7, 8])
  a = seen["x_net"]
  m.training_loss(image, hit, x0_raw * 3 + 1, seeds=[1, 2])
  assert torch.equal(a, seen["x_net"])                          # no x0, no noise: deterministic
  assert torch.equal(a.float(), (~hit)[:, :, None].float().expand_as(a))


def test_one_step_sample_is_the_decoded_prediction():
  m, seen = shell()
  hit = torch.rand(B, L, H, W) > 0.3
  out = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[7, 8])
  assert torch.allclose(out["raw_scale"], torch.full_like(out["raw_scale"], 0.25))
  assert (seen["t"] == 1).all()
  assert torch.equal(seen["x_net"].float(), (~hit)[:, :, None].float().expand_as(seen["x_net"]))
