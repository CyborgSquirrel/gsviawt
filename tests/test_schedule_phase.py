"""diffusion.schedule.phase: 2 trains (and the validations' seeded timesteps use) one t per volume shared by all its layers."""
import types

import torch

import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule


def model(phase):
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.head = lambda feats, x, t: x
  m._feats = lambda image: None
  m.normalize = lambda x: x
  m.schedule = LayerAwareSchedule(6)
  m.phase = phase
  return m


def test_phase_sets_training_and_validation_timesteps():
  B, L = 64, 6
  hit = torch.ones(B, L, 4, 4, dtype=torch.bool)
  image, x0 = torch.zeros(B, 3, 8, 8), torch.zeros(B, L, 14, 4, 4)
  for seeds in (None, list(range(B))):   # training (global stream) and validation (seeded)
    t1 = model(1).training_loss(image, hit, x0, seeds=seeds, return_pred=True)[4]
    t2 = model(2).training_loss(image, hit, x0, seeds=seeds, return_pred=True)[4]
    assert t1.shape == t2.shape == (B, L)
    assert (t1.std(dim=1) > 0).all()                 # phase 1: independent per layer
    assert (t2 == t2[:, :1]).all()                   # phase 2: shared over the layers
    assert t2[:, 0].std() > 0.1                      # ... but different between volumes


def test_phase3_hidden_synced():
  """phase 3: layer 0 its own t, one t shared by the occluded layers 1.. (training and seeded validation; both densities)."""
  B, L = 64, 6
  hit = torch.ones(B, L, 4, 4, dtype=torch.bool)
  image, x0 = torch.zeros(B, 3, 8, 8), torch.zeros(B, L, 14, 4, 4)
  for seeds in (None, list(range(B))):
    t3 = model(3).training_loss(image, hit, x0, seeds=seeds, return_pred=True)[4]
    assert t3.shape == (B, L)
    assert (t3[:, 1:] == t3[:, 1:2]).all()            # the occluded layers share one t
    assert (t3[:, 0] != t3[:, 1]).float().mean() > 0.9  # layer 0 its own
    assert t3[:, 1].std() > 0.1 and t3[:, 0].std() > 0.1
  for uniform in (False, True):
    t = LayerAwareSchedule(L, uniform=uniform).sample(4096, phase=3, generator=torch.Generator().manual_seed(0))
    assert (t[:, 1:] == t[:, 1:2]).all() and float((t[:, 0] - t[:, 1]).abs().mean()) > 0.1
    if uniform:   # both marginals uniform
      assert abs(float(t[:, 0].mean()) - 0.5) < 0.02 and abs(float(t[:, 1].mean()) - 0.5) < 0.02
