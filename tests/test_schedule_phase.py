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
