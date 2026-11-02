"""diffusion.x0_weight: 0 trains with the rendering loss alone (the x0 loss still logged, but no gradient from it); the default 1
keeps x0 + rendering."""
import types

import torch
from omegaconf import OmegaConf

import train_gs


def step(x0_weight):
  a = torch.tensor(2.0, requires_grad=True)
  b = torch.tensor(3.0, requires_grad=True)
  logged = {}
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"x0_weight": x0_weight, "photom": {"enabled": True, "weight": 1.0}}}),
    views_seen=0, pixels_seen=0, _accepted_since_step=False,
    spike_skip=types.SimpleNamespace(check=lambda v: False, average=None),
    _diffusion_losses=lambda batch: (a * 0.5, {}, {}, None, None),       # x0 loss = 1.0
    _photom_active=lambda: True,
    _diffusion_photom=lambda stage, batch, x0_hat, t, gated: b * 0.1,     # rendering loss = 0.3
    _log_metric=lambda stage, k, v, n: logged.__setitem__(k, float(v)), _log_parts=lambda *x: None)
  loss = train_gs.GSLightningModule._diffusion_train_step(fake, {"batch_size": 4, "x0_raw": torch.zeros(1, 14, 6, 8, 8)})
  loss.backward()
  return float(loss), a.grad, b.grad, logged


def test_x0_weight_zero_trains_on_rendering_only():
  loss, ga, gb, logged = step(0.0)
  assert abs(loss - 0.3) < 1e-6 and float(ga) == 0 and float(gb) != 0
  assert abs(logged["loss/x0"] - 1.0) < 1e-6                              # still logged


def test_x0_weight_default_keeps_both():
  loss, ga, gb, _ = step(1.0)
  assert abs(loss - 1.3) < 1e-6 and float(ga) == 0.5 and float(gb) != 0
