"""diffusion_schedule: the logit-normal / plateaued logit-normal densities and samplers, and LayerAwareSchedule."""
import numpy as np
import pytest
import torch

from diffusion_schedule import (LayerAwareSchedule, logit_normal_pdf, plateau_logit_normal_pdf, plateau_mode,
                                sample_logit_normal, sample_plateau_logit_normal, T_MIN)

T = torch.linspace(1e-4, 1 - 1e-4, 200001, dtype=torch.float64)


@pytest.mark.parametrize("pdf", [lambda t: logit_normal_pdf(t), lambda t: plateau_logit_normal_pdf(t),
                                 lambda t: plateau_logit_normal_pdf(t, mirror=True), lambda t: logit_normal_pdf(t, 0.5, 0.8)])
def test_densities_integrate_to_one(pdf):
  assert abs(float(torch.trapezoid(pdf(T), T)) - 1) < 1e-3


def test_plateau_mode_and_shape():
  assert abs(plateau_mode(0.0, 1.0) - 0.5) < 1e-9
  p = plateau_logit_normal_pdf(T)
  above = p[T > 0.5]
  assert float(above.max() - above.min()) < 1e-9                     # constant after the mode
  assert float(p[T < 0.25].max()) < float(above[0])                  # rising before it
  with pytest.raises(ValueError):
    plateau_mode(0.0, 1.5)                                           # bimodal for sigma >= sqrt(2)


@pytest.mark.parametrize("sampler,pdf", [(sample_logit_normal, logit_normal_pdf),
                                         (sample_plateau_logit_normal, plateau_logit_normal_pdf)])
def test_samples_follow_density(sampler, pdf):
  g = torch.Generator().manual_seed(0)
  s = sampler((400000,), generator=g).numpy()
  hist, edges = np.histogram(s, bins=40, range=(0, 1), density=True)
  centers = torch.tensor((edges[:-1] + edges[1:]) / 2)
  assert np.abs(hist - pdf(centers).numpy()).max() < 0.05


def test_layer_aware_schedule():
  s = LayerAwareSchedule(6)
  g = torch.Generator().manual_seed(0)
  t1 = s.sample(1000, phase=1, generator=g)
  assert t1.shape == (1000, 6) and t1.dtype == torch.float32 and float(t1.min()) > 0 and float(t1.max()) < 1
  assert float(t1.std(dim=1).mean()) > 0.05                         # phase 1: independent per layer
  t2 = s.sample(1000, phase=2, generator=g)
  assert (t2 == t2[:, :1]).all()                                     # phase 2: one t per sample
  with pytest.raises(ValueError):
    s.sample(2, phase=3)
  s2 = LayerAwareSchedule(6, standard=dict(mu=0.3), plateau=dict(sigma=0.9))   # overrides merge with the defaults
  assert s2.standard == dict(mu=0.3, sigma=1.0) and s2.plateau == dict(mu=0.0, sigma=0.9, mirror=False)
  d = s.densities(T)
  assert torch.allclose(d["mixture"], 0.5 * (d["standard"] + d["plateau"]))


def test_uniform_phase1_per_layer_phase2_shared():
  s = LayerAwareSchedule(6, uniform=True)
  t = s.sample(20000, phase=1, generator=torch.Generator().manual_seed(0))
  assert t.shape == (20000, 6) and t.dtype == torch.float32
  assert t.min() >= T_MIN and t.max() <= 1 - T_MIN
  for l in range(6):                                                   # every layer uniform, including layer 0 (no plateau)
    assert abs(t[:, l].mean().item() - 0.5) < 0.01
    assert abs((t[:, l] < 0.1).float().mean().item() - 0.1) < 0.01
  assert abs(torch.corrcoef(t[:, :2].T)[0, 1].item()) < 0.03          # independent layers
  t2 = s.sample(1000, phase=2, generator=torch.Generator().manual_seed(1))
  assert (t2 == t2[:, :1]).all() and abs(t2[:, 0].mean().item() - 0.5) < 0.03


def test_uniform_off_is_the_default_and_unchanged():
  g = lambda: torch.Generator().manual_seed(3)
  assert torch.equal(LayerAwareSchedule(6).sample(64, 1, generator=g()), LayerAwareSchedule(6, uniform=False).sample(64, 1, generator=g()))
  assert not LayerAwareSchedule(6).uniform
  assert (LayerAwareSchedule(6, uniform=True, fixed_t=1.0).sample(4, 1) == 1).all()   # fixed_t (the control) wins
