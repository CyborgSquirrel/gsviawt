"""diffusion.photom (the rendering loss on the x0 prediction), on real local grids with the model replaced by a GSDiffusionModel
shell (conftest.photom_setup): the fit's own x0 renders much better than a noised one, the gradient reaches x0_hat, and the gate
(t_range on the max over layers) works."""
import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.data]


@pytest.fixture
def setup(photom_setup):
  return photom_setup


def test_fit_renders_better_than_noise_and_gradient_flows(setup):
  s = setup
  t_lo = torch.full((s.B, 6), 0.1, device=s.dev)
  good = float(s.photom("train", s.batch, s.x0, t_lo))
  noisy = (s.x0 + 0.5 * torch.randn_like(s.x0)).requires_grad_(True)
  bad = s.photom("train", s.batch, noisy, t_lo)
  assert good < float(bad.detach()) / 2
  bad.backward()
  assert torch.isfinite(noisy.grad).all() and noisy.grad.abs().sum() > 0


def test_gate_on_the_noisiest_layer(setup):
  s = setup
  t_hi = torch.full((s.B, 6), 0.9, device=s.dev)
  s.logged.clear()
  assert float(s.photom("train", s.batch, s.x0, t_hi)) == 0 and s.logged["train/loss/photom_frac"] == 0
  t = torch.full((s.B, 6), 0.1, device=s.dev)
  t[0, 3] = 0.9                                                     # one noisy layer closes volume 0
  s.logged.clear()
  s.photom("train", s.batch, s.x0, t)
  assert abs(s.logged["train/loss/photom_frac"] - (s.B - 1) / s.B) < 1e-6
