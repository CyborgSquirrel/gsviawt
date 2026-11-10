"""diffusion.x0_t_weight: none (the loss as before) | inv_t2 (training only: each layer's x0 error times 1 / max(t, clip)^2, divided
by that weight's mean over the training timesteps). The normalizer, the weighted loss against a manual one, the unweighted part,
validations unaffected, and the config."""
import pytest
import torch

import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule
from test_cond_dropout import data, model_cfg, shell, stats


def weighted_shell(clip=0.05, uniform=True):
  m = shell()
  m.schedule = LayerAwareSchedule(m.num_layers, uniform=uniform)
  m.phase = 1
  m.x0_t_weight, m.x0_t_weight_clip = "inv_t2", clip
  m.x0_t_weight_norm = gd.x0_t_weight_mean(m.schedule, 1, clip)
  return m


def test_normalizer_matches_the_uniform_closed_form():
  for clip in (0.05, 0.2):
    got = gd.x0_t_weight_mean(LayerAwareSchedule(6, uniform=True), 1, clip)
    assert abs(got - (2 / clip - 1)) / (2 / clip - 1) < 0.02     # E[1 / max(t, c)^2] = 2/c - 1 for t ~ U[0, 1]


def test_weighted_loss_is_the_manual_one_and_unweighted_is_kept():
  m = weighted_shell()
  feats, hit, x0_raw = data()
  t = torch.tensor([[1.0, 0.5, 0.2, 0.1, 0.02, 0.7]]).expand(x0_raw.shape[0], -1).contiguous()
  seeds = list(range(x0_raw.shape[0]))
  with torch.no_grad():
    plain, parts0, _, x0_hat, _ = m.training_loss(feats, hit, x0_raw, seeds=seeds, t=t, return_pred=True)
    wl, parts, _ = m.training_loss(feats, hit, x0_raw, seeds=seeds, t=t, weighted=True)
  assert "unweighted" not in parts0
  torch.testing.assert_close(parts["unweighted"], plain)
  x0 = m.normalize(x0_raw)
  a = hit[:, :, None].float()
  w = t.clamp_min(0.05).pow(-2) / m.x0_t_weight_norm
  manual = ((x0_hat.float() - x0).square() * a * w[:, :, None, None, None]).sum() / (a.sum() * gd.CHANNELS)
  torch.testing.assert_close(wl, manual)
  for k in parts0:                                                  # the per-field / per-bucket parts stay unweighted
    if k != "unweighted":
      torch.testing.assert_close(parts[k], parts0[k])


def test_none_and_unweighted_calls_are_unchanged():
  m = weighted_shell()
  feats, hit, x0_raw = data()
  seeds = list(range(x0_raw.shape[0]))
  with torch.no_grad():
    a = m.training_loss(feats, hit, x0_raw, seeds=seeds)[0]
    m.x0_t_weight = "none"
    b = m.training_loss(feats, hit, x0_raw, seeds=seeds, weighted=True)[0]
  torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_config():
  assert gd.GSDiffusionModel(model_cfg(), stats()).x0_t_weight == "none"
  sched = {"standard": {"mu": 0.0, "sigma": 1.0}, "plateau": {"mu": 0.0, "sigma": 1.0, "mirror": False}, "uniform": True}
  m = gd.GSDiffusionModel(model_cfg(schedule=sched, x0_t_weight={"mode": "inv_t2", "clip": 0.05}), stats())
  assert m.x0_t_weight == "inv_t2" and abs(m.x0_t_weight_norm - 39.0) < 1.0
  with pytest.raises(SystemExit, match="x0_t_weight"):
    gd.GSDiffusionModel(model_cfg(x0_t_weight={"mode": "snr"}), stats())
  with pytest.raises(SystemExit, match="clip"):
    gd.GSDiffusionModel(model_cfg(x0_t_weight={"mode": "inv_t2", "clip": 0.0}), stats())
