"""fit_3dgs.render's color_floor: hard = the SH path exactly as before; leaky = slope*rgb below 0 with a gradient there;
noise = clamp_min(rgb + sigma*|n|, 0) in training renders, == hard at sigma 0; the sigma schedule."""
import pytest
import torch
import torch.nn.functional as F

import fit_3dgs
from module import SH_C0


def old_render(params, active_sh_degree, viewmats, Ks, width, height):
  """fit_3dgs.render before color_floor existed, verbatim."""
  import gsplat
  colors = torch.cat([params["sh0"], params["shN"]], dim=1)
  rgb, alpha, info = gsplat.rasterization(
    means=params["means"], quats=F.normalize(params["quats"], dim=-1),
    scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"]),
    colors=colors, sh_degree=active_sh_degree,
    viewmats=viewmats, Ks=Ks, width=width, height=height,
    render_mode="RGB", packed=True, backgrounds=None,
  )
  return rgb.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0), info


def scene(rgb, seed=0):
  """len(rgb) Gaussians spread in front of an identity camera (z = 3), DC colour `rgb` (P,3), shN empty."""
  g = torch.Generator().manual_seed(seed)
  n = len(rgb)
  means = torch.cat([torch.rand(n, 2, generator=g) * 1.6 - 0.8, torch.full((n, 1), 3.0)], dim=1)
  params = {
    "means": means, "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
    "scales": torch.full((n, 3), -1.5), "opacities": torch.full((n,), 1.0),
    "sh0": ((torch.as_tensor(rgb, dtype=torch.float32) - 0.5) / SH_C0)[:, None, :], "shN": torch.zeros(n, 0, 3),
  }
  params = {k: v.cuda().requires_grad_(True) for k, v in params.items()}
  W = H = 64
  K = torch.tensor([[[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]]]).cuda()
  return params, torch.eye(4)[None].cuda(), K, W, H


def grads(render_fn, params, *args, **kwargs):
  for p in params.values():
    p.grad = None
  rgb, _, _ = render_fn(params, 0, *args, **kwargs)
  (rgb * torch.linspace(0.5, 1.5, rgb.numel(), device=rgb.device).view_as(rgb)).sum().backward()
  return rgb.detach(), {k: p.grad.clone() for k, p in params.items() if p.grad is not None}


def mixed_colours(n=64, seed=1):
  g = torch.Generator().manual_seed(seed)
  rgb = torch.rand(n, 3, generator=g) * 1.4 - 0.2          # ~14% below 0, some above 1
  rgb[:4] = 0.0                                            # exactly-black seeds: -5.96e-8 after the SH round-trip
  return rgb


@pytest.mark.cuda
def test_hard_is_the_old_render():
  params, viewmat, K, W, H = scene(mixed_colours())
  ref, gref = grads(old_render, params, viewmat, K, W, H)
  out, gout = grads(fit_3dgs.render, params, viewmat, K, W, H)                       # default: hard
  assert torch.equal(out, ref)
  # gradients: equal up to gsplat's own run-to-run backward nondeterminism (atomic adds; ~1e-7 relative between two
  # calls of old_render itself), and exactly zero in the same places
  assert gout.keys() == gref.keys()
  for k in gref:
    assert torch.allclose(gout[k], gref[k], rtol=1e-5, atol=1e-6 * float(gref[k].abs().max()) if gref[k].numel() else 0)
    assert torch.equal(gout[k] == 0, gref[k] == 0)
  out, _ = grads(fit_3dgs.render, params, viewmat, K, W, H, color_floor="noise", color_noise_sigma=0.0)
  assert torch.equal(out, ref)                                                         # noise at sigma 0 == hard


@pytest.mark.cuda
def test_leaky_matches_hard_above_floor_and_has_gradient_below():
  rgb = mixed_colours()
  rgb[8] = torch.tensor([-0.1, 0.4, 0.4])                 # one clearly negative channel
  params, viewmat, K, W, H = scene(rgb)
  hard, ghard = grads(fit_3dgs.render, params, viewmat, K, W, H)
  leaky, gleaky = grads(fit_3dgs.render, params, viewmat, K, W, H, color_floor="leaky", color_floor_slope=0.01)
  assert ghard["sh0"][8, 0, 0] == 0                        # the dead zone this exists to fix
  assert gleaky["sh0"][8, 0, 0] != 0
  assert ghard["sh0"][0].abs().sum() == 0 and gleaky["sh0"][0].abs().sum() > 0   # exactly-black seed: dead vs alive
  assert (leaky - hard).abs().max() < 0.01                 # below-floor colours only render as 1% of themselves
  lin = SH_C0 * params["sh0"][:, 0] + 0.5
  expect = torch.where(lin < 0, 0.01 * lin, lin)
  assert torch.equal(fit_3dgs.leaky_floor_rgb(params["sh0"], 0.01), expect)


@pytest.mark.cuda
def test_noise_is_positive_and_revives_below_floor_colours():
  sh0 = ((torch.full((4096, 3), -0.01) - 0.5) / SH_C0)[:, None, :].cuda().requires_grad_(True)
  torch.manual_seed(0)
  out = fit_3dgs.noisy_floor_rgb(sh0, 0.05)
  lin = (SH_C0 * sh0[:, 0] + 0.5).detach()
  assert (out >= lin.clamp_min(0)).all()                   # |n| >= 0: never darker than the hard floor
  out.sum().backward()
  alive = (sh0.grad[:, 0] != 0).float().mean()
  assert 0.5 < alive < 1.0                                 # -0.01 + 0.05|n| > 0 on most, not all, draws

  rgb = mixed_colours()
  rgb[8] = torch.tensor([-0.01, 0.4, 0.4])
  params, viewmat, K, W, H = scene(rgb)
  _, ghard = grads(fit_3dgs.render, params, viewmat, K, W, H)
  assert ghard["sh0"][8, 0, 0] == 0
  hit = 0
  for s in range(8):
    torch.manual_seed(s)
    _, gnoise = grads(fit_3dgs.render, params, viewmat, K, W, H, color_floor="noise", color_noise_sigma=0.05)
    hit += int(gnoise["sh0"][8, 0, 0] != 0)
  assert hit > 0


def test_noise_sigma_schedule():
  s = fit_3dgs.color_noise_sigma
  assert s(0, 1000, 0.05, 0.5) == 0.05
  assert s(250, 1000, 0.05, 0.5) == pytest.approx(0.025)
  assert s(500, 1000, 0.05, 0.5) == 0.0 and s(999, 1000, 0.05, 0.5) == 0.0
  assert s(0, 1000, 0.05, 0.0) == 0.0


def test_unknown_floor_raises():
  with pytest.raises(ValueError):
    fit_3dgs.render({"shN": torch.zeros(1, 0, 3)}, 0, None, None, 1, 1, color_floor="soft")
