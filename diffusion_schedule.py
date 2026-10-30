"""Layer-aware diffusion-time schedule for the Gaussian-grid diffusion mode (World Tracing's noise curriculum).

Two base densities over t in (0, 1), t = 1 pure noise:
  logit-normal          t = sigmoid(u), u ~ N(mu, sigma^2): concentrates near t = 0.5 (deeper layers).
  plateaued logit-normal  u has a flat top of half-width `half_width` around `mu` and Gaussian tails of scale `sigma`
                        (density ~ exp(-0.5 * (max(|u - mu| - half_width, 0) / sigma)^2)): a broader central
                        plateau, with `mu` < 0 putting more mass at small t (layer 0, the visible surface).
Two phases:
  1  every layer draws its own t: layer 0 from the plateaued logit-normal, layers 1.. from the logit-normal;
  2  one t per sample shared by all layers, from the equal-weight mixture of the two densities.
`LayerAwareSchedule.sample(batch_size, phase)` returns (B, L) timesteps in either phase.

The paper text does not give the plateau's parameters; the defaults here are a guess to be checked against its
density figure (see `densities` and bla/_plot_schedule.py).
"""

import math

import torch

T_MIN = 1e-4   # samples are clamped to [T_MIN, 1 - T_MIN] (the model's own clamp at 0.05 is separate, in the x0 -> v conversion)


def logit_normal_pdf(t, mu=0.0, sigma=1.0):
  """Density of t = sigmoid(u), u ~ N(mu, sigma^2)."""
  t = torch.as_tensor(t, dtype=torch.float64)
  u = torch.log(t) - torch.log1p(-t)
  return torch.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) / (t * (1 - t))


def _plateau_norm(half_width, sigma):
  return 2 * half_width + sigma * math.sqrt(2 * math.pi)


def plateau_logit_normal_pdf(t, mu=-0.5, half_width=0.75, sigma=0.75):
  """Density of t = sigmoid(u), u with density exp(-0.5 * (max(|u - mu| - half_width, 0) / sigma)^2) / Z."""
  t = torch.as_tensor(t, dtype=torch.float64)
  u = torch.log(t) - torch.log1p(-t)
  excess = (u - mu).abs().sub(half_width).clamp_min(0)
  return torch.exp(-0.5 * (excess / sigma) ** 2) / _plateau_norm(half_width, sigma) / (t * (1 - t))


def sample_logit_normal(shape, mu=0.0, sigma=1.0, generator=None, device=None):
  u = torch.randn(shape, generator=generator, device=device, dtype=torch.float64) * sigma + mu
  return torch.sigmoid(u).clamp(T_MIN, 1 - T_MIN)


def sample_plateau_logit_normal(shape, mu=-0.5, half_width=0.75, sigma=0.75, generator=None, device=None):
  """Exact sampling: the flat top (uniform on [mu - half_width, mu + half_width]) with probability
  2 * half_width / Z, otherwise a Gaussian tail |u - mu| = half_width + sigma * |z| with a random side."""
  kw = dict(generator=generator, device=device, dtype=torch.float64)
  top = torch.rand(shape, **kw) < 2 * half_width / _plateau_norm(half_width, sigma)
  inside = (torch.rand(shape, **kw) * 2 - 1) * half_width
  sign = torch.where(torch.rand(shape, **kw) < 0.5, -1.0, 1.0).to(torch.float64)
  tail = sign * (half_width + sigma * torch.randn(shape, **kw).abs())
  return torch.sigmoid(mu + torch.where(top, inside, tail)).clamp(T_MIN, 1 - T_MIN)


class LayerAwareSchedule:
  def __init__(self, num_layers, standard=None, plateau=None):
    self.num_layers = int(num_layers)
    self.standard = dict(mu=0.0, sigma=1.0, **(standard or {}))
    self.plateau = dict(mu=-0.5, half_width=0.75, sigma=0.75, **(plateau or {}))

  def densities(self, t):
    """The two base densities and their 50/50 mixture (phase 2's), on the grid t."""
    std = logit_normal_pdf(t, **self.standard)
    plat = plateau_logit_normal_pdf(t, **self.plateau)
    return {"standard": std, "plateau": plat, "mixture": 0.5 * (std + plat)}

  def sample(self, batch_size, phase, generator=None, device=None):
    """(B, L) float32 timesteps. phase 1: independent per layer (layer 0 plateaued, the rest standard);
    phase 2: one t per sample, from the 50/50 mixture, repeated over the layers."""
    kw = dict(generator=generator, device=device)
    if phase == 1:
      t0 = sample_plateau_logit_normal((batch_size, 1), **self.plateau, **kw)
      rest = sample_logit_normal((batch_size, self.num_layers - 1), **self.standard, **kw)
      return torch.cat([t0, rest], dim=1).float()
    if phase == 2:
      pick = torch.rand((batch_size, 1), generator=generator, device=device, dtype=torch.float64) < 0.5
      t = torch.where(
        pick,
        sample_logit_normal((batch_size, 1), **self.standard, **kw),
        sample_plateau_logit_normal((batch_size, 1), **self.plateau, **kw),
      )
      return t.expand(-1, self.num_layers).float().contiguous()
    raise ValueError(f"phase must be 1 or 2, got {phase!r}")
