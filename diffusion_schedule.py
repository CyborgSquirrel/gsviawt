"""Layer-aware diffusion-time schedule for the Gaussian-grid diffusion mode (World Tracing's noise curriculum).

Two base densities over t in (0, 1), t = 1 pure noise:
  logit-normal            t = sigmoid(u), u ~ N(mu, sigma^2): concentrates near t = 0.5 (deeper layers).
  plateaued logit-normal  BFL's definition (https://bfl.ai/techblog/representation-comparison/): the logit-normal density up to
                          its mode t*, then held constant at the peak value for t > t* ("remaining constant after its mode"),
                          normalized by Z = P_ln(t*) + (1 - t*) * p_ln(t*). Used for layer 0 (visible surface). With mu < 0 the
                          mode sits at small t, so more of the mass is at small t. `mirror` flips t -> 1 - t (plateau on the
                          low-noise side), in case the paper's convention is the other way round.
Two phases:
  1  every layer draws its own t: layer 0 from the plateaued logit-normal, layers 1.. from the logit-normal;
  2  one t per sample shared by all layers, from the equal-weight mixture of the two densities.
`LayerAwareSchedule.sample(batch_size, phase)` returns (B, L) timesteps in either phase.

The plateau starts at the mode t* = 0.5 (mu = 0, sigma = 1) and runs towards t = 1; this was confirmed against the paper's figure
(see `densities` and bla/_plot_schedule.py).
"""

import math

import torch

T_MIN = 1e-4   # samples are clamped to [T_MIN, 1 - T_MIN] (the model's own clamp at 0.05 is separate, in the x0 -> v conversion)


def logit_normal_pdf(t, mu=0.0, sigma=1.0):
  """Density of t = sigmoid(u), u ~ N(mu, sigma^2)."""
  t = torch.as_tensor(t, dtype=torch.float64)
  u = torch.log(t) - torch.log1p(-t)
  return torch.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) / (t * (1 - t))


def _logit(t):
  return torch.log(t) - torch.log1p(-t)


def plateau_mode(mu=0.0, sigma=1.0, iters=50):
  """Mode t* of the logit-normal density in t: the root of logit(t) - mu - sigma^2 * (2t - 1) = 0 (Newton, as in BFL's
  tpPln, for sigma = 1). Needs sigma^2 < 2 for a single mode."""
  if sigma * sigma >= 2:
    raise ValueError(f"the logit-normal has one mode only for sigma < sqrt(2), got sigma={sigma}")
  t = torch.tensor(1 / (1 + math.exp(-mu)), dtype=torch.float64)
  for _ in range(iters):
    g = _logit(t) - mu - sigma ** 2 * (2 * t - 1)
    dg = 1 / (t * (1 - t)) - 2 * sigma ** 2
    t = (t - g / dg).clamp(1e-6, 1 - 1e-6)
  return float(t)


def _plateau_consts(mu, sigma):
  t_star = plateau_mode(mu, sigma)
  p_star = float(logit_normal_pdf(t_star, mu, sigma))
  mass_below = float(0.5 * (1 + math.erf((_logit(torch.tensor(t_star, dtype=torch.float64)).item() - mu) / (sigma * math.sqrt(2)))))
  return t_star, p_star, mass_below, mass_below + (1 - t_star) * p_star   # t*, p_ln(t*), P_ln(t*), Z


def plateau_logit_normal_pdf(t, mu=0.0, sigma=1.0, mirror=False):
  """Plateaued logit-normal density: logit-normal for t <= t*, the constant p_ln(t*) for t > t*, over Z."""
  t = torch.as_tensor(t, dtype=torch.float64)
  if mirror:
    t = 1 - t
  t_star, p_star, _, z = _plateau_consts(mu, sigma)
  return torch.where(t <= t_star, logit_normal_pdf(t, mu, sigma), torch.full_like(t, p_star)) / z


def sample_logit_normal(shape, mu=0.0, sigma=1.0, generator=None, device=None):
  u = torch.randn(shape, generator=generator, device=device, dtype=torch.float64) * sigma + mu
  return torch.sigmoid(u).clamp(T_MIN, 1 - T_MIN)


def sample_plateau_logit_normal(shape, mu=0.0, sigma=1.0, mirror=False, generator=None, device=None):
  """Exact inverse-CDF sampling: below t* the CDF is P_ln(t) / Z, above it grows linearly at p_ln(t*) / Z."""
  t_star, p_star, mass_below, z = _plateau_consts(mu, sigma)
  v = torch.rand(shape, generator=generator, device=device, dtype=torch.float64)
  below = torch.sigmoid(mu + sigma * torch.special.ndtri((v * z).clamp(1e-12, 1 - 1e-12)))
  above = t_star + (v - mass_below / z) * z / p_star
  t = torch.where(v < mass_below / z, below, above).clamp(T_MIN, 1 - T_MIN)
  return 1 - t if mirror else t


class LayerAwareSchedule:
  def __init__(self, num_layers, standard=None, plateau=None):
    self.num_layers = int(num_layers)
    self.standard = dict(mu=0.0, sigma=1.0, **(standard or {}))
    self.plateau = dict(mu=0.0, sigma=1.0, mirror=False, **(plateau or {}))

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
