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
  3  hidden-synced: layer 0 (the visible surface) its own t, ONE t shared by all occluded layers 1.. (layer 0 plateaued, the shared
     one standard; uniform: both uniform) -- the per-layer pattern the Modality-Forcing sampler (hidden_shift) visits.
`LayerAwareSchedule.sample(batch_size, phase)` returns (B, L) timesteps in either phase.
`uniform=True` replaces both densities by the uniform one on [0, 1] (per layer in phase 1, one shared t in phase 2): as much
training at low t (nearly clean input) as anywhere else, where the logit-normal(0, 1) puts ~1.4% of its mass below t = 0.1.

The plateau starts at the mode t* = 0.5 (mu = 0, sigma = 1) and runs towards t = 1, as specified by the user
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


COLOUR_NOISE_MODES = ("joint", "separate")
COLOUR_NOISE_DRAWS = ("cond_uniform", "sorted")


def normalize_colour_noise(cn):
  """diffusion.colour_noise (None: a config without the key) -> {"mode", "draw"}; mode joint (the default): one t per (sample,
  layer) for all channels, as before."""
  cn = {**dict(mode="joint", draw="cond_uniform"), **(dict(cn) if cn is not None else {})}
  if cn["mode"] not in COLOUR_NOISE_MODES:
    raise ValueError(f"diffusion.colour_noise.mode must be one of {COLOUR_NOISE_MODES}, got {cn['mode']!r}")
  if cn["draw"] not in COLOUR_NOISE_DRAWS:
    raise ValueError(f"diffusion.colour_noise.draw must be one of {COLOUR_NOISE_DRAWS}, got {cn['draw']!r}")
  return {"mode": str(cn["mode"]), "draw": str(cn["draw"])}


def sample_colour_geo(batch_size, num_layers, draw, generator=None, device=None):
  """diffusion.colour_noise separate: per (sample, layer) the geometry timestep t_g (every non-colour channel) and the colour one
  t_c <= t_g. Returns (B, L, 2) float32, [..., 0] = t_g, [..., 1] = t_c. From one (B, L, 2) float64 uniform draw (u, v):
    cond_uniform  t_g = u (uniform), t_c = t_g * v (density -ln t_c)
    sorted        t_g = max(u, v) (density 2 t_g), t_c = min(u, v) (density 2 (1 - t_c)): uniform over the triangle t_c <= t_g
  t_g is clamped to [T_MIN, 1 - T_MIN] (as the schedule's), t_c to [0, t_g] (t_c = 0 is a clean colour: nothing divides by it in
  training). The schedule's densities (standard / plateau / uniform) are not used."""
  uv = torch.rand((batch_size, num_layers, 2), generator=generator, device=device, dtype=torch.float64)
  u, v = uv[..., 0], uv[..., 1]
  if draw == "cond_uniform":
    t_g, t_c = u, u * v
  elif draw == "sorted":
    t_g, t_c = torch.maximum(u, v), torch.minimum(u, v)
  else:
    raise ValueError(f"draw must be one of {COLOUR_NOISE_DRAWS}, got {draw!r}")
  t_g = t_g.clamp(T_MIN, 1 - T_MIN)
  t_c = torch.minimum(t_c, t_g).clamp_min(0.0)
  return torch.stack([t_g, t_c], dim=-1).float()


class LayerAwareSchedule:
  def __init__(self, num_layers, standard=None, plateau=None, fixed_t=None, uniform=False):
    """fixed_t: every timestep is this value, in both phases (1.0: always pure noise -- the deterministic control).
    uniform: every timestep uniform on [T_MIN, 1 - T_MIN] instead of the (plateaued) logit-normal; standard / plateau unused."""
    self.num_layers = int(num_layers)
    self.fixed_t = None if fixed_t is None else float(fixed_t)
    self.uniform = bool(uniform)
    self.standard = {**dict(mu=0.0, sigma=1.0), **(standard or {})}
    self.plateau = {**dict(mu=0.0, sigma=1.0, mirror=False), **(plateau or {})}

  def densities(self, t):
    """The two base densities and their 50/50 mixture (phase 2's), on the grid t."""
    std = logit_normal_pdf(t, **self.standard)
    plat = plateau_logit_normal_pdf(t, **self.plateau)
    return {"standard": std, "plateau": plat, "mixture": 0.5 * (std + plat)}

  def sample(self, batch_size, phase, generator=None, device=None):
    """(B, L) float32 timesteps. phase 1: independent per layer (layer 0 plateaued, the rest standard);
    phase 2: one t per sample, from the 50/50 mixture, repeated over the layers; phase 3: layer 0 its own t, one t shared by layers 1.."""
    kw = dict(generator=generator, device=device)
    if self.fixed_t is not None:
      if phase not in (1, 2, 3):
        raise ValueError(f"phase must be 1, 2 or 3, got {phase!r}")
      return torch.full((batch_size, self.num_layers), self.fixed_t, dtype=torch.float32, device=device)
    if self.uniform:
      if phase not in (1, 2, 3):
        raise ValueError(f"phase must be 1, 2 or 3, got {phase!r}")
      n = {1: self.num_layers, 2: 1, 3: 2}[phase]
      t = T_MIN + (1 - 2 * T_MIN) * torch.rand((batch_size, n), generator=generator, device=device, dtype=torch.float64)
      if phase == 3:   # [layer 0, the occluded layers' shared t]
        return torch.cat([t[:, :1], t[:, 1:].expand(-1, self.num_layers - 1)], dim=1).float().contiguous()
      return t.expand(-1, self.num_layers).float().contiguous()
    if phase == 1:
      t0 = sample_plateau_logit_normal((batch_size, 1), **self.plateau, **kw)
      rest = sample_logit_normal((batch_size, self.num_layers - 1), **self.standard, **kw)
      return torch.cat([t0, rest], dim=1).float()
    if phase == 3:
      t0 = sample_plateau_logit_normal((batch_size, 1), **self.plateau, **kw)
      hid = sample_logit_normal((batch_size, 1), **self.standard, **kw)
      return torch.cat([t0, hid.expand(-1, self.num_layers - 1)], dim=1).float().contiguous()
    if phase == 2:
      pick = torch.rand((batch_size, 1), generator=generator, device=device, dtype=torch.float64) < 0.5
      t = torch.where(
        pick,
        sample_logit_normal((batch_size, 1), **self.standard, **kw),
        sample_plateau_logit_normal((batch_size, 1), **self.plateau, **kw),
      )
      return t.expand(-1, self.num_layers).float().contiguous()
    raise ValueError(f"phase must be 1, 2 or 3, got {phase!r}")
