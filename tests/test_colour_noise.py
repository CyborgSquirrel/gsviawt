"""diffusion.colour_noise: separate noise levels for the colour (sh_dc) and the other channels. The draws (t_c <= t_g, the marginals,
seeding), the channel-group-wise noising of training_loss, the time embedding of (t_g, t_c), the joint default unchanged, and the
sampler's colour schedules (colour finished at the right step and then fixed, t_c <= t_g along the path, exact Euler)."""
import math

import pytest
import torch

import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule, normalize_colour_noise, sample_colour_geo

L, PS, GRID, FEAT = 6, 2, 6, 64
H = GRID * PS
COL = gd.X0_SLICES["sh_dc"]
GEO = torch.ones(gd.CHANNELS, dtype=torch.bool)
GEO[COL] = False


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def shell(head, separate=False, draw="cond_uniform", steps=4):
  """A GSDiffusionModel around `head` with the backbone and statistics stubbed out (the "image" is the features, x0 normalized)."""
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.head, m.patch_size, m.num_layers, m.token_drop, m.eval_seed, m.steps, m.phase = head, PS, L, "none", 0, steps, 1
  m._feats = lambda f: f
  m.normalize = lambda x0_raw: x0_raw.permute(0, 2, 1, 3, 4)
  m.schedule = LayerAwareSchedule(L)
  m.colour_noise = normalize_colour_noise({"mode": "separate" if separate else "joint", "draw": draw})
  m.colour_separate = separate
  m.register_buffer("x0_mean", torch.zeros(gd.CHANNELS))
  m.register_buffer("x0_std", torch.ones(gd.CHANNELS))
  m.register_buffer("x0_lo", torch.full((gd.CHANNELS,), -1e6))
  m.register_buffer("x0_hi", torch.full((gd.CHANNELS,), 1e6))
  return m


class Recorder(torch.nn.Module):
  """Records every call's (x_net, t); predicts `value(call index)` everywhere."""
  def __init__(self, value=lambda k: 0.25):
    super().__init__()
    self.calls, self.value = [], value

  def forward(self, feats, x_net, t, pack=None, uncond=None):
    self.calls.append((x_net.clone(), t.clone()))
    return torch.full_like(x_net, float(self.value(len(self.calls) - 1)))


def data(b=2, seed=0):
  g = torch.Generator().manual_seed(seed)
  feats = torch.randn(b, L, GRID * GRID, FEAT, generator=g)
  x0_raw = torch.randn(b, gd.CHANNELS, L, H, H, generator=g)
  hit = torch.rand(b, L, H, H, generator=g) < 0.6
  return feats, hit, x0_raw


def real_head(separate, seed=0):
  torch.manual_seed(seed)
  head = gd.DiffusionHead(FEAT, L, patch_size=PS, colour_separate=separate)
  g = torch.Generator().manual_seed(seed + 1)
  with torch.no_grad():
    for p in head.parameters():
      p.copy_(0.05 * torch.randn(p.shape, generator=g))
  return head.eval()


# ---------------------------------------------------------------- config


def test_normalize_colour_noise():
  assert normalize_colour_noise(None) == {"mode": "joint", "draw": "cond_uniform"}
  assert normalize_colour_noise({"mode": "separate", "draw": "sorted"}) == {"mode": "separate", "draw": "sorted"}
  with pytest.raises(ValueError):
    normalize_colour_noise({"mode": "both"})
  with pytest.raises(ValueError):
    normalize_colour_noise({"mode": "separate", "draw": "normal"})


# ---------------------------------------------------------------- the draws


@pytest.mark.parametrize("draw", ["cond_uniform", "sorted"])
def test_draw_order_and_shape(draw):
  t = sample_colour_geo(5000, L, draw, generator=torch.Generator().manual_seed(0))
  assert t.shape == (5000, L, 2) and t.dtype == torch.float32
  tg, tc = t[..., 0], t[..., 1]
  assert (tc <= tg).all() and (tc >= 0).all() and (tg <= 1).all() and (tg > 0).all()
  assert (tc < tg).float().mean() > 0.99          # not degenerate
  assert t[:, 0].std() > 0 and not torch.equal(t[:, 0], t[:, 1])   # per layer


def _ks(sample, cdf):
  s = sample.double().flatten().sort().values
  n = len(s)
  emp = torch.arange(1, n + 1, dtype=torch.float64) / n
  return float((emp - cdf(s)).abs().max()), 1.63 / math.sqrt(n)   # the 1 % critical value


@pytest.mark.parametrize("draw,cdf_g,cdf_c", [
  ("cond_uniform", lambda x: x, lambda x: x - x * torch.log(x.clamp_min(1e-300))),   # t_c density -ln t
  ("sorted", lambda x: x ** 2, lambda x: 1 - (1 - x) ** 2),                           # densities 2 t_g, 2 (1 - t_c)
])
def test_draw_marginals(draw, cdf_g, cdf_c):
  t = sample_colour_geo(20000, 2, draw, generator=torch.Generator().manual_seed(1))
  for sample, cdf in ((t[..., 0], cdf_g), (t[..., 1], cdf_c)):
    d, crit = _ks(sample, cdf)
    assert d < crit, (draw, d, crit)


def test_eval_noise_seeded_separate_and_joint_unchanged():
  m = shell(Recorder(), separate=True)
  shape = (3, L, gd.CHANNELS, H, H)
  t1, e1 = m.eval_noise([4, 5, 6], shape, "cpu")
  t2, e2 = m.eval_noise([6, 4], (2, *shape[1:]), "cpu")
  assert t1.shape == (3, L, 2)
  torch.testing.assert_close(t2[0], t1[2], rtol=0, atol=0)   # a function of the volume's seed only
  torch.testing.assert_close(t2[1], t1[0], rtol=0, atol=0)
  torch.testing.assert_close(e2[1], e1[0], rtol=0, atol=0)
  assert (t1[..., 1] <= t1[..., 0]).all()
  j = shell(Recorder())
  tj, ej = j.eval_noise([4, 5, 6], shape, "cpu")
  ref = torch.cat([LayerAwareSchedule(L).sample(1, phase=1, generator=torch.Generator().manual_seed(s)) for s in (4, 5, 6)])
  assert tj.shape == (3, L) and torch.equal(tj, ref)
  assert torch.equal(ej, e1)                                   # the noise itself does not depend on the mode


# ---------------------------------------------------------------- training_loss


@pytest.mark.parametrize("seeded", [True, False])
def test_training_loss_noises_colour_with_t_c(seeded):
  feats, hit, x0_raw = data()
  rec = Recorder()
  m = shell(rec, separate=True, draw="sorted")
  torch.manual_seed(3)
  out = m.training_loss(feats, hit, x0_raw, seeds=[7, 8] if seeded else None, return_pred=True)
  x_net, t_in = rec.calls[0]
  assert t_in.shape == (2, L, 2)
  tg, tc = t_in[..., 0], t_in[..., 1]
  assert (tc <= tg).all() and torch.equal(out[4], tg)          # the returned t (the rendering loss's gate): t_g
  x0 = x0_raw.permute(0, 2, 1, 3, 4)
  if seeded:
    _, eps = m.eval_noise([7, 8], x0.shape, "cpu")
  else:   # the same global stream: the draw, then the noise
    torch.manual_seed(3)
    sample_colour_geo(2, L, "sorted")
    eps = torch.randn(x0.shape)
  a = hit[:, :, None].float()
  for sel, tt in ((COL, tc), (GEO, tg)):
    t5 = tt[:, :, None, None, None]
    ref = a * ((1 - t5) * x0 + t5 * eps) + (1 - a) * eps
    torch.testing.assert_close(x_net[:, :, sel], ref[:, :, sel])


def test_training_loss_given_shared_t_is_both():
  feats, hit, x0_raw = data()
  rec = Recorder()
  m = shell(rec, separate=True)
  t = torch.full((2, L), 0.3)
  m.training_loss(feats, hit, x0_raw, seeds=[7, 8], t=t)
  torch.testing.assert_close(rec.calls[0][1], torch.stack([t, t], -1))


def test_joint_training_loss_matches_the_reference_formula():
  """Joint mode: the head sees (B, L) and every channel at t (the pre-colour_noise formula)."""
  feats, hit, x0_raw = data()
  rec = Recorder()
  m = shell(rec)
  m.training_loss(feats, hit, x0_raw, seeds=[7, 8])
  x_net, t = rec.calls[0]
  t_ref, eps = m.eval_noise([7, 8], (2, L, gd.CHANNELS, H, H), "cpu")
  assert torch.equal(t, t_ref)
  a = hit[:, :, None].float()
  t5 = t_ref[:, :, None, None, None]
  x0 = x0_raw.permute(0, 2, 1, 3, 4)
  assert torch.equal(x_net, a * ((1 - t5) * x0 + t5 * eps) + (1 - a) * eps)


# ---------------------------------------------------------------- the time embedding


def test_head_time_embedding_joint_unchanged_and_separate_sees_both():
  joint = gd.DiffusionHead(FEAT, L, patch_size=PS)
  assert joint.time_embed[0].in_features == joint.t_embed_channels   # the old parameters: old checkpoints load
  assert set(joint.state_dict()) == set(gd.DiffusionHead(FEAT, L, patch_size=PS, colour_separate=False).state_dict())
  sep = real_head(True)
  assert sep.time_embed[0].in_features == 2 * sep.t_embed_channels
  assert set(sep.state_dict()) == set(joint.state_dict())
  tg = torch.rand(2, L, generator=torch.Generator().manual_seed(0))
  with torch.no_grad():
    base = sep.adaln(torch.stack([tg, tg * 0.5], -1))
    assert not torch.allclose(base, sep.adaln(torch.stack([tg, tg * 0.2], -1)))   # t_c moves it
    assert not torch.allclose(base, sep.adaln(torch.stack([tg * 0.9, tg * 0.5], -1)))   # t_g moves it
    torch.testing.assert_close(sep.adaln(tg), sep.adaln(torch.stack([tg, tg], -1)), rtol=0, atol=0)   # (B, L): t_c = t_g
    feats, hit, x0_raw = data()
    x_net = x0_raw.permute(0, 2, 1, 3, 4)
    a = sep(feats, x_net, torch.stack([tg, tg * 0.5], -1))
    b = sep(feats, x_net, torch.stack([tg, tg * 0.2], -1))
    assert not torch.allclose(a, b)


def test_separate_real_head_training_and_sampling_run():
  feats, hit, x0_raw = data()
  m = shell(real_head(True), separate=True)
  loss, parts, counts = m.training_loss(feats, hit, x0_raw, seeds=[1, 2])
  assert torch.isfinite(loss)
  g = m.sample(feats, hit, seeds=[1, 2], steps=3, colour_schedule={"kind": "first", "f": 0.5})
  assert all(torch.isfinite(v).all() for v in g.values())
  vals = m.xt_probe(feats, hit, x0_raw, [1, 2], [1.0, 0.5])    # the probe: t_c = t_g
  assert torch.isfinite(vals[0.5]["err_model"]).all()


# ---------------------------------------------------------------- the sampler


def _given_t(rec):
  return [(float(t[0, 0, 0]), float(t[0, 0, 1])) for _, t in rec.calls]


@pytest.mark.parametrize("steps,f,done_at", [(1, "1/N", 1), (5, "1/N", 1), (8, "1/N", 1), (8, 0.25, 2), (8, 0.5, 4), (10, 0.25, 3),
                                             (3, 0.5, 2), (4, 1.0, 4)])
def test_first_schedule_finishes_colour_at_the_right_step(steps, f, done_at):
  """{first, f}: t_c = max(0, 1 - (i / N) / f); colour finished after step ceil(f N) (f = 1/N: after step 1); from then on the model
  gets t_c = 0 and the colour is fixed at the prediction of the step that finished it."""
  feats, hit, _ = data()
  rec = Recorder(value=lambda k: 0.1 * (k + 1))                 # a different prediction at every call
  m = shell(rec, separate=True)
  g = m.sample(feats, hit, seeds=[1, 2], steps=steps, colour_schedule={"kind": "first", "f": f})
  given = _given_t(rec)
  ff = 1.0 / steps if f == "1/N" else f
  for i, (tg, tc) in enumerate(given):
    assert tg == pytest.approx(1 - i / steps, abs=1e-6)
    assert tc == pytest.approx(max(0.0, 1 - (i / steps) / ff), abs=1e-6) and tc <= tg + 1e-7
    assert (tc == 0) == (i >= done_at)
  # every call after the colour finished sees the colour state fixed at the prediction of the step that finished it (call
  # done_at - 1 predicts 0.1 * done_at)
  for x_net, _ in rec.calls[done_at:]:
    col = x_net[:, :, COL][hit[:, :, None].expand(-1, -1, 3, -1, -1)]
    torch.testing.assert_close(col, torch.full_like(col, 0.1 * done_at))
  assert all(torch.isfinite(v).all() for v in g.values())


@pytest.mark.parametrize("gamma", [1.0, 2.0, 4.0])
def test_power_schedule(gamma):
  feats, hit, _ = data()
  rec = Recorder()
  m = shell(rec, separate=True)
  m.sample(feats, hit, seeds=[1, 2], steps=6, colour_schedule={"kind": "power", "gamma": gamma})
  for i, (tg, tc) in enumerate(_given_t(rec)):
    assert tg == pytest.approx(1 - i / 6, abs=1e-6) and tc == pytest.approx(tg ** gamma, abs=1e-6) and tc <= tg + 1e-7


@pytest.mark.parametrize("delta", [0.0, 0.3, 0.5])
def test_lead_schedule(delta):
  feats, hit, _ = data()
  rec = Recorder()
  m = shell(rec, separate=True)
  m.sample(feats, hit, seeds=[1, 2], steps=10, colour_schedule={"kind": "lead", "delta": delta})
  for i, (tg, tc) in enumerate(_given_t(rec)):
    want = 1.0 if i == 0 else max(0.0, tg - delta)
    assert tg == pytest.approx(1 - i / 10, abs=1e-6) and tc == pytest.approx(want, abs=1e-6) and tc <= tg + 1e-7


@pytest.mark.parametrize("cs", [None, {"kind": "first", "f": 0.25}, {"kind": "power", "gamma": 2}, {"kind": "lead", "delta": 0.3}])
def test_exact_euler_lands_on_a_constant_prediction(cs):
  """Exact Euler (t_min 0, the default): a constant x0 prediction is the sample exactly, at any step count, for each group."""
  feats, hit, _ = data()
  m = shell(Recorder(), separate=True)
  for steps in (1, 7, 50):
    x0 = m._sample_separate(feats, hit, [1, 2], steps, None, None, None, None, 0.0, None, gd.normalize_colour_schedule(cs))
    torch.testing.assert_close(x0["raw_scale"], torch.full_like(x0["raw_scale"], 0.25))
  g = m.sample(feats, hit, seeds=[1, 2], steps=50, colour_schedule=cs)
  torch.testing.assert_close(g["raw_scale"], torch.full_like(g["raw_scale"], 0.25))


def test_colour_schedule_validation_and_joint_models():
  for bad in ({"kind": "first", "f": 0.0}, {"kind": "first", "f": 1.5}, {"kind": "power", "gamma": 0.5}, {"kind": "geometry_first"},
              {"kind": "lead", "delta": -0.1}, {"kind": "lead", "delta": 1.0}):
    with pytest.raises(ValueError):
      gd.normalize_colour_schedule(bad)
  assert gd.normalize_colour_schedule(None) == {"kind": "power", "gamma": 1.0}
  feats, hit, _ = data()
  with pytest.raises(ValueError):
    shell(Recorder()).sample(feats, hit, seeds=[1, 2], steps=2, colour_schedule={"kind": "power", "gamma": 2})
  rec = Recorder()
  shell(rec).sample(feats, hit, seeds=[1, 2], steps=3)          # joint: the head still gets (B, L)
  assert all(t.shape == (2, L) for _, t in rec.calls)


def test_one_step_sample_is_the_same_for_every_schedule():
  feats, hit, _ = data()
  m = shell(real_head(True), separate=True)
  ref = m.sample(feats, hit, seeds=[1, 2], steps=1)
  for cs in ({"kind": "first", "f": "1/N"}, {"kind": "first", "f": 0.25}, {"kind": "power", "gamma": 4}, {"kind": "lead", "delta": 0.5}):
    g = m.sample(feats, hit, seeds=[1, 2], steps=1, colour_schedule=cs)
    assert all(torch.equal(g[k], ref[k]) for k in ref)


def test_lock_gives_both_zero():
  feats, hit, x0_raw = data()
  rec = Recorder()
  m = shell(rec, separate=True)
  m.sample(feats, hit, seeds=[1, 2], steps=3, lock=(x0_raw.permute(0, 2, 1, 3, 4), [0]),
           colour_schedule={"kind": "first", "f": 0.5})
  for _, t in rec.calls:
    assert (t[:, 0] == 0).all() and (t[:, 1:, 1] <= t[:, 1:, 0]).all()


def test_colour_schedule_ts_endpoints():
  for cs in ({"kind": "first", "f": "1/N"}, {"kind": "first", "f": 0.3}, {"kind": "power", "gamma": 2.0}, {"kind": "lead", "delta": 0.3}):
    for n in (1, 2, 7, 50):
      tg, tc = gd.colour_schedule_ts(gd.normalize_colour_schedule(cs), n)
      assert tg[0] == 1 and tc[0] == 1 and tg[-1] == 0 and tc[-1] == 0 and (tc <= tg).all()



# ---------------------------------------------------------------- the config (GSDiffusionModel.__init__)


def test_config_keys_and_old_checkpoints():
  from tests.test_cond_dropout import model_cfg, stats
  torch.manual_seed(0)
  old = gd.GSDiffusionModel(model_cfg(), stats())                                   # a config without the keys: joint, frame
  assert not old.colour_separate and old.head.trunk.film_mode == "frame"
  joint = gd.GSDiffusionModel(model_cfg(colour_noise={"mode": "joint", "draw": "sorted"}, film="frame"), stats())
  joint.load_state_dict(old.state_dict(), strict=True)
  sep = gd.GSDiffusionModel(model_cfg(colour_noise={"mode": "separate", "draw": "sorted"}), stats())
  assert sep.colour_separate and sep.colour_noise["draw"] == "sorted"
  assert sep.head.time_embed[0].in_features == 2 * old.head.time_embed[0].in_features
  every = gd.GSDiffusionModel(model_cfg(film="every"), stats())
  assert len(every.head.trunk.film_heads) == 3
  sched = {"standard": {"mu": 0.0, "sigma": 1.0}, "plateau": {"mu": 0.0, "sigma": 1.0, "mirror": False}}
  with pytest.raises(SystemExit, match="phase=1"):
    gd.GSDiffusionModel(model_cfg(colour_noise={"mode": "separate"}, schedule={**sched, "phase": 2}), stats())
