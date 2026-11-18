"""diffusion.cond_dropout (per-sample image-condition dropout in training: zeros / learned null after the z-score) and diffusion.cfg
(classifier-free guidance in the sampler: scale, interval, fields), plus GSDiffusionModel.predict_x0 / training_loss(condition=):
p = 0 is the call, the loss, the gradients and the random streams of before; p = 1 nulls every sample; the per-sample mask; the learned
null's parameter (only when asked for; it trains); guidance scale 1 = no guidance (one forward), scale 0 = the unconditional
prediction, the interval and fields gating; packed (token_drop) vs dense agree on the unconditional path; old configs and
checkpoints load with the new defaults."""
import types

import pytest
import torch
from omegaconf import OmegaConf

import gs_diffusion as gd
import train_gs
from test_token_drop import GRID, FEAT, L, P, PS, hit_masks, inputs, make_head, model_shell, ref_trunk, rel, tokens_of

H = GRID * PS
SEEDS = [5, 6, 7, 8]


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def shell(token_drop="none", p=0.0, null="zeros", seed=0):
  head = make_head(seed)
  if null == "learned":
    head2 = gd.DiffusionHead(FEAT, L, patch_size=PS, cond_null="learned")
    head2.load_state_dict(head.state_dict(), strict=False)   # the same weights, plus the (zero) null
    head = head2
  m = model_shell(head, token_drop)
  m.cond_dropout_p = p
  return m


def data(b=4, seed=1):
  feats, _, _ = inputs(None, b, seed)
  x0_raw = torch.randn(b, gd.CHANNELS, L, H, H, generator=torch.Generator().manual_seed(seed + 100))
  return feats, hit_masks()[:b], x0_raw


def train_step(m, feats, hit, x0_raw, seed=123, **kw):
  """One training-stream loss (seeds None) + backward from a fixed global seed -> (loss, parts, counts, grads, RNG state after)."""
  torch.manual_seed(seed)
  m.head.zero_grad(set_to_none=True)
  loss, parts, counts = m.training_loss(feats, hit, x0_raw, **kw)
  loss.backward()
  grads = {n: p.grad.clone() for n, p in m.head.named_parameters() if p.grad is not None}
  return loss.detach(), parts, counts, grads, torch.get_rng_state()


def record_calls(head):
  calls = []
  real = head.forward
  head.forward = lambda *a, **k: calls.append((len(a), tuple(sorted(k)), k.get("uncond"))) or real(*a, **k)
  return calls


# ---------------------------------------------------------------------------- training: condition dropout

@pytest.mark.parametrize("token_drop", ["none", "empty"])
def test_p0_is_the_old_call_loss_grads_and_rng(token_drop):
  """p = 0: the head's call of before (no uncond), nothing extra drawn (the RNG state after the step is that of the schedule + noise
  draws alone), and the loss / gradients those of the head on the old inputs, bit for bit."""
  m = shell(token_drop, p=0.0)
  feats, hit, x0_raw = data()
  calls = record_calls(m.head)
  loss, parts, counts, grads, rng = train_step(m, feats, hit, x0_raw)
  assert all(k == () for _, k, _ in calls) and "cond_drop" not in parts and "cond_kept" not in parts

  # the reference: master's training_loss body, by hand
  torch.manual_seed(123)
  t = m.schedule.sample(4, phase=1)
  x0 = x0_raw.permute(0, 2, 1, 3, 4)
  eps = torch.randn(x0.shape)
  rng_ref = torch.get_rng_state()
  a = hit[:, :, None].float()
  t5 = t[:, :, None, None, None]
  head = m.head
  head.forward = type(head).forward.__get__(head)
  head.zero_grad(set_to_none=True)
  x0_hat = head(feats, a * ((1 - t5) * x0 + t5 * eps) + (1 - a) * eps, t, *([m.pack(hit)] if token_drop != "none" else []))
  ref = ((x0_hat - x0).square() * a).sum() / (a.sum() * gd.CHANNELS)
  ref.backward()
  assert torch.equal(rng, rng_ref)
  assert torch.equal(loss, ref.detach())
  assert set(grads) == {n for n, p in head.named_parameters() if p.grad is not None}
  for n, p in head.named_parameters():
    if p.grad is None:
      continue
    if token_drop == "none":
      assert torch.equal(grads[n], p.grad), n
    else:   # (not bitwise even run to run: the packed gathers' backward accumulates in a varying order, ~1e-11)
      torch.testing.assert_close(grads[n], p.grad, rtol=1e-6, atol=1e-9, msg=n)


def test_seeded_calls_never_drop():
  """Validations (seeds given) stay conditional whatever p, and draw nothing from the global stream."""
  m = shell(p=1.0)
  feats, hit, x0_raw = data()
  calls = record_calls(m.head)
  state = torch.get_rng_state()
  loss, parts, _ = m.training_loss(feats, hit, x0_raw, seeds=SEEDS)
  assert torch.equal(torch.get_rng_state(), state) and all(k == () for _, k, _ in calls) and "cond_drop" not in parts
  m0 = shell(p=0.0)
  assert torch.equal(loss, m0.training_loss(feats, hit, x0_raw, seeds=SEEDS)[0])


@pytest.mark.parametrize("token_drop", ["none", "empty"])
def test_p1_nulls_every_sample(token_drop):
  """p = 1 with the zeros null: the loss of the old path on all-zero features (a constant z-scores to 0, exactly the null), at the
  same timesteps and noise (the dropout draw comes after them); every sample is dropped and the loss trains."""
  m = shell(token_drop, p=1.0)
  feats, hit, x0_raw = data()
  loss, parts, counts, grads, _ = train_step(m, feats, hit, x0_raw)
  assert parts["cond_drop"].all() and int(counts["cond_kept"]) == 0 and torch.equal(parts["cond_dropped"], loss)
  assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads.values())
  m.cond_dropout_p = 0.0
  ref = train_step(m, torch.zeros_like(feats), hit, x0_raw)[0]
  torch.testing.assert_close(loss, ref, rtol=1e-6, atol=0)


@pytest.mark.parametrize("token_drop", ["none", "empty"])
def test_per_sample_mask(token_drop):
  """A mixed mask: each dropped sample's prediction is the all-null one, each kept sample's the conditional one (samples do not
  interact); the training draw's mask is returned and its kept / dropped x0 losses recombine into the loss."""
  m = shell(token_drop)
  feats, hit, _ = data()
  _, x_net, t = inputs(None, 4, seed=3)
  keep = torch.tensor([True, False, False, True])
  mixed = m.predict_x0(feats, hit, x_net, t, condition=keep)
  cond = m.predict_x0(feats, hit, x_net, t, condition=True)
  null = m.predict_x0(feats, hit, x_net, t, condition=False)
  zeroed = m.predict_x0(torch.zeros_like(feats), hit, x_net, t)
  torch.testing.assert_close(null, zeroed, rtol=1e-6, atol=1e-7)
  v = hit[:, :, None].expand_as(mixed)
  for i in range(4):
    want = cond if keep[i] else null
    torch.testing.assert_close(mixed[i][v[i]], want[i][v[i]], rtol=1e-5, atol=1e-6)
  assert not torch.allclose(cond[1][v[1]], null[1][v[1]])

  m.cond_dropout_p = 0.5
  loss, parts, counts, _, _ = train_step(m, feats, hit, torch.randn(4, gd.CHANNELS, L, H, H))
  drop = parts["cond_drop"]
  assert drop.dtype == torch.bool and drop.shape == (4,)
  nk, nd = counts["cond_kept"], counts["cond_dropped"]
  assert int(nk + nd) == int(hit.sum())
  torch.testing.assert_close((parts["cond_kept"] * nk + parts["cond_dropped"] * nd) / (nk + nd), loss, rtol=1e-5, atol=0)
  # the mask is the uniform draw after the timesteps and the noise, on x0's device
  torch.manual_seed(123)
  m.schedule.sample(4, phase=1)
  torch.randn(4, L, gd.CHANNELS, H, H)
  assert torch.equal(drop, torch.rand(4) < 0.5)


def test_learned_null_only_when_asked_and_trains():
  default = gd.DiffusionHead(FEAT, L, patch_size=PS)
  assert not hasattr(default, "null_cond") and not any("null" in k for k in default.state_dict())
  learned = gd.DiffusionHead(FEAT, L, patch_size=PS, cond_null="learned")
  assert set(learned.state_dict()) - set(default.state_dict()) == {"null_cond"}
  assert learned.null_cond.shape == (FEAT,) and (learned.null_cond == 0).all()
  with pytest.raises(ValueError):
    gd.DiffusionHead(FEAT, L, patch_size=PS, cond_null="noise")

  m = shell(p=1.0, null="learned")
  feats, hit, x0_raw = data()
  _, _, _, grads, _ = train_step(m, feats, hit, x0_raw)
  assert grads["null_cond"].abs().sum() > 0
  m.cond_dropout_p = 0.0   # nothing dropped: no gradient reaches it
  assert "null_cond" not in train_step(m, feats, hit, x0_raw)[3]


def test_learned_null_replaces_the_feature_tokens():
  """The trunk's tokens of a dropped sample: [null | the patch's noisy values] -- the null after the z-score and the scale, the same
  vector in every layer and patch, whatever the features were."""
  m = shell(null="learned")
  with torch.no_grad():
    m.head.null_cond.copy_(torch.randn(FEAT, generator=torch.Generator().manual_seed(9)))
  feats, hit, _ = data()
  _, x_net, t = inputs(None, 4, seed=3)
  seen = []
  hook = m.head.trunk.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach().clone()))
  m.predict_x0(feats, hit, x_net, t, condition=torch.tensor([False, True, True, True]))
  m.predict_x0(feats * 3 + 1, hit, x_net, t, condition=False)
  hook.remove()
  tok = seen[0]
  assert torch.equal(tok[0, ..., :FEAT], m.head.null_cond.detach().expand(L, P, FEAT))
  assert torch.equal(tok[1:], tokens_of(m.head, feats, x_net)[1:])
  assert torch.equal(seen[1][..., :FEAT], m.head.null_cond.detach().expand(4, L, P, FEAT))
  assert torch.equal(tok[0, ..., FEAT:], seen[1][0, ..., FEAT:])   # the noisy values untouched


# ---------------------------------------------------------------------------- dense vs packed, unconditional

@pytest.mark.parametrize("mode", ["empty", "empty_except_layer0"])
def test_packed_unconditional_matches_key_masked_dense(mode):
  head = make_head()
  hit = hit_masks()
  keep = gd.token_keep(hit, PS, mode)
  feats, x_net, t = inputs(head, 4)
  uncond = torch.tensor([True, False, True, True])
  with torch.no_grad():
    rows = head(feats, x_net, t, gd.PackIndex(keep), uncond=uncond)
    tokens = tokens_of(head, torch.where(uncond[:, None, None, None], torch.zeros_like(feats), feats), x_net)
    ref = head.out(ref_trunk(head.trunk, tokens, head.adaln(t), keep))
    ref = ref.reshape(4, L, GRID, GRID, gd.CHANNELS, PS, PS).permute(0, 1, 4, 2, 5, 3, 6).reshape(4, L, gd.CHANNELS, H, H)
    tok = keep.reshape(4, L, GRID, GRID)[:, :, None, :, None, :, None].expand(4, L, gd.CHANNELS, GRID, PS, GRID, PS)
    tok = tok.reshape(4, L, gd.CHANNELS, H, H)
    assert rel(rows[tok], ref[tok]) < 1e-6 and (rows[~tok] == 0).all()
    all_kept = head(feats, x_net, t, gd.PackIndex(torch.ones_like(keep)), uncond=uncond)
    dense = head(feats, x_net, t, uncond=uncond)
    assert rel(all_kept, dense) < 1e-6


# ---------------------------------------------------------------------------- sampling: guidance

def guided_shell(token_drop="none", p=0.1):
  m = shell(token_drop, p=p)
  m.guidance = gd.normalize_guidance(None)
  return m


@pytest.mark.parametrize("token_drop", ["none", "empty"])
def test_scale1_is_no_guidance(token_drop):
  m = guided_shell(token_drop)
  feats, hit, _ = data()
  calls = record_calls(m.head)
  base = m.sample(feats, hit, seeds=SEEDS)
  assert len(calls) == m.steps and all(k == () for _, k, _ in calls)
  for g in ({"scale": 1.0}, {"scale": 1, "interval": [0.2, 0.4], "fields": ["sh_dc"]}):
    calls.clear()
    assert torch.equal(m.sample(feats, hit, seeds=SEEDS, guidance=g), base) and len(calls) == m.steps


@pytest.mark.parametrize("token_drop", ["none", "empty"])
def test_scale0_is_the_unconditional_sampler(token_drop):
  m = guided_shell(token_drop)
  feats, hit, _ = data()
  calls = record_calls(m.head)
  out = m.sample(feats, hit, seeds=SEEDS, guidance={"scale": 0.0})
  assert len(calls) == m.steps and all(u is not None and u.all() for _, _, u in calls)   # one (unconditional) forward per step
  torch.testing.assert_close(out, m.sample(torch.zeros_like(feats), hit, seeds=SEEDS), rtol=1e-5, atol=1e-6)


def test_guidance_mix_interval_and_fields():
  """The guided prediction is x0_u + s (x0_c - x0_u) at the steps in the interval only, on the chosen fields only; one step from
  t = 1 returns the x0 prediction itself, so the mix is visible directly."""
  m = guided_shell()
  feats, hit, _ = data()
  # one step (t = 1): out = eps - (eps - x0_hat)
  c = m.sample(feats, hit, seeds=SEEDS, steps=1)
  u = m.sample(feats, hit, seeds=SEEDS, steps=1, guidance={"scale": 0})
  g = m.sample(feats, hit, seeds=SEEDS, steps=1, guidance={"scale": 2.5})
  torch.testing.assert_close(g, u + 2.5 * (c - u), rtol=1e-5, atol=1e-5)
  f = m.sample(feats, hit, seeds=SEEDS, steps=1, guidance={"scale": 2.5, "fields": ["sh_dc", "opacity"]})
  on = torch.zeros(gd.CHANNELS, dtype=torch.bool)
  on[gd.X0_SLICES["sh_dc"]] = on[gd.X0_SLICES["opacity"]] = True
  assert torch.equal(f[:, :, ~on], c[:, :, ~on])
  torch.testing.assert_close(f[:, :, on], g[:, :, on], rtol=0, atol=0)
  # outside the interval: the conditional sample, one forward
  calls = record_calls(m.head)
  assert torch.equal(m.sample(feats, hit, seeds=SEEDS, steps=1, guidance={"scale": 2.5, "interval": [0.0, 0.9]}), c)
  assert len(calls) == 1
  # 4 steps (t = 1, 0.75, 0.5, 0.25), interval [0.5, 1] inclusive: 3 guided steps (2 forwards each), 1 plain
  calls.clear()
  m.sample(feats, hit, seeds=SEEDS, guidance={"scale": 2.0, "interval": [0.5, 1.0]})
  kinds = ["u" if u is not None else "c" for _, _, u in calls]
  assert kinds == ["u", "c", "u", "c", "u", "c", "c"]


def test_guidance_before_project():
  """project (format_projection) sees the guided prediction."""
  m = guided_shell()
  feats, hit, _ = data()
  seen = []
  m.sample(feats, hit, seeds=SEEDS, steps=1, guidance={"scale": 0}, project=lambda z: seen.append(z) or z)
  u = m.predict_x0(feats, hit, *m.eval_noise(SEEDS, (4, L, gd.CHANNELS, H, H), "cpu")[1:], torch.ones(4, L), condition=False)
  torch.testing.assert_close(seen[0], u.float(), rtol=0, atol=0)


def test_guidance_config_and_warning():
  assert gd.normalize_guidance(None) == {"scale": 1.0, "interval": (0.0, 1.0), "fields": None}
  assert gd.normalize_guidance({"fields": ["sh_dc"]})["fields"] == ("sh_dc",)
  for bad in ({"fields": ["colour"]}, {"fields": []}, {"interval": [0.8, 0.2]}, {"rescale": 0.7}):
    with pytest.raises(ValueError):
      gd.normalize_guidance(bad)
  assert gd.normalize_cond_dropout(None) == {"p": 0.0, "null": "zeros"}
  for bad in ({"p": 1.5}, {"null": "blank"}, {"rate": 0.1}):
    with pytest.raises(ValueError):
      gd.normalize_cond_dropout(bad)
  m = guided_shell(p=0.0)
  feats, hit, _ = data()
  with pytest.warns(UserWarning, match="cond_dropout.p=0"):
    m.sample(feats, hit, seeds=SEEDS, steps=1, guidance={"scale": 2})
  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    gd.warn_guidance_without_dropout(gd.normalize_guidance({"scale": 2}), 0.1)
    gd.warn_guidance_without_dropout(gd.normalize_guidance(None), 0.0)


# ---------------------------------------------------------------------------- configs, checkpoints, train-step logging

def model_cfg(**diffusion):
  return OmegaConf.create({
    "model": {"backbone": {"_target_": "torch.nn.Identity"}, "patch_size": PS, "feature_dim": FEAT,
              "transformer": {"layer_scale_init": 1e-4}},
    "data": {"num_layers": L},
    "diffusion": {"num_blocks": 3, "img_token_scale": 0.3, "steps": 4, "eval_seed": 0,
                  "schedule": {"standard": {"mu": 0.0, "sigma": 1.0}, "plateau": {"mu": 0.0, "sigma": 1.0, "mirror": False}},
                  **diffusion}})


def stats():
  return torch.zeros(gd.CHANNELS), torch.ones(gd.CHANNELS), -torch.ones(gd.CHANNELS), torch.ones(gd.CHANNELS)


def test_old_configs_and_checkpoints_load():
  torch.manual_seed(0)
  old = gd.GSDiffusionModel(model_cfg(), stats())                  # a config from before the keys
  assert old.cond_dropout_p == 0.0 and old.guidance == gd.GUIDANCE_DEFAULTS and not hasattr(old.head, "null_cond")
  new = gd.GSDiffusionModel(model_cfg(cond_dropout={"p": 0.0, "null": "zeros"},
                                      cfg={"scale": 1.0, "interval": [0.0, 1.0], "fields": None}), stats())
  assert set(new.state_dict()) == set(old.state_dict())
  new.load_state_dict(old.state_dict(), strict=True)
  learned = gd.GSDiffusionModel(model_cfg(cond_dropout={"p": 0.1, "null": "learned"}), stats())
  missing, unexpected = learned.load_state_dict(old.state_dict(), strict=False)
  assert missing == ["head.null_cond"] and unexpected == []
  with pytest.warns(UserWarning, match="never trained"):
    gd.GSDiffusionModel(model_cfg(cfg={"scale": 0.0}), stats())


def fake_module(parts, counts, logged, photom=False):
  return types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"photom": {"enabled": photom, "weight": 1.0}}}), views_seen=0, pixels_seen=0,
    _accepted_since_step=False, spike_skip=types.SimpleNamespace(check=lambda v: False, average=None),
    _diffusion_losses=lambda batch: (torch.tensor(0.5, requires_grad=True), parts, counts, None, None),
    _photom_active=lambda: photom,
    _diffusion_photom=lambda *a, **k: logged.append(("photom", k.get("cond_drop"))) or torch.zeros(()),
    _log_metric=lambda stage, k, v, n: logged.append((k, float(v))),
    _log_parts=lambda stage, prefix, p, c: logged.extend((f"{prefix}/{k}", None) for k in p))


def test_train_step_logs_the_dropped_fraction():
  batch = {"batch_size": 4, "x0_raw": torch.zeros(1, 14, 6, 8, 8)}
  logged = []
  drop = torch.tensor([True, False, False, False])
  parts = {"cond_drop": drop, "cond_kept": torch.tensor(1.0), "cond_dropped": torch.tensor(2.0)}
  train_gs.GSLightningModule._diffusion_train_step(fake_module(parts, {}, logged, photom=True), batch)
  keys = dict(logged)
  assert keys["cond_dropped"] == 0.25 and "loss/x0/cond_kept" in keys and "loss/x0/cond_dropped" in keys
  assert "loss/x0/cond_drop" not in keys and ("photom", drop) in logged
  logged.clear()   # p = 0: no key, and the rendering loss called as before
  train_gs.GSLightningModule._diffusion_train_step(fake_module({}, {}, logged, photom=True), batch)
  assert "cond_dropped" not in dict(logged) and ("photom", None) in logged


def test_photom_split_by_condition():
  """_diffusion_photom with cond_drop: the same per-volume losses, also logged over the kept / dropped volumes that count; the
  returned loss is the one without the split."""
  M = train_gs.GSLightningModule
  per = {"l1": torch.tensor([1.0, 2.0, 3.0, 4.0]), "total": torch.tensor([1.0, 2.0, 3.0, 4.0])}
  logged = {}
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"photom": {"t_range": [0.0, 0.7], "t_reduce": "max"}}}), device="cpu",
    model=types.SimpleNamespace(decode=lambda x, floor: x, floor_context=lambda b, d: None),
    _photom_per_volume=lambda batch, gauss: per, _log_metric=lambda stage, k, v, n: logged.__setitem__(k, (float(v), n)))
  fake._log_photom = lambda *a, **k: M._log_photom(fake, *a, **k)
  fake._render_photom = lambda *a, **k: M._render_photom(fake, *a, **k)
  t = torch.tensor([[0.1], [0.2], [0.9], [0.3]])                    # volume 2 does not count
  drop = torch.tensor([True, False, True, False])
  loss = M._diffusion_photom(fake, "train", {"batch_size": 4}, None, t, cond_drop=drop)
  assert float(loss) == pytest.approx((1 + 2 + 4) / 3)
  assert logged["loss/photom_uncond"] == (1.0, 1) and logged["loss/photom_cond"] == (3.0, 2)
  assert logged["loss/photom"] == (pytest.approx(7 / 3), 3)
  logged.clear()
  assert float(M._diffusion_photom(fake, "train", {"batch_size": 4}, None, t)) == pytest.approx(7 / 3)
  assert "loss/photom_cond" not in logged


def test_schedule_uniform_key():
  assert not gd.GSDiffusionModel(model_cfg(), stats()).schedule.uniform                     # a config without the key
  sched = {"standard": {"mu": 0.0, "sigma": 1.0}, "plateau": {"mu": 0.0, "sigma": 1.0, "mirror": False}, "uniform": True}
  assert gd.GSDiffusionModel(model_cfg(schedule=sched), stats()).schedule.uniform
