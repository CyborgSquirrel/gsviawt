"""diffusion.probe, the x_t reliance probe (GSDiffusionModel.xt_probe, GSLightningModule._diffusion_probe / _log_probe): its
values on heads whose answer is known (one that unshrinks x_t, one that ignores it), err_copy's closed form on unit-variance data,
the gain's independence of gain_delta, the packed (token_drop) head call, seeding, and the validation wiring (the first `volumes`
volumes per set, nothing logged or run when off, the validation's own numbers unchanged). GSDiffusionModel shells, no backbone."""
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import gs_diffusion as gd
import train_gs

L, PS, GRID, FEAT = 6, 2, 6, 64          # 6 layers, 6x6 patches of 2x2 slots (the real head: in_dim 64 + 14 * 4 -> dim 128)
H = GRID * PS
TS = [1.0, 0.5, 0.2, 0.1, 0.02]


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def shell(head, token_drop="none"):
  """A GSDiffusionModel around `head` (the real eval_noise / _net_input / pack / _head / xt_probe / training_loss), with the backbone
  and the statistics stubbed out: the "image" is the features already, x0 is normalized already."""
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.head, m.patch_size, m.num_layers, m.token_drop, m.eval_seed, m.steps = head, PS, L, token_drop, 0, 4
  m._feats = lambda feats: feats
  m.normalize = lambda x0_raw: x0_raw.permute(0, 2, 1, 3, 4)
  m.schedule = types.SimpleNamespace(sample=lambda b, phase, generator=None: torch.rand(b, L, generator=generator), fixed_t=None)
  return m


class Unshrink(torch.nn.Module):
  """x0_hat = x_net / (1 - t): exact at t -> 0 (the noisy input itself), linear in x_net with slope 1 / (1 - t)."""
  def forward(self, feats, x_net, t, pack=None):
    return x_net / (1 - t)[:, :, None, None, None]


class ImageOnly(torch.nn.Module):
  """A prediction from the image alone (an image -> 3DGS lookup): x_net is never read."""
  def forward(self, feats, x_net, t, pack=None):
    return feats.mean() + torch.zeros_like(x_net)


def data(b=2, size=H, seed=0, full=False):
  g = torch.Generator().manual_seed(seed)
  feats = torch.randn(b, L, (size // PS) ** 2, FEAT, generator=g)
  x0_raw = torch.randn(b, gd.CHANNELS, L, size, size, generator=g)                # unit variance, as normalized data
  hit = torch.ones(b, L, size, size, dtype=torch.bool) if full else torch.rand(b, L, size, size, generator=g) < 0.6
  return feats, hit, x0_raw, [11 + i for i in range(b)]


def real_head(seed=0):
  """A tiny DiffusionHead with a non-trivial trunk and output (the init's zero output and LayerScale 1e-4 would make it constant)."""
  torch.manual_seed(seed)
  head = gd.DiffusionHead(FEAT, L, patch_size=PS)
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for blk in head.trunk.blocks:
      blk.ls1.fill_(1.0)
      blk.ls2.fill_(1.0)
    head.trunk.film[-1].weight.copy_(0.02 * torch.randn(head.trunk.film[-1].weight.shape, generator=g))
    torch.nn.init.normal_(head.out.weight, std=0.02)
  return head.eval()


def test_unshrink_head_uses_x_t():
  """x0_hat = x_t / (1 - t): err_model = t^2 / (1 - t)^2 mean(eps^2), err_copy ~ t^2 / D (D = (1 - t)^2 + t^2) on unit-variance
  data, so ratio ~ D / (1 - t)^2 = 1 + t^2 / (1 - t)^2 (1.0004 at t = 0.02); the head is linear in x_t: gain = 1 / (1 - t) exactly."""
  feats, hit, x0_raw, seeds = data(size=32)
  ts = [t for t in TS if t < 1]   # (t = 1 divides by zero)
  out = shell(Unshrink()).xt_probe(feats, hit, x0_raw, seeds, ts)
  for t in ts:
    v = out[t]
    torch.testing.assert_close(v["gain"], torch.full((2,), 1 / (1 - t)), rtol=1e-4, atol=0)
    ratio = float(v["err_model"].mean() / v["err_copy"].mean())
    assert abs(ratio - (1 + t ** 2 / (1 - t) ** 2)) < 0.05 * (1 + t ** 2 / (1 - t) ** 2), (t, ratio)
  assert abs(float(out[0.02]["err_model"].mean()) - 0.02 ** 2 / 0.98 ** 2) < 0.05 * 0.02 ** 2 / 0.98 ** 2


def test_image_only_head_has_no_gain():
  feats, hit, x0_raw, seeds = data()
  out = shell(ImageOnly()).xt_probe(feats, hit, x0_raw, seeds, TS)
  for t in TS:
    assert torch.equal(out[t]["gain"], torch.zeros(2))
  assert float(out[0.02]["err_model"].mean() / out[0.02]["err_copy"].mean()) > 1000   # ~ 1 / 4e-4
  assert float(out[1.0]["err_model"].mean() / out[1.0]["err_copy"].mean()) == pytest.approx(1.0, rel=0.1)   # c(1) = 0: both ~ x0^2


def test_err_copy_closed_form():
  """The error of c(t) x_t on unit-variance x0: (c (1 - t) - 1)^2 + c^2 t^2 = t^2 / ((1 - t)^2 + t^2)."""
  feats, hit, x0_raw, seeds = data(size=32, full=True)
  out = shell(ImageOnly()).xt_probe(feats, hit, x0_raw, seeds, TS)
  for t in TS:
    want = t ** 2 / ((1 - t) ** 2 + t ** 2)
    assert abs(float(out[t]["err_copy"].mean()) - want) < 0.03 * want, t


AMBIENT = [
  pytest.param("cpu", None, "none", id="fp32"),
  pytest.param("cpu", torch.bfloat16, "none", id="cpu-bf16-autocast"),
  pytest.param("cuda", torch.bfloat16, "empty", id="cuda-bf16-autocast-packed", marks=pytest.mark.cuda),
]


@pytest.mark.parametrize("device, autocast, token_drop", AMBIENT)
def test_gain_does_not_depend_on_gain_delta(device, autocast, token_drop):
  """A smooth head is locally linear: the gain at half / double the perturbation (the same direction) is the same -- also under
  the validation's bf16 autocast, which the gain's two forwards turn off (in bf16 a perturbation of 0.005-0.02 would be mostly
  rounding; cuda + empty: the err_model forward through the varlen flash kernel, the fp32 pair through the SDPA fallback)."""
  feats, hit, x0_raw, seeds = (x.to(device) if torch.is_tensor(x) else x for x in data())
  m = shell(real_head().to(device), token_drop)
  ctx = torch.autocast(device_type=device, dtype=autocast) if autocast is not None else torch.autocast(device, enabled=False)
  with ctx:
    g = {d: m.xt_probe(feats, hit, x0_raw, seeds, TS, gain_delta=d) for d in (0.005, 0.01, 0.02)}
  for t in TS:
    assert float(g[0.01][t]["gain"].abs().min()) > 1e-4, t          # not vacuous
    for d in (0.005, 0.02):
      torch.testing.assert_close(g[d][t]["gain"], g[0.01][t]["gain"], rtol=2e-2, atol=1e-5)
      assert torch.equal(g[d][t]["err_model"], g[0.01][t]["err_model"])   # only the gain sees it


@pytest.mark.parametrize("mode", ["empty", "empty_except_layer0"])
def test_packed_head_call(mode):
  """token_drop: the probe calls the head as training_loss and the sampler do (with the pack): its err_model is training_loss's
  validation loss at the same shared t; with every token kept the packed probe equals the dense one."""
  head = real_head()
  feats, hit, x0_raw, seeds = data()
  hit[1, 4:] = False                                                    # layers without a valid slot: dropped tokens
  sparse = shell(head, mode)
  out = sparse.xt_probe(feats, hit, x0_raw, seeds, TS)
  for i in range(2):
    for t in TS:
      loss = sparse.training_loss(feats[i:i + 1], hit[i:i + 1], x0_raw[i:i + 1], seeds=[seeds[i]], t=torch.full((1, L), t))[0]
      torch.testing.assert_close(out[t]["err_model"][i], loss, rtol=1e-5, atol=1e-7)
  full = torch.ones_like(hit)
  dense_out = shell(head, "none").xt_probe(feats, full, x0_raw, seeds, TS)
  sparse_out = sparse.xt_probe(feats, full, x0_raw, seeds, TS)
  for t in TS:
    for k in ("err_model", "err_copy", "gain"):
      torch.testing.assert_close(sparse_out[t][k], dense_out[t][k], rtol=1e-4, atol=1e-6, msg=f"{t} {k}")


def test_seeded_and_no_global_random_draws():
  feats, hit, x0_raw, seeds = data()
  m = shell(real_head())
  torch_state, np_state = torch.get_rng_state(), np.random.get_state()
  a = m.xt_probe(feats, hit, x0_raw, seeds, TS)
  assert torch.equal(torch.get_rng_state(), torch_state) and np.random.get_state()[1].tolist() == np_state[1].tolist()
  torch.rand(17)                                                        # the global stream elsewhere: irrelevant
  b = m.xt_probe(feats, hit, x0_raw, seeds, TS)
  for t in TS:
    for k in a[t]:
      assert torch.equal(a[t][k], b[t][k]), (t, k)
  # per volume: the same whatever batch it comes in
  c = m.xt_probe(feats[1:], hit[1:], x0_raw[1:], seeds[1:], TS)
  assert all(torch.equal(c[t][k][0], a[t][k][1]) for t in TS for k in a[t])
  # a volume without a valid slot: NaN (left out of the logged means)
  hit[0] = False
  d = m.xt_probe(feats, hit, x0_raw, seeds, TS)
  assert torch.isnan(d[0.5]["gain"][0]) and float(d[0.5]["valid"][0]) == 0 and torch.equal(d[0.5]["gain"][1], a[0.5]["gain"][1])


# ---------------------------------------------------------------------------- the validation wiring


def fake_module(probe):
  """A fake GSLightningModule (the real _diffusion_val_loss / _diffusion_probe / _log_probe / epoch hooks) on a stand-in model:
  training_loss returns a fixed loss, xt_probe records its calls and returns per-volume values (the volume's seed)."""
  M = train_gs.GSLightningModule
  diffusion = {"photom": {"enabled": False, "validate": False}, "val_t": None}
  if probe is not None:
    diffusion["probe"] = probe
  logged, calls = [], []

  def xt_probe(image, hit, x0_raw, seeds, ts, gain_delta):
    calls.append((list(seeds), list(ts), gain_delta))
    s = torch.tensor(seeds, dtype=torch.float32)
    return {t: {"err_model": s * t, "err_copy": torch.full_like(s, t), "gain": s + t, "valid": torch.ones_like(s)} for t in ts}

  model = types.SimpleNamespace(
    inputs=lambda batch, dev: (batch["image"], batch["hit"], list(batch["source_view"]), None), xt_probe=xt_probe,
    training_loss=lambda *a, **k: (torch.tensor(0.5), {"opacity": torch.tensor(0.1)}, {"opacity": 10}, None, None))
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": diffusion}), device=torch.device("cpu"), model=model,
    trainer=types.SimpleNamespace(sanity_checking=False, datamodule=types.SimpleNamespace(validation="small")),
    _log_metric=lambda stage, key, value, n: logged.append((stage, key, float(value), n)))
  for name in ("_diffusion_losses", "_diffusion_val_loss", "_diffusion_probe", "_log_probe", "_log_parts"):
    setattr(fake, name, (lambda f: lambda *a, **k: f(fake, *a, **k))(getattr(M, name)))

  def validate(sizes={"val_a": (3, 3, 3), "val_b": (2,)}):
    logged.clear(); calls.clear()
    M.on_validation_epoch_start(fake)
    view = 0
    for stage, bs in sizes.items():
      for B in bs:
        batch = {"batch_size": B, "image": torch.zeros(B), "hit": torch.zeros(B), "x0_raw": torch.zeros(B),
                 "source_view": list(range(view, view + B))}
        view += B
        fake._diffusion_val_loss(stage, batch)
    M.on_validation_epoch_end(fake)
    return list(logged), list(calls)
  validate.fake = fake
  return validate


PROBE = {"enabled": True, "ts": [1.0, 0.02], "volumes": 4, "gain_delta": 0.01}


def test_probe_takes_the_first_volumes_and_logs_once_per_set():
  logged, calls = fake_module(PROBE)()
  assert calls == [([0, 1, 2], [1.0, 0.02], 0.01), ([3], [1.0, 0.02], 0.01), ([9, 10], [1.0, 0.02], 0.01)]
  probe = {(s, k): (v, n) for s, k, v, n in logged if k.startswith("probe/")}
  assert set(probe) == {(s, f"probe/t{t}/{k}") for s in ("val_a", "val_b") for t in ("1", "0.02")
                        for k in ("err_model", "err_copy", "ratio", "gain")} | {("val_a", "probe/n_volumes"), ("val_b", "probe/n_volumes")}
  # val_a: volumes 0..3 (err_model = seed * t, err_copy = t, gain = seed + t): means over the 4
  assert probe[("val_a", "probe/t0.02/err_model")] == (pytest.approx(1.5 * 0.02), 4)
  assert probe[("val_a", "probe/t0.02/ratio")][0] == pytest.approx(1.5)
  assert probe[("val_a", "probe/t1/gain")][0] == pytest.approx(2.5)
  assert probe[("val_b", "probe/n_volumes")][0] == 2 and probe[("val_a", "probe/n_volumes")][0] == 4
  assert sum(1 for s, k, *_ in logged if k.startswith("probe/")) == len(probe)       # once per validation and set


@pytest.mark.parametrize("probe", [None, {**PROBE, "enabled": False}])
def test_disabled_runs_and_logs_nothing_and_the_rest_is_unchanged(probe):
  off_logged, off_calls = fake_module(probe)()
  on_logged, _ = fake_module(PROBE)()
  assert off_calls == [] and not any(k.startswith("probe/") for _, k, *_ in off_logged)
  assert off_logged == [e for e in on_logged if not e[1].startswith("probe/")]       # the validation's own logs as they were
  assert train_gs.probe_config(OmegaConf.create({"diffusion": {}})) is None


def test_not_in_the_sanity_check():
  validate = fake_module(PROBE)
  validate.fake.trainer.sanity_checking = True
  logged, calls = validate()
  assert calls == [] and not any(k.startswith("probe/") for _, k, *_ in logged)


def test_config_default_is_on():
  cfg = OmegaConf.load(train_gs.os.path.join(train_gs.os.path.dirname(train_gs.__file__), "conf", "train_gs.yaml"))
  p = cfg.diffusion.probe
  assert p.enabled and list(p.ts) == TS and p.volumes == 8 and p.gain_delta == 0.01
