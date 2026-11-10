"""The PR #97 fixes: D-SSIM in fp32 (its gradient under bf16 autocast was noise), the validation's rendering keys
(loss/photom: the training's term, gated the same way; loss/photom_t<t>: at a shared timestep from a shuffled bag of
diffusion.val_t; loss/photom_full: the full sample -- none colliding), and train/loss/x0 logged only for accepted steps."""
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import train_gs
from module import DSSIMLoss


def smooth_images(n=2, size=128, seed=0):
  g = torch.Generator().manual_seed(seed)
  x = F.interpolate(torch.rand(n, 3, size // 8, size // 8, generator=g), size=size, mode="bilinear", align_corners=False)
  return x.clamp(0, 1)


@pytest.mark.cuda
def test_dssim_gradient_under_bf16_matches_fp32():
  d = DSSIMLoss(reduction="none").cuda()
  x, y = smooth_images(seed=0).cuda(), smooth_images(seed=1).cuda()
  x32 = x.clone().requires_grad_()
  d(x32, y).mean().backward()
  xb = x.clone().requires_grad_()
  with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = d(xb, y).mean()
  loss.backward()
  assert loss.dtype == torch.float32
  cos = F.cosine_similarity(x32.grad.flatten(), xb.grad.flatten(), dim=0)
  assert cos > 0.999, f"bf16 D-SSIM gradient cosine to fp32 {float(cos):.4f}"


def test_train_x0_logged_for_accepted_steps_only():
  logged = []
  a = torch.tensor(2.0, requires_grad=True)
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"photom": {"enabled": False}}}), views_seen=0, pixels_seen=0,
    _accepted_since_step=False, spike_skip=types.SimpleNamespace(check=lambda v: True, average=None),   # every step skipped
    _diffusion_losses=lambda batch: (a * 0.5, {}, {}, None, None), _photom_active=lambda: False,
    _log_metric=lambda stage, k, v, n: logged.append(k), _log_parts=lambda *x: None)
  out = train_gs.GSLightningModule._diffusion_train_step(fake, {"batch_size": 4, "x0_raw": torch.zeros(1, 14, 6, 8, 8)})
  assert out is None and "loss/x0" not in logged and "loss" not in logged


def test_validation_photom_keys():
  """validation_step: the full sample as loss/photom_full (diffusion) / loss/photom (not); _diffusion_val_loss: the training's
  term (loss/photom) and the 1-step sample (loss/photom_1step, every volume, the sampler's own noise seeds)."""
  calls = []
  for enabled, key in ((True, "loss/photom_full"), (False, "loss/photom")):
    fake = types.SimpleNamespace(
      cfg=OmegaConf.create({"diffusion": {"enabled": enabled}}),
      trainer=types.SimpleNamespace(datamodule=types.SimpleNamespace(val_datasets={"val": None})),
      _step=lambda stage, batch, idx, photom_key: calls.append(photom_key) or {"loss": 0.0},
      _diffusion_val_loss=lambda stage, batch: None)
    train_gs.GSLightningModule.validation_step(fake, {}, 0)
    assert calls[-1] == key

  logged = []
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"photom": {"enabled": True, "validate": True}, "val_t": [0.5]}}),
    _diffusion_losses=lambda batch, seeded: (torch.tensor(1.0), {}, {}, "x0_hat", "t"),
    _log_metric=lambda *a: None, _log_parts=lambda *a: None,
    _diffusion_photom=lambda stage, batch, x0_hat, t: logged.append("photom"),
    _diffusion_val_t=lambda stage, batch: logged.append("val_t"))
  train_gs.GSLightningModule._diffusion_val_loss(fake, "val", {"batch_size": 2})
  assert logged == ["photom", "val_t"]


VAL_T = [0.25, 0.5, 0.75, 1.0]
SETS = {"val_a": (3, 4), "val_b": (5, 1)}   # batch sizes: 7 and 6 volumes, neither a multiple of len(val_t) (bags carry over)


def val_t_run(seed=42, eval_seed=0):
  """A fake GSLightningModule (the real val_t methods; a stand-in model recording the t of every forward) on a GSDataModule shell
  (the real reseeding); validate(k) runs the k-th EventValidation's val_t pass (k=None: Lightning's own next) -> {set: [t per volume]}."""
  M, D = train_gs.GSLightningModule, train_gs.GSDataModule
  dm = types.SimpleNamespace(cfg=OmegaConf.create({"seed": seed, "loader": {"batch_size": 4}}), validation="lightning",
                             val_datasets={n: list(range(sum(b))) for n, b in SETS.items()}, lightning_runs=0)
  for m in ("validation_seed", "reseed_validation", "reseed_lightning", "val_dataloader", "load_state_dict"):
    setattr(dm, m, (lambda f: lambda *a, **k: f(dm, *a, **k))(getattr(D, m)))
  dm.val_dataloader()
  forwards = []

  def training_loss(image, hit, x0_raw, seeds, return_pred, t):
    forwards.append(t[:, 0].tolist())
    return None, None, None, torch.zeros(t.shape[0]), t

  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"eval_seed": eval_seed, "val_t": VAL_T}}), device=torch.device("cpu"),
    trainer=types.SimpleNamespace(datamodule=dm, sanity_checking=False),
    model=types.SimpleNamespace(inputs=lambda batch, dev: ("img", torch.ones(batch["batch_size"], 6, 4, 4), [0] * batch["batch_size"], None,
                                                         None),
                                training_loss=training_loss, decode=lambda x, floor=None: x),
    _photom_per_volume=lambda batch, gauss: {"l1": gauss + 1, "total": gauss + 1}, _log_metric=lambda *a: None)
  fake._pop_val_t = lambda stage: M._pop_val_t(fake, stage)
  fake._log_photom = lambda *a: M._log_photom(fake, *a)

  def validate(k=None):
    if k is not None:
      dm.validation = "small"
      dm.reseed_validation(k)
    M.on_validation_epoch_start(fake)
    out = {}
    for stage, sizes in SETS.items():
      forwards.clear()
      for B in sizes:
        M._diffusion_val_t(fake, stage, {"batch_size": B, "x0_raw": torch.zeros(B)})
      out[stage] = sum(forwards, [])
    if k is not None:
      dm.validation = "lightning"
      dm.reseed_lightning()
    else:
      M.on_validation_end(fake)
    return out
  return dm, validate


def test_val_t_bag_balances_each_validation_set():
  """Every validation set's volumes split evenly over diffusion.val_t: a fresh shuffled bag per set and validation, refilled when
  empty, so each run of len(val_t) volumes in a set takes every value once (sets not a multiple of it: no bag shared between sets
  or carried into the next validation)."""
  _, validate = val_t_run()
  for k in (1, 2, 3):
    for stage, ts in validate(k).items():
      assert len(ts) == sum(SETS[stage])
      for i in range(0, len(ts), len(VAL_T)):
        chunk = ts[i:i + len(VAL_T)]
        assert len(set(chunk)) == len(chunk) and set(chunk) <= set(VAL_T), (k, stage, ts)


def test_val_t_logs_one_shared_t_per_volume():
  """One shared t per volume over all its layers, one forward per batch, logged as loss/photom_t<t>[_l1] with its own count."""
  M = train_gs.GSLightningModule
  logged, forwards = [], []

  def training_loss(image, hit, x0_raw, seeds, return_pred, t):
    forwards.append(t.clone())
    return None, None, None, torch.zeros(t.shape[0]), t
  dm = types.SimpleNamespace(validation_seeds={"val_a": 5})
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"eval_seed": 0, "val_t": VAL_T}}), device=torch.device("cpu"),
    trainer=types.SimpleNamespace(datamodule=dm),
    model=types.SimpleNamespace(inputs=lambda batch, dev: ("img", torch.ones(batch["batch_size"], 6, 4, 4), [0] * batch["batch_size"], None,
                                                         None),
                                training_loss=training_loss, decode=lambda x, floor=None: x),
    _photom_per_volume=lambda batch, gauss: {"l1": gauss + 1, "total": gauss + 1},
    _log_metric=lambda stage, k, v, n: logged.append((stage, k, n)))
  fake._pop_val_t = lambda stage: M._pop_val_t(fake, stage)
  fake._log_photom = lambda *a: M._log_photom(fake, *a)
  M.on_validation_epoch_start(fake)
  M._diffusion_val_t(fake, "val_a", {"batch_size": 8, "x0_raw": torch.zeros(8)})
  assert len(forwards) == 1 and forwards[0].shape == (8, 6) and (forwards[0] == forwards[0][:, :1]).all()
  ts = sorted(forwards[0][:, 0].tolist())
  assert ts == sorted(VAL_T * 2)
  ts_names = ("0.25", "0.5", "0.75", "1")
  assert {k for _, k, _ in logged} == {f"loss/photom_t{t}" for t in ts_names} | {f"loss/photom_t{t}_l1" for t in ts_names}
  assert all(n == 2 for _, k, n in logged if k == "loss/photom_t0.5")


def test_val_t_is_paired_across_runs_and_resumes():
  """At its k-th validation every run gives each volume of a set the same t (a function of cfg.seed, eval_seed, k and the set, not
  of the run's history), the next validation a different draw, and a resumed run (counters restored) goes on with the sequence."""
  _, a = val_t_run()
  ref = {k: a(k) for k in (1, 2, 3)}
  lightning = [a() for _ in range(3)]                  # Lightning's own validations 1..3
  _, b = val_t_run()
  np.random.seed(7); torch.rand(9)
  b(); b(); b(5)                                         # another history
  assert b(2) == ref[2] and b(3) == ref[3]
  assert ref[2] != ref[3] and ref[2]["val_a"][:6] != ref[2]["val_b"]   # per validation, per set
  assert lightning[0] != ref[1]                          # Lightning's own validations: their own seeds
  dm, c = val_t_run()                                    # resumed after 2 of Lightning's own validations
  dm.load_state_dict({"lightning_runs": 2})
  assert c() == lightning[2]
  assert val_t_run(seed=7)[1](2) != ref[2] and val_t_run(eval_seed=1)[1](2) != ref[2]


@pytest.mark.cuda
@pytest.mark.data
def test_validation_photom_is_gated_like_training(photom_setup):
  s = photom_setup
  logged, B = s.logged, s.B
  logged.clear()
  s.photom("val", s.batch, s.x0, torch.full((B, 6), 0.9, device=s.dev))   # max t 0.9: out
  assert logged == {"val/loss/photom_frac": 0.0}
  s.photom("val", s.batch, s.x0, torch.full((B, 6), 0.5, device=s.dev))   # in t_range
  assert logged["val/loss/photom_frac"] == 1.0 and {"val/loss/photom", "val/loss/photom_l1", "val/loss/photom_dssim", "val/loss/photom_psnr", "val/loss/photom_fg_psnr", "val/loss/photom_bg_l1"} <= set(logged)
  assert 20 < logged["val/loss/photom_psnr"] < 60 and logged["val/loss/photom_bg_l1"] < 0.05   # the fits themselves: sane values
  assert logged["val/loss/photom"] < 0.15   # the fitted grids themselves (about 0.08): a sane render, not noise


def test_shared_t1_is_the_one_step_sample():
  """loss/photom_t1 (training_loss at t = 1 on every layer, with the volume's seeded noise) renders exactly what the 1-step
  sample (sample(steps=1), the eval's onestep) does: the x0 prediction from the same pure noise."""
  import gs_diffusion as gd
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  w = torch.randn(14, generator=torch.Generator().manual_seed(0))
  m.head = lambda feats, x, t: x * w[None, None, :, None, None] + t[:, :, None, None, None]   # a stand-in network
  m._feats = lambda image: None
  m.normalize = lambda x: x
  m.decode = lambda x, floor=None: x
  m.eval_seed = 0
  m.schedule = types.SimpleNamespace(sample=lambda b, phase, generator=None: torch.rand(b, 6, generator=generator))
  m.schedule.fixed_t = None
  B, L = 2, 6
  hit = torch.rand(B, L, 4, 4, generator=torch.Generator().manual_seed(1)) > 0.3
  image, seeds = torch.zeros(B, 3, 8, 8), [3, 9]
  x0_hat = m.training_loss(image, hit, torch.zeros(B, L, 14, 4, 4), seeds=seeds, return_pred=True, t=torch.ones(B, L))[3]
  one_step = m.sample(image, hit, seeds=seeds, steps=1)
  torch.testing.assert_close(x0_hat.float(), one_step.float())
