"""The PR #97 fixes: D-SSIM in fp32 (its gradient under bf16 autocast was noise), the validation's rendering keys
(loss/photom: the training's term, gated the same way; loss/photom_t<t>: at a shared timestep from a shuffled bag of
diffusion.val_t; loss/photom_full: the full sample -- none colliding), and train/loss/x0 logged only for accepted steps."""
import os
import tempfile
import types

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


def test_val_t_bag_balances_each_validation_set():
  """Every validation set's volumes split evenly over diffusion.val_t (a fresh shuffled bag per set and validation), one shared t
  per volume over all its layers, one forward + one render per batch, logged as loss/photom_t<t> with its own count."""
  M = train_gs.GSLightningModule
  logged, forwards = [], []

  def training_loss(image, hit, x0_raw, seeds, return_pred, t):
    forwards.append(t.clone())
    return None, None, None, torch.zeros(t.shape[0]), t

  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"eval_seed": 0, "val_t": [0.25, 0.5, 0.75, 1.0]}}), device=torch.device("cpu"),
    model=types.SimpleNamespace(inputs=lambda batch, dev: ("img", torch.ones(batch["batch_size"], 6, 4, 4), [0] * batch["batch_size"]),
                                training_loss=training_loss, decode=lambda x: x),
    _photom_per_volume=lambda batch, gauss: {"l1": gauss + 1, "total": gauss + 1},
    _log_metric=lambda stage, k, v, n: logged.append((stage, k, n)))
  fake._pop_val_t = lambda stage: M._pop_val_t(fake, stage)
  fake._log_photom = lambda *a: M._log_photom(fake, *a)
  for _ in range(2):   # two validations
    M.on_validation_epoch_start(fake)
    for stage in ("val_a", "val_b"):
      for B in (3, 5):   # 8 volumes per set
        M._diffusion_val_t(fake, stage, {"batch_size": B, "x0_raw": torch.zeros(B)})
  assert len(forwards) == 8 and all(f.shape[1] == 6 and (f == f[:, :1]).all() for f in forwards)   # shared over the layers
  for k in range(2):
    for s_i, stage in enumerate(("val_a", "val_b")):
      ts = torch.cat(forwards[4 * k + 2 * s_i: 4 * k + 2 * s_i + 2])[:, 0].tolist()
      assert sorted(ts) == [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0], (stage, ts)
  keys = {k for _, k, _ in logged}
  assert keys == {f"loss/photom_t{t}" for t in ("0.25", "0.5", "0.75", "1")} | {f"loss/photom_t{t}_l1" for t in ("0.25", "0.5", "0.75", "1")}


@pytest.mark.cuda
@pytest.mark.data
def test_validation_photom_is_gated_like_training():
  root = os.environ.get("GSVIAWT_TEST_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bla"))
  grids = os.path.join(root, "grids_test", "fit", "grids")
  if not os.path.isdir(grids):
    pytest.skip(f"local test data missing: {grids}")
  import gs_diffusion as gd
  from gs_dataset import WTFeatureDataset, render_catalog
  from util import collate_with_batch_size
  idx, src = gd.build_grid_index(grids)
  if not os.path.exists(src):
    pytest.skip(f"render file of the test grids missing: {src}")
  listing = gd.make_listing(idx, src, os.path.join(tempfile.mkdtemp(), "listing.h5"))
  cat = render_catalog(src)
  cat = cat.take([p for p, v in enumerate(cat.df["view_idx"].to_list()) if v in set(idx)])
  w = gd.GridTargetDataset(WTFeatureDataset(listing, cat, num_layers=6, num_views=2, depth_size=None, image_size=504,
                                            read_features=False), idx)
  mean, std, lo, hi = gd.compute_stats(w, idx)[:4]
  dev = torch.device("cuda")

  class Stub(torch.nn.Module):
    max_sh_degree = 0
    backbone = None
    decode = gd.GSDiffusionModel.decode

    def __init__(self):
      super().__init__()
      for n, v in (("x0_mean", mean), ("x0_std", std), ("x0_lo", lo), ("x0_hi", hi)):
        self.register_buffer(n, v.float())
  logged = {}
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"loss": {"photom": {"for": ["target"], "l1_weight": 1.0, "dssim_weight": 0.85}},
                          "diffusion": {"photom": {"enabled": True, "t_range": [0.0, 0.7], "t_reduce": "max"}}}),
    device=dev, model=Stub().to(dev), dssim=DSSIMLoss(reduction="none").to(dev),
    _log_metric=lambda stage, k, v, n: logged.__setitem__(f"{stage}/{k}", float(v)))
  batch = collate_with_batch_size([w[i] for i in range(len(w))])
  x0 = ((batch["x0_raw"].to(dev).float().permute(0, 2, 1, 3, 4) - mean.to(dev)[None, None, :, None, None])
        / std.to(dev)[None, None, :, None, None])
  fake._render_photom = lambda *a: train_gs.GSLightningModule._render_photom(fake, *a)
  fake._photom_per_volume = lambda *a: train_gs.GSLightningModule._photom_per_volume(fake, *a)
  fake._log_photom = lambda *a: train_gs.GSLightningModule._log_photom(fake, *a)
  B = batch["batch_size"]
  train_gs.GSLightningModule._diffusion_photom(fake, "val", batch, x0, torch.full((B, 6), 0.9, device=dev))   # max t 0.9: out
  assert logged == {"val/loss/photom_frac": 0.0}
  train_gs.GSLightningModule._diffusion_photom(fake, "val", batch, x0, torch.full((B, 6), 0.5, device=dev))   # in t_range
  assert logged["val/loss/photom_frac"] == 1.0 and {"val/loss/photom", "val/loss/photom_l1", "val/loss/photom_dssim"} <= set(logged)
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
  m.decode = lambda x: x
  m.eval_seed = 0
  m.schedule = types.SimpleNamespace(sample=lambda b, phase, generator=None: torch.rand(b, 6, generator=generator))
  if hasattr(gd.GSDiffusionModel, "_net_input"):
    m.schedule.fixed_t = None
  B, L = 2, 6
  hit = torch.rand(B, L, 4, 4, generator=torch.Generator().manual_seed(1)) > 0.3
  image, seeds = torch.zeros(B, 3, 8, 8), [3, 9]
  x0_hat = m.training_loss(image, hit, torch.zeros(B, L, 14, 4, 4), seeds=seeds, return_pred=True, t=torch.ones(B, L))[3]
  one_step = m.sample(image, hit, seeds=seeds, steps=1)
  torch.testing.assert_close(x0_hat.float(), one_step.float())
