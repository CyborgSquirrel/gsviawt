"""PSNR logging (train_gs.image_quality: psnr / fg_psnr / bg_l1 next to every rendering loss, eval_diffusion's definitions) and
the train step's logging fixes (rendering-loss logs on accepted steps only; x0 parts as train/loss/x0/<part>, like the
validations)."""
import types

import torch
from omegaconf import OmegaConf

import train_gs


def test_image_quality_matches_eval_metrics():
  import eval_diffusion
  g = torch.Generator().manual_seed(0)
  B, V, H, W = 2, 3, 16, 16
  gt = torch.rand(B, V, 3, H, W, generator=g)
  pred = (gt + 0.1 * torch.randn(B, V, 3, H, W, generator=g)).clamp(0, 1)
  alpha = (torch.rand(B, V, 1, H, W, generator=g) > 0.4).float()
  q = train_gs.image_quality(pred, gt, alpha)
  for b in range(B):
    ref = eval_diffusion.metrics(pred[b], gt[b], dssim=lambda x, y: (x - y).abs(), lpips=lambda x, y: torch.zeros(()),
                                 gt_alpha=alpha[b])
    for name in ("psnr", "fg_psnr", "bg_l1"):
      torch.testing.assert_close(q[name][b], ref[name])
  q2 = train_gs.image_quality(pred, gt, alpha[:, :, 0])   # (B, V, H, W) alpha too
  torch.testing.assert_close(q2["psnr"], q["psnr"])
  assert q["psnr"].shape == (B, V) and not q["psnr"].requires_grad


def _train_fake(skip):
  logged = []
  fake = types.SimpleNamespace(
    cfg=OmegaConf.create({"diffusion": {"photom": {"enabled": True, "weight": 10.0}}}), views_seen=0, pixels_seen=0,
    _accepted_since_step=False, spike_skip=types.SimpleNamespace(check=lambda v: skip, average=None),
    _diffusion_losses=lambda batch: (torch.tensor(0.5), {"opacity": torch.tensor(0.1)}, {"opacity": 7}, "x0_hat", "t"),
    _photom_active=lambda: True,
    _log_metric=lambda stage, k, v, n: logged.append(f"{stage}/{k}"))
  fake._log_parts = lambda *a: train_gs.GSLightningModule._log_parts(fake, *a)

  def photom(stage, batch, x0_hat, t, log):
    log(stage, "loss/photom_frac", 1.0, 4)
    log(stage, "loss/photom_psnr", 20.0, 4)
    log(stage, "loss/photom", 0.2, 4)
    return torch.tensor(0.2)
  fake._diffusion_photom = photom
  out = train_gs.GSLightningModule._diffusion_train_step(fake, {"batch_size": 4, "x0_raw": torch.zeros(4, 6, 14, 8, 8)})
  return out, logged


def test_train_photom_logs_only_on_accepted_steps():
  out, logged = _train_fake(skip=True)
  assert out is None and not [k for k in logged if "photom" in k]
  out, logged = _train_fake(skip=False)
  torch.testing.assert_close(out, torch.tensor(0.5 + 10 * 0.2))
  assert {"train/loss/photom", "train/loss/photom_frac", "train/loss/photom_psnr"} <= set(logged)
  assert "train/loss/x0/opacity" in logged and "train/loss/opacity" not in logged
