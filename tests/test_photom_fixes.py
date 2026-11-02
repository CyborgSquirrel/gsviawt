"""The PR #97 fixes: D-SSIM in fp32 (its gradient under bf16 autocast was noise), the validation's one-step render logged as
loss/photom_x0 (not colliding with the sample render's loss/photom), and train/loss/x0 logged only for accepted steps."""
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


@pytest.mark.cuda
@pytest.mark.data
def test_validation_logs_one_step_render_as_photom_x0():
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
  train_gs.GSLightningModule._diffusion_photom(fake, "val", batch, x0, torch.full((batch["batch_size"], 6), 0.9, device=dev),
                                               gated=False)
  assert "val/loss/photom_x0" in logged and "val/loss/photom" not in logged and "val/loss/photom_frac" not in logged
