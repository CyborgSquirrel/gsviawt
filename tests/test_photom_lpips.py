"""diffusion.photom.lpips_weight: the optional LPIPS term of the rendering loss. Off (0 / no key): the loss as before, no net
built. On: finite and differentiable, logged as loss/photom_lpips; the net frozen, in eval mode and in no state_dict; the alpha
crop (incl. views without a silhouette: 0, no crash); fp32 under bf16 autocast."""
import contextlib

import pytest
import torch
import torch.nn.functional as F

import train_gs


def smooth_images(n=2, size=128, seed=0):
  g = torch.Generator().manual_seed(seed)
  x = F.interpolate(torch.rand(n, 3, size // 8, size // 8, generator=g), size=size, mode="bilinear", align_corners=False)
  return x.clamp(0, 1)


@contextlib.contextmanager
def photom_keys(cfg, **keys):
  """cfg.diffusion.photom with `keys` set for the block, then as before (the photom_setup cfg is shared by the session)."""
  ph = cfg.diffusion.photom
  old = {k: ph[k] for k in keys if k in ph}
  for k, v in keys.items():
    ph[k] = v
  try:
    yield
  finally:
    for k in keys:
      if k in old:
        ph[k] = old[k]
      else:
        del ph[k]


# --- crop (no GPU) -------------------------------------------------------------------------------------------------------

def test_crop_box_square_with_margin():
  gt = torch.zeros(1, 1, 100, 100)
  gt[0, 0, 40:60, 30:70] = 1                                         # 20 x 40 silhouette
  (box,) = train_gs.alpha_crop_boxes(torch.zeros_like(gt), gt)
  y0, x0, size = box
  assert size == 40 + 2 * 4                                           # longer side + 10 % on each side
  assert (y0, x0) == (50 - 24, 50 - 24)                               # centred on the silhouette's bbox


def test_crop_box_union_of_pred_and_gt():
  gt, pred = torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 64, 64)
  gt[0, 0, 10:20, 10:20] = 1
  pred[0, 0, 40:50, 40:50] = 1                                        # a floater outside the GT silhouette
  (box,) = train_gs.alpha_crop_boxes(pred, gt)
  y0, x0, size = box
  assert y0 <= 10 and x0 <= 10 and y0 + size >= 50 and x0 + size >= 50
  (box,) = train_gs.alpha_crop_boxes(pred, torch.zeros_like(gt))     # GT empty, render not: the render's box
  assert box is not None and box[0] <= 40 and box[0] + box[2] >= 50


def test_crop_box_empty_and_edge():
  a = torch.zeros(2, 64, 64)
  a[1, 0:10, 60:64] = 1                                               # touching the top-right corner
  boxes = train_gs.alpha_crop_boxes(a, a)
  assert boxes[0] is None
  y0, x0, size = boxes[1]
  assert y0 < 0 and x0 + size > 64                                    # reaches outside the frame
  img = torch.ones(3, 64, 64)
  out = train_gs.crop_square(img, boxes[1])
  assert out.shape == (3, size, size)
  assert out[:, -y0:, :64 - x0].eq(1).all() and out[:, :-y0].eq(0).all() and out[:, :, 64 - x0:].eq(0).all()


# --- the LPIPS function (GPU) --------------------------------------------------------------------------------------------

@pytest.mark.cuda
def test_lpips_net_frozen_eval_not_in_state_dict():
  net = train_gs.lpips_net("vgg", torch.device("cuda"))
  assert net is train_gs.lpips_net("vgg", torch.device("cuda"))      # one per (net, device)
  assert all(not p.requires_grad for p in net.parameters())
  net.train(True)
  assert not net.training and not any(m.training for m in net.modules())   # so its dropout never acts
  # the calibrated lin weights are loaded (torchmetrics loads them with strict=False: with use_dropout=False their keys miss)
  import inspect, os
  from torchmetrics.functional.image import lpips as tm
  shipped = torch.load(os.path.join(os.path.dirname(inspect.getfile(tm)), "lpips_models", "vgg.pth"), map_location="cpu")
  sd = net.state_dict()
  assert shipped and all(torch.equal(sd[k].cpu(), v) for k, v in shipped.items())
  x, y = smooth_images(2, 64, 0).cuda(), smooth_images(2, 64, 1).cuda()
  a, b = net(x, y, normalize=True), net(x, y, normalize=True)
  assert torch.equal(a, b) and (a > 0).all()                          # deterministic, non-negative


@pytest.mark.cuda
def test_lpips_empty_views_contribute_zero():
  x, y = smooth_images(3, 64, 0).cuda(), smooth_images(3, 64, 1).cuda()
  x.requires_grad_(True)
  alpha = torch.zeros(3, 1, 64, 64, device="cuda")
  alpha[1, :, 16:48, 16:48] = 1
  v = train_gs.photom_lpips(x, y, alpha, alpha, net="alex", crop=64)
  assert v.shape == (3,) and v[0] == 0 and v[2] == 0 and v[1] > 0
  v.sum().backward()
  assert x.grad[0].eq(0).all() and x.grad[2].eq(0).all() and x.grad[1].abs().sum() > 0
  none = train_gs.photom_lpips(x, y, torch.zeros_like(alpha), torch.zeros_like(alpha), net="alex", crop=64)
  assert none.shape == (3,) and none.eq(0).all()                     # no silhouette anywhere: no crash, 0
  full = train_gs.photom_lpips(x, y, alpha, alpha, net="alex", crop=None)
  assert full.shape == (3,) and (full > 0).all()                     # null: the full frames, silhouette or not


@pytest.mark.cuda
@pytest.mark.parametrize("net", ["vgg", "alex"])
def test_lpips_bf16_autocast_matches_fp32(net):
  x, y = smooth_images(2, 128, 0).cuda(), smooth_images(2, 128, 1).cuda()
  alpha = torch.zeros(2, 1, 128, 128, device="cuda")
  alpha[:, :, 20:100, 30:90] = 1
  x32 = x.clone().requires_grad_()
  train_gs.photom_lpips(x32, y, alpha, alpha, net=net, crop=96).sum().backward()
  xb = x.clone().requires_grad_()
  with torch.autocast("cuda", dtype=torch.bfloat16):
    v = train_gs.photom_lpips(xb, y, alpha, alpha, net=net, crop=96)
  v.sum().backward()
  assert v.dtype == torch.float32
  cos = F.cosine_similarity(x32.grad.flatten(), xb.grad.flatten(), dim=0)
  assert cos > 0.9999, f"LPIPS-{net} gradient under bf16 autocast: cosine {float(cos):.5f} to fp32"


# --- in the rendering loss (GPU, real grids) -----------------------------------------------------------------------------

@pytest.mark.cuda
@pytest.mark.data
def test_off_is_the_loss_as_before(photom_setup, monkeypatch):
  s = photom_setup
  t = torch.full((s.B, 6), 0.1, device=s.dev)
  def no_lpips(*a, **k):
    raise AssertionError("photom_lpips called with lpips_weight 0")
  monkeypatch.setattr(train_gs, "photom_lpips", no_lpips)
  monkeypatch.setattr(train_gs, "lpips_net", no_lpips)
  s.logged.clear()
  without = s.photom("train", s.batch, s.x0, t)                       # a config without the keys
  keys = set(s.logged)
  l1, dssim = s.logged["train/loss/photom_l1"], s.logged["train/loss/photom_dssim"]
  with photom_keys(s.cfg, lpips_weight=0.0, lpips_net="vgg", lpips_crop=256):
    s.logged.clear()
    off = s.photom("train", s.batch, s.x0, t)
  assert torch.equal(without, off) and set(s.logged) == keys
  assert not any("lpips" in k for k in keys)
  assert abs(float(off) - (l1 + 0.85 * dssim)) < 1e-6                # L1 + 0.85 D-SSIM, nothing else


@pytest.mark.cuda
@pytest.mark.data
@pytest.mark.parametrize("crop", [256, None])
def test_on_finite_differentiable_logged(photom_setup, crop):
  s = photom_setup
  t = torch.full((s.B, 6), 0.1, device=s.dev)
  state = set(s.model.state_dict())
  s.logged.clear()
  off = float(s.photom("train", s.batch, s.x0, t))
  with photom_keys(s.cfg, lpips_weight=0.5, lpips_net="vgg", lpips_crop=crop):
    s.logged.clear()
    x0 = s.x0.clone().requires_grad_(True)
    on = s.photom("train", s.batch, x0, t)
    lp = s.logged["train/loss/photom_lpips"]
    on.backward()
    noisy = s.photom("train", s.batch, s.x0 + 0.5 * torch.randn_like(s.x0), t)
    lp_noisy = s.logged["train/loss/photom_lpips"]
  assert torch.isfinite(on) and 0 < lp < 1
  assert abs(float(on.detach()) - (off + 0.5 * lp)) < 1e-5
  assert torch.isfinite(x0.grad).all() and x0.grad.abs().sum() > 0
  assert lp < lp_noisy                                                # the fit's own x0 is perceptually closer than a noised one
  assert set(s.model.state_dict()) == state                           # nothing registered on the model
