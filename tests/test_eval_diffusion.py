"""eval_diffusion: the reproducible volume subset, the metrics (foreground over the TARGET silhouette, so haze cannot dilute it;
bg_l1 / alpha IoU catch it; raw depth error), and the source-to-target view angle."""
import math

import pytest
import torch

import eval_diffusion as ev
from module import DSSIMLoss

NO_LPIPS = lambda a, b: torch.zeros(())   # LPIPS weights are not needed for these checks


def test_volume_subset():
  a = ev.volume_subset(450, 60, 0, "val_meshes")
  assert len(a) == 60 and a == sorted(a) and len(set(a)) == 60 and max(a) < 450
  assert a == ev.volume_subset(450, 60, 0, "val_meshes")                       # reproducible
  assert a != ev.volume_subset(450, 60, 0, "val_views_sval")                   # differs per set
  assert a != ev.volume_subset(450, 60, 1, "val_meshes")                       # and per seed
  assert ev.volume_subset(10, 60, 0, "x") == list(range(10)) == ev.volume_subset(10, None, 0, "x")


def scene():
  gt = torch.zeros(1, 3, 32, 32)
  gt_alpha = torch.zeros(1, 1, 32, 32)
  gt[..., 8:24, 8:24] = 0.6
  gt_alpha[..., 8:24, 8:24] = 1
  return gt, gt_alpha


def test_haze_does_not_inflate_foreground_psnr():
  d = DSSIMLoss(reduction="none")
  gt, gt_alpha = scene()
  pred = gt.clone()
  pred[..., 8:24, 8:24] += 0.1                                               # a fixed error inside the object
  alpha = gt_alpha.clone()
  clean = ev.metrics(pred, gt, d, NO_LPIPS, alpha, gt_alpha)
  hazy_pred, hazy_alpha = pred.clone(), alpha.clone()
  hazy_pred[..., :, :] = torch.where(gt_alpha.bool(), hazy_pred, torch.full_like(hazy_pred, 0.02))   # dim haze everywhere else
  hazy_alpha[:] = 1
  hazy = ev.metrics(hazy_pred, gt, d, NO_LPIPS, hazy_alpha, gt_alpha)
  assert math.isclose(float(hazy["fg_psnr"]), float(clean["fg_psnr"]), rel_tol=1e-6)   # the foreground region is fixed
  assert float(hazy["bg_l1"]) > float(clean["bg_l1"]) + 0.01
  assert float(hazy["alpha_iou"]) < 0.5 < float(clean["alpha_iou"])
  assert math.isclose(float(clean["fg_psnr"]), 20.0, abs_tol=1e-3)            # mse 0.01 over the target silhouette


def test_depth_absrel_and_missing_inputs():
  d = DSSIMLoss(reduction="none")
  gt, gt_alpha = scene()
  gt_depth = torch.full((1, 1, 32, 32), float("nan"))
  gt_depth[..., 8:24, 8:24] = 2.0
  m = ev.metrics(gt, gt, d, NO_LPIPS, gt_alpha, gt_alpha, torch.full((1, 1, 32, 32), 2.2), gt_depth)
  assert math.isclose(float(m["depth_absrel"]), 0.1, rel_tol=1e-5)
  assert math.isinf(float(m["fg_psnr"])) or float(m["fg_psnr"]) > 90
  bare = ev.metrics(gt, gt, d, NO_LPIPS)                                     # no alpha / depth: those metrics are NaN
  assert all(math.isnan(float(bare[k])) for k in ("fg_psnr", "bg_l1", "alpha_iou", "depth_absrel"))


def test_view_angles():
  def viewmat(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])            # rotate the camera about +Y
    return torch.linalg.inv(c2w)
  batch = {"views": {"viewmat": torch.stack([viewmat(0), viewmat(30), viewmat(90), viewmat(180)])[None]}}
  ang = ev.view_angles(batch, [1, 2, 3])
  assert torch.allclose(ang, torch.tensor([30.0, 90.0, 180.0]), atol=1e-3)
