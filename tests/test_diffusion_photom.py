"""diffusion.photom (the rendering loss on the x0 prediction), on real local grids with the model replaced by a stub: the fit's own
x0 renders much better than a noised one, the gradient reaches x0_hat, and the gate (t_range on the max over layers) works."""
import tempfile
import types

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = [pytest.mark.cuda, pytest.mark.data]


@pytest.fixture(scope="module")
def setup():
  import os
  root = os.environ.get("GSVIAWT_TEST_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bla"))
  grids = os.path.join(root, "grids_test", "fit", "grids")
  if not os.path.isdir(grids):
    pytest.skip(f"local test data missing: {grids}")
  import gs_diffusion as gd
  import train_gs
  from gs_dataset import WTFeatureDataset, render_catalog
  from module import DSSIMLoss
  from util import collate_with_batch_size
  idx, src = gd.build_grid_index(grids)
  if not os.path.exists(src):
    pytest.skip(f"render file of the test grids missing: {src}")
  listing = gd.make_listing(idx, src, os.path.join(tempfile.mkdtemp(), "listing.h5"))
  cat = render_catalog(src)
  cat = cat.take([p for p, v in enumerate(cat.df["view_idx"].to_list()) if v in set(idx)])
  ds = WTFeatureDataset(listing, cat, num_layers=6, num_views=2, depth_size=None, image_size=504, read_features=False)
  w = gd.GridTargetDataset(ds, idx)
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

  cfg = OmegaConf.create({"loss": {"photom": {"for": ["target"], "l1_weight": 1.0, "dssim_weight": 0.85}},
                          "diffusion": {"photom": {"enabled": True, "weight": 1.0, "t_range": [0.0, 0.7], "t_reduce": "max",
                                                   "validate": True}}})
  logged = {}
  fake = types.SimpleNamespace(cfg=cfg, device=dev, model=Stub().to(dev), dssim=DSSIMLoss(reduction="none").to(dev),
                               _log_metric=lambda stage, k, v, n: logged.__setitem__(f"{stage}/{k}", float(v)))
  batch = collate_with_batch_size([w[i] for i in range(len(w))])
  x0 = (batch["x0_raw"].to(dev).float().permute(0, 2, 1, 3, 4) - mean.to(dev)[None, None, :, None, None]) \
       / std.to(dev)[None, None, :, None, None]
  photom = lambda *a, **k: train_gs.GSLightningModule._diffusion_photom(fake, *a, **k)
  return types.SimpleNamespace(batch=batch, x0=x0, photom=photom, logged=logged, B=batch["batch_size"], dev=dev)


def test_fit_renders_better_than_noise_and_gradient_flows(setup):
  s = setup
  t_lo = torch.full((s.B, 6), 0.1, device=s.dev)
  good = float(s.photom("train", s.batch, s.x0, t_lo, gated=True))
  noisy = (s.x0 + 0.5 * torch.randn_like(s.x0)).requires_grad_(True)
  bad = s.photom("train", s.batch, noisy, t_lo, gated=True)
  assert good < float(bad.detach()) / 2
  bad.backward()
  assert torch.isfinite(noisy.grad).all() and noisy.grad.abs().sum() > 0


def test_gate_on_the_noisiest_layer(setup):
  s = setup
  t_hi = torch.full((s.B, 6), 0.9, device=s.dev)
  s.logged.clear()
  assert float(s.photom("train", s.batch, s.x0, t_hi, gated=True)) == 0 and s.logged["train/loss/photom_frac"] == 0
  t = torch.full((s.B, 6), 0.1, device=s.dev)
  t[0, 3] = 0.9                                                     # one noisy layer closes volume 0
  s.logged.clear()
  s.photom("train", s.batch, s.x0, t, gated=True)
  assert abs(s.logged["train/loss/photom_frac"] - (s.B - 1) / s.B) < 1e-6
