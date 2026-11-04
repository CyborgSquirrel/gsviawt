"""Shared pytest setup.

Tests import the repo's modules from the repo root (this directory's parent). Markers:
  cuda  needs a CUDA GPU (gsplat rendering, bf16 autocast); skipped without one
  data  needs local data (fitted grids, render files) under data_root(): $GSVIAWT_TEST_DATA if set, else the first existing of
        <repo>/bla, the main checkout's bla (a git worktree has none) and /app/bla (the container's mount); the `data_dir` fixture
        skips the test when the file it asks for is missing, and the run ends with a one-line note of how many were skipped
Run: python -m pytest tests"""
import os
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

DATA_MISSING = "local test data missing"


def _main_checkout_bla():
  try:
    common = subprocess.run(["git", "-C", ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, timeout=10).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return None
  return os.path.join(os.path.dirname(common), "bla") if common else None


def data_root():
  """$GSVIAWT_TEST_DATA (the explicit override), else the first existing candidate (else <repo>/bla, missing)."""
  if os.environ.get("GSVIAWT_TEST_DATA"):
    return os.environ["GSVIAWT_TEST_DATA"]
  for p in (os.path.join(ROOT, "bla"), _main_checkout_bla(), "/app/bla"):
    if p and os.path.isdir(p):
      return p
  return os.path.join(ROOT, "bla")


def data_path(*parts):
  """data_root()/parts; skips the test when it does not exist."""
  p = os.path.join(data_root(), *parts)
  if not os.path.exists(p):
    pytest.skip(f"{DATA_MISSING}: {p} (set GSVIAWT_TEST_DATA to the main checkout's bla/ to run the data tests)")
  return p


def pytest_configure(config):
  config.addinivalue_line("markers", "cuda: needs a CUDA GPU")
  config.addinivalue_line("markers", "data: needs local data under data_root() ($GSVIAWT_TEST_DATA or the main checkout's bla)")


def pytest_collection_modifyitems(config, items):
  import torch
  if torch.cuda.is_available():
    return
  for item in items:
    if "cuda" in item.keywords:
      item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU"))


def pytest_terminal_summary(terminalreporter):
  n = sum(1 for r in terminalreporter.stats.get("skipped", []) if DATA_MISSING in str(r.longrepr))
  if n:
    terminalreporter.write_line(f"{n} data test(s) SKIPPED: their files are missing under {data_root()} "
                                "(set GSVIAWT_TEST_DATA=<main checkout>/bla)", yellow=True)


@pytest.fixture(scope="session")
def data_dir():
  """data_path: a function data_dir(*parts) -> the path under data_root(), skipping the test when it does not exist."""
  return data_path


@pytest.fixture(scope="session")
def photom_setup(data_dir):
  """diffusion.photom on the real local test grids: the train_gs.GSLightningModule rendering helpers bound onto a fake module whose
  model is a GSDiffusionModel shell (class attributes inherited) holding the grids' x0 normalization."""
  import tempfile

  import torch
  from omegaconf import OmegaConf

  import gs_diffusion as gd
  import train_gs
  from gs_dataset import WTFeatureDataset, render_catalog
  from module import DSSIMLoss
  from util import collate_with_batch_size
  idx, src = gd.build_grid_index(data_dir("grids_test", "fit", "grids"))
  if not os.path.exists(src):
    pytest.skip(f"{DATA_MISSING}: the test grids' render file {src}")
  listing = gd.make_listing(idx, src, os.path.join(tempfile.mkdtemp(), "listing.h5"))
  cat = render_catalog(src)
  cat = cat.take([p for p, v in enumerate(cat.df["view_idx"].to_list()) if v in set(idx)])
  w = gd.GridTargetDataset(WTFeatureDataset(listing, cat, num_layers=6, num_views=2, depth_size=None, image_size=504,
                                            read_features=False), idx)
  mean, std, lo, hi = gd.compute_stats(w, idx)[:4]
  dev = torch.device("cuda")
  model = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(model)
  model.backbone = None
  for n, v in (("x0_mean", mean), ("x0_std", std), ("x0_lo", lo), ("x0_hi", hi)):
    model.register_buffer(n, v.float())
  cfg = OmegaConf.create({"loss": {"photom": {"for": ["target"], "l1_weight": 1.0, "dssim_weight": 0.85}},
                          "diffusion": {"photom": {"enabled": True, "weight": 1.0, "t_range": [0.0, 0.7], "t_reduce": "max",
                                                   "validate": True}}})
  logged = {}
  fake = types.SimpleNamespace(cfg=cfg, device=dev, model=model.to(dev), dssim=DSSIMLoss(reduction="none").to(dev),
                               _log_metric=lambda stage, k, v, n: logged.__setitem__(f"{stage}/{k}", float(v)))
  for name in ("_render_photom", "_photom_per_volume", "_log_photom"):   # the helpers _diffusion_photom calls
    setattr(fake, name, (lambda f: lambda *a: f(fake, *a))(getattr(train_gs.GSLightningModule, name)))
  batch = collate_with_batch_size([w[i] for i in range(len(w))])
  x0 = ((batch["x0_raw"].to(dev).float().permute(0, 2, 1, 3, 4) - mean.to(dev)[None, None, :, None, None])
        / std.to(dev)[None, None, :, None, None])
  photom = lambda *a: train_gs.GSLightningModule._diffusion_photom(fake, *a)
  return types.SimpleNamespace(batch=batch, x0=x0, photom=photom, logged=logged, B=batch["batch_size"], dev=dev)
