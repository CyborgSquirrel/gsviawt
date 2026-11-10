"""diffusion.augment.slot_shuffle_target: shuffled (as before: the slot-shuffled grid is noised and is the target) | canonical
(shuffled -> unshuffled: only the noised input is built from the shuffled grid, "x0_input_raw"; the target "x0_raw" is the grid as
fitted). The dataset's items, and training_loss's noised input vs target."""
import numpy as np
import pytest
import torch

import gs_diffusion as gd
from test_augment import scene
from test_cond_dropout import data, shell


class Inner:
  """One render item (source view 3) around test_augment's scene."""
  volumes = type("V", (), {"df": {"view_idx": type("C", (), {"to_list": staticmethod(lambda: [3])})}})()

  def __init__(self, source):
    self.source = source

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    return {"source_view": 3, "source": self.source}


AUG = {"slot_shuffle": True, "slot_shuffle_block": 2, "slot_shuffle_max_px": 3.0}


def dataset(monkeypatch, target, block=2):
  x0, src = scene()
  monkeypatch.setattr(gd, "read_grid", lambda path: None)
  monkeypatch.setattr(gd, "encode_grid", lambda grid, source, rng=None, **enc: x0.copy())
  augment = {**AUG, "slot_shuffle_block": block}
  if target is not None:
    augment["slot_shuffle_target"] = target
  return gd.GridTargetDataset(Inner(src), {3: "x.h5"}, augment=augment), x0


def test_default_and_shuffled_keep_the_old_behaviour(monkeypatch):
  for target in (None, "shuffled"):
    ds, x0 = dataset(monkeypatch, target)
    assert ds.shuffle_target == "shuffled" and "slot_shuffle_target" not in ds.augment
    torch.manual_seed(0)
    item = ds[0]
    assert "x0_input_raw" not in item
    assert not np.array_equal(item["x0_raw"].float().numpy(), x0.astype(np.float16).astype(np.float32))   # the target is shuffled


def test_canonical_shuffles_the_input_only(monkeypatch):
  ds, x0 = dataset(monkeypatch, "canonical")
  torch.manual_seed(0)
  item = ds[0]
  np.testing.assert_array_equal(item["x0_raw"].numpy(), x0.astype(np.float16))                # the target: as fitted
  shuffled = item["x0_input_raw"].float().numpy()
  assert not np.allclose(shuffled, x0.astype(np.float16).astype(np.float32))                   # the input: moved
  _, src = scene()   # shuffling only moves Gaussians between the scene's valid slots: the invalid slots stay NaN
  valid = src["hit"].numpy()
  assert np.isnan(shuffled[:, ~valid]).sum() == np.isnan(x0[:, ~valid]).sum()


def test_canonical_input_is_the_same_gaussians(monkeypatch):
  """Render-invariant: the shuffled input holds the same Gaussians as the target (test_augment's comparison)."""
  from test_augment import gaussians
  ds, x0 = dataset(monkeypatch, "canonical")
  torch.manual_seed(1)
  item = ds[0]
  _, src = scene()
  a = gaussians(item["x0_input_raw"].float().numpy(), src)
  b = gaussians(item["x0_raw"].float().numpy(), src)
  np.testing.assert_allclose(a, b, rtol=0, atol=2e-3)   # (half precision in transit)


def test_block_size_is_passed_through(monkeypatch):
  ds, _ = dataset(monkeypatch, "canonical", block=3)
  assert ds.augment["slot_shuffle_block"] == 3


def test_unknown_target_raises(monkeypatch):
  with pytest.raises(ValueError, match="slot_shuffle_target"):
    dataset(monkeypatch, "nope")


def test_training_loss_noises_the_input_grid_and_scores_the_target():
  m = shell()
  feats, hit, x0_raw = data()
  x0_in = x0_raw + torch.randn(x0_raw.shape, generator=torch.Generator().manual_seed(7))
  seen = {}
  real = m.head.forward
  m.head.forward = lambda f, x_net, t, *a, **k: seen.setdefault("x_net", x_net.clone()) is not None and real(f, x_net, t, *a, **k)
  t = torch.full((x0_raw.shape[0], x0_raw.shape[2]), 0.25)
  seeds = list(range(x0_raw.shape[0]))
  with torch.no_grad():
    loss_in, *_ = m.training_loss(feats, hit, x0_raw, seeds=seeds, t=t, x0_input_raw=x0_in)
    x_net_in = seen.pop("x_net")
    loss_same, *_ = m.training_loss(feats, hit, x0_raw, seeds=seeds, t=t)
    x_net_same = seen.pop("x_net")
    loss_swap, *_ = m.training_loss(feats, hit, x0_in, seeds=seeds, t=t)
    x_net_swap = seen.pop("x_net")
  torch.testing.assert_close(x_net_in, x_net_swap, rtol=0, atol=0)   # the input: built from x0_input_raw
  assert not torch.equal(x_net_in, x_net_same)
  assert not torch.isclose(loss_in, loss_swap)                        # the target: still x0_raw


def test_none_is_the_old_call():
  m = shell()
  feats, hit, x0_raw = data()
  t = torch.full((x0_raw.shape[0], x0_raw.shape[2]), 0.25)
  seeds = list(range(x0_raw.shape[0]))
  with torch.no_grad():
    a = m.training_loss(feats, hit, x0_raw, seeds=seeds, t=t)[0]
    b = m.training_loss(feats, hit, x0_raw, seeds=seeds, t=t, x0_input_raw=None)[0]
  torch.testing.assert_close(a, b, rtol=0, atol=0)
