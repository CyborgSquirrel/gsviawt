"""Paired validations: at its k-th validation every run scores the same volumes (val dataloaders reseeded per validation index and
set), with the same target views per volume (seeded per volume, Python's global random state left as it was)."""
import random
import types

import numpy as np
import torch
from omegaconf import OmegaConf

import gs_diffusion as gd
import train_gs


def datamodule(seed=42):
  dm = types.SimpleNamespace(cfg=OmegaConf.create({"seed": seed, "loader": {"batch_size": 4}}),
                             val_datasets={"val_meshes": list(range(50)), "val_views_sval": list(range(80))})
  for m in ("validation_seed", "reseed_validation", "val_dataloader"):
    setattr(dm, m, (lambda f: lambda *a: f(dm, *a))(getattr(train_gs.GSDataModule, m)))
  return dm


def order(loader):
  return [int(i) for i in loader.sampler]          # the shuffled volume order of one validation


def test_same_volumes_at_the_same_validation_across_runs():
  a, b = datamodule(), datamodule()
  la, lb = a.val_dataloader(), b.val_dataloader()
  torch.rand(123)                                   # the runs' global RNGs are in different states
  a.reseed_validation(3); b.reseed_validation(3)
  first = [order(x) for x in la]
  assert first == [order(x) for x in lb]
  a.reseed_validation(4)
  assert [order(x) for x in la] != first            # the next validation: a different sample
  a.reseed_validation(3)
  assert [order(x) for x in la] == first            # a function of the index only
  assert a.validation_seed(3, "val_meshes") != a.validation_seed(3, "val_views_sval")
  assert datamodule(seed=7).validation_seed(3, "val_meshes") != a.validation_seed(3, "val_meshes")


def test_seeded_items_draw_the_same_targets_and_keep_the_global_random_state(monkeypatch):
  class Column(list):
    def to_list(self):
      return list(self)

  class Inner:
    volumes = types.SimpleNamespace(df={"view_idx": Column([5, 6])})
    def __len__(self):
      return 2
    def __getitem__(self, i):
      return {"source_view": [5, 6][i], "targets": random.sample(range(1000), 3), "source": {}}
  monkeypatch.setattr(gd, "load_grid", lambda path: ({}, {}))
  monkeypatch.setattr(gd, "encode_grid", lambda grid, source, **k: np.zeros((1,), np.float32))
  val = gd.GridTargetDataset(Inner(), {5: "a.h5", 6: "b.h5"}, seeded=True)
  random.seed(0); before = random.random(); random.seed(0)
  t1 = val[0]["targets"]
  after = random.random()
  assert before == after                            # the global stream is untouched
  random.seed(99)
  assert val[0]["targets"] == t1 and val[1]["targets"] != t1
  train = gd.GridTargetDataset(Inner(), {5: "a.h5", 6: "b.h5"})
  assert len({tuple(train[0]["targets"]) for _ in range(5)}) > 1    # training items: fresh draws


def test_eval_can_turn_off_the_target_seeding(monkeypatch):
  """eval_diffusion seeds target views itself (random.seed per volume); with seed_targets off the dataset leaves that alone."""
  class Column(list):
    def to_list(self):
      return list(self)

  class Inner:
    volumes = types.SimpleNamespace(df={"view_idx": Column([5])})
    def __len__(self):
      return 1
    def __getitem__(self, i):
      return {"source_view": 5, "targets": random.sample(range(1000), 3), "source": {}}
  monkeypatch.setattr(gd, "load_grid", lambda path: ({}, {}))
  monkeypatch.setattr(gd, "encode_grid", lambda grid, source, **k: np.zeros((1,), np.float32))
  ds = gd.GridTargetDataset(Inner(), {5: "a.h5"}, seeded=True)
  ds.seed_targets = False
  random.seed("0/val_meshes/0"); a = ds[0]["targets"]
  random.seed("0/val_meshes/0"); b = random.sample(range(1000), 3)
  assert a == b                                     # exactly what the eval's own seed gives
