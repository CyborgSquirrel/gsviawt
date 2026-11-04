"""Paired validations: at its k-th validation every run scores the same volumes (val dataloaders reseeded per validation index and
set), with the same target views per volume (seeded per volume, Python's global random state left as it was)."""
import random
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import gs_diffusion as gd
import train_gs


def datamodule(seed=42):
  dm = types.SimpleNamespace(cfg=OmegaConf.create({"seed": seed, "loader": {"batch_size": 4}}),
                             val_datasets={"val_meshes": list(range(50)), "val_views_sval": list(range(80))})
  dm.lightning_runs = 0
  for m in ("validation_seed", "reseed_validation", "reseed_lightning", "val_dataloader"):
    setattr(dm, m, (lambda f: lambda *a, **k: f(dm, *a, **k))(getattr(train_gs.GSDataModule, m)))
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


def test_validation_counter_survives_a_resume(tmp_path):
  """EventValidation's run counter (the k of reseed_validation) is kept in checkpoints, one entry per validation, so a resumed
  run goes on with validation k+1 instead of repeating the samples of validations 1, 2, ..."""
  import lightning.pytorch as pl
  from module import EventValidation

  class Tiny(pl.LightningModule):
    def __init__(self):
      super().__init__()
      self.w = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
      return self.w(batch[0]).sum()

    def configure_optimizers(self):
      return torch.optim.SGD(self.parameters(), lr=0.1)

  data = lambda: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=2)
  kw = dict(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            default_root_dir=str(tmp_path))
  make = lambda: [EventValidation(name="big", trigger_event=None), EventValidation(name="big_ema", trigger_event=None, ema=True)]
  cbs = make()
  assert cbs[0].state_key != cbs[1].state_key
  trainer = pl.Trainer(max_steps=2, callbacks=cbs, **kw)
  trainer.fit(Tiny(), data())
  cbs[0].runs, cbs[1].runs = 3, 5
  path = str(tmp_path / "c.ckpt")
  trainer.save_checkpoint(path)
  resumed = make()
  pl.Trainer(max_steps=3, callbacks=resumed, **kw).fit(Tiny(), data(), ckpt_path=path)
  assert (resumed[0].runs, resumed[1].runs) == (3, 5)


def test_checkpoints_come_after_the_validations(tmp_path):
  """A checkpoint written at a step where a validation also runs must hold its counter as it is after it: the config lists
  event_checkpoint after the validations, and EventCheckpoint refuses to start before an EventValidation."""
  import lightning.pytorch as pl
  from module import EventCheckpoint, EventValidation
  names = list(OmegaConf.load("conf/train_gs.yaml").callbacks)
  assert all(names.index("event_checkpoint") > i for i, n in enumerate(names) if n.startswith("validation_"))

  class Tiny(pl.LightningModule):
    def __init__(self):
      super().__init__()
      self.w = torch.nn.Linear(2, 1)
    def training_step(self, batch, batch_idx):
      return self.w(batch[0]).sum()
    def configure_optimizers(self):
      return torch.optim.SGD(self.parameters(), lr=0.1)

  def fit(callbacks):
    pl.Trainer(max_steps=1, callbacks=callbacks, accelerator="cpu", logger=False, enable_checkpointing=False,
               enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp_path)).fit(
      Tiny(), torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(4, 2)), batch_size=2))
  v, c = EventValidation(name="big", trigger_event=None), EventCheckpoint(trigger_event=None)
  fit([v, c])                                                   # the right order runs
  with pytest.raises(RuntimeError, match="after every EventValidation"):
    fit([EventCheckpoint(trigger_event=None), EventValidation(name="big", trigger_event=None)])

def test_lightnings_own_validations_are_paired_and_resume(tmp_path):
  """Lightning's own validations (sized by limit_val_batches) score the same volumes in every run at its k-th one, whatever the
  global RNG did before; a different sample at k+1; the sanity check doesn't count; the count is kept in checkpoints."""
  import lightning.pytorch as pl

  class DM(pl.LightningDataModule):
    def __init__(self):
      super().__init__()
      self.cfg = OmegaConf.create({"seed": 42, "loader": {"batch_size": 4}})
      self.val_datasets = {k: [{"i": i} for i in range(n)] for k, n in (("val_meshes", 50), ("val_views_sval", 80))}
      self.validation, self.lightning_runs = "lightning", 0
    train_dataloader = lambda self: torch.utils.data.DataLoader(torch.randn(40, 2), batch_size=2)
  for m in ("validation_seed", "reseed_validation", "reseed_lightning", "state_dict", "load_state_dict", "val_dataloader"):
    setattr(DM, m, getattr(train_gs.GSDataModule, m))

  class Tiny(pl.LightningModule):
    def __init__(self):
      super().__init__()
      self.w, self.seen = torch.nn.Linear(2, 1), []
    def training_step(self, batch, i):
      torch.rand(7)   # global RNG consumption that differs between "runs"
      return self.w(batch).sum()
    def validation_step(self, batch, i, dataloader_idx=0):
      self.seen.append(("sanity" if self.trainer.sanity_checking else self.global_step, dataloader_idx, batch["i"].tolist()))
    def configure_optimizers(self):
      return torch.optim.SGD(self.parameters(), lr=0.1)
    on_validation_end = train_gs.GSLightningModule.on_validation_end

  def run(junk, steps=10, ckpt=None, save=None):
    torch.manual_seed(junk)
    random.seed(junk)
    m, dm = Tiny(), DM()
    t = pl.Trainer(max_steps=steps, val_check_interval=5, limit_val_batches=1, num_sanity_val_steps=1, accelerator="cpu",
                   logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                   default_root_dir=str(tmp_path))
    t.fit(m, dm, ckpt_path=ckpt)
    if save:
      t.save_checkpoint(save)
    return m.seen, dm

  a, dm_a = run(1)
  b, _ = run(2)
  assert a == b and dm_a.lightning_runs == 2
  by = lambda seen, when: [x[2] for x in seen if x[0] == when]
  assert by(a, "sanity") == by(a, 5) != by(a, 10)       # sanity = the first real one; the next one a new sample
  path = str(tmp_path / "half.ckpt")
  run(3, steps=5, save=path)                             # stop after the first validation, resume to the second
  resumed, dm_r = run(4, steps=10, ckpt=path)
  assert by(resumed, 10) == by(a, 10) and dm_r.lightning_runs == 2
