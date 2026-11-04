"""module.SameDirCheckpointIO: checkpoints are written next to their destination (never through the system temp dir), renamed
into place atomically, loadable for a resume; a failed save leaves no temp file and the previous checkpoint untouched."""
import os
import tempfile

import lightning.pytorch as pl
import pytest
import torch

from module import SameDirCheckpointIO


class Tiny(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.w = torch.nn.Linear(2, 1)

  def training_step(self, batch, batch_idx):
    return self.w(batch[0]).sum()

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=0.1)


def data():
  return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=2)


def trainer(tmp_path, steps):
  return pl.Trainer(max_steps=steps, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, default_root_dir=str(tmp_path), plugins=[SameDirCheckpointIO()])


def test_saves_beside_the_destination_and_resumes(tmp_path, monkeypatch):
  def no_system_temp(*a, **k):
    raise AssertionError("the system temp dir was used")
  monkeypatch.setattr(tempfile, "mkstemp", no_system_temp)
  t = trainer(tmp_path, 2)
  t.fit(Tiny(), data())
  path = tmp_path / "out" / "c.ckpt"                     # a directory that doesn't exist yet
  t.save_checkpoint(path)
  assert path.is_file() and sorted(os.listdir(path.parent)) == ["c.ckpt"]   # no temp file left behind
  resumed = trainer(tmp_path, 3)
  resumed.fit(Tiny(), data(), ckpt_path=str(path))
  assert resumed.global_step == 3


def test_failed_save_keeps_the_old_checkpoint_and_no_temp(tmp_path, monkeypatch):
  t = trainer(tmp_path, 1)
  t.fit(Tiny(), data())
  path = tmp_path / "c.ckpt"
  t.save_checkpoint(path)
  before = path.read_bytes()

  def full_disk(obj, f, *a, **k):
    f.write(b"partial")
    raise OSError(28, "No space left on device")
  monkeypatch.setattr(torch, "save", full_disk)
  with pytest.raises(OSError, match="No space"):
    t.save_checkpoint(path)
  assert path.read_bytes() == before and sorted(os.listdir(tmp_path)) == ["c.ckpt"]
