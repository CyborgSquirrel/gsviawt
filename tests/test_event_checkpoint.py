"""EventCheckpoint's keep_last: only the newest n of its own checkpoints are kept, the old ones deleted only after a new one was
written, and a resumed run (into a fresh dir) goes on counting and prunes only in its own dir."""
import os

import lightning.pytorch as pl
import pytest
import torch

from module import EventCheckpoint


class Tiny(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.w = torch.nn.Linear(2, 1)

  def training_step(self, batch, batch_idx):
    return self.w(batch[0]).sum()

  def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=0.1)


def data():
  return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(16, 2)), batch_size=2)


def trainer(root, max_steps, callbacks):
  return pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, default_root_dir=str(root), max_steps=max_steps, callbacks=callbacks)


def checkpoint(keep_last):
  return EventCheckpoint(trigger_event={"every_n_steps": 1}, filename="c{count}-s{step}.ckpt", keep_last=keep_last)


def test_keeps_the_newest_n(tmp_path):
  (tmp_path / "final.ckpt").write_bytes(b"x")       # not this callback's: left alone
  cb = checkpoint(2)
  trainer(tmp_path, 5, [cb]).fit(Tiny(), data())
  assert sorted(os.listdir(tmp_path)) == ["c4-s4.ckpt", "c5-s5.ckpt", "final.ckpt"]
  assert cb.count == 5 and cb.saved == [str(tmp_path / "c4-s4.ckpt"), str(tmp_path / "c5-s5.ckpt")]


def test_keep_last_null_keeps_all(tmp_path):
  trainer(tmp_path, 5, [checkpoint(None)]).fit(Tiny(), data())
  assert sorted(os.listdir(tmp_path)) == [f"c{i}-s{i}.ckpt" for i in range(1, 6)]


def test_resume_counts_on_and_prunes_only_its_own_dir(tmp_path):
  first, second = tmp_path / "first", tmp_path / "second"
  trainer(first, 3, [checkpoint(2)]).fit(Tiny(), data())
  assert sorted(os.listdir(first)) == ["c2-s2.ckpt", "c3-s3.ckpt"]
  resumed = checkpoint(2)
  trainer(second, 6, [resumed]).fit(Tiny(), data(), ckpt_path=str(first / "c3-s3.ckpt"))
  assert resumed.count == 6
  assert sorted(os.listdir(first)) == ["c2-s2.ckpt", "c3-s3.ckpt"]   # the earlier run's: never deleted
  assert sorted(os.listdir(second)) == ["c5-s5.ckpt", "c6-s6.ckpt"]
  assert resumed.saved == [str(second / "c5-s5.ckpt"), str(second / "c6-s6.ckpt")]


def test_old_checkpoint_state_without_saved_list(tmp_path):
  cb = checkpoint(2)
  cb.load_state_dict({"count": 7})
  assert cb.count == 7 and cb.saved == []


def test_failing_save_keeps_the_old_ones_and_raises(tmp_path):
  cb = checkpoint(2)
  t = trainer(tmp_path, 3, [cb])
  t.fit(Tiny(), data())
  before = sorted(os.listdir(tmp_path))
  saved = list(cb.saved)

  def full_disk(path, *a, **k):
    raise OSError(28, "No space left on device")

  t.save_checkpoint = full_disk
  with pytest.raises(OSError, match="No space left"):
    cb.on_train_batch_end(t, t.lightning_module, None, None, 0)
  assert sorted(os.listdir(tmp_path)) == before and cb.saved == saved


def test_already_deleted_file_does_not_crash(tmp_path):
  cb = checkpoint(1)
  t = trainer(tmp_path, 2, [cb])
  t.fit(Tiny(), data())
  cb.saved.insert(0, str(tmp_path / "gone.ckpt"))   # tracked, but already removed by hand
  cb.on_train_batch_end(t, t.lightning_module, None, None, 0)   # due again (global_step 2): writes c3-s2, prunes the rest
  assert len(cb.saved) == 1 and os.path.exists(cb.saved[0])


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_keep_last_validated(bad):
  with pytest.raises(ValueError, match="keep_last"):
    EventCheckpoint(trigger_event=None, keep_last=bad)
