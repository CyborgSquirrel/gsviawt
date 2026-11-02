"""train_gs config: no Lightning ModelCheckpoint (only event_checkpoint and the TrainClock's final.ckpt), and Lightning's
default checkpointing is off -- a short fit with the config's trainer flags writes no checkpoint file."""
import glob

import hydra
import lightning.pytorch as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import train_gs  # noqa: F401  (registers the config's `eval` resolver)
from conftest import ROOT


def compose_train_gs():
  with initialize_config_dir(config_dir=f"{ROOT}/conf", version_base=None):
    return compose(config_name="train_gs", overrides=["output_dir=/tmp/unused"])


def test_no_model_checkpoint_callback():
  cbs = hydra.utils.instantiate(compose_train_gs().callbacks)
  assert "ModelCheckpoint" not in {type(v).__name__ for v in cbs.values() if v is not None}


def test_trainer_flags_write_no_checkpoint(tmp_path):
  class M(pl.LightningModule):
    def __init__(self):
      super().__init__()
      self.l = torch.nn.Linear(2, 1)

    def training_step(self, b, i):
      return self.l(b[0]).pow(2).mean()

    def configure_optimizers(self):
      return torch.optim.SGD(self.parameters(), 0.1)

  kw = OmegaConf.to_container(compose_train_gs().trainer)
  kw.update(max_epochs=2, default_root_dir=str(tmp_path), accelerator="cpu", logger=False, enable_progress_bar=False)
  kw.pop("check_val_every_n_epoch", None)
  data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=4)
  pl.Trainer(**kw).fit(M(), data)
  assert not glob.glob(f"{tmp_path}/**/*.ckpt", recursive=True)
