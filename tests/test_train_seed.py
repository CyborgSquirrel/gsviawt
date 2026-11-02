"""train_seed: seeds torch only (the splits keep `seed`); null or absent falls back to `seed`."""
from omegaconf import OmegaConf

import train_gs


def test_torch_seed():
  assert train_gs.torch_seed(OmegaConf.create({"seed": 42, "train_seed": None})) == 42
  assert train_gs.torch_seed(OmegaConf.create({"seed": 42, "train_seed": 43})) == 43
  assert train_gs.torch_seed(OmegaConf.create({"seed": 42})) == 42                 # configs from before train_seed


def test_config_default_is_null():
  from hydra import compose, initialize_config_dir
  from conftest import ROOT
  with initialize_config_dir(config_dir=f"{ROOT}/conf", version_base=None):
    cfg = compose(config_name="train_gs", overrides=["output_dir=/tmp/unused"])
  assert cfg.train_seed is None and train_gs.torch_seed(cfg) == cfg.seed
