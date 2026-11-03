"""diffusion.scale_encoding.rot_form: which of each Gaussian's 4 equivalent quaternions (two axes flipped) the encoding stores --
keep / canonical (largest |w|) / random. Always the same Gaussian (covariance unchanged); the default and old configs/checkpoints
mean keep."""
import types

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

import gs_diffusion as gd
import train_gs


def cov(q, s):
  R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
  return (R * s[:, None, :] ** 2) @ R.transpose(0, 2, 1)


def quats(n=2000, seed=0):
  q = Rotation.random(n, random_state=seed).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
  return np.where(q[:, :1] < 0, -q, q)


def test_the_four_forms_are_the_same_gaussian():
  q, s = quats(), np.random.default_rng(0).uniform(0.1, 2, (2000, 3))
  forms = gd.quat_forms(q)
  for k in range(4):
    np.testing.assert_allclose(cov(forms[:, k], s), cov(q, s), atol=1e-5)
  # and they are 4 distinct rotations (not q and -q)
  dots = np.abs((forms[:, :, None] * forms[:, None, :]).sum(-1))
  assert (dots[:, ~np.eye(4, dtype=bool)] < 0.999).mean() > 0.99


def test_canonical_and_random():
  q, s = quats(), np.random.default_rng(1).uniform(0.1, 2, (2000, 3))
  c = gd.set_rot_form(q, "canonical")
  np.testing.assert_allclose(cov(c, s), cov(q, s), atol=1e-5)
  assert (np.abs(c).argmax(1) == 0).all() and (c[:, 0] >= 0).all()          # w is the largest component
  np.testing.assert_allclose(gd.set_rot_form(c, "canonical"), c, atol=1e-6)  # idempotent
  np.testing.assert_allclose(gd.set_rot_form(gd.quat_forms(q)[:, 2], "canonical"), c, atol=1e-6)   # whichever form it starts in
  r = gd.set_rot_form(q, "random", np.random.default_rng(0))
  np.testing.assert_allclose(cov(r, s), cov(q, s), atol=1e-5)
  frac = np.bincount(np.abs(gd.set_rot_form(c, "random", np.random.default_rng(0))).argmax(1), minlength=4) / len(q)
  assert (np.abs(frac - 0.25) < 0.05).all()
  assert gd.set_rot_form(q, "keep") is q
  with pytest.raises(ValueError):
    gd.encode_grid({"layer_valid": np.zeros((2, 2, 1), bool)}, {"hit": None}, rot_form="sorted")


def test_tags_defaults_and_checkpoint_compat():
  assert gd.encoding_tag({"scale_sort": True, "scale_floor_px": 0.1}) == "sort_floor0.1px"   # keep: the existing caches
  assert gd.encoding_tag({"scale_sort": True, "scale_floor_px": 0.1, "rot_form": "canonical"}) == "sort_floor0.1px_rotcanonical"
  assert train_gs.x0_encoding(OmegaConf.create({"diffusion": {"enabled": True}}))["rot_form"] == "keep"

  def module(enc):
    return types.SimpleNamespace(cfg=OmegaConf.create({"diffusion": {"enabled": True, "scale_encoding": enc}}), views_seen=0,
                                 pixels_seen=0, ema=None, spike_skip=None)
  new = {"scale_sort": True, "scale_floor_px": 0.1}
  train_gs.GSLightningModule.on_load_checkpoint(module(new), {"x0_encoding": dict(new)})   # saved before rot_form existed: keep
  with pytest.raises(SystemExit, match="rot_form=keep"):
    train_gs.GSLightningModule.on_load_checkpoint(module({**new, "rot_form": "canonical"}), {"x0_encoding": dict(new)})
