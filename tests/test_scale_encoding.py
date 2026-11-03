"""diffusion.scale_encoding (gs_diffusion.sort_scale_axes, encode_grid's scale_sort / scale_floor_px, encoding_tag,
GridTargetDataset's encoding) on synthetic data."""
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import gs_diffusion as gd
from gs_dataset import OPENGL_TO_OPENCV


def cov(q_wxyz, s):
  r = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_matrix()
  return r @ (s[:, :, None] ** 2 * np.eye(3)) @ r.transpose(0, 2, 1)


def test_sort_scale_axes_preserves_covariance():
  rng = np.random.default_rng(0)
  s = np.exp(rng.normal(-5, 2, (2000, 3))).astype(np.float32)
  q = Rotation.random(2000, random_state=0).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
  s2, q2 = gd.sort_scale_axes(s, q)
  assert (np.diff(s2, axis=1) <= 0).all()
  c, c2 = cov(q, s), cov(q2, s2)
  assert (np.abs(c2 - c).max(axis=(1, 2)) / s.max(1) ** 2).max() < 1e-4
  assert np.allclose(np.linalg.norm(q2, axis=1), 1, atol=1e-5)


def synthetic(h=4, w=5, layers=6, seed=1):
  rng = np.random.default_rng(seed)
  hit = rng.random((layers, h, w)) < 0.7
  pose = np.eye(4, dtype=np.float32)
  pose[:3, 3] = [0, 0, 0]
  c2w = pose @ OPENGL_TO_OPENCV
  xyz_cam = (rng.normal(size=(layers, h, w, 3)) + [0, 0, 3]).astype(np.float32)
  world = xyz_cam @ c2w[:3, :3].T + c2w[:3, 3]
  t = lambda a: np.moveaxis(a, 0, 2)
  grid = {"layer_valid": t(hit), "gaussian_means": t(world.astype(np.float32)),
          "gaussian_quats": t(Rotation.random(layers * h * w, random_state=seed).as_quat()[:, [3, 0, 1, 2]]
                              .reshape(layers, h, w, 4).astype(np.float32)),
          "gaussian_scales": t(np.exp(rng.normal(-8, 2, (layers, h, w, 3))).astype(np.float32)),
          "gaussian_opacities": t(rng.uniform(0.05, 0.95, (layers, h, w)).astype(np.float32)),
          "gaussian_colors": t(rng.uniform(0, 1, (layers, h, w, 3)).astype(np.float32))}
  source = {"hit": torch.from_numpy(hit), "xyz_cam": torch.from_numpy(np.where(hit[..., None], xyz_cam, np.nan)),
            "pose_gl": torch.from_numpy(pose), "K_depth": torch.tensor([[500.0, 0, 2], [0, 500.0, 2], [0, 0, 1]])}
  return grid, source, xyz_cam, hit


def test_encode_default_unchanged_and_sort_floor():
  grid, source, xyz_cam, hit = synthetic()
  a = gd.encode_grid(grid, source)
  assert np.array_equal(a, gd.encode_grid(grid, source, scale_sort=False, scale_floor_px=None))
  b = gd.encode_grid(grid, source, scale_sort=True, scale_floor_px=0.1)
  s = np.exp(b[1:4][:, hit].T)
  assert (np.diff(s, axis=1) <= 1e-6 * s[:, :1]).all()                         # sorted largest -> smallest
  footprint = xyz_cam[hit][:, 2] / 500.0
  assert (s >= 0.1 * footprint[:, None] * (1 - 1e-5)).all()                     # floored at 0.1 px
  for ch in (0, 8, 9, 10, 11, 12, 13):                                         # opacity, color, offset untouched
    assert np.array_equal(a[ch], b[ch])
  assert np.isfinite(b).all() and (b[:, ~hit] == 0).all()


def test_encoding_tag_and_dataset_encoding():
  assert gd.encoding_tag(None) == "" and gd.encoding_tag({}) == ""
  assert gd.encoding_tag({"scale_sort": True, "scale_floor_px": 0.1}) == "sort_floor0.1px"
  assert gd.encoding_tag({"scale_floor_px": 0.3}) == "floor0.3px"

  class Inner:                                   # GridTargetDataset only needs .volumes.df["view_idx"] at construction
    class volumes:
      import polars as pl
      df = pl.DataFrame({"view_idx": [3]})
  ds = gd.GridTargetDataset(Inner(), {3: "unused.h5"}, {"scale_sort": True})   # overriding a default must not raise
  assert ds.encoding == {"scale_sort": True, "scale_floor_px": None, "rot_form": "keep"}


def test_default_is_sort_floor_and_checkpoints_keep_their_encoding():
  """conf/train_gs.yaml defaults to sort + 0.1 px floor; a config without the block (older runs, eval's saved configs) means
  off; a checkpoint records its encoding and refuses to resume under another one (checkpoints without the key: off)."""
  import types
  from omegaconf import OmegaConf
  import train_gs
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  conf = OmegaConf.load(os.path.join(root, "conf", "train_gs.yaml"))
  assert train_gs.x0_encoding(conf) == {"scale_sort": True, "scale_floor_px": 0.1, "rot_form": "keep"}
  off = {"scale_sort": False, "scale_floor_px": None, "rot_form": "keep"}
  assert train_gs.x0_encoding(OmegaConf.create({"diffusion": {"enabled": True}})) == off

  def module(enc):
    d = {"enabled": True, "scale_encoding": enc} if enc is not None else {"enabled": True}
    return types.SimpleNamespace(cfg=OmegaConf.create({"diffusion": d}), views_seen=0, pixels_seen=0, ema=None, spike_skip=None)
  M = train_gs.GSLightningModule
  ckpt = {}
  M.on_save_checkpoint(module({"scale_sort": True, "scale_floor_px": 0.1}), ckpt)
  assert ckpt["x0_encoding"] == {"scale_sort": True, "scale_floor_px": 0.1, "rot_form": "keep"}
  M.on_load_checkpoint(module({"scale_sort": True, "scale_floor_px": 0.1}), dict(ckpt))
  with pytest.raises(SystemExit, match="scale_sort=true diffusion.scale_encoding.scale_floor_px=0.1"):
    M.on_load_checkpoint(module(off), dict(ckpt))
  M.on_load_checkpoint(module(off), {})                                   # an old checkpoint, resumed with the encoding off
  with pytest.raises(SystemExit, match="scale_floor_px=null"):
    M.on_load_checkpoint(module({"scale_sort": True, "scale_floor_px": 0.1}), {})
