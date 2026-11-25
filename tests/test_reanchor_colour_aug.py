"""train_reanchor data.colour_aug: the transform's semantics (colour' = colour[p] * g on A's RGB colours and on the premultiplied
GT alike), the draws (6 permutations, gains in range, seeded), the exactness that makes it a valid augmentation (A's grid rendered
with the transformed colours == the transform of A's original render, premultiplied, before any background, to float precision:
gsplat's per-Gaussian colour max(SH_C0 sh + 0.5, 0) has no upper clamp and commutes with gains > 0, the composite is linear in the
colours; the only >1 clamp is the loss's composite clamp, applied to the target as well when the augmentation is on), and a tiny
full-mode run: off (default) == the key absent, step for step; on runs and changes the training loss but not the step-0 eval."""
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import reanchor as ra
import train_reanchor as tr
from gaussians import SH_C0
from test_reanchor_full import FULL_TEST, GRIDS79, NEW100_H5, SHARD2_H5, has_gpu_data


def test_rows_and_images_semantics():
  rng = np.random.default_rng(0)
  rgb = torch.tensor(rng.uniform(-0.2, 1.2, (50, 3)), dtype=torch.float64)
  rows = torch.tensor(rng.normal(size=(50, 14)), dtype=torch.float64)
  rows[:, ra.SH_DC] = (rgb - 0.5) / SH_C0
  p, g = np.array([2, 0, 1]), np.array([0.6, 1.3, 1.1])
  out = ra.colour_aug_rows(rows, p, g)
  keep = [c for c in range(14) if c not in range(14)[ra.SH_DC]]
  assert torch.equal(out[:, keep], rows[:, keep])                       # only the colour channels change
  want = torch.stack([rgb[:, 2] * 0.6, rgb[:, 0] * 1.3, rgb[:, 1] * 1.1], 1)
  assert torch.allclose(out[:, ra.SH_DC] * SH_C0 + 0.5, want, atol=1e-12, rtol=0)
  assert torch.equal(rows[:, ra.SH_DC], (rgb - 0.5) / SH_C0)            # input untouched
  img = torch.tensor(rng.uniform(0, 1, (4, 3, 5, 6)), dtype=torch.float32)
  oi = ra.colour_aug_images(img, p, g)
  for c, (src, gain) in enumerate(zip(p, g)):
    assert torch.allclose(oi[:, c], img[:, src] * float(gain), atol=1e-7, rtol=0)
  ident = ra.colour_aug_rows(rows, np.arange(3), np.ones(3))
  assert torch.allclose(ident, rows, atol=1e-12, rtol=0)


def test_draws():
  rng = np.random.default_rng(1)
  d = [ra.draw_colour_aug(rng, (0.5, 1.5), True) for _ in range(6000)]
  perms = [tuple(p) for p, _ in d]
  counts = np.array([perms.count(q) for q in set(perms)])
  assert len(counts) == 6 and counts.min() > 850 and counts.max() < 1150
  gs = np.stack([g for _, g in d])
  assert gs.min() >= 0.5 and gs.max() < 1.5 and abs(gs.mean() - 1.0) < 0.02
  assert all(tuple(p) == (0, 1, 2) for p, _ in (ra.draw_colour_aug(rng, (0.5, 1.5), False) for _ in range(20)))
  a = [ra.draw_colour_aug(np.random.default_rng([0, 0, 0xC0A6])) for _ in range(5)]
  b = [ra.draw_colour_aug(np.random.default_rng([0, 0, 0xC0A6])) for _ in range(5)]
  assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_colour_aug_rng_switch():
  base = {"train": {"seed": 0}, "data": {}}
  assert tr.colour_aug_rng(OmegaConf.create(base), 0) is None                                   # key absent
  base["data"]["colour_aug"] = {"enabled": False, "gain": [0.5, 1.5], "permute": True}
  assert tr.colour_aug_rng(OmegaConf.create(base), 0) is None                                   # the default
  assert OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf/train_reanchor.yaml")).data.colour_aug.enabled is False
  base["data"]["colour_aug"]["enabled"] = True
  rng, gain, perm = tr.colour_aug_rng(OmegaConf.create(base), 0)
  assert gain == [0.5, 1.5] and perm is True
  base["data"]["colour_aug"]["gain"] = [1.5, 0.5]
  with pytest.raises(SystemExit):
    tr.colour_aug_rng(OmegaConf.create(base), 0)


def test_photometric_clamp_target():
  dssim = tr.DSSIMLoss(reduction="none")
  cfg = OmegaConf.create({"loss": {"l1_weight": 1.0, "dssim_weight": 0.85}})
  g = torch.Generator().manual_seed(0)
  rgb, alpha = torch.rand(2, 3, 16, 16, generator=g) * 1.4, torch.rand(2, 1, 16, 16, generator=g)
  gt = rgb.clone()
  bg = torch.rand(2, 3, 1, 1, generator=g)
  off = tr.photometric(dssim, cfg, rgb, alpha, gt, alpha, bg)
  on = tr.photometric(dssim, cfg, rgb, alpha, gt, alpha, bg, clamp_target=True)
  assert float(off[1]) > 1e-3                 # unclamped target > 1 cannot be reached by the clamped render
  assert float(on[1]) == 0.0                  # the render of a GT-explaining A matches the clamped target exactly
  gt1 = gt.clamp(0, 1) * alpha                # targets within [0, 1]: the clamp is a no-op
  a = tr.photometric(dssim, cfg, rgb, alpha, gt1, alpha, bg)
  b = tr.photometric(dssim, cfg, rgb, alpha, gt1, alpha, bg, clamp_target=True)
  assert all(torch.equal(x, y) for x, y in zip(a, b))


@has_gpu_data
def test_render_transformed_a_equals_transformed_render():
  """A's real grid (mesh 79, in B's frame) rendered at 4 views with transformed colours == the transform of its original render
  (premultiplied, before the background). Includes Gaussians whose raw colour is < 0 (clamped to 0 by gsplat): max(x g, 0) =
  max(x, 0) g. No upper clamp exists in the renderer (it may exceed 1); the only >1 clamp is photometric's composite clamp."""
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  data = ra.MeshData(NEW100_H5, GRIDS79, 79, [3, 17, 40], device="cuda")
  views = [4, 17, 30, 44]
  rng = np.random.default_rng(3)
  n_neg = 0
  for a, b in ((3, 17), (40, 3), (17, 40)):
    rows, xyz = data.a_rows_in_b(a, b)
    n_neg += int(((rows[:, ra.SH_DC] * SH_C0 + 0.5) < 0).sum())
    vm = data.viewmats(b, views)
    rgb, alpha = ra.render(ra.raw_gaussians(rows, xyz), vm, data.K[views], data.W, data.H)
    for _ in range(3):
      p, g = ra.draw_colour_aug(rng)
      rgb2, alpha2 = ra.render(ra.raw_gaussians(ra.colour_aug_rows(rows, p, g), xyz), vm, data.K[views], data.W, data.H)
      want = ra.colour_aug_images(rgb, p, g)
      assert torch.equal(alpha2, alpha)
      err = (rgb2 - want).abs().max().item()
      assert err < 1e-5, err
      assert float(rgb.amax()) > 0.5
  # the real grids have no raw colour < 0 (n_neg); force some to exercise gsplat's clamp_min(., 0): max(x g, 0) == max(x, 0) g
  rows = rows.clone()
  sel = torch.rand(len(rows), device=rows.device, generator=torch.Generator("cuda").manual_seed(0)) < 0.1
  rows[sel, ra.SH_DC.start] = (-0.3 - 0.5) / SH_C0
  rgb, alpha = ra.render(ra.raw_gaussians(rows, xyz), vm, data.K[views], data.W, data.H)
  p, g = np.array([1, 2, 0]), np.array([1.4, 0.55, 1.2])
  rgb2, _ = ra.render(ra.raw_gaussians(ra.colour_aug_rows(rows, p, g), xyz), vm, data.K[views], data.W, data.H)
  assert (rgb2 - ra.colour_aug_images(rgb, p, g)).abs().max().item() < 1e-5
  print(f"{n_neg} Gaussians with raw colour < 0 among the tested A grids")


@pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isdir(f"{FULL_TEST}/new100") and os.path.exists(SHARD2_H5)),
                    reason="needs CUDA + bla/reanchor/full_test grids + the shard2 h5")
def test_full_mode_off_is_unchanged_on_changes_training_only(tmp_path):
  g = f"{FULL_TEST}/new100/anchored_wd0.3_rbg/grids_sparse"
  base = [sys.executable, "train_reanchor.py", "--config-name", "train_reanchor_full", f"data.grids_dir={g}", "data.train_meshes=[0,1,79]",
          "data.workers=2", "eval.sets.train_heldout.n_meshes=1", "eval.sets.train_heldout.pairs_per_mesh=1", "eval.sets.val.meshes=[3]",
          "eval.sets.val.pairs_per_mesh=1", f"eval.sets.shard2.render_h5={SHARD2_H5}",
          f"eval.sets.shard2.grids_dir={FULL_TEST}/shard2/anchored_wd0.3_rbg/grids_sparse", "eval.sets.shard2.meshes=[0]",
          "eval.sets.shard2.pairs_per_mesh=1", "model.dim=128", "model.num_blocks=2", "train.pairs_per_step=2", "train.views_per_pair=2",
          "train.max_steps=4", "train.eval_every=10000", "train.log_every=1", "wandb.mode=disabled"]
  cwd = os.path.dirname(os.path.dirname(__file__))

  def run(name, *extra):
    r = subprocess.run(base + [f"output_dir={tmp_path / name}"] + list(extra), capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, r.stderr[-3000:]
    out = r.stderr + r.stdout
    losses = [float(m) for m in re.findall(r"step \d+ loss ([0-9.]+)", out)]
    ev = re.findall(r"eval step 0 \(start.*?\): (.*)", out)
    assert len(losses) == 4 and len(ev) == 1, out[-3000:]
    return losses, ev[0], out

  off, ev_off, _ = run("off")
  absent, ev_absent, _ = run("absent", "~data.colour_aug")
  on, ev_on, out_on = run("on", "data.colour_aug.enabled=true")
  assert off[0] == absent[0] and ev_off == ev_absent
  assert np.allclose(off, absent, rtol=0, atol=2e-4)       # later steps: identical up to nondeterministic backward kernels
  assert "colour augmentation: per training sample" in out_on
  assert ev_on == ev_off                                   # eval untouched (same init, no augmentation in eval)
  assert on[0] != off[0]
