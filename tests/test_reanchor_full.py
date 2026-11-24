"""train_reanchor data.dataset full: the streaming sampler (one mesh per sample, A != B, only training meshes), the mesh split (==
train_gs's), the unseen-mesh check, the streamed pair == MeshData's (bit for bit, model inputs included), deterministic eval pairs,
the LR schedule (one cycle unchanged; cycles on the training clock) and a stop / resume run (clock, LR, troughs continue)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import reanchor as ra
import train_reanchor as tr

NEW100_H5 = "/app/bla/new100/trellis_new100_50views_eevee_504.h5"
SHARD2_H5 = "/app/bla/shard2/trellis_shard2_50views_eevee_504.h5"
GRIDS79 = "/app/bla/reanchor/grids79"
FULL_TEST = "/app/bla/reanchor/full_test"          # rbg grids_sparse of new100 meshes 0, 1, 3, 79 and shard2 meshes 0, 1
VAL_MESHES = [3, 7, 8, 34, 37, 42, 91, 92, 95, 98]  # gs_dataset.split_meshes_by_fraction(new100, 0.1, seed 42)

has_h5 = pytest.mark.skipif(not os.path.exists(NEW100_H5), reason="needs the local new100 h5")
has_gpu_data = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(NEW100_H5) and os.path.isdir(GRIDS79)),
                                  reason="needs CUDA + the local new100 h5 and mesh 79's grids")


def test_draw_pair_one_mesh_a_ne_b_training_meshes_only():
  train = [m for m in range(100) if m not in VAL_MESHES]
  ds = ra.StreamPairs("unused.h5", "unused", train, 45, 8, seed=[0, 0])
  it = ds.draws()
  d = [next(it) for _ in range(20000)]
  ms = np.array([x[0] for x in d])
  assert set(ms) == set(train) and not set(ms) & set(VAL_MESHES)
  assert all(a != b and 0 <= a < 45 and 0 <= b < 45 for _, a, b, _ in d)
  assert all(len(set(v)) == 8 and all(0 <= x < 45 for x in v) for *_, v in d)
  counts = np.bincount(ms, minlength=100)[train]
  assert counts.min() > 0.7 * counts.mean() and counts.max() < 1.3 * counts.mean()
  ab = np.array([(a, b) for _, a, b, _ in d])
  assert set(ab[:, 0]) == set(range(45)) and set(ab[:, 1]) == set(range(45))
  it2 = ra.StreamPairs("unused.h5", "unused", train, 45, 8, seed=[0, 0]).draws()
  assert [next(it2) for _ in range(50)] == d[:50]                                   # seeded
  it3 = ra.StreamPairs("unused.h5", "unused", train, 45, 8, seed=[0, 1000]).draws()   # a resumed run: a fresh stream
  assert [next(it3) for _ in range(50)] != d[:50]


@has_h5
def test_mesh_split_matches_train_gs():
  train, val = ra.mesh_split(NEW100_H5, 0.1, 42)
  assert val == VAL_MESHES and len(train) == 90 and not set(train) & set(val) and 79 in train


def test_check_unseen(tmp_path):
  import h5py
  for name, paths in (("a", ["/m/x.glb", "/m/y.glb"]), ("b", ["/m/z.glb"]), ("c", ["/m/z.glb", "/m/y.glb"])):
    with h5py.File(tmp_path / f"{name}.h5", "w") as f:
      f["mesh_paths"] = np.array(paths, dtype=object).astype("S")
  tr.check_unseen(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"), "b")
  with pytest.raises(SystemExit):
    tr.check_unseen(str(tmp_path / "a.h5"), str(tmp_path / "c.h5"), "c")


@pytest.mark.skipif(not (os.path.exists(NEW100_H5) and os.path.exists(SHARD2_H5)), reason="needs the new100 and shard2 h5")
def test_shard2_meshes_unseen():
  tr.check_unseen(NEW100_H5, SHARD2_H5, "shard2")


def eval_cfg(**sets):
  return OmegaConf.create({"data": {"train_views": 45}, "eval": {"seed": 0, "sets": sets}})


def test_eval_pairs_deterministic():
  train = [m for m in range(100) if m not in VAL_MESHES]
  s = {"train_heldout": {"meshes": "train", "n_meshes": 20, "pairs_per_mesh": 2}, "val": {"meshes": "val", "n_meshes": None, "pairs_per_mesh": 4}}
  cfg = eval_cfg(**s)
  p1 = {n: tr.eval_set_plan(cfg, n, cfg.eval.sets[n], train, VAL_MESHES) for n in s}
  p2 = {n: tr.eval_set_plan(cfg, n, cfg.eval.sets[n], train, VAL_MESHES) for n in s}
  assert p1 == p2
  assert [m for m, _ in p1["val"]] == VAL_MESHES and all(len(p) == 4 for _, p in p1["val"])
  th = [m for m, _ in p1["train_heldout"]]
  assert len(th) == 20 == len(set(th)) and set(th) <= set(train)
  assert all(a != b and 0 <= a < 45 and 0 <= b < 45 for _, ps in p1["train_heldout"] + p1["val"] for a, b in ps)
  cfg1 = eval_cfg(**s)
  cfg1.eval.seed = 1
  assert tr.eval_set_plan(cfg1, "val", cfg1.eval.sets.val, train, VAL_MESHES) != p1["val"]


def old_lr_factor(cfg, step, seconds):
  """train_reanchor.lr_factor before cycles (7f82b56), verbatim."""
  w = min(1.0, (step + 1) / max(1, int(cfg.train.warmup_steps)))
  frac = seconds / float(cfg.train.max_seconds)
  df = float(cfg.train.decay_fraction)
  if frac > 1 - df:
    m = float(cfg.train.min_lr) / float(cfg.train.lr)
    w *= m + (1 - m) * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - (1 - df)) / df)))
  return w


def lr_cfg(**kw):
  t = {"warmup_steps": 500, "decay_fraction": 0.1, "min_lr": 1e-6, "lr": 1e-4, "max_seconds": 8700, "cycle_seconds": None,
       "cycle_warmup_seconds": 150}
  t.update(kw)
  return OmegaConf.create({"train": t})


def test_lr_single_cycle_unchanged():
  cfg = lr_cfg()
  for step, sec in [(0, 0), (100, 30), (499, 150), (5000, 1500), (20000, 7900), (25000, 8500), (29000, 8700), (30000, 9000)]:
    assert tr.lr_factor(cfg, step, sec) == old_lr_factor(cfg, step, sec)
  assert tr.n_cycles(cfg) == 1 and tr.cycle_of(cfg, 9000) == 0


def test_lr_cycles_on_the_clock():
  cfg = lr_cfg(max_seconds=28800, cycle_seconds=7200, decay_fraction=0.2)
  m = 1e-6 / 1e-4
  assert tr.n_cycles(cfg) == 4
  # cycle 0 == the single-cycle schedule over 7200 s
  c0 = lr_cfg(max_seconds=7200, decay_fraction=0.2)
  for step, sec in [(0, 0), (200, 60), (3000, 1000), (15000, 5800), (20000, 6500), (23000, 7199.9)]:
    assert tr.lr_factor(cfg, step, sec) == pytest.approx(old_lr_factor(c0, step, sec))
  for k in (1, 2, 3):
    t = 7200 * k
    assert tr.cycle_of(cfg, t - 1e-6) == k - 1 and tr.cycle_of(cfg, t) == k
    assert tr.lr_factor(cfg, 10 ** 6, t - 1e-6) == pytest.approx(m, abs=1e-6)       # the trough before the boundary
    assert tr.lr_factor(cfg, 10 ** 6, t) == pytest.approx(m)                        # warms up again from min_lr
    assert tr.lr_factor(cfg, 10 ** 6, t + 75) == pytest.approx(m + (1 - m) * 0.5)   # half way through cycle_warmup_seconds
    assert tr.lr_factor(cfg, 10 ** 6, t + 150) == 1.0 and tr.lr_factor(cfg, 10 ** 6, t + 5760) == 1.0
    assert tr.lr_factor(cfg, 10 ** 6, t + 5760 + 720) == pytest.approx(m + (1 - m) * 0.5)
  assert tr.cycle_of(cfg, 30000) == 3 and tr.lr_factor(cfg, 10 ** 6, 30000) == pytest.approx(m)
  # continuous in the clock (no jumps but at the step-warmup start)
  ts = np.linspace(0, 28800, 200001)
  lr = np.array([tr.lr_factor(cfg, 10 ** 6, t) for t in ts])
  assert np.abs(np.diff(lr)).max() < 1e-3
  with pytest.raises(SystemExit):
    tr.n_cycles(lr_cfg(max_seconds=28800, cycle_seconds=7000))


@has_gpu_data
def test_streamed_pair_equals_meshdata():
  """A pair streamed by load_pair + stream_bundle == the same pair from MeshData: bundle, render targets and model inputs, bit
  for bit (so the full mode feeds the model exactly what the one-mesh mode did)."""
  import h5py
  views = [4, 17, 30, 44]
  data = ra.MeshData(NEW100_H5, GRIDS79, 79, [3, 17, 40], device="cuda")
  rows = ra.mesh_rows(NEW100_H5)
  mo = data.own_moments()
  in_mean, in_std = data.paired_moments([(3, 17), (40, 3)])
  batcher = ra.Batcher(data, in_mean, in_std, mo)
  empty = torch.randn(14, device="cuda")
  with h5py.File(NEW100_H5, "r") as f:
    for a, b in ((3, 17), (40, 3), (17, 40)):
      s = ra.load_pair(f, rows, GRIDS79, 79, a, b, views)
      got = ra.stream_bundle(s, "cuda")
      want = data.bundle(a, b, True, views)
      assert set(got) == set(want)
      for k in want:
        assert torch.equal(got[k], want[k]), k
      x1 = batcher.tokens([got], empty)
      x2 = batcher.inputs([(a, b)], empty)
      for u, v in zip(x1[:4], x2[:4]):
        assert torch.equal(u, v)
      nb = ra.stream_bundle(s, "cuda", use_a=False)
      assert "a_rows" not in nb and torch.equal(nb["gt"], want["gt"])


@pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isdir(f"{FULL_TEST}/new100") and os.path.exists(SHARD2_H5)),
                    reason="needs CUDA + bla/reanchor/full_test grids + the shard2 h5")
def test_full_stop_and_resume(tmp_path):
  """A tiny full-mode run stopped by train.max_wall_seconds, then resumed: the step / clock / trough count continue, the first
  resumed step's LR is the schedule's at the saved clock, a finished run resumes to a no-op; every trough evaluated once."""
  g = f"{FULL_TEST}/new100/anchored_wd0.3_rbg/grids_sparse"
  out = tmp_path / "run"
  args = [sys.executable, "train_reanchor.py", "--config-name", "train_reanchor_full", f"data.grids_dir={g}", "data.train_meshes=[0,1,79]",
          "data.workers=2", "eval.sets.train_heldout.n_meshes=1", "eval.sets.train_heldout.pairs_per_mesh=1", "eval.sets.val.meshes=[3]",
          "eval.sets.val.pairs_per_mesh=1", f"eval.sets.shard2.render_h5={SHARD2_H5}",
          f"eval.sets.shard2.grids_dir={FULL_TEST}/shard2/anchored_wd0.3_rbg/grids_sparse", "eval.sets.shard2.meshes=[0]",
          "eval.sets.shard2.pairs_per_mesh=1", "model.dim=128", "model.num_blocks=2", "train.pairs_per_step=2", "train.views_per_pair=2",
          "train.max_seconds=24", "train.cycle_seconds=8", "train.cycle_warmup_seconds=2", "train.warmup_steps=10", "train.eval_every=10000",
          "train.log_every=1", "wandb.mode=disabled", f"output_dir={out}"]
  run = lambda *extra: subprocess.run(args + list(extra), capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
  r1 = run("train.max_wall_seconds=20")
  assert r1.returncode == 0, r1.stderr[-3000:]
  st = torch.load(out / "last.pt", weights_only=False)
  assert not st["done"] and 0 < st["clock"] < 24
  step1, clock1, troughs1 = st["step"], st["clock"], st["troughs"]
  cfg = OmegaConf.create(st["cfg"])
  r2 = run()
  assert r2.returncode == 0, r2.stderr[-3000:]
  log = r2.stderr + r2.stdout
  assert f"resumed {out}/last.pt at step {step1} ({clock1:.0f} s training clock, {troughs1} troughs done)" in log
  st2 = torch.load(out / "last.pt", weights_only=False)
  assert st2["done"] and st2["clock"] >= 24 and st2["step"] > step1 and st2["troughs"] == 3
  recs = [json.loads(x) for x in open(out / "eval.jsonl")]
  tags = [r["tag"] for r in recs]
  assert tags == ["start", "trough1", "trough2", "final"], tags
  assert all(recs[i]["clock"] < recs[i + 1]["clock"] for i in range(len(recs) - 1))
  for k, r in enumerate(recs[1:3], 1):
    assert 8 * k <= r["clock"] < 8 * k + 1
  # the first resumed step's LR == the schedule at the saved clock (no reset)
  first = [x for x in log.splitlines() if f"step {step1 + 1} loss" in x]
  assert first, "no log line for the first resumed step"
  lr = float(first[0].split(" lr ")[1].split()[0])
  assert lr == pytest.approx(float(cfg.train.lr) * tr.lr_factor(cfg, step1, clock1), rel=2e-2)
  r3 = run()
  assert r3.returncode == 0 and "a finished run: nothing to do" in r3.stderr + r3.stdout
