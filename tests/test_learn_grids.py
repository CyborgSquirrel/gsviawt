"""diffusion.learn_grids (learnable_grids.py): off = the training batches, loss and gradients of before, bit for bit; the fresh init's
values per slot; the gradient reaches the grids only through the noisy input x_t (zero at t = 1, and never through the x0 target);
frozen grids have no parameters and no optimizer; the grid optimizer is skipped with a skipped loss (SpikeSkip); the grids and their
optimizer state round-trip through a checkpoint; the dump is the grid files' format (encode_grid gives the rows back)."""
import os
import types

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from scipy.spatial.transform import Rotation

import gs_diffusion as gd
import learnable_grids as lgm
import train_gs
from test_token_drop import FEAT, GRID, L, PS, hit_masks, make_head, model_shell

H = GRID * PS
VIEWS = [10, 11, 12, 13]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def synthetic_init(seed=0):
  """build_init's dict for the 4 hit masks of test_token_drop, random raw rows, extent 2 for every volume."""
  g = np.random.default_rng(seed)
  hit = hit_masks().numpy()
  init = {k: [] for k in ("views", "mesh", "rows", "extent", "foot", "hit", "anchor", "pose")}
  pairs, start = [], 0
  for v, h in zip(VIEWS, hit):
    n = int(h.sum())
    init["views"].append(v)
    init["mesh"].append(0)
    init["rows"].append(g.normal(size=(n, gd.CHANNELS)).astype(np.float32))
    init["extent"].append(2.0)
    init["foot"].append(np.full(n, 0.01, np.float32))
    init["hit"].append(h)
    init["anchor"].append(g.normal(size=(n, 3)).astype(np.float32))
    init["pose"].append(np.eye(4, dtype=np.float32))
    pairs.append(lgm.neighbour_pairs(h, start))
    start += n
  init["pairs"] = np.concatenate(pairs)
  init["ref_rows"] = None
  return init


def batch_of(seed=1):
  g = torch.Generator().manual_seed(seed)
  return {"image": torch.randn(4, L, GRID * GRID, FEAT, generator=g), "source": {"hit": hit_masks()},
          "source_view": torch.tensor(VIEWS), "x0_raw": torch.randn(4, gd.CHANNELS, L, H, H, generator=g).half(), "batch_size": 4}


def fake_module(grids=None, t=None):
  m = model_shell(make_head(), "none")
  if t is not None:   # every layer of every volume at this t
    m.schedule = types.SimpleNamespace(sample=lambda b, phase, generator=None: torch.full((b, L), float(t)), fixed_t=None)
  fake = types.SimpleNamespace(model=m, device=torch.device("cpu"))
  if grids is not None:
    fake.grids = grids
  return fake


def losses(fake, batch, seed=123):
  torch.manual_seed(seed)
  fake.model.head.zero_grad(set_to_none=True)
  loss, parts, counts, x0_hat, t = train_gs.GSLightningModule._diffusion_losses(fake, batch)
  return loss, x0_hat, t


# ---------------------------------------------------------------------------- off: exactly as before

def test_off_is_the_old_loss_and_grads():
  batch = batch_of()
  ref = fake_module()
  torch.manual_seed(123)
  img, hit, _, _ = ref.model.inputs(batch, "cpu")
  loss0, *_ = ref.model.training_loss(img, hit, batch["x0_raw"], return_pred=True)   # master's _diffusion_losses call
  loss0.backward()
  g0 = {n: p.grad.clone() for n, p in ref.model.head.named_parameters() if p.grad is not None}
  for fake in (fake_module(), types.SimpleNamespace(**{**vars(fake_module()), "grids": None})):   # no attribute / grids None
    loss, _, _ = losses(fake, batch)
    loss.backward()
    assert torch.equal(loss, loss0)
    g = {n: p.grad for n, p in fake.model.head.named_parameters() if p.grad is not None}
    assert g.keys() == g0.keys() and all(torch.equal(g[k], g0[k]) for k in g)


def test_off_by_default_and_without_the_key():
  with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
    cfg = compose("train_gs")
  assert lgm.learn_grids_config(cfg) is None and cfg.diffusion.learn_grids.enabled is False
  from omegaconf import OmegaConf
  assert lgm.learn_grids_config(OmegaConf.create({"diffusion": {"enabled": True}})) is None


def test_validation_batch_untouched_without_grids():
  batch = batch_of()
  from omegaconf import OmegaConf
  fake = types.SimpleNamespace(cfg=OmegaConf.create({"diffusion": {"enabled": True}}), seen=[],
                               trainer=types.SimpleNamespace(datamodule=types.SimpleNamespace(val_datasets={"val": None})))
  fake._step = lambda stage, b, idx, photom_key: fake.seen.append(b) or {"loss": 0.0}
  fake._diffusion_val_loss = lambda stage, b: fake.seen.append(b)
  train_gs.GSLightningModule.validation_step(fake, batch, 0)
  assert all(b is batch for b in fake.seen)


# ---------------------------------------------------------------------------- the fresh init

def test_fresh_rows_per_slot():
  from test_augment import FX, scene
  _, src = scene()
  src = {"hit": src["hit"].numpy(), "xyz_cam": src["xyz_cam"].numpy(), "K_depth": src["K_depth"]}
  colors = np.random.default_rng(0).random((*src["hit"].shape, 3)).astype(np.float32)
  rows = lgm.fresh_rows(src, colors, opacity=0.1)
  li, yi, xi = np.nonzero(src["hit"])
  z = src["xyz_cam"][li, yi, xi, 2]
  assert rows.shape == (len(li), gd.CHANNELS)
  np.testing.assert_allclose(1 / (1 + np.exp(-rows[:, 0])), 0.1, rtol=1e-6)                   # opacity 0.1 (a logit)
  np.testing.assert_allclose(np.exp(rows[:, gd.SCALE]), np.repeat((z / FX)[:, None], 3, 1), rtol=1e-6)   # isotropic, 1 px
  np.testing.assert_array_equal(rows[:, gd.ROTATION], np.tile([1, 0, 0, 0], (len(li), 1)))        # identity
  np.testing.assert_allclose(rows[:, 8:11] * gd.SH_C0 + 0.5, colors[li, yi, xi], atol=1e-6)    # the slot's colour
  np.testing.assert_array_equal(rows[:, gd.OFFSET], 0)                                          # the mean on the hit


def test_dense_scatters_rows_in_hit_order_and_offsets_by_extent():
  init = synthetic_init()
  g = lgm.LearnableGrids(init)
  x = g.dense(VIEWS[::-1], hit_masks().flip(0))
  for b, i in enumerate(range(3, -1, -1)):
    h = hit_masks()[i]
    torch.testing.assert_close(x[b][:, h].T, torch.from_numpy(init["rows"][i]))
    assert (x[b][:, ~h] == 0).all()
  # the stored offsets are offset / extent: the optimizer sees the fitter's lr.offset x scene_extent
  torch.testing.assert_close(g.params["offset"][:3] * 2, torch.from_numpy(init["rows"][0][:3, gd.OFFSET]))
  with pytest.raises(KeyError):
    g.dense([99], hit_masks()[:1])
  with pytest.raises(ValueError):
    g.dense([VIEWS[0]], hit_masks()[1:2])


# ---------------------------------------------------------------------------- gradients: only through x_t

def test_grads_reach_the_grids_through_x_t_only():
  batch = batch_of()
  for t, expect_grad in ((0.3, True), (1.0, False)):
    g = lgm.LearnableGrids(synthetic_init())
    fake = fake_module(g, t=t)
    loss, x0_hat, _ = losses(fake, batch)
    x0_hat.float().square().mean().backward()        # stands in for the rendering loss of the x0 prediction
    total = sum(float(p.grad.abs().sum()) for p in g.parameters())
    assert (total > 0) == expect_grad, (t, total)
    # the x0 loss: at t = 1 its prediction doesn't see the grid, and the target is detached -> no gradient at all
    g.zero_grad(set_to_none=True)
    loss, _, _ = losses(fake, batch)
    loss.backward()
    if t == 1.0:
      assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in g.parameters())


def test_training_uses_the_grids_not_the_dataset():
  batch = batch_of()
  g = lgm.LearnableGrids(synthetic_init())
  fake = fake_module(g, t=0.0)
  loss_g, _, _ = losses(fake, batch)
  ref = fake_module(t=0.0)
  img, hit, _, _ = ref.model.inputs(batch, "cpu")
  torch.manual_seed(123)
  loss_r, *_ = ref.model.training_loss(img, hit, g.dense(VIEWS, hit).detach(), return_pred=True)
  torch.testing.assert_close(loss_g, loss_r)


def test_frozen_has_no_parameters_and_no_grad():
  g = lgm.LearnableGrids(synthetic_init(), learnable=False)
  assert list(g.parameters()) == [] and not g.dense(VIEWS, hit_masks()).requires_grad
  fake = fake_module(g, t=0.3)
  loss, x0_hat, _ = losses(fake, batch_of())
  x0_hat.float().square().mean().backward()   # still trains the model
  assert any(p.grad is not None for p in fake.model.head.parameters())
  assert g.state_dict() == {}   # (nothing to checkpoint: a resume rebuilds the frozen grids from their init)


# ---------------------------------------------------------------------------- the module: optimizer, SpikeSkip, checkpoint

def module(tmp_path, learnable=True, seed=0):
  with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
    cfg = compose("train_gs", overrides=[f"output_dir={tmp_path}", "diffusion.learn_grids.enabled=true",
                                         f"diffusion.learn_grids.learnable={str(learnable).lower()}"])
  init = synthetic_init(seed)
  init["eval"] = {"source_h5": None, "volumes": [], "sets": {}}
  return train_gs.GSLightningModule(cfg, grid_init=init)


def fill_grads(m, seed=0):
  g = torch.Generator().manual_seed(seed)
  for p in m.grids.parameters():
    p.grad = torch.randn(p.shape, generator=g)


def test_grid_optimizer_steps_with_the_model_and_skips_with_spikeskip(tmp_path, monkeypatch):
  monkeypatch.setattr(pl.LightningModule, "optimizer_step", lambda self, *a, **k: None)   # (the model's own step: not under test)
  m = module(tmp_path)
  m.on_train_start()
  assert isinstance(m.grid_opt, torch.optim.Adam)
  groups = {gr["name"]: gr for gr in m.grid_opt.param_groups}
  assert {k: gr["lr"] for k, gr in groups.items()} == {"grids/opacity": 0.025, "grids/scale": 0.005, "grids/rotation": 0.001,
                                                        "grids/sh_dc": 0.0025, "grids/offset": 0.00016}
  assert all(gr["eps"] == 1e-15 and gr["weight_decay"] == 0 for gr in groups.values())
  before = [p.detach().clone() for p in m.grids.parameters()]
  fill_grads(m)
  m._accepted_since_step = False                       # SpikeSkip skipped this step's loss
  m.optimizer_step(0, 0, None)
  assert all(torch.equal(a, p) for a, p in zip(before, m.grids.parameters()))
  assert all(p.grad is None for p in m.grids.parameters()) and not m.grid_opt.state
  fill_grads(m)
  m._accepted_since_step = True
  m.optimizer_step(0, 1, None)
  assert all(not torch.equal(a, p) for a, p in zip(before, m.grids.parameters()))
  assert all(p.grad is None for p in m.grids.parameters()) and not m._accepted_since_step


def test_frozen_has_no_grid_optimizer(tmp_path):
  m = module(tmp_path, learnable=False)
  m.on_train_start()
  assert m.grid_opt is None and not any(k.startswith("grids.") for k in m.state_dict())


def test_grids_round_trip_through_a_checkpoint(tmp_path, monkeypatch):
  monkeypatch.setattr(pl.LightningModule, "optimizer_step", lambda self, *a, **k: None)
  a = module(tmp_path)
  a.on_train_start()
  for step in range(3):
    fill_grads(a, step)
    a._accepted_since_step = True
    a.optimizer_step(0, step, None)
  ckpt = {"state_dict": a.state_dict()}
  a.on_save_checkpoint(ckpt)
  assert "grid_optimizer" in ckpt and any(k.startswith("grids.params.") for k in ckpt["state_dict"])
  torch.save(ckpt, tmp_path / "c.ckpt")
  ckpt = torch.load(tmp_path / "c.ckpt", weights_only=False)

  b = module(tmp_path)   # the same init (a resume rebuilds it), then Lightning's restore order: hooks, weights, train start
  b.on_load_checkpoint(ckpt)
  b.load_state_dict(ckpt["state_dict"])
  b.on_train_start()
  for pa, pb in zip(a.grids.parameters(), b.grids.parameters()):
    assert torch.equal(pa, pb)
  sa, sb = a.grid_opt.state_dict(), b.grid_opt.state_dict()
  for k in sa["state"]:
    for n in ("exp_avg", "exp_avg_sq", "step"):
      assert torch.equal(sa["state"][k][n], sb["state"][k][n])
  fill_grads(a, 9)
  fill_grads(b, 9)
  for m in (a, b):
    m._accepted_since_step = True
    m.optimizer_step(0, 9, None)
  assert all(torch.equal(pa, pb) for pa, pb in zip(a.grids.parameters(), b.grids.parameters()))


# ---------------------------------------------------------------------------- statistics and the dump

def test_param_stats_drift_and_smoothness():
  init = synthetic_init()
  g = lgm.LearnableGrids(init)
  s0 = lgm.param_stats(g.rows().detach(), g.foot, g.pairs, init_rows=g.rows(init=True))
  assert all(s0[f"drift/{k}"] == 0 for k in gd.FIELDS)
  with torch.no_grad():
    g.params["opacity"] += 1.0
  s1 = lgm.param_stats(g.rows().detach(), g.foot, g.pairs, init_rows=g.rows(init=True))
  assert s1["drift/opacity"] > 0 and s1["drift/scale"] == 0
  assert abs(s1["smooth/scale"] - s0["smooth/scale"]) < 1e-9 and 0 <= s1["frac_degenerate"] <= 1


def test_dump_is_the_grid_format(tmp_path):
  from test_augment import scene
  x0, src = scene()
  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
  pose[:3, 3] = [0.5, -1.0, 2.0]
  hit = src["hit"].numpy()
  rows = x0[:, hit].T.copy()
  n = len(rows)
  init = {"views": [7], "mesh": [0], "rows": [rows], "extent": [1.5], "foot": [np.full(n, 0.02, np.float32)], "hit": [hit],
          "anchor": [src["xyz_cam"].numpy()[hit]], "pose": [pose], "pairs": lgm.neighbour_pairs(hit, 0), "ref_rows": None}
  g = lgm.LearnableGrids(init)
  grid, raw = g.to_grid(0, {"primary_view_row": 7})
  from grid_io import read_grid, write_sparse
  write_sparse(tmp_path / "g.h5", grid)
  back = gd.encode_grid(read_grid(tmp_path / "g.h5"), {**src, "pose_gl": torch.from_numpy(pose)})
  np.testing.assert_allclose(back[:, hit].T, rows, atol=2e-5)
  np.testing.assert_allclose(raw, rows, rtol=1e-6, atol=1e-8)   # (offsets: stored / extent, x extent back)


# ---------------------------------------------------------------------------- scale_encoding.scale_param cumulative

def world_grid(seed=0):
  """test_augment's scene as a grid file's Grid (world frame, a random pose) and its source dict (with pose_gl)."""
  from test_augment import scene
  x0, src = scene(seed)
  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
  pose[:3, 3] = [0.5, -1.0, 2.0]
  hit = src["hit"].numpy()
  n = int(hit.sum())
  init = {"views": [7], "mesh": [0], "rows": [x0[:, hit].T.copy()], "extent": [1.0], "foot": [np.full(n, 0.02, np.float32)],
          "hit": [hit], "anchor": [src["xyz_cam"].numpy()[hit]], "pose": [pose], "pairs": lgm.neighbour_pairs(hit, 0), "ref_rows": None}
  grid, _ = lgm.LearnableGrids(init).to_grid(0)
  grid.rows["gaussian_scales"][:5] = grid.rows["gaussian_scales"][:5, :1]   # a few tied (isotropic) Gaussians
  return grid, {**src, "pose_gl": torch.from_numpy(pose)}


def as_log(x0, scale_param):
  """(C, L, H, W) x0 with its scale channels turned into log-scales (test_augment.gaussians reads those)."""
  out = x0.copy()
  out[gd.SCALE] = np.moveaxis(gd.cumulative_to_log(np.moveaxis(x0[gd.SCALE], 0, -1).reshape(-1, 3)).reshape(
    *x0.shape[1:], 3), -1, 0) if scale_param == "cumulative" else x0[gd.SCALE]
  return out


def test_cumulative_encode_decode_round_trip():
  from test_augment import gaussians
  grid, src = world_grid()
  hit = src["hit"].numpy()
  x_log = gd.encode_grid(grid, src)
  x_cum = gd.encode_grid(grid, src, scale_param="cumulative")
  logs = as_log(x_cum, "cumulative")[:, hit][gd.SCALE].T
  assert (np.diff(logs, axis=1) > 0).all()                                     # sorted ascending (ties: the 1e-4 floor apart)
  np.testing.assert_allclose(gaussians(as_log(x_cum, "cumulative"), src), gaussians(x_log, src), rtol=1e-3, atol=1e-6)   # render-equal
  # decode_x0 reads them the same way
  c = gd.CHANNELS
  g_cum = gd.decode_x0(torch.from_numpy(x_cum).permute(1, 0, 2, 3)[None], torch.zeros(c), torch.ones(c), scale_param="cumulative")
  torch.testing.assert_close(g_cum["raw_scale"][0].permute(1, 0, 2, 3)[:, torch.from_numpy(hit)].T, torch.from_numpy(logs))


def test_cumulative_decode_is_sorted_for_any_prediction():
  g = torch.Generator().manual_seed(0)
  x = 5 * torch.randn(3, 2, gd.CHANNELS, 8, 8, generator=g)
  c = gd.CHANNELS
  lo, hi = -2 * torch.ones(c), 2 * torch.ones(c)
  for kw in ({}, {"lo": lo, "hi": hi}):
    s = gd.decode_x0(x, torch.zeros(c), torch.ones(c), scale_param="cumulative", **kw)["raw_scale"]   # (B, L, 3, H, W)
    assert (s[:, :, 1] >= s[:, :, 0]).all() and (s[:, :, 2] >= s[:, :, 1]).all()
  with pytest.raises(ValueError):
    gd.decode_x0(x, torch.zeros(c), torch.ones(c), lo, hi, clamp="scale", scale_param="cumulative")
  with pytest.raises(ValueError):
    gd.normalize_encoding({"scale_param": "cumulative", "scale_order": "sorted"})


def test_default_log_encoding_and_decode_unchanged():
  grid, src = world_grid()
  assert np.array_equal(gd.encode_grid(grid, src), gd.encode_grid(grid, src, scale_param="log"))
  assert gd.normalize_encoding(None)["scale_param"] == "log" and gd.encoding_tag(None) == ""
  assert gd.encoding_tag({"scale_param": "cumulative"}) == "cumulative"
  x = torch.randn(2, 2, gd.CHANNELS, 4, 4)
  c = gd.CHANNELS
  a = gd.decode_x0(x, torch.zeros(c), torch.ones(c))
  torch.testing.assert_close(a["raw_scale"], x[:, :, gd.SCALE])


def cumulative_init(seed=0):
  init = synthetic_init(seed)
  for r in init["rows"]:
    r[:, gd.SCALE] = gd.log_to_cumulative(np.sort(r[:, gd.SCALE], axis=1).astype(np.float64))
  return init


def test_cumulative_learnable_grid_stays_sorted():
  g = lgm.LearnableGrids(cumulative_init(), scale_param="cumulative")
  opt = torch.optim.Adam(g.param_groups({"offset": 1e-3, "sh0": 1e-2, "opacities": 1e-2, "scales": 0.5, "quats": 1e-2}), eps=1e-15)
  gen = torch.Generator().manual_seed(0)
  for _ in range(50):   # big steps, random gradients: the channels go anywhere, the decoded axes stay sorted
    for p in g.parameters():
      p.grad = torch.randn(p.shape, generator=gen)
    opt.step()
  s = lgm.log_scales(g.rows().detach(), "cumulative")
  assert (s[:, 1] >= s[:, 0]).all() and (s[:, 2] >= s[:, 1]).all()
  assert float((g.params["scale"].detach() - g.init_scale).abs().max()) > 1
  st = lgm.param_stats(g.rows().detach(), g.foot, g.pairs, init_rows=g.rows(init=True), scale_param="cumulative")
  assert np.isfinite(list(st.values())).all() and st["drift/scale"] > 0


def test_fresh_rows_cumulative():
  from test_augment import FX, scene
  _, src = scene()
  src = {"hit": src["hit"].numpy(), "xyz_cam": src["xyz_cam"].numpy(), "K_depth": src["K_depth"]}
  colors = np.zeros((*src["hit"].shape, 3), np.float32)
  rows = lgm.fresh_rows(src, colors, scale_param="cumulative")
  z = src["xyz_cam"][src["hit"]][:, 2]
  logs = lgm.log_scales(rows, "cumulative")
  np.testing.assert_allclose(logs[:, 0], np.log(z / FX), rtol=1e-6)
  np.testing.assert_allclose(np.diff(logs, axis=1), gd.CUM_GAP_FLOOR, rtol=1e-2)   # isotropic up to the floor gaps (float32)


def test_cumulative_gradient_path_unchanged():
  batch = batch_of()
  for t, expect in ((0.3, True), (1.0, False)):
    g = lgm.LearnableGrids(cumulative_init(), scale_param="cumulative")
    fake = fake_module(g, t=t)
    loss, x0_hat, _ = losses(fake, batch)
    (loss + x0_hat.float().square().mean()).backward()
    total = sum(float(p.grad.abs().sum()) for p in g.parameters() if p.grad is not None)
    assert (total > 0) == expect, (t, total)


# ---------------------------------------------------------------------------- learn_grids.x0_grad

def train_step(g, x0_weight, x0_grad, t=0.3):
  """_diffusion_train_step on a shell module (rendering loss stand-in: mean square of the x0 prediction) -> (loss, the fake)."""
  from omegaconf import OmegaConf
  fake = fake_module(g, t=t)
  fake.cfg = OmegaConf.create({"diffusion": {"x0_weight": x0_weight, "photom": {"enabled": True, "weight": 10.0}}})
  fake.views_seen = fake.pixels_seen = 0
  fake._accepted_since_step = False
  fake._grid_x0_grad = x0_grad
  fake._grid_grads = None
  fake.spike_skip = types.SimpleNamespace(check=lambda v: False, average=None)
  fake._photom_active = lambda: True
  fake._diffusion_photom = lambda stage, batch, x0_hat, t, log=None: x0_hat.float().square().mean()
  fake._log_metric = lambda *a: None
  fake._log_parts = lambda *a: None
  for name in ("_diffusion_losses", "_rendering_grid_grads"):
    setattr(fake, name, (lambda f: lambda *a, **k: f(fake, *a, **k))(getattr(train_gs.GSLightningModule, name)))
  torch.manual_seed(123)
  return train_gs.GSLightningModule._diffusion_train_step(fake, batch_of()), fake


def test_x0_grad_none_gives_the_grids_the_rendering_loss_gradient_only():
  ref_g = lgm.LearnableGrids(synthetic_init())
  loss, _ = train_step(ref_g, 0.0, "through_input")   # the rendering loss alone
  loss.backward()
  ref = [p.grad.clone() for p in ref_g.parameters()]
  g = lgm.LearnableGrids(synthetic_init())
  loss, fake = train_step(g, 1.0, "none")
  assert fake._grid_grads is not None
  for a, b in zip(fake._grid_grads, ref):
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)
  # the model still gets both losses: the total loss's backward (graph kept by the extra pass) differs from the rendering-only one
  loss.backward()
  # through_input: unchanged -- nothing stashed, the grids' gradient is the whole loss's (x0 loss included: not the reference)
  g2 = lgm.LearnableGrids(synthetic_init())
  loss2, fake2 = train_step(g2, 1.0, "through_input")
  loss2.backward()
  assert fake2._grid_grads is None
  assert any(not torch.allclose(p.grad, r) for p, r in zip(g2.parameters(), ref))


def test_backward_keeps_the_total_loss_off_the_grids(tmp_path, monkeypatch):
  monkeypatch.setattr(pl.LightningModule, "optimizer_step", lambda self, *a, **k: None)
  m = module(tmp_path)
  m.on_train_start()
  w = next(p for n, p in m.named_parameters() if not n.startswith("grids.") and p.requires_grad)
  stash = [torch.full_like(p, 0.5) for p in m.grids.parameters()]
  m._grid_grads = stash
  m.backward(w.sum() + sum(p.sum() for p in m.grids.parameters()))
  assert w.grad is not None and all(p.grad is None for p in m.grids.parameters())
  before = [p.detach().clone() for p in m.grids.parameters()]
  m._accepted_since_step = True
  m.optimizer_step(0, 0, None)   # installs the stash and steps
  assert all(not torch.equal(a, p) for a, p in zip(before, m.grids.parameters())) and m._grid_grads is None
