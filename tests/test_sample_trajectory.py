"""GSDiffusionModel.sample's trajectory hook (eval_diffusion +save_trajectory): an observer only -- the sample is bit-identical with
and without it (lockstep, hidden_shift, lock, inject, project) -- called steps + 1 times with the state the model is given, its
per-layer t and the model's prediction from it, the last call with the output state; and eval_diffusion.TrajectoryRecorder's npz."""
import json

import numpy as np
import pytest
import torch

import eval_diffusion as ev
import gs_diffusion as gd
from diffusion_schedule import LayerAwareSchedule

B, L, H, W = 2, 6, 8, 8


def shell(heads=None):
  m = object.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.schedule, m.num_layers, m.steps, m.eval_seed = LayerAwareSchedule(L), L, 1, 0
  m.register_buffer("x0_mean", torch.zeros(gd.CHANNELS))
  m.register_buffer("x0_std", torch.ones(gd.CHANNELS))
  m.register_buffer("x0_lo", torch.full((gd.CHANNELS,), -10.0))
  m.register_buffer("x0_hi", torch.full((gd.CHANNELS,), 10.0))

  class Head(torch.nn.Module):
    def forward(self, feats, x_net, t):
      out = 0.5 * x_net[:, :, :gd.CHANNELS] + 0.1 * t[:, :, None, None, None] + 0.05   # state- and t-dependent
      if heads is not None:
        heads.append((x_net.clone(), t.clone(), out.clone()))
      return out
  m.head = Head()
  m._feats = lambda image: None
  return m


class Keep:
  def __init__(self):
    self.calls = []

  def __call__(self, step, t, x, x0_hat):
    self.calls.append((step, t.clone(), x.clone(), None if x0_hat is None else x0_hat.clone()))


VARIANTS = {
  "lockstep": {},
  "hidden_shift": {"hidden_shift": 32.0},
  "t_min": {"t_min": 0.05},
  "lock": "lock",
  "inject": "inject",
  "inject_onestep": "inject_onestep",
  "project": {"project": lambda x: x.clamp(-0.3, 0.3)},
}


def kwargs_of(name):
  v = VARIANTS[name]
  if v == "lock":
    return {"lock": (torch.randn(B, L, gd.CHANNELS, H, W, generator=torch.Generator().manual_seed(1)), [0, 2])}
  if v == "inject":
    return {"inject": (torch.randn(B, L, gd.CHANNELS, H, W, generator=torch.Generator().manual_seed(2)), ["sh_dc"])}
  if v == "inject_onestep":
    return {"inject": ("onestep", ["sh_dc", "offset"])}
  return dict(v)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_sample_bit_identical_with_and_without_the_hook(variant):
  hit = torch.rand(B, L, H, W, generator=torch.Generator().manual_seed(0)) > 0.3
  a = shell().sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=7, **kwargs_of(variant))
  keep = Keep()
  b = shell().sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=7, trajectory=keep, **kwargs_of(variant))
  assert a.keys() == b.keys()
  for k in a:
    assert torch.equal(a[k], b[k]), k
  assert [c[0] for c in keep.calls] == list(range(8))


@pytest.mark.parametrize("alpha", [None, 32.0])
def test_hook_sees_the_model_inputs_predictions_and_schedule(alpha):
  hit = torch.rand(B, L, H, W, generator=torch.Generator().manual_seed(0)) > 0.3
  steps = 6
  heads, keep = [], Keep()
  m = shell(heads)
  g = m.sample(torch.zeros(B, 3, 4, 4), hit, seeds=[3, 4], steps=steps, hidden_shift=alpha, trajectory=keep)
  assert len(keep.calls) == steps + 1 and len(heads) == steps
  _, eps = m.eval_noise([3, 4], (B, L, gd.CHANNELS, H, W), torch.device("cpu"))
  assert torch.equal(keep.calls[0][2], eps)                                       # k = 0: the pure noise
  ts = torch.linspace(1, 0, steps + 1)
  for k, (step, t, x, x0_hat) in enumerate(keep.calls):
    assert step == k and t.shape == (L,)
    torch.testing.assert_close(t[0], ts[k])
    if alpha is not None:
      torch.testing.assert_close(t[1:], (alpha * ts[k] / (1 + (alpha - 1) * ts[k])).expand(L - 1))
    if k < steps:
      x_net, t_model, out = heads[k]
      assert torch.equal(x0_hat, out.float())                                     # the model's prediction from this state
      assert torch.equal(t_model, t[None].expand(B, L))                            # at this t
      a = hit[:, :, None].float()
      assert torch.equal(x_net[:, :, :gd.CHANNELS], a * x + (1 - a) * eps)         # the state the model is given (noise off-hit)
      nxt = keep.calls[k + 1][2]
      dt = (t - keep.calls[k + 1][1])[None, :, None, None, None]
      torch.testing.assert_close(nxt, x - dt * (x - x0_hat) / t[None, :, None, None, None])   # the exact Euler step
    else:
      assert x0_hat is None
      ref = m.decode(x)                                                           # the last state is the output
      for key in g:
        assert torch.equal(g[key], ref[key])
  torch.testing.assert_close(keep.calls[-1][2], keep.calls[-2][3])                # lands on the last prediction (up to rounding)


def test_hook_refused_for_colour_separate():
  m = shell()
  m.colour_separate = True
  with pytest.raises(ValueError, match="trajectory"):
    m.sample(torch.zeros(B, 3, 4, 4), torch.ones(B, L, H, W, dtype=torch.bool), seeds=[3, 4], steps=2, trajectory=Keep())


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_recorder_npz(tmp_path, monkeypatch, dtype):
  steps = 5
  hit = torch.rand(1, L, H, W, generator=torch.Generator().manual_seed(0)) > 0.3
  xyz = torch.randn(1, L, H, W, 3, generator=torch.Generator().manual_seed(5))
  batch = {"source": {"xyz_cam": xyz, "hit": hit}}
  m = shell()
  m.decode_clamp, m.max_sh_degree = "channel", 0
  decoded = []
  monkeypatch.setattr(ev, "render", lambda model, batch, gauss, views, device: decoded.append(gauss) or "r")
  score = lambda r: {k: torch.full((2,), float(len(decoded))) for k in ev.METRICS}
  rec = ev.TrajectoryRecorder(m, batch, hit, [1, 2], torch.device("cpu"), None, score)
  keep = Keep()
  both = lambda **kw: (rec(**kw), keep(**kw))
  g = m.sample(torch.zeros(1, 3, 4, 4), hit, seeds=[9], steps=steps, hidden_shift=32.0, trajectory=both)
  path = str(tmp_path / "trajectory" / "val_meshes_3.npz")
  rec.save(path, np.dtype(dtype), {"set": "val_meshes", "volume": 3})
  f = np.load(path)
  n = int(hit.sum())
  assert f["x_t"].shape == (steps + 1, 1, n, gd.CHANNELS) and f["x0_hat"].shape == (steps, 1, n, gd.CHANNELS)
  assert str(f["x_t"].dtype) == dtype == str(f["dtype"]) and f["t"].shape == (steps + 1, L)
  assert json.loads(str(f["meta"])) == {"set": "val_meshes", "volume": 3}
  sel = f["hit"]
  assert sel.shape == (L, H, W) and np.array_equal(sel, hit[0].numpy())
  assert np.array_equal(f["xyz_cam"], xyz[0].numpy()[sel])
  tol = {"float16": 2e-3, "float32": 0}[dtype]
  for k in range(steps + 1):
    dense = keep.calls[k][2][0].permute(0, 2, 3, 1).numpy()                     # (L, H, W, C)
    np.testing.assert_allclose(f["x_t"][k, 0].astype(np.float32), dense[sel], rtol=tol, atol=tol)   # np.nonzero order
    if k < steps:
      np.testing.assert_allclose(f["x0_hat"][k, 0].astype(np.float32), keep.calls[k][3][0].permute(0, 2, 3, 1).numpy()[sel],
                                 rtol=tol, atol=tol)
    np.testing.assert_allclose(f["t"][k], keep.calls[k][1].numpy())
  # decoding the slots from the npz alone (as (1, 1, C, N, 1)) = the model's decode of the state at the hit slots
  st = lambda a: torch.from_numpy(a)
  slots = st(f["x_t"][steps, 0].astype(np.float32)).T[None, None, :, :, None]
  dec = gd.decode_x0(slots, st(f["x0_mean"]), st(f["x0_std"]), st(f["x0_lo"]), st(f["x0_hi"]), clamp=str(f["decode_clamp"]))
  sel_t = torch.from_numpy(sel)
  for key in g:
    ref = g[key][0].permute(0, 2, 3, 1)[sel_t]                                     # (N, c)
    torch.testing.assert_close(dec[key][0, 0, :, :, 0].T, ref, rtol=tol or 1e-6, atol=tol * 10 or 1e-6)
  # one metrics row per decoded state: steps + 1 states and steps predictions, in call order
  assert [(r["step"], r["kind"]) for r in rec.rows] == [p for k in range(steps + 1)
                                                         for p in ([(k, "x_t"), (k, "x0_hat")] if k < steps else [(k, "x_t")])]
  assert len(decoded) == 2 * steps + 1 and rec.last_x0_hat is not None
  torch.testing.assert_close(rec.last_x0_hat, keep.calls[steps - 1][3])
