"""fit_3dgs param_mode: direct = self.params are the raw parameters, renders exactly as before; latent_mlp = each
Gaussian's 14 raw numbers are f(code), f(z) = z + g(z) the identity at init (first render bit-identical to direct),
gradients reach the codes and the MLP, and everything downstream (render, export) reads f(z), not z."""
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import fit_3dgs
from module import SH_C0

CONF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf", "fit_3dgs.yaml")


def make_cfg(**over):
  cfg = OmegaConf.load(CONF)
  for k, v in {"gaussian_layout": "layered", "sh_degree": 0, "opacity_reset_interval": None, "iters": 100,
               "lr.means_weight_decay": 0.3, **over}.items():
    OmegaConf.update(cfg, k, v)
  return cfg


def seed_g(n=96, seed=0):
  """A layered-style seed: n Gaussians in front of an identity camera (z ~ 3), raw parameters like init_gaussians'."""
  r = np.random.default_rng(seed)
  return {
    "means": np.concatenate([r.uniform(-0.8, 0.8, (n, 2)), r.uniform(2.8, 3.2, (n, 1))], 1).astype(np.float32),
    "scales": np.full((n, 3), -2.0, np.float32) + r.normal(0, 0.1, (n, 3)).astype(np.float32),
    "quats": np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
    "opacities": np.full(n, fit_3dgs._logit(np.float32(0.1)), np.float32),
    "sh0": ((r.uniform(0, 1, (n, 3)).astype(np.float32) - 0.5) / SH_C0)[:, None, :],
    "shN": np.zeros((n, 0, 3), np.float32),
  }


def make_model(mode, **over):
  m = fit_3dgs.Fit3DGSLightningModule(make_cfg(param_mode=mode, **over), seed_g(), 1.0, {"train": [], "val": []})
  return m.cuda() if torch.cuda.is_available() else m


def camera():
  K = torch.tensor([[[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]]]).cuda()
  return torch.eye(4)[None].cuda(), K, 64, 64


def render(m):
  viewmat, K, W, H = camera()
  return fit_3dgs.render(m._render_params(), 0, viewmat, K, W, H)[0]


def test_pack_unpack_roundtrip():
  g = {k: torch.from_numpy(v) for k, v in seed_g(7).items()}
  x = fit_3dgs.pack_latent(g)
  assert x.shape == (7, 14) and fit_3dgs.LATENT_DIM == 14
  back = fit_3dgs.unpack_latent(x)
  for k in ("opacities", "scales", "quats", "sh0", "means"):
    assert back[k].shape == g[k].shape and torch.equal(back[k], g[k])


def test_decoder_is_exact_identity_at_init_and_seeded():
  d = fit_3dgs.LatentDecoder(14, 64, 2, seed=3)
  z = torch.randn(1000, 14) * 5
  assert torch.equal(d(z), z)
  assert [tuple(l.weight.shape) for l in d.linears] == [(64, 14), (64, 64), (14, 64)]
  state = torch.random.get_rng_state()
  d2 = fit_3dgs.LatentDecoder(14, 64, 2, seed=3)
  assert torch.equal(torch.random.get_rng_state(), state)          # the global RNG (DataLoader shuffle) untouched
  for a, b in zip(d.parameters(), d2.parameters()):
    assert torch.equal(a, b)


def test_direct_decoded_is_params_itself():
  m = fit_3dgs.Fit3DGSLightningModule(make_cfg(), seed_g(), 1.0, {"train": [], "val": []})
  assert m.param_mode == "direct" and m.latent is None
  assert m._decoded() is m.params
  rp = m._render_params()
  assert torch.equal(rp["means"], m.means_original + m.params["means"])
  for k in ("scales", "quats", "opacities", "sh0", "shN"):
    assert rp[k] is m.params[k]


@pytest.mark.cuda
def test_latent_first_render_identical_to_direct():
  a, b = make_model("direct"), make_model("latent_mlp")
  pa, pb = a._render_params(), b._render_params()
  for k in pa:
    assert torch.equal(pa[k], pb[k]), k
  assert torch.equal(render(a), render(b))
  out_a, out_b = a.params_np(), b.params_np()
  for k in out_a:
    assert np.array_equal(out_a[k], out_b[k]), k


@pytest.mark.cuda
def test_gradients_reach_codes_and_mlp():
  m = make_model("latent_mlp")
  m.configure_optimizers()
  assert len(m.all_optimizers) == len(m.optimizers_dict) + 1
  target = torch.rand(1, 64, 64, 3, device="cuda")

  def step():
    for o in m.all_optimizers:
      o.zero_grad(set_to_none=True)
    ((render(m) - target) ** 2).mean().backward()

  step()
  for k in ("opacities", "scales", "quats", "sh0", "means"):
    assert m.params[k].grad is not None and m.params[k].grad.abs().sum() > 0, k
  lin = m.latent.linears
  # at init only g's (zero) last layer gets a gradient: the earlier layers sit behind its zero weight
  assert lin[-1].weight.grad.abs().sum() > 0 and lin[-1].bias.grad.abs().sum() > 0
  assert all(l.weight.grad is None or l.weight.grad.abs().sum() == 0 for l in lin[:-1])
  for o in m.all_optimizers:
    o.step()
  step()
  for l in lin:
    assert l.weight.grad.abs().sum() > 0


@pytest.mark.cuda
def test_downstream_reads_decoded_not_codes():
  m = make_model("latent_mlp")
  with torch.no_grad():
    torch.nn.init.normal_(m.latent.linears[-1].weight, std=0.1)
  z = fit_3dgs.pack_latent(m.params)
  p = m.latent(z).detach()
  assert not torch.allclose(p, z)
  dec = fit_3dgs.unpack_latent(p)
  rp = m._render_params()
  assert torch.allclose(rp["means"], m.means_original + dec["means"])
  for k in ("scales", "quats", "opacities", "sh0"):
    assert torch.allclose(rp[k], dec[k]), k
  out = m.params_np()
  assert np.allclose(out["sh0"], dec["sh0"].cpu().numpy())
  assert np.allclose(out["means"], (m.means_original + dec["means"]).cpu().numpy())
  ln = m.latent_np()
  assert np.allclose(ln["decoded"], p.cpu().numpy()) and np.allclose(ln["codes"], z.detach().cpu().numpy())
  assert np.allclose(ln["codes"] + ln["residual"], ln["decoded"], atol=1e-5)
  # the render really uses f(z): differs from a render of the codes themselves
  viewmat, K, W, H = camera()
  codes = {**m.params, "means": m.means_original + m.params["means"]}
  assert not torch.equal(fit_3dgs.render(rp, 0, viewmat, K, W, H)[0], fit_3dgs.render(codes, 0, viewmat, K, W, H)[0])


def test_latent_rejects_set_layout_and_opacity_reset():
  with pytest.raises(SystemExit):
    fit_3dgs.Fit3DGSLightningModule(make_cfg(param_mode="latent_mlp", gaussian_layout="set"), seed_g(), 1.0,
                                     {"train": [], "val": []})
  with pytest.raises(SystemExit):
    fit_3dgs.Fit3DGSLightningModule(make_cfg(param_mode="latent_mlp", opacity_reset_interval=50), seed_g(), 1.0,
                                     {"train": [], "val": []})
  with pytest.raises(SystemExit):
    make_model("bogus")


def test_out_dim_follows_sh_degree():
  assert [fit_3dgs.latent_out_dim(d) for d in (0, 1, 2, 3)] == [14, 23, 38, 59]


def test_sh_degree_warmup_option():
  assert make_model("direct", sh_degree=1).active_sh_degree == 0                          # default: the ramp
  assert make_model("direct", sh_degree=1, sh_degree_warmup=False).active_sh_degree == 1   # active from iteration 0


def seed_g_sh(deg, **kw):
  g = seed_g(**kw)
  g["shN"] = np.zeros((len(g["means"]), (deg + 1) ** 2 - 1, 3), np.float32)
  return g


@pytest.mark.cuda
def test_latent_sh1_identity_and_zero_shN_at_init():
  kw = dict(sh_degree=1, sh_degree_warmup=False)
  ms = {mode: fit_3dgs.Fit3DGSLightningModule(make_cfg(param_mode=mode, **kw), seed_g_sh(1), 1.0,
                                              {"train": [], "val": []}).cuda() for mode in ("direct", "latent_mlp")}
  m = ms["latent_mlp"]
  assert m.latent.out_dim == 23 and m.latent.linears[-1].weight.shape == (23, 64)
  z = fit_3dgs.pack_latent(m.params)
  out = m.latent(z)
  assert torch.equal(out[:, :14], z) and torch.equal(out[:, 14:], torch.zeros_like(out[:, 14:]))
  rp, rd = m._render_params(), ms["direct"]._render_params()
  assert rp["shN"].shape == (96, 3, 3) and torch.equal(rp["shN"], torch.zeros_like(rp["shN"]))
  for k in rd:
    assert torch.equal(rp[k], rd[k]), k
  viewmat, K, W, H = camera()
  r1 = fit_3dgs.render(rp, 1, viewmat, K, W, H)[0]
  assert torch.equal(r1, fit_3dgs.render(rp, 0, viewmat, K, W, H)[0])   # SH1 == 0: the SH0 render
  assert torch.equal(r1, fit_3dgs.render(rd, 1, viewmat, K, W, H)[0])
  # gradients reach the SH1 rows of g's last layer, and the decoded shN is what render/export use
  m.configure_optimizers()
  target = torch.rand(1, 64, 64, 3, device="cuda")
  ((fit_3dgs.render(m._render_params(), 1, viewmat, K, W, H)[0] - target) ** 2).mean().backward()
  assert m.latent.linears[-1].weight.grad[14:].abs().sum() > 0
  with torch.no_grad():
    torch.nn.init.normal_(m.latent.linears[-1].weight, std=0.1)
  sh = m.latent(fit_3dgs.pack_latent(m.params))[:, 14:].detach().reshape(96, 3, 3)
  assert torch.allclose(m._render_params()["shN"], sh)
  assert np.allclose(m.params_np()["shN"], sh.cpu().numpy())
  assert m.latent_np()["decoded"].shape == (96, 23)
