"""diffusion.rope: 2d (the frame / global blocks' RoPE on the patch row, column; as before) | xyz_mean (on each token's mean 3D hit
point, token_positions): the tables (rope_xyz_cos_sin: axis split, frequency range, apply_rope's pair layout), the positions (means,
scale, centring, the no-hit fallback), the head (pos required / refused, translation invariance, dense = packed, positions matter),
apply_rope with per-token tables, and the model's inputs()."""
import math

import pytest
import torch

import gs_diffusion as gd
from gs_decoder import apply_rope, rope_cos_sin
from test_token_drop import GRID, FEAT, L, P, PS, hit_masks, inputs, nontrivial_
from test_cond_dropout import model_cfg, stats

H = GRID * PS


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def head(rope="xyz_mean", seed=0):
  torch.manual_seed(seed)
  h = gd.DiffusionHead(FEAT, L, patch_size=PS, rope=rope)
  nontrivial_(h.trunk, seed)
  torch.nn.init.normal_(h.out.weight, std=0.02)
  return h


def positions(b=4, seed=3):
  return 5.0 * torch.randn(b, L, P, 3, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------- the tables

def test_tables_split_range_and_layout():
  hd = 64
  pos = torch.zeros(1, 3)
  cos, sin = gd.rope_xyz_cos_sin(pos, hd)
  assert cos.shape == (1, hd) and torch.equal(cos, torch.ones(1, hd)) and torch.equal(sin, torch.zeros(1, hd))
  angles = {}
  for a in range(3):                                   # which pairs move with which axis, and their frequencies
    p = torch.zeros(1, 3)
    p[0, a] = 1.0
    c, s = gd.rope_xyz_cos_sin(p, hd)
    ang = torch.atan2(s, c)[0]
    torch.testing.assert_close(ang[:16], ang[16:32])   # rotate-half: dim k pairs with k + 16 in each half
    torch.testing.assert_close(ang[32:48], ang[48:])
    angles[a] = torch.cat([ang[:16], ang[32:48]])      # the 32 pairs' angles at unit offset
  moved = {a: (angles[a].abs() > 0).sum().item() for a in range(3)}
  assert moved == {0: 11, 1: 11, 2: 10}
  assert all(((angles[a] != 0).int() * sum((angles[b] != 0).int() for b in range(3) if b != a)).sum() == 0 for a in range(3))
  lo = math.exp(-math.log(100.0) * 30 / 32)            # the 2D RoPE's range: 1 .. 100^(-30/32) rad per unit
  for a in range(3):
    f = angles[a][angles[a] != 0]
    torch.testing.assert_close(f.max(), torch.tensor(1.0))
    torch.testing.assert_close(f.min(), torch.tensor(lo))
  c2, _ = rope_cos_sin(36, hd)
  torch.testing.assert_close(torch.acos(c2[36, :16].clamp(-1, 1)).min(), torch.tensor(lo), rtol=1e-3, atol=1e-5)   # row 1, col 0


def test_apply_rope_per_token_tables_match_shared():
  t = torch.randn(3, 5, 2, 64)
  cos, sin = rope_cos_sin(3, 64)
  cos, sin = cos[:5], sin[:5]
  old = t * cos[:, None, :] + torch.cat([-t.unflatten(-1, (2, -1))[..., 16:], t.unflatten(-1, (2, -1))[..., :16]], -1).flatten(-2) \
      * sin[:, None, :]
  torch.testing.assert_close(apply_rope(t, cos, sin), old, rtol=0, atol=0)          # (S, hd): exactly as before
  torch.testing.assert_close(apply_rope(t, cos.expand(3, 5, 64), sin.expand(3, 5, 64)), old, rtol=0, atol=0)


# ---------------------------------------------------------------------------- the positions

def test_token_positions_means_scale_centre_and_fallback():
  ps, g = 2, 3
  K = torch.tensor([[[10.0, 0, 3.0], [0, 10.0, 3.0], [0, 0, 1]]])
  xyz = torch.full((1, 2, g * ps, g * ps, 3), float("nan"))
  vals = torch.arange(1, 1 + 2 * g * g * ps * ps * 3, dtype=torch.float32).reshape(2, g * ps, g * ps, 3) / 10
  xyz[0] = vals
  xyz[0, 1, :ps, :ps] = float("nan")                   # layer 1, patch 0: no hit at all
  xyz[0, 0, 0, 0] = float("nan")                       # layer 0, patch 0: 3 of 4 slots hit
  pos = gd.token_positions(xyz, K, ps, z_ref=2.0)
  assert pos.shape == (1, 2, g * g, 3) and torch.isfinite(pos).all()
  scale = 10.0 / (ps * 2.0)
  pts = vals[0, :ps, :ps].reshape(-1, 3)[1:]
  raw = torch.stack([vals[l, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps].reshape(-1, 3).mean(0)
                     for l in range(2) for r in range(g) for c in range(g)]).reshape(2, g * g, 3)
  raw[0, 0] = pts.mean(0)                              # only the hit slots
  has = torch.ones(2, g * g, dtype=torch.bool)
  has[1, 0] = False
  centre = raw[has].nanmedian(0).values
  torch.testing.assert_close(pos[0][has], (raw[has] - centre) * scale)
  z_med = raw[has][:, 2].nanmedian()
  ray = torch.linalg.inv(K[0]) @ torch.tensor([0.5 * ps, 0.5 * ps, 1.0])            # patch 0's centre pixel
  torch.testing.assert_close(pos[0, 1, 0], (ray * z_med - centre) * scale)


def test_token_positions_translation_of_the_scene_moves_nothing():
  K = torch.tensor([[[50.0, 0, 6.0], [0, 50.0, 6.0], [0, 0, 1]]])
  xyz = torch.randn(1, L, H, H, 3) + torch.tensor([0, 0, 3.0])
  xyz[:, :, :2] = float("nan")
  a = gd.token_positions(xyz, K, PS, 1.8)
  b = gd.token_positions(xyz + torch.tensor([0.3, -0.2, 0.5]), K, PS, 1.8)
  hit = torch.isfinite(xyz[..., 2]).reshape(1, L, GRID, PS, GRID, PS).any(5).any(3).reshape(1, L, P)
  torch.testing.assert_close(a[hit], b[hit], rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------- the head

def test_rope_argument_checks():
  feats, x_net, t = inputs(None, 2)
  with pytest.raises(ValueError, match="pos"):
    head()(feats, x_net, t)
  with pytest.raises(ValueError, match="pos"):
    head("2d")(feats, x_net, t, pos=positions(2))
  with pytest.raises(ValueError, match="rope"):
    gd.DiffusionHead(FEAT, L, patch_size=PS, rope="3d")


def test_2d_is_the_default_and_has_the_same_parameters():
  assert gd.DiffusionHead(FEAT, L, patch_size=PS).rope == "2d"
  assert set(head("2d").state_dict()) == set(head("xyz_mean").state_dict())


def test_translation_invariance():
  h = head()
  feats, x_net, t = inputs(None, 4)
  pos = positions()
  with torch.no_grad():
    a = h(feats, x_net, t, pos=pos)
    b = h(feats, x_net, t, pos=pos + torch.tensor([3.0, -7.0, 11.0]))
  torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_positions_matter():
  h = head()
  feats, x_net, t = inputs(None, 4)
  pos = positions()
  with torch.no_grad():
    a = h(feats, x_net, t, pos=pos)
    b = h(feats, x_net, t, pos=pos.flip(2))
  assert (a - b).abs().max() > 1e-3


def test_dense_and_packed_agree():
  """On a full hit mask (every token kept) the packed path computes the dense one."""
  h = head()
  feats, x_net, t = inputs(None, 4)
  pos = positions()
  full = torch.ones(4, L, H, H, dtype=torch.bool)
  with torch.no_grad():
    dense = h(feats, x_net, t, pos=pos)
    packed = h(feats, x_net, t, gd.PackIndex(gd.token_keep(full, PS, "empty")), pos=pos)
  torch.testing.assert_close(packed, dense, rtol=1e-4, atol=1e-5)


def test_packed_with_dropped_tokens_uses_each_rows_own_position():
  """Dropped tokens' positions don't matter (they never enter the trunk); the kept ones' do."""
  h = head()
  hit = hit_masks()
  feats, x_net, t = inputs(None, 4)
  pack = gd.PackIndex(gd.token_keep(hit, PS, "empty"))
  keep = gd.token_keep(hit, PS, "empty")
  pos = positions()
  moved = torch.where(keep[..., None], pos, pos + 100.0)
  with torch.no_grad():
    a = h(feats, x_net, t, pack, pos=pos)
    b = h(feats, x_net, t, pack, pos=moved)
  torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_gradients_flow():
  h = head()
  feats, x_net, t = inputs(None, 2)
  h(feats, x_net, t, pos=positions(2)).square().mean().backward()
  assert all(p.grad is not None for n, p in h.named_parameters() if p.requires_grad and "null" not in n)


# ---------------------------------------------------------------------------- the model

def batch(b=2):
  K = torch.tensor([[50.0, 0, 6.0], [0, 50.0, 6.0], [0, 0, 1]]).expand(b, 3, 3)
  xyz = torch.randn(b, L, H, H, 3) + torch.tensor([0, 0, 3.0])
  return {"image": torch.randn(b, L, P, FEAT), "source_view": list(range(b)),
          "source": {"hit": torch.ones(b, L, H, H, dtype=torch.bool), "xyz_cam": xyz, "K_depth": K}}


def test_model_inputs_and_config():
  m2 = gd.GSDiffusionModel(model_cfg(), stats())       # a config without the keys: 2d, no positions
  assert m2.rope == "2d" and m2.inputs(batch(), torch.device("cpu"))[4] is None
  m3 = gd.GSDiffusionModel(model_cfg(rope="xyz_mean", rope_z_ref=2.0), stats())
  pos = m3.inputs(batch(), torch.device("cpu"))[4]
  assert pos.shape == (2, L, P, 3) and m3.rope_z_ref == 2.0
  m3.load_state_dict(m2.state_dict(), strict=True)     # no parameters either way
