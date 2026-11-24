"""reanchor.py: the A -> B frame transform (rotate_rows + moved anchors == encode_grid with A's anchors in B's camera and B's pose),
the packed cross-attention (each sample's B rows attend only to its own A rows, == a per-sample dense reference; varlen == SDPA
under bf16 on CUDA), the joint packing (a batch == its samples run alone) and the B-only switch (no cross-attention parameters,
output independent of A; with A the output depends on it)."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import reanchor as ra
from cameras import OPENGL_TO_OPENCV
from grid_io import Grid
from gs_diffusion import PackIndex, encode_grid

PS, L, G = 14, 2, 2          # patch, layers, patches per side
H = PS * G


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def pose(rng):
  """A random OpenGL camera-to-world (rotation + translation)."""
  p = np.eye(4, dtype=np.float32)
  p[:3, :3] = Rotation.random(random_state=rng).as_matrix()
  p[:3, 3] = rng.normal(size=3) * 2
  return p


def synthetic(rng, pose_gl):
  """A random grid (world rows) on a random hit mask, with the camera-frame anchors of a camera at pose_gl."""
  hit = rng.random((L, H, H)) < 0.4
  n = int(hit.sum())
  c2w = pose_gl @ OPENGL_TO_OPENCV
  xyz_cam = np.full((L, H, H, 3), np.nan, np.float32)
  xyz_cam[hit] = rng.normal(size=(n, 3)).astype(np.float32) + [0, 0, 3]
  world_anchor = xyz_cam[hit] @ c2w[:3, :3].T + c2w[:3, 3]
  rows = {"gaussian_means": (world_anchor + 0.05 * rng.normal(size=(n, 3))).astype(np.float32),
          "gaussian_scales": rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32),
          "gaussian_quats": Rotation.random(n, random_state=rng).as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
          "gaussian_opacities": rng.uniform(0.05, 0.95, n).astype(np.float32),
          "gaussian_colors": rng.uniform(0, 1, (n, 3)).astype(np.float32)}
  return Grid(hit, np.nonzero(hit), rows), xyz_cam, hit


def source(xyz_cam, hit, pose_gl):
  return {"xyz_cam": torch.from_numpy(xyz_cam), "hit": torch.from_numpy(hit), "pose_gl": torch.from_numpy(pose_gl),
          "K_depth": torch.tensor([[500.0, 0, H / 2], [0, 500.0, H / 2], [0, 0, 1]])}


def test_frame_transform_matches_encode_grid():
  rng = np.random.default_rng(0)
  pa, pb = pose(rng), pose(rng)
  grid, xyz_a, hit = synthetic(rng, pa)
  rows_a = encode_grid(grid, source(xyz_a, hit, pa))[:, hit].T                                   # A's own frame
  R, t = ra.relative(ra.c2w_cv(torch.from_numpy(pa).double()), ra.c2w_cv(torch.from_numpy(pb).double()))
  got = ra.rotate_rows(torch.from_numpy(rows_a).double(), R).numpy()
  xyz_ab = xyz_a.copy()
  xyz_ab[hit] = (torch.from_numpy(xyz_a[hit]).double() @ R.T + t).numpy()
  want = encode_grid(grid, source(xyz_ab, hit, pb))[:, hit].T                                    # A's grid in B's camera
  np.testing.assert_allclose(got, want, atol=2e-5)
  # and the moved anchors are the same world points
  c2w_b = pb @ OPENGL_TO_OPENCV
  c2w_a = pa @ OPENGL_TO_OPENCV
  np.testing.assert_allclose(xyz_ab[hit] @ c2w_b[:3, :3].T + c2w_b[:3, 3], xyz_a[hit] @ c2w_a[:3, :3].T + c2w_a[:3, 3], atol=1e-4)


def test_quat_mul_matches_scipy():
  rng = np.random.default_rng(1)
  a, b = Rotation.random(5, random_state=rng), Rotation.random(5, random_state=rng)
  wxyz = lambda r: torch.from_numpy(r.as_quat()[:, [3, 0, 1, 2]])
  got = ra.quat_mul(wxyz(a), wxyz(b))
  want = wxyz(a * b)
  got = torch.where(got[:, :1] < 0, -got, got)
  want = torch.where(want[:, :1] < 0, -want, want)
  torch.testing.assert_close(got, want)


def test_cross_attention_packed_matches_per_sample():
  torch.manual_seed(0)
  heads, hd = 2, 16
  lens_q, lens_k = [3, 5], [4, 2]
  q = torch.randn(sum(lens_q), heads, hd)
  k, v = torch.randn(sum(lens_k), heads, hd), torch.randn(sum(lens_k), heads, hd)
  cu = lambda n: torch.tensor([0, *np.cumsum(n)], dtype=torch.int32)
  out = ra.cross_segment_attention(q, k, v, cu(lens_q), cu(lens_k), max(lens_q), max(lens_k))
  ref = []
  for i in range(2):
    qi = q[sum(lens_q[:i]):sum(lens_q[:i + 1])]
    ki, vi = (x[sum(lens_k[:i]):sum(lens_k[:i + 1])] for x in (k, v))
    att = torch.einsum("qhd,khd->hqk", qi, ki) * hd ** -0.5
    ref.append(torch.einsum("hqk,khd->qhd", att.softmax(-1), vi))
  torch.testing.assert_close(out, torch.cat(ref))


@pytest.mark.cuda
def test_cross_attention_varlen_matches_sdpa():
  if not torch.cuda.is_available():
    pytest.skip("no CUDA")
  torch.manual_seed(0)
  lens_q, lens_k = [7, 33], [12, 5]
  q, k, v = (torch.randn(sum(n), 2, 64, device="cuda") for n in (lens_q, lens_k, lens_k))
  cu = lambda n: torch.tensor([0, *np.cumsum(n)], dtype=torch.int32, device="cuda")
  ref = ra.cross_segment_attention(q, k, v, cu(lens_q), cu(lens_k), max(lens_q), max(lens_k))
  with torch.autocast("cuda", dtype=torch.bfloat16):
    out = ra.cross_segment_attention(q, k, v, cu(lens_q), cu(lens_k), max(lens_q), max(lens_k))
  torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def randomize_(net, seed=3):
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for _, p in net.named_parameters():
      p.copy_(0.05 * torch.randn(p.shape, generator=g))
    for blk in net.trunk.blocks:
      blk.ls1.fill_(1.0)
      blk.ls2.fill_(1.0)
    for c in net.cross or []:
      c.ls.fill_(1.0)
  return net


def batch(seed, b=2):
  g = torch.Generator().manual_seed(seed)
  keep = lambda: torch.rand(b, L, G * G, generator=g) < 0.7
  ka, kb = keep(), keep()
  ka[:, 0, 0] = kb[:, 0, 0] = True
  packs = ra.Packs(kb, ka)
  f = PS * PS * ra.SLOT_FEATS
  return (torch.randn(packs.a.T, f, generator=g), torch.randn(packs.a.T, 3, generator=g),
          torch.randn(packs.b.T, f, generator=g), torch.randn(packs.b.T, 3, generator=g), packs, ka, kb)


def net(use_a, seed=0):
  torch.manual_seed(seed)
  return randomize_(ra.ReanchorNet(dim=128, num_blocks=3, num_layers=L, patch_size=PS, num_heads=2, use_a=use_a,
                                   checkpoint_activations=False)).eval()


def test_batch_equals_samples_alone():
  m = net(True)
  ca, pa, cb, pb, packs, ka, kb = batch(0)
  out = m(ca, pa, cb, pb, packs)
  for i in range(2):
    sa = PackIndex(ka).b == i
    sb = PackIndex(kb).b == i
    alone = m(ca[sa], pa[sa], cb[sb], pb[sb], ra.Packs(kb[i:i + 1], ka[i:i + 1]))
    torch.testing.assert_close(out[sb], alone, atol=1e-5, rtol=1e-4)


def test_output_depends_on_own_a_only():
  m = net(True)
  ca, pa, cb, pb, packs, ka, kb = batch(1)
  out = m(ca, pa, cb, pb, packs)
  s1a, sb = packs.a.b == 1, packs.b.b
  ca2 = ca.clone()
  ca2[s1a] += 1.0                                                     # sample 1's A changed
  out2 = m(ca2, pa, cb, pb, packs)
  torch.testing.assert_close(out2[sb == 0], out[sb == 0])             # sample 0 untouched
  assert (out2[sb == 1] - out[sb == 1]).abs().max() > 1e-3            # sample 1 reads its A


def test_b_only_switch():
  m = net(False)
  assert m.cross is None and not any("cross" in n for n, _ in m.named_parameters())
  full = net(True)
  extra = sum(p.numel() for n, p in full.named_parameters() if n.startswith("cross."))
  assert sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in m.parameters()) == extra > 0
  ca, pa, cb, pb, _, ka, kb = batch(2)
  packs = ra.Packs(kb)                                                # no A pack
  out = m(None, None, cb, pb, packs)
  out2 = m(ca, pa, cb, pb, packs)                                     # an A given anyway is ignored
  torch.testing.assert_close(out, out2)
  assert out.shape == (packs.b.T, PS * PS * 14)


def test_fresh_model_readout_zero_and_branch_tokens():
  m = ra.ReanchorNet(dim=128, num_blocks=3, num_layers=L, patch_size=PS, num_heads=2, use_a=True, checkpoint_activations=False)
  ca, pa, cb, pb, packs, _, _ = batch(3)
  assert m(ca, pa, cb, pb, packs).abs().max() == 0                   # zero-initialized readout: the mean Gaussian everywhere
  # branch_tokens: per-slot relative anchors average to 0 over a token's hit slots, the keep mask = any hit
  rng = np.random.default_rng(0)
  hit = torch.from_numpy(rng.random((L, H, H)) < 0.3)
  xyz = torch.randn(int(hit.sum()), 3)
  keep, content, pos = ra.branch_tokens(torch.zeros(14), xyz, hit, torch.zeros(3), 1.0, 1.0, PS)
  c = content.reshape(L, G * G, PS * PS, ra.SLOT_FEATS)
  assert torch.equal(keep, c[..., -1].sum(-1) > 0)
  torch.testing.assert_close((c[..., 14:17] * c[..., -1:]).sum(2)[keep], torch.zeros(int(keep.sum()), 3), atol=1e-5, rtol=0)
