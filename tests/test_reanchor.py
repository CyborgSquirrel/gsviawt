"""reanchor.py: the A -> B frame transform (rotate_rows + moved anchors == encode_grid with A's anchors in B's camera and B's pose),
the packed cross-attention (each sample's B rows attend only to its own A rows, == a per-sample dense reference; varlen == SDPA
under bf16 on CUDA), the joint packing (a batch == its samples run alone) and the B-only switch (no cross-attention parameters,
output independent of A; with A the output depends on it)."""
import os

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


# ---------------------------------------------------------------------------
# donor A (train_reanchor data.a_source other_mesh)
# ---------------------------------------------------------------------------

NEW100_H5 = "/app/bla/new100/trellis_new100_50views_eevee_504.h5"
GRIDS79 = "/app/bla/reanchor/grids79"
DONOR_GRIDS = "/app/bla/new100/anchored_wd0.3/grids_sparse"


def test_donor_draws_never_target_and_uniform():
  pool = [m for m in range(100) if m != 79]
  ds = ra.DonorViews("unused.h5", "unused", pool, 45, seed=[0, 0])
  it = ds.draws()
  d = np.array([next(it) for _ in range(30000)])
  assert 79 not in d[:, 0] and set(d[:, 0]) == set(pool) and set(d[:, 1]) == set(range(45))
  counts = np.bincount(d[:, 0], minlength=100)[pool]
  assert counts.min() > 0.7 * counts.mean() and counts.max() < 1.3 * counts.mean()
  it2 = ra.DonorViews("unused.h5", "unused", pool, 45, seed=[0, 0]).draws()
  assert [next(it2) for _ in range(100)] == [tuple(x) for x in d[:100]]             # seeded: a fixed sequence


def test_donor_pool_excludes_target_and_checks_grids(tmp_path):
  import h5py
  h5 = tmp_path / "r.h5"
  with h5py.File(h5, "w") as f:
    f["mesh_index"] = np.repeat([3, 79, 5], 4)
  for m in (3, 79, 5):
    for v in range(4):
      (tmp_path / f"mesh{m}_view{v}.h5").touch()
  assert ra.donor_pool(str(h5), str(tmp_path), 79, 4) == [3, 5]
  (tmp_path / "mesh5_view2.h5").unlink()
  with pytest.raises(SystemExit):
    ra.donor_pool(str(h5), str(tmp_path), 79, 4)


def test_eval_donors_deterministic():
  pool = [m for m in range(100) if m != 79]
  a, b = ra.eval_donors(pool, 45, 30, 0), ra.eval_donors(pool, 45, 30, 0)
  assert a == b and len(a) == 30 and all(m != 79 and 0 <= v < 45 for m, v in a)
  assert ra.eval_donors(pool, 45, 30, 1) != a


def test_a_source_same_inputs_unchanged():
  """rows_to_frame (the shared A -> B transform) == the pre-donor a_rows_in_b formula, bit for bit."""
  rng = np.random.default_rng(5)
  ca = torch.as_tensor(pose(rng) @ OPENGL_TO_OPENCV)
  cb = torch.as_tensor(pose(rng) @ OPENGL_TO_OPENCV)
  x0, xyz = torch.randn(500, 14), torch.randn(500, 3)
  x0[:, ra.ROTATION] = torch.nn.functional.normalize(x0[:, ra.ROTATION], dim=1)
  R, t = ra.relative(ca, cb)
  rows, pts = ra.rows_to_frame(x0, xyz, ca, cb)
  assert torch.equal(rows, ra.rotate_rows(x0, R)) and torch.equal(pts, xyz @ R.T + t)


needs_new100 = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(NEW100_H5) and os.path.isdir(GRIDS79)),
                                  reason="needs CUDA + the local new100 h5 and mesh 79's grids")


@needs_new100
def test_donor_path_with_own_mesh_reproduces_real_a():
  data = ra.MeshData(NEW100_H5, GRIDS79, 79, [3, 17], device="cuda")
  ds = ra.DonorViews(NEW100_H5, GRIDS79, [79], 45, seed=0)
  for a, b in ((3, 17), (17, 3), (3, 3)):
    rows, xyz, hit = ra.donor_in_b(ds.load(79, a), data.c2w[b], "cuda")
    r0, x0 = data.a_rows_in_b(a, b)
    assert torch.equal(rows, r0) and torch.equal(xyz, x0) and torch.equal(hit, data.hit[a])


@needs_new100
@pytest.mark.skipif(not os.path.isdir(DONOR_GRIDS), reason="needs the local new100 donor grids")
def test_donor_in_b_frame_matches_world_render():
  """A donor (another mesh) moved into B's frame and rendered from B-relative cameras == its world grid rendered from the same
  cameras in the world: the donor sits at its world position relative to B."""
  from grid_io import read_grid
  data = ra.MeshData(NEW100_H5, GRIDS79, 79, [11], device="cuda")
  ds = ra.DonorViews(NEW100_H5, DONOR_GRIDS, [5], 45, seed=0)
  b, views = 11, [11, 30, 46]
  d = ds.load(5, 20)
  rows, xyz, _ = ra.donor_in_b(d, data.c2w[b], "cuda")
  r1, _ = ra.render(ra.raw_gaussians(rows, xyz), data.viewmats(b, views), data.K[views], data.W, data.H)
  world = {k: torch.as_tensor(np.asarray(v, np.float32), device="cuda") for k, v in read_grid(f"{DONOR_GRIDS}/mesh5_view20.h5").rows.items()}
  r2, a2 = ra.render(ra.world_gaussians(world), torch.linalg.inv(data.c2w[views]), data.K[views], data.W, data.H)
  assert float(a2.mean()) > 0.02                                                    # the donor is in view
  psnr = -10 * torch.log10((r1.clamp(0, 1) - r2.clamp(0, 1)).pow(2).flatten(1).mean(1))
  assert float(psnr.min()) > 50, psnr
