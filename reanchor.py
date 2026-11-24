"""The re-anchoring task (train_reanchor.py): view A's pixel-aligned layered 3DGS grid + view B's depth-peel point cloud -> B's grid,
in ONE forward pass, trained photometrically only (no x0 target). Neither A's nor B's RGB image is an input.

Frames. Everything the network sees and predicts is in B's camera frame (OpenCV axes): A's per-slot offsets and quaternions,
stored in A's camera frame by gs_diffusion.encode_grid, are rotated into it (a_rows_in_b; exactly what encode_grid gives for A's
grid with A's anchors moved into B's camera and B's pose, tests/test_reanchor.py), A's anchors moved into it, and both branches'
token positions are centred on ONE centre (B's median token position) and scaled with ONE scale (B's fx / (patch * z_ref)).

Tokens (both branches, the same layout and the same in_proj): one token per (layer, 14 x 14 patch) with >= 1 hit slot
(token_drop empty); per slot [14 normalized params | slot anchor minus the token's mean anchor (in units of slot_scale_px pixels at
z_ref) | hit]. A's params: its fitted Gaussians in B's frame, normalized with moments computed in the paired frame
(paired_moments). B's params: a learned "empty" 14-vector at every hit slot (zero at the others). A learned branch embedding is
added after in_proj. RoPE: the 3D RoPE on each token's mean anchor (rope_xyz_cos_sin, ported from branch rope-xyz fe6956a).

Trunk (ReanchorNet): gs_diffusion.DiTTrunk's blocks (all-global attention, QK-norm, LayerScale, per-layer FiLM at blocks 0/3/6),
shared by both branches; no timestep: every block gets a CONSTANT AdaLN input (gates 1, shifts / scales 0: a plain pre-LN block
with LayerScale whose learned `modulation` is still applied around it). Wiring per block k (A and B packed as 2B segments, one
self-attention call):
    A_k' = A_{k-1} + selfattn(A_{k-1})        B_k' = B_{k-1} + selfattn(B_{k-1})      (each branch attends within itself)
                                              B_k'' = B_k' + crossattn(B_k' -> A_k')    (B queries, A keys / values; new code)
    A_k = A_k' + MLP(A_k')                    B_k = B_k'' + MLP(B_k'')
i.e. B reads A's SAME-depth states (after that block's self-attention). Cross-attention: its own pre-norms, QK-norm, RoPE at B's
token positions for q and A's for k (one frame), LayerScale init layer_scale_init, varlen flash (bf16) or a per-sample SDPA loop.
Readout: a zero-initialized per-token Linear on B's rows -> 196 x 14 normalized params, decoded by gs_diffusion.decode_x0 with
the B-own-frame moments and clamp=scale (only the scale channels clamped: offsets, colours, opacities keep their gradient).

B-only control (use_a false): the A branch is absent -- no A tokens, no cross-attention sublayers (not built: 4 D^2 per block
fewer parameters) -- the rest identical. (Zeroed A tokens would still carry A's geometry through their positions.)
"""

import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from cameras import OPENGL_TO_OPENCV, dense_unproject_camera
from gaussians import SH_C0, X0_CHANNELS, X0_SLICES
from grid_io import Grid, read_grid
from gs_decoder import apply_rope
from gs_diffusion import DiTTrunk, PackIndex, decode_x0, encode_grid, matrix_to_quat, segment_attention

OFFSET, ROTATION, SCALE = X0_SLICES["offset"], X0_SLICES["rotation"], X0_SLICES["scale"]
OPACITY, SH_DC = X0_SLICES["opacity"], X0_SLICES["sh_dc"]
SLOT_FEATS = X0_CHANNELS + 3 + 1   # params | relative anchor | hit


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------

def rope_xyz_cos_sin(pos, head_dim, base=100.0):
  """(ported unchanged from branch rope-xyz fe6956a, gs_diffusion.rope_xyz_cos_sin) 3D RoPE tables from continuous positions:
  pos (..., 3) (x, y, z, in patch units) -> (cos, sin), each (..., head_dim), in the layout apply_rope rotates (rotate-half within
  each half of head_dim). The head_dim / 2 rotation pairs are split over the axes (z gets a third, rounded down; x and y share the
  rest: 11 / 11 / 10 at head_dim 64), each axis with its own geometric ladder of frequencies over the 2D RoPE's range."""
  half = head_dim // 2
  nz = half // 3
  nx = (half - nz + 1) // 2
  ny = half - nz - nx
  lo = math.log(base) * (half - 2) / half
  ladder = lambda n: torch.exp(-torch.linspace(0.0, lo, n, device=pos.device, dtype=torch.float32))
  ang = torch.cat([pos[..., a:a + 1].float() * ladder(n) for a, n in enumerate((nx, ny, nz))], dim=-1)
  q = half // 2
  ang = torch.cat([ang[..., :q], ang[..., :q], ang[..., q:], ang[..., q:]], dim=-1)
  return ang.cos(), ang.sin()


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def c2w_cv(pose_gl):
  """(4, 4) OpenGL camera-to-world (the render h5's camera_pose) -> OpenCV camera-to-world."""
  return pose_gl @ torch.as_tensor(OPENGL_TO_OPENCV, device=pose_gl.device, dtype=pose_gl.dtype)


def quat_mul(a, b):
  """Hamilton product of (..., 4) wxyz quaternions: the rotation a after b."""
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def rotate_rows(rows, R):
  """Raw x0 rows (N, 14) of a grid in one camera frame -> the same Gaussians' rows in another, R (3, 3) the rotation from the
  first frame to the second: offsets rotated, quaternions R q (sign w >= 0); opacity, scales, colours unchanged."""
  out = rows.clone()
  out[:, OFFSET] = rows[:, OFFSET] @ R.T
  q = quat_mul(matrix_to_quat(R[None].float()).to(rows.dtype).expand(len(rows), 4), rows[:, ROTATION])
  out[:, ROTATION] = torch.where(q[:, :1] < 0, -q, q)
  return out


def relative(c2w_from, c2w_to):
  """(R, t) taking points in camera `from`'s frame to camera `to`'s (both (4, 4) OpenCV camera-to-world)."""
  T = torch.linalg.inv(c2w_to) @ c2w_from
  return T[:3, :3], T[:3, 3]


# ---------------------------------------------------------------------------
# one mesh, GPU-resident
# ---------------------------------------------------------------------------

class MeshData:
  """One mesh of a render h5 and its per-view grids, on the GPU. Per view v (the mesh's 0-based view number): gt (3, H, W)
  premultiplied over black (rgb * alpha: the fitter's convention; the h5 images are straight alpha), alpha (1, H, W), K_image,
  c2w (OpenCV). Per grid view (the ones with a grid file): xyz (N, 3) the hit slots' anchors in the view's own camera frame and
  hit (L, H, W), in np.nonzero(hit) order -- the grid's row order --, raw x0 rows (N, 14) in the own frame (encode_grid, default
  encoding), and the world-frame grid rows (for the no-learning references)."""

  def __init__(self, render_h5, grids_dir, mesh, grid_views, all_views=50, device="cuda", num_layers=6):
    self.device = device
    with h5py.File(render_h5, "r") as f:
      rows = np.flatnonzero(np.asarray(f["mesh_index"]) == mesh)
      if len(rows) < all_views:
        raise SystemExit(f"mesh {mesh}: {len(rows)} rows in {render_h5}, expected {all_views}")
      self.rows = rows[:all_views]
      t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
      gt, alpha, K, c2w, Kd = [], [], [], [], []
      for r in self.rows:
        im = np.asarray(f["images"][r]).astype(np.float32) / 255
        a = im[..., 3:4]
        gt.append(t((im[..., :3] * a).transpose(2, 0, 1)))
        alpha.append(t(a.transpose(2, 0, 1)))
        K.append(t(f["image_intrinsics"][r]))
        Kd.append(t(f["depth_intrinsics"][r]))
        c2w.append(c2w_cv(t(f["camera_pose"][r])))
      self.gt, self.alpha, self.K, self.c2w, self.K_depth = (torch.stack(x) for x in (gt, alpha, K, c2w, Kd))
      self.H, self.W = self.gt.shape[-2:]
      self.grid_views = list(grid_views)
      self.xyz, self.hit, self.x0, self.world = {}, {}, {}, {}
      for v in self.grid_views:
        r = int(self.rows[v])
        depth = np.asarray(f["depth_peel"][r]).astype(np.float32)
        xyz_cam, hit = dense_unproject_camera(depth, np.asarray(f["depth_intrinsics"][r], np.float32))
        xyz_cam, hit = xyz_cam.transpose(2, 0, 1, 3), hit.transpose(2, 0, 1)                     # (L, H, W, .)
        if hit.shape[0] != num_layers:
          raise SystemExit(f"view {v}: {hit.shape[0]} depth-peel layers, expected {num_layers}")
        path = os.path.join(grids_dir, f"mesh{mesh}_view{v}.h5")
        grid = read_grid(path)
        if int(grid.attrs.get("primary_view_row", r)) != r:
          raise SystemExit(f"{path}: primary_view_row {grid.attrs.get('primary_view_row')} != render row {r}")
        source = {"xyz_cam": torch.from_numpy(xyz_cam), "hit": torch.from_numpy(hit), "pose_gl": torch.from_numpy(
          np.asarray(f["camera_pose"][r], np.float32)), "K_depth": torch.from_numpy(np.asarray(f["depth_intrinsics"][r], np.float32))}
        x0 = encode_grid(grid, source)                                                              # (14, L, H, W)
        li, yi, xi = np.nonzero(hit)
        self.hit[v] = torch.from_numpy(hit).to(device)
        self.xyz[v] = t(xyz_cam[li, yi, xi])
        self.x0[v] = t(x0[:, li, yi, xi].T)
        self.world[v] = {k: t(val) for k, val in grid.rows.items()}

  def own_moments(self):
    """(mean, std, lo, hi) (14,) of the raw x0 rows over every grid view (each in its own frame): the output's normalization
    and decode_x0's clamp range (as gs_diffusion.compute_stats over these grids)."""
    x = torch.cat([self.x0[v] for v in self.grid_views]).double()
    return (x.mean(0).float(), x.std(0, unbiased=False).clamp_min(1e-4).float(), x.amin(0).float(), x.amax(0).float())

  def a_rows_in_b(self, a, b):
    """A's raw x0 rows and anchors (N_A, 14), (N_A, 3) in B's camera frame."""
    R, t = relative(self.c2w[a], self.c2w[b])
    return rotate_rows(self.x0[a], R), self.xyz[a] @ R.T + t

  def paired_moments(self, pairs):
    """(mean, std) (14,) of A's raw rows in B's frame over the given (a, b) pairs: the A input's normalization."""
    x = torch.cat([self.a_rows_in_b(a, b)[0] for a, b in pairs]).double()
    return x.mean(0).float(), x.std(0, unbiased=False).clamp_min(1e-4).float()

  def viewmats(self, b, views):
    """(V, 4, 4) world-to-camera of `views` with "world" = B's camera frame (cameras.relative_viewmats' convention)."""
    return torch.linalg.inv(self.c2w[list(views)]) @ self.c2w[b]


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

def dense_rows(rows, hit, fill=0.0):
  """(N, C) rows in np.nonzero(hit) order -> (L, H, W, C), `fill` elsewhere."""
  out = rows.new_full(tuple(hit.shape) + (rows.shape[1],), fill)
  out[hit] = rows
  return out


def branch_tokens(params, xyz, hit, centre, scale, rel_scale, patch):
  """One branch of one sample -> (keep (L, P) bool, content (L, P, 196 * SLOT_FEATS) float32, pos (L, P, 3)).
  params (N, 14) normalized rows (or one (14,) vector: B's empty token, put at every hit slot), xyz (N, 3) anchors in the shared
  frame, hit (L, H, W); centre (3,), scale (pos units per metre), rel_scale (slot-feature units per metre)."""
  n = int(hit.sum())
  p = params.expand(n, X0_CHANNELS) if params.dim() == 1 else params
  dense = dense_rows(torch.cat([p.float(), xyz, torch.ones(n, 1, device=xyz.device)], 1), hit)   # (L, H, W, 18)
  pt = rearrange(dense, "l (gh py) (gw px) c -> l (gh gw) (py px) c", py=patch, px=patch)
  h = pt[..., -1:]                                                                                  # (L, P, S, 1)
  cnt = h.sum(2)                                                                                    # (L, P, 1)
  mean = (pt[..., X0_CHANNELS:X0_CHANNELS + 3] * h).sum(2) / cnt.clamp_min(1)                      # (L, P, 3)
  rel = (pt[..., X0_CHANNELS:X0_CHANNELS + 3] - mean[:, :, None]) * h * rel_scale
  content = torch.cat([pt[..., :X0_CHANNELS], rel, h], -1).flatten(2)
  return cnt[..., 0] > 0, content, (mean - centre) * scale


def token_centre(xyz, hit, patch):
  """The median of a branch's token mean anchors (3,) (token_positions' centre)."""
  dense = dense_rows(torch.cat([xyz, torch.ones(len(xyz), 1, device=xyz.device)], 1), hit)
  pt = rearrange(dense, "l (gh py) (gw px) c -> l (gh gw) (py px) c", py=patch, px=patch)
  cnt = pt[..., 3].sum(2)
  mean = pt[..., :3].sum(2) / cnt.clamp_min(1)[..., None]
  return mean[cnt > 0].median(0).values


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def cross_segment_attention(q, k, v, cu_q, cu_k, max_q, max_k):
  """q (Tq, H, hd), k / v (Tk, H, hd) -> (Tq, H, hd): sample i's query rows cu_q[i]:cu_q[i+1] attend to its key rows
  cu_k[i]:cu_k[i+1]. Under CUDA bf16 / fp16 autocast torch's varlen flash kernel, otherwise SDPA per sample."""
  if q.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16):
    from torch.nn.attention.varlen import varlen_attn
    dt = torch.get_autocast_dtype("cuda")
    return varlen_attn(q.to(dt).contiguous(), k.to(dt).contiguous(), v.to(dt).contiguous(), cu_q, cu_k, max_q, max_k)
  bq, bk = cu_q.tolist(), cu_k.tolist()
  return torch.cat([F.scaled_dot_product_attention(q[a:e].transpose(0, 1), k[c:d].transpose(0, 1), v[c:d].transpose(0, 1))
                    .transpose(0, 1) for a, e, c, d in zip(bq[:-1], bq[1:], bk[:-1], bk[1:])])


class CrossAttention(nn.Module):
  """Pre-norm cross-attention sublayer: queries from x (B's rows), keys / values from ctx (A's rows); QK-norm, RoPE at each
  side's own positions, LayerScale (init layer_scale_init, as the blocks' ls1 / ls2)."""

  def __init__(self, dim, num_heads, layer_scale_init=1e-4):
    super().__init__()
    self.num_heads = num_heads
    hd = dim // num_heads
    self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.q = nn.Linear(dim, dim, bias=False)
    self.kv = nn.Linear(dim, 2 * dim, bias=False)
    self.q_norm, self.k_norm = nn.LayerNorm(hd), nn.LayerNorm(hd)
    self.proj = nn.Linear(dim, dim)
    self.ls = nn.Parameter(layer_scale_init * torch.ones(dim))

  def forward(self, x, ctx, rope_q, rope_k, cu_q, cu_k, max_q, max_k):
    tq, d = x.shape
    hd = d // self.num_heads
    q = self.q_norm(self.q(self.norm_q(x)).reshape(tq, self.num_heads, hd))
    k, v = self.kv(self.norm_kv(ctx)).reshape(len(ctx), 2, self.num_heads, hd).unbind(1)
    k = self.k_norm(k)
    q, k = apply_rope(q, *rope_q), apply_rope(k, *rope_k)
    out = cross_segment_attention(q, k, v, cu_q, cu_k, max_q, max_k)
    return self.ls * self.proj(out.reshape(tq, d))


class Packs:
  """The index tables of one batch: per branch a PackIndex over its (B, L, P) keep mask (gather / scatter, per-sample cu), and the
  joint one over [A samples; B samples] (2B segments: the shared self-attention runs once on all rows, A's rows first)."""

  def __init__(self, keep_b, keep_a=None):
    self.b = PackIndex(keep_b)
    self.a = PackIndex(keep_a) if keep_a is not None else None
    for name, p in (("B", self.b), ("A", self.a)):
      if p is not None and len(p.cu_global) != p.B + 1:   # PackIndex._cu drops empty samples: the cu would misalign
        raise ValueError(f"branch {name}: a sample without any kept token")
    self.joint = PackIndex(torch.cat([keep_a, keep_b])) if keep_a is not None else self.b
    self.ta = self.a.T if keep_a is not None else 0


class ReanchorNet(nn.Module):
  """See the module docstring. forward(content_a, pos_a, content_b, pos_b, packs) -> B's normalized params (T_B, 196 * 14) in the
  tokens' (py px c) slot order."""

  def __init__(self, dim=2048, num_blocks=9, num_layers=6, patch_size=14, num_heads=None, use_a=True, mlp_ratio=4.0,
               layer_scale_init=1e-4, checkpoint_activations=True):
    super().__init__()
    self.patch_size, self.use_a, self.checkpoint_activations = patch_size, bool(use_a), checkpoint_activations
    self.num_layers = num_layers
    num_heads = num_heads or dim // 64
    s = patch_size ** 2
    self.in_proj = nn.Linear(s * SLOT_FEATS, dim)
    self.branch = nn.Parameter(torch.randn(2, dim) * 0.02)          # [A, B]
    self.empty = nn.Parameter(torch.zeros(X0_CHANNELS))              # B's slot params (normalized units)
    self.trunk = DiTTrunk(dim, num_layers, num_blocks, num_heads, mlp_ratio, layer_scale_init, False, "global", "frame")
    self.cross = nn.ModuleList(CrossAttention(dim, num_heads, layer_scale_init) for _ in range(num_blocks)) if self.use_a else None
    self.out = nn.Linear(dim, s * X0_CHANNELS)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)
    ada = torch.zeros(6, dim)
    ada[2] = ada[5] = 1.0                                            # gates 1: no timestep, a plain pre-LN LayerScale block
    self.register_buffer("adaln", ada, persistent=False)

  def _head_dim(self):
    return self.trunk.blocks[0].qkv.in_features // self.trunk.num_heads

  def _unit(self, k, x, rope, film, packs, rope_a, rope_b):
    """Block k on the joint rows x (T, D) (A's rows first), with B's cross-attention to A between attention and MLP."""
    block = self.trunk.blocks[k]
    if film is not None:
      x = x * (1 + film[0]) + film[1]
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (self.adaln + block.modulation).unbind(0)
    h = (block.norm1(x).float() * (1 + scale_a) + shift_a).to(x.dtype)
    x = x + (block.ls1 * block._mix_packed(h, packs.joint, *rope) * gate_a).to(x.dtype)
    if self.use_a:
      xa, xb = x[:packs.ta], x[packs.ta:]
      xb = xb + self.cross[k](xb, xa, rope_b, rope_a, packs.b.cu_global, packs.a.cu_global, packs.b.max_global,
                              packs.a.max_global).to(x.dtype)
      x = torch.cat([xa, xb])
    h = (block.norm2(x).float() * (1 + scale_m) + shift_m).to(x.dtype)
    return x + (block.ls2 * block.mlp(h) * gate_m).to(x.dtype)

  def forward(self, content_a, pos_a, content_b, pos_b, packs):
    hd = self._head_dim()
    xb = self.in_proj(content_b) + self.branch[1]
    rope_b = rope_xyz_cos_sin(pos_b, hd)
    if self.use_a:
      xa = self.in_proj(content_a) + self.branch[0]
      rope_a = rope_xyz_cos_sin(pos_a, hd)
      x = torch.cat([xa, xb])
      rope = tuple(torch.cat([u, w]) for u, w in zip(rope_a, rope_b))
    else:
      x, rope, rope_a = xb, rope_b, None
    films = self.trunk._films(self.num_layers, x.device, lambda v: v[packs.joint.l])
    for k, film in enumerate(films):
      if self.checkpoint_activations and torch.is_grad_enabled():
        x = torch.utils.checkpoint.checkpoint(self._unit, k, x, rope, film, packs, rope_a, rope_b, use_reentrant=False)
      else:
        x = self._unit(k, x, rope, film, packs, rope_a, rope_b)
    return self.out(x[packs.ta:])

  def readout_grid(self, rows, packs):
    """forward's (T_B, 196 * 14) -> (B, L, 14, H, W) normalized x0."""
    s = self.patch_size
    out = packs.b.scatter(rows.float())                                                         # (B, L, P, 196 * 14)
    g = int(round(packs.b.P ** 0.5))
    return rearrange(out, "b l (gh gw) (py px c) -> b l c (gh py) (gw px)", gh=g, gw=g, py=s, px=s, c=X0_CHANNELS)


# ---------------------------------------------------------------------------
# a batch of pairs -> model inputs -> Gaussians
# ---------------------------------------------------------------------------

class Batcher:
  """(a, b) pairs of a MeshData -> the model's inputs, and the decoded prediction -> per-sample render dicts (B's frame)."""

  def __init__(self, data, in_mean, in_std, out_moments, patch_size=14, z_ref=1.8, slot_scale_px=7.0, decode_clamp="scale"):
    self.d = data
    self.in_mean, self.in_std = in_mean, in_std
    self.out_mean, self.out_std, self.lo, self.hi = out_moments
    self.patch, self.z_ref, self.slot_scale_px, self.decode_clamp = patch_size, z_ref, slot_scale_px, decode_clamp

  def inputs(self, pairs, empty, use_a=True, a_override=None):
    """pairs: [(a, b)]; empty: the model's B token (14,). a_override (optional): {i: (rows (N, 14) raw in B's frame, xyz (N, 3),
    hit)} replacing sample i's A (identity / ablation evals)."""
    ka, ca, pa, kb, cb, pb = [], [], [], [], [], []
    for i, (a, b) in enumerate(pairs):
      fx = self.d.K_depth[b][0, 0]
      scale = fx / (self.patch * self.z_ref)                       # one unit = one patch width at depth z_ref
      rel = fx / (self.slot_scale_px * self.z_ref)
      centre = token_centre(self.d.xyz[b], self.d.hit[b], self.patch)
      k, c, p = branch_tokens(empty, self.d.xyz[b], self.d.hit[b], centre, scale, rel, self.patch)
      kb.append(k); cb.append(c); pb.append(p)
      if use_a:
        if a_override is not None and i in a_override:
          rows, xyz, hit = a_override[i]
        else:
          (rows, xyz), hit = self.d.a_rows_in_b(a, b), self.d.hit[a]
        k, c, p = branch_tokens((rows - self.in_mean) / self.in_std, xyz, hit, centre, scale, rel, self.patch)
        ka.append(k); ca.append(c); pa.append(p)
    keep_b = torch.stack(kb)
    keep_a = torch.stack(ka) if use_a else None
    packs = Packs(keep_b, keep_a)
    content_b, pos_b = packs.b.gather(torch.stack(cb)), packs.b.gather(torch.stack(pb))
    content_a = packs.a.gather(torch.stack(ca)) if use_a else None
    pos_a = packs.a.gather(torch.stack(pa)) if use_a else None
    return content_a, pos_a, content_b, pos_b, packs

  def decode(self, x0n):
    """(B, L, 14, H, W) normalized -> gs_diffusion.decode_x0's Gaussian dict."""
    return decode_x0(x0n, self.out_mean, self.out_std, self.lo, self.hi, clamp=self.decode_clamp)

  def gaussians(self, gauss, i, b):
    """Sample i (anchored on view b) of a decoded dict -> render params in B's frame (gsplat's inputs)."""
    hit = self.d.hit[b]
    take = lambda k: gauss[k][i].permute(0, 2, 3, 1)[hit]                                         # (N, C)
    return {"means": self.d.xyz[b] + take("offset"), "quats": take("rotation"), "scales": take("scale"),
            "opacities": take("opacity")[:, 0], "colors": take("sh_dc")[:, None]}


def raw_gaussians(rows, xyz):
  """Raw x0 rows (N, 14) + anchors (N, 3) of one frame -> render params in that frame (no normalization, no clamp)."""
  q = rows[:, ROTATION]
  return {"means": xyz + rows[:, OFFSET], "quats": q / q.norm(dim=1, keepdim=True).clamp_min(1e-12),
          "scales": rows[:, SCALE].exp(), "opacities": torch.sigmoid(rows[:, OPACITY][:, 0]), "colors": rows[:, None, SH_DC]}


def world_gaussians(w):
  """A grid's world-frame rows (MeshData.world) -> render params in the world frame, exactly as fitted (the review's path)."""
  return {"means": w["gaussian_means"], "quats": w["gaussian_quats"] / w["gaussian_quats"].norm(dim=1, keepdim=True),
          "scales": w["gaussian_scales"], "opacities": w["gaussian_opacities"].reshape(-1),
          "colors": ((w["gaussian_colors"] - 0.5) / SH_C0)[:, None]}


def render(p, viewmats, Ks, W, H):
  """gsplat over black: (V, 3, H, W) rgb (unclamped), (V, 1, H, W) alpha."""
  import gsplat
  rgb, alpha, _ = gsplat.rasterization(means=p["means"], quats=p["quats"], scales=p["scales"], opacities=p["opacities"],
                                       colors=p["colors"], viewmats=viewmats, Ks=Ks, width=int(W), height=int(H), sh_degree=0,
                                       packed=True)
  return rgb.permute(0, 3, 1, 2), alpha.permute(0, 3, 1, 2)


def psnr_bw(rgb, alpha, gt, gt_alpha):
  """Per-view full-image PSNR over black and over white: (V,), (V,). rgb / gt premultiplied over black."""
  mse = lambda x, y: (x - y).pow(2).flatten(1).mean(1)
  black = -10 * torch.log10(mse(rgb.clamp(0, 1), gt))
  white = -10 * torch.log10(mse((rgb + 1 - alpha).clamp(0, 1), (gt + 1 - gt_alpha).clamp(0, 1)))
  return black, white


# ---------------------------------------------------------------------------
# the matched copy (bla/_review_nn_baseline.py's floor reference)
# ---------------------------------------------------------------------------

def matched_copy_world(data, a, b, k=32):
  """A's Gaussians (world rows) moved one-to-one into B's slots: in descending opacity each takes its nearest FREE B slot among
  its k nearest (anchor distance, world), keeping its own mean / everything; unassigned A Gaussians dropped, unused B slots get
  opacity 9.1e-4 (the review's). Returns ({field: (N_B, C)} world rows in B's slot order, info)."""
  from scipy.spatial import cKDTree
  wa = {kk: v.cpu().numpy() for kk, v in data.world[a].items()}
  pa = (data.xyz[a] @ data.c2w[a][:3, :3].T + data.c2w[a][:3, 3]).cpu().numpy()
  pb = (data.xyz[b] @ data.c2w[b][:3, :3].T + data.c2w[b][:3, 3]).cpu().numpy()
  _, cand = cKDTree(pb).query(pa, k=k)
  free = np.ones(len(pb), bool)
  src = np.full(len(pb), -1)
  for ai in np.argsort(-wa["gaussian_opacities"].reshape(-1)):
    for bj in cand[ai]:
      if free[bj]:
        free[bj] = False
        src[bj] = ai
        break
  used = src >= 0
  take = np.where(used, src, 0)
  rows = {kk: v[take] for kk, v in wa.items()}
  rows["gaussian_opacities"] = np.where(used, rows["gaussian_opacities"].reshape(-1), 9.1e-4).astype(np.float32)
  op = wa["gaussian_opacities"].reshape(-1)
  dropped = np.setdiff1d(np.arange(len(pa)), src[used])
  return rows, {"dropped": int(len(dropped)), "dropped_opacity_frac": float(op[dropped].sum() / op.sum())}


def world_rows_to_b(data, rows, b):
  """World-frame grid rows in B's slot order -> raw x0 rows (N_B, 14) in B's frame (encode_grid with B's source)."""
  hit = data.hit[b].cpu().numpy()
  grid = Grid(hit, np.nonzero(hit), {k: np.asarray(v, np.float32) for k, v in rows.items()})
  xyz_cam = np.full(hit.shape + (3,), np.nan, np.float32)
  xyz_cam[hit] = data.xyz[b].cpu().numpy()
  pose_gl = (data.c2w[b].cpu().numpy() @ OPENGL_TO_OPENCV).astype(np.float32)   # OPENGL_TO_OPENCV is its own inverse
  source = {"xyz_cam": torch.from_numpy(xyz_cam), "hit": torch.from_numpy(hit), "pose_gl": torch.from_numpy(pose_gl),
            "K_depth": data.K_depth[b].cpu()}
  x0 = encode_grid(grid, source)
  li, yi, xi = np.nonzero(hit)
  return torch.as_tensor(x0[:, li, yi, xi].T, device=data.device)
