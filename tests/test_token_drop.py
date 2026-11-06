"""diffusion.token_drop: the packed trunk (PackIndex rows, varlen / per-segment attention, the ray table) against a dense reference in
which the dropped tokens are masked as KEYS in every block (what packing computes for the kept tokens), forward and gradients; the
keep mask's token order; token_drop none unchanged; the edge cases (a layer without a kept token, a sample with one, nothing kept);
and that nothing downstream (the loss, the sampler) reads a prediction at a dropped token."""
import types

import pytest
import torch
import torch.nn.functional as F

import gs_diffusion as gd
from gs_decoder import apply_rope

L, GRID, PS, FEAT = 6, 6, 2, 64          # 6 layers, 6x6 patches of 2x2 slots: in_dim 64 + 14 * 4 = 120 -> dim 128, 2 heads
P = GRID * GRID


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def nontrivial_(trunk, seed=0):
  """LayerScale 1 and a random FiLM: no near-identity block (the comparisons would be vacuous otherwise)."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for blk in trunk.blocks:
      blk.ls1.fill_(1.0)
      blk.ls2.fill_(1.0)
    trunk.film[-1].weight.copy_(0.02 * torch.randn(trunk.film[-1].weight.shape, generator=g))
    trunk.film[-1].bias.copy_(0.1 * torch.randn(trunk.film[-1].bias.shape, generator=g))
  return trunk


def make_head(seed=0, device="cpu"):
  torch.manual_seed(seed)
  head = gd.DiffusionHead(FEAT, L, patch_size=PS).to(device)
  nontrivial_(head.trunk, seed)
  torch.nn.init.normal_(head.out.weight, std=0.02)
  return head


def hit_masks(seed=0):
  """(4, L, H, W) hit masks with the real masks' structure (the kept layers of a column a prefix): sample 0 random columns of
  1..6 layers, sample 1 layers 4-5 without any valid slot, sample 2 a single valid slot (one token), sample 3 every slot."""
  g = torch.Generator().manual_seed(seed)
  H = GRID * PS
  depth = torch.randint(0, L + 1, (4, H, H), generator=g)                       # per pixel: how many layers are hit
  depth[1] = depth[1].clamp(max=4)
  depth[2] = 0
  depth[2, 5, 7] = 1
  depth[3] = L
  hit = torch.arange(L)[None, :, None, None] < depth[:, None]
  hit &= torch.rand(hit.shape, generator=g) < 0.9                               # partially filled tokens
  hit[2, 0, 5, 7] = True
  hit[3] = True
  return hit


def inputs(head, b, seed=1, device="cpu"):
  g = torch.Generator().manual_seed(seed)
  feats = torch.randn(b, L, P, FEAT, generator=g).to(device)
  x_net = torch.randn(b, L, gd.CHANNELS, GRID * PS, GRID * PS, generator=g).to(device)
  t = torch.rand(b, L, generator=g).to(device)
  return feats, x_net, t


# ---------------------------------------------------------------------------- the reference: dense, dropped tokens masked as keys

def _masked_sdpa(q, k, v, key_ok):
  """q, k, v (N, S, H, hd), key_ok (N, S). Every query also sees its own key: a dropped query's row would be NaN with all its keys
  masked, and the NaN would leak into the kept tokens through later blocks (0 * NaN); a kept query's own key is kept anyway."""
  m = key_ok[:, None, None, :] | torch.eye(key_ok.shape[1], dtype=torch.bool, device=q.device)[None, None]
  return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=m).transpose(1, 2)


def _ref_block(block, x, cos, sin, adaln, keep):
  b, l, p, d = x.shape
  hd = d // block.num_heads
  mod = (block.modulation + adaln.float()).unbind(2)
  shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (m[:, :, None, :] for m in mod)
  h = (block.norm1(x).float() * (1 + scale_a) + shift_a).to(x.dtype)

  def qkv(hh, n, s):
    q, k, v = block.qkv(hh).reshape(n, s, 3, block.num_heads, hd).unbind(2)
    return block.q_norm(q), block.k_norm(k), v

  if block.mode == "frame":
    q, k, v = qkv(h.reshape(b * l, p, d), b * l, p)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    y = _masked_sdpa(q, k, v, keep.reshape(b * l, p)).reshape(b, l, p, d)
  elif block.mode == "ray":
    q, k, v = qkv(h.transpose(1, 2).reshape(b * p, l, d), b * p, l)
    att = torch.einsum("nshd,nthd->nhst", q, k) * hd ** -0.5
    ok = keep.transpose(1, 2).reshape(b * p, l)[:, None, None, :] | torch.eye(l, dtype=torch.bool, device=x.device)[None, None]
    att = att.masked_fill(~ok, float("-inf"))
    y = torch.einsum("nhst,nthd->nshd", att.softmax(-1).to(v.dtype), v).reshape(b, p, l, d).transpose(1, 2)
  else:
    q, k, v = qkv(h.reshape(b, l * p, d), b, l * p)
    q, k = apply_rope(q, cos.repeat(l, 1), sin.repeat(l, 1)), apply_rope(k, cos.repeat(l, 1), sin.repeat(l, 1))
    y = _masked_sdpa(q, k, v, keep.reshape(b, l * p)).reshape(b, l, p, d)
  x = x + (block.ls1 * block.proj(y) * gate_a).to(x.dtype)
  h = (block.norm2(x).float() * (1 + scale_m) + shift_m).to(x.dtype)
  return x + (block.ls2 * block.mlp(h) * gate_m).to(x.dtype)


def ref_trunk(trunk, x, adaln, keep):
  cos, sin = trunk._rope_tables(GRID, x.device)
  gamma, beta = trunk.film(trunk.layer_embed(torch.arange(L, device=x.device))).chunk(2, dim=-1)
  adaln_mean = adaln.mean(1, keepdim=True)
  for block in trunk.blocks:
    if block.mode == "frame":
      x = x * (1 + gamma[None, :, None]) + beta[None, :, None]
    x = _ref_block(block, x, cos, sin, adaln if block.mode == "frame" else adaln_mean, keep)
  return x


def tokens_of(head, feats, x_net):
  """DiffusionHead.forward's dense tokens (B, L, P, D)."""
  feats = (feats - feats.mean((1, 2, 3), keepdim=True)) / (feats.var((1, 2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
  patches = x_net.reshape(*x_net.shape[:3], GRID, PS, GRID, PS).permute(0, 1, 3, 5, 2, 4, 6).reshape(*x_net.shape[:2], P, -1)
  return F.pad(torch.cat([feats * head.img_token_scale, patches], -1), (0, head.pad))


def rel(a, b):
  return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------- tests

def test_keep_mask_follows_the_token_order():
  hit = torch.zeros(2, L, GRID * PS, GRID * PS, dtype=torch.bool)
  hit[1, 3, 7, 4] = True                                                         # slot (y 7, x 4): patch row 3, column 2
  keep = gd.token_keep(hit, PS, "empty")
  assert keep.shape == (2, L, P) and keep.sum() == 1 and keep[1, 3, 3 * GRID + 2]
  # ... the patch the tokens' rearrange puts that slot in
  x = torch.zeros(2, L, gd.CHANNELS, GRID * PS, GRID * PS)
  x[:, :, 0][hit] = 1.0
  patches = gd.rearrange(x, "b l c (h py) (w px) -> b l (h w) (c py px)", py=PS, px=PS)
  assert torch.equal(patches.abs().sum(-1) > 0, keep)
  k0 = gd.token_keep(hit, PS, "empty_except_layer0")
  assert k0[:, 0].all() and torch.equal(k0[:, 1:], keep[:, 1:])
  assert gd.token_keep(hit, PS, "none") is None
  with pytest.raises(ValueError):
    gd.token_keep(hit, PS, "all")


def test_pack_index_segments():
  keep = gd.token_keep(hit_masks(), PS, "empty")
  pi = gd.PackIndex(keep)
  assert pi.T == int(keep.sum()) and torch.equal(pi.gather(torch.arange(4 * L * P).reshape(4, L, P)), pi.flat)
  counts = keep.sum(2).flatten()
  assert pi.cu_frame.diff().tolist() == counts[counts > 0].tolist()            # zero-length (sample, layer)s left out
  assert pi.cu_global.diff().tolist() == keep.sum((1, 2)).tolist()
  assert keep[1, 4:].sum() == 0 and keep[2].sum() == 1                          # the edge cases are really there
  # every row sits in its own column's slot of the ray table, and the table has nothing else
  assert torch.equal(pi.ray_table[pi.ray_col, pi.l], torch.arange(pi.T))
  assert int(pi.ray_valid.sum()) == pi.T and pi.ray_valid.any(1).all()
  assert torch.equal(pi.scatter(pi.gather(keep.float())), keep.float())


@pytest.mark.parametrize("mode", ["empty", "empty_except_layer0"])
def test_packed_trunk_matches_key_masked_dense(mode):
  head = make_head()
  hit = hit_masks()
  keep = gd.token_keep(hit, PS, mode)
  feats, x_net, t = inputs(head, 4)
  tokens = tokens_of(head, feats, x_net)
  adaln = head.adaln(t)
  pi = gd.PackIndex(keep)
  with torch.no_grad():
    ref = ref_trunk(head.trunk, tokens, adaln, keep)
    rows = head.trunk.forward_packed(pi.gather(tokens), adaln, pi)
  assert rel(rows, ref[keep]) < 1e-6
  # the whole head: the same prediction at the kept tokens, 0 at the dropped ones
  with torch.no_grad():
    out = head(feats, x_net, t, pi)
    ref_out = gd.rearrange(head.out(ref), "b l (h w) (c py px) -> b l c (h py) (w px)", h=GRID, w=GRID, py=PS, px=PS)
  slot_keep = keep.reshape(4, L, GRID, 1, GRID, 1).expand(4, L, GRID, PS, GRID, PS).reshape(4, L, 1, GRID * PS, GRID * PS)
  slot_keep = slot_keep.expand_as(out)
  assert rel(out[slot_keep], ref_out[slot_keep]) < 1e-6
  assert (out[~slot_keep] == 0).all()
  # every valid slot is in a kept token
  assert slot_keep[:, :, 0][hit].all()


def test_packed_gradients_match_key_masked_dense():
  head = make_head()
  keep = gd.token_keep(hit_masks(), PS, "empty")
  feats, x_net, t = inputs(head, 4)
  tokens = tokens_of(head, feats, x_net)
  t_req = t.clone()
  w = torch.randn(4, L, P, head.dim, generator=torch.Generator().manual_seed(7))
  pi = gd.PackIndex(keep)

  def grads(packed):
    head.zero_grad(set_to_none=True)
    x = tokens.clone().requires_grad_(True)
    adaln = head.adaln(t_req)
    out = pi.scatter(head.trunk.forward_packed(pi.gather(x), adaln, pi)) if packed else ref_trunk(head.trunk, x, adaln, keep)
    (out * w * keep[..., None]).sum().backward()                                # a loss on the kept tokens only
    return {n: p.grad.clone() for n, p in head.named_parameters() if p.grad is not None}, x.grad.clone()

  g_ref, gx_ref = grads(False)
  g_pk, gx_pk = grads(True)
  assert set(g_ref) == set(g_pk) and "time_proj.1.weight" in g_pk               # the AdaLN gather carries the gradient back too
  for n in g_ref:
    # the ray block's k_norm.bias: its true gradient is 0 (softmax is shift invariant), both are rounding noise
    tol = dict(atol=1e-5, rtol=0) if n.endswith("k_norm.bias") else dict(atol=1e-6 * g_ref[n].abs().max().item(), rtol=1e-4)
    torch.testing.assert_close(g_pk[n], g_ref[n], **tol, msg=n)
  assert rel(gx_pk[keep], gx_ref[keep]) < 1e-5
  assert (gx_pk[~keep] == 0).all() and (gx_ref[~keep] == 0).all()


def test_all_kept_packed_equals_dense_trunk():
  head = make_head()
  feats, x_net, t = inputs(head, 2)
  pi = gd.PackIndex(torch.ones(2, L, P, dtype=torch.bool))
  with torch.no_grad():
    dense = head(feats, x_net, t)
    packed = head(feats, x_net, t, pi)
  assert rel(packed, dense) < 1e-6


def test_nothing_kept_falls_back_to_dense():
  head = make_head()
  feats, x_net, t = inputs(head, 2)
  pi = gd.PackIndex(torch.zeros(2, L, P, dtype=torch.bool))
  assert pi.T == 0
  torch.testing.assert_close(head(feats, x_net, t, pi), head(feats, x_net, t), rtol=0, atol=0)


def model_shell(head, token_drop):
  """A GSDiffusionModel around `head`, with the backbone / statistics / schedule stubbed out."""
  m = gd.GSDiffusionModel.__new__(gd.GSDiffusionModel)
  torch.nn.Module.__init__(m)
  m.head = head
  m.patch_size = PS
  m.num_layers = L
  m.token_drop = token_drop
  m.eval_seed = 0
  m.steps = 4
  m._feats = lambda feats: feats                                                # the "image" is the features already
  m.normalize = lambda x0_raw: x0_raw.permute(0, 2, 1, 3, 4)
  m.decode = lambda x, floor=None: x
  m.schedule = types.SimpleNamespace(sample=lambda b, phase, generator=None: torch.rand(b, L, generator=generator), fixed_t=None)
  return m


def test_token_drop_none_is_the_dense_head_call():
  head = make_head()
  m = model_shell(head, "none")
  hit = hit_masks()
  feats, _, _ = inputs(head, 4)
  x0_raw = torch.randn(4, gd.CHANNELS, L, GRID * PS, GRID * PS)
  calls = []
  real = head.forward
  head.forward = lambda *a, **k: calls.append((len(a), tuple(sorted(k)))) or real(*a, **k)
  m.training_loss(feats, hit, x0_raw, seeds=[1, 2, 3, 4])
  m.sample(feats, hit, seeds=[1, 2, 3, 4])
  assert m.pack(hit) is None and set(calls) == {(3, ())}                        # exactly the 3-argument call of before


@pytest.mark.parametrize("mode", ["empty", "empty_except_layer0"])
def test_loss_and_sampler_never_read_dropped_tokens(mode, monkeypatch):
  """The loss (and its gradients) and the sample at the valid slots are the same whatever the prediction at a dropped token is."""
  head = make_head()
  m = model_shell(head, mode)
  hit = hit_masks()
  feats, _, _ = inputs(head, 4)
  x0_raw = torch.randn(4, gd.CHANNELS, L, GRID * PS, GRID * PS)
  seeds = [5, 6, 7, 8]

  def run():
    head.zero_grad(set_to_none=True)
    loss, parts, _ = m.training_loss(feats, hit, x0_raw, seeds=seeds)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in head.named_parameters() if p.grad is not None}
    with torch.no_grad():
      x = m.sample(feats, hit, seeds=seeds)                                     # (B, L, C, H, W), decode stubbed out
    return loss.detach(), parts, grads, x.permute(0, 1, 3, 4, 2)[hit]

  base = run()
  scatter = gd.PackIndex.scatter

  def junk_scatter(self, rows):   # 1e3 instead of 0 at the dropped tokens
    return scatter(self, rows).masked_fill(scatter(self, torch.ones_like(rows)) == 0, 1e3)

  monkeypatch.setattr(gd.PackIndex, "scatter", junk_scatter)
  junk = run()
  assert torch.equal(base[0], junk[0])
  assert all(torch.equal(base[1][k], junk[1][k]) for k in base[1])
  assert set(base[2]) == set(junk[2])
  for n in base[2]:   # (not bitwise: the gathers' backward accumulates in a run-to-run varying order, ~1e-11)
    torch.testing.assert_close(junk[2][n], base[2][n], rtol=1e-6, atol=1e-9, msg=n)
  assert torch.equal(base[3], junk[3])


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["empty", "empty_except_layer0"])
def test_varlen_bf16_matches_key_masked_dense(mode):
  """The varlen flash path (CUDA, bf16 autocast) against the dense key-masked reference under the same autocast; calibrated by
  the reference's own bf16-vs-fp32 gap. Forward and backward run (the kernel's autograd)."""
  dev = "cuda"
  head = make_head(device=dev)
  keep = gd.token_keep(hit_masks().to(dev), PS, mode)
  feats, x_net, t = inputs(head, 4, device=dev)
  tokens = tokens_of(head, feats, x_net)
  adaln = head.adaln(t)
  pi = gd.PackIndex(keep)
  with torch.no_grad():
    ref32 = ref_trunk(head.trunk, tokens, adaln, keep)[keep]
    with torch.autocast("cuda", dtype=torch.bfloat16):
      refb = ref_trunk(head.trunk, tokens, adaln, keep)[keep].float()
      rows = head.trunk.forward_packed(pi.gather(tokens), adaln, pi).float()
  gap = rel(refb, ref32)
  assert rel(rows, refb) < max(0.5 * gap, 1e-3) and rel(rows, ref32) < 2 * gap + 1e-3
  x = tokens.clone().requires_grad_(True)
  with torch.autocast("cuda", dtype=torch.bfloat16):
    out = head.trunk.forward_packed(pi.gather(x), head.adaln(t), pi)
  out.float().square().mean().backward()
  assert torch.isfinite(x.grad).all() and x.grad[keep].abs().sum() > 0
