"""diffusion.mask_attn: each token's hit bits -> a shared MLP -> per block a zero-initialized Linear whose output is added to the
attention's q, k (qk) or q, k, v (qkv), after the QK-norm and before RoPE. The default (none) bit-identical to the code before the
option (parameters, dense forward + gradients, packed forward, the head's call); a fresh qk / qkv model torch.equal to none (and the
other parameters' init the same); trained, the output depends on the mask (q / k and v each); packed agrees with a key-masked dense
reference, with the mask; the model passes hit through the training loss, the guided and plain samplers and the probe."""
import pytest
import torch

import gs_diffusion as gd
from gs_decoder import apply_rope
from tests.test_token_drop import FEAT, GRID, L, P, PS, _masked_sdpa, hit_masks, inputs, model_shell, rel

H = GRID * PS


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def randomize_(head, seed=3, skip=("mask_",)):
  """Random values in every parameter (no near-identity block, no zero output projection), except those whose name contains one of
  `skip` (the mask parameters stay at their init unless asked)."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for n, p in head.named_parameters():
      if not any(s in n for s in skip):
        p.copy_(0.05 * torch.randn(p.shape, generator=g))
    for blk in head.trunk.blocks:
      blk.ls1.fill_(1.0)
      blk.ls2.fill_(1.0)
  return head


def heads(mode, attention="global", seed=0):
  """(none, mode) heads with the same (randomized) shared parameters; the mode head's mask parameters at their zero init."""
  torch.manual_seed(seed)
  base = randomize_(gd.DiffusionHead(FEAT, L, patch_size=PS, attention=attention)).eval()
  torch.manual_seed(seed)
  masked = gd.DiffusionHead(FEAT, L, patch_size=PS, attention=attention, mask_attn={"mode": mode, "dim": 32, "hidden": 48}).eval()
  missing, unexpected = masked.load_state_dict(base.state_dict(), strict=False)
  assert not unexpected and missing and all("mask_" in k for k in missing)
  return base, masked


def perturb_mask_(head, seed=11, scale=0.3, rows=None):
  """Random mask parameters; rows: only these thirds of every mask_proj's output (0 q, 1 k, 2 v), the others zero."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for n, p in head.named_parameters():
      if "mask_" in n:
        p.copy_(scale * torch.randn(p.shape, generator=g))
    if rows is not None:
      for blk in head.trunk.blocks:
        d = blk.qkv.in_features
        keep = torch.zeros(blk.mask_proj.out_features, dtype=torch.bool)
        for r in rows:
          keep[r * d:(r + 1) * d] = True
        blk.mask_proj.weight[~keep] = 0
        blk.mask_proj.bias[~keep] = 0
  return head


# ---------------------------------------------------------------- config and parameters


def test_normalize_mask_attn():
  assert gd.normalize_mask_attn(None) == {"mode": "none", "dim": 256, "hidden": 256}
  assert gd.normalize_mask_attn({"mode": "qkv"}) == {"mode": "qkv", "dim": 256, "hidden": 256}
  for bad in ({"mode": "q"}, {"mode": "qk", "size": 3}, {"mode": "qk", "dim": 0}):
    with pytest.raises(ValueError):
      gd.normalize_mask_attn(bad)


def test_none_has_no_new_parameters():
  plain = gd.DiffusionHead(FEAT, L, patch_size=PS)
  none = gd.DiffusionHead(FEAT, L, patch_size=PS, mask_attn={"mode": "none"})
  assert set(plain.state_dict()) == set(none.state_dict()) and none.mask_attn == "none"
  assert not any("mask_" in k for k in none.state_dict())


@pytest.mark.parametrize("mode,n", [("qk", 2), ("qkv", 3)])
def test_parameters_and_zero_init(mode, n):
  head = gd.DiffusionHead(FEAT, L, patch_size=PS, num_blocks=4, mask_attn={"mode": mode, "dim": 32, "hidden": 48})
  plain = gd.DiffusionHead(FEAT, L, patch_size=PS, num_blocks=4)
  new = {k: v for k, v in head.state_dict().items() if k not in plain.state_dict()}
  assert set(plain.state_dict()) <= set(head.state_dict())
  assert set(new) == ({"mask_embed.0.weight", "mask_embed.0.bias", "mask_embed.2.weight", "mask_embed.2.bias"}
                      | {f"trunk.blocks.{k}.mask_proj.{w}" for k in range(4) for w in ("weight", "bias")})
  assert head.mask_embed[0].in_features == PS ** 2 and head.mask_embed[2].out_features == 32
  for blk in head.trunk.blocks:
    assert blk.mask_proj.weight.shape == (n * head.dim, 32)
    assert (blk.mask_proj.weight == 0).all() and (blk.mask_proj.bias == 0).all()
  d, h, m, blocks = 2048, 256, 256, 9   # the runs' size (reported in conf/train_gs.yaml)
  count = (196 * h + h) + (h * m + m) + blocks * n * (m * d + d)
  assert count == {"qk": 9_590_272, "qkv": 14_327_296}[mode]


@pytest.mark.parametrize("mode", ["qk", "qkv"])
def test_init_rng_unchanged(mode):
  """The mask modules are built on a forked RNG: the shared parameters' init and the random stream after it are the same."""
  torch.manual_seed(7)
  a = gd.DiffusionHead(FEAT, L, patch_size=PS)
  after_a = torch.rand(5)
  torch.manual_seed(7)
  b = gd.DiffusionHead(FEAT, L, patch_size=PS, mask_attn={"mode": mode})
  after_b = torch.rand(5)
  sb = b.state_dict()
  assert all(torch.equal(v, sb[k]) for k, v in a.state_dict().items())
  assert torch.equal(after_a, after_b)


# ---------------------------------------------------------------- the default bit-identical to the code before the option


def _old_attend_normed(blk, h, cos, sin):
  n, s, d = h.shape
  q, k, v = blk.qkv(h).reshape(n, s, 3, blk.num_heads, d // blk.num_heads).unbind(2)
  q, k = blk.q_norm(q), blk.k_norm(k)
  if cos is not None:
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
  out = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
  return blk.proj(out.transpose(1, 2).reshape(n, s, d))


def _old_attend_short(blk, h):
  n, s, d = h.shape
  hd = d // blk.num_heads
  q, k, v = blk.qkv(h).reshape(n, s, 3, blk.num_heads, hd).unbind(2)
  q, k = blk.q_norm(q), blk.k_norm(k)
  att = torch.einsum("nshd,nthd->nhst", q, k) * hd ** -0.5
  out = torch.einsum("nhst,nthd->nshd", att.softmax(-1).to(v.dtype), v)
  return blk.proj(out.reshape(n, s, d))


def _old_block(blk, x, cos, sin, adaln):
  b, l, p, d = x.shape
  mod = (blk.modulation + adaln.float()).unbind(2)
  shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (m[:, :, None, :] for m in mod)
  h = (blk.norm1(x).float() * (1 + scale_a) + shift_a).to(x.dtype)
  if blk.mode == "frame":
    y = _old_attend_normed(blk, h.reshape(b * l, p, d), cos, sin).reshape(b, l, p, d)
  elif blk.mode == "ray":
    y = _old_attend_short(blk, h.transpose(1, 2).reshape(b * p, l, d)).reshape(b, p, l, d).transpose(1, 2)
  else:
    y = _old_attend_normed(blk, h.reshape(b, l * p, d), cos.repeat(l, 1), sin.repeat(l, 1)).reshape(b, l, p, d)
  x = x + (blk.ls1 * y * gate_a).to(x.dtype)
  h = (blk.norm2(x).float() * (1 + scale_m) + shift_m).to(x.dtype)
  return x + (blk.ls2 * blk.mlp(h) * gate_m).to(x.dtype)


def _old_block_packed(blk, x, pack, cos_r, sin_r, adaln, seg):
  mod = blk.modulation + adaln.float()
  row = lambda i: mod[:, i][seg]
  h = (blk.norm1(x).float() * (1 + row(1)) + row(0)).to(x.dtype)
  t, d = h.shape
  hd = d // blk.num_heads
  q, k, v = blk.qkv(h).reshape(t, 3, blk.num_heads, hd).unbind(1)
  q, k = blk.q_norm(q), blk.k_norm(k)
  if blk.mode == "ray":
    idx = pack.ray_table.clamp_min(0)
    att = torch.einsum("cshd,cthd->chst", q[idx], k[idx]) * hd ** -0.5
    att = att.masked_fill(~pack.ray_valid[:, None, None, :], float("-inf"))
    out = torch.einsum("chst,cthd->cshd", att.softmax(-1).to(v.dtype), v[idx])[pack.ray_col, pack.l]
  else:
    q, k = apply_rope(q, cos_r, sin_r), apply_rope(k, cos_r, sin_r)
    cu, mx = (pack.cu_frame, pack.max_frame) if blk.mode == "frame" else (pack.cu_global, pack.max_global)
    out = gd.segment_attention(q, k, v, cu, mx)
  x = x + (blk.ls1 * blk.proj(out.reshape(t, d)) * row(2)).to(x.dtype)
  h = (blk.norm2(x).float() * (1 + row(4)) + row(3)).to(x.dtype)
  return x + (blk.ls2 * blk.mlp(h) * row(5)).to(x.dtype)


@pytest.mark.parametrize("attention", ["global", "frame_ray_global"])
def test_default_blocks_bit_identical_to_before(attention):
  """mask_attn none: every block's dense forward and gradients and its packed forward torch.equal to the pre-option block code."""
  torch.manual_seed(0)
  head = randomize_(gd.DiffusionHead(FEAT, L, patch_size=PS, attention=attention))
  tr = head.trunk
  g = torch.Generator().manual_seed(4)
  x = torch.randn(2, L, P, head.dim, generator=g)
  t = torch.rand(2, L, generator=g)
  keep = torch.rand(2, L, P, generator=g) < 0.5
  keep[:, 0, 0] = True
  pack = gd.PackIndex(keep)
  cos, sin = tr._rope_tables(GRID, x.device)
  adaln = head.adaln(t).detach()
  params = list(head.parameters())
  for blk, cond in zip(tr.blocks, tr.cond):
    a = adaln if cond == "frame" else adaln.mean(1, keepdim=True)
    with torch.no_grad():
      assert torch.equal(blk(x, cos, sin, a), _old_block(blk, x, cos, sin, a))
      ap, seg = (adaln.reshape(2 * L, 6, -1), pack.frame) if cond == "frame" else (adaln.mean(1), pack.b)
      xp = pack.gather(x)
      assert torch.equal(blk.forward_packed(xp, pack, cos[pack.p], sin[pack.p], ap, seg),
                         _old_block_packed(blk, xp, pack, cos[pack.p], sin[pack.p], ap, seg))
    new, old = (torch.autograd.grad(f().square().sum(), params, allow_unused=True)
                for f in (lambda: blk(x, cos, sin, a), lambda: _old_block(blk, x, cos, sin, a)))
    assert all((u is None and w is None) or torch.equal(u, w) for u, w in zip(new, old))


def test_default_model_calls_head_as_before():
  """mask_attn none: the model never gives the head hit (its call exactly as before; head stubs without the argument work)."""
  torch.manual_seed(0)
  head = randomize_(gd.DiffusionHead(FEAT, L, patch_size=PS))
  for token_drop, n_args in (("none", 3), ("empty", 4)):
    m = model_shell(head, token_drop)
    calls = []
    real = head.forward
    head.forward = lambda *a, **k: calls.append((len(a), tuple(sorted(k)))) or real(*a, **k)
    feats, _, _ = inputs(head, 4)
    hit = hit_masks()
    m.training_loss(feats, hit, torch.randn(4, gd.CHANNELS, L, H, H), seeds=[1, 2, 3, 4])
    m.sample(feats, hit, seeds=[1, 2, 3, 4])
    del head.forward
    assert set(calls) == {(n_args, ())}


# ---------------------------------------------------------------- zero init = baseline; trained: the mask matters


@pytest.mark.parametrize("attention", ["global", "frame_ray_global"])
@pytest.mark.parametrize("mode", ["qk", "qkv"])
def test_zero_init_equals_none(mode, attention):
  base, masked = heads(mode, attention)
  feats, x_net, t = inputs(base, 4)
  hit = hit_masks()
  pack = gd.PackIndex(gd.token_keep(hit, PS, "empty"))
  with torch.no_grad():
    assert torch.equal(masked(feats, x_net, t, hit=hit), base(feats, x_net, t))
    assert torch.equal(masked(feats, x_net, t, pack, hit=hit), base(feats, x_net, t, pack))
    uncond = torch.tensor([True, False, True, False])
    assert torch.equal(masked(feats, x_net, t, pack, uncond=uncond, hit=hit), base(feats, x_net, t, pack, uncond=uncond))
  with pytest.raises(ValueError):
    masked(feats, x_net, t)                       # on: hit is required
  # the gradients of the shared parameters too (the zero mask_proj passes nothing back into the trunk)
  grads = {}
  for name, h, kw in (("base", base, {}), ("masked", masked, {"hit": hit})):
    h.zero_grad(set_to_none=True)
    h(feats, x_net, t, **kw).square().sum().backward()
    grads[name] = {n: p.grad for n, p in h.named_parameters() if p.grad is not None}
  assert all(torch.equal(grads["masked"][n], g) for n, g in grads["base"].items())
  assert any(grads["masked"][n].abs().sum() > 0 for n in grads["masked"] if "mask_proj" in n)   # it can learn


@pytest.mark.parametrize("attention", ["global", "frame_ray_global"])
@pytest.mark.parametrize("mode", ["qk", "qkv"])
def test_mask_changes_the_output(mode, attention):
  base, masked = heads(mode, attention)
  perturb_mask_(masked)
  feats, x_net, t = inputs(base, 4)
  hit = hit_masks()
  other = hit.clone()
  other[0] = ~other[0]           # sample 0 another mask; the others the same
  with torch.no_grad():
    a, b = masked(feats, x_net, t, hit=hit), masked(feats, x_net, t, hit=other)
    # (the randomized trunk's v vectors are much alike, so re-weighting the attention moves the output only a little: q / k alone
    # change it by ~1e-5..1e-4 relative, still 100x above the fp32 rounding of the equal-path comparisons, which are 0)
    assert rel(a, base(feats, x_net, t)) > 1e-5
    assert rel(a[0], b[0]) > 1e-5
    assert torch.equal(a[1:], b[1:])             # (the dense trunk: no attention across samples)


@pytest.mark.parametrize("rows", [(0,), (1,), (2,)])
def test_each_projection_is_used(rows):
  """qkv: the q, the k and the v term each change the output on their own."""
  _, masked = heads("qkv")
  feats, x_net, t = inputs(masked, 2)
  hit = hit_masks()[:2]
  with torch.no_grad():
    ref = masked(feats, x_net, t, hit=hit)
    perturb_mask_(masked, rows=rows)
    assert rel(masked(feats, x_net, t, hit=hit), ref) > 5e-6


def test_mask_embedding_order():
  """The embedding's input is the token's patch_size^2 hit bits in the (py px) order of the noisy values (its token's patch)."""
  _, masked = heads("qk")
  seen = []
  masked.mask_embed.register_forward_hook(lambda mod, inp, out: seen.append(inp[0]))
  hit = torch.zeros(1, L, H, H, dtype=torch.bool)
  hit[0, 2, 7, 4] = True                         # slot (y 7, x 4): patch (3, 2), position (py 1, px 0) in it
  feats, x_net, t = inputs(masked, 1)
  with torch.no_grad():
    masked(feats, x_net, t, hit=hit)
  e = seen[0]
  assert e.shape == (1, L, P, PS * PS) and e.sum() == 1 and e[0, 2, 3 * GRID + 2, 1 * PS + 0] == 1


# ---------------------------------------------------------------- packed = key-masked dense, with the mask


def _ref_block_mask(blk, x, cos, sin, adaln, keep, mask):
  """test_token_drop's key-masked dense block, with the mask terms added after the QK-norm, before RoPE."""
  b, l, p, d = x.shape
  hd = d // blk.num_heads
  mod = (blk.modulation + adaln.float()).unbind(2)
  shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (m[:, :, None, :] for m in mod)
  h = (blk.norm1(x).float() * (1 + scale_a) + shift_a).to(x.dtype)
  add = blk.mask_proj(mask)                                                     # (B, L, P, n * D)

  def qkv(hh, aa, n, s):
    q, k, v = blk.qkv(hh).reshape(n, s, 3, blk.num_heads, hd).unbind(2)
    aa = aa.reshape(n, s, -1, blk.num_heads, hd)
    q, k = blk.q_norm(q) + aa[:, :, 0], blk.k_norm(k) + aa[:, :, 1]
    return q, k, (v + aa[:, :, 2] if aa.shape[2] == 3 else v)

  if blk.mode == "frame":
    q, k, v = qkv(h.reshape(b * l, p, d), add.reshape(b * l, p, -1), b * l, p)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    y = _masked_sdpa(q, k, v, keep.reshape(b * l, p)).reshape(b, l, p, d)
  elif blk.mode == "ray":
    q, k, v = qkv(h.transpose(1, 2).reshape(b * p, l, d), add.transpose(1, 2).reshape(b * p, l, -1), b * p, l)
    att = torch.einsum("nshd,nthd->nhst", q, k) * hd ** -0.5
    ok = keep.transpose(1, 2).reshape(b * p, l)[:, None, None, :] | torch.eye(l, dtype=torch.bool)[None, None]
    att = att.masked_fill(~ok, float("-inf"))
    y = torch.einsum("nhst,nthd->nshd", att.softmax(-1).to(v.dtype), v).reshape(b, p, l, d).transpose(1, 2)
  else:
    q, k, v = qkv(h.reshape(b, l * p, d), add.reshape(b, l * p, -1), b, l * p)
    q, k = apply_rope(q, cos.repeat(l, 1), sin.repeat(l, 1)), apply_rope(k, cos.repeat(l, 1), sin.repeat(l, 1))
    y = _masked_sdpa(q, k, v, keep.reshape(b, l * p)).reshape(b, l, p, d)
  x = x + (blk.ls1 * blk.proj(y) * gate_a).to(x.dtype)
  h = (blk.norm2(x).float() * (1 + scale_m) + shift_m).to(x.dtype)
  return x + (blk.ls2 * blk.mlp(h) * gate_m).to(x.dtype)


@pytest.mark.parametrize("attention", ["global", "frame_ray_global"])
@pytest.mark.parametrize("mode", ["qk", "qkv"])
def test_packed_matches_key_masked_dense(mode, attention):
  _, masked = heads(mode, attention)
  perturb_mask_(masked)
  tr = masked.trunk
  hit = hit_masks()
  keep = gd.token_keep(hit, PS, "empty")
  pi = gd.PackIndex(keep)
  g = torch.Generator().manual_seed(9)
  x = torch.randn(4, L, P, masked.dim, generator=g)
  t = torch.rand(4, L, generator=g)
  bits = gd.rearrange(hit.float(), "b l (h py) (w px) -> b l (h w) (py px)", py=PS, px=PS)
  with torch.no_grad():
    mask = masked.mask_embed(bits)
    adaln = masked.adaln(t)
    rows = tr.forward_packed(pi.gather(x), adaln, pi, pi.gather(mask))
    cos, sin = tr._rope_tables(GRID, x.device)
    films = tr._films(L, x.device, lambda v: v[None, :, None])
    ref, am = x, adaln.mean(1, keepdim=True)
    for blk, cond, gb in zip(tr.blocks, tr.cond, films):
      if gb is not None:
        ref = ref * (1 + gb[0]) + gb[1]
      ref = _ref_block_mask(blk, ref, cos, sin, adaln if cond == "frame" else am, keep, mask)
    assert rel(rows, ref[keep]) < 1e-6
    # every token kept: the packed head equals the dense head
    feats, x_net, tt = inputs(masked, 4)
    full = gd.PackIndex(torch.ones(4, L, P, dtype=torch.bool))
    assert rel(masked(feats, x_net, tt, full, hit=hit), masked(feats, x_net, tt, hit=hit)) < 1e-6


# ---------------------------------------------------------------- the model passes hit everywhere


@pytest.mark.parametrize("token_drop", ["none", "empty"])
def test_model_passes_hit(token_drop):
  """training_loss, sample (plain and guided), sample_staged, xt_probe and predict_x0 all give the head the
  batch's hit mask (the same at every step)."""
  _, masked = heads("qk")
  perturb_mask_(masked)
  m = model_shell(masked, token_drop)
  m.mask_attn = "qk"
  m.guidance = gd.normalize_guidance(None)
  m.cond_dropout_p = 0.0
  hit = hit_masks()
  feats, _, _ = inputs(masked, 4)
  x0_raw = torch.randn(4, gd.CHANNELS, L, H, H)
  seen = []
  real = masked.forward
  masked.forward = lambda *a, **k: seen.append(k.get("hit")) or real(*a, **k)
  loss, _, _ = m.training_loss(feats, hit, x0_raw, seeds=[1, 2, 3, 4])
  assert torch.isfinite(loss)
  m.sample(feats, hit, seeds=[1, 2, 3, 4], steps=2)
  m.sample(feats, hit, seeds=[1, 2, 3, 4], steps=2, guidance={"scale": 2.0})
  m.sample_staged(feats, hit, [[0], [1, 2, 3, 4, 5]], seeds=[1, 2, 3, 4], steps=2)
  m.xt_probe(feats, hit, x0_raw, seeds=[1, 2, 3, 4], ts=[0.5])
  m.predict_x0(feats, hit, torch.randn(4, L, gd.CHANNELS, H, H), torch.rand(4, L), condition=False)
  del masked.forward
  assert len(seen) == 1 + 2 + 4 + 4 + 4 * 3 + 1 and all(s is not None for s in seen)
  assert all(s.shape == (1, *hit.shape[1:]) or torch.equal(s, hit) for s in seen)
