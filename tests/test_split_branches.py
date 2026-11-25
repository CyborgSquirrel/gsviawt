"""diffusion.split_branches: 4 branch transformers (offset + opacity, colour + opacity, scale + opacity, rotation) with joint global
attention, and the hybrids (mid shared | merged: input branch blocks -> a shared-weight / a merged-token middle -> output branch
blocks), every test for each of the three. The default (off) is the single trunk bit for bit (parameters, loss, gradients, sample); the packed head agrees with an
independent dense reference (per-branch loops, the dropped tokens masked as keys, the joint attention over the concatenated branch
sequences), forward and gradients, and with itself on every token kept; the attention mixes the branches; the merged opacity is the
mean of the three; the x0 loss is every branch on its own channels; the decode / sampler path; a state_dict round trip; bf16 varlen
on CUDA."""
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import gs_diffusion as gd
from gs_decoder import apply_rope
from tests.test_cond_dropout import model_cfg, stats
from tests.test_token_drop import FEAT, GRID, L, P, PS, _masked_sdpa, hit_masks, rel

H = GRID * PS
SPLIT = {"enabled": True, "dim": 64, "num_blocks": 4, "image_dim": 16}   # 1 head of 64, blocks 0 and 3 at the frame positions


@pytest.fixture(autouse=True)
def no_tf32():
  saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  yield
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def split_head(seed=0, device="cpu", randomize=True, **kw):
  torch.manual_seed(seed)
  head = gd.SplitBranchHead(FEAT, L, patch_size=PS, dim=SPLIT["dim"], num_blocks=SPLIT["num_blocks"], image_dim=SPLIT["image_dim"],
                            **kw).to(device)
  if randomize:   # no near-identity block, no zero readout / FiLM: the comparisons would be vacuous otherwise
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
      for n, p in head.named_parameters():
        if n.endswith(("ls1", "ls2")):
          p.fill_(1.0)
        elif n.endswith(("out_w", "out_b", "film2_w", "film2_b", "unmerge_w", "unmerge_b")):
          p.copy_(0.05 * torch.randn(p.shape, generator=g))
  return head


def inputs(b, seed=1, device="cpu"):
  g = torch.Generator().manual_seed(seed)
  feats = torch.randn(b, L, P, FEAT, generator=g).to(device)
  x_net = torch.randn(b, L, gd.CHANNELS, H, H, generator=g).to(device)
  t = torch.rand(b, L, generator=g).to(device)
  return feats, x_net, t


# ---------------------------------------------------------------- the dense reference, written per branch

def ref_joint_block(blk, xs, ads, keep, cos, sin, pidx):
  """One block on the dense tokens xs (a list of (B, L, P, D_i)), ONE masked SDPA over their sequences concatenated (list order:
  branch-major, a different order from the packed rows'); token i uses blk's parameter slice pidx(i), modulated by ads[i] ((B, L | 1,
  6, D)). keep (B, L, P): the dropped tokens masked as keys."""
  b, l, p, _ = xs[0].shape
  ln = lambda x, w, bb: F.layer_norm(x, (x.shape[-1],)) * w + bb
  key_ok = keep.reshape(b, l * p).repeat(1, len(xs))
  qs, ks, vs, mods = [], [], [], []
  for i, x in enumerate(xs):
    k, D = pidx(i), x.shape[-1]
    mod = [m[:, :, None] for m in (blk.modulation[k] + ads[i]).unbind(2)]          # (B, L | 1, 1, D) each
    mods.append(mod)
    h = ln(x, blk.norm1_w[k], blk.norm1_b[k]) * (1 + mod[1]) + mod[0]
    q, kk, v = (h.reshape(b, l * p, D) @ blk.qkv_w[k]).reshape(b, l * p, 3, blk.num_heads, -1).unbind(2)
    q, kk = ln(q, blk.q_norm_w[k], blk.q_norm_b[k]), ln(kk, blk.k_norm_w[k], blk.k_norm_b[k])
    qs.append(apply_rope(q, cos, sin))
    ks.append(apply_rope(kk, cos, sin))
    vs.append(v)
  y = _masked_sdpa(torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1), key_ok).reshape(b, len(xs), l, p, -1)
  out = []
  for i, x in enumerate(xs):
    k, mod = pidx(i), mods[i]
    x = x + blk.ls1[k] * (y[:, i] @ blk.proj_w[k] + blk.proj_b[k]) * mod[2]
    h = ln(x, blk.norm2_w[k], blk.norm2_b[k]) * (1 + mod[4]) + mod[3]
    out.append(x + blk.ls2[k] * ((F.gelu(h @ blk.fc1_w[k] + blk.fc1_b[k]) @ blk.fc2_w[k] + blk.fc2_b[k]) * mod[5]))
  return out


def ref_forward(head, feats, x_net, t, keep):
  """(merged (B, L, 14, H, W), branches (B, L, 16, H, W)) from head's parameters: per-branch dense tensors, every block's attention
  one masked SDPA (ref_joint_block); the middle (mid shared | merged) written out the same way."""
  b, l, p, _ = feats.shape
  K, D = head.k, head.dim
  feats = (feats - feats.mean((1, 2, 3), keepdim=True)) / (feats.var((1, 2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
  feats = feats * head.img_token_scale
  emb = F.silu(head.time_embed(gd.sinusoidal_embedding(t.reshape(-1) * 1000.0, head.t_embed_channels)))
  xs, ads, films = [], [], []
  for k, (_, chans) in enumerate(gd.SPLIT_BRANCHES):
    patches = rearrange(x_net[:, :, list(chans)], "b l c (h py) (w px) -> b l (h w) (c py px)", py=PS, px=PS)
    tok = torch.cat([feats @ head.in_image_w[k] + head.in_image_b[k], patches @ head.in_noise_w[k] + head.in_noise_b[k]], -1)
    xs.append(tok + head.branch_embed[k])
    ads.append((emb @ head.time_proj_w[k] + head.time_proj_b[k]).reshape(b, l, 6, D))
    e = F.gelu(head.layer_embed[k] @ head.film1_w[k] + head.film1_b[k]) @ head.film2_w[k] + head.film2_b[k]
    films.append(e.chunk(2, -1))
  cos, sin = head._rope_tables(GRID, feats.device)
  cos, sin = cos.repeat(l, 1), sin.repeat(l, 1)

  def middle(xs):
    ad = head.mid_time_proj(emb).reshape(b, l, 6, head.mid_dim)
    pick = lambda cond: ad if cond == "frame" else ad.mean(1, keepdim=True)
    if head.mid == "shared":                                                       # the same weights (slice 0) for every branch
      for blk, cond in zip(head.mid_blocks, head.mid_cond):
        xs = ref_joint_block(blk, xs, [pick(cond)] * K, keep, cos, sin, lambda i: 0)
      return xs
    m = [head.merge(torch.cat(xs, -1))]                                            # (B, L, P, mid_dim)
    for blk, cond in zip(head.mid_blocks, head.mid_cond):
      m = ref_joint_block(blk, m, [pick(cond)], keep, cos, sin, lambda i: 0)
    return [xs[k] + m[0] @ head.unmerge_w[k] + head.unmerge_b[k] for k in range(K)]

  for j, (blk, cond) in enumerate(zip(head.blocks, head.cond)):
    if j == head.n_in and head.mid != "none":
      xs = middle(xs)
    if cond == "frame":
      xs = [xs[k] * (1 + films[k][0][None, :, None]) + films[k][1][None, :, None] for k in range(K)]
    a = [ads[k] if cond == "frame" else ads[k].mean(1, keepdim=True) for k in range(K)]
    xs = ref_joint_block(blk, xs, a, keep, cos, sin, lambda i: i)
  if head.mid != "none" and head.n_in == len(head.blocks):
    xs = middle(xs)
  outs = [rearrange(xs[k] @ head.out_w[k] + head.out_b[k], "b l (h w) (c py px) -> b l c (h py) (w px)", h=GRID, w=GRID, py=PS, px=PS)
          for k in range(K)]
  br = torch.cat(outs, 2)
  merged = torch.zeros(b, l, gd.CHANNELS, H, H)
  for j, c in enumerate(gd.SPLIT_INDEX):
    if c != 0:
      merged[:, :, c] = br[:, :, j]
  merged[:, :, 0] = br[:, :, [j for j, c in enumerate(gd.SPLIT_INDEX) if c == 0]].mean(2)
  return merged, br


# the hybrids at test size: 2 input blocks -> 3 middle blocks -> 1 output block (positions 0 1 | 2 3 4 | 5: frame at 0 and 3)
MIDS = {"none": {}, "shared": dict(mid="shared", in_blocks=2, mid_blocks=3, out_blocks=1),
        "merged": dict(mid="merged", in_blocks=2, mid_blocks=3, mid_dim=128, out_blocks=1)}


def slot_keep(keep):
  """(B, L, P) token keep -> (B, L, 1, H, W) slot mask."""
  return rearrange(keep, "b l (h w) -> b l () h w", h=GRID)[..., None, :, None].expand(-1, -1, -1, -1, PS, -1, PS).reshape(
    *keep.shape[:2], 1, H, H)


# ---------------------------------------------------------------- config / default


def test_normalize_split_branches():
  assert gd.normalize_split_branches(None) == {"enabled": False, "dim": 1024, "num_blocks": 9, "image_dim": 256, "mid": "none",
                                               "in_blocks": 4, "mid_blocks": 12, "mid_dim": None, "out_blocks": 2}
  assert gd.normalize_split_branches({"mid": "shared"})["mid_dim"] == 1024
  for bad in ({"enabled": True, "dim": 100}, {"image_dim": 1024}, {"depth": 3}, {"num_blocks": 0}, {"mid": "both"},
              {"mid": "shared", "mid_dim": 2048}, {"mid": "merged"}, {"mid": "merged", "mid_dim": 100}, {"mid": "shared", "mid_blocks": 0}):
    with pytest.raises(ValueError):
      gd.normalize_split_branches(bad)
  assert gd.SPLIT_INDEX == (11, 12, 13, 0, 8, 9, 10, 0, 1, 2, 3, 0, 4, 5, 6, 7)   # offset, colour, scale + opacity; rotation


def test_default_is_the_single_trunk_bit_for_bit():
  torch.manual_seed(0)
  old = gd.GSDiffusionModel(model_cfg(), stats())                                  # a config without the key
  torch.manual_seed(0)
  off = gd.GSDiffusionModel(model_cfg(split_branches={"enabled": False, "dim": 64, "num_blocks": 2, "image_dim": 16}), stats())
  assert isinstance(off.head, gd.DiffusionHead) and not off.split_branches
  sd_old, sd_off = old.state_dict(), off.state_dict()
  assert list(sd_old) == list(sd_off) and all(torch.equal(sd_old[k], sd_off[k]) for k in sd_old)
  torch.nn.init.normal_(old.head.out.weight, std=0.02)
  off.head.out.weight.data.copy_(old.head.out.weight.data)
  feats, _, _ = inputs(2)
  hit = hit_masks()[:2]
  x0_raw = torch.randn(2, gd.CHANNELS, L, H, H)
  for drop in ("none", "empty"):
    res = []
    for m in (old, off):
      m._feats = lambda f: f
      m.token_drop = drop
      m.zero_grad(set_to_none=True)
      torch.manual_seed(5)
      loss, parts, counts = m.training_loss(feats, hit, x0_raw)
      loss.backward()
      res.append((loss.detach(), parts, [p.grad.clone() for p in m.head.parameters()], m.sample(feats, hit, seeds=[1, 2], steps=2)))
    (l0, p0, g0, s0), (l1, p1, g1, s1) = res
    assert torch.equal(l0, l1) and set(p0) == set(p1) and "merged" not in p1 and all(torch.equal(p0[k], p1[k]) for k in p0)
    assert all(torch.equal(s0[k], s1[k]) for k in s0)
    # (the packed backward's index gathers accumulate in a nondeterministic order on the CPU: equal up to rounding there)
    same = torch.equal if drop == "none" else lambda a, b: rel(a, b) < 1e-6
    assert all(same(a, b) for a, b in zip(g0, g1))


def test_config_builds_the_split_head_and_rejects_other_options():
  m = gd.GSDiffusionModel(model_cfg(split_branches=SPLIT, attention="global"), stats())
  assert isinstance(m.head, gd.SplitBranchHead) and m.split_branches and len(m.head.blocks) == 4
  for extra in ({}, {"attention": "global", "film": "every"}, {"attention": "global", "mask_attn": {"mode": "qk"}},
                {"attention": "global", "colour_noise": {"mode": "separate"}}, {"attention": "global", "readout": "slice"}):
    with pytest.raises(SystemExit, match="split_branches"):
      gd.GSDiffusionModel(model_cfg(split_branches=SPLIT, **extra), stats())


def test_mid_none_is_the_plain_split_head():
  """mid none (with the hybrid keys given) builds the plain split head: the same parameters, the same init, the same output."""
  torch.manual_seed(3)
  plain = gd.SplitBranchHead(FEAT, L, patch_size=PS, dim=64, num_blocks=4, image_dim=16)
  torch.manual_seed(3)
  none = gd.SplitBranchHead(FEAT, L, patch_size=PS, dim=64, num_blocks=4, image_dim=16, mid="none", in_blocks=1, mid_blocks=5,
                            mid_dim=128, out_blocks=7)
  a, b = plain.state_dict(), none.state_dict()
  assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a) and none.n_in == 4 and not none.mid_cond
  m = gd.GSDiffusionModel(model_cfg(split_branches={**SPLIT, "mid": "none", "in_blocks": 1, "out_blocks": 1}, attention="global"),
                          stats())
  assert len(m.head.blocks) == 4 and m.head.mid == "none" and not hasattr(m.head, "mid_blocks")


def test_hybrid_layouts():
  a = split_head(**MIDS["shared"])
  assert len(a.blocks) == 3 and len(a.mid_blocks) == 3 and a.n_in == 2 and a.mid_blocks[0].k == 1
  assert a.cond == ["frame", "ray", "global"] and a.mid_cond == ["global", "frame", "ray"]      # positions 0 1 | 2 3 4 | 5
  b = split_head(**MIDS["merged"])
  assert b.mid_blocks[0].num_heads == 2 and b.merge.in_features == 4 * 64 and b.unmerge_w.shape == (4, 128, 64)
  m = gd.GSDiffusionModel(model_cfg(split_branches={**SPLIT, **MIDS["merged"]}, attention="global"), stats())
  assert m.head.mid == "merged" and m.head.mid_dim == 128


def test_full_size_parameter_count():
  with torch.device("meta"):
    head = gd.SplitBranchHead(1024, 6, 14)                                          # the defaults: 1024 x 9, image 256
  n = sum(p.numel() for p in head.parameters())
  per_block_branch = sum(p.numel() for p in head.blocks[0].parameters()) / 4
  assert per_block_branch == 12_601_600 and n == 499_460_160 and len(head.blocks) == 9 and head.num_heads == 16
  count = lambda **kw: sum(p.numel() for p in gd.SplitBranchHead(1024, 6, 14, **kw).parameters())
  with torch.device("meta"):
    assert count(mid="shared", in_blocks=4, mid_blocks=12, out_blocks=2) == 505_757_760                     # variant A
    assert count(mid="merged", in_blocks=4, mid_blocks=3, mid_dim=2048, out_blocks=2) == 528_725_824        # variant B


# ---------------------------------------------------------------- the forward


@pytest.mark.parametrize("which", ["partial", "all"])
@pytest.mark.parametrize("mid", list(MIDS))
def test_packed_matches_the_dense_reference(which, mid):
  head = split_head(**MIDS[mid])
  feats, x_net, t = inputs(4)
  hit = hit_masks()
  keep = gd.token_keep(hit, PS, "empty") if which == "partial" else torch.ones(4, L, P, dtype=torch.bool)
  ref_m, ref_b = ref_forward(head, feats, x_net, t, keep)
  m, br = head(feats, x_net, t, gd.PackIndex(keep), branches=True)
  s = slot_keep(keep)
  assert rel(br * s, ref_b * s) < 1e-5 and rel(m * s, ref_m * s) < 1e-5
  if which == "all":   # pack None: every token, the same code path
    assert torch.equal(head(feats, x_net, t), m)


@pytest.mark.parametrize("mid", list(MIDS))
def test_packed_gradients_match_the_dense_reference(mid):
  head = split_head(**MIDS[mid])
  feats, x_net, t = inputs(4)
  keep = gd.token_keep(hit_masks(), PS, "empty")
  s = slot_keep(keep)
  w = torch.randn(4, L, 16, H, H)
  grads = []
  for fn in (lambda x: ref_forward(head, feats, x, t, keep)[1], lambda x: head(feats, x, t, gd.PackIndex(keep), branches=True)[1]):
    head.zero_grad()
    x = x_net.clone().requires_grad_(True)
    (fn(x) * s * w).sum().backward()
    grads.append((x.grad * s, {n: p.grad.clone() for n, p in head.named_parameters()}))
  (gx0, gp0), (gx1, gp1) = grads
  assert rel(gx1, gx0) < 1e-5
  for n in gp0:
    assert rel(gp1[n], gp0[n]) < 1e-4, n


@pytest.mark.parametrize("mid", list(MIDS))
def test_joint_attention_mixes_the_branches(mid):
  """Perturbing only the rotation channels of the input moves the colour branch's output (and, with every token alone in its
  own sample, nothing else could carry it: the branches' only contact is the attention)."""
  head = split_head(**MIDS[mid])
  feats, x_net, t = inputs(2)
  pack = gd.PackIndex(gd.token_keep(hit_masks()[:2], PS, "empty"))
  x2 = x_net.clone()
  x2[:, :, gd.X0_SLICES["rotation"]] += torch.randn_like(x2[:, :, gd.X0_SLICES["rotation"]])
  with torch.no_grad():
    a, b = head(feats, x_net, t, pack), head(feats, x2, t, pack)
  s = slot_keep(gd.token_keep(hit_masks()[:2], PS, "empty"))
  for f in ("sh_dc", "offset", "scale", "opacity"):
    assert ((a[:, :, gd.X0_SLICES[f]] - b[:, :, gd.X0_SLICES[f]]) * s).abs().max() > 1e-4, f
  # the control: without the attention's cross-branch terms (every block's attention output zeroed: ls1 = 0; merged: the middle's
  # way back too) the colour is untouched
  with torch.no_grad():
    for blk in list(head.blocks) + list(getattr(head, "mid_blocks", [])):
      blk.ls1.zero_()
    if mid == "merged":
      head.unmerge_w.zero_()
      head.unmerge_b.zero_()
    a, b = head(feats, x_net, t, pack), head(feats, x2, t, pack)
  assert torch.equal(a[:, :, gd.X0_SLICES["sh_dc"]], b[:, :, gd.X0_SLICES["sh_dc"]])


@pytest.mark.parametrize("mid", list(MIDS))
def test_merged_opacity_is_the_mean_and_channels_in_place(mid):
  head = split_head(**MIDS[mid])
  feats, x_net, t = inputs(2)
  with torch.no_grad():
    m, br = head(feats, x_net, t, branches=True)
  assert m.shape == (2, L, gd.CHANNELS, H, H) and br.shape == (2, L, 16, H, H)
  opac = [j for j, c in enumerate(gd.SPLIT_INDEX) if c == 0]
  assert len(opac) == 3 and torch.allclose(m[:, :, 0], br[:, :, opac].mean(2))
  assert not torch.allclose(br[:, :, opac[0]], br[:, :, opac[1]])                  # three different predictions
  for j, c in enumerate(gd.SPLIT_INDEX):
    if c:
      assert torch.equal(m[:, :, c], br[:, :, j])


@pytest.mark.parametrize("mid", list(MIDS))
def test_fresh_head_predicts_zero(mid):
  head = split_head(randomize=False, **MIDS[mid])                                               # the zero-init readout
  feats, x_net, t = inputs(2)
  with torch.no_grad():
    assert (head(feats, x_net, t) == 0).all()


# ---------------------------------------------------------------- the model: loss, decode, sampler, checkpoint


def split_model(seed=0, mid="none"):
  torch.manual_seed(seed)
  m = gd.GSDiffusionModel(model_cfg(split_branches={**SPLIT, **MIDS[mid]}, attention="global", token_drop="empty"), stats())
  m._feats = lambda f: f
  with torch.no_grad():
    m.head.load_state_dict(split_head(seed, **MIDS[mid]).state_dict())
  return m


@pytest.mark.parametrize("mid", list(MIDS))
def test_x0_loss_is_every_branch_on_its_own_channels(mid):
  m = split_model(mid=mid)
  feats, _, _ = inputs(2)
  hit = hit_masks()[:2]
  x0_raw = torch.randn(2, gd.CHANNELS, L, H, H)
  loss, parts, counts, x0_hat, t = m.training_loss(feats, hit, x0_raw, seeds=[3, 4], return_pred=True)
  # the same call's head output, recomputed: the loss from the branch outputs by hand
  _, eps = m.eval_noise([3, 4], (2, L, gd.CHANNELS, H, H), "cpu")
  x0 = m.normalize(x0_raw)
  a = hit[:, :, None].float()
  x_net = a * ((1 - t[..., None, None, None]) * x0 + t[..., None, None, None] * eps) + (1 - a) * eps
  merged, br = m.head(feats, x_net, t, m.pack(hit), branches=True)
  err = (br - x0[:, :, list(gd.SPLIT_INDEX)]).square()
  assert torch.allclose(loss, (err * a).sum() / (a.sum() * 16))
  assert torch.allclose(x0_hat, merged)                                            # the rendering loss sees the merged opacity
  assert torch.allclose(parts["merged"], ((merged - x0).square() * a).sum() / (a.sum() * gd.CHANNELS))
  assert torch.allclose(parts["opacity"], ((merged[:, :, :1] - x0[:, :, :1]).square() * a).sum() / a.sum())
  for j, name in zip((3, 7, 11), ("offset", "colour", "scale")):
    assert torch.allclose(parts[f"opacity_{name}"], (err[:, :, j:j + 1] * a).sum() / a.sum())
  assert all(int(counts[k]) for k in parts)
  loss.backward()
  for n, p in m.head.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), n
  # each opacity prediction is supervised: the readout rows of all three branches' opacity get a gradient
  ps2 = PS * PS
  for k in range(3):
    assert m.head.out_w.grad[k, :, 3 * ps2:].abs().sum() > 0


@pytest.mark.parametrize("mid", list(MIDS))
def test_decode_and_sampler(mid):
  m = split_model(mid=mid)
  feats, _, _ = inputs(2)
  hit = hit_masks()[:2]
  g = m.sample(feats, hit, seeds=[1, 2], steps=3)
  assert g["opacity"].shape == (2, L, 1, H, H) and all(torch.isfinite(v).all() for v in g.values())
  x0_hat = m.training_loss(feats, hit, torch.randn(2, gd.CHANNELS, L, H, H), seeds=[1, 2], return_pred=True)[3]
  d = m.decode(x0_hat)
  assert torch.allclose(d["opacity"], torch.sigmoid(x0_hat[:, :, :1]).clamp(torch.sigmoid(torch.tensor(-1.0)),
                                                                             torch.sigmoid(torch.tensor(1.0))))
  probe = m.xt_probe(feats, hit, torch.randn(2, gd.CHANNELS, L, H, H), [1, 2], [1.0, 0.5])
  assert torch.isfinite(probe[0.5]["err_model"]).all()


@pytest.mark.parametrize("mid", list(MIDS))
def test_state_dict_round_trip(tmp_path, mid):
  m = split_model(seed=1, mid=mid)
  path = tmp_path / "m.pt"
  torch.save({"state_dict": m.state_dict(), "ema": gd.EMAWeights(m.head).state_dict()}, path)
  torch.manual_seed(7)
  fresh = gd.GSDiffusionModel(model_cfg(split_branches={**SPLIT, **MIDS[mid]}, attention="global", token_drop="empty"), stats())
  fresh._feats = lambda f: f
  ck = torch.load(path)
  fresh.load_state_dict(ck["state_dict"], strict=True)
  ema = gd.EMAWeights(fresh.head)
  ema.load_state_dict(ck["ema"])
  feats, _, _ = inputs(2)
  hit = hit_masks()[:2]
  a, b = m.sample(feats, hit, seeds=[1, 2], steps=2), fresh.sample(feats, hit, seeds=[1, 2], steps=2)
  assert all(torch.equal(a[k], b[k]) for k in a)
  assert not any("branch_index" in k for k in ck["state_dict"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (varlen flash)")
@pytest.mark.parametrize("mid", list(MIDS))
def test_varlen_bf16_matches_fp32(mid):
  head = split_head(device="cuda", **MIDS[mid])
  head.checkpoint_activations = True
  feats, x_net, t = inputs(4, device="cuda")
  keep = gd.token_keep(hit_masks().cuda(), PS, "empty")
  pack = gd.PackIndex(keep)
  with torch.no_grad():
    ref = head(feats, x_net, t, pack)
  x = x_net.clone().requires_grad_(True)
  with torch.autocast("cuda", dtype=torch.bfloat16):
    out = head(feats, x, t, pack)
  out.float().sum().backward()
  s = slot_keep(keep.cpu()).cuda()
  assert rel(out.float() * s, ref * s) < 0.03 and torch.isfinite(x.grad).all()
