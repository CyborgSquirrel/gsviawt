#!/usr/bin/env python3
"""The re-anchoring task on one mesh: view A's layered 3DGS grid + view B's depth-peel point cloud -> B's grid, one forward pass,
photometric loss only (reanchor.py has the model, the frames and the token layout). Hydra: conf/train_reanchor.yaml.

Training: per step `pairs_per_step` ordered pairs (A, B) of the mesh's training views (A != B, uniform; data.identity: A = B), one
forward each, the predicted B grid rendered at `views_per_pair` random training views (0 .. train_views-1) over a fresh random
solid colour per view (render and premultiplied GT composited over it), L1 + 0.85 D-SSIM. bf16 autocast on the network, fp32
decode / render / loss; AdamW, linear warmup, cosine decay over the last decay_fraction of max_seconds (training time).

Donor-A control (data.a_source other_mesh): every training sample's A is a fresh random view (0 .. train_views-1) of a random
OTHER mesh of the render h5 (uniform over them; reanchor.py's donor section for the frame), loaded on the fly by DataLoader
workers; the eval pairs get fixed seeded donors (eval_donors.json) and ablate-A zeroes the donor's params at its slots.

Eval (every eval_every steps and at the end): a fixed set of eval_pairs (A, B) pairs (eval_seed), the predicted B grid rendered
at the held-out views (eval_views, never fitted) over black and over white, full-image PSNR, mean over views then pairs; the
main model also with A's params zeroed (the mean Gaussian at A's slots: does the output follow A?). The references on the same
pairs (computed once, cached in <output_dir>/references.json): A's fit as is, B's own fit, the matched copy (k = 32,
bla/_review_nn_baseline.py's floor). Everything goes to W&B (project gsviawt) and <output_dir>/eval.jsonl.

  docker exec -w /app/.claude/worktrees/reanchor gsviawt-app-gpu-1 python train_reanchor.py model.dim=256 model.num_blocks=3 ...
"""

import json
import logging
import math
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import reanchor as ra
from module import DSSIMLoss

log = logging.getLogger("train_reanchor")


def eval_pair_list(n_views, n, seed, identity=False):
  rng = np.random.default_rng(seed)
  out = []
  while len(out) < n:
    a, b = (int(x) for x in rng.choice(n_views, 2, replace=False))
    out.append((b, b) if identity else (a, b))
  return out


def sample_pairs(rng, n_views, n, identity):
  out = []
  for _ in range(n):
    a, b = (int(x) for x in rng.choice(n_views, 2, replace=False))
    out.append((b, b) if identity else (a, b))
  return out


def render_eval(data, p, b, views, frame="b"):
  """Mean PSNR over black / white at `views` of render params p in B's frame (frame b) or the world (frame world)."""
  vm = data.viewmats(b, views) if frame == "b" else torch.linalg.inv(data.c2w[list(views)])
  rgb, alpha = ra.render(p, vm, data.K[list(views)], data.W, data.H)
  black, white = ra.psnr_bw(rgb, alpha, data.gt[list(views)], data.alpha[list(views)])
  return float(black.mean()), float(white.mean())


@torch.no_grad()
def references(data, pairs, views, path):
  """{name: [black, white] per pair} for A as is, B's fit, the matched copy (k 32); cached in `path`."""
  if os.path.exists(path):
    with open(path) as f:
      ref = json.load(f)
    if ref.get("pairs") == [list(p) for p in pairs]:
      return ref
  ref = {"pairs": [list(p) for p in pairs], "A_asis": [], "B_fit": [], "matched32": [], "matched32_info": []}
  for a, b in pairs:
    ref["A_asis"].append(render_eval(data, ra.world_gaussians(data.world[a]), b, views, "world"))
    ref["B_fit"].append(render_eval(data, ra.world_gaussians(data.world[b]), b, views, "world"))
    rows, info = ra.matched_copy_world(data, a, b, 32)
    ref["matched32"].append(render_eval(data, ra.world_gaussians({k: torch.as_tensor(v, device=data.device) for k, v in rows.items()}),
                                        b, views, "world"))
    ref["matched32_info"].append(info)
    log.info("references (%d, %d): A %s B %s matched %s", a, b, ref["A_asis"][-1], ref["B_fit"][-1], ref["matched32"][-1])
  with open(path, "w") as f:
    json.dump(ref, f, indent=1)
  return ref


def ref_means(ref):
  return {k: np.mean(ref[k], 0).tolist() for k in ("A_asis", "B_fit", "matched32")}


def photometric(dssim, cfg, rgb, alpha, gt, gt_alpha, bg):
  """L1 + D-SSIM (loss weights) of a render (V, 3, H, W) over per-view backgrounds bg (V, 3, 1, 1) against the GT over them."""
  pred = (rgb + (1 - alpha) * bg).clamp(0, 1)
  target = gt + (1 - gt_alpha) * bg
  l1 = (pred - target).abs().mean()
  ds = dssim(pred, target).mean()
  with torch.no_grad():
    psnr = -10 * torch.log10((pred - target).pow(2).flatten(1).mean(1)).mean()
  return cfg.loss.l1_weight * l1 + cfg.loss.dssim_weight * ds, l1, ds, psnr


def backgrounds(cfg, n, device):
  if cfg.loss.background == "random":
    return torch.rand(n, 3, 1, 1, device=device)
  if cfg.loss.background == "black":
    return torch.zeros(n, 3, 1, 1, device=device)
  raise SystemExit(f"loss.background must be random | black, got {cfg.loss.background!r}")


def setup(cfg, device):
  data = ra.MeshData(cfg.data.render_h5, cfg.data.grids_dir, int(cfg.data.mesh), range(int(cfg.data.train_views)), device=device)
  nv = int(cfg.data.train_views)
  rng = np.random.default_rng(12345)
  mp = [tuple(int(x) for x in rng.choice(nv, 2, replace=False)) for _ in range(int(cfg.data.moment_pairs))]
  if cfg.data.identity:
    mp = [(b, b) for _, b in mp]
  in_mean, in_std = data.paired_moments(mp)
  out = data.own_moments()
  log.info("output moments mean %s std %s; A-input (paired) mean %s std %s", out[0].cpu().numpy().round(3),
           out[1].cpu().numpy().round(3), in_mean.cpu().numpy().round(3), in_std.cpu().numpy().round(3))
  batcher = ra.Batcher(data, in_mean, in_std, out, z_ref=float(cfg.model.z_ref), slot_scale_px=float(cfg.model.slot_scale_px),
                       decode_clamp=str(cfg.model.decode_clamp))
  return data, batcher


def init_wandb(cfg):
  import wandb
  return wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode, name=cfg.wandb.name, id=cfg.wandb.id, resume="allow" if cfg.wandb.id else None,
                    tags=list(cfg.wandb.tags), dir=cfg.output_dir, config=OmegaConf.to_container(cfg, resolve=True))


# ---------------------------------------------------------------------------
# mode roundtrip
# ---------------------------------------------------------------------------

@torch.no_grad()
def roundtrip(cfg, data, batcher):
  views = list(cfg.data.eval_views)
  pairs = eval_pair_list(int(cfg.data.train_views), int(cfg.train.eval_pairs), int(cfg.train.eval_seed))
  res = {"fit": [], "path": [], "frame_psnr_between": []}
  for a, b in pairs[:10]:
    fit = render_eval(data, ra.world_gaussians(data.world[b]), b, views, "world")
    rows = data.x0[b]
    xn = ra.dense_rows((rows - batcher.out_mean) / batcher.out_std, data.hit[b]).permute(0, 3, 1, 2)[None]
    p = batcher.gaussians(batcher.decode(xn), 0, b)
    path = render_eval(data, p, b, views, "b")
    # A moved into B's frame (rotated rows, moved anchors) rendered from B-relative cameras vs A rendered in the world
    ra_rows, ra_xyz = data.a_rows_in_b(a, b)
    r1, a1 = ra.render(ra.raw_gaussians(ra_rows, ra_xyz), data.viewmats(b, views), data.K[views], data.W, data.H)
    r2, a2 = ra.render(ra.world_gaussians(data.world[a]), torch.linalg.inv(data.c2w[views]), data.K[views], data.W, data.H)
    between = float((-10 * torch.log10((r1.clamp(0, 1) - r2.clamp(0, 1)).pow(2).flatten(1).mean(1))).mean())
    res["fit"].append(fit); res["path"].append(path); res["frame_psnr_between"].append(between)
    log.info("roundtrip B=%d: fit %.4f / %.4f  path %.4f / %.4f  (diff %+.4f / %+.4f dB); A=%d in B's frame vs world: %.1f dB",
             b, *fit, *path, path[0] - fit[0], path[1] - fit[1], a, between)
  d = np.array(res["path"]) - np.array(res["fit"])
  res["max_abs_diff_db"] = float(np.abs(d).max())
  log.info("roundtrip: max |path - fit| %.4f dB, min A frame agreement %.1f dB", res["max_abs_diff_db"], min(res["frame_psnr_between"]))
  with open(os.path.join(cfg.output_dir, "roundtrip.json"), "w") as f:
    json.dump(res, f, indent=1)


# ---------------------------------------------------------------------------
# mode direct
# ---------------------------------------------------------------------------

def direct(cfg, data, batcher, device):
  views = list(cfg.data.eval_views)
  pairs = eval_pair_list(int(cfg.data.train_views), int(cfg.train.eval_pairs), int(cfg.train.eval_seed))
  a, b = (tuple(cfg.direct.pair) if cfg.direct.pair is not None else pairs[0])
  ref = references(data, [(a, b)], views, os.path.join(cfg.output_dir, "references_direct.json"))
  rows, info = ra.matched_copy_world(data, a, b, 32)
  x0 = ra.world_rows_to_b(data, rows, b)
  xn = (x0 - batcher.out_mean) / batcher.out_std
  chans = ra.X0_SLICES
  params = {k: torch.nn.Parameter(xn[:, chans[k]].clone()) for k in ("opacity", "scale", "rotation", "sh_dc", "offset")}
  groups = [{"params": [params[k]], "lr": float(cfg.direct.lr[k]) / float(batcher.out_std[chans[k]].mean())} for k in params]
  opt = torch.optim.Adam(groups, eps=1e-15)
  base = [g["lr"] for g in groups]
  dssim = DSSIMLoss(reduction="none").to(device)
  rng = np.random.default_rng(int(cfg.train.seed))
  nv = int(cfg.data.train_views)
  run = init_wandb(cfg)
  hist = []

  def current():
    full = torch.cat([params[k] for k in ("opacity", "scale", "rotation", "sh_dc", "offset")], 1)
    xd = ra.dense_rows(full, data.hit[b]).permute(0, 3, 1, 2)[None]
    return batcher.gaussians(batcher.decode(xd), 0, b)

  steps = int(cfg.direct.steps)
  for step in range(steps + 1):
    if step % int(cfg.direct.eval_every) == 0 or step == steps:
      with torch.no_grad():
        e = render_eval(data, current(), b, views, "b")
      hist.append({"step": step, "black": e[0], "white": e[1]})
      run.log({"eval/black": e[0], "eval/white": e[1], "ref/B_fit_black": ref["B_fit"][0][0], "ref/B_fit_white": ref["B_fit"][0][1],
               "ref/matched32_black": ref["matched32"][0][0], "ref/A_asis_black": ref["A_asis"][0][0]}, step=step)
      log.info("direct step %d: held-out %.3f / %.3f (B fit %.3f / %.3f, matched %.3f / %.3f, A %.3f / %.3f)", step, *e,
               *ref["B_fit"][0], *ref["matched32"][0], *ref["A_asis"][0])
    if step == steps:
      break
    f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
    for g, lr in zip(opt.param_groups, base):
      g["lr"] = lr * f
    vs = rng.choice(nv, int(cfg.train.views_per_pair), replace=False).tolist()
    rgb, alpha = ra.render(current(), data.viewmats(b, vs), data.K[vs], data.W, data.H)
    loss, l1, ds, psnr = photometric(dssim, cfg, rgb, alpha, data.gt[vs], data.alpha[vs], backgrounds(cfg, len(vs), device))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    run.log({"train/loss": float(loss), "train/psnr": float(psnr)}, step=step)
  with open(os.path.join(cfg.output_dir, "direct.json"), "w") as fh:
    json.dump({"pair": [a, b], "matched_info": info, "references": ref, "history": hist}, fh, indent=1)
  run.finish()


# ---------------------------------------------------------------------------
# mode train
# ---------------------------------------------------------------------------

def forward(model, batcher, pairs, use_a, bf16, a_override=None):
  ca, pa, cb, pb, packs = batcher.inputs(pairs, model.empty, use_a, a_override)
  with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
    rows = model(ca, pa, cb, pb, packs)
  return batcher.decode(model.readout_grid(rows, packs)), packs


@torch.no_grad()
def evaluate(model, batcher, data, pairs, views, use_a, bf16, ablate_a=False, donors=None):
  """donors (a_source other_mesh): per pair the fixed donor A (an a_override triple in B's frame) used in place of the real A."""
  model.eval()
  out = []
  for j, (a, b) in enumerate(pairs):
    override = {0: donors[j]} if donors is not None else None
    if ablate_a:
      rows, xyz, hit = donors[j] if donors is not None else (*data.a_rows_in_b(a, b), data.hit[a])
      override = {0: (batcher.in_mean.expand_as(rows), xyz, hit)}   # normalized 0: the mean Gaussian at A's (the donor's) slots
    gauss, _ = forward(model, batcher, [(a, b)], use_a, bf16, override)
    out.append(render_eval(data, batcher.gaussians(gauss, 0, b), b, views, "b"))
  model.train()
  return np.array(out)


def diagnostics(gauss, batcher, data, pairs):
  """Per-step diagnostics of the prediction (valid B slots): fraction of scales at the clamp, median |offset| in B pixel footprints,
  mean opacity, fraction of near-empty (opacity < 0.004)."""
  return bundle_diagnostics(gauss, batcher, [data.bundle(a, b, use_a=False) for a, b in pairs])


def bundle_diagnostics(gauss, batcher, bundles):
  lo, hi = float(batcher.lo[ra.SCALE].min()), float(batcher.hi[ra.SCALE].max())
  d = {"clamp": [], "off": [], "op": [], "empty": []}
  for i, bd in enumerate(bundles):
    hit = bd["b_hit"]
    raw = gauss["raw_scale"][i].permute(0, 2, 3, 1)[hit]
    d["clamp"].append(float(((raw <= lo + 1e-6) | (raw >= hi - 1e-6)).float().mean()))
    off = gauss["offset"][i].permute(0, 2, 3, 1)[hit].norm(dim=1)
    foot = bd["b_xyz"][:, 2] / bd["fx"]
    d["off"].append(float((off / foot).median()))
    op = gauss["opacity"][i].permute(0, 2, 3, 1)[hit]
    d["op"].append(float(op.mean()))
    d["empty"].append(float((op < 0.004).float().mean()))
  return {f"diag/{k}": float(np.mean(v)) for k, v in d.items()}


def n_cycles(cfg):
  """The number of LR cycles: 1 without train.cycle_seconds, else max_seconds / cycle_seconds (must divide it)."""
  cyc = cfg.train.get("cycle_seconds")
  if cyc is None:
    return 1
  n = float(cfg.train.max_seconds) / float(cyc)
  if float(cyc) <= 0 or abs(n - round(n)) > 1e-9:
    raise SystemExit(f"train.cycle_seconds={cyc} must divide train.max_seconds={cfg.train.max_seconds}")
  return int(round(n))


def cycle_of(cfg, seconds):
  """The LR cycle the training clock is in (0-based; the last cycle's index once the clock is past max_seconds)."""
  cyc = cfg.train.get("cycle_seconds")
  return 0 if cyc is None else min(int(seconds // float(cyc)), n_cycles(cfg) - 1)


def lr_factor(cfg, step, seconds):
  """LR / train.lr at a step and training clock. Without train.cycle_seconds: a linear warmup over warmup_steps steps, flat, a
  cosine to min_lr over the last decay_fraction of max_seconds. With cycle_seconds: max_seconds cut into cycles of that length
  (warmup-stable-decay each, decay_fraction of each cycle): the first cycle exactly as above over its own length; every later
  one warms up linearly from min_lr over cycle_warmup_seconds of the clock, then flat, then the cosine to min_lr."""
  w = min(1.0, (step + 1) / max(1, int(cfg.train.warmup_steps)))
  m = float(cfg.train.min_lr) / float(cfg.train.lr)
  df = float(cfg.train.decay_fraction)
  cyc = cfg.train.get("cycle_seconds")
  if cyc is None:
    frac = seconds / float(cfg.train.max_seconds)
    if frac > 1 - df:
      w *= m + (1 - m) * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - (1 - df)) / df)))
    return w
  k = cycle_of(cfg, seconds)
  frac = (seconds - k * float(cyc)) / float(cyc)
  d = 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - (1 - df)) / df))) if frac > 1 - df else 1.0
  if k == 0:
    return w * (m + (1 - m) * d)
  wt = min(1.0, (seconds - k * float(cyc)) / max(1e-9, float(cfg.train.cycle_warmup_seconds)))
  return m + (1 - m) * wt * d


def donor_setup(cfg, nv, start_step, n_eval):
  """a_source other_mesh: (an endless iterator of random donor views (DataLoader workers), the DonorViews, the eval pairs' fixed
  donors [(mesh, view)]). Donors: data.donor_grids_dir (null: data.grids_dir), every mesh of the render h5 but data.mesh."""
  gdir = cfg.data.donor_grids_dir or cfg.data.grids_dir
  pool = ra.donor_pool(cfg.data.render_h5, gdir, int(cfg.data.mesh), nv)
  ds = ra.DonorViews(cfg.data.render_h5, gdir, pool, nv, seed=[int(cfg.train.seed), int(start_step)])
  nw = int(cfg.data.donor_workers)
  loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw, pin_memory=True, prefetch_factor=8 if nw else None,
                                       persistent_workers=nw > 0)
  ev = ra.eval_donors(pool, nv, n_eval, int(cfg.train.eval_seed))
  log.info("donor A: %d donor meshes (%s .. %s) from %s, %d workers; eval donors %s", len(pool), pool[0], pool[-1], gdir, nw, ev)
  return iter(loader), ds, ev


def train(cfg, data, batcher, device):
  torch.manual_seed(int(cfg.train.seed))
  use_a = bool(cfg.model.use_a)
  a_source = str(cfg.data.a_source)
  if a_source not in ("same", "other_mesh"):
    raise SystemExit(f"data.a_source must be same | other_mesh, got {a_source!r}")
  if a_source == "other_mesh" and (not use_a or cfg.data.identity):
    raise SystemExit("data.a_source other_mesh needs model.use_a true and data.identity false")
  model = ra.ReanchorNet(dim=int(cfg.model.dim), num_blocks=int(cfg.model.num_blocks), num_heads=cfg.model.num_heads, use_a=use_a,
                         layer_scale_init=float(cfg.model.layer_scale_init),
                         checkpoint_activations=bool(cfg.model.checkpoint_activations)).to(device)
  n_params = sum(p.numel() for p in model.parameters())
  log.info("ReanchorNet: %.1f M parameters (use_a %s)", n_params / 1e6, use_a)
  opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
  dssim = DSSIMLoss(reduction="none").to(device)
  views = list(cfg.data.eval_views)
  nv = int(cfg.data.train_views)
  pairs_eval = eval_pair_list(nv, int(cfg.train.eval_pairs), int(cfg.train.eval_seed), bool(cfg.data.identity))
  ref = references(data, pairs_eval, views, os.path.join(cfg.output_dir, "references.json"))
  refm = ref_means(ref)
  log.info("references (mean over %d pairs, black / white): %s", len(pairs_eval), refm)
  run = init_wandb(cfg)
  run.summary["n_params"] = n_params
  start_step, clock = 0, 0.0
  ckpt_path = os.path.join(cfg.output_dir, "last.pt")
  if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    start_step, clock = st["step"], st["clock"]
    log.info("resumed %s at step %d (%.0f s)", ckpt_path, start_step, clock)
  rng = np.random.default_rng([int(cfg.train.seed), start_step])
  donor_iter = eval_a = None
  if a_source == "other_mesh":
    donor_iter, donor_ds, ev = donor_setup(cfg, nv, start_step, len(pairs_eval))
    eval_a = [ra.donor_in_b(donor_ds.load(m, v), data.c2w[b], device) for (m, v), (_, b) in zip(ev, pairs_eval)]
    with open(os.path.join(cfg.output_dir, "eval_donors.json"), "w") as fh:
      json.dump({"pairs": [list(p) for p in pairs_eval], "donors": [list(d) for d in ev]}, fh)
  eval_file = open(os.path.join(cfg.output_dir, "eval.jsonl"), "a")
  bf16 = bool(cfg.train.bf16)
  max_steps = cfg.train.max_steps
  last_ckpt = time.time()

  def do_eval(step):
    e = evaluate(model, batcher, data, pairs_eval, views, use_a, bf16, donors=eval_a)
    rec = {"step": step, "clock": clock, "black": float(e[:, 0].mean()), "white": float(e[:, 1].mean()),
           "black_se": float(e[:, 0].std(ddof=1) / np.sqrt(len(e))), "white_se": float(e[:, 1].std(ddof=1) / np.sqrt(len(e))),
           "per_pair": e.tolist()}
    logd = {"eval/black": rec["black"], "eval/white": rec["white"]}
    if use_a:
      z = evaluate(model, batcher, data, pairs_eval, views, use_a, bf16, ablate_a=True, donors=eval_a)
      rec["ablate_a_black"], rec["ablate_a_white"] = float(z[:, 0].mean()), float(z[:, 1].mean())
      logd.update({"eval/ablate_a_black": rec["ablate_a_black"], "eval/ablate_a_white": rec["ablate_a_white"]})
    for k, (bl, wh) in refm.items():
      logd[f"ref/{k}_black"], logd[f"ref/{k}_white"] = bl, wh
    run.log(logd, step=step)
    eval_file.write(json.dumps(rec) + "\n"); eval_file.flush()
    log.info("eval step %d: %.3f / %.3f dB (refs %s)%s", step, rec["black"], rec["white"], refm,
             f"  ablate-A {rec['ablate_a_black']:.3f} / {rec['ablate_a_white']:.3f}" if use_a else "")

  def save(step):
    tmp = ckpt_path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "clock": clock,
                "cfg": OmegaConf.to_container(cfg, resolve=True)}, tmp)
    os.replace(tmp, ckpt_path)

  step = start_step
  if step == 0:
    do_eval(0)
  torch.cuda.reset_peak_memory_stats()
  while clock < float(cfg.train.max_seconds) and (max_steps is None or step < int(max_steps)):
    torch.cuda.synchronize()
    t0 = time.time()
    lrf = lr_factor(cfg, step, clock)
    for g in opt.param_groups:
      g["lr"] = float(cfg.train.lr) * lrf
    pairs = sample_pairs(rng, nv, int(cfg.train.pairs_per_step), bool(cfg.data.identity))
    override = None
    if donor_iter is not None:   # a fresh random donor A per sample (the pair's a is drawn but unused: the same B / view stream as same)
      override = {i: ra.donor_in_b(next(donor_iter), data.c2w[b], device) for i, (_, b) in enumerate(pairs)}
    gauss, packs = forward(model, batcher, pairs, use_a, bf16, override)
    loss = torch.zeros((), device=device)
    stats = np.zeros(3)
    for i, (a, b) in enumerate(pairs):
      vs = rng.choice(nv, int(cfg.train.views_per_pair), replace=False).tolist()
      rgb, alpha = ra.render(batcher.gaussians(gauss, i, b), data.viewmats(b, vs), data.K[vs], data.W, data.H)
      li, l1, ds, ps = photometric(dssim, cfg, rgb, alpha, data.gt[vs], data.alpha[vs], backgrounds(cfg, len(vs), device))
      loss = loss + li / len(pairs)
      stats += [float(l1), float(ds), float(ps)]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    clock += dt
    step += 1
    if step % int(cfg.train.log_every) == 0:
      stats /= len(pairs)
      d = {"train/loss": float(loss), "train/l1": stats[0], "train/dssim": stats[1], "train/psnr": stats[2], "train/lr": opt.param_groups[0]["lr"],
           "train/grad_norm": float(gn), "train/step_seconds": dt, "train/clock": clock, "train/tokens_b": packs.b.T,
           "train/tokens_a": packs.ta, "train/max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
      if step % 50 == 0 or step < 5:
        with torch.no_grad():
          d.update(diagnostics({k: v.detach() for k, v in gauss.items()}, batcher, data, pairs))
      run.log(d, step=step)
    if step % 50 == 0 or step < 5:
      log.info("step %d loss %.4f psnr %.2f lr %.2e gn %.3f %.3f s/step mem %.1f GB tokens A %d B %d", step, float(loss), stats[2],
               opt.param_groups[0]["lr"], float(gn), dt, torch.cuda.max_memory_allocated() / 2 ** 30, packs.ta, packs.b.T)
    if step % int(cfg.train.eval_every) == 0:
      do_eval(step)
    if time.time() - last_ckpt > float(cfg.train.checkpoint_every_seconds):
      save(step)
      last_ckpt = time.time()
  if step % int(cfg.train.eval_every) != 0:
    do_eval(step)
  save(step)
  run.summary["final_step"] = step
  run.finish()
  log.info("done: %d steps, %.0f s training", step, clock)


# ---------------------------------------------------------------------------
# data.dataset full: the training meshes of a render h5, streamed (conf/train_reanchor_full.yaml)
# ---------------------------------------------------------------------------

T_START = time.time()


def full_meshes(cfg):
  """(train_meshes, val_meshes) of data.render_h5 by train_gs's mesh split; checks no eval set's other h5 shares a mesh path."""
  sp = cfg.data.mesh_split
  train_m, val_m = ra.mesh_split(cfg.data.render_h5, float(sp.val_fraction), int(sp.seed))
  if cfg.data.get("train_meshes") is not None:   # a subset of the split's training meshes (local tests)
    sub = sorted(int(m) for m in cfg.data.train_meshes)
    if set(sub) - set(train_m):
      raise SystemExit(f"data.train_meshes {sorted(set(sub) - set(train_m))} not in the split's training meshes")
    train_m = sub
  for name, s in cfg.eval.sets.items():
    if os.path.realpath(s.render_h5) != os.path.realpath(cfg.data.render_h5):
      check_unseen(cfg.data.render_h5, s.render_h5, name)
  return train_m, val_m


def check_unseen(train_h5, other_h5, name):
  """SystemExit if the h5 `other_h5` (an eval set of meshes never trained on) shares a mesh path with the training h5."""
  shared = set(ra.mesh_paths(train_h5)) & set(ra.mesh_paths(other_h5))
  if shared:
    raise SystemExit(f"eval set {name}: {len(shared)} mesh paths also in the training h5 (first {sorted(shared)[:3]})")


def full_moments_cached(cfg, train_m, path):
  """The training meshes' normalization (reanchor.full_moments), computed once and cached at `path` (a resumed run reuses it)."""
  if os.path.exists(path):
    mo = torch.load(path, weights_only=False)
    if mo["train_meshes"] != list(train_m):
      raise SystemExit(f"{path}: cached for other training meshes")
    return mo
  mc = cfg.data.moments
  t0 = time.time()
  mo = ra.full_moments(cfg.data.render_h5, cfg.data.grids_dir, train_m, int(cfg.data.train_views), int(mc.own_views_per_mesh),
                       int(mc.pairs_per_mesh), int(mc.seed), int(cfg.data.workers))
  mo["train_meshes"] = list(train_m)
  tok = np.concatenate([np.asarray(v) for v in mo["tokens"].values()])
  log.info("moments over %d meshes (%d own rows, %d paired rows) in %.0f s; tokens per view: mean %.0f median %.0f p95 %.0f max %.0f "
           "(largest mesh %d)", len(train_m), mo["n_rows"], mo["n_pair_rows"], time.time() - t0, tok.mean(), np.median(tok),
           np.percentile(tok, 95), tok.max(), max(mo["tokens"], key=lambda m: max(mo["tokens"][m])))
  torch.save(mo, path + ".tmp")
  os.replace(path + ".tmp", path)
  return mo


def eval_set_plan(cfg, name, s, train_m, val_m):
  """Eval set `name` -> [(mesh, [(a, b)])]: fixed seeded meshes and pairs (a != b of 0 .. train_views-1), the same every eval."""
  nv = int(cfg.data.train_views)
  if s.meshes == "train":
    pool = list(train_m)
  elif s.meshes == "val":
    pool = list(val_m)
  elif s.meshes == "all":
    pool = sorted(ra.mesh_rows(s.render_h5))
  elif not isinstance(s.meshes, str):
    pool = [int(m) for m in s.meshes]                     # an explicit list (local tests)
    if os.path.realpath(s.render_h5) == os.path.realpath(cfg.data.render_h5) and set(pool) & set(train_m) and name != "train_heldout":
      log.warning("eval set %s: explicit meshes include training meshes", name)
  else:
    raise SystemExit(f"eval set {name}: meshes must be train | val | all | a list, got {s.meshes!r}")
  rng = np.random.default_rng([int(cfg.eval.seed), *name.encode()])
  if s.n_meshes is not None and int(s.n_meshes) < len(pool):
    pool = sorted(int(m) for m in rng.choice(pool, int(s.n_meshes), replace=False))
  return [(int(m), [tuple(int(x) for x in rng.choice(nv, 2, replace=False)) for _ in range(int(s.pairs_per_mesh))]) for m in pool]


def cpu_bundle(bd):
  return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in bd.items()}


def eval_sets_setup(cfg, train_m, val_m, device):
  """{set: {"bundles": [CPU pair bundles at eval views], "ref": {ref: [[black, white]] per pair}, "pairs": [(mesh, a, b)]}}.
  Each mesh is loaded once (MeshData of its pairs' views), its references computed (cached per set and mesh in output_dir:
  references_<set>_mesh<M>.json) and its pairs cut down to CPU bundles, then freed."""
  views = list(cfg.data.eval_views)
  out = {}
  t0 = time.time()
  for name, s in cfg.eval.sets.items():
    plan = eval_set_plan(cfg, name, s, train_m, val_m)
    st = {"bundles": [], "ref": {k: [] for k in ("A_asis", "B_fit", "matched32")}, "pairs": []}
    for m, pairs in plan:
      gv = sorted({v for p in pairs for v in p})
      data = ra.MeshData(s.render_h5, s.grids_dir, m, gv, device=device)
      ref = references(data, pairs, views, os.path.join(cfg.output_dir, f"references_{name}_mesh{m}.json"))
      for j, (a, b) in enumerate(pairs):
        st["bundles"].append(cpu_bundle(data.bundle(a, b, True, views)))
        st["pairs"].append((m, a, b))
        for k in st["ref"]:
          st["ref"][k].append(ref[k][j])
      del data
      torch.cuda.empty_cache()
    out[name] = st
    log.info("eval set %s: %d meshes, %d pairs (%s); refs %s", name, len(plan), len(st["pairs"]), s.render_h5,
             {k: np.round(np.mean(v, 0), 3).tolist() for k, v in st["ref"].items()})
  log.info("eval sets ready in %.0f s", time.time() - t0)
  return out


def to_dev(bd, device):
  return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in bd.items()}


@torch.no_grad()
def evaluate_bundles(model, batcher, bundles, use_a, bf16, device, ablate_a=False):
  """Per pair (black, white) PSNR of the predicted B grid at each bundle's render targets."""
  model.eval()
  out = []
  for bd in bundles:
    bd = to_dev(bd, device)
    if ablate_a:
      bd["a_rows"] = batcher.in_mean.expand_as(bd["a_rows"])
    ca, pa, cb, pb, packs = batcher.tokens([bd], model.empty, use_a)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
      rows = model(ca, pa, cb, pb, packs)
    gauss = batcher.decode(model.readout_grid(rows, packs))
    H, W = bd["gt"].shape[-2:]
    rgb, alpha = ra.render(ra.slot_gaussians(gauss, 0, bd["b_xyz"], bd["b_hit"]), bd["viewmats"], bd["K"], W, H)
    black, white = ra.psnr_bw(rgb, alpha, bd["gt"], bd["alpha"])
    out.append((float(black.mean()), float(white.mean())))
  model.train()
  return np.array(out)


def train_full(cfg, device):
  torch.manual_seed(int(cfg.train.seed))
  use_a = bool(cfg.model.use_a)
  if cfg.data.identity or str(cfg.data.a_source) != "same":
    raise SystemExit("data.dataset full: data.identity / data.a_source other_mesh are one-mesh options")
  nc = n_cycles(cfg)
  train_m, val_m = full_meshes(cfg)
  log.info("mesh split (val_fraction %s, seed %s): %d train meshes, val meshes %s", cfg.data.mesh_split.val_fraction,
           cfg.data.mesh_split.seed, len(train_m), val_m)
  mo = full_moments_cached(cfg, train_m, os.path.join(cfg.output_dir, "moments.pt"))
  dv = lambda x: x.to(device)
  batcher = ra.Batcher(None, dv(mo["in_mean"]), dv(mo["in_std"]), tuple(dv(mo[k]) for k in ("out_mean", "out_std", "lo", "hi")),
                       z_ref=float(cfg.model.z_ref), slot_scale_px=float(cfg.model.slot_scale_px), decode_clamp=str(cfg.model.decode_clamp))
  log.info("output moments mean %s std %s; A-input mean %s std %s", mo["out_mean"].numpy().round(3), mo["out_std"].numpy().round(3),
           mo["in_mean"].numpy().round(3), mo["in_std"].numpy().round(3))
  sets = eval_sets_setup(cfg, train_m, val_m, device)
  with open(os.path.join(cfg.output_dir, "eval_pairs.json"), "w") as fh:
    json.dump({k: v["pairs"] for k, v in sets.items()}, fh)
  model = ra.ReanchorNet(dim=int(cfg.model.dim), num_blocks=int(cfg.model.num_blocks), num_heads=cfg.model.num_heads, use_a=use_a,
                         layer_scale_init=float(cfg.model.layer_scale_init),
                         checkpoint_activations=bool(cfg.model.checkpoint_activations)).to(device)
  n_params = sum(p.numel() for p in model.parameters())
  log.info("ReanchorNet: %.1f M parameters (use_a %s)", n_params / 1e6, use_a)
  opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
  dssim = DSSIMLoss(reduction="none").to(device)
  start_step, clock, troughs = 0, 0.0, 0
  ckpt_path = os.path.join(cfg.output_dir, "last.pt")
  st = None
  if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    start_step, clock, troughs = st["step"], st["clock"], st.get("troughs", 0)
    log.info("resumed %s at step %d (%.0f s training clock, %d troughs done)", ckpt_path, start_step, clock, troughs)
    if st.get("done"):
      log.info("the checkpoint is a finished run: nothing to do")
      return
  torch.manual_seed(int(cfg.train.seed) + start_step)
  if st is not None and "torch_rng" in st:
    torch.set_rng_state(st["torch_rng"].cpu())
    torch.cuda.set_rng_state(st["cuda_rng"].cpu())
  del st
  run = init_wandb(cfg)
  run.summary["n_params"] = n_params
  run.summary["train_meshes"], run.summary["val_meshes"] = len(train_m), val_m
  ds = ra.StreamPairs(cfg.data.render_h5, cfg.data.grids_dir, train_m, int(cfg.data.train_views), int(cfg.train.views_per_pair),
                      seed=[int(cfg.train.seed), int(start_step)])
  nw = int(cfg.data.workers)
  loader = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw, pin_memory=True,
                                            prefetch_factor=int(cfg.data.prefetch_factor) if nw else None, persistent_workers=nw > 0))
  eval_file = open(os.path.join(cfg.output_dir, "eval.jsonl"), "a")
  bf16 = bool(cfg.train.bf16)
  max_steps = cfg.train.max_steps
  max_wall = cfg.train.get("max_wall_seconds")
  last_ckpt = time.time()
  stop = {"signal": None}

  def on_term(signum, frame):
    stop["signal"] = signum
  import signal
  signal.signal(signal.SIGTERM, on_term)
  signal.signal(signal.SIGUSR1, on_term)

  def wall_out():
    return stop["signal"] is not None or (max_wall is not None and time.time() - T_START > float(max_wall))

  def do_eval(step, tag):
    t0 = time.time()
    rec = {"step": step, "clock": clock, "tag": tag, "sets": {}}
    logd = {}
    for name, sset in sets.items():
      e = evaluate_bundles(model, batcher, sset["bundles"], use_a, bf16, device)
      r = {"black": float(e[:, 0].mean()), "white": float(e[:, 1].mean()), "n": len(e),
           "black_se": float(e[:, 0].std(ddof=1) / np.sqrt(len(e))), "white_se": float(e[:, 1].std(ddof=1) / np.sqrt(len(e))),
           "per_pair": e.tolist()}
      logd[f"eval/{name}/black"], logd[f"eval/{name}/white"] = r["black"], r["white"]
      if use_a and name in list(cfg.eval.ablate_sets):
        z = evaluate_bundles(model, batcher, sset["bundles"], use_a, bf16, device, ablate_a=True)
        r["ablate_a_black"], r["ablate_a_white"] = float(z[:, 0].mean()), float(z[:, 1].mean())
        logd[f"eval/{name}/ablate_a_black"], logd[f"eval/{name}/ablate_a_white"] = r["ablate_a_black"], r["ablate_a_white"]
      for k, v in sset["ref"].items():
        logd[f"ref/{name}/{k}_black"], logd[f"ref/{name}/{k}_white"] = np.mean(v, 0).tolist()
      rec["sets"][name] = r
    rec["seconds"] = time.time() - t0
    logd["eval/seconds"] = rec["seconds"]
    run.log(logd, step=step)
    eval_file.write(json.dumps(rec) + "\n"); eval_file.flush()
    log.info("eval step %d (%s, %.0f s): %s", step, tag, rec["seconds"],
             "  ".join(f"{n} {r['black']:.3f} / {r['white']:.3f}" for n, r in rec["sets"].items()))

  def save(step, done=False):
    t0 = time.time()
    tmp = ckpt_path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "clock": clock, "troughs": troughs, "done": done,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "cfg": OmegaConf.to_container(cfg, resolve=True)}, tmp)
    os.replace(tmp, ckpt_path)
    log.info("saved %s at step %d (%.0f s training clock) in %.0f s", ckpt_path, step, clock, time.time() - t0)

  step = start_step
  if step == 0:
    do_eval(0, "start")
  torch.cuda.reset_peak_memory_stats()
  reason = "done"
  while clock < float(cfg.train.max_seconds) and (max_steps is None or step < int(max_steps)):
    if wall_out():
      reason = "wall"
      break
    tw = time.time()
    items = [next(loader) for _ in range(int(cfg.train.pairs_per_step))]
    torch.cuda.synchronize()
    t0 = time.time()
    wait = t0 - tw
    lrf = lr_factor(cfg, step, clock)
    for g in opt.param_groups:
      g["lr"] = float(cfg.train.lr) * lrf
    bundles = [ra.stream_bundle(x, device, use_a) for x in items]
    ca, pa, cb, pb, packs = batcher.tokens(bundles, model.empty, use_a)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
      rows = model(ca, pa, cb, pb, packs)
    gauss = batcher.decode(model.readout_grid(rows, packs))
    loss = torch.zeros((), device=device)
    stats = np.zeros(3)
    for i, bd in enumerate(bundles):
      H, W = bd["gt"].shape[-2:]
      rgb, alpha = ra.render(ra.slot_gaussians(gauss, i, bd["b_xyz"], bd["b_hit"]), bd["viewmats"], bd["K"], W, H)
      li, l1, dsv, ps = photometric(dssim, cfg, rgb, alpha, bd["gt"], bd["alpha"], backgrounds(cfg, len(rgb), device))
      loss = loss + li / len(bundles)
      stats += [float(l1), float(dsv), float(ps)]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    k0 = cycle_of(cfg, clock)
    clock += dt
    step += 1
    if step % int(cfg.train.log_every) == 0 or step < 5:
      stats /= len(bundles)
      d = {"train/loss": float(loss), "train/l1": stats[0], "train/dssim": stats[1], "train/psnr": stats[2], "train/lr": opt.param_groups[0]["lr"],
           "train/grad_norm": float(gn), "train/step_seconds": dt, "train/data_wait_seconds": wait, "train/clock": clock,
           "train/tokens_b": packs.b.T, "train/tokens_a": packs.ta, "train/max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
           "train/cycle": cycle_of(cfg, clock)}
      if step % 50 == 0 or step < 5:
        with torch.no_grad():
          d.update(bundle_diagnostics({k: v.detach() for k, v in gauss.items()}, batcher, bundles))
      run.log(d, step=step)
    if step % 50 == 0 or step < 5 or step == start_step + 1:
      log.info("step %d loss %.4f psnr %.2f lr %.2e gn %.3f %.3f s/step (wait %.3f) mem %.1f GB tokens A %d B %d clock %.0f", step,
               float(loss), stats[2], opt.param_groups[0]["lr"], float(gn), dt, wait, torch.cuda.max_memory_allocated() / 2 ** 30,
               packs.ta, packs.b.T, clock)
    trough = nc > 1 and cycle_of(cfg, clock) > k0 and clock < float(cfg.train.max_seconds)
    if trough:
      troughs += 1
      log.info("LR trough %d / %d at step %d (clock %.0f s)", troughs, nc, step, clock)
      do_eval(step, f"trough{troughs}")
      save(step)
      last_ckpt = time.time()
    elif step % int(cfg.train.eval_every) == 0:
      do_eval(step, "regular")
    if time.time() - last_ckpt > float(cfg.train.checkpoint_every_seconds):
      save(step)
      last_ckpt = time.time()
  if reason == "done":
    if nc > 1:
      troughs += 1
    do_eval(step, "final")
    save(step, done=True)
    run.summary["final_step"] = step
  else:
    save(step)
    log.info("stopping at step %d (%.0f s training clock, %.0f s wall): %s", step, clock, time.time() - T_START,
             f"signal {stop['signal']}" if stop["signal"] is not None else "train.max_wall_seconds")
  run.finish()
  log.info("%s: %d steps, %.0f s training", reason, step, clock)


@hydra.main(version_base=None, config_path="conf", config_name="train_reanchor")
def main(cfg):
  os.makedirs(cfg.output_dir, exist_ok=True)
  device = "cuda"
  if cfg.data.get("dataset", "mesh") == "full":
    if cfg.mode != "train":
      raise SystemExit("data.dataset full supports mode train only")
    train_full(cfg, device)
    return
  data, batcher = setup(cfg, device)
  if cfg.mode == "roundtrip":
    roundtrip(cfg, data, batcher)
  elif cfg.mode == "direct":
    direct(cfg, data, batcher, device)
  elif cfg.mode == "train":
    train(cfg, data, batcher, device)
  else:
    raise SystemExit(f"mode must be train | direct | roundtrip, got {cfg.mode!r}")


if __name__ == "__main__":
  main()
