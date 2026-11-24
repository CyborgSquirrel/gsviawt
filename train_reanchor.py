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
  lo, hi = float(batcher.lo[ra.SCALE].min()), float(batcher.hi[ra.SCALE].max())
  d = {"clamp": [], "off": [], "op": [], "empty": []}
  for i, (_, b) in enumerate(pairs):
    hit = data.hit[b]
    raw = gauss["raw_scale"][i].permute(0, 2, 3, 1)[hit]
    d["clamp"].append(float(((raw <= lo + 1e-6) | (raw >= hi - 1e-6)).float().mean()))
    off = gauss["offset"][i].permute(0, 2, 3, 1)[hit].norm(dim=1)
    foot = data.xyz[b][:, 2] / data.K_depth[b][0, 0]
    d["off"].append(float((off / foot).median()))
    op = gauss["opacity"][i].permute(0, 2, 3, 1)[hit]
    d["op"].append(float(op.mean()))
    d["empty"].append(float((op < 0.004).float().mean()))
  return {f"diag/{k}": float(np.mean(v)) for k, v in d.items()}


def lr_factor(cfg, step, seconds):
  w = min(1.0, (step + 1) / max(1, int(cfg.train.warmup_steps)))
  frac = seconds / float(cfg.train.max_seconds)
  df = float(cfg.train.decay_fraction)
  if frac > 1 - df:
    m = float(cfg.train.min_lr) / float(cfg.train.lr)
    w *= m + (1 - m) * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - (1 - df)) / df)))
  return w


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


@hydra.main(version_base=None, config_path="conf", config_name="train_reanchor")
def main(cfg):
  os.makedirs(cfg.output_dir, exist_ok=True)
  device = "cuda"
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
