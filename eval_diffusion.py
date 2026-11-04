"""Render-space evaluation of diffusion-mode train_gs.py checkpoints (conf/eval_diffusion.yaml).

Every checkpoint x weights (raw / EMA) is scored on the run's validation volumes with FIXED target views and noise (functions of
seed, set and volume, so every run and checkpoint sees the same ones) and with metrics computed here, in fp32 with autocast off --
the same numbers whatever code version trained the run. Per target view, of renders premultiplied over black against the
dataset images:
  l1, dssim (module.DSSIMLoss), psnr, loss = l1 + 0.85 dssim   the whole frame (dominated by the black background)
  fg_l1, fg_psnr       over the TARGET's silhouette (alpha > 0.5) only -- a fixed region per view, so spreading coverage
                       (haze) cannot dilute the error (a union with the render's silhouette would reward it)
  bg_l1                over the rest of the frame: what the render puts where the target is empty (floaters, haze)
  lpips                LPIPS (VGG, torchmetrics), whole frame
  alpha_iou            IoU of the render's silhouette with the target's (alpha > 0.5)
  depth_absrel         raw (unaligned) |depth - mesh depth| / mesh depth, the render's expected depth against the target view's
                       first depth-peel layer, over the pixels both cover
  angle                the angle (degrees) between the source and the target camera's viewing directions
Methods:
  fit        the volume's own fitted grid (the target), encoded -> decoded: the reference the model is trained towards
  onestep    the x0 prediction from pure noise (t = 1), one sampler step
  sample     one full sampler run (k = 0)
  meanK      the mean of the K samples' RENDERS
  bestK      the best of the K samples by loss (an oracle: it looks at the target)
Writes <out>/views.csv (one row per checkpoint, weights, set, volume, method, target view), <out>/volumes.csv (the same averaged
over the volume's target views) and <out>/summary.json (mean and standard error over volumes per checkpoint, weights, set, method;
volumes, not views, are the unit). mean<K> has no depth (an average of renders has no single depth)."""
import contextlib as ctl
import csv
import json
import math
import os
import random

import gsplat
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import gs_diffusion as gd
import train_gs
from module import DSSIMLoss
from util import collate_with_batch_size


METRICS = ("l1", "dssim", "psnr", "loss", "fg_l1", "fg_psnr", "bg_l1", "lpips", "alpha_iou", "depth_absrel")


def render(model, batch, gauss, views, device):
  """(rgb (V, 3, H, W), alpha (V, 1, H, W), expected depth (V, 1, H, W)) of one volume's Gaussian dict (batch dim 1) at the given
  view indices of batch["views"]."""
  flat = next(train_gs.flatten_gaussians(1, gauss, batch["source"]["xyz_cam"].to(device), batch["source"]["hit"].to(device)))
  h, w = batch["views"]["rgb"].shape[-2:]
  out, alpha, _ = gsplat.rasterization(
    means=flat["means"].float(), quats=flat["quats"].float(), scales=flat["scales"].float(), opacities=flat["opacities"].float(),
    colors=flat["colors"].float(), viewmats=batch["views"]["viewmat"][0, views].to(device),
    Ks=batch["views"]["K_image"][0, views].to(device), width=int(w), height=int(h), sh_degree=model.max_sh_degree,
    render_mode="RGB+ED", packed=True)
  out = out.permute(0, 3, 1, 2)
  return out[:, :3].clamp(0, 1), alpha.permute(0, 3, 1, 2).clamp(0, 1), out[:, 3:4]


def metrics(pred, gt, dssim, lpips, alpha=None, gt_alpha=None, depth=None, gt_depth=None):
  """Per view (see the module docstring), fp32 with autocast off. pred / gt (V, 3, H, W); alpha, gt_alpha, depth, gt_depth
  (V, 1, H, W) or None (then the metrics that need them are NaN). gt_depth: NaN where the mesh is not hit."""
  nan = torch.full((pred.shape[0],), float("nan"), device=pred.device)
  with torch.autocast(device_type=pred.device.type, enabled=False):
    pred, gt = pred.float(), gt.float()
    l1 = (pred - gt).abs().flatten(1).mean(1)
    ds = dssim(pred, gt).flatten(1).mean(1)
    mse = (pred - gt).square().flatten(1).mean(1)
    lp = torch.stack([lpips(pred[v:v + 1], gt[v:v + 1]) for v in range(pred.shape[0])]).float()
    out = {"l1": l1, "dssim": ds, "psnr": -10 * torch.log10(mse.clamp_min(1e-10)), "loss": l1 + 0.85 * ds, "lpips": lp,
           "fg_l1": nan, "fg_psnr": nan, "bg_l1": nan, "alpha_iou": nan, "depth_absrel": nan}
    if gt_alpha is not None:
      g = gt_alpha.float() > 0.5
      p = alpha.float() > 0.5 if alpha is not None else g
      fg, bg = g.float(), (~g).float()
      nf, nb = fg.flatten(1).sum(1).clamp_min(1), bg.flatten(1).sum(1).clamp_min(1)
      err1 = (pred - gt).abs().mean(1, keepdim=True)
      out["fg_l1"] = (err1 * fg).flatten(1).sum(1) / nf
      out["bg_l1"] = (err1 * bg).flatten(1).sum(1) / nb
      fg_mse = ((pred - gt).square().mean(1, keepdim=True) * fg).flatten(1).sum(1) / nf
      out["fg_psnr"] = -10 * torch.log10(fg_mse.clamp_min(1e-10))
      if alpha is not None:
        out["alpha_iou"] = (g & p).float().flatten(1).sum(1) / (g | p).float().flatten(1).sum(1).clamp_min(1)
      if depth is not None and gt_depth is not None:
        both = (torch.isfinite(gt_depth) & (gt_depth > 0) & p)
        rel = ((depth.float() - gt_depth).abs() / gt_depth).nan_to_num(0) * both
        out["depth_absrel"] = rel.flatten(1).sum(1) / both.float().flatten(1).sum(1).clamp_min(1)
  return out


def view_angles(batch, views):
  """(V,) angle in degrees between the source camera's viewing direction and each target camera's (viewmats are relative to the
  source camera: viewmat[0] = I)."""
  vm = batch["views"]["viewmat"][0, views].float()                       # world(=source camera)-to-camera
  fwd = torch.linalg.inv(vm)[:, :3, 2]                                   # the target cameras' +Z axes in the source frame
  cos = (fwd[:, 2] / fwd.norm(dim=1)).clamp(-1, 1)
  return torch.rad2deg(torch.acos(cos))


def volume_subset(n, max_volumes, seed, set_name):
  """The dataset indices of a set to score: all of them, or max_volumes of them drawn at random but reproducibly (a function
  of seed and the set's name only), so a subset is spread over all of the set's meshes and is the same for every run with
  the same split. Sorted."""
  if max_volumes is None or int(max_volumes) >= n:
    return list(range(n))
  rng = np.random.default_rng([int(seed), *map(ord, set_name)])
  return sorted(rng.choice(n, int(max_volumes), replace=False).tolist())


def load_checkpoint(module, path):
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
  if missing or unexpected:
    raise SystemExit(f"{path}: state_dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
  has_ema = "ema" in ckpt and module.ema is not None
  if has_ema:
    module.ema.load_state_dict(ckpt["ema"])
  return has_ema


@hydra.main(version_base=None, config_path="conf", config_name="eval_diffusion")
def main(cfg):
  device = torch.device("cuda")
  run = OmegaConf.load(cfg.run_config)
  os.makedirs(cfg.out, exist_ok=True)
  with open_dict(run):
    run.output_dir = cfg.out                      # listing.h5 goes here, not into the run's own directory
    run.loader.num_workers = 0
    if cfg.grids_dir is not None:
      run.diffusion.grids_dir = cfg.grids_dir
    if cfg.source_h5 is not None:
      run.diffusion.source_h5 = cfg.source_h5
  if not train_gs.diffusion_enabled(run):
    raise SystemExit("not a diffusion-mode run")
  dm = train_gs.GSDataModule(run)
  dm.setup()
  for ds in dm.val_datasets.values():   # the target views come from this script's own per-volume seed (random.seed below), as in
    ds.seed_targets = False              # every eval so far -- not the validations' seeding, so evals stay paired across code versions
  module = train_gs.GSLightningModule(run, dm.diffusion_stats).to(device).eval()
  model = module.model
  dssim = DSSIMLoss(reduction="none").to(device)
  from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
  lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device).eval()
  K, steps = int(cfg.samples), (int(cfg.steps) if cfg.steps is not None else model.steps)
  sets = list(cfg.val_sets) if cfg.val_sets is not None else list(dm.val_datasets)

  rows, view_rows = [], []
  for ckpt_path in cfg.checkpoints:
    has_ema = False if ckpt_path == "init" else load_checkpoint(module, ckpt_path)
    for weights in cfg.weights:
      if weights == "ema" and not has_ema:
        print(f"{ckpt_path}: no EMA weights, skipping 'ema'", flush=True)
        continue
      ctx = module.ema.applied(model) if weights == "ema" else ctl.nullcontext()
      with ctx, torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for set_name in sets:
          ds = dm.val_datasets[set_name]
          idx = volume_subset(len(ds), cfg.max_volumes, cfg.seed, set_name)
          n = len(idx)
          for k, i in enumerate(idx):
            random.seed(f"{cfg.seed}/{set_name}/{i}")       # the dataset draws the target views from the global `random`
            item = ds[i]
            batch = collate_with_batch_size([item])
            views = list(range(1, batch["views"]["rgb"].shape[1]))   # the target views (index 0 is the source)
            gt = batch["views"]["rgb"][0, views].to(device)
            gt_alpha = batch["views"]["alpha"][0, views].to(device)
            with h5py.File(ds.inner.source_h5, "r") as f:   # the target views' first depth-peel layer (z-depth, NaN = no hit)
              gt_depth = torch.from_numpy(np.stack([f["depth_peel"][int(r)][..., 0] for r in item["target_views"]]))
            gt_depth = gt_depth.float().to(device)[:, None]
            if gt_depth.shape[-2:] != gt.shape[-2:]:
              gt_depth = None                                # the depth peel is at another resolution than the images
            angle = view_angles(batch, views)
            image, hit, _ = model.inputs(batch, device)
            project = None
            if cfg.get("project_format"):   # the sampler's intermediate x0 predictions put into the training grids' format
              enc = train_gs.x0_encoding(run)
              project = gd.format_projection(model.x0_mean, model.x0_std, hit, item["source"]["xyz_cam"][..., 2].to(device),
                                             float(item["source"]["K_depth"][0, 0]), **enc)
            base = 1_000_003 * (int(cfg.seed) + 1) + int(item["source_view"]) * 101
            seeds = [base + k for k in range(K)]
            out = {"fit": [render(model, batch, model.decode(model.normalize(batch["x0_raw"].to(device).float())), views, device)],
                   "onestep": [render(model, batch, model.sample(image, hit, seeds=seeds[:1], steps=1, project=project), views, device)]}
            g = model.sample(image.expand(K, *image.shape[1:]), hit.expand(K, *hit.shape[1:]), seeds=seeds, steps=steps, project=project)
            out["samples"] = [render(model, batch, {k: v[j:j + 1] for k, v in g.items()}, views, device) for j in range(K)]
            M = lambda r: metrics(r[0], gt, dssim, lpips, r[1], gt_alpha, r[2], gt_depth)
            per = {m: M(r[0]) for m, r in out.items() if m != "samples"}
            ks = [M(r) for r in out["samples"]]
            per["sample"] = ks[0]
            per[f"mean{K}"] = metrics(torch.stack([r[0] for r in out["samples"]]).mean(0), gt, dssim, lpips,
                                      torch.stack([r[1] for r in out["samples"]]).mean(0), gt_alpha)
            best = int(np.argmin([float(m["loss"].mean()) for m in ks]))
            per[f"best{K}"] = ks[best]
            for method, m in per.items():
              key = {"checkpoint": ckpt_path, "weights": weights, "set": set_name, "volume": i,
                     "mesh": int(item["mesh_index"]), "source_view": int(item["source_view"])}
              rows.append({**key, "target_views": " ".join(map(str, item["target_views"])), "method": method,
                           **{k: float(m[k].nanmean()) if torch.isfinite(m[k]).any() else float("nan") for k in METRICS}})
              for v, tv in enumerate(item["target_views"]):
                view_rows.append({**key, "target_view": int(tv), "angle": float(angle[v]), "method": method,
                                  **{k: float(m[k][v]) for k in METRICS}})
            print(f"{os.path.basename(str(ckpt_path))} {weights} {set_name} {k + 1}/{n} (volume {i}):",
                  {m: round(float(v["loss"].mean()), 4) for m, v in per.items()}, flush=True)

  for name, rs in (("volumes.csv", rows), ("views.csv", view_rows)):
    with open(os.path.join(cfg.out, name), "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(rs[0]))
      w.writeheader()
      w.writerows(rs)
  summary = {}
  for r in rows:
    summary.setdefault((r["checkpoint"], r["weights"], r["set"], r["method"]), []).append(r)
  out = []
  for (c, wt, s, m), rs in summary.items():
    entry = {"checkpoint": c, "weights": wt, "set": s, "method": m, "n_volumes": len(rs)}
    for k in METRICS:
      v = np.array([r[k] for r in rs], dtype=float)
      v = v[np.isfinite(v)]
      entry[k] = float(v.mean()) if len(v) else None
      entry[f"{k}_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
    out.append(entry)
  with open(os.path.join(cfg.out, "summary.json"), "w") as f:
    json.dump({"config": OmegaConf.to_container(cfg), "results": out}, f, indent=1)
  print(f"wrote {cfg.out}", flush=True)


if __name__ == "__main__":
  main()
