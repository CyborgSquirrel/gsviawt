"""Render-space evaluation of diffusion-mode train_gs.py checkpoints (conf/eval_diffusion.yaml).

Every checkpoint x weights (raw / EMA) is scored on the run's validation volumes with FIXED target views and noise (functions of
seed, set and volume, so every run and checkpoint sees the same ones) and with metrics computed here, in fp32 with autocast off --
the same numbers whatever code version trained the run. Per volume and target view: L1, D-SSIM (module.DSSIMLoss), PSNR, and
loss = L1 + 0.85 D-SSIM, of renders premultiplied over black against the dataset images. Methods:
  fit        the volume's own fitted grid (the target), encoded -> decoded: the reference the model is trained towards
  onestep    the x0 prediction from pure noise (t = 1), one sampler step
  sample     one full sampler run (k = 0)
  meanK      the mean of the K samples' RENDERS
  bestK      the best of the K samples by loss (an oracle: it looks at the target)
Writes <out>/volumes.csv (one row per checkpoint, weights, set, volume, method) and <out>/summary.json (mean and standard error
over volumes per checkpoint, weights, set, method; volumes, not views, are the unit)."""
import contextlib as ctl
import csv
import json
import math
import os
import random

import gsplat
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

import train_gs
from module import DSSIMLoss
from util import collate_with_batch_size


def render(model, batch, gauss, views, device):
  """(len(views), 3, H, W) renders of one volume's Gaussian dict (batch dim 1) at the given view indices of batch["views"]."""
  flat = next(train_gs.flatten_gaussians(1, gauss, batch["source"]["xyz_cam"].to(device), batch["source"]["hit"].to(device)))
  h, w = batch["views"]["rgb"].shape[-2:]
  rgb, _, _ = gsplat.rasterization(
    means=flat["means"].float(), quats=flat["quats"].float(), scales=flat["scales"].float(), opacities=flat["opacities"].float(),
    colors=flat["colors"].float(), viewmats=batch["views"]["viewmat"][0, views].to(device),
    Ks=batch["views"]["K_image"][0, views].to(device), width=int(w), height=int(h), sh_degree=model.max_sh_degree,
    render_mode="RGB", packed=True)
  return rgb.clamp(0, 1).permute(0, 3, 1, 2)


def metrics(pred, gt, dssim):
  """Per view: L1, D-SSIM, PSNR, loss (fp32, autocast off)."""
  with torch.autocast(device_type=pred.device.type, enabled=False):
    pred, gt = pred.float(), gt.float()
    l1 = (pred - gt).abs().flatten(1).mean(1)
    ds = dssim(pred, gt).flatten(1).mean(1)
    mse = (pred - gt).square().flatten(1).mean(1)
  psnr = -10 * torch.log10(mse.clamp_min(1e-10))
  return {"l1": l1, "dssim": ds, "psnr": psnr, "loss": l1 + 0.85 * ds}


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
  module = train_gs.GSLightningModule(run, dm.diffusion_stats).to(device).eval()
  model = module.model
  dssim = DSSIMLoss(reduction="none").to(device)
  K, steps = int(cfg.samples), (int(cfg.steps) if cfg.steps is not None else model.steps)
  sets = list(cfg.val_sets) if cfg.val_sets is not None else list(dm.val_datasets)

  rows = []
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
          n = len(ds) if cfg.max_volumes is None else min(len(ds), int(cfg.max_volumes))
          for i in range(n):
            random.seed(f"{cfg.seed}/{set_name}/{i}")       # the dataset draws the target views from the global `random`
            item = ds[i]
            batch = collate_with_batch_size([item])
            views = list(range(1, batch["views"]["rgb"].shape[1]))   # the target views (index 0 is the source)
            gt = batch["views"]["rgb"][0, views].to(device)
            image, hit, _ = model.inputs(batch, device)
            base = 1_000_003 * (int(cfg.seed) + 1) + int(item["source_view"]) * 101
            seeds = [base + k for k in range(K)]
            out = {"fit": [render(model, batch, model.decode(model.normalize(batch["x0_raw"].to(device).float())), views, device)],
                   "onestep": [render(model, batch, model.sample(image, hit, seeds=seeds[:1], steps=1), views, device)]}
            g = model.sample(image.expand(K, *image.shape[1:]), hit.expand(K, *hit.shape[1:]), seeds=seeds, steps=steps)
            out["samples"] = [render(model, batch, {k: v[j:j + 1] for k, v in g.items()}, views, device) for j in range(K)]
            per = {m: metrics(r[0], gt, dssim) for m, r in out.items() if m != "samples"}
            ks = [metrics(r, gt, dssim) for r in out["samples"]]
            per["sample"] = ks[0]
            per[f"mean{K}"] = metrics(torch.stack(out["samples"]).mean(0), gt, dssim)
            best = int(np.argmin([float(m["loss"].mean()) for m in ks]))
            per[f"best{K}"] = ks[best]
            for method, m in per.items():
              rows.append({"checkpoint": ckpt_path, "weights": weights, "set": set_name, "volume": i,
                           "mesh": int(item["mesh_index"]), "source_view": int(item["source_view"]),
                           "target_views": " ".join(map(str, item["target_views"])), "method": method,
                           **{k: float(v.mean()) for k, v in m.items()}})
            print(f"{os.path.basename(str(ckpt_path))} {weights} {set_name} {i + 1}/{n}:",
                  {m: round(float(v["loss"].mean()), 4) for m, v in per.items()}, flush=True)

  with open(os.path.join(cfg.out, "volumes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  summary = {}
  for r in rows:
    summary.setdefault((r["checkpoint"], r["weights"], r["set"], r["method"]), []).append(r)
  out = []
  for (c, wt, s, m), rs in summary.items():
    entry = {"checkpoint": c, "weights": wt, "set": s, "method": m, "n_volumes": len(rs)}
    for k in ("l1", "dssim", "psnr", "loss"):
      v = np.array([r[k] for r in rs])
      entry[k] = float(v.mean())
      entry[f"{k}_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
    out.append(entry)
  with open(os.path.join(cfg.out, "summary.json"), "w") as f:
    json.dump({"config": OmegaConf.to_container(cfg), "results": out}, f, indent=1)
  print(f"wrote {cfg.out}", flush=True)


if __name__ == "__main__":
  main()
