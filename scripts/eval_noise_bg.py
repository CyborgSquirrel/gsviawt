#!/usr/bin/env python3
"""Evaluates fit_anchored.py loss.background arms of one mesh: held-out PSNR over black and over white, Gaussian colour / opacity
stats, a histogram plot and side-by-side held-out renders.

    python scripts/eval_noise_bg.py H5 MESH OUT_DIR name=root[,shared_root] ...

Each root holds the mesh's grids as <root>/mesh<M>_view<V>/gaussians.grid.h5 or <root>/grids/mesh<M>_view<V>.h5. The held-out views
are the mesh's last round(0.1 x n) views (fit_anchored.py's split). Grids are rendered straight from their activated values
(gsplat, RGB colours, no SH), clamped to [0,1] like fit_3dgs.render. Writes OUT_DIR/eval.json, fit_noise_bg.png, renders.png.
"""

import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import anchored_3dgs as an  # noqa: E402

COLORS = {0: "#2a78d6", 1: "#eb6834"}


def grid_path(root, M, v):
  for p in (os.path.join(root, f"mesh{M}_view{v}", "gaussians.grid.h5"), os.path.join(root, "grids", f"mesh{M}_view{v}.h5")):
    if os.path.exists(p):
      return p
  raise FileNotFoundError(f"{root}: no grid for mesh {M} view {v}")


def load_flat(path, dev):
  with h5py.File(path, "r") as f:
    valid = np.asarray(f["layer_valid"])
    g = {k: np.asarray(f["gaussian_" + k])[valid] for k in ("means", "scales", "quats", "opacities", "colors")}
  return {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in g.items()}


def render(g, cams, i):
  import gsplat
  rgb, alpha, _ = gsplat.rasterization(
    means=g["means"], quats=F.normalize(g["quats"], dim=-1), scales=g["scales"], opacities=g["opacities"].reshape(-1),
    colors=g["colors"], sh_degree=None, viewmats=cams.viewmat[[i]], Ks=cams.K[[i]], width=cams.W, height=cams.H,
    render_mode="RGB", packed=True)
  return rgb[0].clamp(0, 1), alpha[0].clamp(0, 1)


def psnr(a, b):
  return float(-10 * torch.log10((a - b).pow(2).mean()))


@torch.no_grad()
def heldout(g, cams):
  """(mean PSNR over black, mean PSNR over white) over the held-out cams"""
  pb, pw = [], []
  for i in range(len(cams)):
    rgb, a = render(g, cams, i)
    pb.append(psnr(rgb, cams.gt[i]))
    pw.append(psnr(rgb + (1 - a), cams.gt[i] + (1 - cams.alpha[i])))
  return float(np.mean(pb)), float(np.mean(pw))


def main():
  h5, M, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
  arms = {}
  for a in sys.argv[4:]:
    name, roots = a.split("=", 1)
    arms[name] = roots.split(",")
  dev = torch.device("cuda")
  rows = an.mesh_rows(h5, M)
  train, held = an.split_views(len(rows), 0.1, None)
  with h5py.File(h5, "r") as f:
    held_views = [an.load_view(f, rows[v]) for v in held]
    train_centers = np.stack([np.asarray(f["camera_pose"][rows[v]])[:3, 3] for v in train])
  cams = an.Cameras(held_views, dev)
  res = {"mesh_index": M, "held_views": held, "n_train_grids": len(train)}
  pooled = {}
  for name, roots in arms.items():
    r = res[name] = {}
    for kind, root in zip(("finetuned", "shared"), roots):
      per = [heldout(load_flat(grid_path(root, M, v), dev), cams) for v in train]
      pb, pw = np.array(per).T
      r[kind] = {"psnr_black_mean": float(pb.mean()), "psnr_black_se": float(pb.std(ddof=1) / np.sqrt(len(pb))),
                 "psnr_white_mean": float(pw.mean()), "psnr_white_se": float(pw.std(ddof=1) / np.sqrt(len(pw))),
                 "psnr_black_per_view": pb.tolist(), "psnr_white_per_view": pw.tolist()}
      ops, cols = [], []
      for v in train:
        with h5py.File(grid_path(root, M, v), "r") as f:
          valid = np.asarray(f["layer_valid"])
          ops.append(np.asarray(f["gaussian_opacities"])[valid].reshape(-1))
          cols.append(np.asarray(f["gaussian_colors"])[valid])
      op, col = np.concatenate(ops), np.concatenate(cols)
      r[kind].update({"n_gaussians_pooled": int(len(op)), "colour_all_zero": float((col == 0).all(1).mean()),
                      "colour_any_zero": float((col == 0).any(1).mean()), "opacity_lt_0.02": float((op < 0.02).mean()),
                      "opacity_gt_0.98": float((op > 0.98).mean()), "opacity_median": float(np.median(op))})
      if kind == "finetuned":
        pooled[name] = (op, col.max(1))
      print(name, kind, {k: v for k, v in r[kind].items() if "per_view" not in k}, flush=True)
  with open(os.path.join(out, "eval.json"), "w") as f:
    json.dump(res, f, indent=1)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1, 2, figsize=(11, 4))
  cmax = max(float(np.percentile(c, 99.9)) for _, c in pooled.values())
  for k, (name, (op, cm)) in enumerate(pooled.items()):
    lbl = f"{name} background"
    ax[0].hist(op, bins=np.linspace(0, 1, 101), histtype="step", lw=2, color=COLORS[k], label=lbl, density=True)
    ax[1].hist(np.clip(cm, 0, cmax), bins=np.linspace(0, cmax, 101), histtype="step", lw=2, color=COLORS[k], label=lbl,
               density=True)
  ax[0].set(title="Gaussian opacity", xlabel="opacity", ylabel="density (log scale)", yscale="log")
  ax[1].set(title="Gaussian colour (brightest channel)", xlabel=f"max(R, G, B), clipped at {cmax:.2f}",
            ylabel="density (log scale)", yscale="log")
  for a in ax:
    a.legend(frameon=False)
    a.spines[["top", "right"]].set_visible(False)
  fig.suptitle(f"Mesh {M}, all {len(train)} fine-tuned grids pooled, training loss over black vs over random noise")
  fig.tight_layout()
  fig.savefig(os.path.join(out, "fit_noise_bg.png"), dpi=130)

  # held-out renders: views held[0], held[2], each by the fine-tuned grid of the nearest training camera
  show = [0, 2]
  fig, axs = plt.subplots(2 * len(show), 1 + len(arms), figsize=(3.2 * (1 + len(arms)), 3.2 * 2 * len(show)))
  with torch.no_grad():
    for r_i, i in enumerate(show):
      pose = held_views[i]["pose"]
      src = train[int(np.argmin(np.linalg.norm(train_centers - pose[:3, 3], axis=1)))]
      renders = {name: render(load_flat(grid_path(roots[0], M, src), dev), cams, i) for name, roots in arms.items()}
      # crop: the GT's bounding box with a margin of 50% of its size (floaters around it stay visible); PSNRs are full-image
      ys, xs = np.nonzero(cams.alpha[i][..., 0].cpu().numpy() > 0)
      cy, cx, half = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2, 0.75 * max(np.ptp(ys), np.ptp(xs)) + 8
      y0, y1 = int(max(0, cy - half)), int(min(cams.H, cy + half))
      x0, x1 = int(max(0, cx - half)), int(min(cams.W, cx + half))
      for b_i, (bgname, bg) in enumerate((("black", 0.0), ("white", 1.0))):
        row = axs[2 * r_i + b_i]
        gt = cams.gt[i] + (1 - cams.alpha[i]) * bg
        row[0].imshow(gt[y0:y1, x0:x1].cpu().numpy().clip(0, 1))
        row[0].set_title(f"GT, held-out view {held[i]}, over {bgname}", fontsize=9)
        for c, (name, (rgb, a)) in enumerate(renders.items()):
          im = rgb + (1 - a) * bg
          row[c + 1].imshow(im[y0:y1, x0:x1].cpu().numpy().clip(0, 1))
          row[c + 1].set_title(f"{name} arm (grid of view {src}): {psnr(im, gt):.2f} dB", fontsize=9)
        for a_ in row:
          a_.axis("off")
  fig.tight_layout()
  fig.savefig(os.path.join(out, "renders.png"), dpi=110)


if __name__ == "__main__":
  main()
