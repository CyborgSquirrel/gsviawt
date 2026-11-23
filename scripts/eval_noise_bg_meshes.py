#!/usr/bin/env python3
"""Multi-mesh version of eval_noise_bg.py: per mesh and arm, held-out PSNR over black and over white, rendered alpha outside the
GT silhouette and Gaussian colour / opacity stats (fine-tuned and shared grids); then per-arm means over the meshes, paired
differences between arms (SE paired by mesh) and two plots.

    python scripts/eval_noise_bg_meshes.py H5 OUT_DIR MESHES name=root[,shared_root] ...

MESHES is comma-separated (e.g. 1,3,7). Per-mesh results are cached in OUT_DIR/per_mesh/mesh<M>.json (delete to recompute).
Writes OUT_DIR/eval.json, psnr_per_mesh.png, alpha_outside_per_mesh.png.
"""

import itertools
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchored_3dgs as an  # noqa: E402
import eval_noise_bg as ev  # noqa: E402

METRICS = ("psnr_black", "psnr_white", "alpha_outside", "colour_all_zero", "opacity_median", "opacity_lt_0.02",
           "opacity_gt_0.98")


def eval_mesh(h5, M, arms, dev):
  rows = an.mesh_rows(h5, M)
  train, held = an.split_views(len(rows), 0.1, None)
  with h5py.File(h5, "r") as f:
    cams = an.Cameras([an.load_view(f, rows[v]) for v in held], dev)
  res = {"mesh_index": M, "held_views": held, "n_train_grids": len(train)}
  for name, roots in arms.items():
    res[name] = {}
    for kind, root in zip(("finetuned", "shared"), roots):
      per, ops, cols = [], [], []
      for v in train:
        p = ev.grid_path(root, M, v)
        per.append(ev.heldout(ev.load_flat(p, dev), cams))
        with h5py.File(p, "r") as f:
          valid = np.asarray(f["layer_valid"])
          ops.append(np.asarray(f["gaussian_opacities"])[valid].reshape(-1))
          cols.append(np.asarray(f["gaussian_colors"])[valid])
      pb, pw, sp = np.array(per).T
      op, col = np.concatenate(ops), np.concatenate(cols)
      res[name][kind] = {"psnr_black": float(pb.mean()), "psnr_white": float(pw.mean()), "alpha_outside": float(sp.mean()),
                         "colour_all_zero": float((col == 0).all(1).mean()), "opacity_median": float(np.median(op)),
                         "opacity_lt_0.02": float((op < 0.02).mean()), "opacity_gt_0.98": float((op > 0.98).mean()),
                         "psnr_black_se_over_grids": float(pb.std(ddof=1) / np.sqrt(len(pb))),
                         "psnr_white_se_over_grids": float(pw.std(ddof=1) / np.sqrt(len(pw)))}
      print(M, name, kind, res[name][kind], flush=True)
  return res


def main():
  h5, out, meshes = sys.argv[1], sys.argv[2], [int(m) for m in sys.argv[3].split(",")]
  arms = {}
  for a in sys.argv[4:]:
    name, roots = a.split("=", 1)
    arms[name] = roots.split(",")
  dev = torch.device("cuda")
  os.makedirs(os.path.join(out, "per_mesh"), exist_ok=True)
  per_mesh = []
  for M in meshes:
    p = os.path.join(out, "per_mesh", f"mesh{M}.json")
    if not os.path.exists(p):
      with open(p, "w") as f:
        json.dump(eval_mesh(h5, M, arms, dev), f, indent=1)
    with open(p) as f:
      per_mesh.append(json.load(f))

  names = list(arms)
  summary = {"meshes": meshes, "arms": {n: r for n, r in arms.items()}, "n_meshes": len(meshes)}
  for kind in ("finetuned", "shared"):
    s = summary[kind] = {"mean": {}, "paired_diff": {}}
    val = {n: {k: np.array([r[n][kind][k] for r in per_mesh]) for k in METRICS} for n in names}
    for n in names:
      s["mean"][n] = {k: float(v.mean()) for k, v in val[n].items()}
      s["mean"][n]["alpha_outside_median_over_meshes"] = float(np.median(val[n]["alpha_outside"]))
    for a, b in itertools.combinations(names, 2):
      # differences b - a (later arm minus earlier), paired by mesh; alpha also as a log10 ratio
      d = {k: val[b][k] - val[a][k] for k in METRICS}
      d["log10_alpha_outside_ratio"] = np.log10(val[b]["alpha_outside"] / val[a]["alpha_outside"])
      s["paired_diff"][f"{b}-{a}"] = {k: {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))),
                                          "min": float(x.min()), "max": float(x.max())} for k, x in d.items()}
    s["per_mesh"] = {n: {k: v.tolist() for k, v in val[n].items()} for n in names}
  with open(os.path.join(out, "eval.json"), "w") as f:
    json.dump(summary, f, indent=1)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  x = np.arange(len(meshes))
  pm = summary["finetuned"]["per_mesh"]
  fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
  for i, (key, title) in enumerate((("psnr_white", "over a white background"), ("psnr_black", "over a black background"))):
    for k, n in enumerate(names):
      ax[i].plot(x, pm[n][key], "o-", color=ev.COLORS[k], lw=1.2, ms=6, label=f"trained over {n}")
    ax[i].set(title=f"Held-out PSNR {title}", xlabel="mesh", ylabel="PSNR (dB)", xticks=x, xticklabels=meshes)
    ax[i].spines[["top", "right"]].set_visible(False)
    ax[i].grid(axis="y", alpha=0.3)
  ax[0].legend(frameon=False)
  fig.suptitle("Fine-tuned grids, held-out views 45-49, mean over the 45 grids of each mesh")
  fig.tight_layout()
  fig.savefig(os.path.join(out, "psnr_per_mesh.png"), dpi=130)

  fig, ax = plt.subplots(figsize=(7, 4.2))
  for k, n in enumerate(names):
    ax.plot(x, pm[n]["alpha_outside"], "o-", color=ev.COLORS[k], lw=1.2, ms=6, label=f"trained over {n}")
  ax.set(title="Opacity rendered outside the object's silhouette", xlabel="mesh",
         ylabel="summed alpha per held-out image\n(fully opaque pixel equivalents, log scale)", yscale="log", xticks=x,
         xticklabels=meshes)
  ax.spines[["top", "right"]].set_visible(False)
  ax.grid(axis="y", alpha=0.3)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(out, "alpha_outside_per_mesh.png"), dpi=130)


if __name__ == "__main__":
  main()
