#!/usr/bin/env python3
"""Single-panel figure: per-view mean surface depth, World Tracing prediction
vs capture_turntable render (ground truth), grouped by mesh.

Just the mean-depth view from compare_wt_depth.py's summary figure, pulled out
on its own and laid out so the N-views-per-mesh structure is obvious: each mesh
is one x-slot, its views fanned out within the slot and joined by a light line,
with alternating background bands per mesh.

Depths are RAW (no scale alignment): GT is in the render's Blender units, WT is
its own normalised output -- so the vertical gap per mesh is exactly the scale
mismatch that compare_wt_depth.py's `fit scale` corrects.

    python plot_mean_depth.py bla/lite.h5 bla/lite.h5.wt.h5 [--layer 0] [--out ...]
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import compare_view


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--layer", type=int, default=0)
  p.add_argument("--out", default=None, help="default: <wt_h5>.meandepth.png")
  args = p.parse_args()

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    depth_peel, points = rf["depth_peel"], wf["points"]
    n = min(depth_peel.shape[0], points.shape[0])
    mesh_index = (rf["mesh_index"][:n] if "mesh_index" in rf
                  else np.zeros(n, dtype=int))
    gt_md, pred_md, mi = [], [], []
    for i in range(n):
      m, *_ = compare_view(depth_peel[i], points[i], args.layer, "none")
      if np.isfinite(m["gt_mean_depth"]):
        gt_md.append(m["gt_mean_depth"])
        pred_md.append(m["pred_mean_depth"])
        mi.append(int(mesh_index[i]))

  gt_md = np.array(gt_md)
  pred_md = np.array(pred_md)
  mi = np.array(mi)

  # x = mesh slot + a small fan-out offset per view within the mesh
  meshes = sorted(set(mi.tolist()))
  x = np.empty(len(mi))
  for mesh in meshes:
    sel = np.where(mi == mesh)[0]
    k = len(sel)
    off = (np.arange(k) - (k - 1) / 2) * (0.6 / max(k, 1))
    x[sel] = mesh + off

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(max(9, len(meshes) * 1.15), 5.5))

  # alternating background band per mesh
  for j, mesh in enumerate(meshes):
    ax.axvspan(mesh - 0.5, mesh + 0.5, color="0.5" if j % 2 else "1.0",
               alpha=0.08, zorder=0)

  # per-mesh connectors: join that mesh's views, GT and WT separately
  for mesh in meshes:
    sel = np.where(mi == mesh)[0]
    order = sel[np.argsort(x[sel])]
    ax.plot(x[order], gt_md[order], "-", color="tab:blue", lw=1, alpha=0.5, zorder=1)
    ax.plot(x[order], pred_md[order], "-", color="tab:red", lw=1, alpha=0.5, zorder=1)

  ax.scatter(x, gt_md, marker="o", s=44, facecolors="none", edgecolors="tab:blue",
             linewidths=1.6, label="render (GT), Blender units", zorder=3)
  ax.scatter(x, pred_md, marker="x", s=40, color="tab:red", linewidths=1.6,
             label="WT prediction, raw", zorder=4)

  ax.set_yscale("log")
  ax.set_xticks(meshes)
  ax.set_xticklabels([f"mesh {m}\n({int((mi == m).sum())} views)" for m in meshes])
  ax.set_xlim(meshes[0] - 0.5, meshes[-1] + 0.5)
  ax.set_ylabel("mean surface depth over shared valid pixels")
  ax.set_title(f"WT vs render mean depth, {len(gt_md)} views, {len(meshes)} meshes "
               f"(raw — no scale alignment)")
  ax.grid(alpha=0.3, axis="y", which="both")
  ax.legend(loc="upper left", framealpha=0.9)
  fig.tight_layout()

  out = args.out or f"{args.wt_h5}.meandepth.png"
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[fig] {out}")


if __name__ == "__main__":
  main()
