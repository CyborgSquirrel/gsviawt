#!/usr/bin/env python3
"""Boxplots of per-pixel surface depth, World Tracing prediction vs
capture_turntable render (ground truth), one box pair per view, grouped by
mesh -- the distribution analogue of plot_mean_depth.py's point plot (which
only shows the mean).

Each view contributes two boxes (quartiles + whiskers, outliers hidden) side
by side: render (GT, blue) and WT prediction (red). Same mesh-slot x-axis
layout as plot_mean_depth.py, so the N-views-per-mesh structure stays
readable.

    python plot_depth_boxplots.py bla/lite.h5 bla/lite.h5.wt.h5 [--align none] [--out ...]
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _align, _layer0_depth_gt, _layer0_depth_pred


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--layer", type=int, default=0)
  p.add_argument(
    "--align", choices=["none", "median", "scale", "affine"], default="none",
    help="Align WT to GT per view before boxing (default: none, i.e. raw "
         "pixel values -- shows the same units both were saved in)")
  p.add_argument(
    "--max-points", type=int, default=20000,
    help="Subsample each box to at most this many pixels (quantiles are "
         "stable well below full resolution; keeps rendering fast)")
  p.add_argument("--out", default=None, help="default: <wt_h5>.depthbox.png")
  args = p.parse_args()

  rng = np.random.default_rng(0)

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    depth_peel, points = rf["depth_peel"], wf["points"]
    n = min(depth_peel.shape[0], points.shape[0])
    mesh_index = (rf["mesh_index"][:n] if "mesh_index" in rf
                  else np.zeros(n, dtype=int))

    gt_data, pred_data, mi = [], [], []
    for i in range(n):
      gt, gt_valid = _layer0_depth_gt(depth_peel[i], args.layer)
      pred, pred_valid = _layer0_depth_pred(points[i], args.layer)
      both = gt_valid & pred_valid & np.isfinite(pred) & (gt > 0)
      if both.sum() < 50:
        continue
      gv, pv = gt[both], pred[both]
      if args.align != "none":
        s, t = _align(pv, gv, args.align)
        pv = s * pv + t
      if len(gv) > args.max_points:
        idx = rng.choice(len(gv), args.max_points, replace=False)
        gv, pv = gv[idx], pv[idx]
      gt_data.append(gv)
      pred_data.append(pv)
      mi.append(int(mesh_index[i]))

  mi = np.array(mi)
  meshes = sorted(set(mi.tolist()))

  # x = mesh slot + a small fan-out offset per view within the mesh
  # (matches plot_mean_depth.py's layout).
  x = np.empty(len(mi), dtype=float)
  max_k = 1
  for mesh in meshes:
    sel = np.where(mi == mesh)[0]
    k = len(sel)
    max_k = max(max_k, k)
    off = (np.arange(k) - (k - 1) / 2) * (0.6 / max(k, 1))
    x[sel] = mesh + off
  spacing = 0.6 / max_k
  box_w = min(spacing * 0.75, 0.13)

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(max(9, len(meshes) * 1.3), 5.5))

  for j, mesh in enumerate(meshes):
    ax.axvspan(mesh - 0.5, mesh + 0.5, color="0.5" if j % 2 else "1.0",
               alpha=0.08, zorder=0)

  # GT and WT boxes sit on the exact same x (one view = one vertical line);
  # translucent fill + lines so both are legible where they overlap.
  def _style(bp, color, zorder):
    for key in ("whiskers", "caps"):
      for artist in bp[key]:
        artist.set_color(color)
        artist.set_alpha(0.5)
        artist.set_linewidth(1.6)
        artist.set_zorder(zorder)
    for artist in bp["medians"]:
      artist.set_color(color)
      artist.set_alpha(0.7)
      artist.set_linewidth(2.2)
      artist.set_zorder(zorder + 0.1)
    for patch in bp["boxes"]:
      patch.set_facecolor(color)
      patch.set_edgecolor(color)
      patch.set_alpha(0.35)
      patch.set_zorder(zorder)

  common = dict(positions=x, widths=box_w, patch_artist=True, showfliers=False)
  bp_gt = ax.boxplot(gt_data, **common)
  bp_wt = ax.boxplot(pred_data, **common)
  _style(bp_gt, "tab:blue", zorder=3)
  _style(bp_wt, "tab:red", zorder=4)

  ax.plot([], [], color="tab:blue", lw=6, alpha=0.35,
          label="render (GT)" + (" [aligned]" if args.align != "none" else ""))
  ax.plot([], [], color="tab:red", lw=6, alpha=0.35,
          label="WT prediction" + (" [aligned]" if args.align != "none" else " [raw]"))

  ax.set_yscale("log")
  ax.set_xticks(meshes)
  ax.set_xticklabels([f"mesh {m}\n({int((mi == m).sum())} views)" for m in meshes])
  ax.set_xlim(meshes[0] - 0.5, meshes[-1] + 0.5)
  ax.set_ylabel("per-pixel surface depth")
  align_note = "raw -- no scale alignment" if args.align == "none" else f"aligned per view ({args.align})"
  ax.set_title(f"WT vs render depth distribution, {len(mi)} views, {len(meshes)} meshes ({align_note})")
  ax.grid(alpha=0.3, axis="y", which="both")
  ax.legend(loc="best", framealpha=0.9)
  fig.tight_layout()

  out = args.out or f"{args.wt_h5}.depthbox.png"
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[fig] {out}")


if __name__ == "__main__":
  main()
