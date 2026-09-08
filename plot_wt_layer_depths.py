#!/usr/bin/env python3
"""Boxplot of per-pixel depth for each depth-peel layer of a
wt_infer_layers.py prediction (`points[..., L, 2]`).

Layer 0 is the visible surface; deeper layers are the geometry occluded
behind it. Shows how World Tracing stacks its layers in depth and how the
spread changes with occlusion depth.

Default: one box per layer, per view (grouped in mesh bands when --render
gives a mesh_index). --pooled collapses every view into one box per layer.

    python plot_wt_layer_depths.py bla/lite.h5.wt.h5 --render bla/lite.h5
    python plot_wt_layer_depths.py bla/lite.h5.wt.h5 --render bla/lite.h5 --align median
    python plot_wt_layer_depths.py bla/lite.h5.wt.h5 --pooled
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _align, _layer0_depth_gt, _layer0_depth_pred


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("wt_h5")
  p.add_argument("--render", default=None,
                 help="capture_turntable render h5; overlays GT boxes per layer + mesh bands")
  p.add_argument("--pooled", action="store_true",
                 help="one box per layer over all views (default: per view)")
  p.add_argument("--align", choices=["none", "median", "scale", "affine", "mad"], default="none",
                 help="align WT to GT per (view, layer) before boxing (needs --render)")
  p.add_argument("--max-points", type=int, default=40000,
                 help="cap on pixels per box (subsampled); quantiles stay stable well below")
  p.add_argument("--out", default=None, help="default: <wt_h5>.layerbox.png")
  args = p.parse_args()
  if args.align != "none" and args.render is None:
    p.error("--align needs --render")

  rng = np.random.default_rng(0)

  with h5py.File(args.wt_h5, "r") as wf:
    points = wf["points"]
    n, _, _, num_layers, _ = points.shape
    rf = h5py.File(args.render, "r") if args.render else None
    depth_peel = rf["depth_peel"] if rf else None
    mesh_index = (rf["mesh_index"][:] if rf is not None and "mesh_index" in rf else None)
    if depth_peel is not None:
      n = min(n, depth_peel.shape[0])

    # wt[v][L] / gt[v][L] -> 1-D pixel-depth array (possibly empty)
    wt, gt = [], []
    for i in range(n):
      pts_view = points[i]
      dp_view = depth_peel[i] if depth_peel is not None else None
      wt_v, gt_v = [], []
      for L in range(num_layers):
        pred, pred_valid = _layer0_depth_pred(pts_view, L)
        if dp_view is not None:
          g, g_valid = _layer0_depth_gt(dp_view, L)
          both = g_valid & pred_valid & np.isfinite(pred) & (g > 0)
          gv, pv = g[both], pred[both]
          if args.align != "none" and len(gv) >= 50:
            s, t = _align(pv, gv, args.align)
            pv = s * pv + t
        else:
          gv = np.empty(0, np.float32)
          pv = pred[pred_valid & np.isfinite(pred)]
        wt_v.append(pv)
        gt_v.append(gv)
      wt.append(wt_v)
      gt.append(gt_v)
    if rf:
      rf.close()

  def _sub(a, cap):
    if len(a) > cap:
      a = a[rng.choice(len(a), cap, replace=False)]
    return a

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  cmap = plt.get_cmap("viridis")
  lcol = [cmap(L / max(num_layers - 1, 1)) for L in range(num_layers)]

  def _style(bp, colors, fill_a, line_a, z):
    colors = colors if isinstance(colors, list) else [colors] * len(bp["boxes"])
    for i, box in enumerate(bp["boxes"]):
      box.set(facecolor=colors[i], edgecolor=colors[i], alpha=fill_a, zorder=z)
    for i, med in enumerate(bp["medians"]):
      med.set(color="k", alpha=0.7, linewidth=1.4, zorder=z + 0.1)
    for key in ("whiskers", "caps"):
      for i, art in enumerate(bp[key]):
        art.set(color=colors[i // 2], alpha=line_a, linewidth=1.2, zorder=z)

  # -------- pooled --------
  if args.pooled:
    wt_boxes = [_sub(np.concatenate([wt[v][L] for v in range(n) if len(wt[v][L])] or [np.array([np.nan])]),
                     args.max_points * 8) for L in range(num_layers)]
    fig, ax = plt.subplots(figsize=(max(7, num_layers * 1.25), 5))
    pos = np.arange(num_layers, dtype=float)
    if args.render:
      gt_boxes = [_sub(np.concatenate([gt[v][L] for v in range(n) if len(gt[v][L])] or [np.array([np.nan])]),
                       args.max_points * 8) for L in range(num_layers)]
      _style(ax.boxplot(gt_boxes, positions=pos, widths=0.5, patch_artist=True, showfliers=False),
             "tab:blue", 0.28, 0.5, z=3)
      _style(ax.boxplot(wt_boxes, positions=pos, widths=0.5, patch_artist=True, showfliers=False),
             "tab:red", 0.34, 0.8, z=4)
      ax.plot([], [], color="tab:blue", lw=6, alpha=0.4, label="render (GT)")
      ax.plot([], [], color="tab:red", lw=6, alpha=0.5,
              label="WT" + (f" [{args.align}-aligned]" if args.align != "none" else " [raw]"))
      ax.legend(loc="best")
    else:
      _style(ax.boxplot(wt_boxes, positions=pos, widths=0.62, patch_artist=True, showfliers=False),
             lcol, 0.75, 0.9, z=3)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"layer {L}" for L in range(num_layers)])
    scope = f"{n} views pooled"

  # -------- per view --------
  else:
    slot = 0.82
    step = slot / num_layers
    bw = step * 0.8
    wt_data, wt_pos, wt_lyr = [], [], []
    gt_data, gt_pos = [], []
    for v in range(n):
      for L in range(num_layers):
        x = v + (L - (num_layers - 1) / 2) * step
        if len(wt[v][L]) >= 20:
          wt_data.append(_sub(wt[v][L], args.max_points))
          wt_pos.append(x)
          wt_lyr.append(L)
        if args.render and len(gt[v][L]) >= 20:
          gt_data.append(_sub(gt[v][L], args.max_points))
          gt_pos.append(x)

    fig, ax = plt.subplots(figsize=(max(11, n * 0.82), 5.6))

    # mesh (or 4-view) bands
    grp = mesh_index[:n] if mesh_index is not None else (np.arange(n) // 4)
    prev, start = grp[0], 0
    for idx in range(1, n + 1):
      if idx == n or grp[idx] != prev:
        ax.axvspan(start - 0.5, idx - 0.5,
                   color="0.5" if int(prev) % 2 else "1.0", alpha=0.07, zorder=0)
        if idx < n:
          prev, start = grp[idx], idx

    if args.render and gt_data:
      bp = ax.boxplot(gt_data, positions=gt_pos, widths=bw, patch_artist=True, showfliers=False)
      for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor="0.45", linewidth=0.8, zorder=2)
      for art in bp["whiskers"] + bp["caps"]:
        art.set(color="0.45", linewidth=0.6, zorder=2)
      for med in bp["medians"]:
        med.set(color="0.25", linewidth=0.9, zorder=2)

    bp = ax.boxplot(wt_data, positions=wt_pos, widths=bw, patch_artist=True, showfliers=False)
    for box, L in zip(bp["boxes"], wt_lyr):
      box.set(facecolor=lcol[L], edgecolor=lcol[L], alpha=0.7, zorder=4)
    lyr2 = list(np.repeat(wt_lyr, 2))  # 2 whiskers + 2 caps per box
    for art, L in zip(bp["whiskers"] + bp["caps"], lyr2 + lyr2):
      art.set(color=lcol[L], alpha=0.8, linewidth=0.9, zorder=4)
    for med in bp["medians"]:
      med.set(color="k", alpha=0.75, linewidth=1.0, zorder=4.1)

    ax.set_xticks(np.arange(n))
    if mesh_index is not None:
      ax.set_xticklabels([f"m{int(mesh_index[v])}" for v in range(n)], fontsize=7)
    else:
      ax.set_xticklabels([str(v) for v in range(n)], fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)
    handles = [plt.Line2D([], [], color=lcol[L], lw=6, alpha=0.7, label=f"WT layer {L}")
               for L in range(num_layers)]
    if args.render:
      handles.append(plt.Line2D([], [], color="0.45", lw=1.5, label="render GT (per layer)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=max(1, (num_layers + 1) // 2), framealpha=0.9)
    scope = f"per view, {n} views"

  ax.set_yscale("log")
  ax.set_ylabel("per-pixel depth")
  ax.grid(alpha=0.3, axis="y", which="both")
  align_note = "" if args.align == "none" else f", {args.align}-aligned"
  ax.set_title(f"WT per-layer depth distribution ({scope}{align_note})")
  fig.tight_layout()

  out = args.out or f"{args.wt_h5}.layerbox.png"
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[fig] {out}")


if __name__ == "__main__":
  main()
