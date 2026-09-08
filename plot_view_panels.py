#!/usr/bin/env python3
"""Per-view panel: RGB render | RGB WT-input | GT depth | aligned WT depth |
|delta|, for one or more views of a capture_turntable render vs its
wt_infer_layers.py prediction.

WT's layer-0 depth (`points[..., 0, 2]`) is aligned to the render's layer-0
`depth_peel` per view on the shared valid pixels (--align, default median)
before display, so the two depth panels and the delta are on the render's
scale.

    python plot_view_panels.py bla/lite_blackbg.h5 bla/lite_blackbg.h5.wt.h5 --views 1 8 12 24
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _align, _layer0_depth_gt, _layer0_depth_pred


def panel(rgb_r, rgb_w, gt, wt_al, both, title, out, wt_label="depth WT (aligned)"):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec

  plt.rcParams.update({"font.size": 16})

  gt_m = np.where(gt >= 0, gt, np.nan)
  wt_m = np.where(np.isfinite(wt_al), wt_al, np.nan)  # nan outside WT's mask
  delta = np.where(both, np.abs(gt - wt_al), np.nan)

  finite = np.concatenate([gt_m[np.isfinite(gt_m)], wt_al[both]])
  vmin, vmax = np.percentile(finite, [1, 99])
  dmax = np.nanpercentile(delta, 98) if np.isfinite(delta).any() else 1.0

  fig = plt.figure(figsize=(13, 10))
  gs = GridSpec(3, 3, height_ratios=[1, 1, 0.06], width_ratios=[1, 1, 1],
                hspace=0.32, wspace=0.14, figure=fig)
  ax_rgb1 = fig.add_subplot(gs[0, 0])
  ax_rgb2 = fig.add_subplot(gs[0, 1])
  ax_delta = fig.add_subplot(gs[0:2, 2])
  ax_d1 = fig.add_subplot(gs[1, 0])
  ax_d2 = fig.add_subplot(gs[1, 1])
  cax_d = fig.add_subplot(gs[2, 0:2])
  cax_delta = fig.add_subplot(gs[2, 2])

  def show(ax, img, t, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(t, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(True); s.set_color("black"); s.set_linewidth(1.5)
    return im

  show(ax_rgb1, rgb_r, "RGB render")
  show(ax_rgb2, rgb_w, "RGB WT input")
  im1 = show(ax_d1, gt_m, "depth GT", cmap="turbo", vmin=vmin, vmax=vmax)
  show(ax_d2, wt_m, wt_label, cmap="turbo", vmin=vmin, vmax=vmax)
  im_d = show(ax_delta, delta, "|delta|", cmap="turbo", vmin=0, vmax=dmax)

  fig.colorbar(im1, cax=cax_d, orientation="horizontal", label="depth")
  fig.colorbar(im_d, cax=cax_delta, orientation="horizontal", label="|delta|")
  fig.suptitle(title, y=0.99)
  fig.savefig(out, dpi=100, bbox_inches="tight")
  plt.close(fig)
  print(f"[fig] {out}")


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--views", type=int, nargs="+", required=True)
  p.add_argument("--layer", type=int, default=0)
  p.add_argument("--align", choices=["none", "median", "scale", "affine", "mad"], default="median")
  p.add_argument("--out-prefix", default=None,
                 help="default: <wt_h5>.panel  ->  <prefix>.viewN.png")
  args = p.parse_args()
  prefix = args.out_prefix or f"{args.wt_h5}.panel"

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    mesh_index = rf["mesh_index"][:] if "mesh_index" in rf else None
    for v in args.views:
      rgb_r = rf["images"][v]
      rgb_w = wf["images"][v]
      gt, gt_valid = _layer0_depth_gt(rf["depth_peel"][v], args.layer)
      pred, pred_valid = _layer0_depth_pred(wf["points"][v], args.layer)
      both = gt_valid & pred_valid & np.isfinite(pred) & (gt > 0)
      s, t = _align(pred[both], gt[both], args.align)
      wt_al = s * pred + t

      absrel = float(np.mean(np.abs(wt_al[both] - gt[both]) / gt[both]))
      m = f"mesh {int(mesh_index[v])}" if mesh_index is not None else ""
      title = (f"view {v}  {m}   align={args.align} (s={s:.3g}, t={t:.3g})   "
               f"AbsRel={absrel:.3f}   n={int(both.sum())}")
      wt_label = "depth WT (raw)" if args.align == "none" else f"depth WT ({args.align}-aligned)"
      panel(rgb_r, rgb_w, gt, wt_al, both, title, f"{prefix}.view{v}.png", wt_label=wt_label)


if __name__ == "__main__":
  main()
