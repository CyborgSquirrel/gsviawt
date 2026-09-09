#!/usr/bin/env python3
"""Per-view panel: RGB render + RGB WT-input on the first row, then one row
per depth-peel layer -- GT depth | WT depth | |delta| -- for one or more
views of a render_objaverse render vs its wt_infer_layers.py prediction.

Both depth panels are RAW by default (`--align none`): the point of these
views is to see how well the geometry we generate already matches what World
Tracing predicts, in absolute terms. Scale/shift alignment (`--align
median|scale|affine|mad`, fit on layer 0 and applied to every layer) fits
that mismatch away, so it's opt-in only -- use it to inspect residual
*shape* error after the scale is taken out.

All the depth panels (GT + WT, every layer) share one colour range; all the
|delta| panels share another.

    python plot_view_panels.py bla/obj_rand40.h5 bla/obj_rand40.h5.wt.h5 --views 1 8 12 24
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _align, _layer0_depth_gt, _layer0_depth_pred


def _mask(a, valid):
  return np.where(valid, a, np.nan)


def panel(rgb_r, rgb_w, layers, title, out, wt_label="depth WT (raw)"):
  """layers: list of dicts {idx, gt, gtv, wt, wtv, both, absrel, n}, one per
  depth-peel layer to draw (gt/wt/gtv/wtv/both are H x W arrays)."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec

  plt.rcParams.update({"font.size": 15})

  # shared ranges: depths over every GT+WT valid pixel of every layer,
  # deltas over every shared-valid pixel of every layer.
  depth_pool = np.concatenate(
    [L["gt"][L["gtv"] & (L["gt"] > 0)] for L in layers]
    + [L["wt"][L["wtv"]] for L in layers])
  delta_pool = np.concatenate([np.abs(L["gt"] - L["wt"])[L["both"]] for L in layers])
  vmin, vmax = np.percentile(depth_pool, [1, 99]) if depth_pool.size else (0.0, 1.0)
  dmax = float(np.percentile(delta_pool, 98)) if delta_pool.size else 1.0

  nL = len(layers)
  fig = plt.figure(figsize=(13, 2.0 + 3.3 * nL))
  gs = GridSpec(2 + nL, 3, height_ratios=[1] + [1] * nL + [0.08],
                width_ratios=[1, 1, 1], hspace=0.28, wspace=0.14,
                top=0.95, bottom=0.05, left=0.06, right=0.98, figure=fig)

  def show(ax, img, t=None, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if t:
      ax.set_title(t, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_visible(True); s.set_color("black"); s.set_linewidth(1.5)
    return im

  show(fig.add_subplot(gs[0, 0]), rgb_r, "RGB render")
  show(fig.add_subplot(gs[0, 1]), rgb_w, "RGB WT input")

  im_depth = im_delta = None
  for row, L in enumerate(layers, start=1):
    first = row == 1
    ax_gt = fig.add_subplot(gs[row, 0])
    ax_wt = fig.add_subplot(gs[row, 1])
    ax_dl = fig.add_subplot(gs[row, 2])
    im_depth = show(ax_gt, _mask(L["gt"], L["gtv"] & (L["gt"] > 0)),
                    "depth GT" if first else None, "turbo", vmin, vmax)
    show(ax_wt, _mask(L["wt"], L["wtv"]),
         wt_label if first else None, "turbo", vmin, vmax)
    im_delta = show(ax_dl, _mask(np.abs(L["gt"] - L["wt"]), L["both"]),
                    "|delta|" if first else None, "turbo", 0.0, dmax)
    ar = f"AbsRel {L['absrel']:.3f}" if np.isfinite(L["absrel"]) else "AbsRel --"
    ax_gt.set_ylabel(f"layer {L['idx']}\n{ar}   n={L['n']}", fontsize=12)

  fig.colorbar(im_depth, cax=fig.add_subplot(gs[1 + nL, 0:2]),
               orientation="horizontal", label="depth")
  fig.colorbar(im_delta, cax=fig.add_subplot(gs[1 + nL, 2]),
               orientation="horizontal", label="|delta|")
  fig.suptitle(title, y=0.985)
  fig.savefig(out, dpi=100)
  plt.close(fig)
  print(f"[fig] {out}")


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--views", type=int, nargs="+", required=True)
  p.add_argument("--layers", type=int, nargs="+", default=None,
                 help="Which depth-peel layers to show (default: every layer "
                      "that has data in GT or WT for the view)")
  p.add_argument("--align", choices=["none", "median", "scale", "affine", "mad"], default="none",
                 help="Scale/shift-fit WT onto GT (on layer 0, applied to all "
                      "layers) before display. Default none -- these views are "
                      "for seeing the raw match to WT.")
  p.add_argument("--out-prefix", default=None,
                 help="default: <wt_h5>.panel  ->  <prefix>.viewN.png")
  args = p.parse_args()
  prefix = args.out_prefix or f"{args.wt_h5}.panel"

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    mesh_index = rf["mesh_index"][:] if "mesh_index" in rf else None
    for v in args.views:
      dp = rf["depth_peel"][v]      # (H, W, L)
      pts = wf["points"][v]         # (H, W, L, 3)
      n_layers = min(dp.shape[2], pts.shape[2])

      # one alignment for the view, fit on layer 0's shared valid pixels
      gt0, gtv0 = _layer0_depth_gt(dp, 0)
      pr0, prv0 = _layer0_depth_pred(pts, 0)
      both0 = gtv0 & prv0 & np.isfinite(pr0) & (gt0 > 0)
      s, t = _align(pr0[both0], gt0[both0], args.align)

      layers = []
      for li in (args.layers if args.layers is not None else range(n_layers)):
        gt, gtv = _layer0_depth_gt(dp, li)
        pr, prv = _layer0_depth_pred(pts, li)
        wt = s * pr + t
        wtv = prv & np.isfinite(wt)
        both = gtv & wtv & (gt > 0)
        if not (gtv.any() or wtv.any()):
          continue  # layer empty in both -> skip
        absrel = (float(np.mean(np.abs(wt[both] - gt[both]) / gt[both]))
                  if both.any() else np.nan)
        layers.append(dict(idx=li, gt=gt, gtv=gtv, wt=wt, wtv=wtv, both=both,
                           absrel=absrel, n=int(both.sum())))
      if not layers:
        print(f"[skip] view {v}: no populated layers")
        continue

      m = f"mesh {int(mesh_index[v])}" if mesh_index is not None else ""
      title = (f"view {v}  {m}   align={args.align} (s={s:.3g}, t={t:.3g})   "
               f"{len(layers)} layer(s)")
      wt_label = "depth WT (raw)" if args.align == "none" else f"depth WT ({args.align}-aligned)"
      panel(rf["images"][v], wf["images"][v], layers, title,
            f"{prefix}.view{v}.png", wt_label=wt_label)


if __name__ == "__main__":
  main()
