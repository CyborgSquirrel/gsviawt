#!/usr/bin/env python3
"""Box plot (+ the individual per-view points jittered on top) of the raw
Chamfer distance between a render_objaverse render's unprojected depth-peel
point cloud and World Tracing's predicted XYZ, over every view in the file.

One box for the whole predicted cloud ("all"), then one per depth-peel
layer. Chamfer is symmetric (mean_a min_b|a-b| + mean_b min_a|a-b|), raw --
no alignment -- and subsampled per side for speed (see plot_view_panels).

    python plot_chamfer_box.py bla/obj_rand40.h5 bla/obj_rand40.h5.wt.h5

Every run also writes a `<out>.csv` with the per-view / per-layer numbers.
To only re-draw the figure (e.g. after a style change) without recomputing
any Chamfer distances, pass that CSV back:

    python plot_chamfer_box.py --replot bla/obj_rand40.h5.wt.h5.chamferbox.csv
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _layer0_depth_gt
from plot_view_panels import _chamfer, _unproject, _xyz_cloud
from util import intrinsics_name


def _compute(render_h5, wt_h5, cap):
  """Returns (cd_all (n,), cd_layer (n, n_layers))."""
  with h5py.File(render_h5, "r") as rf, h5py.File(wt_h5, "r") as wf:
    n = min(rf["depth_peel"].shape[0], wf["points"].shape[0])
    n_layers = min(rf["depth_peel"].shape[3], wf["points"].shape[3])
    K_ds = rf[intrinsics_name(rf, "depth")]

    cd_all = np.full(n, np.nan)
    cd_layer = np.full((n, n_layers), np.nan)
    for v in range(n):
      dp = rf["depth_peel"][v]
      pts = wf["points"][v]
      K = K_ds[v]
      gt_all = np.concatenate([_unproject(dp[..., li], K) for li in range(n_layers)], axis=0)
      cd_all[v] = _chamfer(gt_all, _xyz_cloud(pts), cap=cap)
      for li in range(n_layers):
        cd_layer[v, li] = _chamfer(_unproject(dp[..., li], K),
                                   _xyz_cloud(pts[:, :, li, :]), cap=cap)
      print(f"  view {v:>2}: all={cd_all[v]:.4f}")
  return cd_all, cd_layer


def _read_csv(path):
  rows = np.genfromtxt(path, delimiter=",", names=True)
  cols = [c for c in rows.dtype.names if c.startswith("cd_L")]
  cd_all = np.atleast_1d(rows["cd_all"])
  cd_layer = np.column_stack([np.atleast_1d(rows[c]) for c in cols])
  return cd_all, cd_layer


def _write_csv(path, cd_all, cd_layer):
  n, n_layers = cd_layer.shape
  with open(path, "w") as f:
    f.write("view,cd_all," + ",".join(f"cd_L{li}" for li in range(n_layers)) + "\n")
    for v in range(n):
      cells = [f"{cd_all[v]:.6f}"] + [
        ("" if not np.isfinite(cd_layer[v, li]) else f"{cd_layer[v, li]:.6f}")
        for li in range(n_layers)]
      f.write(f"{v}," + ",".join(cells) + "\n")
  print(f"[csv] {path}")


def _draw(groups, n, out):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(1.4 * len(groups) + 2, 6))
  rng = np.random.default_rng(0)
  positions = np.arange(len(groups))

  ax.boxplot([g[1] for g in groups], positions=positions, widths=0.55,
             showfliers=False, medianprops=dict(color="tab:red", lw=2),
             boxprops=dict(color="0.35"), whiskerprops=dict(color="0.35"),
             capprops=dict(color="0.35"))
  for x, (_, vals) in zip(positions, groups):
    if len(vals) == 0:
      continue
    jitter = rng.uniform(-0.16, 0.16, len(vals))
    # small dots -- a 500-view run puts hundreds of points on each box
    ax.scatter(x + jitter, vals, s=4, color="tab:blue", alpha=0.35,
               edgecolors="none", zorder=3)
    ax.text(x, vals.max(), f"  med {np.median(vals):.3f}\n  n={len(vals)}",
            fontsize=8, va="bottom", ha="center", color="0.3")

  ax.set_xticks(positions)
  ax.set_xticklabels([g[0] for g in groups])
  ax.set_ylabel("Chamfer distance (raw, WT metric units)")
  ax.set_ylim(0, max(v.max() for _, v in groups if len(v)) * 1.15)
  ax.set_title(f"render vs WT Chamfer distance, {n} views (box + per-view points)")
  ax.grid(alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[fig] {out}")


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5", nargs="?")
  p.add_argument("wt_h5", nargs="?")
  p.add_argument("--replot", metavar="CSV",
                 help="skip all Chamfer computation; re-draw the figure from a "
                      "CSV a previous run wrote")
  p.add_argument("--cap", type=int, default=40000, help="points per side for Chamfer (default 40000)")
  p.add_argument("--out", default=None, help="default: <wt_h5>.chamferbox.png "
                 "(or the CSV path with .png for --replot)")
  args = p.parse_args()

  if args.replot:
    cd_all, cd_layer = _read_csv(args.replot)
    out = args.out or (args.replot[:-4] if args.replot.endswith(".csv")
                       else args.replot) + ".png"
    write_csv = False
  else:
    if not (args.render_h5 and args.wt_h5):
      p.error("render_h5 and wt_h5 are required unless --replot is given")
    cd_all, cd_layer = _compute(args.render_h5, args.wt_h5, args.cap)
    out = args.out or f"{args.wt_h5}.chamferbox.png"
    write_csv = True

  n, n_layers = cd_layer.shape
  groups = [("all", cd_all[np.isfinite(cd_all)])] + [
    (f"L{li}", cd_layer[np.isfinite(cd_layer[:, li]), li]) for li in range(n_layers)]

  _draw(groups, n, out)
  if write_csv:
    _write_csv(out.rsplit(".", 1)[0] + ".csv", cd_all, cd_layer)

  a = groups[0][1]
  print(f"\nall-layers Chamfer: median {np.median(a):.4f}  mean {np.mean(a):.4f}  "
        f"min {np.min(a):.4f}  max {np.max(a):.4f}")


if __name__ == "__main__":
  main()
