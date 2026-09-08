#!/usr/bin/env python3
"""Compare candidate depth-anchor statistics for matching a render_objaverse.py
render's scale to a World Tracing prediction.

WT outputs depth in its own canonical frame (surface ~1.9 regardless of the
object's true size); our render is in arbitrary units. To align them we pick
one statistic C of the render's depth, declare `C == target` (default 1.904 =
wt.inference.XYZ_MEAN[2]), and scale the render by target/C. This asks which
statistic WT's training used: the right one leaves a single global leftover
factor across all views; the wrong one leaves per-view scatter.

For each candidate C and view i:
    f_i = (raw_render_L0mean_i * target / C_i) / WT_L0mean_i
i.e. the factor still needed to make WT match the C-normalised render. A good
anchor -> f_i tight around some constant (low coefficient of variation);
`mean f` is a free global correction, `CoV(f)` is the irreducible per-view
error.

    python plot_anchor_sweep.py bla/lite_blackbg.h5 bla/lite_blackbg.h5.wt.h5
"""

from argparse import ArgumentParser

import h5py
import numpy as np

from compare_wt_depth import _layer0_depth_gt, _layer0_depth_pred


def _candidates(layers):
  """layers: list of 1-D raw-depth arrays, layers[0] = visible surface.
  Returns {name: value}."""
  d0 = layers[0]
  alld = np.concatenate(layers) if len(layers) > 1 else d0
  logd0 = np.log(d0[d0 > 0])
  lo, hi = np.percentile(d0, [10, 90])
  trimmed = d0[(d0 >= lo) & (d0 <= hi)]
  return {
    "L0 mean": d0.mean(),
    "L0 median": np.median(d0),
    "L0 geo-mean": float(np.exp(logd0.mean())),
    "L0 trim mean\n(10-90)": trimmed.mean() if len(trimmed) else d0.mean(),
    "L0 p5\n(near)": np.percentile(d0, 5),
    "L0 min\n(nearest)": d0.min(),
    "L0 range mid\n(min+max)/2": 0.5 * (d0.min() + d0.max()),
    "all-layer\nmean": alld.mean(),
    "all-layer\nmedian": np.median(alld),
    "all-layer\nrange mid": 0.5 * (alld.min() + alld.max()),
  }


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--target", type=float, default=1.904)
  p.add_argument("--out", default=None, help="default: <wt_h5>.anchorsweep.png")
  args = p.parse_args()

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    depth_peel = rf["depth_peel"]
    points = wf["points"]
    scale = rf["depth_scale"][:] if "depth_scale" in rf else np.ones(depth_peel.shape[0])
    mesh_index = rf["mesh_index"][:] if "mesh_index" in rf else None
    n = min(depth_peel.shape[0], points.shape[0])
    L = depth_peel.shape[3]

    rows = []          # per view: {cand_name: f_i}
    mi = []
    for i in range(n):
      dp = depth_peel[i]
      raw_layers = []
      for l in range(L):
        d = dp[..., l]
        d = d[d >= 0] / scale[i]
        if len(d) >= 50:
          raw_layers.append(d)
      if not raw_layers:
        continue
      gt0 = raw_layers[0]

      predz, pred_valid = _layer0_depth_pred(points[i], 0)
      wt0 = predz[pred_valid & np.isfinite(predz)]
      if len(wt0) < 50:
        continue
      wt0_mean = wt0.mean()

      cands = _candidates(raw_layers)
      r = {name: (gt0.mean() * args.target / C) / wt0_mean for name, C in cands.items()}
      rows.append(r)
      mi.append(int(mesh_index[i]) if mesh_index is not None else i)

  names = list(rows[0].keys())
  mi = np.array(mi)
  F = {name: np.array([r[name] for r in rows]) for name in names}
  cov = {name: F[name].std() / F[name].mean() for name in names}
  # within-mesh CoV
  wm_cov = {}
  for name in names:
    cs = []
    for m in np.unique(mi):
      v = F[name][mi == m]
      if len(v) >= 2:
        cs.append(v.std() / v.mean())
    wm_cov[name] = float(np.mean(cs)) if cs else np.nan

  order = sorted(names, key=lambda k: cov[k])

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.15), 6))
  cmap = plt.get_cmap("tab10")
  pos = np.arange(len(order))

  bp = ax.boxplot([F[k] for k in order], positions=pos, widths=0.55,
                  patch_artist=True, showfliers=False, zorder=2)
  for i, box in enumerate(bp["boxes"]):
    win = i == 0
    box.set(facecolor="tab:green" if win else "0.8", edgecolor="0.3", alpha=0.5 if win else 0.35)
  for med in bp["medians"]:
    med.set(color="k", linewidth=1.5)

  for i, k in enumerate(order):
    jit = (np.random.default_rng(i).random(len(F[k])) - 0.5) * 0.28
    ax.scatter(pos[i] + jit, F[k], c=[cmap(m % 10) for m in mi], s=14, alpha=0.7, zorder=3)
    ax.text(pos[i], ax.get_ylim()[0], f"CoV\n{cov[k]:.3f}\n({wm_cov[k]:.3f})",
            ha="center", va="bottom", fontsize=7,
            color="darkgreen" if i == 0 else "0.35")

  ax.axhline(1.0, color="0.5", ls=":", lw=1)
  ax.set_xticks(pos)
  ax.set_xticklabels(order, fontsize=8)
  ax.set_ylabel(f"leftover factor f  (anchor stat == {args.target})")
  ax.set_title("Anchor-statistic sweep: f per view for each candidate, "
               "sorted by scatter (lowest = best anchor). "
               "CoV = all-views (within-mesh). Point colour = mesh.")
  ax.grid(alpha=0.3, axis="y")
  fig.tight_layout()

  out = args.out or f"{args.wt_h5}.anchorsweep.png"
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[fig] {out}")
  print(f"\n{'anchor':<24} {'mean f':>8} {'CoV all':>9} {'CoV/mesh':>9}")
  for k in order:
    print(f"{k.replace(chr(10),' '):<24} {F[k].mean():>8.3f} {cov[k]:>9.3f} {wm_cov[k]:>9.3f}")


if __name__ == "__main__":
  main()
