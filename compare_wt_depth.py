#!/usr/bin/env python3
"""Compare the surface-layer (layer 0) depth of a capture_turntable.py render
against World Tracing's prediction (wt_infer_layers.py output).

Inputs
------
render.h5  -- has `depth_peel` (N, H, W, L) float32, -1.0 = no hit. Layer 0 is
              planar Z in OpenCV camera space (X right, Y down, Z forward): see
              debug_pointcloud.unproject_depth_peel, where `points_cv.z` is
              exactly `depth_peel` (K^-1 @ [u, v, 1] has z == 1, then scaled by
              depth).
wt.h5      -- has `points` (N, H, W, L, 3) float32, NaN = invalid. Camera-space
              XYZ in the same OpenCV convention. The layer-0 depth map is
              `points[..., 0, 2]`.

Both share the same image pixel grid -- wt_infer_layers.py runs without
`--center-crop` at the render's own resolution -- so pixel (v, u) refers to the
same surface in both maps.

Alignment
---------
World Tracing predicts object geometry only up to an unknown global scale (and
under its own assumed intrinsics, not the render camera's), and the render's
absolute units are arbitrary Blender units that vary wildly per mesh. So the
prediction is aligned to the GT on the shared valid pixels before scoring:

  --align median  (default) : s = median(gt / pred)                  scale only, robust
  --align scale             : s = argmin || s*pred - gt ||           scale only, least squares
  --align affine            : (s, t) = argmin || s*pred + t - gt ||  scale + shift
  --align none              : compare raw

Metrics (per view, and mean over views), computed on the intersection of the
two valid masks:

  n        shared valid pixel count (and its fraction of the GT mask)
  scale    fitted alignment factor (shift too, for --align affine)
  AbsRel   mean(|pred - gt| / gt)
  RMSE     sqrt(mean((pred - gt)^2))          -- in GT units, after alignment
  RMSElog  sqrt(mean((ln pred - ln gt)^2))
  d1/d2/d3 fraction with max(pred/gt, gt/pred) < 1.25 / 1.25^2 / 1.25^3

Usage
-----
    # every shared view, metrics table + summary plot
    python compare_wt_depth.py bla/lite.h5 bla/lite.h5.wt.h5

    # one view, detailed 4-panel figure (RGB | GT | aligned pred | error)
    python compare_wt_depth.py bla/lite.h5 bla/lite.h5.wt.h5 --index 7

    # dump per-view metrics
    python compare_wt_depth.py bla/lite.h5 bla/lite.h5.wt.h5 --csv /tmp/depthcmp.csv
"""

from argparse import ArgumentParser

import h5py
import numpy as np


def _align(pred, gt, mode):
  """Return (scale, shift) mapping pred -> gt on the given 1-D valid samples."""
  if mode == "none":
    return 1.0, 0.0
  if mode == "median":
    return float(np.median(gt / pred)), 0.0
  if mode == "scale":
    return float(np.dot(pred, gt) / np.dot(pred, pred)), 0.0
  if mode == "affine":
    # least squares [pred, 1] @ [s, t] ~= gt
    a = np.stack([pred, np.ones_like(pred)], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, gt, rcond=None)
    return float(s), float(t)
  raise ValueError(f"unknown align mode {mode!r}")


def _layer0_depth_gt(depth_peel_view, layer):
  """depth_peel_view: (H, W, L). Returns (depth HxW float32, valid HxW bool)."""
  d = depth_peel_view[..., layer].astype(np.float32)
  return d, d >= 0.0


def _layer0_depth_pred(points_view, layer):
  """points_view: (H, W, L, 3). Returns (Z HxW float32, valid HxW bool)."""
  xyz = points_view[:, :, layer, :].astype(np.float32)
  valid = ~np.isnan(xyz).any(axis=-1)
  z = np.where(valid, xyz[..., 2], np.nan).astype(np.float32)
  return z, valid


def compare_view(depth_peel_view, points_view, layer, align):
  gt, gt_valid = _layer0_depth_gt(depth_peel_view, layer)
  pred, pred_valid = _layer0_depth_pred(points_view, layer)

  both = gt_valid & pred_valid & np.isfinite(pred) & (gt > 0)
  n = int(both.sum())
  out = {
    "n": n,
    "gt_mask_px": int(gt_valid.sum()),
    "pred_mask_px": int(pred_valid.sum()),
    "iou": float((gt_valid & pred_valid).sum() / max((gt_valid | pred_valid).sum(), 1)),
  }
  if n < 50:
    out.update(scale=np.nan, shift=np.nan, abs_rel=np.nan, rmse=np.nan,
               rmse_log=np.nan, d1=np.nan, d2=np.nan, d3=np.nan,
               gt_mean_depth=np.nan, pred_mean_depth=np.nan)
    return out, gt, pred, both, (np.nan, np.nan)

  gv, pv = gt[both], pred[both]
  s, t = _align(pv, gv, align)
  pred_al = s * pred + t
  pa = pred_al[both]

  # scale/shift can push a few samples <= 0; drop them from ratio-based metrics
  pos = pa > 0
  ratio = np.maximum(pa[pos] / gv[pos], gv[pos] / pa[pos])

  out.update(
    scale=s,
    shift=t,
    abs_rel=float(np.mean(np.abs(pa - gv) / gv)),
    rmse=float(np.sqrt(np.mean((pa - gv) ** 2))),
    rmse_log=float(np.sqrt(np.mean((np.log(pa[pos]) - np.log(gv[pos])) ** 2))),
    d1=float(np.mean(ratio < 1.25)),
    d2=float(np.mean(ratio < 1.25 ** 2)),
    d3=float(np.mean(ratio < 1.25 ** 3)),
    # mean depth over the shared valid pixels, BEFORE alignment: GT in
    # Blender units vs WT's raw (normalized) output. The gap between them
    # is what `fit scale` corrects -- shown so the raw scale mismatch per
    # view/mesh is visible.
    gt_mean_depth=float(gv.mean()),
    pred_mean_depth=float(pv.mean()),
  )
  return out, gt, pred_al, both, (s, t)


def _detail_figure(rgb, gt, pred_al, both, m, out_path, title):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  vis = both
  lo, hi = np.percentile(gt[vis], [2, 98])

  def masked(a):
    return np.ma.masked_where(~vis, a)

  fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
  ax[0].imshow(rgb)
  ax[0].set_title("model input RGB")
  im1 = ax[1].imshow(masked(gt), cmap="turbo", vmin=lo, vmax=hi)
  ax[1].set_title("render depth (layer 0)")
  fig.colorbar(im1, ax=ax[1], fraction=0.046)
  im2 = ax[2].imshow(masked(pred_al), cmap="turbo", vmin=lo, vmax=hi)
  ax[2].set_title(f"WT depth, aligned (s={m['scale']:.4g})")
  fig.colorbar(im2, ax=ax[2], fraction=0.046)
  err = pred_al - gt
  elim = np.percentile(np.abs(err[vis]), 95) or 1.0
  im3 = ax[3].imshow(masked(err), cmap="RdBu", vmin=-elim, vmax=elim)
  ax[3].set_title("pred - gt")
  fig.colorbar(im3, ax=ax[3], fraction=0.046)
  for a in ax:
    a.set_xticks([])
    a.set_yticks([])
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=110)
  plt.close(fig)


def _summary_figure(rows, mesh_index, out_path, align):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  idx = np.array([r["index"] for r in rows])
  absrel = np.array([r["abs_rel"] for r in rows])
  rmse = np.array([r["rmse"] for r in rows])
  scale = np.array([r["scale"] for r in rows])
  shift = np.array([r["shift"] for r in rows])
  gt_md = np.array([r["gt_mean_depth"] for r in rows])
  pred_md = np.array([r["pred_mean_depth"] for r in rows])
  mi = np.array([mesh_index[i] for i in idx]) if mesh_index is not None else np.zeros_like(idx)

  panels = [
    (absrel, "AbsRel", "linear"),
    (rmse, "RMSE (gt units)", "linear"),
    (scale, f"fit scale ({align})", "log" if np.all(scale > 0) else "linear"),
    (shift, f"fit shift ({align})", "linear"),
  ]
  fig, ax = plt.subplots(len(panels) + 1, 1, figsize=(max(8, len(idx) * 0.28), 13), sharex=True)
  for a, (y, name, yscale) in zip(ax, panels):
    a.scatter(idx, y, c=mi, cmap="tab10", s=36)
    a.set_ylabel(name)
    a.set_yscale(yscale)
    a.grid(alpha=0.3, which="both")
    if name.startswith("fit shift") and np.allclose(shift, 0):
      a.text(0.99, 0.9, "identically 0 (shift only fitted for --align affine)",
             transform=a.transAxes, ha="right", va="top", fontsize=8, color="0.4")

  # mean depth over shared valid pixels, BEFORE alignment: render (GT) in
  # Blender units vs WT's raw normalized output. The vertical gap per view
  # is the raw scale mismatch that `fit scale` corrects.
  a = ax[-1]
  a.vlines(idx, np.minimum(gt_md, pred_md), np.maximum(gt_md, pred_md), color="0.7", lw=1, zorder=1)
  a.scatter(idx, gt_md, marker="o", s=40, facecolors="none", edgecolors="tab:blue",
            linewidths=1.5, label="render (GT)", zorder=2)
  a.scatter(idx, pred_md, marker="x", s=36, color="tab:red", linewidths=1.5,
            label="WT (raw, pre-align)", zorder=3)
  a.set_ylabel("mean depth\n(shared px, raw)")
  a.set_yscale("log" if np.all(np.r_[gt_md, pred_md][np.isfinite(np.r_[gt_md, pred_md])] > 0) else "linear")
  a.grid(alpha=0.3, which="both")
  a.legend(loc="best", fontsize=8)

  ax[-1].set_xlabel("view index (color = mesh_index)")
  ax[0].set_title(f"WT depth vs render depth -- {len(idx)} views, align={align}")
  fig.tight_layout()
  fig.savefig(out_path, dpi=110)
  plt.close(fig)


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("--index", type=int, default=None, help="Compare just this view (default: every shared view)")
  p.add_argument("--layer", type=int, default=0, help="Depth-peel / points layer to compare (default 0, the surface)")
  p.add_argument("--align", choices=["median", "scale", "affine", "none"], default="median")
  p.add_argument("--out", default=None, help="Figure path (default: <wt_h5>.depthcmp[.viewN].png; '-' to skip)")
  p.add_argument("--csv", default=None, help="Also write per-view metrics as CSV")
  args = p.parse_args()

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    depth_peel = rf["depth_peel"]
    points = wf["points"]
    rgb_all = wf["images"]
    mesh_index = rf["mesh_index"][:] if "mesh_index" in rf else None

    n_render, n_wt = depth_peel.shape[0], points.shape[0]
    n = min(n_render, n_wt)
    if n_render != n_wt:
      print(f"[warn] view count differs: render={n_render}, wt={n_wt}; comparing first {n}")
    if depth_peel.shape[1:3] != points.shape[1:3]:
      raise SystemExit(
        f"resolution mismatch: render depth {depth_peel.shape[1:3]} vs "
        f"wt points {points.shape[1:3]} -- rerun wt_infer_layers.py without "
        "--center-crop at the render resolution")
    if args.layer >= min(depth_peel.shape[3], points.shape[3]):
      raise SystemExit(f"--layer {args.layer} out of range")

    indices = [args.index] if args.index is not None else list(range(n))
    rows = []
    single = None
    for i in indices:
      m, gt, pred_al, both, _ = compare_view(depth_peel[i], points[i], args.layer, args.align)
      m["index"] = i
      m["mesh"] = int(mesh_index[i]) if mesh_index is not None else -1
      rows.append(m)
      if args.index is not None:
        single = (rgb_all[i], gt, pred_al, both, m)

  # ---- report ----
  hdr = ["view", "mesh", "n", "gtpx", "scale", "shift", "AbsRel", "RMSE", "RMSElog", "d1", "d2", "d3"]
  print("  ".join(f"{h:>8}" for h in hdr))
  for r in rows:
    print("  ".join(f"{v:>8}" for v in [
      r["index"], r["mesh"], r["n"], r["gt_mask_px"],
      f"{r['scale']:.4g}", f"{r['shift']:.3g}",
      f"{r['abs_rel']:.4f}", f"{r['rmse']:.4g}", f"{r['rmse_log']:.4f}",
      f"{r['d1']:.3f}", f"{r['d2']:.3f}", f"{r['d3']:.3f}",
    ]))

  ok = [r for r in rows if np.isfinite(r["abs_rel"])]
  if ok:
    def mean(k):
      return float(np.mean([r[k] for r in ok]))
    print("-" * 96)
    print(f"mean over {len(ok)} views:  AbsRel={mean('abs_rel'):.4f}  "
          f"RMSE={mean('rmse'):.4g}  RMSElog={mean('rmse_log'):.4f}  "
          f"d1={mean('d1'):.3f}  d2={mean('d2'):.3f}  d3={mean('d3'):.3f}  "
          f"| median fit scale={np.median([r['scale'] for r in ok]):.4g}")

  if args.csv:
    import csv

    with open(args.csv, "w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["index", "mesh", "n", "gt_mask_px", "pred_mask_px", "iou",
                  "scale", "shift", "abs_rel", "rmse", "rmse_log", "d1", "d2", "d3"])
      for r in rows:
        w.writerow([r["index"], r["mesh"], r["n"], r["gt_mask_px"], r["pred_mask_px"],
                    f"{r['iou']:.4f}", r["scale"], r["shift"], r["abs_rel"], r["rmse"],
                    r["rmse_log"], r["d1"], r["d2"], r["d3"]])
    print(f"[csv] {args.csv}")

  if args.out == "-":
    return
  if args.index is not None:
    out = args.out or f"{args.wt_h5}.depthcmp.view{args.index}.png"
    rgb, gt, pred_al, both, m = single
    if np.isfinite(m["abs_rel"]):
      _detail_figure(rgb, gt, pred_al, both, m,
                     out, f"view {args.index} (mesh {m['mesh']}) -- AbsRel {m['abs_rel']:.3f}, d1 {m['d1']:.2f}")
      print(f"[fig] {out}")
    else:
      print("[fig] skipped -- too few overlapping pixels")
  else:
    out = args.out or f"{args.wt_h5}.depthcmp.png"
    if ok:
      _summary_figure(ok, mesh_index, out, args.align)
      print(f"[fig] {out}")


if __name__ == "__main__":
  main()
