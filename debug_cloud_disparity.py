#!/usr/bin/env python3
"""Colour two point clouds by how well they agree, point by point.

For one view you have two clouds in the same OpenCV camera frame:
  GT  -- the render's depth peel, unprojected with the render camera K.
  WT  -- World Tracing's predicted XYZ for that view (stored, already
         camera space).

Every GT point is coloured by the Euclidean distance to its nearest WT
point (pale = coincident, deep **blue** = far); every WT point by the
distance to its nearest GT point (pale = coincident, deep **red** = far).
So blue clumps are geometry the render has that WT missed, red clumps are
geometry WT invented, and pale regions are where the two agree. It's the
per-point form of the symmetric Chamfer distance that plot_chamfer_box.py
reduces to one number per view -- and mean(GT->WT) + mean(WT->GT) printed
here should match that number.

Raw, no alignment: a Z-only scale/shift isn't a valid 3-D transform, so
(like every other Chamfer tool here) the clouds are compared as-is.

    python debug_cloud_disparity.py bla/obj_rand500.h5 bla/obj_rand500.h5.wt.h5 400
    python debug_cloud_disparity.py render.h5 render.h5.wt.h5 7 --layers 0 1
    python debug_cloud_disparity.py render.h5 render.h5.wt.h5 7 -e glb -o /tmp/v7
"""

from argparse import ArgumentParser

import h5py
import numpy as np
import trimesh

from debug_pointcloud import extract_valid_points, unproject_depth_peel
from util import intrinsics_name


def _nn_dist(a, b):
  """(Na, 3), (Nb, 3) -> (Na,) nearest-neighbour Euclidean distance from every
  row of `a` into `b` (k=1, every point, no subsampling). Empty `b` -> +inf."""
  if len(a) == 0:
    return np.zeros(0, np.float64)
  if len(b) == 0:
    return np.full(len(a), np.inf)
  from scipy.spatial import cKDTree

  d, _ = cKDTree(b).query(a)
  return d


def _colors_by_distance(dist, cmap_name, lo, hi):
  """Colormap per-point distance `dist` with `cmap_name` (pale -> saturated).
  Returns (N, 4) uint8 RGBA. Mirrors debug_pointcloud.colors_by_depth."""
  import matplotlib

  t = np.clip((dist - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
  rgba = matplotlib.colormaps[cmap_name](np.nan_to_num(t, nan=1.0))
  return np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _stats(name, dist):
  """Print mean / median / p95 of the finite entries; return the mean (or nan)."""
  f = dist[np.isfinite(dist)]
  if f.size == 0:
    print(f"  {name}: (empty)")
    return np.nan
  print(f"  {name}: mean {f.mean():.4f}  median {np.median(f):.4f}  "
        f"p95 {np.percentile(f, 95):.4f}  n={f.size}")
  return float(f.mean())


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("render_h5")
  p.add_argument("wt_h5")
  p.add_argument("index", type=int, help="view index (same in both files)")
  p.add_argument("--layers", type=int, nargs="+", default=None,
                 help="depth-peel layer indices to keep, on BOTH clouds (default: all)")
  p.add_argument("--dist-range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                 help="pin the shared distance colour scale (default: LO=0, HI=99th "
                      "percentile of the pooled GT->WT and WT->GT distances)")
  p.add_argument("-e", "--export-format", choices=["ply", "glb"], default="ply",
                 help="ply (default): two flat files <prefix>.gt.ply / <prefix>.wt.ply. "
                      "glb: one file with \"gt\" and \"wt\" nodes.")
  p.add_argument("-o", "--out", default=None,
                 help="output path PREFIX (default: <wt_h5>.view<index>.disparity); "
                      "the .gt.ply/.wt.ply or .glb suffix is appended")
  args = p.parse_args()
  if args.dist_range and args.dist_range[1] <= args.dist_range[0]:
    p.error("--dist-range HI must be > LO")

  with h5py.File(args.render_h5, "r") as rf, h5py.File(args.wt_h5, "r") as wf:
    k_name = intrinsics_name(rf, "depth", None)
    if k_name is None:
      raise SystemExit(f"{args.render_h5}: no depth_intrinsics or camera_intrinsics")

    depth_peel = rf["depth_peel"][args.index]        # (H, W, L)
    K = rf[k_name][args.index]                       # (3, 3)
    pts_grid = wf["points"][args.index]              # (H, W, L, 3)
    mesh = int(rf["mesh_index"][args.index]) if "mesh_index" in rf else None

    n_layers = min(depth_peel.shape[2], pts_grid.shape[2])
    if args.layers is None:
      sel = list(range(n_layers))
    else:
      sel = sorted(set(args.layers) & set(range(n_layers)))
      dropped = sorted(set(args.layers) - set(sel))
      if dropped:
        print(f"[warn] layers {dropped} out of range [0, {n_layers})")
      if not sel:
        raise SystemExit(f"--layers {args.layers} selected nothing in [0, {n_layers})")

  # unproject the FULL peel, then filter by point->layer (slicing depth_peel
  # by `sel` would renumber the layers)
  gt, _, _, gt_layer = unproject_depth_peel(depth_peel, K, None, "camera")
  wt, _, _, wt_layer = extract_valid_points(pts_grid)

  suffix = "" if args.layers is None else f" (layers {sel})"
  if args.layers is not None:
    gt, gt_layer = gt[np.isin(gt_layer, sel)], gt_layer[np.isin(gt_layer, sel)]
    wt = wt[np.isin(wt_layer, sel)]

  # cKDTree rejects non-finite input; the loaders already drop NaN/-1, this
  # is cheap insurance
  gt = gt[np.isfinite(gt).all(1)]
  wt = wt[np.isfinite(wt).all(1)]
  if len(gt) == 0:
    raise SystemExit(f"GT cloud has 0 valid points for view {args.index}{suffix}")
  if len(wt) == 0:
    raise SystemExit(f"WT cloud has 0 valid points for view {args.index}{suffix}")

  d_gt = _nn_dist(gt, wt)   # GT -> nearest WT
  d_wt = _nn_dist(wt, gt)   # WT -> nearest GT

  if args.dist_range is not None:
    lo, hi = args.dist_range
  else:
    pooled = np.concatenate([d_gt[np.isfinite(d_gt)], d_wt[np.isfinite(d_wt)]])
    lo, hi = 0.0, (float(np.percentile(pooled, 99)) if pooled.size else 1.0)

  gt_colors = _colors_by_distance(d_gt, "Blues", lo, hi)
  wt_colors = _colors_by_distance(d_wt, "Reds", lo, hi)

  head = f"view {args.index}"
  if mesh is not None:
    head += f"  mesh {mesh}"
  print(f"{head}  GT {len(gt)} pts  WT {len(wt)} pts{suffix}")
  print(f"  distance colour scale (shared): [{lo:.4g}, {hi:.4g}]")
  m_gt = _stats("GT->WT", d_gt)
  m_wt = _stats("WT->GT", d_wt)
  print(f"  per-point Chamfer  mean(GT->WT) + mean(WT->GT) = {m_gt + m_wt:.4f}  "
        f"(cf. plot_chamfer_box; a few % off from its 40k/side subsampling)")

  prefix = args.out or f"{args.wt_h5}.view{args.index}.disparity"
  if args.export_format == "ply":
    for tag, xyz, col in (("gt", gt, gt_colors), ("wt", wt, wt_colors)):
      out = f"{prefix}.{tag}.ply"
      trimesh.points.PointCloud(xyz, colors=col).export(out)
      print(f"[cloud] {out}")
  else:
    scene = trimesh.Scene()
    scene.graph.update(frame_to="points", frame_from=scene.graph.base_frame,
                       matrix=np.eye(4))
    for tag, xyz, col in (("gt", gt, gt_colors), ("wt", wt, wt_colors)):
      scene.add_geometry(trimesh.points.PointCloud(xyz, colors=col),
                         node_name=tag, parent_node_name="points")
    out = f"{prefix}.glb"
    scene.export(out)
    print(f"[cloud] {out}")


if __name__ == "__main__":
  main()
