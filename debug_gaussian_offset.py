#!/usr/bin/env python3
"""Colour a gaussian_layout=layered fit_3dgs.py point cloud by how far each
Gaussian moved from its depth-peel seed position to its final trained
position -- the magnitude of Fit3DGSLightningModule._active_means()'s
means_original + offset split (see fit_3dgs.py's module docstring), one
scalar per Gaussian, pale (near-zero) -> saturated (moved a lot).

means_original is never saved anywhere (the flat .h5 only has the final,
resolved means) -- this script REPRODUCES it by replaying the training
run's own seeding (resolve_views + init_gaussians_layered) with the exact
config the run itself saved to its output .h5 (attrs["config_json"]).
That replay is deterministic (same hdf5_path/seed/split_fn -> same
primary_view_idx -> same depth-peel unprojection), so no extra metadata
from the run is needed. Only meaningful for a layered-mode output --
"set" mode has no means_original at all (means was always fully free).

    python debug_gaussian_offset.py outputs/fit3dgs_layered_porsche_250views_sh0_noreset_meanopt_reg.h5
    python debug_gaussian_offset.py outputs/foo.h5 --cmap viridis -o /tmp/foo.offset.ply
"""

import json
from argparse import ArgumentParser

import h5py
import numpy as np
import trimesh
from omegaconf import OmegaConf

import fit_3dgs


def _colors_by_distance(dist, cmap_name, lo, hi):
  """Colormap per-point distance `dist` with `cmap_name` (pale -> saturated).
  Mirrors debug_cloud_disparity._colors_by_distance / debug_pointcloud.colors_by_depth."""
  import matplotlib

  t = np.clip((dist - lo) / ((hi - lo) or 1.0), 0.0, 1.0)
  rgba = matplotlib.colormaps[cmap_name](np.nan_to_num(t, nan=1.0))
  return np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main():
  p = ArgumentParser(description=__doc__)
  p.add_argument("h5", help="fit_3dgs.py layered-mode flat output .h5 (means/.../config_json attr)")
  p.add_argument("--cmap", default="turbo", help="matplotlib colormap name (default: turbo)")
  p.add_argument("--dist-range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                 help="pin the colour scale (default: LO=0, HI=99th percentile of the offset distances)")
  p.add_argument("-o", "--out", default=None, help="output .ply path (default: <h5>.offset.ply)")
  args = p.parse_args()
  if args.dist_range and args.dist_range[1] <= args.dist_range[0]:
    p.error("--dist-range HI must be > LO")

  with h5py.File(args.h5, "r") as f:
    cfg = OmegaConf.create(json.loads(f.attrs["config_json"]))
    means_final = np.asarray(f["means"])

  if str(cfg.gaussian_layout) != "layered":
    raise SystemExit(f"{args.h5}: gaussian_layout={cfg.gaussian_layout!r}, not \"layered\" -- "
                     "means_original (the depth-peel seed) only exists in layered mode")

  # Replay the exact seeding the training run did.
  primary_view_idx, _secondary, _validation = fit_3dgs.resolve_views(
    cfg.hdf5_path, cfg.data.get("primary_view"), cfg.data.get("secondary_views"),
    cfg.data.get("validation_views"), float(cfg.data.split_fn.val_fraction), int(cfg.seed))
  seed, _meta = fit_3dgs.init_gaussians_layered(
    cfg.hdf5_path, primary_view_idx, int(cfg.sh_degree), float(cfg.init_opacity), int(cfg.knn_k))
  means_original = seed["means"]

  if means_original.shape != means_final.shape:
    raise SystemExit(
      f"seed/final Gaussian count mismatch ({len(means_original)} vs {len(means_final)}) -- "
      "either this isn't really a layered-mode output, or densify/prune wasn't disabled")

  dist = np.linalg.norm(means_final - means_original, axis=-1)

  lo, hi = args.dist_range if args.dist_range else (0.0, float(np.percentile(dist, 99)))
  colors = _colors_by_distance(dist, args.cmap, lo, hi)

  print(f"{args.h5}: {len(dist)} Gaussians (primary view {primary_view_idx})")
  print(f"  offset distance: mean {dist.mean():.5f}  median {np.median(dist):.5f}  "
        f"p95 {np.percentile(dist, 95):.5f}  max {dist.max():.5f}")
  print(f"  colour scale: [{lo:.4g}, {hi:.4g}] ({args.cmap})")

  out = args.out or f"{args.h5}.offset.ply"
  trimesh.points.PointCloud(means_final, colors=colors).export(out)
  print(f"[cloud] {out}")


if __name__ == "__main__":
  main()
