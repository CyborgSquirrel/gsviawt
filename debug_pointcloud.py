#!/usr/bin/env python3
"""Visualize one view of a capture_turntable.py render HDF5 as a colored
point cloud .glb.

Three formats (-f/--format):
  auto   (default) -- detect depth vs. points from which dataset is present
    in the file (error if both or neither are).
  depth  -- unproject every depth-peel layer using that view's camera
    intrinsics/pose, color the surface layer (layer 0) from the RGB image,
    and color deeper layers with a white -> red gradient by layer index.
  points -- take an existing (H, W, L, 3) XYZ point grid (NaN marks an
    invalid entry; same (H, W, L) layout as depth_peel, see
    wt_infer_layers.py) and color it from the RGB image exactly like the
    depth case, no unprojection.

Each input (image/depth/intrinsics/pose/points) is read from `hdf5_path` at
its default dataset name unless overridden with the matching flag, given as
a different dataset name within that same file.

Output format (-e/--export-format):
  glb (default) -- a "points" root node with one child node per layer index
    ("layer0", "layer1", ...), so a viewer can toggle layers individually.
  ply -- every layer merged into a single flat point cloud (the format has
    no scene graph, so there is nothing to attach separate layer nodes to).
"""

from argparse import ArgumentParser

import h5py
import numpy as np
import trimesh


DEFAULT_DATASETS = {
  "image": "images",
  "depth": "depth_peel",
  "intrinsics": "camera_intrinsics",
  "pose": "camera_pose",
  "points": "points",
}


def detect_format(f, datasets):
  """Pick "depth" or "points" based on which of the two dataset names is
  present in the open HDF5 file `f`; error if both or neither are."""
  has_depth = datasets["depth"] in f
  has_points = datasets["points"] in f
  match (has_depth, has_points):
    case (True, False):
      return "depth"
    case (False, True):
      return "points"
    case _:
      raise ValueError(
        f"Cannot auto-detect format: depth dataset present={has_depth}, "
        f"points dataset present={has_points} (need exactly one); pass "
        "-f/--format explicitly")


def unproject_depth_peel(depth_peel, intrinsics, pose, space):
  """depth_peel: (H, W, L) float32, -1.0 marks no hit.
  intrinsics: (3, 3) pinhole K matrix (pixel row 0 = top, standard
    computer-vision convention: u right, v down, cx/cy in pixel units).
  pose: (4, 4) camera-to-world matrix (Blender camera-local axes: +X
    right, +Y up, -Z forward -- glTF/OpenGL convention, not OpenCV).
  space: "camera" to return points_cv as-is (OpenCV convention: X right,
    Y down, Z forward -- the frame World Tracing itself predicts in, see
    wt_infer_layers.py), or "world" to additionally flip into Blender's
    camera-local axes and apply `pose`.

  Returns (points, u, v, layer_idx), each length N (one entry per hit),
  where points is (N, 3) in the requested space.
  """
  height, width, max_layers = depth_peel.shape
  v_idx, u_idx, l_idx = np.meshgrid(
    np.arange(height), np.arange(width), np.arange(max_layers), indexing="ij")

  hit = depth_peel >= 0
  u = u_idx[hit].astype(np.float32)
  v = v_idx[hit].astype(np.float32)
  layer_idx = l_idx[hit]
  depth = depth_peel[hit]

  # K^-1 maps homogeneous pixel coords to a ray in OpenCV-style camera space
  # (X right, Y down, Z forward) at unit depth; scale by depth to place the
  # point.
  pixels_h = np.stack([u, v, np.ones_like(u)], axis=1)
  rays_cv = pixels_h @ np.linalg.inv(intrinsics).T
  points_cv = rays_cv * depth[:, None]

  match space:
    case "camera":
      points = points_cv
    case "world":
      # Flip Y/Z into Blender's actual camera-local axes (X right, Y up,
      # Z backward) before applying `pose`.
      local = np.concatenate([points_cv * [1, -1, -1], np.ones_like(depth[:, None])], axis=1)
      points = (local @ pose.T)[:, :3]
    case _:
      raise ValueError(f"Unknown space: {space!r}")

  return points, u.astype(np.int64), v.astype(np.int64), layer_idx


def colors_for(image, u, v, layer_idx, max_layers):
  """image: (H, W, 3) uint8. Returns (N, 4) uint8 RGBA."""
  surface_rgb = image[v, u].astype(np.float32)  # 0-255

  denom = max(max_layers - 1, 1)
  t = (layer_idx.astype(np.float32) / denom)[:, None]
  white = np.array([255.0, 255.0, 255.0])
  red = np.array([255.0, 0.0, 0.0])
  gradient_rgb = white * (1 - t) + red * t

  is_surface = (layer_idx == 0)[:, None]
  rgb = np.where(is_surface, surface_rgb, gradient_rgb)

  alpha = np.full((rgb.shape[0], 1), 255.0)
  return np.clip(np.concatenate([rgb, alpha], axis=1), 0, 255).astype(np.uint8)


def extract_valid_points(points_grid):
  """points_grid: (H, W, L, 3) float32 XYZ, NaN in any component marks an
  invalid entry (mirroring depth_peel's (H, W, L) layout and -1.0 sentinel,
  but a scalar sentinel can't sit inside real XYZ, hence NaN -- see
  wt_infer_layers.py). Returns (xyz, u, v, layer_idx), each length N (one
  entry per valid hit), analogous to unproject_depth_peel's outputs.
  """
  height, width, max_layers, _ = points_grid.shape
  v_idx, u_idx, l_idx = np.meshgrid(
    np.arange(height), np.arange(width), np.arange(max_layers), indexing="ij")

  hit = ~np.isnan(points_grid).any(axis=-1)
  xyz = points_grid[hit].astype(np.float32)
  u = u_idx[hit].astype(np.int64)
  v = v_idx[hit].astype(np.int64)
  layer_idx = l_idx[hit]

  return xyz, u, v, layer_idx


def main():
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_path", help="Default HDF5 source for any input not overridden below")
  parser.add_argument("index", type=int)
  parser.add_argument(
    "-f", "--format", choices=["auto", "depth", "points"], default="auto",
    help="auto: detect from which dataset is present (default); "
    "depth: unproject a depth-peel layer stack; "
    "points: re-export a flat point cloud dataset")
  parser.add_argument(
    "-s", "--space", choices=["world", "camera"], default=None,
    help="Space to unproject depth-peel points into (required for -f depth, "
    "invalid for -f points)")
  parser.add_argument(
    "-e", "--export-format", choices=["glb", "ply"], default="glb",
    help="glb: one scene node per layer under a \"points\" root (default); "
    "ply: merge every layer into a single flat point cloud (ply has no "
    "scene graph)")
  parser.add_argument("-o", "--out", default=None, help="Output path (default: <hdf5>.view<index>.<export-format>)")
  parser.add_argument("--image", default=None, metavar="DATASET", help="Override the RGB image dataset name (default: images)")
  parser.add_argument("--depth", default=None, metavar="DATASET", help="Override the depth-peel dataset name (default: depth_peel)")
  parser.add_argument("--intrinsics", default=None, metavar="DATASET", help="Override the camera intrinsics dataset name (default: camera_intrinsics)")
  parser.add_argument("--pose", default=None, metavar="DATASET", help="Override the camera pose dataset name (default: camera_pose)")
  parser.add_argument("--points", default=None, metavar="DATASET", help="Override the raw point cloud dataset name (default: points)")
  args = parser.parse_args()

  datasets = dict(DEFAULT_DATASETS)
  for kind in datasets:
    override = getattr(args, kind)
    if override is not None:
      datasets[kind] = override

  with h5py.File(args.hdf5_path, "r") as f:
    fmt = args.format
    if fmt == "auto":
      fmt = detect_format(f, datasets)

    match fmt:
      case "depth":
        if args.space is None:
          raise ValueError("-s/--space (world or camera) is required for -f depth")

        image = f[datasets["image"]][args.index]
        depth_peel = f[datasets["depth"]][args.index]
        intrinsics = f[datasets["intrinsics"]][args.index]
        pose = f[datasets["pose"]][args.index]

        max_layers = depth_peel.shape[2]
        points, u, v, layer_idx = unproject_depth_peel(depth_peel, intrinsics, pose, args.space)
        colors = colors_for(image, u, v, layer_idx, max_layers)
        detail = f" ({(layer_idx == 0).sum()} surface)"
      case "points":
        if args.space is not None:
          raise ValueError("-s/--space is not valid for -f points")

        image = f[datasets["image"]][args.index]
        raw_points = f[datasets["points"]][args.index]

        max_layers = raw_points.shape[2]
        points, u, v, layer_idx = extract_valid_points(raw_points)
        colors = colors_for(image, u, v, layer_idx, max_layers)
        detail = f" ({(layer_idx == 0).sum()} surface)"
      case _:
        raise ValueError(f"Unknown format: {fmt!r}")

  out_path = args.out or f"{args.hdf5_path}.view{args.index}.{args.export_format}"

  match args.export_format:
    case "glb":
      scene = trimesh.Scene()
      scene.graph.update(frame_to="points", frame_from=scene.graph.base_frame, matrix=np.eye(4))
      layers = np.unique(layer_idx)
      for layer in layers:
        mask = layer_idx == layer
        point_cloud = trimesh.points.PointCloud(points[mask], colors=colors[mask])
        scene.add_geometry(point_cloud, node_name=f"layer{layer}", parent_node_name="points")
      scene.export(out_path)
      print(f"Wrote {len(points)}{detail} points across {len(layers)} layers to {out_path}")
    case "ply":
      point_cloud = trimesh.points.PointCloud(points, colors=colors)
      point_cloud.export(out_path)
      print(f"Wrote {len(points)}{detail} points to {out_path}")
    case _:
      raise ValueError(f"Unknown export format: {args.export_format!r}")


if __name__ == "__main__":
  main()
