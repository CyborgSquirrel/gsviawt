#!/usr/bin/env python3
"""Visualize one view of a capture_turntable.py render HDF5 as a colored
point cloud .glb.

Two formats (-f/--format):
  depth  (default) -- unproject every depth-peel layer using that view's
    camera intrinsics/pose, color the surface layer (layer 0) from the RGB
    image, and color deeper layers with a white -> red gradient by layer
    index.
  points -- re-export an existing flat point cloud dataset ((N, 3) XYZ, or
    (N, 6) XYZRGB) straight to .glb, no unprojection.

Each input (image/depth/intrinsics/pose/points) is read from `hdf5_path` at
its default dataset name unless overridden with the matching flag, given as
a different dataset name within that same file.
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


def load_dataset(f, index, kind, override):
  """Load one view's array for `kind` from the open HDF5 file `f`, at
  `override` if given, otherwise at its default dataset name."""
  dataset = override or DEFAULT_DATASETS[kind]
  return f[dataset][index]


def unproject_depth_peel(depth_peel, intrinsics, pose):
  """depth_peel: (H, W, L) float32, -1.0 marks no hit.
  intrinsics: (3, 3) pinhole K matrix (pixel row 0 = top, standard
    computer-vision convention: u right, v down, cx/cy in pixel units).
  pose: (4, 4) camera-to-world matrix (Blender camera-local axes: +X
    right, +Y up, -Z forward -- glTF/OpenGL convention, not OpenCV).

  Returns (points_world, u, v, layer_idx), each length N (one entry per
  hit), where points_world is (N, 3).
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
  # point, then flip Y/Z into Blender's actual camera-local axes (X right,
  # Y up, Z backward) before applying `pose`.
  pixels_h = np.stack([u, v, np.ones_like(u)], axis=1)
  rays_cv = pixels_h @ np.linalg.inv(intrinsics).T
  points_cv = rays_cv * depth[:, None]

  local = np.concatenate([points_cv * [1, -1, -1], np.ones_like(depth[:, None])], axis=1)
  points_world = (local @ pose.T)[:, :3]

  return points_world, u.astype(np.int64), v.astype(np.int64), layer_idx


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


def colors_for_points(points):
  """points: (N, 3) or (N, 6) array; columns 3:6, if present, are RGB (either
  0-255 or 0-1 float). Returns (xyz (N, 3) float32, colors (N, 4) uint8 RGBA).
  """
  xyz = np.asarray(points[:, :3], dtype=np.float32)

  if points.shape[1] >= 6:
    rgb = np.asarray(points[:, 3:6], dtype=np.float32)
    if np.issubdtype(points.dtype, np.floating) and rgb.max() <= 1.0:
      rgb = rgb * 255.0
    alpha = np.full((rgb.shape[0], 1), 255.0)
    colors = np.clip(np.concatenate([rgb, alpha], axis=1), 0, 255).astype(np.uint8)
  else:
    colors = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (xyz.shape[0], 1))

  return xyz, colors


def main():
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_path", help="Default HDF5 source for any input not overridden below")
  parser.add_argument("index", type=int)
  parser.add_argument(
    "-f", "--format", choices=["depth", "points"], default="depth",
    help="depth: unproject a depth-peel layer stack (default); "
    "points: re-export a flat point cloud dataset")
  parser.add_argument("-o", "--out", default=None, help="Output .glb path (default: <hdf5>.view<index>.glb)")
  parser.add_argument("--image", default=None, metavar="DATASET", help="Override the RGB image dataset name (default: images)")
  parser.add_argument("--depth", default=None, metavar="DATASET", help="Override the depth-peel dataset name (default: depth_peel)")
  parser.add_argument("--intrinsics", default=None, metavar="DATASET", help="Override the camera intrinsics dataset name (default: camera_intrinsics)")
  parser.add_argument("--pose", default=None, metavar="DATASET", help="Override the camera pose dataset name (default: camera_pose)")
  parser.add_argument("--points", default=None, metavar="DATASET", help="Override the raw point cloud dataset name (default: points)")
  args = parser.parse_args()

  with h5py.File(args.hdf5_path, "r") as f:
    match args.format:
      case "depth":
        image = load_dataset(f, args.index, "image", args.image)
        depth_peel = load_dataset(f, args.index, "depth", args.depth)
        intrinsics = load_dataset(f, args.index, "intrinsics", args.intrinsics)
        pose = load_dataset(f, args.index, "pose", args.pose)

        max_layers = depth_peel.shape[2]
        points, u, v, layer_idx = unproject_depth_peel(depth_peel, intrinsics, pose)
        colors = colors_for(image, u, v, layer_idx, max_layers)
        detail = f" ({(layer_idx == 0).sum()} surface)"
      case "points":
        raw_points = load_dataset(f, args.index, "points", args.points)
        points, colors = colors_for_points(raw_points)
        detail = ""
      case _:
        raise ValueError(f"Unknown format: {args.format!r}")

  point_cloud = trimesh.points.PointCloud(points, colors=colors)

  out_path = args.out or f"{args.hdf5_path}.view{args.index}.glb"
  point_cloud.export(out_path)
  print(f"Wrote {len(points)}{detail} points to {out_path}")


if __name__ == "__main__":
  main()
