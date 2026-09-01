#!/usr/bin/env python3
"""Visualize one view of a capture_turntable.py render HDF5 as a colored
point cloud .glb: unproject every depth-peel layer using that view's camera
intrinsics/pose, color the surface layer (layer 0) from the RGB image, and
color deeper layers with a white -> red gradient by layer index.
"""

from argparse import ArgumentParser

import h5py
import numpy as np
import trimesh


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


def main():
  parser = ArgumentParser(description=__doc__)
  parser.add_argument("hdf5_path")
  parser.add_argument("index", type=int)
  parser.add_argument("-o", "--out", default=None, help="Output .glb path (default: <hdf5>.view<index>.glb)")
  args = parser.parse_args()

  with h5py.File(args.hdf5_path, "r") as f:
    image = f["images"][args.index]
    depth_peel = f["depth_peel"][args.index]
    intrinsics = f["camera_intrinsics"][args.index]
    pose = f["camera_pose"][args.index]

  max_layers = depth_peel.shape[2]
  points, u, v, layer_idx = unproject_depth_peel(depth_peel, intrinsics, pose)
  colors = colors_for(image, u, v, layer_idx, max_layers)

  point_cloud = trimesh.points.PointCloud(points, colors=colors)

  out_path = args.out or f"{args.hdf5_path}.view{args.index}.glb"
  point_cloud.export(out_path)
  print(f"Wrote {len(points)} points ({(layer_idx == 0).sum()} surface) to {out_path}")


if __name__ == "__main__":
  main()
