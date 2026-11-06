"""Camera conventions and the depth-peel unprojection, numpy only.

The one home of the helpers module.py, gs_dataset.py and debug_pointcloud.py used to define themselves (each still
re-exports its old names, so `from module import make_viewmats`, `from gs_dataset import OPENGL_TO_OPENCV`,
`from debug_pointcloud import unproject_depth_peel` keep working). Only numpy: gs_dataset.py / gs_diffusion.py import this
without pulling in lightning, wandb or trimesh.

Conventions: poses are camera-to-world in Blender/OpenGL camera axes (X right, Y up, Z back), the h5's `camera_pose`;
gsplat wants world-to-camera viewmats in OpenCV axes (X right, Y down, Z forward); intrinsics K are pinhole, pixel row 0
at the top.

Left where they are on purpose (not bit-identical to the ones here, or pinned):
  - orbit_video.look_at_c2w: divides by the bare norm (no +1e-8) and falls back to up=[0, 1, 0] only, so its frames can
    differ from look_at_c2w's in the last bits.
  - gs_diffusion.encode_grid's own world -> camera matmul: its float order is pinned bit for bit (the cached x0 moments).
  - the inline `inv(pose @ OPENGL_TO_OPENCV)` copies in fit_3dgs.py, anchored_3dgs.py, train_gs.py and module.py's
    _orbit_viewmats (a single vs a batched matmul can differ in the last bit).
  - extractors.py's and plot_view_panels.py's unprojections (integer pixel positions, World Tracing's convention).
"""

import numpy as np

# Camera-local axis flip: Blender/OpenGL (X right, Y up, Z back) <-> OpenCV
# (X right, Y down, Z forward). Canonical copy -- the deleted fit_gsplat.py and
# gs_dataset.py each used to define their own identical constant.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
WORLD_UP = np.array([0.0, 0.0, 1.0], np.float32)  # Blender / render_objaverse is Z-up


def look_at_c2w(eye, target, up=WORLD_UP):
  """OpenGL camera-to-world (X right, Y up, -Z forward) looking from `eye`
  at `target`. Canonical copy -- the deleted fit_gsplat.py had its own; orbit_video.py
  still has a near-copy (bare-norm divide, a different fallback up)."""
  z = eye - target
  z = z / (np.linalg.norm(z) + 1e-8)
  if abs(np.dot(z, up)) > 0.999:
    up = np.array([0.0, 1.0, 0.0], np.float32) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
  x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-8)
  y = np.cross(z, x)
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return c2w


def make_viewmats(poses):
  """poses: (V,4,4) camera-to-world, OpenGL cam axes. Returns (V,4,4)
  world-to-camera in OpenCV convention -- what gsplat wants."""
  c2w_cv = poses @ OPENGL_TO_OPENCV
  return np.linalg.inv(c2w_cv).astype(np.float32)


def relative_viewmats(poses):
  """poses: (V,4,4) camera-to-world, OpenGL axes, poses[0] = the source
  view. Returns viewmats: (V,4,4) world-to-camera in OpenCV convention, but
  with "world" redefined as the source camera's own frame -- viewmats[0] is
  always the identity matrix."""
  c2w_cv = poses @ OPENGL_TO_OPENCV                # (V,4,4), still world-referenced
  rel_c2w = np.linalg.inv(c2w_cv[0]) @ c2w_cv       # (V,4,4), source-frame-referenced
  return np.linalg.inv(rel_c2w).astype(np.float32)  # (V,4,4), viewmats[0] == eye(4)


# ---------------------------------------------------------------------------
# depth-peel unprojection. Two variants, kept apart because their outputs differ:
#   unproject_depth_peel    SPARSE: one row per hit, (N, 3) points (camera or world space) + their (u, v, layer);
#                           pixel_offset selectable (0.5 = pixel centres, 0.0 = World Tracing's integer convention).
#   dense_unproject_camera  DENSE: (H, W, L, 3) float32 in the camera frame, NaN where not hit, always pixel centres
#                           (+0.5). Same ray formula as the sparse one with space="camera".
# ---------------------------------------------------------------------------

def unproject_depth_peel(depth_peel, intrinsics, pose, space, pixel_offset=0.5):
  """depth_peel: (H, W, L) float32, NaN marks no hit (older renders used
    -1.0; the `depth_peel > 0` test below rejects both).
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

  hit = depth_peel > 0  # NaN and -1.0 both fail this
  u = u_idx[hit].astype(np.float32)
  v = v_idx[hit].astype(np.float32)
  layer_idx = l_idx[hit]
  depth = depth_peel[hit]

  # K^-1 maps homogeneous pixel coords to a ray in OpenCV-style camera space
  # (X right, Y down, Z forward) at unit depth; scale by depth to place the
  # point. Pixel (u, v)'s sample sits at its centre (u + 0.5, v + 0.5): the
  # renderer casts each pixel's ray through the centre (cx = W/2), and gsplat
  # rasterizes at pixel centres too (cross-view check: bla/_pixel_convention_test.py).
  # pixel_offset=0.0 reproduces World Tracing's own integer-index convention
  # (wt/inference.py), for comparisons against its point maps.
  pixels_h = np.stack([u + pixel_offset, v + pixel_offset, np.ones_like(u)], axis=1)
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


def dense_unproject_camera(depth_peel, intrinsics):
  """depth_peel: (H,W,L) f32, <=0/NaN = no hit. intrinsics: (3,3), this
  view's own camera. Returns (xyz_cam, hit): xyz_cam is (H,W,L,3) f32 in
  this camera's own OpenCV frame (X right, Y down, Z forward), NaN where not
  hit; hit is (H,W,L) bool. Same ray formula as
  unproject_depth_peel(space="camera"), densified."""
  h, w, l = depth_peel.shape
  uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
  pixels_h = np.stack([uu + 0.5, vv + 0.5, np.ones_like(uu)], axis=-1)   # (H,W,3), pixel centres
  rays_cv = pixels_h @ np.linalg.inv(intrinsics).T                  # (H,W,3), unit-depth rays
  hit = depth_peel > 0                                              # NaN and legacy -1.0 both fail
  xyz_cam = rays_cv[:, :, None, :] * depth_peel[:, :, :, None]      # (H,W,L,3)
  return np.where(hit[..., None], xyz_cam, np.nan).astype(np.float32), hit
