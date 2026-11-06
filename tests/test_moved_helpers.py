"""The helpers moved into cameras.py / gaussians.py: bit-identical to their old implementations (verbatim copies of the
f0300b1 versions below, on seeded random inputs), and every old import path still gives the very same object."""
import numpy as np
import pytest

import cameras

# ---------------------------------------------------------------------------
# the old implementations, verbatim @ f0300b1
# ---------------------------------------------------------------------------

OLD_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
OLD_WORLD_UP = np.array([0.0, 0.0, 1.0], np.float32)


def old_look_at_c2w(eye, target, up=OLD_WORLD_UP):   # module.py
  z = eye - target
  z = z / (np.linalg.norm(z) + 1e-8)
  if abs(np.dot(z, up)) > 0.999:
    up = np.array([0.0, 1.0, 0.0], np.float32) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
  x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-8)
  y = np.cross(z, x)
  c2w = np.eye(4, dtype=np.float32)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
  return c2w


def old_make_viewmats(poses):   # module.py
  c2w_cv = poses @ OLD_OPENGL_TO_OPENCV
  return np.linalg.inv(c2w_cv).astype(np.float32)


def old_relative_viewmats(poses):   # gs_dataset.py
  c2w_cv = poses @ OLD_OPENGL_TO_OPENCV
  rel_c2w = np.linalg.inv(c2w_cv[0]) @ c2w_cv
  return np.linalg.inv(rel_c2w).astype(np.float32)


def old_dense_unproject_camera(depth_peel, intrinsics):   # gs_dataset.py
  h, w, l = depth_peel.shape
  uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
  pixels_h = np.stack([uu + 0.5, vv + 0.5, np.ones_like(uu)], axis=-1)
  rays_cv = pixels_h @ np.linalg.inv(intrinsics).T
  hit = depth_peel > 0
  xyz_cam = rays_cv[:, :, None, :] * depth_peel[:, :, :, None]
  return np.where(hit[..., None], xyz_cam, np.nan).astype(np.float32), hit


def old_unproject_depth_peel(depth_peel, intrinsics, pose, space, pixel_offset=0.5):   # debug_pointcloud.py
  height, width, max_layers = depth_peel.shape
  v_idx, u_idx, l_idx = np.meshgrid(
    np.arange(height), np.arange(width), np.arange(max_layers), indexing="ij")
  hit = depth_peel > 0
  u = u_idx[hit].astype(np.float32)
  v = v_idx[hit].astype(np.float32)
  layer_idx = l_idx[hit]
  depth = depth_peel[hit]
  pixels_h = np.stack([u + pixel_offset, v + pixel_offset, np.ones_like(u)], axis=1)
  rays_cv = pixels_h @ np.linalg.inv(intrinsics).T
  points_cv = rays_cv * depth[:, None]
  match space:
    case "camera":
      points = points_cv
    case "world":
      local = np.concatenate([points_cv * [1, -1, -1], np.ones_like(depth[:, None])], axis=1)
      points = (local @ pose.T)[:, :3]
    case _:
      raise ValueError(f"Unknown space: {space!r}")
  return points, u.astype(np.int64), v.astype(np.int64), layer_idx


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def random_poses(rng, n):
  """(n, 4, 4) float32 camera-to-world poses looking at the origin from random directions."""
  poses = []
  for _ in range(n):
    eye = rng.normal(size=3).astype(np.float32)
    eye *= np.float32(rng.uniform(1.0, 3.0)) / np.linalg.norm(eye)
    poses.append(old_look_at_c2w(eye, rng.normal(scale=0.1, size=3).astype(np.float32)))
  return np.stack(poses)


def random_depth(rng, h=12, w=10, l=4):
  depth = rng.uniform(0.5, 3.0, size=(h, w, l)).astype(np.float32)
  depth[rng.random((h, w, l)) < 0.3] = np.nan
  depth[rng.random((h, w, l)) < 0.05] = -1.0        # legacy "no hit"
  return depth


def random_K(rng, h=12, w=10):
  f = rng.uniform(8.0, 20.0)
  return np.array([[f, 0, w / 2], [0, f * rng.uniform(0.9, 1.1), h / 2], [0, 0, 1]], np.float32)


def same(a, b):
  """Bit-identical: same dtype, shape and bytes (NaN in the same places counts as equal)."""
  a, b = np.asarray(a), np.asarray(b)
  return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# cameras.py
# ---------------------------------------------------------------------------

def test_constants():
  assert same(cameras.OPENGL_TO_OPENCV, OLD_OPENGL_TO_OPENCV)
  assert same(cameras.WORLD_UP, OLD_WORLD_UP)


@pytest.mark.parametrize("seed", range(5))
def test_look_at_c2w(seed):
  rng = np.random.default_rng(seed)
  for _ in range(20):
    eye = rng.normal(size=3).astype(np.float32)
    target = rng.normal(size=3).astype(np.float32)
    assert same(cameras.look_at_c2w(eye, target), old_look_at_c2w(eye, target))
    up = rng.normal(size=3).astype(np.float32)
    assert same(cameras.look_at_c2w(eye, target, up), old_look_at_c2w(eye, target, up))
  for eye in ([0, 0, 2], [0, 0, -2], [0, 2, 0]):                  # the degenerate (vertical) fallbacks
    eye = np.array(eye, np.float32)
    for up in (OLD_WORLD_UP, np.array([0, 1, 0], np.float32)):
      assert same(cameras.look_at_c2w(eye, np.zeros(3, np.float32), up), old_look_at_c2w(eye, np.zeros(3, np.float32), up))


@pytest.mark.parametrize("seed", range(5))
def test_viewmats(seed):
  rng = np.random.default_rng(seed)
  poses = random_poses(rng, 7)
  assert same(cameras.make_viewmats(poses), old_make_viewmats(poses))
  assert same(cameras.make_viewmats(poses[0]), old_make_viewmats(poses[0]))
  assert same(cameras.relative_viewmats(poses), old_relative_viewmats(poses))
  poses64 = poses.astype(np.float64)
  assert same(cameras.make_viewmats(poses64), old_make_viewmats(poses64))


@pytest.mark.parametrize("seed", range(5))
def test_unprojection(seed):
  rng = np.random.default_rng(seed)
  depth, K, pose = random_depth(rng), random_K(rng), random_poses(rng, 1)[0]
  for new, old in zip(cameras.dense_unproject_camera(depth, K), old_dense_unproject_camera(depth, K)):
    assert same(new, old)
  for space in ("camera", "world"):
    for offset in (0.5, 0.0):
      new = cameras.unproject_depth_peel(depth, K, pose, space, pixel_offset=offset)
      old = old_unproject_depth_peel(depth, K, pose, space, pixel_offset=offset)
      assert all(same(a, b) for a, b in zip(new, old))
  with pytest.raises(ValueError):
    cameras.unproject_depth_peel(depth, K, pose, "nope")


def test_camera_reexports_are_the_same_objects():
  import debug_pointcloud
  import gs_dataset
  import gs_diffusion
  import module
  for name in ("OPENGL_TO_OPENCV", "WORLD_UP", "look_at_c2w", "make_viewmats", "unproject_depth_peel"):
    assert getattr(module, name) is getattr(cameras, name), name
  for name in ("OPENGL_TO_OPENCV", "dense_unproject_camera", "relative_viewmats"):
    assert getattr(gs_dataset, name) is getattr(cameras, name), name
  assert gs_diffusion.OPENGL_TO_OPENCV is cameras.OPENGL_TO_OPENCV
  assert debug_pointcloud.unproject_depth_peel is cameras.unproject_depth_peel
  import orbit_video
  assert orbit_video.WORLD_UP is cameras.WORLD_UP and orbit_video.make_viewmats is cameras.make_viewmats
