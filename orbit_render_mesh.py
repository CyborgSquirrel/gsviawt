#!/usr/bin/env python3
"""Blender-side helper for `orbit_video.py`'s mesh-vs-3DGS comparison.

`orbit_video.py` runs in the container venv (gsplat, no `bpy`), so it shells
out to Blender for the mesh render:

    /opt/blender/blender --background --python orbit_render_mesh.py -- <job.npz> <out.npy>

`job.npz` (written by orbit_video.py) carries the orbit it already used:
`poses` (N,4,4 camera-to-world, OpenGL cam axes -- the same array orbit_video
feeds gsplat), `fx`, `width`, `height`, plus the render knobs
(`lighting` / `normalize_object` / `cycles_samples` / `render_engine` /
`device`). We rebuild render_objaverse.py's scene (same normalise + multi-sun
rig the 3DGS was fit against), render one RGBA frame per pose, and dump an
(N, H, W, 4) uint8 stack to `out.npy`.

The camera pose is set straight from `poses[i]` (matrix_world), so the mesh
and the splat see pixel-identical cameras.
"""

def _head():
  import os
  import sys
  extra = os.environ.get("BLENDER_USER_PYTHON", "")
  if extra.strip():
    sys.path.append(extra)
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_head(); del _head

import logging
import os
import sys
from tempfile import TemporaryDirectory

import bpy
import numpy as np
from mathutils import Matrix

from render_objaverse import (
  configure_device, load_object, normalize_object, render_rgba, reset_scene,
  setup_lighting,
)

log = logging.getLogger("orbit_render_mesh")


def setup_camera_from_fx(fx, width, height):
  for o in list(bpy.data.objects):
    if o.type == "CAMERA":
      bpy.data.objects.remove(o, do_unlink=True)
  cam_data = bpy.data.cameras.new("Camera")
  cam_data.sensor_fit = "HORIZONTAL"
  cam_data.sensor_width = cam_data.sensor_height = 32.0
  cam_data.lens = float(fx) * cam_data.sensor_width / float(width)  # fx = w*lens/sw
  cam_data.shift_x = cam_data.shift_y = 0.0                          # principal point = centre
  cam = bpy.data.objects.new("Camera", cam_data)
  bpy.context.collection.objects.link(cam)
  bpy.context.scene.camera = cam
  return cam


def main():
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  sep = sys.argv.index("--")
  job_path, out_path = sys.argv[sep + 1], sys.argv[sep + 2]
  job = np.load(job_path, allow_pickle=False)

  poses = job["poses"].astype(np.float64)          # (N,4,4) c2w, OpenGL cam axes
  fx = float(job["fx"])
  W, H = int(job["width"]), int(job["height"])
  mesh_path = str(job["mesh_path"])
  lighting = str(job["lighting"])
  do_normalize = bool(job["normalize_object"])
  samples = int(job["cycles_samples"])
  engine = str(job["render_engine"])
  device = str(job["device"])

  log.info("mesh=%s  %d poses  %dx%d  fx=%.3f  engine=%s samples=%d lighting=%s normalize=%s",
           mesh_path, len(poses), W, H, fx, engine, samples, lighting, do_normalize)

  scene = bpy.context.scene
  scene.render.engine = engine
  scene.render.resolution_percentage = 100
  scene.render.film_transparent = True
  scene.render.use_persistent_data = False        # don't cache per-frame render data
  if scene.render.engine == "CYCLES":
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
  configure_device(device)

  reset_scene()
  load_object(mesh_path)
  if do_normalize:
    normalize_object()
  setup_lighting(lighting)
  cam = setup_camera_from_fx(fx, W, H)

  frames = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                     shape=(len(poses), H, W, 4))
  with TemporaryDirectory() as tmp:
    png = os.path.join(tmp, "f.png")
    for i, c2w in enumerate(poses):
      cam.matrix_world = Matrix(c2w.tolist())
      bpy.context.view_layer.update()
      frames[i] = render_rgba(png, W, H)
      # keep RSS flat over 100s of frames: drop any stray 0-user image
      # render_rgba's readback left behind, and flush the mmap page cache.
      for img in list(bpy.data.images):
        if img.users == 0 and not img.name.startswith(("Render Result", "Viewer Node")):
          bpy.data.images.remove(img)
      if (i + 1) % 24 == 0:
        frames.flush()
      if (i + 1) % max(1, len(poses) // 10) == 0 or i == len(poses) - 1:
        log.info("rendered %d/%d mesh frames", i + 1, len(poses))
  frames.flush()
  log.info("wrote %s  %s", out_path, frames.shape)


if __name__ == "__main__":
  main()
