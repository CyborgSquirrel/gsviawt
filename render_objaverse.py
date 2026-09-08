#!/usr/bin/env python3
"""Render turntables of meshes the way Objaverse-XL's `blender_script.py`
does -- normalise the object into a unit cube at the origin, light it with a
fixed multi-sun rig, render RGBA on a transparent film -- but driven by our
Hydra config and view strategies, and dumped to a single HDF5 file laid out
like capture_turntable.py's renders.h5 (images / depth_peel / camera_pose /
camera_intrinsics / mesh_index).

Runs *inside* Blender (no rpyc server):

    /opt/blender/blender --background --python render_objaverse.py -- \\
        view_strategy=turntable mesh_strategy=random \\
        output_path=/app/bla/obj.h5

Needs h5py + hydra-core in Blender's Python. Install once:

    /opt/blender/4.2/python/bin/python3.11 -m pip install \\
        --target="$BLENDER_USER_PYTHON" h5py hydra-core

(the Dockerfile does this; `_head()` below puts $BLENDER_USER_PYTHON on the
path, same trick render_server.py uses).
"""

# --- let Blender's bundled Python see our extra packages + repo modules ---
def _head():
  import os
  import sys
  extra = os.environ.get("BLENDER_USER_PYTHON", "")
  if extra.strip():
    sys.path.append(extra)  # append: Blender's own numpy still wins
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_head(); del _head

import contextlib as ctl
import json
import logging
import math
import os
import sys
from tempfile import TemporaryDirectory

import bpy
import numpy as np
from mathutils import Vector

logger = logging.getLogger("render_objaverse")

IMPORT_FUNCS = {
  "glb": bpy.ops.import_scene.gltf,
  "gltf": bpy.ops.import_scene.gltf,
  "fbx": bpy.ops.import_scene.fbx,
  "obj": bpy.ops.wm.obj_import,
  "ply": bpy.ops.wm.ply_import,
  "stl": bpy.ops.wm.stl_import,
}


# ---------------------------------------------------------------------------
# scene setup  (adapted from Objaverse-XL blender_script.py)
# ---------------------------------------------------------------------------

def reset_scene():
  for obj in list(bpy.data.objects):
    if obj.type not in {"CAMERA", "LIGHT"}:
      bpy.data.objects.remove(obj, do_unlink=True)
  for coll in (bpy.data.materials, bpy.data.textures, bpy.data.images):
    for item in list(coll):
      coll.remove(item, do_unlink=True)
  bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def load_object(path):
  ext = path.split(".")[-1].lower()
  fn = IMPORT_FUNCS.get(ext)
  if fn is None:
    raise ValueError(f"unsupported mesh extension: {ext!r} ({path})")
  if ext in ("glb", "gltf"):
    fn(filepath=path, merge_vertices=True, disable_bone_shape=True)
  else:
    fn(filepath=path)


def scene_root_objects():
  return [o for o in bpy.context.scene.objects if o.parent is None
          and o.type not in {"CAMERA", "LIGHT"}]


def scene_meshes():
  return [o for o in bpy.context.scene.objects if isinstance(o.data, bpy.types.Mesh)]


def scene_bbox():
  lo = Vector((math.inf,) * 3)
  hi = Vector((-math.inf,) * 3)
  found = False
  for obj in scene_meshes():
    found = True
    for c in obj.bound_box:
      w = obj.matrix_world @ Vector(c)
      lo = Vector(min(a, b) for a, b in zip(lo, w))
      hi = Vector(max(a, b) for a, b in zip(hi, w))
  if not found:
    raise RuntimeError("no meshes in scene to bound")
  return lo, hi


def normalize_object(target_longest=1.0):
  """Recenter on the origin and scale so the longest bbox axis == target.
  Objaverse/Shap-E convention. Returns the applied scale factor."""
  roots = scene_root_objects()
  if len(roots) > 1:
    parent = bpy.data.objects.new("ParentEmpty", None)
    bpy.context.scene.collection.objects.link(parent)
    for o in roots:
      o.parent = parent
    roots = [parent]

  lo, hi = scene_bbox()
  scale = target_longest / max(max(hi - lo), 1e-12)
  for o in roots:
    o.scale = o.scale * scale
  bpy.context.view_layer.update()

  lo, hi = scene_bbox()
  offset = -(lo + hi) / 2.0
  for o in roots:
    o.matrix_world.translation += offset
  bpy.context.view_layer.update()
  return scale


def _sun(name, rotation, energy):
  d = bpy.data.lights.new(name, type="SUN")
  d.energy = energy
  d.use_shadow = True
  o = bpy.data.objects.new(name, d)
  o.rotation_euler = rotation
  bpy.context.collection.objects.link(o)
  return o


def setup_lighting(mode):
  for o in list(bpy.data.objects):
    if o.type == "LIGHT":
      bpy.data.objects.remove(o, do_unlink=True)
  world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
  bpy.context.scene.world = world
  world.use_nodes = True
  bg = world.node_tree.nodes["Background"]

  if mode == "objaverse":
    # blender_script.py's key/fill/rim/bottom SUN rig (energies fixed to the
    # midpoint of its random.choice ranges, for reproducibility).
    _sun("Key", (0.7854, 0.0, -0.7854), 4.0)
    _sun("Fill", (0.7854, 0.0, 2.3562), 3.0)
    _sun("Rim", (-0.7854, 0.0, -3.9270), 4.0)
    _sun("Bottom", (3.1416, 0.0, 0.0), 2.0)
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
  elif mode == "env":
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    world.cycles_visibility.camera = False
    _sun("Key", (0.0, 0.0, 0.0), 1.5)
  else:
    raise ValueError(f"unknown lighting mode {mode!r}")

  world.cycles_visibility.camera = False  # background renders pure black


def setup_camera(cfg):
  for o in list(bpy.data.objects):
    if o.type == "CAMERA":
      bpy.data.objects.remove(o, do_unlink=True)
  cam_data = bpy.data.cameras.new("Camera")
  cam_data.sensor_fit = "HORIZONTAL"
  cam_data.lens = float(cfg.camera_lens)
  cam_data.sensor_width = cam_data.sensor_height = float(cfg.camera_sensor_width)
  cam = bpy.data.objects.new("Camera", cam_data)
  bpy.context.collection.objects.link(cam)
  bpy.context.scene.camera = cam
  return cam


def aim_camera(cam, target=(0.0, 0.0, 0.0)):
  d = Vector(target) - cam.location
  cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def camera_intrinsics(cam_data, w, h):
  fx = w * cam_data.lens / cam_data.sensor_width
  fy = fx
  return np.array([[fx, 0.0, w / 2.0],
                   [0.0, fy, h / 2.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# depth peeling  (ported verbatim-ish from render_server.py)
# ---------------------------------------------------------------------------

def build_depth_peel_material():
  mat = bpy.data.materials.new("DepthPeelMaterial")
  mat.use_nodes = True
  nt = mat.node_tree
  nt.nodes.clear()
  out = nt.nodes.new("ShaderNodeOutputMaterial")
  cam = nt.nodes.new("ShaderNodeCameraData")
  tc = nt.nodes.new("ShaderNodeTexCoord")
  tex = nt.nodes.new("ShaderNodeTexImage")
  tex.interpolation = "Closest"
  tex.extension = "EXTEND"
  eps = nt.nodes.new("ShaderNodeMath")
  eps.operation = "ADD"
  eps.inputs[1].default_value = 1e-4
  gt = nt.nodes.new("ShaderNodeMath")
  gt.operation = "GREATER_THAN"
  transp = nt.nodes.new("ShaderNodeBsdfTransparent")
  diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
  mix = nt.nodes.new("ShaderNodeMixShader")
  L = nt.links
  L.new(tc.outputs["Window"], tex.inputs["Vector"])
  L.new(tex.outputs["Color"], eps.inputs[0])
  L.new(cam.outputs["View Z Depth"], gt.inputs[0])
  L.new(eps.outputs[0], gt.inputs[1])
  L.new(gt.outputs[0], mix.inputs[0])
  L.new(transp.outputs["BSDF"], mix.inputs[1])
  L.new(diff.outputs["BSDF"], mix.inputs[2])
  L.new(mix.outputs["Shader"], out.inputs["Surface"])
  return mat, tex, eps


def setup_depth_compositor(scene, view_layer):
  view_layer.use_pass_z = True
  scene.use_nodes = True
  nt = scene.node_tree
  nt.nodes.clear()
  rl = nt.nodes.new("CompositorNodeRLayers")
  rl.layer = view_layer.name
  comp = nt.nodes.new("CompositorNodeComposite")
  nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
  return rl, comp


def _write_depth_image(img, depth):
  h, w = depth.shape
  rgba = np.empty((h, w, 4), np.float32)
  rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = depth
  rgba[..., 3] = 1.0
  img.pixels.foreach_set(rgba.ravel())
  img.update()


def _read_exr_channel0(path, w, h):
  img = bpy.data.images.load(path, check_existing=False)
  try:
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[:, :, 0]
  finally:
    bpy.data.images.remove(img)


def depth_peel(mesh_objs, peel_mat, peel_tex, peel_eps, rl, comp,
               width, height, max_layers, scratch_dir):
  scene = bpy.context.scene
  cyc = scene.cycles
  ims = scene.render.image_settings
  saved_fp, saved_ff, saved_cd, saved_cm = (
    scene.render.filepath, ims.file_format, ims.color_depth, ims.color_mode)
  ims.file_format, ims.color_depth, ims.color_mode = "OPEN_EXR", "32", "BW"
  scene.node_tree.links.new(rl.outputs["Depth"], comp.inputs["Image"])

  lo, hi = scene_bbox()
  peel_eps.inputs[1].default_value = max((hi - lo).length * 1e-4, 1e-6)

  saved_slots = []
  for obj in mesh_objs:
    mats = obj.data.materials
    added = len(mats) == 0
    if added:
      mats.append(None)
    saved_slots.append((obj, list(mats), added))
    for i in range(len(mats)):
      mats[i] = peel_mat

  saved_cyc = {k: getattr(cyc, k) for k in (
    "samples", "use_denoising", "max_bounces", "diffuse_bounces",
    "glossy_bounces", "transmission_bounces", "volume_bounces",
    "transparent_max_bounces")}
  cyc.samples = 1
  cyc.use_denoising = False
  cyc.max_bounces = cyc.diffuse_bounces = cyc.glossy_bounces = 0
  cyc.transmission_bounces = cyc.volume_bounces = 0
  cyc.transparent_max_bounces = max_layers + 2

  prev = np.full((height, width), -1e6, np.float32)
  vol = np.full((height, width, max_layers), -1.0, np.float32)
  found = 0
  prev_img = bpy.data.images.new("DepthPeelPrev", width=width, height=height, float_buffer=True)
  peel_tex.image = prev_img
  try:
    for k in range(max_layers):
      _write_depth_image(prev_img, prev)
      p = os.path.join(scratch_dir, f"_peel_{k}.exr")
      scene.render.filepath = p
      bpy.ops.render.render(write_still=True)
      d = _read_exr_channel0(p, width, height)
      os.remove(p)
      hit = d < 1e9
      if not hit.any():
        break
      vol[:, :, k] = np.where(hit, d, -1.0)
      found = k + 1
      prev = np.where(hit, d, prev)
  finally:
    for obj, saved, added in saved_slots:
      mats = obj.data.materials
      for i, m in enumerate(saved):
        mats[i] = m
      if added:
        mats.pop(index=len(mats) - 1)
    for k, v in saved_cyc.items():
      setattr(cyc, k, v)
    scene.node_tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    scene.render.filepath = saved_fp
    ims.file_format, ims.color_depth, ims.color_mode = saved_ff, saved_cd, saved_cm
    peel_tex.image = None
    bpy.data.images.remove(prev_img)

  return np.flip(vol, axis=0), found


# ---------------------------------------------------------------------------
# rgb read-back
# ---------------------------------------------------------------------------

def render_rgba(path, width, height):
  """Render the current frame to `path` (PNG, RGBA) and read it back as
  (H, W, 4) uint8, top-row-first. 'Non-Color' colorspace so pixels come back
  as the stored 8-bit values (sRGB-encoded) rather than linearised."""
  scene = bpy.context.scene
  ims = scene.render.image_settings
  saved = (scene.render.filepath, ims.file_format, ims.color_mode, ims.color_depth)
  ims.file_format, ims.color_mode, ims.color_depth = "PNG", "RGBA", "8"
  scene.render.filepath = path
  bpy.ops.render.render(write_still=True)
  scene.render.filepath, ims.file_format, ims.color_mode, ims.color_depth = saved

  img = bpy.data.images.load(path, check_existing=False)
  try:
    img.colorspace_settings.name = "Non-Color"
    px = np.empty(width * height * 4, np.float32)
    img.pixels.foreach_get(px)
  finally:
    bpy.data.images.remove(img)
  rgba = np.flipud(px.reshape(height, width, 4))
  return np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

def configure_device(device):
  scene = bpy.context.scene
  if scene.render.engine != "CYCLES":
    return  # EEVEE has no device selection to make
  if device == "cpu":
    scene.cycles.device = "CPU"
    return
  prefs = bpy.context.preferences.addons["cycles"].preferences
  for backend in ("OPTIX", "CUDA"):
    try:
      prefs.compute_device_type = backend
      prefs.get_devices()
    except Exception:
      continue
    gpus = [d for d in prefs.devices if d.type == backend]
    if gpus:
      for d in prefs.devices:
        d.use = d.type == backend
      scene.cycles.device = "GPU"
      logger.info("rendering on GPU (%s)", backend)
      return
  if device == "gpu":
    raise RuntimeError("device='gpu' but no CUDA/OptiX device found")
  scene.cycles.device = "CPU"
  logger.info("no GPU found, rendering on CPU")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(cfg):
  import hydra
  import h5py
  from util import LazyDataset, timed
  from omegaconf import OmegaConf

  logging.basicConfig(level=logging.INFO)
  scene = bpy.context.scene
  scene.render.engine = str(cfg.render_engine)
  scene.render.resolution_x = int(cfg.width)
  scene.render.resolution_y = int(cfg.height)
  scene.render.resolution_percentage = 100
  scene.render.film_transparent = bool(cfg.film_transparent)
  if scene.render.engine == "CYCLES":
    scene.cycles.samples = int(cfg.cycles_samples)
    scene.cycles.use_denoising = True
  configure_device(str(cfg.device))

  view_strategy = hydra.utils.instantiate(cfg.view_strategy)
  meshes = list(hydra.utils.instantiate(cfg.mesh_strategy).meshes())
  W, H, Lmax = int(cfg.width), int(cfg.height), int(cfg.max_peel_layers)

  os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
  img_kw = dict(chunks=(1, H, W, 4), compression="gzip", compression_opts=4)
  depth_kw = dict(chunks=(1, H, W, Lmax), compression="gzip", compression_opts=4)

  with ctl.ExitStack() as stack:
    tmp = stack.enter_context(TemporaryDirectory())
    hf = stack.enter_context(h5py.File(cfg.output_path, "w"))
    hf.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    hf.create_dataset("mesh_paths", data=meshes, dtype=h5py.string_dtype("utf-8"))

    ds_img = stack.enter_context(LazyDataset(hf, "images", dataset_kwargs=img_kw)) if cfg.render else None
    ds_pose = stack.enter_context(LazyDataset(hf, "camera_pose"))
    ds_intr = stack.enter_context(LazyDataset(hf, "camera_intrinsics"))
    ds_depth = stack.enter_context(LazyDataset(hf, "depth_peel", dataset_kwargs=depth_kw))
    ds_mesh = stack.enter_context(LazyDataset(hf, "mesh_index"))
    ds_scale = stack.enter_context(LazyDataset(hf, "depth_scale"))

    for mi, mesh_path in enumerate(meshes):
      reset_scene()  # purges bpy.data objects/materials/images -> rebuild the rig
      load_object(mesh_path)
      if cfg.normalize_object:
        normalize_object()
      cam = setup_camera(cfg)
      setup_lighting(str(cfg.lighting))
      rl, comp = setup_depth_compositor(scene, bpy.context.view_layer)
      peel_mat, peel_tex, peel_eps = build_depth_peel_material()
      mesh_objs = scene_meshes()

      for tilt_deg, az_deg, view_dir in view_strategy.views():
        vd = np.asarray(view_dir, np.float64)
        vd = vd / (np.linalg.norm(vd) or 1.0)
        cam.location = Vector(vd * float(cfg.camera_distance))
        aim_camera(cam)
        bpy.context.view_layer.update()
        label = f"mesh={mi} tilt={tilt_deg:.1f} az={az_deg:.1f}"

        with timed(f"{label} render"):
          if cfg.render:
            rgba = render_rgba(os.path.join(tmp, "rgb.png"), W, H)
          depth_vol, _ = depth_peel(mesh_objs, peel_mat, peel_tex, peel_eps,
                                    rl, comp, W, H, Lmax, tmp)

        pose = np.array(cam.matrix_world, np.float32)
        intr = camera_intrinsics(cam.data, W, H)

        depth_scale = 1.0
        if cfg.camera_depth_target is not None:
          surf = depth_vol[..., 0]
          hit = surf >= 0
          if hit.any():
            depth_scale = float(cfg.camera_depth_target) / float(surf[hit].mean())
            depth_vol = np.where(depth_vol >= 0, depth_vol * depth_scale, depth_vol)
            pose[:3, 3] *= depth_scale

        if ds_img is not None:
          ds_img.append(rgba)
        ds_pose.append(pose)
        ds_intr.append(intr)
        ds_depth.append(depth_vol.astype(np.float32))
        ds_mesh.append(np.int64(mi))
        ds_scale.append(np.float32(depth_scale))

  logger.info("wrote %s", cfg.output_path)


def main():
  from hydra import compose, initialize_config_dir

  sep = sys.argv.index("--") if "--" in sys.argv else len(sys.argv) - 1
  overrides = sys.argv[sep + 1:]
  conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
  with initialize_config_dir(version_base=None, config_dir=conf_dir):
    cfg = compose(config_name="objaverse", overrides=overrides)
  run(cfg)


if __name__ == "__main__":
  main()
