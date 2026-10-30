#!/usr/bin/env python3
"""Render meshes the way Objaverse-XL's `blender_script.py` does --
normalise the object into a unit cube at the origin, light it with a fixed
multi-sun rig, render RGBA on a transparent film -- but driven by our Hydra
config and view strategies, and dumped to a single HDF5 file (images / depth_peel / camera_pose /
depth_intrinsics / image_intrinsics / mesh_index / depth_scale) that
wt_infer_layers.py + the compare_wt_depth.py tooling read directly.
depth_intrinsics matches depth_peel, image_intrinsics matches images; both
are always written and are identical when the two resolutions match.

Runs *inside* Blender (no rpyc server):

    /opt/blender/blender --background --python render_objaverse.py -- \\
        view_strategy=turntable-lite mesh_strategy=random \\
        output_path=/app/bla/obj.h5

To render the RGB pass with EEVEE instead of Cycles add
`render_engine=BLENDER_EEVEE_NEXT` (the depth peel stays on Cycles). To A/B the
engines on the *same* meshes + camera poses, re-render an existing h5:

    ... -- source_path=/app/bla/obj.h5 source_rerender_rgb=true \\
        render_engine=BLENDER_EEVEE_NEXT output_path=/app/bla/obj_eevee.h5

and diff the two with compare_renders.py. EEVEE needs a GPU EGL context; the
Dockerfile registers the NVIDIA EGL vendor for that (without it Blender falls
back to Mesa software GL and EEVEE takes minutes per frame).

Needs h5py + hydra-core in Blender's Python. Install once:

    /opt/blender/4.2/python/bin/python3.11 -m pip install \\
        --target="$BLENDER_USER_PYTHON" h5py hydra-core

(the Dockerfile does this; a sitecustomize.py it drops into Blender's own
site-packages puts $BLENDER_USER_PYTHON on sys.path automatically).
"""

# --- let Blender's bundled Python see our repo modules ---
def _head():
  import os
  import sys
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
from mathutils import Matrix, Vector

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
# depth peeling
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


# shaders that let a ray through (or end it without a surface): the peel material has none
_SEE_THROUGH_NODES = ("BSDF_TRANSPARENT", "HOLDOUT", "BSDF_RAY_PORTAL")


def _make_opaque(nt, groups):
  """In place on node tree `nt`, recursing into node groups (copied first and appended to
  `groups`, so the originals are untouched): Principled metallic / transmission / alpha forced to
  0 / 0 / 1, every _SEE_THROUGH_NODES shader replaced by a black diffuse, and the volume /
  displacement outputs unlinked. The material is then opaque wherever the peel material is and
  doesn't move the surface, so it peels the same layers; the Diffuse Color pass is the plain
  base colour."""
  L = nt.links
  for node in list(nt.nodes):
    if node.type == "GROUP" and node.node_tree is not None:
      node.node_tree = node.node_tree.copy()
      groups.append(node.node_tree)
      _make_opaque(node.node_tree, groups)
    elif node.type == "BSDF_PRINCIPLED":
      for name, value in (("Metallic", 0.0), ("Transmission Weight", 0.0), ("Alpha", 1.0)):
        for link in list(node.inputs[name].links):
          L.remove(link)
        node.inputs[name].default_value = value
    elif node.type in _SEE_THROUGH_NODES:
      black = nt.nodes.new("ShaderNodeBsdfDiffuse")
      black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
      for link in list(node.outputs[0].links):
        L.new(black.outputs["BSDF"], link.to_socket)
    elif node.type == "OUTPUT_MATERIAL":
      for name in ("Volume", "Displacement"):
        for link in list(node.inputs[name].links):
          L.remove(link)


def build_gated_material(src_mat, eps_value, groups):
  """A copy of `src_mat` (None = Blender's default grey) whose surface is transparent wherever
  View Z <= prev depth + eps and the original shader (made opaque by _make_opaque, which appends
  its node-group copies to `groups`) elsewhere: build_depth_peel_material's gate, but around the
  real material. Returns (material, previous-depth image texture node)."""
  mat = bpy.data.materials.new("DepthPeelColorMaterial") if src_mat is None else src_mat.copy()
  mat.use_nodes = True
  nt = mat.node_tree
  L = nt.links
  cam = nt.nodes.new("ShaderNodeCameraData")
  tc = nt.nodes.new("ShaderNodeTexCoord")
  tex = nt.nodes.new("ShaderNodeTexImage")
  tex.interpolation = "Closest"
  tex.extension = "EXTEND"
  eps = nt.nodes.new("ShaderNodeMath")
  eps.operation = "ADD"
  eps.inputs[1].default_value = eps_value
  gt = nt.nodes.new("ShaderNodeMath")
  gt.operation = "GREATER_THAN"
  L.new(tc.outputs["Window"], tex.inputs["Vector"])
  L.new(tex.outputs["Color"], eps.inputs[0])
  L.new(cam.outputs["View Z Depth"], gt.inputs[0])
  L.new(eps.outputs[0], gt.inputs[1])
  _make_opaque(nt, groups)
  for out in [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"]:
    surf = out.inputs["Surface"]
    if surf.links:
      orig = surf.links[0].from_socket
    else:
      orig = nt.nodes.new("ShaderNodeBsdfDiffuse").outputs["BSDF"]
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    L.new(gt.outputs[0], mix.inputs[0])
    L.new(transp.outputs["BSDF"], mix.inputs[1])
    L.new(orig, mix.inputs[2])
    L.new(mix.outputs["Shader"], surf)
  return mat, tex


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


def _prev_depth_image(depth, images):
  """A new float image holding `depth` (appended to `images`; the caller removes them). The peel
  renders with persistent data, under which Cycles keeps serving an image it has already loaded
  even after its pixels change, so every render gets a fresh image instead of a rewritten one."""
  h, w = depth.shape
  img = bpy.data.images.new(f"DepthPeelPrev{len(images)}", width=w, height=h, float_buffer=True)
  images.append(img)
  rgba = np.empty((h, w, 4), np.float32)
  rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = depth
  rgba[..., 3] = 1.0
  img.pixels.foreach_set(rgba.ravel())
  img.update()
  return img


def _read_exr_channel0(path, w, h):
  img = bpy.data.images.load(path, check_existing=False)
  try:
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[:, :, 0]
  finally:
    bpy.data.images.remove(img)


def _read_exr_rgb(path, w, h):
  img = bpy.data.images.load(path, check_existing=False)
  try:
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[:, :, :3]
  finally:
    bpy.data.images.remove(img)


def _linear_to_srgb8(lin):
  lin = np.clip(lin, 0.0, 1.0)
  srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
  return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def depth_peel(mesh_objs, peel_mat, peel_tex, peel_eps, rl, comp,
               width, height, max_layers, scratch_dir, with_rgb=False):
  """Returns (depth (H, W, max_layers) float32, NaN = no hit; layers found; rgb). With `with_rgb`,
  rgb is (H, W, max_layers, 3) uint8: the sRGB base colour (Diffuse Color pass, no lighting) of
  the surface at each depth hit, 0 where depth is NaN; else None. The colour comes from the same
  renders as the depth: the sweep then runs on gated copies of the original materials
  (build_gated_material) instead of `peel_mat`, the Diffuse Color pass going to a File Output
  node next to the depth EXR."""
  scene = bpy.context.scene
  view_layer = bpy.context.view_layer
  cyc = scene.cycles
  ims = scene.render.image_settings
  # The peel needs Cycles' exact transparent-BSDF ray continuation; the RGB
  # pass may be EEVEE, so switch for the duration and restore afterwards.
  saved_engine = scene.render.engine
  scene.render.engine = "CYCLES"
  saved_fp, saved_ff, saved_cd, saved_cm = (
    scene.render.filepath, ims.file_format, ims.color_depth, ims.color_mode)
  saved_res = (scene.render.resolution_x, scene.render.resolution_y)
  # Persistent data keeps the Cycles session (device, kernels, BVH) alive between the sweep's
  # renders; without it every render pays a few hundred ms of GPU setup / teardown.
  saved_persistent = scene.render.use_persistent_data
  scene.render.use_persistent_data = True
  ims.file_format, ims.color_depth, ims.color_mode = "OPEN_EXR", "32", "BW"
  scene.render.resolution_x, scene.render.resolution_y = width, height
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

  prev = np.full((height, width), -1e6, np.float32)  # fed back to the peel shader; not the output
  vol = np.full((height, width, max_layers), np.nan, np.float32)  # NaN = no hit at this layer
  found = 0
  rgb = None
  prev_imgs = []
  gated = {}  # original material -> (gated copy, its previous-depth texture node)
  groups = []  # node-group copies the gated materials use
  saved_pass = view_layer.use_pass_diffuse_color
  col_out = None
  try:
    texs = [peel_tex]
    if with_rgb:
      for obj, orig_mats, _ in saved_slots:
        for i, orig in enumerate(orig_mats):
          if orig not in gated:
            gated[orig] = build_gated_material(orig, peel_eps.inputs[1].default_value, groups)
          obj.data.materials[i] = gated[orig][0]
      texs = [tex for _, tex in gated.values()]
      view_layer.use_pass_diffuse_color = True
      col_out = scene.node_tree.nodes.new("CompositorNodeOutputFile")
      col_out.base_path = scratch_dir
      col_out.format.file_format, col_out.format.color_depth = "OPEN_EXR", "32"
      col_out.format.color_mode = "RGB"
      col_out.file_slots[0].path = "_peel_rgb_"
      scene.node_tree.links.new(rl.outputs["DiffCol"], col_out.inputs[0])
      col_path = os.path.join(scratch_dir, f"_peel_rgb_{scene.frame_current:04d}.exr")
      rgb = np.zeros((height, width, max_layers, 3), np.uint8)
    for k in range(max_layers):
      img = _prev_depth_image(prev, prev_imgs)
      for tex in texs:
        tex.image = img
      p = os.path.join(scratch_dir, f"_peel_{k}.exr")
      scene.render.filepath = p
      bpy.ops.render.render(write_still=True)
      d = _read_exr_channel0(p, width, height)
      os.remove(p)
      hit = d < 1e9
      if with_rgb:
        c = _read_exr_rgb(col_path, width, height)
        os.remove(col_path)
      if not hit.any():
        break
      vol[:, :, k] = np.where(hit, d, np.nan)
      if with_rgb:
        rgb[:, :, k] = np.where(hit[..., None], _linear_to_srgb8(c), 0)
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
    scene.render.engine = saved_engine
    scene.node_tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    scene.render.filepath = saved_fp
    ims.file_format, ims.color_depth, ims.color_mode = saved_ff, saved_cd, saved_cm
    scene.render.resolution_x, scene.render.resolution_y = saved_res
    peel_tex.image = None
    view_layer.use_pass_diffuse_color = saved_pass
    if col_out is not None:
      scene.node_tree.nodes.remove(col_out)
    # switching persistent data off frees the session, so the images (whose addresses a later
    # sweep's images may reuse) can't outlive it in Cycles' image cache
    scene.render.use_persistent_data = saved_persistent
    for img in prev_imgs:
      bpy.data.images.remove(img)
    for mat, _ in gated.values():
      bpy.data.materials.remove(mat)
    for group in groups:
      bpy.data.node_groups.remove(group)

  return np.flip(vol, axis=0), found, None if rgb is None else np.flip(rgb, axis=0)


# ---------------------------------------------------------------------------
# rgb read-back
# ---------------------------------------------------------------------------

def render_rgba(path, width, height):
  """Render the current frame to `path` (PNG, RGBA) and read it back as
  (H, W, 4) uint8, top-row-first. 'Non-Color' colorspace so pixels come back
  as the stored 8-bit values (sRGB-encoded) rather than linearised."""
  scene = bpy.context.scene
  ims = scene.render.image_settings
  saved = (scene.render.filepath, ims.file_format, ims.color_mode, ims.color_depth,
           scene.render.resolution_x, scene.render.resolution_y)
  ims.file_format, ims.color_mode, ims.color_depth = "PNG", "RGBA", "8"
  scene.render.resolution_x, scene.render.resolution_y = width, height
  scene.render.filepath = path
  bpy.ops.render.render(write_still=True)
  (scene.render.filepath, ims.file_format, ims.color_mode, ims.color_depth,
   scene.render.resolution_x, scene.render.resolution_y) = saved

  img = bpy.data.images.load(path, check_existing=False)
  try:
    img.colorspace_settings.name = "Non-Color"
    px = np.empty(width * height * 4, np.float32)
    img.pixels.foreach_get(px)
  finally:
    bpy.data.images.remove(img)
  rgba = np.flipud(px.reshape(height, width, 4))
  return np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_props(target, props):
  """setattr every key of the (Omega)dict `props` on a bpy struct; nested
  dicts recurse into the sub-struct (e.g. ray_tracing_options)."""
  for k, v in props.items():
    if hasattr(v, "items"):
      apply_props(getattr(target, k), v)
    else:
      setattr(target, k, v)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

def configure_device(device):
  # Cycles is configured whatever the RGB engine is: the depth peel always
  # runs on it. (EEVEE itself has no device to select; it uses the EGL GPU.)
  scene = bpy.context.scene
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
  from omegaconf import OmegaConf
  from util import TransactionalWriter, size_chunks

  logging.basicConfig(level=logging.INFO)

  with ctl.ExitStack() as stack:
    # Depth-only re-render: the scene (meshes, poses, lens, normalize, ...)
    # comes from the source file's config_json + datasets; only the depth
    # resolution, device and output path come from this run's cfg.
    src = None
    if cfg.source_path is not None:
      if os.path.abspath(cfg.source_path) == os.path.abspath(cfg.output_path):
        raise ValueError("output_path must differ from source_path")
      src = stack.enter_context(h5py.File(cfg.source_path, "r"))
      cfg = OmegaConf.merge(
        json.loads(src.attrs["config_json"]),
        {k: cfg[k] for k in ("depth_width", "depth_height", "device",
                             "output_path", "source_path", "render_engine",
                             "eevee", "eevee_materials", "source_rerender_rgb",
                             "source_max_meshes", "depth_peel_rgb")})
      if cfg.source_max_meshes is not None and not cfg.source_rerender_rgb:
        raise ValueError("source_max_meshes needs source_rerender_rgb=true "
                         "(copied images can't be subset)")

    # Queue mode: the config (and the mesh list the jobs index into) comes from the queue; only this
    # worker's own keys come from the command line.
    queue = None
    if cfg.queue_path is not None:
      if src is not None:
        raise ValueError("queue_path doesn't combine with source_path")
      from job_queue import JobQueue
      queue = JobQueue(cfg.queue_path)
      cfg = OmegaConf.merge(queue.get_meta("config"), {k: cfg[k] for k in ("queue_path", "queue_worker", "output_path")})

    scene = bpy.context.scene
    # Per-engine settings are all set up front; _render_mesh switches
    # scene.render.engine per pass (RGB: cfg.render_engine, depth: Cycles).
    scene.render.engine = str(cfg.render_engine)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(cfg.film_transparent)
    scene.cycles.samples = int(cfg.cycles_samples)
    scene.cycles.use_denoising = True
    apply_props(scene.eevee, cfg.eevee)
    configure_device(str(cfg.device))

    W, H = int(cfg.width), int(cfg.height)
    if cfg.depth_width is None or cfg.depth_height is None:
      raise ValueError("depth_width/depth_height must be set (config default is "
                       "${width}/${height}; use that to match the RGB pass)")
    DW, DH = int(cfg.depth_width), int(cfg.depth_height)
    worker = None
    if queue is not None:
      worker = str(os.getpid()) if cfg.queue_worker is None else str(cfg.queue_worker)
    if src is None:
      view_strategy = hydra.utils.instantiate(cfg.view_strategy)
      # queue mode: the mesh list the jobs index into is stored in the queue
      meshes = queue.get_meta("meshes") if queue is not None else list(hydra.utils.instantiate(cfg.mesh_strategy).meshes())
      Lmax = int(cfg.max_peel_layers)
      mesh_views = lambda mi: _strategy_views(cfg, view_strategy, meshes[mi])
    else:
      meshes = list(src["mesh_paths"].asstr()[()])
      if cfg.source_max_meshes is not None:
        meshes = meshes[:int(cfg.source_max_meshes)]
      Lmax = int(src["depth_peel"].shape[-1])
      mesh_views = lambda mi: _source_views(src, mi)
    logger.info("RGB %dx%d, depth-peel %dx%d", W, H, DW, DH)

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    img_kw = dict(compression="gzip", compression_opts=4)
    depth_kw = dict(compression="gzip", compression_opts=4)

    tmp = stack.enter_context(TemporaryDirectory())
    hf = stack.enter_context(h5py.File(cfg.output_path, "w"))
    hf.attrs["config_json"] = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    hf.create_dataset("mesh_paths", data=meshes, dtype=h5py.string_dtype("utf-8"))

    # when re-rendering depth only, the RGB images are copied from the source
    render_rgb = cfg.render and (src is None or cfg.source_rerender_rgb)
    writer = TransactionalWriter(hf)
    if render_rgb:
      writer.init("images", dataset_kwargs=img_kw)
    writer.init("camera_pose")
    # two intrinsics datasets, both always written (identical when the RGB and
    # depth-peel resolutions match) so consumers never have to guess which K a
    # dataset goes with.
    writer.init("depth_intrinsics")  # matches depth_peel
    if cfg.render:
      writer.init("image_intrinsics")  # matches images
    writer.init("depth_peel", dataset_kwargs=depth_kw)
    if cfg.depth_peel_rgb:
      writer.init("depth_peel_rgb", dataset_kwargs=depth_kw)  # pixel- and layer-aligned with depth_peel
    writer.init("mesh_index", chunks=size_chunks)
    writer.init("depth_scale")

    failed = []
    for mi, job in _mesh_jobs(meshes, queue, worker, cfg.queue_max_meshes):
      mesh_path = meshes[mi]
      views = list(mesh_views(mi))
      if not views:
        continue  # re-render: mesh has no rows in the source (it failed there)
      try:
        with writer.transaction():
          _render_mesh(cfg, scene, mi, mesh_path, views, W, H, DW, DH, Lmax, tmp, writer)
      except Exception as e:
        if src is not None:
          raise  # skipping would misalign rows with the copied images
        logger.exception("mesh %d (%s) failed -- skipping", mi, mesh_path)
        failed.append(mi)
        if job is not None:
          queue.fail(job, worker, repr(e))
      else:
        if job is not None:
          queue.done(job, worker)   # only after the rows are committed to the file

    if src is not None:
      if "images" in src and not render_rgb:
        src.copy("images", hf)
      if "failed_mesh_indices" in src.attrs:
        hf.attrs["failed_mesh_indices"] = src.attrs["failed_mesh_indices"]
    if failed:
      hf.attrs["failed_mesh_indices"] = json.dumps(failed)
      logger.warning("%d/%d meshes failed: %s", len(failed), len(meshes), failed)

  logger.info("wrote %s", cfg.output_path)


def _mesh_jobs(meshes, queue, worker, max_meshes):
  """(mesh index, job) per mesh to render: every mesh in order with job None, or -- with a queue --
  the meshes this worker claims, until the queue has none ready or max_meshes were claimed."""
  if queue is None:
    for mi in range(len(meshes)):
      yield mi, None
    return
  claimed = 0
  while max_meshes is None or claimed < max_meshes:
    job = queue.claim(worker)
    if job is None:
      return
    claimed += 1
    yield int(job["payload"]["mesh_index"]), job


def _strategy_views(cfg, view_strategy, mesh_path):
  """(label, place(cam)) per view the strategy yields for one mesh."""
  for tilt_deg, az_deg, view_dir in view_strategy.views(mesh_path):
    vd = np.asarray(view_dir, np.float64)
    r = float(np.linalg.norm(vd)) or 1.0
    # A view strategy may encode a per-view camera distance as the view_dir
    # magnitude (see view_strategy.RandomViews); unit length means "use
    # cfg.camera_distance".
    dist = float(cfg.camera_distance) if abs(r - 1.0) < 1e-3 else r

    def place(cam, loc=Vector(vd / r * dist)):
      cam.location = loc
      aim_camera(cam)
    yield f"tilt={tilt_deg:.1f} az={az_deg:.1f}", place


def _source_views(src, mi):
  """(label, place(cam)) per row of mesh `mi` in a source render h5, putting
  the camera back where it was for that row."""
  for row in np.flatnonzero(src["mesh_index"][()] == mi):
    pose = src["camera_pose"][row].astype(np.float64)
    if "depth_scale" in src:
      pose[:3, 3] /= float(src["depth_scale"][row])  # undo camera_depth_target -> scene units

    def place(cam, m=Matrix(pose.tolist())):
      cam.matrix_world = m
    yield f"row={row}", place


def _render_mesh(cfg, scene, mi, mesh_path, views, W, H, DW, DH, Lmax, tmp, writer):
  """Load, normalize and render one mesh for every (label, place(cam)) in
  `views`, appending a row per view to the open datasets. Raises on any
  Blender failure (bad glb, degenerate bbox, ...) so run() can skip the mesh.
  RGB is rendered only when the writer has an "images" dataset.

  W,H is the RGB resolution; DW,DH the depth-peel resolution (equal unless
  cfg.depth_width/height override)."""
  from util import timed

  reset_scene()  # purges bpy.data objects/materials/images -> rebuild the rig
  load_object(mesh_path)
  if cfg.normalize_object:
    normalize_object()
  if cfg.render_engine != "CYCLES":  # Cycles output must not depend on these
    for mat in bpy.data.materials:
      apply_props(mat, cfg.eevee_materials)
  cam = setup_camera(cfg)
  setup_lighting(str(cfg.lighting))
  rl, comp = setup_depth_compositor(scene, bpy.context.view_layer)
  peel_mat, peel_tex, peel_eps = build_depth_peel_material()
  mesh_objs = scene_meshes()

  for view_label, place in views:
    place(cam)
    bpy.context.view_layer.update()
    label = f"mesh={mi} {view_label}"

    if "images" in writer.datasets:
      scene.render.engine = str(cfg.render_engine)
      with timed(f"{label} rgb ({scene.render.engine})"):
        rgba = render_rgba(os.path.join(tmp, "rgb.png"), W, H)
    with timed(f"{label} depth peel"):
      depth_vol, _, peel_rgb = depth_peel(mesh_objs, peel_mat, peel_tex, peel_eps,
                                          rl, comp, DW, DH, Lmax, tmp,
                                          with_rgb="depth_peel_rgb" in writer.datasets)

    pose = np.array(cam.matrix_world, np.float32)
    depth_intr = camera_intrinsics(cam.data, DW, DH)  # matches depth_vol
    image_intr = camera_intrinsics(cam.data, W, H)    # matches rgba (== depth_intr when equal res)

    depth_scale = 1.0
    if cfg.camera_depth_target is not None:
      hit = np.isfinite(depth_vol[..., 0])
      if hit.any():
        depth_scale = float(cfg.camera_depth_target) / float(depth_vol[..., 0][hit].mean())
        depth_vol = depth_vol * depth_scale  # NaN * s = NaN, so no-hit stays no-hit
        pose[:3, 3] *= depth_scale

    if "images" in writer.datasets:
      writer.append("images", rgba)
    if "image_intrinsics" in writer.datasets:
      writer.append("image_intrinsics", image_intr)
    writer.append("camera_pose", pose)
    writer.append("depth_intrinsics", depth_intr)
    writer.append("depth_peel", depth_vol.astype(np.float32))
    if peel_rgb is not None:
      writer.append("depth_peel_rgb", peel_rgb)
    writer.append("mesh_index", np.int64(mi))
    writer.append("depth_scale", np.float32(depth_scale))


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
