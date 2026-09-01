def _head():
  import os
  import sys
  path = os.environ.get("BLENDER_USER_PYTHON", "")
  if len(path.strip()) <= 0:
    return
  sys.path.append(path)
_head(); del _head

import contextlib as ctl
import math
import os
import sys
from argparse import ArgumentParser

import bpy
import mathutils
import numpy as np
import rpyc
from rpyc.utils.server import ThreadPoolServer


def get_mesh_objects(root_obj):
  mesh_objects = [root_obj] if root_obj.type == 'MESH' else []
  mesh_objects += [child for child in root_obj.children_recursive if child.type == 'MESH']
  return mesh_objects

def get_combined_bbox_corners_world(root_obj):
  """World-space bbox corners across root_obj and all its mesh descendants."""
  all_corners = []
  for obj in get_mesh_objects(root_obj):
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    all_corners.extend(corners)

  return all_corners

def get_combined_bbox_center_and_extents(root_obj):
  corners = get_combined_bbox_corners_world(root_obj)
  if not corners:
    return mathutils.Vector((0, 0, 0)), mathutils.Vector((0, 0, 0))

  xs = [c.x for c in corners]
  ys = [c.y for c in corners]
  zs = [c.z for c in corners]

  min_v = mathutils.Vector((min(xs), min(ys), min(zs)))
  max_v = mathutils.Vector((max(xs), max(ys), max(zs)))
  center = (min_v + max_v) / 2
  extents = max_v - min_v
  return center, extents

# def get_bbox_corners_world(obj):
#   """World-space bounding box corners."""
#   return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def get_bbox_center(corners):
  return sum(corners, mathutils.Vector()) / len(corners)

def look_at(cam_obj, target_point):
  direction = target_point - cam_obj.location
  rot_quat = direction.to_track_quat('-Z', 'Y')
  cam_obj.rotation_euler = rot_quat.to_euler()

def get_camera_intrinsics(cam_data, width, height):
  """3x3 pinhole intrinsics matrix for cam_data at the given render
  resolution (assumes square pixels, i.e. render.pixel_aspect_x/y == 1,
  which is what reset() leaves them at). Same sensor_fit handling as
  frame_object_robust's FOV computation, so cross-check against that if
  the camera setup ever changes.
  """
  aspect = width / height
  sensor_fit = cam_data.sensor_fit
  if sensor_fit == 'AUTO':
    sensor_fit = 'HORIZONTAL' if aspect >= 1.0 else 'VERTICAL'

  if sensor_fit == 'HORIZONTAL':
    focal_px = width * cam_data.lens / cam_data.sensor_width
  else:
    focal_px = height * cam_data.lens / cam_data.sensor_height

  cx = width / 2 - cam_data.shift_x * max(width, height)
  cy = height / 2 + cam_data.shift_y * max(width, height)

  return [
    [focal_px, 0.0, cx],
    [0.0, focal_px, cy],
    [0.0, 0.0, 1.0],
  ]

def frame_object_robust(
  obj,
  cam_obj,
  scene,
  view_dir=None,
  margin=1.05,
):
  """
  Position cam_obj so obj is fully framed, using actual FOV/sensor/aspect,
  not a rough distance multiplier.

  view_dir: unit vector pointing FROM object TOWARD camera (i.e. the direction
            you want to view from). Defaults to a nice 3/4 angle.
  margin: >1.0 adds headroom (1.05 = 5% padding around the object)
  """
  cam_data = cam_obj.data
  corners = get_combined_bbox_corners_world(obj)
  center = get_bbox_center(corners)

  # Handle view_dir
  if view_dir is None:
    view_dir = (1, -1, 0.6)
  if isinstance(view_dir, tuple):
    view_dir = mathutils.Vector(view_dir)
  if isinstance(view_dir, mathutils.Vector):
    view_dir = view_dir.normalized()

  # Orient the camera first (rotation doesn't depend on distance)
  # temp placement just to get correct basis vectors
  cam_obj.location = center + view_dir
  look_at(cam_obj, center)
  bpy.context.view_layer.update()  # ensure matrix_world is current

  # Camera's local axes in world space, after orientation
  cam_matrix = cam_obj.matrix_world
  cam_right = cam_matrix.to_3x3() @ mathutils.Vector((1, 0, 0))
  cam_up    = cam_matrix.to_3x3() @ mathutils.Vector((0, 1, 0))
  cam_fwd   = cam_matrix.to_3x3() @ mathutils.Vector((0, 0, -1))  # camera looks down -Z

  # Half-FOV angles from actual sensor/lens/resolution
  render = scene.render
  res_x, res_y = render.resolution_x, render.resolution_y
  aspect = res_x / res_y

  sensor_fit = cam_data.sensor_fit
  if sensor_fit == 'AUTO':
    sensor_fit = 'HORIZONTAL' if aspect >= 1.0 else 'VERTICAL'

  if cam_data.type != 'ORTHO':
    if sensor_fit == 'HORIZONTAL':
      half_fov_h = math.atan((cam_data.sensor_width / 2) / cam_data.lens)
      half_fov_v = math.atan(math.tan(half_fov_h) / aspect)
    else:
      half_fov_v = math.atan((cam_data.sensor_height / 2) / cam_data.lens)
      half_fov_h = math.atan(math.tan(half_fov_v) * aspect)
  else:
    # Orthographic: no perspective distance solve needed, just scale.
    # Project corners onto right/up axes relative to center, find max extents.
    max_right = max(abs((c - center).dot(cam_right)) for c in corners)
    max_up    = max(abs((c - center).dot(cam_up)) for c in corners)
    cam_data.ortho_scale = 2 * max(max_right, max_up / aspect) * margin
    cam_obj.location = center + view_dir * (max((c - center).length for c in corners) * 2)
    return

  # For each corner, find the distance along cam_fwd needed so it stays
  # within the horizontal and vertical half-angle, then take the max
  # (i.e. the most constraining corner) required distance.
  max_dist = 0.0
  for c in corners:
    rel = c - center
    d_fwd   = rel.dot(cam_fwd)     # depth offset from center along view dir (usually small)
    d_right = abs(rel.dot(cam_right))
    d_up    = abs(rel.dot(cam_up))

    # distance needed so this corner's lateral offset fits within the half-FOV cone,
    # accounting for its own depth offset relative to the object's center
    dist_for_h = d_right / math.tan(half_fov_h) - d_fwd
    dist_for_v = d_up / math.tan(half_fov_v) - d_fwd

    max_dist = max(max_dist, dist_for_h, dist_for_v)

  max_dist *= margin
  cam_obj.location = center + view_dir * max_dist
  look_at(cam_obj, center)

@ctl.contextmanager
def track_imported_objects():
  """Yields a list that gets populated with the top-level objects
  (objects with no parent) created during the `with` block.
  """
  before = set(obj for obj in bpy.data.objects if obj.parent is None)
  imported_objects = []
  try:
    yield imported_objects
  finally:
    after = set(obj for obj in bpy.data.objects if obj.parent is None)
    imported_objects.extend(after - before)

def build_depth_peel_material():
  """A shared material implementing shader-based depth peeling: a fragment
  renders opaque (Diffuse) if its camera-space depth is strictly ahead of
  the previous peel pass's recorded depth (sampled from an image texture via
  screen-space "Window" coordinates), otherwise it's discarded (Transparent),
  letting the primary ray continue to the next surface behind it.

  Returns (material, image_texture_node, epsilon_node) so callers can swap
  the texture's source image and tune epsilon per mesh.
  """
  mat = bpy.data.materials.new("DepthPeelMaterial")
  mat.use_nodes = True
  nt = mat.node_tree
  nt.nodes.clear()

  output = nt.nodes.new("ShaderNodeOutputMaterial")
  cam_data = nt.nodes.new("ShaderNodeCameraData")
  tex_coord = nt.nodes.new("ShaderNodeTexCoord")
  img_tex = nt.nodes.new("ShaderNodeTexImage")
  img_tex.interpolation = 'Closest'
  img_tex.extension = 'EXTEND'
  add_eps = nt.nodes.new("ShaderNodeMath")
  add_eps.operation = 'ADD'
  add_eps.inputs[1].default_value = 1e-4
  gt = nt.nodes.new("ShaderNodeMath")
  gt.operation = 'GREATER_THAN'
  transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
  diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
  mix = nt.nodes.new("ShaderNodeMixShader")

  links = nt.links
  links.new(tex_coord.outputs["Window"], img_tex.inputs["Vector"])
  links.new(img_tex.outputs["Color"], add_eps.inputs[0])
  links.new(cam_data.outputs["View Z Depth"], gt.inputs[0])
  links.new(add_eps.outputs[0], gt.inputs[1])
  links.new(gt.outputs[0], mix.inputs[0])
  links.new(transparent.outputs["BSDF"], mix.inputs[1])
  links.new(diffuse.outputs["BSDF"], mix.inputs[2])
  links.new(mix.outputs["Shader"], output.inputs["Surface"])

  return mat, img_tex, add_eps

def setup_depth_compositor(scene, view_layer):
  """Set up a Render Layers -> Composite compositor tree, defaulting the
  Composite input to the normal combined Image pass (so plain write_still
  RGB renders are unaffected -- an unconnected Composite node input saves
  as black, so it must always have *something* valid linked). Depth-peel
  passes temporarily reroute Composite's input to the Z (Depth) output and
  write it out as an EXR via the same write_still=True + explicit filepath
  mechanism the RGB render already uses -- the Viewer Node approach doesn't
  reliably update its pixel buffer in --background mode (confirmed
  empirically: it stays stuck at a placeholder 256x256 buffer regardless of
  actual render resolution).

  Returns (render_layers_node, composite_node).
  """
  view_layer.use_pass_z = True
  scene.use_nodes = True
  nt = scene.node_tree
  nt.nodes.clear()
  rl = nt.nodes.new("CompositorNodeRLayers")
  rl.layer = view_layer.name
  composite = nt.nodes.new("CompositorNodeComposite")
  nt.links.new(rl.outputs["Image"], composite.inputs["Image"])
  return rl, composite

class Service(rpyc.Service):
  def reset(self, path: str):
    # Remove all default objects (Cube, Camera, Light)
    for obj in list(bpy.data.objects):
      bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Light
    light_data = bpy.data.lights.new("Light", type="SUN")
    light_obj = bpy.data.objects.new("Light", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, -5, 10)

    # Load mesh
    with track_imported_objects() as imported_objects:
      bpy.ops.import_scene.gltf(filepath=path)

    if len(imported_objects) != 1:
      raise RuntimeError()

    self.obj = imported_objects[0]

    self._peel_rl, self._peel_composite = setup_depth_compositor(bpy.context.scene, bpy.context.view_layer)
    self._peel_mat, self._peel_img_tex, self._peel_eps_node = build_depth_peel_material()

  def render(
    self,
    *,
    width: int,
    height: int,
    path: str,
    view_dir,
    max_layers: int,
    depth_path: str,
  ):
    scene = bpy.context.scene

    # Set render engine and output
    scene.render.engine = "CYCLES"  # or "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = path

    frame_object_robust(self.obj, scene.camera, scene, view_dir=view_dir)

    # Render a single frame (real materials, normal quality)
    bpy.ops.render.render(write_still=True)

    pose_matrix = [list(row) for row in scene.camera.matrix_world]
    intrinsics_matrix = get_camera_intrinsics(scene.camera.data, width, height)

    depth_volume, num_layers_found = self._depth_peel(width, height, max_layers, os.path.dirname(depth_path))
    np.save(depth_path, depth_volume)

    return {
      "pose_matrix": pose_matrix,
      "intrinsics_matrix": intrinsics_matrix,
      "num_layers_found": num_layers_found,
    }
    # bpy.ops.wm.save_as_mainfile(filepath="/app/bla/curr.blend")

  def _depth_peel(self, width, height, max_layers, scratch_dir):
    """Shader-based depth peeling from the current scene.camera (as just
    positioned by frame_object_robust). Returns (volume, num_layers_found)
    where volume has shape (height, width, max_layers), float32, with -1.0
    marking rays that had fewer than max_layers intersections.
    """
    scene = bpy.context.scene
    cyc = scene.cycles
    img_settings = scene.render.image_settings

    saved_filepath = scene.render.filepath
    saved_file_format = img_settings.file_format
    saved_color_depth = img_settings.color_depth
    saved_color_mode = img_settings.color_mode
    img_settings.file_format = 'OPEN_EXR'
    img_settings.color_depth = '32'
    img_settings.color_mode = 'BW'

    scene.node_tree.links.new(
      self._peel_rl.outputs["Depth"], self._peel_composite.inputs["Image"])

    mesh_objects = get_mesh_objects(self.obj)
    extents = get_combined_bbox_center_and_extents(self.obj)[1]
    eps = max(extents.length * 1e-4, 1e-6)
    self._peel_eps_node.inputs[1].default_value = eps

    # Swap in the peel material on every mesh object, saving originals to restore after.
    saved_slots = []
    for obj in mesh_objects:
      materials = obj.data.materials
      added_slot = len(materials) == 0
      if added_slot:
        materials.append(None)
      saved = list(materials)
      saved_slots.append((obj, saved, added_slot))
      for i in range(len(materials)):
        materials[i] = self._peel_mat

    # Strip Cycles down for speed: primary-ray Z depth doesn't need samples,
    # bounces, or denoising. transparent_max_bounces is the one budget that
    # MUST stay high -- the primary ray needs to pass through one already-
    # peeled (transparent) surface per prior layer to reach the next one.
    saved_settings = {
      "samples": cyc.samples,
      "use_denoising": cyc.use_denoising,
      "max_bounces": cyc.max_bounces,
      "diffuse_bounces": cyc.diffuse_bounces,
      "glossy_bounces": cyc.glossy_bounces,
      "transmission_bounces": cyc.transmission_bounces,
      "volume_bounces": cyc.volume_bounces,
      "transparent_max_bounces": cyc.transparent_max_bounces,
    }
    cyc.samples = 1
    cyc.use_denoising = False
    cyc.max_bounces = 0
    cyc.diffuse_bounces = 0
    cyc.glossy_bounces = 0
    cyc.transmission_bounces = 0
    cyc.volume_bounces = 0
    cyc.transparent_max_bounces = max_layers + 2

    prev_depth = np.full((height, width), -1e6, dtype=np.float32)
    layers = np.full((height, width, max_layers), -1.0, dtype=np.float32)
    num_layers_found = 0

    prev_img = bpy.data.images.new("DepthPeelPrev", width=width, height=height, float_buffer=True)
    self._peel_img_tex.image = prev_img

    try:
      for layer_idx in range(max_layers):
        self._write_depth_image(prev_img, prev_depth)

        layer_path = os.path.join(scratch_dir, f"_depth_peel_layer_{layer_idx}.exr")
        scene.render.filepath = layer_path
        bpy.ops.render.render(write_still=True)
        depth = self._read_exr_depth(layer_path, width, height)
        os.remove(layer_path)

        hit_mask = depth < 1e9
        if not hit_mask.any():
          break

        layers[:, :, layer_idx] = np.where(hit_mask, depth, -1.0)
        num_layers_found = layer_idx + 1
        prev_depth = np.where(hit_mask, depth, prev_depth)
    finally:
      for obj, saved, added_slot in saved_slots:
        materials = obj.data.materials
        for i, mat in enumerate(saved):
          materials[i] = mat
        if added_slot:
          materials.pop(index=len(materials) - 1)

      for key, value in saved_settings.items():
        setattr(cyc, key, value)

      scene.node_tree.links.new(
        self._peel_rl.outputs["Image"], self._peel_composite.inputs["Image"])
      scene.render.filepath = saved_filepath
      img_settings.file_format = saved_file_format
      img_settings.color_depth = saved_color_depth
      img_settings.color_mode = saved_color_mode

      self._peel_img_tex.image = None
      bpy.data.images.remove(prev_img)

    # Blender image buffers are bottom-row-first; flip to top-row-first so
    # depth_volume[i, j] lines up with the RGB image's row i.
    layers = np.flip(layers, axis=0)
    return layers, num_layers_found

  def _write_depth_image(self, img, depth):
    height, width = depth.shape
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., 0] = depth
    rgba[..., 1] = depth
    rgba[..., 2] = depth
    rgba[..., 3] = 1.0
    img.pixels.foreach_set(rgba.ravel())
    img.update()

  def _read_exr_depth(self, path, width, height):
    img = bpy.data.images.load(path, check_existing=False)
    try:
      pixels = np.empty(width * height * 4, dtype=np.float32)
      img.pixels.foreach_get(pixels)
      pixels = pixels.reshape(height, width, 4)
      return pixels[:, :, 0]
    finally:
      bpy.data.images.remove(img)

def main():
  sep_idx = sys.argv.index("--")
  if sep_idx == -1:
    raise RuntimeError("Missing command-line args separator")
  raw_args = sys.argv[sep_idx+1:]

  parser = ArgumentParser()
  parser.add_argument("socket_path")
  args = parser.parse_args(raw_args)

  server = ThreadPoolServer(
    Service,
    socket_path=args.socket_path,
    nbThreads=1,  # NOTE: Only one thread!!! Otherwise they will crash into eachother.
    protocol_config={"allow_public_attrs": True},
  )
  server.start()

main()
