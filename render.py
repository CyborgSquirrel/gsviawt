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
import sys
from argparse import ArgumentParser

import bpy
import mathutils
import rpyc
from rpyc.utils.server import ThreadPoolServer


def get_combined_bbox_corners_world(root_obj):
  """World-space bbox corners across root_obj and all its mesh descendants."""
  mesh_objects = [root_obj] if root_obj.type == 'MESH' else []
  mesh_objects += [child for child in root_obj.children_recursive if child.type == 'MESH']

  all_corners = []
  for obj in mesh_objects:
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

  def render(
    self,
    *,
    width: int,
    height: int,
    path: str,
    view_dir,
  ):
    scene = bpy.context.scene

    # Set render engine and output
    scene.render.engine = "CYCLES"  # or "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = path

    frame_object_robust(self.obj, scene.camera, scene, view_dir=view_dir)

    # Render a single frame
    bpy.ops.render.render(write_still=True)

    return {
      "pose_matrix": [list(row) for row in scene.camera.matrix_world],
    }
    # bpy.ops.wm.save_as_mainfile(filepath="/app/bla/curr.blend")

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
