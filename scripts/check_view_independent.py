"""Blender-side check of render_objaverse.py's view_dependent=false material rewrite (no rendering):
for each mesh given, every material copy has no view-dependent Principled lobe / BSDF left, and the
originals are untouched. Run inside Blender:

    /opt/blender/blender --background --python scripts/check_view_independent.py -- a.glb [b.glb ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import render_objaverse as ro  # noqa: E402


def principled_state(nt, out):
  for n in nt.nodes:
    if n.type == "GROUP" and n.node_tree is not None:
      principled_state(n.node_tree, out)
    elif n.type == "BSDF_PRINCIPLED":
      out.append({name: (n.inputs[name].is_linked, n.inputs[name].default_value)
                  for name, _ in ro._VIEW_INDEPENDENT_PRINCIPLED if name in n.inputs})
    elif n.type in ro._VIEW_DEPENDENT_BSDFS:
      out.append(n.type)
  return out


def main():
  paths = sys.argv[sys.argv.index("--") + 1:]
  bad = 0
  for path in paths:
    ro.reset_scene()
    ro.load_object(path)
    objs = ro.scene_meshes()
    before = {m.name: principled_state(m.node_tree, []) for m in bpy_materials()}
    copies, _ = ro.build_view_independent_materials(objs)
    after = {m.name: principled_state(m.node_tree, []) for m in bpy_materials() if m not in copies.values()}
    if before != after:
      print("FAIL originals changed:", path); bad += 1
    for orig, mat in copies.items():
      for st in principled_state(mat.node_tree, []):
        if isinstance(st, str) or any(linked or abs(v - dict(ro._VIEW_INDEPENDENT_PRINCIPLED)[k]) > 1e-6
                                      for k, (linked, v) in st.items()):
          print("FAIL", path, getattr(orig, "name", None), st); bad += 1
    with ro._swapped_materials(objs, copies):
      assert all(m in copies.values() for o in objs for m in o.data.materials)
    assert all(m not in copies.values() for o in objs for m in o.data.materials)
    print("ok", path, f"{len(copies)} materials")
  print("FAILURES:", bad)
  sys.exit(1 if bad else 0)


def bpy_materials():
  import bpy
  return list(m for m in bpy.data.materials if m.use_nodes and m.node_tree is not None)


main()
