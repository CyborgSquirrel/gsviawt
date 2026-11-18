"""render_objaverse view_dependent option, the parts that run without Blender: the config default keeps
the old behaviour, and render_merge carries the output's view_dependent attr from the parts into the
merged file (the Blender side is scripts/check_view_independent.py)."""
import json
import os

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from render_merge import merge_parts, scan_parts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_is_view_dependent():
  cfg = OmegaConf.load(os.path.join(ROOT, "conf", "objaverse.yaml"))
  assert cfg.view_dependent is True


def _part(path, mesh, view_dependent):
  with h5py.File(path, "w") as f:
    f.attrs["config_json"] = json.dumps({"view_dependent": view_dependent})
    if view_dependent is not None:
      f.attrs["view_dependent"] = view_dependent
    f.create_dataset("mesh_paths", data=["a.glb", "b.glb"], dtype=h5py.string_dtype("utf-8"))
    f.create_dataset("mesh_index", data=np.full(2, mesh, np.int64))
    f.create_dataset("images", data=np.full((2, 4, 4, 4), mesh, np.uint8))


@pytest.mark.parametrize("view_dependent", [True, False, None])
def test_merge_keeps_view_dependent_attr(tmp_path, view_dependent):
  parts = [str(tmp_path / f"part_{i}.h5") for i in range(2)]
  for i, p in enumerate(parts):
    _part(p, i, view_dependent)
  out = str(tmp_path / "merged.h5")
  merge_parts(scan_parts(parts), ["a.glb", "b.glb"], out)
  with h5py.File(out, "r") as f:
    if view_dependent is None:  # files from before the option: no attr, read as true
      assert "view_dependent" not in f.attrs
    else:
      assert bool(f.attrs["view_dependent"]) is view_dependent
    assert json.loads(f.attrs["config_json"])["view_dependent"] == view_dependent
    assert f["images"].shape[0] == 4
