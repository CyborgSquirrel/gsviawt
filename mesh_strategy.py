"""Mesh-selection strategies for render_objaverse.py, instantiated via
Hydra `_target_` from conf/mesh_strategy/*.yaml.
"""

import glob
import random
from dataclasses import dataclass
from typing import List, Optional


class MeshStrategy:
  def meshes(self) -> List[str]:
    raise NotImplementedError


@dataclass
class ListMeshes(MeshStrategy):
  """The given `paths`, or -- if `file` is set -- the paths in that text file (one per line)."""
  paths: List[str] = ()
  file: Optional[str] = None

  def meshes(self):
    if self.file is not None:
      with open(self.file) as f:
        return [line.strip() for line in f if line.strip()]
    return list(self.paths)


@dataclass
class RandomMeshes(MeshStrategy):
  """Picks `n` unique meshes matching `pattern` (e.g.
  "/app/data/objaverse/hf-objaverse-v1/glbs/*/*.glb"), sampled with a
  fixed `seed` for reproducibility.
  """
  pattern: str
  n: int
  seed: int

  def meshes(self):
    # Sorted so the sample is reproducible across filesystems/runs -- glob
    # order isn't guaranteed stable.
    candidates = sorted(glob.glob(self.pattern, recursive=True))
    return random.Random(self.seed).sample(candidates, self.n)
