"""Mesh-selection strategies for capture_turntable.py, instantiated via
Hydra `_target_` from conf/mesh_strategy/*.yaml.
"""

import glob
import random
from dataclasses import dataclass
from typing import List


class MeshStrategy:
  def meshes(self) -> List[str]:
    raise NotImplementedError


@dataclass
class ListMeshes(MeshStrategy):
  paths: List[str]

  def meshes(self):
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
