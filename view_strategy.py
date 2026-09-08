"""View-sampling strategies for capture_turntable.py, instantiated via Hydra
`_target_` from conf/view_strategy/*.yaml.
"""

import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


def view_dir_for(azimuth_rad: float, tilt_rad: float) -> Tuple[float, float, float]:
  return (
    math.cos(tilt_rad) * math.cos(azimuth_rad),
    math.cos(tilt_rad) * math.sin(azimuth_rad),
    math.sin(tilt_rad),
  )


class ViewStrategy:
  def views(self) -> Iterator[Tuple[float, float, Tuple[float, float, float]]]:
    """Yields (tilt_deg, azimuth_deg, view_dir) tuples."""
    raise NotImplementedError


@dataclass
class SingleView(ViewStrategy):
  """A single fixed view. Useful for debugging mesh loading without paying
  for a whole turntable. Defaults to a level shot from the front (tilt 0,
  azimuth 0), matching the Turntable's first frame.
  """
  tilt_deg: float = 0.0
  azimuth_deg: float = 0.0

  def views(self):
    tilt_rad = math.radians(self.tilt_deg)
    azimuth_rad = math.radians(self.azimuth_deg)
    yield self.tilt_deg, self.azimuth_deg, view_dir_for(azimuth_rad, tilt_rad)


@dataclass
class RandomViews(ViewStrategy):
  """Objaverse-XL blender_script.py's camera sampling: `num_views` i.i.d.
  random viewpoints, no turntable structure. Each direction is a
  normalized `uniform(-1, 1, 3)` cube sample -- exactly as upstream, so
  mildly biased toward the cube corners rather than perfectly uniform on
  the sphere -- and every camera is aimed at the origin.

  Upstream also jitters the camera radius per view. Set `radius_min` /
  `radius_max` (upstream: 1.5 / 2.2) to reproduce that: the sampled radius
  is returned as the view_dir's magnitude, and render_objaverse.py reads a
  non-unit magnitude as an absolute camera distance (overriding
  cfg.camera_distance). Left None (default), every view_dir is unit length
  and the pipeline's fixed camera_distance is used -- consistent framing,
  which is what the WT comparison wants.

  `maxz` / `minz` clamp the (post-radius) camera height, matching
  upstream's `_sample_spherical`; the defaults (+/-2.2) are a no-op for a
  unit sphere. `seed` makes the draw reproducible; None leaves numpy's
  global RNG untouched.

  The RNG stream is shared across `views()` calls, so when a renderer
  calls this once per mesh each mesh gets fresh viewpoints (still
  reproducible from `seed`) rather than an identical repeated draw.
  """
  num_views: int = 12
  seed: Optional[int] = 0
  only_northern_hemisphere: bool = False
  maxz: float = 2.2
  minz: float = -2.2
  radius_min: Optional[float] = None
  radius_max: Optional[float] = None

  def __post_init__(self):
    self._rng = np.random.default_rng(self.seed) if self.seed is not None else np.random

  def views(self):
    rng = self._rng
    jitter = self.radius_min is not None and self.radius_max is not None

    produced = 0
    while produced < self.num_views:
      vec = rng.uniform(-1.0, 1.0, 3)
      norm = float(np.linalg.norm(vec))
      if norm == 0.0:
        continue
      radius = float(rng.uniform(self.radius_min, self.radius_max)) if jitter else 1.0
      vec = vec / norm * radius
      if not (self.maxz > vec[2] > self.minz):
        continue
      if self.only_northern_hemisphere:
        vec[2] = abs(vec[2])

      unit = vec / radius
      tilt_deg = math.degrees(math.asin(float(np.clip(unit[2], -1.0, 1.0))))
      az_deg = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
      yield tilt_deg, az_deg, (float(vec[0]), float(vec[1]), float(vec[2]))
      produced += 1


@dataclass
class Turntable(ViewStrategy):
  tilts_deg: List[float]
  num_azi: int

  def views(self):
    for tilt_deg in self.tilts_deg:
      tilt_rad = math.radians(tilt_deg)
      for az_idx in range(self.num_azi):
        azimuth_rad = az_idx * 2 * math.pi / self.num_azi
        yield tilt_deg, math.degrees(azimuth_rad), view_dir_for(azimuth_rad, tilt_rad)
