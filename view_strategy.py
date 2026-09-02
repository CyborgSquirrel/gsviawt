"""View-sampling strategies for capture_turntable.py, instantiated via Hydra
`_target_` from conf/view_strategy/*.yaml.
"""

import math
from dataclasses import dataclass
from typing import Iterator, List, Tuple


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
class Turntable(ViewStrategy):
  tilts_deg: List[float]
  num_azi: int

  def views(self):
    for tilt_deg in self.tilts_deg:
      tilt_rad = math.radians(tilt_deg)
      for az_idx in range(self.num_azi):
        azimuth_rad = az_idx * 2 * math.pi / self.num_azi
        yield tilt_deg, math.degrees(azimuth_rad), view_dir_for(azimuth_rad, tilt_rad)
