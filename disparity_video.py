#!/usr/bin/env python3
"""Orbit a camera around two coloured point clouds and write a side-by-side
[GT | WT | both] turntable .mp4 -- the moving-camera companion to the static
.ply pair that debug_cloud_disparity.py writes.

Input is just two .ply point clouds; their per-vertex colours are used as-is
(so debug_cloud_disparity's pale->blue / pale->red disparity colouring
carries straight through, but any two coloured clouds work). Each cloud is
drawn as a cloud of small isotropic dots via `gsplat.rasterization` along the
same orbit, and the two are also composited into an `both` panel so you can
watch where the blue (render-only) and red (WT-only) geometry sits relative
to the shared surface.

The clouds are assumed to be OpenCV camera space (what debug_cloud_disparity
emits); `upright: true` (default) remaps screen-up onto the orbit axis so the
turntable spins the object about its vertical.

Config is Hydra (`conf/disparity_video.yaml`); run in the container venv:

    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python \\
        disparity_video.py \\
        gt=/app/bla/x.disparity.gt.ply wt=/app/bla/x.disparity.wt.ply \\
        caption="view 400  CD 0.131" seconds=8
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

# orbit_video pulls in fit_gsplat (CUDA toolchain for gsplat's JIT) on import.
from orbit_video import (  # noqa: E402
  orbit_poses, render_frames, write_frames_dir, write_mp4,
)
from util import timed  # noqa: E402

log = logging.getLogger("disparity_video")

# camera space (X right, Y down, Z fwd) -> Z-up world (X right, Y fwd, Z up):
# screen-up (-Y_cam) maps to +Z, the orbit axis.
_CAM_TO_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)


def load_cloud(path):
  """.ply -> (xyz (N,3) float32, rgb (N,3) float32 in [0,1]). Drops non-finite
  points."""
  import trimesh

  pc = trimesh.load(path, process=False)
  if not hasattr(pc, "vertices"):
    raise SystemExit(f"{path}: not a point cloud / mesh trimesh can read")
  xyz = np.asarray(pc.vertices, np.float32)
  col = np.asarray(pc.colors, np.float32)
  if col.ndim != 2 or col.shape[0] != len(xyz):
    col = np.full((len(xyz), 3), 200.0, np.float32)   # uncoloured -> light grey
  rgb = np.clip(col[:, :3] / 255.0, 0.0, 1.0)
  ok = np.isfinite(xyz).all(1)
  if not ok.all():
    xyz, rgb = xyz[ok], rgb[ok]
  if len(xyz) == 0:
    raise SystemExit(f"{path}: no finite points")
  return xyz, rgb


def as_gaussians(xyz, rgb, point_size):
  """A point cloud dressed up as the gaussian dict render_frames expects:
  isotropic `point_size` dots, opaque, identity rotation."""
  n = len(xyz)
  return {
    "means": np.ascontiguousarray(xyz, np.float32),
    "scales": np.full((n, 3), float(point_size), np.float32),
    "quats": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),   # wxyz identity
    "opacities": np.ones(n, np.float32),
    "colors": np.ascontiguousarray(rgb, np.float32),
  }


def _strip(panels, labels, caption):
  """panels: list of equal-length lists of (H,W,3) uint8. Yields one hstacked
  frame per index with a label bar (+ optional caption) on top."""
  from PIL import Image, ImageDraw

  n = len(panels[0])
  h, w = panels[0][0].shape[:2]
  strip = 34 if caption else 20
  for i in range(n):
    row = np.concatenate([p[i] for p in panels], axis=1)
    canvas = Image.new("RGB", (row.shape[1], h + strip), (16, 16, 18))
    canvas.paste(Image.fromarray(row), (0, strip))
    d = ImageDraw.Draw(canvas)
    if caption:
      d.text((6, 4), str(caption), fill=(150, 210, 255))
    labely = 18 if caption else 5
    for j, lab in enumerate(labels):
      d.text((j * w + 6, labely), lab, fill=(230, 230, 230))
    yield np.asarray(canvas)


@hydra.main(version_base=None, config_path="conf", config_name="disparity_video")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  np.random.seed(int(cfg.seed))
  torch.manual_seed(int(cfg.seed))

  fmt = str(cfg.format).lower()
  if fmt not in ("mp4", "frames"):
    raise SystemExit(f"unknown format {fmt!r} (want mp4 or frames)")
  if fmt == "mp4":  # yuv420p needs even dimensions
    cfg.width = int(cfg.width) + int(cfg.width) % 2
    cfg.height = int(cfg.height) + int(cfg.height) % 2

  with timed("load"):
    gt_xyz, gt_rgb = load_cloud(cfg.gt)
    wt_xyz, wt_rgb = load_cloud(cfg.wt)

  # shared frame: recenter on the combined centroid, remap to Z-up world.
  both_xyz = np.concatenate([gt_xyz, wt_xyz])
  center = both_xyz.mean(0)
  gt_xyz, wt_xyz = gt_xyz - center, wt_xyz - center
  if bool(cfg.upright):
    gt_xyz = gt_xyz @ _CAM_TO_UP.T
    wt_xyz = wt_xyz @ _CAM_TO_UP.T
  log.info("GT %d pts  WT %d pts", len(gt_xyz), len(wt_xyz))

  poses, K, dist, orbit_center, n = orbit_poses(cfg, np.concatenate([gt_xyz, wt_xyz]))
  log.info("%d frames  %dx%d  elevation %.1f deg  dist %.3f",
           n, int(cfg.width), int(cfg.height), float(cfg.elevation_deg), dist)

  device = "cuda" if (str(cfg.device) == "cuda" and torch.cuda.is_available()) else "cpu"
  if device != str(cfg.device):
    log.warning("cuda not available, falling back to cpu")

  ps = float(cfg.point_size)
  g_gt = as_gaussians(gt_xyz, gt_rgb, ps)
  g_wt = as_gaussians(wt_xyz, wt_rgb, ps)
  g_both = {k: np.concatenate([g_gt[k], g_wt[k]]) for k in g_gt}

  with timed("render GT"):
    f_gt = render_frames(cfg, g_gt, poses, K, device)
  with timed("render WT"):
    f_wt = render_frames(cfg, g_wt, poses, K, device)
  with timed("render both"):
    f_both = render_frames(cfg, g_both, poses, K, device)
  if device == "cuda":
    torch.cuda.empty_cache()

  out = cfg.output_path
  if not out:
    stem = os.path.splitext(cfg.gt)[0]
    stem = stem[:-3] if stem.endswith(".gt") else stem   # ...view400.disparity
    out = stem + ".orbit" + (".mp4" if fmt == "mp4" else ".frames")

  frames = _strip([f_gt, f_wt, f_both],
                  ["GT  (dist->blue)", "WT  (dist->red)", "overlaid"], cfg.caption)
  with timed("encode"):
    if fmt == "frames":
      write_frames_dir(frames, out)
    else:
      write_mp4(frames, out, cfg.fps, cfg.crf)
  log.info("wrote %s", out)


if __name__ == "__main__":
  main()
