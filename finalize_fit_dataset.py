#!/usr/bin/env python3
"""Collect a dataset of layered 3DGS fits (fit_3dgs.py gaussian_layout=layered, one run dir per source view, named `mesh<M>_view<V>`)
into the form gs_diffusion.py reads:
    <root>/grids/<name>.h5       the fit's gaussian grid (H, W, L, .) with the fit's metadata stamped into its attributes
    <root>/extra/<name>/...      everything else the run wrote (flat gaussians.h5, .ply, logs, .hydra); moved, not deleted

Only complete fits are collected (grid, flat h5 and ply all there, and no writer holding the files); the others are listed and left
where they are, so it can be run again later, also while fits are still running. An existing grids/<name>.h5 is never overwritten
(overwrite=true to replace it).

    python finalize_fit_dataset.py root=/scratch/vc16/t26rb00/fit_dataset
"""

import json
import logging
import os
import shutil

import h5py
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

NEEDED = ("gaussians.grid.h5", "gaussians.h5", "gaussians.ply")


def finalize_one(root, name, overwrite):
  """Moves one fit into grids/ + extra/. Returns None when done, else the reason it was left alone."""
  run = os.path.join(root, name)
  missing = [f for f in NEEDED if not os.path.exists(os.path.join(run, f))]
  if missing:
    return f"incomplete (no {', '.join(missing)})"
  target = os.path.join(root, "grids", name + ".h5")
  if os.path.exists(target) and not overwrite:
    return "grids/ already has it (overwrite=true to replace)"
  grid, flat = os.path.join(run, "gaussians.grid.h5"), os.path.join(run, "gaussians.h5")
  try:
    with h5py.File(flat, "r") as f:
      attrs = dict(f.attrs)
    cfg = json.loads(attrs["config_json"])
    if cfg.get("gaussian_layout") != "layered":
      return f"not a layered fit (gaussian_layout={cfg.get('gaussian_layout')!r})"
    mesh, view = (int(x) for x in name[len("mesh"):].split("_view"))
    with h5py.File(grid, "r+") as g:   # also fails (BlockingIOError) while a fit still writes it
      g.attrs.update({
        "source_h5": cfg["hdf5_path"],
        "mesh_number": mesh,
        "primary_view_in_mesh": view,
        "primary_view_row": int(cfg["data"]["primary_view"]),
        "supervision_views_rows": json.dumps(cfg["data"]["secondary_views"]),   # the fit's other training views
        "num_supervision_views": len(cfg["data"]["secondary_views"]) + 1,       # + the primary
        "iters": int(cfg["iters"]),
        "sh_degree": int(cfg["sh_degree"]),
        "mean_offset_reg_weight": float(cfg["mean_offset_reg_weight"]),
        "means_optimized": True,   # layered: means = the depth-peel seed + a trained offset
        "final_loss": float(attrs["final_loss"]),
        "num_gaussians": int(attrs["num_gaussians"]),
        "config_json": attrs["config_json"],
      })
  except (OSError, KeyError, ValueError) as e:
    return f"{type(e).__name__}: {e}"
  os.makedirs(os.path.dirname(target), exist_ok=True)
  extra = os.path.join(root, "extra", name)
  if os.path.exists(extra):
    shutil.rmtree(extra)
  shutil.move(grid, target)
  os.makedirs(os.path.dirname(extra), exist_ok=True)
  shutil.move(run, extra)
  return None


@hydra.main(version_base=None, config_path="conf", config_name="finalize_fit_dataset")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  names = sorted(n for n in os.listdir(cfg.root) if n.startswith("mesh") and "_view" in n and os.path.isdir(os.path.join(cfg.root, n)))
  done, left = 0, {}
  for name in names:
    reason = finalize_one(cfg.root, name, cfg.overwrite)
    if reason is None:
      done += 1
    else:
      left[name] = reason
  log.info("finalized %d fits into %s/grids; %d left where they are", done, cfg.root, len(left))
  for name, reason in left.items():
    log.info("  %s: %s", name, reason)


if __name__ == "__main__":
  main()
