#!/usr/bin/env python3
"""Collect a dataset of layered 3DGS fits (fit_3dgs.py gaussian_layout=layered, one run dir per source view, named `mesh<M>_view<V>`)
into the form gs_diffusion.py reads:
    <root>/grids/<name>.h5       the fit's gaussian grid (H, W, L, .) with the fit's metadata stamped into its attributes
    <root>/extra/<name>/...      everything else the run wrote (flat gaussians.h5, .ply, logs, .hydra); moved, not deleted
                                 (grid-only fits, e.g. fit_anchored.py's, get no extra/ entry when nothing else is left)

Only complete fits are collected (grid, flat h5 and ply all there -- or, for a grid-only fit, the grid with the fit's metadata
(config_json, final_loss, num_gaussians) in its own attributes -- and no writer holding the files); the others are listed and left
where they are, so it can be run again later, also while fits are still running. An existing grids/<name>.h5 is never overwritten
(overwrite=true to replace it).

grid_format=sparse writes grids/<name>.h5 in grid_io.py's sparse layout instead (verified against the fit's dense grid, which is
kept: it goes to extra/<name>/gaussians.grid.h5 with the rest of the run). Default dense, as before.

    python finalize_fit_dataset.py root=/scratch/vc16/t26rb00/fit_dataset
"""

import json
import logging
import os
import shutil

import h5py
import hydra
from omegaconf import DictConfig

import grid_io

log = logging.getLogger(__name__)

NEEDED = ("gaussians.grid.h5", "gaussians.h5", "gaussians.ply")


GRID_FORMATS = ("dense", "sparse")


def finalize_one(root, name, overwrite, grid_format="dense"):
  """Moves one fit into grids/ + extra/. Returns None when done, else the reason it was left alone."""
  run = os.path.join(root, name)
  grid_only = os.path.exists(os.path.join(run, NEEDED[0])) and not os.path.exists(os.path.join(run, "gaussians.h5"))
  if grid_only:
    try:
      with h5py.File(os.path.join(run, NEEDED[0]), "r") as g:
        grid_only = "config_json" in g.attrs
    except OSError as e:   # e.g. still being written
      return f"{type(e).__name__}: {e}"
  missing = [f for f in (NEEDED[:1] if grid_only else NEEDED) if not os.path.exists(os.path.join(run, f))]
  if missing:
    return f"incomplete (no {', '.join(missing)})"
  target = os.path.join(root, "grids", name + ".h5")
  if os.path.exists(target) and not overwrite:
    return "grids/ already has it (overwrite=true to replace)"
  grid, flat = os.path.join(run, "gaussians.grid.h5"), os.path.join(run, "gaussians.h5")
  try:
    with h5py.File(grid if grid_only else flat, "r") as f:
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
        # how the layered offsets were pulled toward their slot: AdamW weight decay (current) or, in fits made
        # before it replaced it, an L2 loss weight
        **({"means_weight_decay": float(cfg["lr"]["means_weight_decay"])} if "means_weight_decay" in cfg.get("lr", {})
           else {"mean_offset_reg_weight": float(cfg["mean_offset_reg_weight"])}),
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
  if grid_format == "sparse":   # the dense grid stays in the run dir, so it moves to extra/ below
    try:
      grid_io.convert_file(grid, target)
    except (OSError, ValueError, RuntimeError) as e:
      return f"sparse conversion failed: {type(e).__name__}: {e}"
  else:
    shutil.move(grid, target)
  if os.listdir(run):
    os.makedirs(os.path.dirname(extra), exist_ok=True)
    shutil.move(run, extra)
  else:
    os.rmdir(run)
  return None


@hydra.main(version_base=None, config_path="conf", config_name="finalize_fit_dataset")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  if cfg.grid_format not in GRID_FORMATS:
    raise SystemExit(f"grid_format must be one of {GRID_FORMATS}, got {cfg.grid_format!r}")
  names = sorted(n for n in os.listdir(cfg.root) if n.startswith("mesh") and "_view" in n and os.path.isdir(os.path.join(cfg.root, n)))
  done, left = 0, {}
  for name in names:
    reason = finalize_one(cfg.root, name, cfg.overwrite, cfg.grid_format)
    if reason is None:
      done += 1
    else:
      left[name] = reason
  log.info("finalized %d fits into %s/grids; %d left where they are", done, cfg.root, len(left))
  for name, reason in left.items():
    log.info("  %s: %s", name, reason)


if __name__ == "__main__":
  main()
