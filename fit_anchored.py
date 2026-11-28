#!/usr/bin/env python3
"""Anchored layered 3DGS for one mesh (anchored_3dgs.py): merge the training views' depth-peel slots into anchors, fit them
jointly (combined fit), then cut out and fine-tune every training view's grid. Config: conf/fit_anchored.yaml.

    python fit_anchored.py hdf5_path=/app/bla/new90/trellis_new90_50views_eevee_504.h5 mesh_index=3 \\
        output_dir=/app/bla/new90/anchored
    -> train with diffusion.grids_dir=/app/bla/new90/anchored/grids_sparse (no finalize / convert step)

Shard mode (build_dataset.py's fitter): mesh_indices=[...] (or "all": every mesh with rows in the h5) fits the meshes one after
another in this process, skipping DONE ones and resuming a partly fitted one per view; a fit error raises (nothing is skipped).
With packed=<file>, once every listed mesh is DONE their grids are packed into that one file (grid_io.pack_dir: temp -> verify
-> fsync -> rename) and then the per-view grid files and the fit intermediates (anchors.npz, combined.npz) are deleted --
regression.json, timing.json and DONE stay. A rerun on a packed shard is a no-op (it only finishes an interrupted clean-up).

    python fit_anchored.py hdf5_path=<shard>/render.h5 mesh_indices=all output_dir=<shard>/fit packed=<shard>/grids.h5

Stages, each skipped when its result is already there (rerun the same command to resume):
  merge     -> <output_dir>/_anchored/mesh<M>/anchors.npz (slot -> anchor labels, per-slot levels, anchor stats, the config
               fingerprint it was built with; a mismatch with the current config is an error, not a silent reuse)
  combined  -> <output_dir>/_anchored/mesh<M>/combined.npz + combined.done
  shared    -> <shared_output_dir>/grids_sparse/mesh<M>_view<V>.h5 (shared_output_dir set; default off)
  finetune  -> <output_dir>/grids_sparse/mesh<M>_view<V>.h5 (always from the combined fit's parameters)
The grids are written finalized, in grid_io.py's sparse layout (the file finalize_fit_dataset.py + convert_grids_sparse.py made
from the old dense output, bit for bit); no dense grid is written unless save_dense=true (then also the old dense run dir
<root>/mesh<M>_view<V>/gaussians.grid.h5, for finalize_fit_dataset.py). A view counts as done when its sparse grid is there; a
view with only a dense grid (old dense-only roots) is never refit: an error by default, converted with save_dense=true (see
anchored_3dgs.view_done).
Stage timings accumulate in <output_dir>/_anchored/mesh<M>/timing.json (the fine-tune per view; time lost to kills is not
counted); regression mode writes regression.json there, updated per view before that view's grid is written (so a kill never
leaves a grid without its score; the means are over every view). The last thing written for a mesh is its DONE marker
(<output_dir>/_anchored/mesh<M>/DONE, after regression.json): a mesh counts as finished only with it.
"""

import glob
import json
import logging
import os
import shutil
import time

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import anchored_3dgs as an
import grid_io

log = logging.getLogger(__name__)


DONE = "DONE"
INTERMEDIATES = ("anchors.npz", "combined.npz", "combined.done")   # deleted after packing (~100 MB per mesh)


def state_dir(out, M):
  return os.path.join(out, "_anchored", f"mesh{M}")


def mesh_done(out, M):
  """The mesh's DONE marker exists: every grid and (regression mode) regression.json were written."""
  return os.path.exists(os.path.join(state_dir(out, M), DONE))


def _fingerprint(cfg, hdf5_path, rows, train):
  """What the anchors depend on: a saved state is only reused when this matches exactly."""
  return {"hdf5_path": hdf5_path, "mesh_index": int(cfg.mesh_index), "rows": rows, "train_views": train,
          "merge": OmegaConf.to_container(cfg.merge, resolve=True), "levels": int(cfg.levels)}


def _check(what, saved, now):
  if saved != now:
    raise SystemExit(f"{what} was made with a different setup:\n  saved {saved}\n  now   {now}\n"
                     "(delete it, or use another output_dir)")


def _view_config(cfg, hdf5_path, rows, train, held, v, stage, iters, extra):
  """config_json of a view's gaussians.grid.h5: the fit_3dgs keys finalize_fit_dataset.py reads, + the full anchored config."""
  return {
    "gaussian_layout": "layered", "hdf5_path": hdf5_path,
    "data": {"primary_view": rows[v], "secondary_views": [rows[w] for w in train if w != v],
             "validation_views": [rows[w] for w in held]},
    "iters": int(iters), "sh_degree": 0, "lr": {"means_weight_decay": float(cfg.lr.offset_weight_decay)},
    "lambda_dssim": float(cfg.loss.lambda_dssim), "seed": int(cfg.seed),
    "anchored": {**OmegaConf.to_container(cfg, resolve=True), "stage": stage, "view_in_mesh": int(v), **extra},
  }


def _with_means(reg):
  """regression.json's dict with the *_mean entries recomputed over every view's score."""
  reg = {k: v for k, v in reg.items() if not k.endswith("_mean")}
  for k in ("shared", "finetuned"):
    if reg.get(k):
      reg[f"{k}_mean"] = float(np.mean(list(reg[k].values())))
  return reg


def fit_mesh(cfg: DictConfig, M: int) -> None:
  """Fit (or resume) mesh M of cfg.hdf5_path into cfg.output_dir; returns at once when its DONE marker exists. Raises on any
  error."""
  cfg = OmegaConf.merge(cfg, {"mesh_index": int(M)})
  hdf5_path = os.path.abspath(str(cfg.hdf5_path))
  out = str(cfg.output_dir)
  if mesh_done(out, M):
    log.info("mesh %d: DONE", M)
    return
  shared = None if cfg.shared_output_dir is None else str(cfg.shared_output_dir)
  regression = cfg.regression.held_views is not None
  save_dense = bool(cfg.save_dense)
  rows = an.mesh_rows(hdf5_path, M)
  train, held = an.split_views(len(rows), cfg.val_fraction, cfg.regression.held_views)
  name = {v: f"mesh{M}_view{v}" for v in train}
  state = state_dir(out, M)
  reg_path = os.path.join(state, "regression.json")
  reg = json.load(open(reg_path)) if (regression and os.path.exists(reg_path)) else {}
  # a view is done when its grid is there and (regression mode) its score is in regression.json (a view without one -- a root
  # resumed with the old code, which wrote the file only at the end -- is fitted again)
  scored = lambda v, k="finetuned": not regression or str(v) in reg.get(k, {})
  todo_ft = [v for v in train if not (an.view_done(out, name[v], save_dense) and scored(v) and scored(v, "shared"))]
  todo_sh = [v for v in train if not an.view_done(shared, name[v], save_dense)] if shared else []
  log.info("mesh %d: %d views, %d train, excluded %s; to do: %d fine-tuned, %d shared grids",
           M, len(rows), len(train), held, len(todo_ft), len(todo_sh))

  def _finish():
    if regression:
      unscored = [v for v in train if not (scored(v) and scored(v, "shared"))]
      if unscored:
        raise RuntimeError(f"mesh {M}: views {unscored} have no held-out score in {reg_path}")
      an.write_json_atomic(reg_path, _with_means(reg))
    os.makedirs(state, exist_ok=True)
    an.write_json_atomic(os.path.join(state, DONE), {"mesh_index": M, "train_views": len(train), "time": time.time()})
    log.info("mesh %d DONE -> %s", M, os.path.join(out, an.SPARSE_DIR))

  if not todo_ft and not todo_sh:
    log.info("mesh %d: all views done", M)
    _finish()
    return
  os.makedirs(state, exist_ok=True)
  timing_path = os.path.join(state, "timing.json")
  timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}

  def _timed(stage, t0, quiet=False):
    timing[stage] = timing.get(stage, 0.0) + time.time() - t0
    an.write_json_atomic(timing_path, timing)
    if not quiet:
      log.info("mesh %d: %s %.1f s", M, stage, time.time() - t0)

  wandb_run = None
  if cfg.wandb.mode != "disabled":
    import wandb
    wandb_run = wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode, tags=list(cfg.wandb.tags),
                           name=cfg.wandb.name or f"anchored-mesh{M}", config=OmegaConf.to_container(cfg, resolve=True))

  def _log_step(stage, extra=None):
    def fn(it, photo, loss):
      if wandb_run is not None:
        wandb_run.log({f"{stage}/loss": float(loss), f"{stage}/loss/photom": float(photo), **(extra or {})})
      if it % int(cfg.log_every) == 0:
        log.info("%s step %d: loss %.5f (photometric %.5f)", stage, it, float(loss), float(photo))
    return fn

  # --- load the training views (one mesh's rows only)
  t0 = time.time()
  with h5py.File(hdf5_path, "r") as f:
    views = [an.load_view(f, rows[v]) for v in train]
    held_views = [an.load_view(f, rows[v]) for v in held] if regression else []
    mesh_path = ""
    if "mesh_paths" in f and 0 <= M < f["mesh_paths"].shape[0]:
      mp = f["mesh_paths"][M]
      mesh_path = mp.decode() if isinstance(mp, bytes) else str(mp)
  scene_extent = an.scene_extent_of(np.stack([w["pose"] for w in views]))
  log.info("loaded %d training views (%d slots), scene_extent %.4f", len(views), sum(len(w["u"]) for w in views), scene_extent)
  _timed("load", t0)

  # --- merge
  fp = _fingerprint(cfg, hdf5_path, rows, train)
  anchors_path = os.path.join(state, "anchors.npz")
  slot_counts = np.array([len(w["u"]) for w in views], np.int64)
  if os.path.exists(anchors_path):
    z = np.load(anchors_path)
    _check(anchors_path, json.loads(str(z["fingerprint"])), fp)
    anchors = {k: z[k] for k in ("labels", "slot_counts", "pos", "foot", "spacing", "color", "levels")}
    if not np.array_equal(anchors["slot_counts"], slot_counts):
      raise SystemExit(f"{anchors_path}: slot counts {anchors['slot_counts'].tolist()} != the h5's {slot_counts.tolist()}")
    log.info("anchors: %s (%d anchors)", anchors_path, len(anchors["pos"]))
  else:
    t0 = time.time()
    anchors = an.build_anchors(views, float(cfg.merge.r), int(cfg.levels), int(cfg.merge.k), int(cfg.merge.max_rounds))
    an.save_npz_atomic(anchors_path, fingerprint=json.dumps(fp), **anchors)
    _timed("merge", t0)
  n_anchor = len(anchors["pos"])
  digest = an.labels_digest(anchors["labels"])
  dev = torch.device(str(cfg.device))
  anchors_t = {
    "pos": torch.tensor(anchors["pos"], device=dev, dtype=torch.float32),
    "labels": [torch.tensor(x, device=dev) for x in an.per_view(anchors["labels"], slot_counts)],
    "levels": [torch.tensor(x, device=dev) for x in an.per_view(anchors["levels"], slot_counts)],
  }
  cams = an.Cameras(views, dev)
  held_cams = an.Cameras(held_views, dev) if regression else None

  # --- combined fit
  combined_path = os.path.join(state, "combined.npz")   # written atomically and fingerprinted: its presence = done
  # everything else the combined fit depends on (a saved fit is only reused when this matches exactly)
  combined_setup = {k: OmegaConf.to_container(cfg, resolve=True)[k]
                    for k in ("levels", "init_opacity", "seed", "combined", "lr", "loss")}
  if an.check_background(cfg) == "black":   # the default: states saved before loss.background existed stay resumable
    combined_setup["loss"].pop("background", None)
  if os.path.exists(combined_path):
    z = np.load(combined_path)
    _check(combined_path, {"labels_sha1": str(z["labels_sha1"]), "n_anchors": int(z["n_anchors"]), **json.loads(str(z["setup"]))},
           {"labels_sha1": digest, "n_anchors": n_anchor, **combined_setup})
    params = {k: torch.tensor(z[k], device=dev) for k in an.PARAM_KEYS}
    combined_loss = z["loss"]
    log.info("combined fit: %s", combined_path)
  else:
    t0 = time.time()
    torch.manual_seed(int(cfg.seed))
    params = {k: v.requires_grad_(True) for k, v in an.init_params(anchors, int(cfg.levels), float(cfg.init_opacity), dev).items()}
    photo, combined_loss = an.fit_combined(params, anchors_t, cams, cfg, scene_extent, np.random.default_rng(int(cfg.seed)),
                                           _log_step("combined"), gen=an.noise_generator(dev, int(cfg.seed)))
    params = {k: v.detach() for k, v in params.items()}
    an.save_npz_atomic(combined_path, labels_sha1=digest, n_anchors=n_anchor, setup=json.dumps(combined_setup),
                       photom=photo, loss=combined_loss,
                       **{k: v.cpu().numpy() for k, v in params.items()})
    _timed("combined", t0)

  def _extra(j):
    lv = anchors_t["levels"][j]
    return {"n_anchors": n_anchor, "num_slots": int(len(lv)), "scene_extent": scene_extent,
            "level_counts": torch.bincount(lv, minlength=int(cfg.levels)).tolist()}

  meta = {"mesh_index": M, "mesh_path": mesh_path}

  # --- shared grids (the combined fit's subsets, no fine-tune)
  if todo_sh:
    t0 = time.time()
    for v in todo_sh:
      j = train.index(v)
      flat = an.view_params(params, anchors_t["pos"], anchors_t["labels"][j], anchors_t["levels"][j])
      if regression:
        reg.setdefault("shared", {})[str(v)] = an.heldout_psnr(an.flat_render_params(flat), held_cams)
      config = _view_config(cfg, hdf5_path, rows, train, held, v, "shared", cfg.combined.steps, _extra(j))
      an.write_view(shared, name[v], an.flat_numpy(flat), views[j], meta, config, scene_extent, float(combined_loss[-1]),
                    save_dense)
    _timed("shared", t0)

  # --- per-view fine-tunes, each from the combined fit
  for n_done, v in enumerate(todo_ft):
    t0 = time.time()
    j = train.index(v)
    flat = an.view_params(params, anchors_t["pos"], anchors_t["labels"][j], anchors_t["levels"][j])
    if regression:   # the subset straight from the combined fit, also without shared_output_dir
      reg.setdefault("shared", {})[str(v)] = an.heldout_psnr(an.flat_render_params(flat), held_cams)
    rng = np.random.default_rng([int(cfg.seed), M, v])
    photo, loss = an.finetune(flat, cams, cfg, scene_extent, rng, _log_step("finetune", {"finetune/view": v}),
                              gen=an.noise_generator(dev, [int(cfg.seed), M, v]))
    if regression:
      reg.setdefault("finetuned", {})[str(v)] = an.heldout_psnr(an.flat_render_params(flat), held_cams)
    if regression:   # before the grid: a kill in between refits this view, which overwrites its score
      an.write_json_atomic(reg_path, _with_means(reg))
    config = _view_config(cfg, hdf5_path, rows, train, held, v, "finetune",
                          int(cfg.combined.steps) + int(cfg.finetune.steps), _extra(j))
    an.write_view(out, name[v], an.flat_numpy(flat), views[j], meta, config, scene_extent, float(loss[-1]), save_dense)
    _timed("finetune", t0, quiet=True)
    log.info("mesh %d view %d fine-tuned (%d/%d): loss %.5f%s", M, v, n_done + 1, len(todo_ft), float(loss[-1]),
             f", held-out PSNR {reg['finetuned'][str(v)]:.3f}" if regression else "")

  if regression:
    reg = _with_means(reg)
    log.info("regression: held-out PSNR shared %s, fine-tuned %s (%s)", reg.get("shared_mean"), reg.get("finetuned_mean"),
             reg_path)
  if wandb_run is not None:
    wandb_run.finish()
  _finish()


def shard_meshes(cfg):
  """The meshes of shard mode: cfg.mesh_indices, or "all" = every mesh index with rows in the h5 (sorted)."""
  if cfg.mesh_indices == "all":
    with h5py.File(str(cfg.hdf5_path), "r") as f:
      return [int(m) for m in np.unique(np.asarray(f["mesh_index"]))]
  return [int(m) for m in cfg.mesh_indices]


def _expected_grids(cfg, meshes):
  """{grid file name} the listed meshes' training views make."""
  with h5py.File(str(cfg.hdf5_path), "r") as f:
    mesh_index = np.asarray(f["mesh_index"])
  names = set()
  for M in meshes:
    train, _ = an.split_views(int((mesh_index == M).sum()), cfg.val_fraction, cfg.regression.held_views)
    names |= {f"mesh{M}_view{v}.h5" for v in train}
  return names


def _cleanup(out, meshes):
  """After packing: the per-view grid files and the fit intermediates go; regression.json / timing.json / DONE / logs stay."""
  sparse = os.path.join(out, an.SPARSE_DIR)
  n = 0
  if os.path.isdir(sparse):
    n = len(os.listdir(sparse))
    shutil.rmtree(sparse)
  for M in meshes:
    st = state_dir(out, M)
    for p in [os.path.join(st, k) for k in INTERMEDIATES] + glob.glob(os.path.join(st, "*.tmp*")):
      if os.path.exists(p):
        os.remove(p)
  return n


def fit_shard(cfg: DictConfig) -> None:
  """Shard mode: fit every listed mesh (DONE ones skipped), then -- with cfg.packed -- pack and clean up (see the docstring).
  Raises on any fit error: a mesh that rendered fine must fit."""
  out = str(cfg.output_dir)
  meshes = shard_meshes(cfg)
  packed = None if cfg.packed is None else os.path.abspath(str(cfg.packed))
  if packed is not None and os.path.exists(packed):
    not_done = [M for M in meshes if not mesh_done(out, M)]
    if not_done:
      raise SystemExit(f"{packed} exists but meshes {not_done[:20]} are not DONE (a different mesh list?)")
    tmp = packed + ".tmp"
    if os.path.exists(tmp) and os.path.samefile(packed, tmp):   # killed between pack_dir's link and the temp's removal
      os.remove(tmp)
    n = _cleanup(out, meshes)
    log.info("%s: packed already, %d meshes DONE%s", packed, len(meshes), f" (removed {n} leftover grid files)" if n else "")
    return
  t0 = time.time()
  for i, M in enumerate(meshes):
    if not mesh_done(out, M):
      log.info("=== mesh %d (%d/%d)", M, i + 1, len(meshes))
      fit_mesh(cfg, M)
  log.info("all %d meshes DONE (%.0f s in this run)", len(meshes), time.time() - t0)
  if packed is None:
    return
  sparse = os.path.join(out, an.SPARSE_DIR)
  for p in glob.glob(os.path.join(sparse, "*.tmp")):
    os.remove(p)   # a grid write killed mid-way (its view was refitted since)
  have = {os.path.basename(p) for p in glob.glob(os.path.join(sparse, "*.h5"))}
  want = _expected_grids(cfg, meshes)
  if have != want:
    raise SystemExit(f"{sparse}: {len(want - have)} grids missing (e.g. {sorted(want - have)[:3]}), {len(have - want)} not of "
                     f"the listed meshes (e.g. {sorted(have - want)[:3]}): not packing")
  r = grid_io.pack_dir(sparse, packed)
  log.info("packed %d grids -> %s (%.2f GB, verified, %.0f s; %d grids not one contiguous byte range)", r["n_grids"], packed,
           r["bytes"] / 1e9, r["seconds"], len(r["overlapping"]))
  n = _cleanup(out, meshes)
  log.info("removed %d per-view grid files and the fit intermediates", n)


@hydra.main(version_base=None, config_path="conf", config_name="fit_anchored")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
  if (cfg.mesh_index is None) == (cfg.mesh_indices is None):
    raise SystemExit("set exactly one of mesh_index (one mesh) and mesh_indices (shard mode: a list, or all)")
  if cfg.mesh_index is not None:
    if cfg.packed is not None:
      raise SystemExit("packed needs shard mode (mesh_indices)")
    fit_mesh(cfg, int(cfg.mesh_index))
  else:
    fit_shard(cfg)


if __name__ == "__main__":
  main()
