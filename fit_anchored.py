#!/usr/bin/env python3
"""Anchored layered 3DGS for one mesh (anchored_3dgs.py): merge the training views' depth-peel slots into anchors, fit them
jointly (combined fit), then cut out and fine-tune every training view's grid. Config: conf/fit_anchored.yaml.

    python fit_anchored.py hdf5_path=/app/bla/new90/trellis_new90_50views_eevee_504.h5 mesh_index=3 \\
        output_dir=/app/bla/new90/anchored shared_output_dir=/app/bla/new90/anchored_shared

Stages, each skipped when its result is already there (rerun the same command to resume):
  merge     -> <output_dir>/_anchored/mesh<M>/anchors.npz (slot -> anchor labels, per-slot levels, anchor stats, the config
               fingerprint it was built with; a mismatch with the current config is an error, not a silent reuse)
  combined  -> <output_dir>/_anchored/mesh<M>/combined.npz + combined.done
  shared    -> <shared_output_dir>/mesh<M>_view<V>/ (shared_output_dir set)
  finetune  -> <output_dir>/mesh<M>_view<V>/ (always from the combined fit's parameters)
A view counts as done when its gaussians.grid.h5 is in the run dir, or finalize_fit_dataset.py already moved it to grids/.
Stage timings accumulate in <output_dir>/_anchored/mesh<M>/timing.json; regression mode writes regression.json there.
"""

import json
import logging
import os
import time

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import anchored_3dgs as an

log = logging.getLogger(__name__)


def _fingerprint(cfg, hdf5_path, rows, train):
  """What the anchors depend on: a saved state is only reused when this matches exactly."""
  return {"hdf5_path": hdf5_path, "mesh_index": int(cfg.mesh_index), "rows": rows, "train_views": train,
          "merge": OmegaConf.to_container(cfg.merge, resolve=True), "levels": int(cfg.levels)}


def _check(what, saved, now):
  if saved != now:
    raise SystemExit(f"{what} was made with a different setup:\n  saved {saved}\n  now   {now}\n"
                     "(delete it, or use another output_dir)")


def _view_config(cfg, hdf5_path, rows, train, held, v, stage, iters, extra):
  """config_json of a view's gaussians.h5: the fit_3dgs keys finalize_fit_dataset.py reads, + the full anchored config."""
  return {
    "gaussian_layout": "layered", "hdf5_path": hdf5_path,
    "data": {"primary_view": rows[v], "secondary_views": [rows[w] for w in train if w != v],
             "validation_views": [rows[w] for w in held]},
    "iters": int(iters), "sh_degree": 0, "lr": {"means_weight_decay": float(cfg.lr.offset_weight_decay)},
    "lambda_dssim": float(cfg.loss.lambda_dssim), "seed": int(cfg.seed),
    "anchored": {**OmegaConf.to_container(cfg, resolve=True), "stage": stage, "view_in_mesh": int(v), **extra},
  }


@hydra.main(version_base=None, config_path="conf", config_name="fit_anchored")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
  hdf5_path = os.path.abspath(str(cfg.hdf5_path))
  M = int(cfg.mesh_index)
  out = str(cfg.output_dir)
  shared = None if cfg.shared_output_dir is None else str(cfg.shared_output_dir)
  regression = cfg.regression.held_views is not None
  rows = an.mesh_rows(hdf5_path, M)
  train, held = an.split_views(len(rows), cfg.val_fraction, cfg.regression.held_views)
  name = {v: f"mesh{M}_view{v}" for v in train}
  todo_ft = [v for v in train if not an.view_done(out, name[v])]
  todo_sh = [v for v in train if not an.view_done(shared, name[v])] if shared else []
  log.info("mesh %d: %d views, %d train, excluded %s; to do: %d fine-tuned, %d shared grids",
           M, len(rows), len(train), held, len(todo_ft), len(todo_sh))
  if not todo_ft and not todo_sh:
    log.info("mesh %d: all done", M)
    return
  state = os.path.join(out, "_anchored", f"mesh{M}")
  os.makedirs(state, exist_ok=True)
  timing_path = os.path.join(state, "timing.json")
  timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}

  def _timed(stage, t0):
    timing[stage] = timing.get(stage, 0.0) + time.time() - t0
    an.write_json_atomic(timing_path, timing)
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
  combined_path, done_path = os.path.join(state, "combined.npz"), os.path.join(state, "combined.done")
  # everything else the combined fit depends on (a saved fit is only reused when this matches exactly)
  combined_setup = {k: OmegaConf.to_container(cfg, resolve=True)[k]
                    for k in ("levels", "init_opacity", "seed", "combined", "lr", "loss")}
  if os.path.exists(done_path):
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
                                           _log_step("combined"))
    params = {k: v.detach() for k, v in params.items()}
    an.save_npz_atomic(combined_path, labels_sha1=digest, n_anchors=n_anchor, setup=json.dumps(combined_setup),
                       photom=photo, loss=combined_loss,
                       **{k: v.cpu().numpy() for k, v in params.items()})
    with open(done_path, "w") as f:
      f.write(digest + "\n")
    _timed("combined", t0)

  reg_path = os.path.join(state, "regression.json")
  reg = json.load(open(reg_path)) if (regression and os.path.exists(reg_path)) else {}

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
      an.write_view(shared, name[v], an.flat_numpy(flat), views[j], meta, config, scene_extent, float(combined_loss[-1]))
    _timed("shared", t0)

  # --- per-view fine-tunes, each from the combined fit
  t0 = time.time()
  for n_done, v in enumerate(todo_ft):
    j = train.index(v)
    flat = an.view_params(params, anchors_t["pos"], anchors_t["labels"][j], anchors_t["levels"][j])
    if regression:   # the subset straight from the combined fit, also without shared_output_dir
      reg.setdefault("shared", {})[str(v)] = an.heldout_psnr(an.flat_render_params(flat), held_cams)
    rng = np.random.default_rng([int(cfg.seed), M, v])
    photo, loss = an.finetune(flat, cams, cfg, scene_extent, rng, _log_step("finetune", {"finetune/view": v}))
    if regression:
      reg.setdefault("finetuned", {})[str(v)] = an.heldout_psnr(an.flat_render_params(flat), held_cams)
    config = _view_config(cfg, hdf5_path, rows, train, held, v, "finetune",
                          int(cfg.combined.steps) + int(cfg.finetune.steps), _extra(j))
    an.write_view(out, name[v], an.flat_numpy(flat), views[j], meta, config, scene_extent, float(loss[-1]),
                  {"photom": photo, "loss": loss})
    log.info("mesh %d view %d fine-tuned (%d/%d): loss %.5f%s", M, v, n_done + 1, len(todo_ft), float(loss[-1]),
             f", held-out PSNR {reg['finetuned'][str(v)]:.3f}" if regression else "")
  if todo_ft:
    _timed("finetune", t0)

  if regression:
    for k in ("shared", "finetuned"):
      if reg.get(k):
        reg[f"{k}_mean"] = float(np.mean(list(reg[k].values())))
    an.write_json_atomic(reg_path, reg)
    log.info("regression: held-out PSNR shared %s, fine-tuned %s (%s)", reg.get("shared_mean"), reg.get("finetuned_mean"),
             reg_path)
  if wandb_run is not None:
    wandb_run.finish()
  log.info("mesh %d done -> %s", M, out)


if __name__ == "__main__":
  main()
