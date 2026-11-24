#!/usr/bin/env python3
"""Does a one-mesh re-anchoring model (train_reanchor.py) generalise to OTHER meshes? Eval only, no training.

For the trained mesh and every mesh in `meshes`: the first n_pairs of train_reanchor's eval pairs (A, B) of views
0 .. train_views-1, PSNR over black and over white at the held-out views (the checkpoint's data.eval_views) of the model's
predicted B grid, and the same references as training's eval (train_reanchor.references: A's fit as is, B's own fit, the matched
copy k 32). The model's normalization is the TRAINED mesh's (train_reanchor.setup on the checkpoint cfg), reused unchanged for
every other mesh (a Batcher over that mesh's data with the trained moments). The other meshes' grids are linked into
<output_dir>/grids/mesh<M>_view<V>.h5 (MeshData's layout) from the `other_grids` template.

Writes <output_dir>/generalisation.json, references_mesh<M>.json and generalisation.png. Hydra: conf/eval_reanchor_generalisation.yaml.

  docker exec -w /app/.claude/worktrees/reanchor gsviawt-app-gpu-1 python eval_reanchor_generalisation.py
"""

import json
import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import reanchor as ra
import train_reanchor as tr

log = logging.getLogger("eval_reanchor_generalisation")
NAMES = {"model": "model (predicted B)", "A_asis": "A as is", "B_fit": "B's own fit (ceiling)", "matched32": "matched copy (floor)",
         "empty": "nothing rendered"}


def link_grids(template, mesh, n_views, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  for v in range(n_views):
    src = template.format(mesh=mesh, view=v)
    if not os.path.exists(src):
      raise SystemExit(f"missing grid {src}")
    dst = os.path.join(out_dir, f"mesh{mesh}_view{v}.h5")
    if os.path.islink(dst) and os.readlink(dst) != src:
      os.remove(dst)
    if not os.path.lexists(dst):
      os.symlink(src, dst)
  return out_dir


def eval_mesh(model, rcfg, data, moments, pairs, views, bf16, ref_path):
  in_mean, in_std, out = moments
  batcher = ra.Batcher(data, in_mean, in_std, out, z_ref=float(rcfg.model.z_ref), slot_scale_px=float(rcfg.model.slot_scale_px),
                       decode_clamp=str(rcfg.model.decode_clamp))
  e = tr.evaluate(model, batcher, data, pairs, views, bool(rcfg.model.use_a), bf16)
  ref = tr.references(data, pairs, views, ref_path)
  per = {"model": e.tolist(), **{k: ref[k] for k in ("A_asis", "B_fit", "matched32")}}
  z = torch.zeros_like(data.gt[views])                                           # an empty render: what "no object" scores
  bl, wh = ra.psnr_bw(z, z[:, :1], data.gt[views], data.alpha[views])
  per["empty"] = [[float(bl.mean()), float(wh.mean())]] * len(pairs)
  return {"per_pair": per, "mean": {k: np.mean(v, 0).tolist() for k, v in per.items()},
          "se": {k: (np.std(v, 0, ddof=1) / np.sqrt(len(v))).tolist() for k, v in per.items()}}


def plot(res, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  colours = {"model": "#2a78d6", "A_asis": "#eb6834", "B_fit": "#1baf7a", "matched32": "#eda100", "empty": "#8a8984"}
  markers = {"model": "o", "A_asis": "s", "B_fit": "^", "matched32": "D", "empty": "_"}
  order = [res["trained_mesh"]] + list(res["meshes"])
  keys = [str(m) for m in order]
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
  for ax, (bg, ci) in zip(axes, (("white", 1), ("black", 0))):
    x = np.arange(len(keys))
    for j, k in enumerate(NAMES):
      y = [res["per_mesh"][m]["mean"][k][ci] for m in keys]
      line = k == "empty"
      ax.plot(x + (j - 2) * 0.12, y, linestyle="none", marker=markers[k], markersize=14 if line else 8, color=colours[k],
              label=NAMES[k], markeredgecolor=colours[k] if line else "white", markeredgewidth=2.5 if line else 1.5)
    ax.axvspan(-0.5, 0.5, color="#ececea", zorder=0)
    ax.text(0, ax.get_ylim()[0], "trained", ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_xticks(x, [f"{m}" for m in keys])
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_xlabel("new100 mesh")
    ax.set_title(f"held-out views 45-49, over {bg}", fontsize=11)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)
  axes[0].set_ylabel(f"PSNR (dB), mean of {res['n_pairs']} (A, B) pairs")
  fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(NAMES), fontsize=9, frameon=False)
  fig.suptitle(f"Re-anchoring model trained on mesh {res['trained_mesh']} only, applied to other meshes", fontsize=12)
  fig.tight_layout(rect=(0, 0.07, 1, 1))
  fig.savefig(path, dpi=150)
  log.info("wrote %s", path)


@hydra.main(version_base=None, config_path="conf", config_name="eval_reanchor_generalisation")
def main(cfg):
  os.makedirs(cfg.output_dir, exist_ok=True)
  if cfg.plot_only:
    with open(os.path.join(cfg.output_dir, "generalisation.json")) as f:
      plot(json.load(f), os.path.join(cfg.output_dir, "generalisation.png"))
    return
  device = "cuda"
  st = torch.load(cfg.checkpoint, map_location="cpu", mmap=True, weights_only=False)
  rcfg = OmegaConf.create(st["cfg"])
  rcfg.data.grids_dir = cfg.train_grids_dir
  rcfg.data.render_h5 = cfg.render_h5
  log.info("checkpoint step %d (%.0f s); trained mesh %d", st["step"], st["clock"], int(rcfg.data.mesh))
  model = ra.ReanchorNet(dim=int(rcfg.model.dim), num_blocks=int(rcfg.model.num_blocks), num_heads=rcfg.model.num_heads,
                         use_a=bool(rcfg.model.use_a), layer_scale_init=float(rcfg.model.layer_scale_init),
                         checkpoint_activations=False)
  model.load_state_dict(st["model"])
  del st
  model.to(device)
  log.info("ReanchorNet %.1f M parameters on GPU", sum(p.numel() for p in model.parameters()) / 1e6)

  data, batcher = tr.setup(rcfg, device)        # the trained mesh, with training's moments
  moments = (batcher.in_mean, batcher.in_std, (batcher.out_mean, batcher.out_std, batcher.lo, batcher.hi))
  nv, views = int(rcfg.data.train_views), list(rcfg.data.eval_views)
  pairs = tr.eval_pair_list(nv, int(rcfg.train.eval_pairs), int(rcfg.train.eval_seed))[:int(cfg.n_pairs)]
  bf16 = bool(cfg.bf16)
  trained = int(rcfg.data.mesh)
  res = {"checkpoint": cfg.checkpoint, "trained_mesh": trained, "meshes": list(cfg.meshes), "pairs": [list(p) for p in pairs],
         "views": views, "n_pairs": len(pairs), "columns": ["black", "white"], "per_mesh": {}}

  def run(m, d):
    r = eval_mesh(model, rcfg, d, moments, pairs, views, bf16, os.path.join(cfg.output_dir, f"references_mesh{m}.json"))
    res["per_mesh"][str(m)] = r
    log.info("mesh %d: %s", m, {k: np.round(v, 2).tolist() for k, v in r["mean"].items()})

  run(trained, data)
  del data, batcher
  torch.cuda.empty_cache()
  for m in cfg.meshes:
    gdir = link_grids(cfg.other_grids, int(m), nv, os.path.join(cfg.output_dir, "grids"))
    d = ra.MeshData(cfg.render_h5, gdir, int(m), range(nv), device=device)
    run(int(m), d)
    del d
    torch.cuda.empty_cache()
  with open(os.path.join(cfg.output_dir, "generalisation.json"), "w") as f:
    json.dump(res, f, indent=1)
  plot(res, os.path.join(cfg.output_dir, "generalisation.png"))


if __name__ == "__main__":
  main()
