"""Render-space evaluation of diffusion-mode train_gs.py checkpoints (conf/eval_diffusion.yaml).

Every checkpoint x weights (raw / EMA) is scored on the run's validation volumes with FIXED target views and noise (functions of
seed, set and volume, so every run and checkpoint sees the same ones) and with metrics computed here, in fp32 with autocast off --
the same numbers whatever code version trained the run. Per target view, of renders premultiplied over black against the
dataset images premultiplied by their alpha (the h5 stores straight alpha; until gt-premult the straight colour was used, which
capped a perfect render near 30.5 dB psnr and made edge pixels count as errors):
  l1, dssim (module.DSSIMLoss), psnr, loss = l1 + 0.85 dssim   the whole frame (dominated by the black background)
  fg_l1, fg_psnr       over the TARGET's silhouette (alpha > 0.5) only -- a fixed region per view, so spreading coverage
                       (haze) cannot dilute the error (a union with the render's silhouette would reward it)
  bg_l1                over the rest of the frame: what the render puts where the target is empty (floaters, haze)
  lpips                LPIPS (VGG, torchmetrics), whole frame
  alpha_iou            IoU of the render's silhouette with the target's (alpha > 0.5)
  depth_absrel         raw (unaligned) |depth - mesh depth| / mesh depth, the render's expected depth against the target view's
                       first depth-peel layer, over the pixels both cover
  angle                the angle (degrees) between the source and the target camera's viewing directions
Methods:
  fit        the volume's own fitted grid (the target), encoded -> decoded: the reference the model is trained towards
  onestep    the x0 prediction from pure noise (t = 1), one sampler step
  sample     one full sampler run (k = 0)
  meanK      the mean of the K samples' RENDERS
  bestK      the best of the K samples by loss (an oracle: it looks at the target)
steps: one sampler step count, or a list of them (the run's diffusion.steps when null): with a list, every step count gets its own
<out>/steps<N>/ with the same files as a single run (the same numbers: the noise and target views do not depend on the step
count), but the volumes, the fit and the onestep are loaded / rendered / scored once for all of them.
colour_schedule / colour_schedules (diffusion.colour_noise separate runs): the sampler's colour schedule (GSDiffusionModel.sample),
or several, each to <out>/cs_<name>/ (steps<N>/ below it as above).
Writes <out>/views.csv (one row per checkpoint, weights, set, volume, method, target view), <out>/volumes.csv (the same averaged
over the volume's target views) and <out>/summary.json (mean and standard error over volumes per checkpoint, weights, set, method;
volumes, not views, are the unit). mean<K> has no depth (an average of renders has no single depth).
extra_sets: more eval sets from other render files (extra_dataset: e.g. unseen meshes), the x0 normalization still the run's.
skip_methods (and extra_sets.<name>.skip_methods): leave out fit / onestep (e.g. a set without grids has no fit).
kid (kid.enabled): the sets (val_sets; null: val_views_sval, val_views_strain and the extra sets) become one volume per mesh with
kid.views_per_mesh fixed target views (kid_selection: a function of kid.seed, set and mesh only); the noise base is kid.noise_seed
(null: seed) instead of seed; besides the usual outputs every <out>/[cs_<name>/][steps<N>/] gets kid_features.npz: the DINOv2 features
(kid_features.py) of the sample (k = 0) and of the GT at every target view, scored by kid_compare.py."""
import contextlib as ctl
import csv
import json
import math
import os
import random

import gsplat
import h5py
import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf, open_dict

import gs_dataset
import gs_diffusion as gd
import kid_features
import train_gs
from module import DSSIMLoss
from util import collate_with_batch_size


METRICS = ("l1", "dssim", "psnr", "loss", "fg_l1", "fg_psnr", "bg_l1", "lpips", "alpha_iou", "depth_absrel")


def render(model, batch, gauss, views, device):
  """(rgb (V, 3, H, W), alpha (V, 1, H, W), expected depth (V, 1, H, W)) of one volume's Gaussian dict (batch dim 1) at the given
  view indices of batch["views"]."""
  flat = next(train_gs.flatten_gaussians(1, gauss, batch["source"]["xyz_cam"].to(device), batch["source"]["hit"].to(device)))
  h, w = batch["views"]["rgb"].shape[-2:]
  out, alpha, _ = gsplat.rasterization(
    means=flat["means"].float(), quats=flat["quats"].float(), scales=flat["scales"].float(), opacities=flat["opacities"].float(),
    colors=flat["colors"].float(), viewmats=batch["views"]["viewmat"][0, views].to(device),
    Ks=batch["views"]["K_image"][0, views].to(device), width=int(w), height=int(h), sh_degree=model.max_sh_degree,
    render_mode="RGB+ED", packed=True)
  out = out.permute(0, 3, 1, 2)
  return out[:, :3].clamp(0, 1), alpha.permute(0, 3, 1, 2).clamp(0, 1), out[:, 3:4]


def metrics(pred, gt, dssim, lpips, alpha=None, gt_alpha=None, depth=None, gt_depth=None):
  """Per view (see the module docstring), fp32 with autocast off. pred / gt (V, 3, H, W); alpha, gt_alpha, depth, gt_depth
  (V, 1, H, W) or None (then the metrics that need them are NaN). gt_depth: NaN where the mesh is not hit."""
  nan = torch.full((pred.shape[0],), float("nan"), device=pred.device)
  with torch.autocast(device_type=pred.device.type, enabled=False):
    pred, gt = pred.float(), gt.float()
    l1 = (pred - gt).abs().flatten(1).mean(1)
    ds = dssim(pred, gt).flatten(1).mean(1)
    mse = (pred - gt).square().flatten(1).mean(1)
    lp = torch.stack([lpips(pred[v:v + 1], gt[v:v + 1]) for v in range(pred.shape[0])]).float()
    out = {"l1": l1, "dssim": ds, "psnr": -10 * torch.log10(mse.clamp_min(1e-10)), "loss": l1 + 0.85 * ds, "lpips": lp,
           "fg_l1": nan, "fg_psnr": nan, "bg_l1": nan, "alpha_iou": nan, "depth_absrel": nan}
    if gt_alpha is not None:
      g = gt_alpha.float() > 0.5
      p = alpha.float() > 0.5 if alpha is not None else g
      fg, bg = g.float(), (~g).float()
      nf, nb = fg.flatten(1).sum(1).clamp_min(1), bg.flatten(1).sum(1).clamp_min(1)
      err1 = (pred - gt).abs().mean(1, keepdim=True)
      out["fg_l1"] = (err1 * fg).flatten(1).sum(1) / nf
      out["bg_l1"] = (err1 * bg).flatten(1).sum(1) / nb
      fg_mse = ((pred - gt).square().mean(1, keepdim=True) * fg).flatten(1).sum(1) / nf
      out["fg_psnr"] = -10 * torch.log10(fg_mse.clamp_min(1e-10))
      if alpha is not None:
        out["alpha_iou"] = (g & p).float().flatten(1).sum(1) / (g | p).float().flatten(1).sum(1).clamp_min(1)
      if depth is not None and gt_depth is not None:
        both = (torch.isfinite(gt_depth) & (gt_depth > 0) & p)
        rel = ((depth.float() - gt_depth).abs() / gt_depth).nan_to_num(0) * both
        out["depth_absrel"] = rel.flatten(1).sum(1) / both.float().flatten(1).sum(1).clamp_min(1)
  return out


def view_angles(batch, views):
  """(V,) angle in degrees between the source camera's viewing direction and each target camera's (viewmats are relative to the
  source camera: viewmat[0] = I)."""
  vm = batch["views"]["viewmat"][0, views].float()                       # world(=source camera)-to-camera
  fwd = torch.linalg.inv(vm)[:, :3, 2]                                   # the target cameras' +Z axes in the source frame
  cos = (fwd[:, 2] / fwd.norm(dim=1)).clamp(-1, 1)
  return torch.rad2deg(torch.acos(cos))


def volume_subset(n, max_volumes, seed, set_name):
  """The dataset indices of a set to score: all of them, or max_volumes of them drawn at random but reproducibly (a function
  of seed and the set's name only), so a subset is spread over all of the set's meshes and is the same for every run with
  the same split. Sorted."""
  if max_volumes is None or int(max_volumes) >= n:
    return list(range(n))
  rng = np.random.default_rng([int(seed), *map(ord, set_name)])
  return sorted(rng.choice(n, int(max_volumes), replace=False).tolist())


def kid_selection(volumes, pool, set_name, seed, views_per_mesh):
  """The KID's volumes of a set: ONE per mesh (every mesh the set has), its source one of the set's own volumes of that mesh, its
  targets views_per_mesh of the mesh's pool (the set's target views: the held-out views for val_views_*) other than the source,
  all of it drawn from (seed, set, mesh) only -- the same for every run with the same split, whatever the noise. Fewer pool views
  than views_per_mesh (+ the source): all of them. -> [(dataset index, sorted target views)], by mesh."""
  by_mesh = {}
  for i, (m, v) in enumerate(zip(volumes.df["mesh_id"].to_list(), volumes.df["view_idx"].to_list())):
    by_mesh.setdefault(int(m), []).append((int(v), i))
  out = []
  for m in sorted(by_mesh):
    rng = np.random.default_rng([int(seed), *map(ord, set_name), m])
    cands = sorted(by_mesh[m])
    src, i = cands[int(rng.integers(len(cands)))]
    others = sorted(int(t) for t in pool[m] if int(t) != src)
    out.append((i, sorted(rng.choice(others, min(int(views_per_mesh), len(others)), replace=False).tolist())))
  return out


def fixed_targets_dataset(ds, selection):
  """ds (a GridTargetDataset or WTFeatureDataset) cut to the volumes of `selection` [(index, target views)] (one per mesh), each
  item with exactly its listed target views: the mesh's pool is those views (the source not among them), all of which an item
  draws (in an order of Python's global random); otherwise the same items as ds's."""
  inner = getattr(ds, "inner", ds)
  if len({int(inner.volumes.df["mesh_id"][i]) for i, _ in selection}) != len(selection):
    raise ValueError("fixed_targets_dataset: one volume per mesh")
  rows = [int(inner.volumes.df["row"][i]) for i, _ in selection]
  targets = sorted({t for _, ts in selection for t in ts})
  views = gs_dataset.render_catalog(inner.source_h5).filter(pl.col("view_idx").is_in(targets))
  out = gs_dataset.WTFeatureDataset(
    inner.features_h5, views, rows=rows, num_layers=inner.num_layers, num_views=max(len(ts) for _, ts in selection),
    allow_source_as_target=False, depth_size=inner.depth_size, image_size=inner.image_size, read_features=inner.read_features)
  if inner is ds:
    return out
  out = gd.GridTargetDataset(out, ds.grid_index, ds.encoding, seeded=True)
  out.seed_targets = False
  return out


def extra_dataset(name, spec, like, encoding, patch_size, out_dir):
  """An extra eval set (extra_sets.<name>) from another render file: max_meshes of its meshes (a random subset by (spec.seed, name);
  null: all), as sources their views with view number < max_source_view_num (null: all), targets drawn from all of the mesh's
  other views (the run's data.num_views per volume; kid: kid_selection). The grids of spec.grids_dir give the 'fit' and the
  oracles; grids_dir null: none (the set has to skip 'fit'). The x0 normalization stays the run's (its training grids). `like`:
  one of the run's validation datasets, for the item settings (layers, sizes, num_views). The listing goes to
  <out_dir>/listing_<name>.h5."""
  inner_like = getattr(like, "inner", like)
  source_h5 = str(spec.source_h5)
  cat = gs_dataset.render_catalog(source_h5)
  meshes = sorted(cat.df["mesh_id"].unique().to_list())
  if spec.get("max_meshes") is not None and int(spec.max_meshes) < len(meshes):
    rng = np.random.default_rng([int(spec.get("seed", 0)), *map(ord, name)])
    meshes = sorted(rng.choice(meshes, int(spec.max_meshes), replace=False).tolist())
  cat = cat.filter(pl.col("mesh_id").is_in(meshes))
  src = cat if spec.get("max_source_view_num") is None else cat.filter(pl.col("view_num") < int(spec.max_source_view_num))
  rows = sorted(src.df["view_idx"].to_list())
  grid_index = None
  if spec.get("grids_dir") is not None:
    grid_index, _ = gd.build_grid_index(str(spec.grids_dir), source_h5)
    gd.check_grids(grid_index, int(inner_like.image_size), int(patch_size), int(inner_like.num_layers))
    missing = sorted(set(rows) - set(grid_index))
    if missing:
      raise SystemExit(f"extra set {name}: source views (rows) {missing[:5]}... have no grid in {spec.grids_dir!r} (max_source_view_num)")
  listing = gd.make_listing({r: None for r in rows}, source_h5, os.path.join(out_dir, f"listing_{name}.h5"))
  ds = gs_dataset.WTFeatureDataset(
    listing, cat, num_layers=inner_like.num_layers, num_views=inner_like.num_views,
    allow_source_as_target=inner_like.allow_source_as_target, depth_size=inner_like.depth_size, image_size=inner_like.image_size,
    read_features=inner_like.read_features)
  if grid_index is None:
    return ds
  ds = gd.GridTargetDataset(ds, grid_index, encoding, seeded=True)
  ds.seed_targets = False
  return ds


def load_checkpoint(module, path):
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
  if missing or unexpected:
    raise SystemExit(f"{path}: state_dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
  has_ema = "ema" in ckpt and module.ema is not None
  if has_ema:
    module.ema.load_state_dict(ckpt["ema"])
  return has_ema


@hydra.main(version_base=None, config_path="conf", config_name="eval_diffusion")
def main(cfg):
  device = torch.device("cuda")
  run = OmegaConf.load(cfg.run_config)
  os.makedirs(cfg.out, exist_ok=True)
  with open_dict(run):
    run.output_dir = cfg.out                      # listing.h5 goes here, not into the run's own directory
    run.loader.num_workers = 0
    if cfg.grids_dir is not None:
      run.diffusion.grids_dir = cfg.grids_dir
    if cfg.source_h5 is not None:
      run.diffusion.source_h5 = cfg.source_h5
    if cfg.val_sets is not None and "val_train" in cfg.val_sets:
      run.data.val_train = True                   # scoreable for a run trained without data.val_train too
    if cfg.get("guidance") is not None:           # the samplers' classifier-free guidance (the model reads diffusion.cfg)
      run.diffusion.cfg = OmegaConf.merge(run.diffusion.get("cfg") or {}, cfg.guidance)
  if not train_gs.diffusion_enabled(run):
    raise SystemExit("not a diffusion-mode run")
  dm = train_gs.GSDataModule(run)
  dm.setup()
  for ds in dm.val_datasets.values():   # the target views come from this script's own per-volume seed (random.seed below), as in
    ds.seed_targets = False              # every eval so far -- not the validations' seeding, so evals stay paired across code versions
  module = train_gs.GSLightningModule(run, dm.diffusion_stats).to(device).eval()
  model = module.model
  dssim = DSSIMLoss(reduction="none").to(device)
  from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
  lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device).eval()
  K = int(cfg.samples)
  if cfg.steps is None or isinstance(cfg.steps, (int, str)):
    outs = {int(cfg.steps) if cfg.steps is not None else model.steps: cfg.out}
  else:
    outs = {int(s): os.path.join(cfg.out, f"steps{int(s)}") for s in cfg.steps}
  # the colour schedules (colour_noise separate runs): one (name, schedule) per sampler sweep; name None: outs as they are
  if cfg.get("colour_schedule") is not None and cfg.get("colour_schedules") is not None:
    raise SystemExit("colour_schedule and colour_schedules: give one of them")
  if cfg.get("colour_schedules") is not None:
    schedules = [(str(n), OmegaConf.to_container(c)) for n, c in cfg.colour_schedules.items()]
  else:
    cs = cfg.get("colour_schedule")
    schedules = [(None, None if cs is None else OmegaConf.to_container(cs))]
  if any(c is not None for _, c in schedules) and not model.colour_separate:
    raise SystemExit("colour_schedule(s): only for a diffusion.colour_noise.mode=separate run")
  outs = {(n, st): d if n is None else os.path.join(cfg.out, f"cs_{n}", os.path.relpath(d, cfg.out))
          for n, _ in schedules for st, d in outs.items()}
  schedule_of = dict(schedules)
  datasets, set_skip = dict(dm.val_datasets), {}
  for name, spec in (cfg.get("extra_sets") or {}).items():   # extra eval sets from other render files (extra_dataset)
    if name in datasets:
      raise SystemExit(f"extra_sets.{name}: the run has a validation set of that name")
    datasets[name] = extra_dataset(name, spec, next(iter(dm.val_datasets.values())), train_gs.x0_encoding(run), run.model.patch_size,
                                   cfg.out)
    set_skip[name] = set(spec.get("skip_methods") or [])
  kid = cfg.get("kid")
  kid_on = kid is not None and bool(kid.get("enabled", False))
  if cfg.val_sets is not None:
    sets = list(cfg.val_sets)
  elif kid_on:
    sets = [s for s in ("val_views_sval", "val_views_strain") if s in datasets] + list(cfg.get("extra_sets") or {})
  else:
    sets = list(dm.val_datasets) + list(cfg.get("extra_sets") or {})
  for name in [*sets, *set_skip]:
    if not set(cfg.get("skip_methods") or []) | set_skip.get(name, set()) <= {"fit", "onestep"}:
      raise SystemExit("skip_methods: fit and / or onestep only")
  noise_seed = int(cfg.seed)
  if kid_on:   # the KID's volumes: one per mesh with fixed target views (kid_selection), features of the sample and the GT
    for name in sets:
      inner = getattr(datasets[name], "inner", datasets[name])
      sel = kid_selection(inner.volumes, inner.pool, name, kid.seed, kid.views_per_mesh)
      counts = sorted({len(ts) for _, ts in sel})
      print(f"kid {name}: {len(sel)} meshes, {sum(len(ts) for _, ts in sel)} target views (per mesh {counts})", flush=True)
      if counts != [int(kid.views_per_mesh)]:
        print(f"WARNING kid {name}: not {int(kid.views_per_mesh)} target views for every mesh (the set's pool is smaller)", flush=True)
      datasets[name] = fixed_targets_dataset(datasets[name], sel)
    noise_seed = int(cfg.seed if kid.get("noise_seed") is None else kid.noise_seed)
    gt_cache = kid_features.GTCache(kid.get("gt_cache"))
    kid_model = kid_features.load_dinov2(kid.get("dinov2_repo"), device)
  kid_rows = {s: [] for s in outs}

  rows, view_rows = {s: [] for s in outs}, {s: [] for s in outs}
  for ckpt_path in cfg.checkpoints:
    has_ema = False if ckpt_path == "init" else load_checkpoint(module, ckpt_path)
    for weights in cfg.weights:
      if weights == "ema" and not has_ema:
        print(f"{ckpt_path}: no EMA weights, skipping 'ema'", flush=True)
        continue
      ctx = module.ema.applied(model) if weights == "ema" else ctl.nullcontext()
      with ctx, torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for set_name in sets:
          ds = datasets[set_name]
          skip = set(cfg.get("skip_methods") or []) | set_skip.get(set_name, set())
          idx = volume_subset(len(ds), cfg.max_volumes, cfg.seed, set_name)
          n = len(idx)
          for k, i in enumerate(idx):
            random.seed(f"{cfg.seed}/{set_name}/{i}")       # the dataset draws the target views from the global `random`
            item = ds[i]
            batch = collate_with_batch_size([item])
            views = list(range(1, batch["views"]["rgb"].shape[1]))   # the target views (index 0 is the source)
            gt = batch["views"]["rgb"][0, views].to(device)
            gt_alpha = batch["views"]["alpha"][0, views].to(device)
            gt_pm = gt * gt_alpha   # the h5 stores STRAIGHT alpha; the metrics compare against the premultiplied GT, like the render
            with h5py.File(getattr(ds, "inner", ds).source_h5, "r") as f:   # the target views' first depth-peel layer (z-depth, NaN = no hit)
              gt_depth = torch.from_numpy(np.stack([f["depth_peel"][int(r)][..., 0] for r in item["target_views"]]))
            gt_depth = gt_depth.float().to(device)[:, None]
            if gt_depth.shape[-2:] != gt.shape[-2:]:
              gt_depth = None                                # the depth peel is at another resolution than the images
            angle = view_angles(batch, views)
            image, hit, _, floor = model.inputs(batch, device)
            project = None
            if cfg.get("project_format"):   # the sampler's intermediate x0 predictions put into the training grids' format
              enc = train_gs.x0_encoding(run)
              project = gd.format_projection(model.x0_mean, model.x0_std, hit, item["source"]["xyz_cam"][..., 2].to(device),
                                             float(item["source"]["K_depth"][0, 0]), **enc,
                                             quat=str(cfg.get("project_quat", "unit")))
            base = 1_000_003 * (noise_seed + 1) + int(item["source_view"]) * 101
            seeds = [base + k for k in range(K)]
            M = lambda r: metrics(r[0], gt_pm, dssim, lpips, r[1], gt_alpha, r[2], gt_depth)
            if "x0_raw" not in batch and ("fit" not in skip or cfg.get("lock_layers") or cfg.get("inject_fields")):
              raise SystemExit(f"{set_name}: no grids (grids_dir null): 'fit', lock_layers and inject_fields need them (skip_methods: [fit])")
            once = {}
            if "fit" not in skip:
              once["fit"] = M(render(model, batch, model.decode(model.normalize(batch["x0_raw"].to(device).float()), floor), views, device))
            if "onestep" not in skip:
              once["onestep"] = M(render(model, batch, model.sample(image, hit, seeds=seeds[:1], steps=1, project=project, floor=floor),
                                         views, device))
            key = {"checkpoint": ckpt_path, "weights": weights, "set": set_name, "volume": i,
                   "mesh": int(item["mesh_index"]), "source_view": int(item["source_view"])}
            if kid_on:   # the GT's features (the same in every run: cached per render file and view)
              src_h5 = getattr(ds, "inner", ds).source_h5
              gt_feats = gt_cache.get(src_h5, item["target_views"])
              if gt_feats is None:
                gt_feats = kid_features.features(kid_model, kid_features.white_uint8(gt, gt_alpha, premultiplied=False), device)
                gt_cache.put(src_h5, item["target_views"], gt_feats)
            inject = None
            if cfg.get("inject_fields"):   # those fields fed in at every step, noised to the step's t: the ground truth (oracle) or
              src = str(cfg.get("inject_source", "gt"))   # the model's own one-step prediction
              if src not in ("gt", "onestep"):
                raise ValueError(f"inject_source must be gt or onestep, got {src!r}")
              inject = (model.normalize(batch["x0_raw"].to(device).float()) if src == "gt" else "onestep", list(cfg.inject_fields))
            lock = None
            if cfg.get("lock_layers"):   # oracle: those depth layers held at the ground truth, only the others denoised
              lock = (model.normalize(batch["x0_raw"].to(device).float()), [int(v) for v in cfg.lock_layers])
            for out_key in outs:
              cs_name, steps = out_key
              if cfg.get("layer_stages"):   # the depth-layer groups denoised one after the other, `steps` steps each
                g = model.sample_staged(image.expand(K, *image.shape[1:]), hit.expand(K, *hit.shape[1:]),
                                        [list(map(int, grp)) for grp in cfg.layer_stages], seeds=seeds, steps=steps, floor=floor,
                                        t_min=float(cfg.get("sampler_t_min", 0.0)))
              else:
                g = model.sample(image.expand(K, *image.shape[1:]), hit.expand(K, *hit.shape[1:]), seeds=seeds, steps=steps,
                                 project=project, floor=floor, inject=inject, t_min=float(cfg.get("sampler_t_min", 0.0)), lock=lock,
                                 hidden_shift=cfg.get("hidden_shift"),
                                 **({"colour_schedule": schedule_of[cs_name]} if schedule_of[cs_name] is not None else {}))
              samples = [render(model, batch, {k: v[j:j + 1] for k, v in g.items()}, views, device) for j in range(K)]
              per = dict(once)
              ks = [M(r) for r in samples]
              per["sample"] = ks[0]
              if kid_on:
                gen_feats = kid_features.features(kid_model, kid_features.white_uint8(samples[0][0], samples[0][1], premultiplied=True),
                                                  device)
                for v, tv in enumerate(item["target_views"]):
                  kid_rows[out_key].append({**key, "view": int(tv), "gen": gen_feats[v], "gt": gt_feats[v]})
              per[f"mean{K}"] = metrics(torch.stack([r[0] for r in samples]).mean(0), gt_pm, dssim, lpips,
                                        torch.stack([r[1] for r in samples]).mean(0), gt_alpha)
              best = int(np.argmin([float(m["loss"].mean()) for m in ks]))
              per[f"best{K}"] = ks[best]
              if lock is not None:   # plain_swap: the plain sample (no lock, same noise) with the locked layers swapped to the ground truth
                gp = model.sample(image.expand(K, *image.shape[1:]), hit.expand(K, *hit.shape[1:]), seeds=seeds, steps=steps,
                                  project=project, floor=floor, inject=inject, t_min=float(cfg.get("sampler_t_min", 0.0)),
                                  **({"colour_schedule": schedule_of[cs_name]} if schedule_of[cs_name] is not None else {}))
                gt_g = model.decode(lock[0], floor)
                gp = {name: v[:1].clone() for name, v in gp.items()}
                for name in gp:
                  gp[name][:, lock[1]] = gt_g[name][:, lock[1]].to(gp[name].dtype)
                per["plain_swap"] = M(render(model, batch, gp, views, device))
              if inject is not None and not isinstance(inject[0], str):   # sample_gt: the sample with the injected fields' final values set to the ground truth
                gt_g = model.decode(inject[0], floor)
                g0 = {name: v[:1] for name, v in g.items()}
                for f in inject[1]:
                  for name in (f, "raw_scale") if f == "scale" else (f,):
                    g0[name] = gt_g[name]
                per["sample_gt"] = M(render(model, batch, g0, views, device))
              for method, m in per.items():
                rows[out_key].append({**key, "target_views": " ".join(map(str, item["target_views"])), "method": method,
                                    **{k: float(m[k].nanmean()) if torch.isfinite(m[k]).any() else float("nan") for k in METRICS}})
                for v, tv in enumerate(item["target_views"]):
                  view_rows[out_key].append({**key, "target_view": int(tv), "angle": float(angle[v]), "method": method,
                                           **{k: float(m[k][v]) for k in METRICS}})
            print(f"{os.path.basename(str(ckpt_path))} {weights} {set_name} {k + 1}/{n} (volume {i}), {out_key}:",
                  {m: round(float(v["loss"].mean()), 4) for m, v in per.items()}, flush=True)

  for out_key, out_dir in outs.items():
    write_outputs(cfg, out_dir, rows[out_key], view_rows[out_key])
    if kid_on:
      write_kid(out_dir, kid_rows[out_key], {"feature": kid_features.FEATURE_TAG, "steps": out_key[1], "colour_schedule": out_key[0],
                                             "noise_seed": noise_seed, "kid": OmegaConf.to_container(kid)})


def write_kid(out_dir, rows, meta):
  """<out_dir>/kid_features.npz: per target view image (rows), gen / gt (N, 1024) float32 features of the sample and of the GT, and
  checkpoint, weights, set, volume, mesh, source_view, view (the target view's row of the set's render file); meta: a JSON string."""
  os.makedirs(out_dir, exist_ok=True)
  cols = ("checkpoint", "weights", "set", "volume", "mesh", "source_view", "view")
  path = os.path.join(out_dir, "kid_features.npz")
  np.savez(path, gen=np.stack([r["gen"] for r in rows]).astype(np.float32), gt=np.stack([r["gt"] for r in rows]).astype(np.float32),
           meta=json.dumps(meta), **{c: np.array([r[c] for r in rows]) for c in cols})
  print(f"wrote {path} ({len(rows)} images)", flush=True)


def write_outputs(cfg, out_dir, rows, view_rows):
  """<out_dir>/volumes.csv, views.csv and summary.json (mean and SE over volumes per checkpoint, weights, set, method)."""
  os.makedirs(out_dir, exist_ok=True)
  for name, rs in (("volumes.csv", rows), ("views.csv", view_rows)):
    with open(os.path.join(out_dir, name), "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(rs[0]))
      w.writeheader()
      w.writerows(rs)
  summary = {}
  for r in rows:
    summary.setdefault((r["checkpoint"], r["weights"], r["set"], r["method"]), []).append(r)
  out = []
  for (c, wt, s, m), rs in summary.items():
    entry = {"checkpoint": c, "weights": wt, "set": s, "method": m, "n_volumes": len(rs)}
    for k in METRICS:
      v = np.array([r[k] for r in rs], dtype=float)
      v = v[np.isfinite(v)]
      entry[k] = float(v.mean()) if len(v) else None
      entry[f"{k}_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
    out.append(entry)
  with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump({"config": OmegaConf.to_container(cfg), "results": out}, f, indent=1)
  print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
  main()
