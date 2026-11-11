#!/usr/bin/env python3
"""Train a point-cloud-anchored 3D Gaussian Splatting prediction model on
cached World Tracing decoder tokens.

Input is a wt_features.py output h5 (the (layers, patches, dim) token volume
World Tracing's decoder produced for one view, from a noised GT layered
point cloud). A per-patch linear layer (gs_decoder.PatchGaussianHead)
maps each token to per-pixel per-layer Gaussian appearance/shape parameters
(opacity, scale, rotation, SH color), with the same activations/init as the
Flash3D-style decoder this script used to train. Gaussian positions are
anchored to the source view's own depth-peel point cloud (see gs_dataset.py's
dense_unproject_camera); with model.predict_mean_offset the head also
predicts a residual offset on top of that anchor (see flatten_gaussians).

data.item picks the rows of the features h5 the datasets hold (one row, a
list of rows, or null for all of them; gs_dataset.WTFeatureDataset), split
into train/val by data.mesh_split / data.view_split. With model/backbone set
(data.input=image) the head's input is the source image instead of cached
features, and with diffusion.enabled the model is gs_diffusion.py's
diffusion model over fitted layered 3DGS grids.

Training objective (Flash3D's cross-view photometric setup): decode the
source view's Gaussians, render them into the source view itself
(self-reconstruction) plus other views of the same mesh, and supervise with
L1 + D-SSIM against the real renders. Rendering uses gsplat, with all
geometry kept in the source camera's own frame (see
gs_dataset.relative_viewmats) -- no world coordinates involved.

Usage:
    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python train_gs.py \\
        data.features_h5=/app/bla/porsche_250views.wt_features.5views10seeds.h5 \\
        data.item=0 trainer.max_epochs=1000

    # Resume from an earlier run's checkpoint (model/optimizer/scheduler +
    # epoch/step state; continues into a fresh output_dir):
    docker exec -w /app gsviawt-app-gpu-1 /home/user/venv/bin/python train_gs.py \\
        ckpt_path=/app/outputs/3dgs-model-2026-09-01_12-00-00/final.ckpt

gsplat JIT-compiles CUDA kernels on first import; the CUDA toolchain
it needs is set up at the image level (see Dockerfile).
"""

import collections
import contextlib as ctl
import functools as ft
import logging
import math
import os
import sys

import gsplat
import h5py
import hydra
import lightning.pytorch as pl
import numpy as np
import polars
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import pack, rearrange
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import ConcatDataset, DataLoader

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from gs_dataset import (OPENGL_TO_OPENCV, WTFeatureDataset, _build_item, _EmptyDataset, _get_h5,  # noqa: E402
                        render_catalog, rotate_quats_wxyz, sample_matching_sources)
from gs_decoder import PatchGaussianHead  # noqa: E402
import gs_diffusion  # noqa: E402
import learnable_grids  # noqa: E402
from module import (DSSIMLoss, Events, ProgressWandbLogger, SameDirCheckpointIO, TrainClock, _completed_steps,  # noqa: E402
                    make_scheduler, needs_clock)
from util import collate_with_batch_size, pipe, set_mode, timed  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def image_quality(pred_rgb, gt_rgb, gt_alpha):
  """Per view (B, V), in fp32 and without a graph: psnr, fg_psnr (over the target silhouette, gt_alpha > 0.5; 100 dB for a view
  without one) and bg_l1 (the L1 outside it) -- eval_diffusion.metrics' definitions. pred_rgb / gt_rgb (B, V, 3, H, W), gt_alpha
  (B, V, [1,] H, W)."""
  with torch.no_grad(), torch.autocast(device_type=pred_rgb.device.type, enabled=False):
    diff = pred_rgb.detach().float() - gt_rgb.float()
    sq, ab = diff.square().mean(2), diff.abs().mean(2)                                 # (B, V, H, W)
    fg = (gt_alpha.float().reshape(sq.shape) > 0.5).float()
    nf, nb = fg.flatten(2).sum(2).clamp_min(1), (1 - fg).flatten(2).sum(2).clamp_min(1)
    return {
      "psnr": -10 * torch.log10(sq.flatten(2).mean(2).clamp_min(1e-10)),
      "fg_psnr": -10 * torch.log10(((sq * fg).flatten(2).sum(2) / nf).clamp_min(1e-10)),
      "bg_l1": (ab * (1 - fg)).flatten(2).sum(2) / nb,
    }


def x0_encoding(cfg):
  """cfg.diffusion.scale_encoding, completed with the defaults of a config without it (sort off, no floor: runs before it existed)."""
  enc = OmegaConf.select(cfg, "diffusion.scale_encoding", default=None)
  return gs_diffusion.normalize_encoding(OmegaConf.to_container(enc) if enc is not None else None)


def diffusion_enabled(cfg):
  """cfg.diffusion.enabled; configs without the block (older ones) are not diffusion runs."""
  return bool(OmegaConf.select(cfg, "diffusion.enabled", default=False))


def probe_config(cfg):
  """cfg.diffusion.probe (the x_t reliance probe of the validations, GSDiffusionModel.xt_probe) when it is enabled, else None. A
  config without the key (a run from before the probe) has it off: its validations, logged keys and cost stay as they were."""
  p = OmegaConf.select(cfg, "diffusion.probe", default=None)
  return p if p is not None and bool(p.get("enabled", False)) else None


CATALOG_ROW = "__catalog_row"   # the row-index column catalog_transform's expression must keep


def catalog_transform_keys(df, expr):
  """The (mesh_id, view_idx) keys of the catalogue rows data.catalog_transform keeps: `expr` is a Python expression evaluated with
  `df` (the catalogue dataframe, plus a row-index column CATALOG_ROW) and `pl` (polars) in scope, which must return a DataFrame of
  some of df's rows (still carrying CATALOG_ROW), e.g. 'df.filter(pl.col("mesh_id") == 13).filter(pl.col("view_idx") == 12)'."""
  indexed = df.with_row_index(CATALOG_ROW)
  try:
    out = eval(expr, {"df": indexed, "pl": polars})
  except Exception as e:
    raise SystemExit(f"data.catalog_transform {expr!r} failed: {type(e).__name__}: {e}") from e
  if isinstance(out, polars.LazyFrame):
    out = out.collect()
  if not isinstance(out, polars.DataFrame):
    raise SystemExit(f"data.catalog_transform {expr!r} must return a polars DataFrame, got {type(out).__name__}")
  if CATALOG_ROW not in out.columns:
    raise SystemExit(f"data.catalog_transform {expr!r}: the result lost the {CATALOG_ROW!r} column (filter df's rows, don't "
                     "select columns away)")
  kept = sorted(set(out[CATALOG_ROW].to_list()))
  if kept and not 0 <= kept[0] <= kept[-1] < len(df):
    raise SystemExit(f"data.catalog_transform {expr!r}: {CATALOG_ROW} values out of range")
  sub = df[kept]
  return set(zip(sub["mesh_id"].to_list(), sub["view_idx"].to_list()))


def check_val_targets(ds, name, cfg):
  """A validation set whose volumes have no target view to render (loss.photom.for has no "source") would make gsplat render zero
  cameras and crash, e.g. when a mesh has too few grid views left after the view split."""
  if "source" in cfg.loss.photom["for"]:
    return
  inner = ds.inner
  bad = []
  for row in inner.volumes.df.iter_rows(named=True):
    pool = inner.pool_sets[row["mesh_id"]]
    n = len(pool) if inner.allow_source_as_target else len(pool - {row["view_idx"]})
    if n == 0:
      bad.append((row["mesh_id"], row["view_idx"]))
  if bad:
    raise SystemExit(
      f"validation set {name}: {len(bad)} volumes have no target view to render (mesh, source view) {bad[:5]}...: a mesh has too few "
      "grid views for data.view_split / data.num_views, or put 'source' in loss.photom.for")


class GSModel(nn.Module):
  """Per-patch head (a linear layer, optionally behind a transformer trunk --
  cfg.model.transformer.num_blocks) over cached World Tracing decoder tokens: maps the
  (B, L, P, D) token volume of the source view to per-pixel per-layer
  Gaussian appearance/shape parameters. Positions are anchored to the input
  point cloud (gs_dataset.dense_unproject_camera); with
  cfg.model.predict_mean_offset the output also carries a residual
  camera-space "offset" on top of that anchor -- see flatten_gaussians below."""

  def __init__(self, cfg):
    super().__init__()
    self.num_layers = int(cfg.data.num_layers)
    self.max_sh_degree = int(cfg.model.max_sh_degree)
    self.min_scale_mult = float(cfg.model.min_scale_mult)
    self.feature_dim = int(cfg.model.feature_dim)

    self.head = PatchGaussianHead(
      feature_dim=self.feature_dim,
      num_layers=self.num_layers,
      max_sh_degree=self.max_sh_degree,
      patch_size=int(cfg.model.patch_size),
      opacity_scale=cfg.model.opacity_scale, opacity_bias=cfg.model.opacity_bias,
      scale_scale=cfg.model.scale_scale, scale_bias=cfg.model.scale_bias,
      sh_scale=cfg.model.sh_scale, scale_lambda=cfg.model.scale_lambda,
      zero_init_last_conv=cfg.model.zero_init_last_conv,
      predict_mean_offset=cfg.model.predict_mean_offset,
      num_blocks=int(cfg.model.transformer.num_blocks),
      num_heads=int(cfg.model.transformer.num_heads),
      mlp_ratio=float(cfg.model.transformer.mlp_ratio),
      layer_scale_init=float(cfg.model.transformer.layer_scale_init),
      checkpoint_activations=bool(cfg.model.transformer.get("checkpoint_activations", False)),
      layer_embed_dim=cfg.model.layer_embed_dim,
      input_proj_dim=cfg.model.get("input_proj_dim"),
      input_proj_identity_init=bool(cfg.model.get("input_proj_identity_init")),
    )

    # model/backbone selected: the encoder is part of the model and trained with the head (the batch carries the
    # source "image" instead of cached "features"); Hydra builds it from the config group's _target_
    backbone = cfg.model.get("backbone")
    self.backbone = None if backbone is None else hydra.utils.instantiate(backbone)
    self.input_key = "features" if self.backbone is None else "image"

    # model.rgb_patch: every patch token gets its own 14x14x3 RGB pixels concatenated (in [-1, 1],
    # the same patch for all L layers); feature_dim is then the cached feature dim + 3 * patch_size^2
    self.rgb_patch = bool(cfg.model.get("rgb_patch"))
    # model.rgb_patch_noise: only Gaussian layer 0 gets the RGB patch; layers 1.. get N(0, 1) noise
    # of the same shape, freshly sampled on every forward (train and eval)
    self.rgb_patch_noise = bool(cfg.model.get("rgb_patch_noise"))
    if self.rgb_patch_noise and not self.rgb_patch:
      raise ValueError("model.rgb_patch_noise needs model.rgb_patch")
    self.patch_size = int(cfg.model.patch_size)

  def inputs(self, batch, device):
    """The model input of a collated batch: cached features, the source image, or
    (model.rgb_patch) both."""
    if self.rgb_patch:
      return batch["features"].to(device), batch["image"].to(device)
    return batch[self.input_key].to(device)

  def forward(self, x):
    """x: cached features (B, L, P, D), or with a backbone the source image (B, 3, S, S);
    -> dict of (B, L, C, H, W) Gaussian parameters, H = W = sqrt(P) * patch_size."""
    if self.rgb_patch:
      features, image = x
      patches = rearrange(
        image * 2 - 1, "b c (h py) (w px) -> b 1 (h w) (py px c)", py=self.patch_size, px=self.patch_size)
      patches = patches.expand(-1, features.shape[1], -1, -1)
      if self.rgb_patch_noise:
        patches = torch.cat([patches[:, :1], torch.randn_like(patches[:, 1:])], dim=1)
      features = torch.cat([features, patches.to(features.dtype)], dim=-1)
    else:
      features = x if self.backbone is None else self.backbone(x)
    if features.shape[-1] != self.feature_dim:
      raise ValueError(
        f"features have dim {features.shape[-1]}, model.feature_dim={self.feature_dim}")
    return self.head(features)


def _check_grid(gauss, xyz_cam):
  """The head's pixel grid must be exactly the depth-peel grid -- everything
  downstream indexes gauss and xyz_cam/hit pixel-for-pixel."""
  gh, gw = gauss["opacity"].shape[-2:]
  _, _, dh, dw, _ = xyz_cam.shape
  if (gh, gw) != (dh, dw):
    raise ValueError(
      f"head output grid {gh}x{gw} != depth-peel grid {dh}x{dw}; the features "
      "must come from a render at the model's input size (504x504 for r75b)")


def model_forward(
  model,
  batch,
  *,
  need_render: bool=False,
  render_view_idx = None,
  bg_colors = None,
  gauss = None,
  device,
):
  """gauss: the Gaussian dict to render instead of the model's own prediction (diffusion.photom renders the diffusion model's x0
  prediction; the model is not called then)."""
  if render_view_idx is None:
    render_view_idx = slice(None)

  B = batch["batch_size"]

  out = {}
  out["gauss"] = model(model.inputs(batch, device)) if gauss is None else gauss
  _check_grid(out["gauss"], batch["source"]["xyz_cam"])

  # render 3DGS if necessary
  if need_render:
    rvi = render_view_idx

    pred_rgb = []
    pred_alpha = []

    # bg_colors: (B,3) or None -- one background color per BATCH item (not
    # per view; every view of a given item composites onto the same color),
    # used only by _step's train-stage bg-compositing option (see
    # loss.background). None (default): unchanged behavior, no background
    # passed to gsplat (whatever it renders against internally).
    bg_iter = bg_colors if bg_colors is not None else [None] * B

    for (
      bg,
      x_viewmats,
      x_Ks,
      x_gauss_flat,
    ) in zip(
      bg_iter,
      batch["views"]["viewmat"][:,rvi].to(device), # (B,V,N,M)
      batch["views"]["K_image"][:,rvi].to(device), # (B,V,N,M)
      flatten_gaussians(
        B,
        out["gauss"],
        batch["source"]["xyz_cam"].to(device),
        batch["source"]["hit"].to(device),
      ),
    ):
      _b, _v, _c, ih, iw = batch["views"]["rgb"].shape

      def rasterize(means, quats, scales, opacities, colors, viewmats, Ks, bg=bg, iw=iw, ih=ih):
        return gsplat.rasterization(
          means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
          viewmats=viewmats, Ks=Ks,
          width=int(iw),
          height=int(ih),
          sh_degree=model.max_sh_degree,
          render_mode="RGB",
          packed=True,
          # packed=True (always, here) -> gsplat's low-level rasterizer wants
          # backgrounds shaped just (channels,), shared across every view in
          # this call, not one per view -- which is exactly the granularity
          # we want (one call per batch item already).
          backgrounds=bg if bg is not None else None,
        )

      render_args = (
        x_gauss_flat["means"], x_gauss_flat["quats"], x_gauss_flat["scales"],
        x_gauss_flat["opacities"], x_gauss_flat["colors"], x_viewmats, x_Ks)
      if model.backbone is not None and torch.is_grad_enabled():
        # a trainable backbone leaves little GPU memory for the renderer's activations:
        # recompute each batch item's render in the backward pass instead of keeping all of them
        x_pred_rgb, x_pred_alpha, _ = torch.utils.checkpoint.checkpoint(rasterize, *render_args, use_reentrant=False)
      else:
        x_pred_rgb, x_pred_alpha, _ = rasterize(*render_args)

      x_pred_rgb   = rearrange(x_pred_rgb, "v h w c -> v c h w")
      x_pred_alpha = rearrange(x_pred_alpha, "v h w c -> v c h w")

      pred_rgb.append(x_pred_rgb)
      pred_alpha.append(x_pred_alpha)

    pred_rgb = torch.stack(pred_rgb).clamp(0.0, 1.0)
    pred_alpha = torch.stack(pred_alpha).clamp(0.0, 1.0)

    out["rgb"] = pred_rgb
    out["alpha"] = pred_alpha

  return out


def flatten_gaussians(b, gauss, xyz_cam, hit):
  """
  xyz_cam: (B,L,H,W,3) f32 camera-space positions (NaN where invalid)
  same layer/pixel order as `hit` and every entry of `gauss` (each (L,C,H,W)).
  Filters to valid entries and returns gsplat.rasterization-ready tensors.
  """

  # NOTE(andrei): There are different ways one could go about flattening the
  # Gaussians with our setup. We could put the entire batch of Gaussians into
  # arrays, since gsplat.rasterize() does support this kind of batched
  # rendering.
  #
  # However in this case it would be necessary to set the opacity of the invalid
  # Gaussians to zero to prevent them from showing up.
  #
  # The alternative is to just call gsplat.rasterize() in a for loop which is
  # what I went with. We have a lot of invalid Gaussians so I think this makes
  # more sense.

  for idx in range(b):
    means_flat = rearrange(xyz_cam[idx], "l h w c -> (l h w) c")
    if "offset" in gauss:
      # model.predict_mean_offset: means = depth-peel position (frozen
      # anchor, no grad) + predicted residual offset. The one place the two
      # get resolved -- every render/preview goes through here.
      means_flat = means_flat + rearrange(gauss["offset"][idx], "l c h w -> (l h w) c")
    valid_flat = rearrange(hit[idx], "l h w -> (l h w)")

    opacity_flat  = rearrange(gauss["opacity"] [idx], "l c h w -> (l h w c)") * valid_flat.to(gauss["opacity"].dtype)
    scale_flat    = rearrange(gauss["scale"]   [idx], "l c h w -> (l h w) c")
    rotation_flat = rearrange(gauss["rotation"][idx], "l c h w -> (l h w) c")
    colors_flat   = rearrange(gauss["sh_dc"]   [idx], "l c h w -> (l h w) 1 c")
    if "sh_rest" in gauss:   # max_sh_degree > 0: (N, 1 + K, 3), the DC band first
      colors_flat, _ = pack(
        [
          colors_flat,
          rearrange(gauss["sh_rest"][idx], "l (k c) h w -> (l h w) k c", c=3)
        ],
        "n * c",
      )

    keep = valid_flat
    yield {
      "means": means_flat[keep],
      "quats": rotation_flat[keep],
      "scales": scale_flat[keep],
      "opacities": opacity_flat[keep],
      "colors": colors_flat[keep],
    }


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

class GSDataModule(pl.LightningDataModule):
  """Train/val over cached feature volumes (data.item: one row, a list of rows, or null for all of them),
  split with master's two stages on the render catalog the volumes' source views are rows of:
    data/mesh_split: all meshes -> train_meshes + val_meshes
    data/view_split: the train meshes' views -> train_views + val_views
  (see conf/data/mesh_split/*.yaml and conf/data/view_split/*.yaml). A volume goes where its SOURCE view is,
  and samples its data.num_views supervision views from the views of that half:
    train:            volumes whose source is a train view (targets from the train views)
    val_meshes:       volumes of the held-out meshes (targets from their views)
    val_views_strain: volumes with a train-view source, as many per mesh as it has val views (see
                      gs_dataset.sample_matching_sources); targets from the val views
    val_views_sval:   volumes with a val-view source, targets from the val views
    val_train:        (data.val_train) the training volumes themselves, targets from the train views
  data.catalog_transform (a polars expression, see catalog_transform_keys) then cuts the training set and every validation
  set down to the volumes of the catalogue rows it keeps; the splits, pools and x0 stats stay the full catalogue's.
  Every validation set is complete (the empty ones are dropped); how much of it a validation scores is
  that validation's own choice (module.EventValidation volumes; trainer.limit_val_batches for Lightning's own
  ones, which a run with an at_end validation doesn't have): the val dataloaders shuffle, so each validation scores a fresh random sample. The datasets are in
  self.val_datasets, in that order, one val dataloader each. self.validation names the validation running
  (module.EventValidation sets it; "lightning" is Lightning's own), it is part of the logged metric names."""

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.train_ds = None
    self.val_datasets = None
    self.validation = "lightning"
    self.lightning_runs = 0   # Lightning's own validations so far (kept in checkpoints: state_dict)

  def setup(self, stage=None):
    if self.train_ds is not None:
      return
    cfg = self.cfg
    d = cfg.data
    self.diffusion_stats = None
    self.grid_init = None   # diffusion.learn_grids: the learnable grids' init (learnable_grids.build_init), for the module
    grid_index = None
    if not diffusion_enabled(cfg):
      features_h5 = d.features_h5
    else:
      # diffusion mode: the volumes are the fitted grids (one per source view), listed in a features-less listing file
      if d.input != "image":
        raise SystemExit("diffusion mode needs data.input=image")
      ph = cfg.diffusion.get("photom")
      if float(cfg.diffusion.get("x0_weight", 1.0)) == 0 and (ph is None or not ph.enabled):
        raise SystemExit("diffusion.x0_weight=0 without diffusion.photom: nothing would train the model")
      if ph is not None and ph.enabled:
        if "target" in cfg.loss.photom["for"] and int(d.num_views) < 1:
          raise SystemExit("diffusion.photom renders the target views: it needs data.num_views >= 1 (target views per volume)")
        if cfg.loss.background is not None:
          raise SystemExit("diffusion.photom compares with the images as rendered (over black): loss.background must be null")
      if d.item is not None:
        log.warning("data.item=%s: the diffusion run trains on that one listed grid only (data.item=null: all of them)", d.item)
      if d.depth_size not in (None, d.image_size):
        raise SystemExit("diffusion mode needs data.depth_size=null (or the grids' own size): the grids are at the depth peel's own resolution")
      lg = learnable_grids.learn_grids_config(cfg)
      if lg is not None:
        if {**x0_encoding(cfg), "scale_param": "log"} != gs_diffusion.normalize_encoding(None):
          raise SystemExit(f"diffusion.learn_grids needs the raw encoding (diffusion.scale_encoding.scale_order=keep, scale_floor_px=null, "
                           f"rot_form=keep; scale_param log or cumulative), got {x0_encoding(cfg)}")
        if str(lg.get("x0_grad", "through_input")) not in learnable_grids.X0_GRADS:
          raise SystemExit(f"diffusion.learn_grids.x0_grad must be one of {learnable_grids.X0_GRADS}, got {lg.x0_grad!r}")
        aug = cfg.diffusion.get("augment")
        if aug is not None and aug.get("slot_shuffle"):
          raise SystemExit("diffusion.learn_grids: no augmentation (diffusion.augment.slot_shuffle=false)")
        if bool(lg.learnable) and (ph is None or not ph.enabled):
          raise SystemExit("diffusion.learn_grids.learnable: the grids train only through diffusion.photom, which is off")
      grid_index, grids_source = gs_diffusion.build_grid_index(cfg.diffusion.grids_dir, cfg.diffusion.get("source_h5"))
      gs_diffusion.check_grids(grid_index, int(d.image_size), int(cfg.model.patch_size), int(d.num_layers))
      os.makedirs(cfg.output_dir, exist_ok=True)   # the module makes it too, but later
      features_h5 = gs_diffusion.make_listing(grid_index, grids_source, os.path.join(cfg.output_dir, "listing.h5"))
    with h5py.File(features_h5, "r") as f:
      source_h5 = str(f.attrs.get("source_h5", f.attrs["input"]))
      volume_views, volume_meshes = f["view_indices"][:], f["mesh_index"][:]
    catalog = render_catalog(source_h5)
    full_catalog = catalog
    if grid_index is not None:
      # the splits and the supervision views of the validation sets are over the views that have a grid
      views_with_grid = set(grid_index)
      catalog = catalog.take([pos for pos, v in enumerate(catalog.df["view_idx"].to_list()) if v in views_with_grid])
    rows = np.arange(len(volume_views)) if d.item is None else np.array(
      [d.item] if isinstance(d.item, int) else list(d.item))
    kwargs = dict(
      num_layers=d.num_layers, num_views=d.num_views,
      allow_source_as_target=d.allow_source_as_target, depth_size=d.depth_size,
      image_size=d.image_size if d.input in ("image", "features_image") else None,
      read_features=d.input != "image",
    )

    # the split of master's two stages (conf/data/mesh_split, conf/data/view_split) on the render catalog;
    # a volume goes where its SOURCE view is, and samples its supervision views from the half's views
    train_meshes, val_meshes = hydra.utils.instantiate(d.mesh_split)(catalog, seed=cfg.seed)
    train_views, val_views = hydra.utils.instantiate(d.view_split)(train_meshes, seed=cfg.seed)
    volume_keys = list(zip(volume_meshes[rows].tolist(), volume_views[rows].tolist()))

    def rows_in(half):
      keys = set(zip(half.df["mesh_id"].to_list(), half.df["view_idx"].to_list()))
      return [r for r, k in zip(rows, volume_keys) if k in keys]

    train_rows = rows_in(train_views)
    if not train_rows:
      raise SystemExit("no training volumes: check data.item / data.mesh_split / data.view_split")
    # val_meshes:       every view of the held-out meshes (source and targets)
    # val_views_strain: train meshes, sources = train views (as many per mesh as it has val views),
    #                   targets = val views
    # val_views_sval:   train meshes, sources and targets = val views
    sets = {}
    if len(val_meshes):
      sets["val_meshes"] = (val_meshes, rows_in(val_meshes))
    if len(val_views):
      sets["val_views_strain"] = (val_views, rows_in(sample_matching_sources(train_views, val_views, seed=cfg.seed)))
      sets["val_views_sval"] = (val_views, rows_in(val_views))
    # data.catalog_transform: the splits, the supervision-view pools and the x0 stats (below) are the full catalogue's; the
    # transform only picks which of their volumes (source views) are trained / validated on
    stats_rows = train_rows
    transform = d.get("catalog_transform")
    if transform is not None:
      keys = catalog_transform_keys(catalog.df, str(transform))
      key_of = dict(zip(rows.tolist(), volume_keys))
      kept = lambda rs: [r for r in rs if key_of[r] in keys]
      before = {"train": len(train_rows), **{n: len(r) for n, (_, r) in sets.items()}}
      train_rows = kept(train_rows)
      sets = {n: (v, kept(r)) for n, (v, r) in sets.items()}
      after = {"train": len(train_rows), **{n: len(r) for n, (_, r) in sets.items()}}
      log.info("data.catalog_transform %r keeps %d of %d catalogue rows; volumes kept (of): %s", transform, len(keys),
               len(catalog), {n: f"{after[n]} ({before[n]})" for n in before})
      if not train_rows:
        raise SystemExit(
          f"data.catalog_transform {transform!r} leaves no training volume (kept per set: {after}): its rows are not training "
          "source views -- e.g. a held-out mesh of data.mesh_split or a held-out view of data.view_split, or no volume of the data")
    repeat = d.get("repeat", 1)
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
      raise SystemExit(f"data.repeat must be a whole number >= 1, got {repeat!r}")
    train_pool = train_views
    if grid_index is not None and cfg.diffusion.get("train_targets", "grid") == "all":
      # diffusion.train_targets=all: the training volumes' supervision views are ALL renders of their mesh except the held-out
      # (validation) grid views -- the views without a grid are never a source and never validated
      held_out = set(zip(val_views.df["mesh_id"].to_list(), val_views.df["view_idx"].to_list()))
      train_mesh_ids = set(train_views.df["mesh_id"].to_list())
      train_pool = full_catalog.take([
        pos for pos, (m, v) in enumerate(zip(full_catalog.df["mesh_id"].to_list(), full_catalog.df["view_idx"].to_list()))
        if m in train_mesh_ids and (m, v) not in held_out])
    elif cfg.diffusion.get("train_targets", "grid") not in ("grid", "all"):
      raise SystemExit(f"diffusion.train_targets must be grid or all, got {cfg.diffusion.train_targets!r}")
    if d.get("val_train", False):
      # val_train: the training volumes themselves (after data.catalog_transform), targets from the training views' pool
      sets["val_train"] = (train_pool, train_rows)
    self.train_ds = WTFeatureDataset(features_h5, train_pool, rows=train_rows, **kwargs)
    self.val_datasets = {
      name: WTFeatureDataset(features_h5, v, rows=r, **kwargs) for name, (v, r) in sets.items() if len(r)}
    if grid_index is not None:
      enc = x0_encoding(cfg)
      augment = cfg.diffusion.get("augment")
      self.train_ds = gs_diffusion.GridTargetDataset(
        self.train_ds, grid_index, enc, augment=OmegaConf.to_container(augment) if augment is not None else None)
      self.val_datasets = {n: gs_diffusion.GridTargetDataset(ds, grid_index, enc, seeded=True) for n, ds in self.val_datasets.items()}
      for name, ds in self.val_datasets.items():
        check_val_targets(ds, name, cfg)
      # normalization of the x0 channels: the valid slots of the TRAIN grids (moments cached per grid, next to the grids) -- all of
      # them, also under data.catalog_transform, so a run on a few volumes sees them normalized like a run on all
      stats_ds = self.train_ds if stats_rows == train_rows else gs_diffusion.GridTargetDataset(
        WTFeatureDataset(features_h5, train_pool, rows=stats_rows, **kwargs), grid_index, enc)
      self.diffusion_stats = gs_diffusion.compute_stats(
        stats_ds, grid_index, os.path.join(cfg.diffusion.grids_dir, ".x0_moments"))
      lg = learnable_grids.learn_grids_config(cfg)
      if lg is not None:
        self.grid_init = self._grid_init(lg, grid_index, enc, val_views, full_catalog)
    log.info(
      "volumes: %d train, validation sets %s", len(self.train_ds),
      {n: len(ds) for n, ds in self.val_datasets.items()})
    if not self.val_datasets:
      self.val_datasets = {"val": _EmptyDataset()}

  def _grid_init(self, lg, grid_index, enc, val_views, full_catalog):
    """diffusion.learn_grids: learnable_grids.build_init over the training volumes, plus "eval": what the grids' own render metrics
    render -- "volumes", the source views of learn_grids.metrics.volumes training volumes (evenly spaced over them in source-view
    order; null: all), and per set, for each of them, its target views: train_views (the training pool of its mesh, without the
    source unless data.allow_source_as_target: the views its training renders), val_views (the view split's held-out views of the
    mesh) and nogrid_views (the mesh's renders that have no grid file: never seen by the fit either)."""
    inner = self.train_ds.inner
    init = learnable_grids.build_init(lg, inner, grid_index, enc, gs_diffusion.encode_grid, gs_diffusion.read_grid)
    init["paths"] = [grid_index[v] for v in init["views"]]
    n = lg.metrics.volumes
    pick = (list(range(len(init["views"]))) if n is None or int(n) >= len(init["views"])
            else sorted({int(round(i)) for i in np.linspace(0, len(init["views"]) - 1, int(n))}))
    vols = [(init["views"][i], init["mesh"][i]) for i in pick]
    of_mesh = lambda cat, m: sorted(v for mm, v in zip(cat.df["mesh_id"].to_list(), cat.df["view_idx"].to_list()) if mm == m)
    sets = {"train_views": {}, "val_views": {}, "nogrid_views": {}}
    for v, m in vols:
      pool = inner.pool[m]
      sets["train_views"][v] = sorted(u for u in pool if u != v or self.cfg.data.allow_source_as_target)
      sets["val_views"][v] = of_mesh(val_views, m)
      sets["nogrid_views"][v] = [u for u in of_mesh(full_catalog, m) if u not in grid_index]
    init["eval"] = {"source_h5": inner.source_h5, "volumes": vols, "sets": sets}
    log.info("learn_grids metrics: %d volumes, views per set %s", len(vols),
             {k: sum(len(t) for t in s.values()) for k, s in sets.items()})
    return init

  def train_dataloader(self):
    # data.repeat: an epoch is that many passes over the training volumes (each access draws its own targets, t and noise);
    # self.train_ds itself (the previews' "train" stage) is not repeated
    repeat = int(self.cfg.data.get("repeat", 1))
    return DataLoader(
      self.train_ds if repeat == 1 else ConcatDataset([self.train_ds] * repeat),
      shuffle=True,
      collate_fn=collate_with_batch_size,
      **OmegaConf.to_container(self.cfg.loader, resolve=True),
    )

  def validation_seed(self, index, name):
    """The shuffle seed of validation set `name` at the index-th validation (module.EventValidation counts them): a function of
    cfg.seed, the index and the set only -- so at the same validation every run scores the same volumes (paired comparisons), and
    each validation a different random sample."""
    words = [ord(c) for c in name]   # (name: the set's; "lightning/<set>" for Lightning's own validations)
    return int(np.random.SeedSequence([int(self.cfg.seed), int(index), *words]).generate_state(1)[0])

  def reseed_validation(self, index, kind=None):
    """Seeds every val dataloader's shuffle for the index-th validation (called by module.EventValidation before it runs);
    kind="lightning": for the index-th of Lightning's own validations (their own seeds). The seeds stay in self.validation_seeds
    (per set) for the validation's other draws (GSLightningModule's val_t bags)."""
    gens = getattr(self, "_val_generators", {})
    self.validation_seeds = {name: self.validation_seed(index, name if kind is None else f"{kind}/{name}") for name in gens}
    for name, g in gens.items():
      g.manual_seed(self.validation_seeds[name])

  def reseed_lightning(self):
    """Seeds the val dataloaders for Lightning's NEXT own validation. Lightning builds a validation's iterator (and fetches its
    first batch) before any hook runs, so the generators are kept in this state whenever no EventValidation is running: after
    val_dataloader, after every EventValidation and after every Lightning validation (GSLightningModule.on_validation_end)."""
    self.reseed_validation(self.lightning_runs + 1, kind="lightning")

  def state_dict(self):
    return {"lightning_runs": self.lightning_runs}

  def load_state_dict(self, state):
    self.lightning_runs = int(state.get("lightning_runs", 0))
    self.reseed_lightning()

  def val_dataloader(self):
    self._val_generators = {name: torch.Generator() for name in self.val_datasets}
    self.reseed_lightning()
    return [
      DataLoader(
        ds,
        shuffle=True,   # every validation iterates it afresh: a new random order, so a partial one is a random sample
        generator=self._val_generators[name],   # seeded per validation by reseed_validation (paired across runs)
        collate_fn=collate_with_batch_size,
        num_workers=0,
        batch_size=self.cfg.loader.batch_size,
        drop_last=False,
      )
      for name, ds in self.val_datasets.items()
    ]


class GSLightningModule(pl.LightningModule):
  def __init__(self, cfg, diffusion_stats=None, grid_init=None):
    """grid_init: diffusion.learn_grids' init (GSDataModule.grid_init); None: the grids are the dataset's x0_raw, as always."""
    super().__init__()
    self.cfg = cfg
    self.clock = None   # a module.TrainClock, set by main() for time-driven LR schedulers
    self.views_seen = 0
    self.pixels_seen = 0  # pixels of the supervised target images seen so far (views x H x W)

    self.ema = None
    self._ema_fresh = True    # false once an EMA is restored from a checkpoint
    self.spike_skip = None
    self._accepted_since_step = False   # a loss went into the gradients the next optimizer step applies
    with timed("build_model"):
      if diffusion_enabled(cfg):
        self.model = gs_diffusion.GSDiffusionModel(cfg, diffusion_stats)
        self.ema = gs_diffusion.EMAWeights(self.model, cfg.diffusion.ema.decay)
        spike = cfg.diffusion.spike_skip
        self.spike_skip = gs_diffusion.SpikeSkip(spike.factor, spike.decay, spike.warmup_steps)
      else:
        self.model = GSModel(cfg)
    self.dssim = DSSIMLoss(reduction="none")

    # diffusion.learn_grids: the training volumes' grids as parameters (learnable_grids.LearnableGrids; frozen: buffers), with their
    # own Adam (made in on_train_start, on the device), stepped with the model's optimizer
    self.grids = None
    self.grid_opt = None
    self._grid_opt_state = None   # a resumed run's grid optimizer state, loaded once the optimizer exists
    if grid_init is not None:
      lg = learnable_grids.learn_grids_config(cfg)
      self.grids = learnable_grids.LearnableGrids(grid_init, learnable=bool(lg.learnable), scale_param=x0_encoding(cfg)["scale_param"])
      # x0_grad none: the grids' gradient is the rendering loss's alone (_diffusion_train_step computes it, backward() keeps the
      # total loss's off the grids, optimizer_step installs it)
      self._grid_x0_grad = str(lg.get("x0_grad", "through_input"))
      self._grid_grads = None
      self._grid_eval = grid_init["eval"]
      self._grid_logged = {}   # what the grid metrics logged once already (the reference / init) and the last step they ran

    # One directory for everything this run produces (currently just
    # checkpoints, but named generically for whatever else lands here later
    # -- orbit videos/panels are wandb-only right now).
    # Defaults to the Unix timestamp at startup, so back-to-back runs never
    # share a directory unless cfg.output_dir is set explicitly.
    self.output_dir = cfg.output_dir
    os.makedirs(self.output_dir, exist_ok=True)

  def configure_optimizers(self):
    out = {}

    # optim_groups: per-module optimizer settings (e.g. the pretrained backbone's own learning rate); the head
    # and everything else use `optim`'s
    groups = OmegaConf.to_container(self.cfg.get("optim_groups") or {}, resolve=True)
    modules = {name: getattr(self.model, name, None) for name in groups}
    modules = {name: module for name, module in modules.items() if module is not None}
    if not modules:
      params = self.model.parameters()
    else:
      params = [
        {"params": self.model.head.parameters()},
        *({"params": [p for p in module.parameters() if p.requires_grad], **groups[name]} for name, module in modules.items()),
      ]
    out["optimizer"] = hydra.utils.instantiate(self.cfg.optim)(params)
    if OmegaConf.select(self.cfg, "sched") is not None:
      out["lr_scheduler"] = {
        "scheduler": make_scheduler(hydra.utils.instantiate(self.cfg.sched), out["optimizer"], self.clock),
        "interval": "step",
      }

    return out

  def on_save_checkpoint(self, checkpoint):
    # views_seen is a plain Python int, not part of state_dict() -- persist
    # it explicitly so it keeps climbing (not resetting to 0) across resume.
    checkpoint["views_seen"] = self.views_seen
    checkpoint["pixels_seen"] = self.pixels_seen
    if diffusion_enabled(self.cfg):
      checkpoint["x0_encoding"] = x0_encoding(self.cfg)   # the x0 statistics in state_dict are this encoding's
    if self.ema is not None:
      checkpoint["ema"] = self.ema.state_dict()   # chained resumes must not restart the average from the raw weights
      for name, live in self.ema.stash.items():   # saved while the average is swapped in (an EMA validation): hold the live weights
        checkpoint["state_dict"][f"model.{name}"] = live.detach()
    if self.spike_skip is not None:
      checkpoint["spike_skip"] = self.spike_skip.state_dict()   # nor the loss average and its warmup
    if getattr(self, "grid_opt", None) is not None:   # (the learnable grids themselves are parameters: in state_dict)
      checkpoint["grid_optimizer"] = self.grid_opt.state_dict()

  def on_load_checkpoint(self, checkpoint):
    # .get: checkpoints written before this hook existed have no such key.
    self.views_seen = checkpoint.get("views_seen", 0)
    self.pixels_seen = checkpoint.get("pixels_seen", 0)
    if diffusion_enabled(self.cfg):
      # checkpoints from before x0_encoding was saved were all trained without the scale encoding
      # (keys a checkpoint predates mean their defaults, as in a config without them)
      saved = gs_diffusion.normalize_encoding(checkpoint.get("x0_encoding"))
      now = x0_encoding(self.cfg)
      if saved != now:
        raise SystemExit(
          f"the checkpoint was trained with diffusion.scale_encoding={saved}, this config has {now}: its x0 statistics and targets "
          f"would not match. Resume with the checkpoint's own encoding, e.g. diffusion.scale_encoding.scale_order="
          f"{saved['scale_order']} diffusion.scale_encoding.scale_floor_px={saved['scale_floor_px'] if saved['scale_floor_px'] is not None else 'null'}"
          f" diffusion.scale_encoding.scale_floor_mode={saved['scale_floor_mode']} diffusion.scale_encoding.rot_form={saved['rot_form']}"
          f" diffusion.scale_encoding.scale_param={saved['scale_param']}")
    if self.ema is not None and "ema" in checkpoint:
      self.ema.load_state_dict(checkpoint["ema"])
      self._ema_fresh = False
    if self.spike_skip is not None and "spike_skip" in checkpoint:   # without: a fresh filter, warming up from zero
      self.spike_skip.load_state_dict(checkpoint["spike_skip"])
    if self.cfg.get("reset_scheduler"):
      checkpoint["lr_schedulers"] = []   # Lightning then restores no scheduler state: the config's scheduler stays as built
    if getattr(self, "grids", None) is not None and self.grids.learnable:
      if "grid_optimizer" not in checkpoint:
        raise SystemExit("diffusion.learn_grids.learnable: the checkpoint has no grid optimizer state (not a learnable-grid run's)")
      self._grid_opt_state = checkpoint["grid_optimizer"]   # loaded in on_train_start, once the optimizer is on the device
      if self.grid_opt is not None:
        self.grid_opt.load_state_dict(self._grid_opt_state)
        self._grid_opt_state = None

  def on_fit_start(self):
    if self.spike_skip is not None and self.spike_skip.factor is not None and self.trainer.world_size > 1:
      raise ValueError(
        "diffusion.spike_skip skips a step by returning None from training_step, which Lightning does not support with "
        f"world_size > 1 (here {self.trainer.world_size}): set diffusion.spike_skip.factor=null")

  def on_train_start(self):
    if self.ema is not None and self._ema_fresh:
      self.ema = gs_diffusion.EMAWeights(self.model, self.cfg.diffusion.ema.decay)   # from the weights training starts with
    if getattr(self, "grids", None) is not None and self.grids.learnable and self.grid_opt is None:
      self.grid_opt = self.make_grid_optimizer()
      if self._grid_opt_state is not None:
        self.grid_opt.load_state_dict(self._grid_opt_state)
        self._grid_opt_state = None

  def make_grid_optimizer(self):
    """The learnable grids' Adam: the anchored fitter's (anchored_3dgs.make_adam: eps 1e-15, no weight decay) with its per-group
    learning rates (diffusion.learn_grids.lr), constant (no schedule, no gradient clipping: neither the model's LR schedule nor
    trainer.gradient_clip_val touch it)."""
    lr = learnable_grids.learn_grids_config(self.cfg).lr
    return torch.optim.Adam(self.grids.param_groups(lr), eps=1e-15, weight_decay=0.0)

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
    super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)   # the closure runs training_step + backward
    # Right after the optimizer step (before on_train_batch_end, so validations there see the current average); not
    # after a step whose loss was skipped, which left the weights as they were.
    if self.ema is not None and self._accepted_since_step:
      self.ema.update(self.model)
    if getattr(self, "grid_opt", None) is not None:   # diffusion.learn_grids: the grids step with the model (a skipped loss moves neither)
      if self._accepted_since_step:
        if self._grid_grads is not None:   # x0_grad none: the rendering loss's gradient
          for p, g in zip(self.grids.parameters(), self._grid_grads):
            p.grad = g
        self.grid_opt.step()
      self.grid_opt.zero_grad(set_to_none=True)
      self._grid_grads = None
    self._accepted_since_step = False

  def training_step(self, batch, batch_idx):
    if diffusion_enabled(self.cfg):
      return self._diffusion_train_step(batch)
    return self._step("train", batch, batch_idx)

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    # dataloader_idx: Lightning omits it with a single val loader, hence the default.
    stage = list(self.trainer.datamodule.val_datasets)[dataloader_idx]
    if getattr(self, "grids", None) is not None:   # (getattr: the tests' module shells have no grids)
      batch = self._with_grids(batch)
    # the sampled Gaussians, rendered and compared with the views; diffusion: the full sample, as loss/photom_full (loss/photom is
    # the training's rendering term, see _diffusion_val_loss)
    out = self._step(stage, batch, batch_idx, photom_key="loss/photom_full" if diffusion_enabled(self.cfg) else "loss/photom")
    if diffusion_enabled(self.cfg):
      self._diffusion_val_loss(stage, batch)
    return out

  def _log_metric(self, stage, key, value, batch_size):
    name = f"{stage}/{key}" if stage == "train" else f"{stage}/{self.trainer.datamodule.validation}/{key}"
    self.log(name, value, batch_size=batch_size, add_dataloader_idx=False)

  def _diffusion_losses(self, batch, seeded=False):
    """(loss, parts, counts, x0_hat, t); seeded: the timesteps and noise of each volume are a function of the volume (validation)."""
    image, hit, seeds, _ = self.model.inputs(batch, self.device)
    x0_in = batch.get("x0_input_raw")   # augment.slot_shuffle_target canonical (training items only): the shuffled grid, noised
    x0_raw = batch["x0_raw"].to(self.device)
    if getattr(self, "grids", None) is not None and not seeded:
      # diffusion.learn_grids, training: the volumes' grids instead of the dataset's; learnable: the noised input carries the graph to
      # them, the x0 target is the detached copy (stop-gradient on the target only; learn_grids.x0_grad none: the x0 loss's
      # gradient is kept from the grids altogether, see _diffusion_train_step)
      x0_in = self.grids.dense(batch["source_view"], hit)
      x0_raw = x0_in.detach()
      if not self.grids.learnable:
        x0_in = None
    return self.model.training_loss(image, hit, x0_raw, seeds=seeds if seeded else None, return_pred=True,
                                    x0_input_raw=None if x0_in is None else x0_in.to(self.device))

  def _photom_active(self):
    """diffusion.photom is on, and the train clock is past its start_seconds."""
    p = self.cfg.diffusion.get("photom")
    if p is None or not p.enabled:
      return False
    seconds = self.clock.seconds() if self.clock is not None else 0.0
    return seconds >= float(p.start_seconds)

  def _diffusion_photom(self, stage, batch, x0_hat, t, log=None, cond_drop=None):
    """The photometric loss of the diffusion model's x0 prediction (at the timesteps t), rendered: per volume, only the volumes
    whose timestep, reduced over the layers by diffusion.photom.t_reduce (max: the noisiest layer, mean, layer0), is in
    diffusion.photom.t_range count (a mean over them; zero when there is none). Logs <prefix>/photom[_l1|_dssim] (over the
    volumes that count) and photom_frac (the fraction of volumes that count); the validations log it the same way, so
    train/loss/photom and <set>/<name>/loss/photom are one quantity. log: called instead of _log_metric (the train step holds
    its logs until the step is accepted). Returns the loss (with its graph). cond_drop: the (B,) mask of the samples trained with
    the null image condition (diffusion.cond_dropout): the same renders' loss is also logged over the conditioned / dropped
    volumes that count, as <prefix>/photom_cond[_<term>] / photom_uncond[_<term>] (the loss itself is unchanged)."""
    log = log or self._log_metric
    lo, hi = self.cfg.diffusion.photom.t_range
    reduce = self.cfg.diffusion.photom.t_reduce
    tv = {"max": lambda: t.amax(dim=1), "mean": lambda: t.mean(dim=1), "layer0": lambda: t[:, 0]}[reduce]()   # (B,)
    keep = ((tv >= float(lo)) & (tv <= float(hi))).float()
    log(stage, "loss/photom_frac", keep.mean(), batch["batch_size"])
    gauss = self.model.decode(x0_hat, self.model.floor_context(batch, self.device))
    if cond_drop is None:
      return self._render_photom(stage, batch, gauss, keep, "loss/photom", log)
    if keep.sum() == 0:
      return torch.zeros((), device=self.device)
    per = self._photom_per_volume(batch, gauss)
    dropped = cond_drop.to(keep.device, keep.dtype)
    self._log_photom(stage, per, keep * (1 - dropped), "loss/photom_cond", log)
    self._log_photom(stage, per, keep * dropped, "loss/photom_uncond", log)
    return self._log_photom(stage, per, keep, "loss/photom", log)

  def _render_photom(self, stage, batch, gauss, keep, key, log=None):
    """loss.photom (L1 + D-SSIM on the views of loss.photom.for, against the dataset's rendered images) of the Gaussian dict gauss,
    rendered, over the volumes with keep (B,) = 1. Logs <key>[_l1|_dssim|_psnr|_fg_psnr|_bg_l1]; returns the loss (with its graph)."""
    if keep.sum() == 0:
      return torch.zeros((), device=self.device)
    return self._log_photom(stage, self._photom_per_volume(batch, gauss), keep, key, log)

  def _photom_per_volume(self, batch, gauss):
    """{"l1", "dssim" (the terms with a weight), "total", and the image_quality metrics (no graph)}: per volume (B,), of gauss
    rendered; the image metrics are means over the volume's views of the per-view values, as in eval_diffusion."""
    cfg, device, B = self.cfg, self.device, batch["batch_size"]
    V = batch["views"]["rgb"].shape[1]
    rvi = ([0] if "source" in cfg.loss.photom["for"] else []) + (list(range(1, V)) if "target" in cfg.loss.photom["for"] else [])
    pred = model_forward(self.model, batch, need_render=True, render_view_idx=rvi, gauss=gauss, device=device)
    pred_rgb = pred["rgb"]                                                              # (B, RV, 3, H, W)
    gt_rgb = batch["views"]["rgb"][:, rvi].to(device)
    per = {}
    if cfg.loss.photom.l1_weight > 0:
      per["l1"] = (pred_rgb - gt_rgb).abs().flatten(1).mean(1)                          # (B,)
    if cfg.loss.photom.dssim_weight > 0:
      per["dssim"] = self.dssim(pred_rgb.flatten(0, 1), gt_rgb.flatten(0, 1)).reshape(B, -1).mean(1)
    per["total"] = cfg.loss.photom.l1_weight * per.get("l1", 0) + cfg.loss.photom.dssim_weight * per.get("dssim", 0)
    for name, v in image_quality(pred_rgb, gt_rgb, batch["views"]["alpha"][:, rvi].to(device)).items():
      per[name] = v.mean(1)
    return per

  def _log_photom(self, stage, per, keep, key, log=None):
    """Logs the mean of _photom_per_volume's terms over the volumes with keep = 1 as <key>[_<term>]; returns the total's."""
    log = log or self._log_metric
    n = keep.sum()
    if n == 0:
      return torch.zeros((), device=self.device)
    loss = (keep * per["total"]).sum() / n
    for name, v in per.items():
      if name != "total":
        log(stage, f"{key}_{name}", ((keep * v).sum() / n).detach(), int(n))
    log(stage, key, loss.detach(), int(n))
    return loss

  def _log_parts(self, stage, prefix, parts, counts):
    """Per-field / per-t-bucket losses, each weighted by its number of valid slots (epoch means are then slot means); a bucket
    that no slot fell in is not a mean of anything and is not logged."""
    for name, v in parts.items():
      n = int(counts[name])
      if n:
        self._log_metric(stage, f"{prefix}/{name}", v, n)

  def _diffusion_train_step(self, batch):
    """WT's loss: the squared distance of the predicted x0 to x0 on the valid slots. A loss above `factor` times its running
    average is skipped (no optimizer step), as in WT's recipe, and so is a non-finite one (gs_diffusion.SpikeSkip)."""
    B = batch["batch_size"]
    loss, parts, counts, x0_hat, t = self._diffusion_losses(batch)
    # diffusion.cond_dropout (p > 0 only): the (B,) mask of the samples whose image condition was the null this step
    cond_drop = parts.pop("cond_drop", None)
    x0_loss = loss.detach()
    loss = float(self.cfg.diffusion.get("x0_weight", 1.0)) * loss   # 0: the x0 loss is only logged (diffusion.photom alone trains)
    held = []   # the rendering loss's logs: like train/loss and the x0 loss's, accepted steps only
    photom = None
    if self._photom_active():
      photom = float(self.cfg.diffusion.photom.weight) * self._diffusion_photom(
        "train", batch, x0_hat, t, log=lambda *a: held.append(a), **({"cond_drop": cond_drop} if cond_drop is not None else {}))
      loss = loss + photom
    value = float(loss.detach())
    skipped = self.spike_skip.check(value)
    self._log_metric("train", "skipped", float(skipped), B)
    self._log_metric("train", "nonfinite", float(not math.isfinite(value)), B)
    if self.spike_skip.average is not None:
      self._log_metric("train", "loss_ema", self.spike_skip.average, B)
    if skipped:
      return None
    self._accepted_since_step = True
    self.views_seen += B
    self.pixels_seen += B * int(np.prod(batch["x0_raw"].shape[-2:]))
    self._log_metric("train", "loss", loss.detach(), B)
    self._log_metric("train", "loss/x0", x0_loss, B)   # like train/loss and its parts: accepted steps only
    self._log_parts("train", "loss/x0", parts, counts)   # the validations' keys: <set>/<name>/loss/x0/<part>
    if cond_drop is not None:
      self._log_metric("train", "cond_dropped", cond_drop.float().mean(), B)   # the realized fraction of null-condition samples
    for a in held:
      self._log_metric(*a)
    if getattr(self, "grids", None) is not None and self.grids.learnable and self._grid_x0_grad == "none":
      self._grid_grads = self._rendering_grid_grads(photom)
    return loss

  def _rendering_grid_grads(self, photom):
    """diffusion.learn_grids.x0_grad none: the grids' gradient from the weighted rendering loss `photom` alone (zeros when it is
    off or has no graph), one extra backward pass through the model to x_t (the graph is kept for the total loss's backward)."""
    params = list(self.grids.parameters())
    if photom is None or not photom.requires_grad:
      return [torch.zeros_like(p) for p in params]
    grads = torch.autograd.grad(photom, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]

  def backward(self, loss, *args, **kwargs):
    # diffusion.learn_grids.x0_grad none: the total loss's gradient goes to the model only; the grids get the rendering loss's
    # (_rendering_grid_grads, installed by optimizer_step)
    if getattr(self, "_grid_grads", None) is not None:
      kwargs = {**kwargs, "inputs": [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("grids.")]}
    super().backward(loss, *args, **kwargs)

  @torch.no_grad()
  def _diffusion_val_loss(self, stage, batch):
    """The same loss on a validation batch. The timesteps and noise of a volume are a function of the volume (its source view and
    diffusion.eval_seed), so every validation scores each volume at the same noise levels, whatever the subset or the order."""
    loss, parts, counts, x0_hat, t = self._diffusion_losses(batch, seeded=True)
    self._log_metric(stage, "loss/x0", loss.detach(), batch["batch_size"])
    self._log_parts(stage, "loss/x0", parts, counts)
    if self.cfg.diffusion.photom.enabled and self.cfg.diffusion.photom.validate:
      self._diffusion_photom(stage, batch, x0_hat, t)   # the training's own rendering term, gated the same way
    if self.cfg.diffusion.get("val_t"):
      self._diffusion_val_t(stage, batch)
    if probe_config(self.cfg) is not None:
      self._diffusion_probe(stage, batch)

  @torch.no_grad()
  def _diffusion_probe(self, stage, batch):
    """diffusion.probe: GSDiffusionModel.xt_probe on the validation set's first diffusion.probe.volumes volumes in the validation's
    own (seeded) order -- the first batches it scores, so no other draw and no other volume -- collected per set, logged once per
    validation by on_validation_epoch_end. Not in Lightning's sanity check (its logs are thrown away)."""
    p = probe_config(self.cfg)
    if p is None or self.trainer.sanity_checking:
      return
    probe = self.__dict__.setdefault("_probe", {})
    rows = probe.setdefault(stage, [])
    k = min(int(p.volumes) - sum(int(r["n"]) for r in rows), int(batch["batch_size"]))
    if k <= 0:
      return
    image, hit, seeds, _ = self.model.inputs(batch, self.device)
    vals = self.model.xt_probe(image[:k], hit[:k], batch["x0_raw"][:k], seeds[:k], [float(t) for t in p.ts], float(p.gain_delta))
    rows.append({"n": k, "vals": vals})

  def _log_probe(self):
    """Logs every set's collected probe values (_diffusion_probe) as <set>/<name>/probe/t<t>/{err_model, err_copy, ratio, gain}:
    err_model, err_copy, gain the means over the probed volumes (those with a valid slot), ratio = mean err_model / mean err_copy;
    and <set>/<name>/probe/n_volumes, how many volumes they are a mean over."""
    probe, self._probe = self.__dict__.get("_probe") or {}, {}
    for stage, rows in probe.items():
      count = None
      for t in rows[0]["vals"]:
        v = {k: torch.cat([r["vals"][t][k] for r in rows]) for k in ("err_model", "err_copy", "gain", "valid")}
        ok = v["valid"] > 0
        count = int(ok.sum())
        if not count:
          continue
        m = {k: float(v[k][ok].mean()) for k in ("err_model", "err_copy", "gain")}
        m["ratio"] = m["err_model"] / m["err_copy"] if m["err_copy"] > 0 else float("nan")
        for k, value in m.items():
          self._log_metric(stage, f"probe/t{t:g}/{k}", value, count)
      if count is not None:
        self._log_metric(stage, "probe/n_volumes", float(count), 1)

  def on_validation_end(self):
    dm = getattr(self.trainer, "datamodule", None)
    if dm is not None and getattr(dm, "validation", None) == "lightning" and hasattr(dm, "reseed_lightning"):
      if not self.trainer.sanity_checking:   # the sanity check doesn't count: the first real one gets the same sample
        dm.lightning_runs += 1
      dm.reseed_lightning()

  def on_validation_epoch_start(self):
    # a fresh bag per validation set at every validation (each splits its volumes evenly over diffusion.val_t), shuffled by an RNG
    # of this validation and set
    self._val_t_bags, self._val_t_rngs = {}, {}
    self._probe = {}   # diffusion.probe: this validation's values per set (_diffusion_probe)

  def on_validation_epoch_end(self):
    if self.__dict__.get("_probe"):
      self._log_probe()
    if getattr(self, "grids", None) is not None and not self.trainer.sanity_checking:
      self._log_grid_metrics()

  # ---- diffusion.learn_grids -------------------------------------------------------------------------------------------------

  @torch.no_grad()
  def _with_grids(self, batch):
    """A validation batch with x0_raw (float32) the volumes' learnable grids where they have one (the training volumes: val_train,
    val_views_strain); the others (held-out sources) keep the dataset's fitted grid with the pretrained init, and get their fresh
    init with the fresh one (what a training volume started from), so every x_t a validation builds is from the run's own data."""
    lg = learnable_grids.learn_grids_config(self.cfg)
    x0 = batch["x0_raw"].to(self.device).float().clone()
    hit = batch["source"]["hit"].to(self.device)
    for b, v in enumerate(batch["source_view"]):
      v = int(v)
      if self.grids.has(v):
        x0[b] = self.grids.dense([v], hit[b:b + 1])[0]
      elif str(lg.init) == "fresh":
        src = {k: batch["source"][k][b].cpu().numpy() for k in ("hit", "xyz_cam", "K_depth")}
        colors = learnable_grids.read_colors(_get_h5(self._grid_eval["source_h5"]), v, str(lg.fresh.color), src["hit"].shape[0])
        rows = torch.from_numpy(learnable_grids.fresh_rows(src, colors, float(lg.fresh.opacity), self.grids.scale_param)).to(self.device)
        x0[b] = 0
        x0[b][:, hit[b]] = rows.T
    return {**batch, "x0_raw": x0}

  def _grid_render(self, rows, view, mesh, targets):
    """Per target view (V,) image metrics of one volume's raw rows (n, CHANNELS) rendered as they are (no clamp, no model) at the
    render rows `targets`: psnr, fg_psnr, bg_l1 (image_quality), dssim (1 - SSIM, the photometric loss's), lpips
    (learn_grids.metrics.lpips's network; left out when it is null or unavailable)."""
    lg = learnable_grids.learn_grids_config(self.cfg)
    path = self._grid_eval["source_h5"]
    item = _build_item(_get_h5(path), path, view, targets, int(self.cfg.data.num_layers), mesh,
                       depth_size=self.cfg.data.depth_size, target_depth=False)
    batch = collate_with_batch_size([item])
    hit = batch["source"]["hit"].to(self.device)
    x = rows.new_zeros((1, gs_diffusion.CHANNELS, *hit.shape[1:]))
    x[0][:, hit[0]] = rows.T
    c = gs_diffusion.CHANNELS
    gauss = gs_diffusion.decode_x0(x.permute(0, 2, 1, 3, 4).float(), torch.zeros(c, device=self.device), torch.ones(c, device=self.device))
    per = collections.defaultdict(list)
    chunk = int(lg.metrics.chunk)
    for lo in range(1, len(targets) + 1, chunk):
      rvi = list(range(lo, min(lo + chunk, len(targets) + 1)))
      pred = model_forward(self.model, batch, need_render=True, render_view_idx=rvi, gauss=gauss, device=self.device)["rgb"][0]
      gt = batch["views"]["rgb"][0, rvi].to(self.device)
      for k, v in image_quality(pred[None], gt[None], batch["views"]["alpha"][0, rvi].to(self.device)[None]).items():
        per[k].append(v[0])
      per["dssim"].append(self.dssim(pred, gt).flatten(1).mean(1))
      lp = self._lpips()
      if lp is not None:
        per["lpips"].append(torch.cat([lp(pred[i:i + 1], gt[i:i + 1]) for i in range(len(rvi))]))   # (one at a time: VGG at 504^2)
    return {k: torch.cat(v) for k, v in per.items()}

  def _lpips(self):
    """learn_grids.metrics.lpips's per-image LPIPS (torchmetrics, inputs in [0, 1]), made once and kept out of the module's state
    (no checkpoint entry); None when off or when it can't be made (warned once)."""
    if "_lpips_fn" not in self.__dict__:
      net = learnable_grids.learn_grids_config(self.cfg).metrics.get("lpips")
      fn = None
      if net is not None:
        try:
          from torchmetrics.functional.image.lpips import _NoTrainLpips
          lp = _NoTrainLpips(net=str(net)).to(self.device).eval()
          fn = lambda a, b: lp(a, b, normalize=True).flatten()
        except Exception as e:   # no weights offline, ...: the other metrics still run
          log.warning("learn_grids: LPIPS (%s) unavailable, not logged: %s: %s", net, type(e).__name__, e)
      self.__dict__["_lpips_fn"] = fn
    return self.__dict__["_lpips_fn"]

  @torch.no_grad()
  def _grid_quality(self, rows_of):
    """{<set>/<metric>: mean over the metric volumes of the mean over the set's views} of the grids rows_of(i) (volume i's raw rows)."""
    out = collections.defaultdict(list)
    for view, mesh in self._grid_eval["volumes"]:
      rows = rows_of(self.grids.index[view]).to(self.device).float()
      for name, targets in self._grid_eval["sets"].items():
        if targets[view]:
          with torch.autocast(device_type=self.device.type, enabled=False):
            for k, v in self._grid_render(rows, view, mesh, targets[view]).items():
              out[f"{name}/{k}"].append(float(v.mean()))
    return {k: float(np.mean(v)) for k, v in out.items()} | {"n_volumes": float(len(self._grid_eval["volumes"]))}

  @torch.no_grad()
  def _log_grid_metrics(self):
    """At every validation step (once, whichever validation runs first there): the learnable grids' own render quality (_grid_quality)
    and parameter statistics (learnable_grids.param_stats, over all training volumes) as grids/<...>; once, at the first one, the
    same of the fitted grids as grids_ref/<...> and of the init as grids_init/<...> (fresh init only: pretrained, the init is the
    reference)."""
    step = int(self.global_step)
    if self._grid_logged.get("step") == step:
      return
    self._grid_logged["step"] = step
    g = self.grids
    with timed("learn_grids metrics"):
      metrics = {}
      if not self._grid_logged.get("once"):
        self._grid_logged["once"] = True
        ref = (lambda i: g.volume_rows(i, init=True)) if g.ref_rows is None else (
          lambda i: g.ref_rows[int(g.starts[i]):int(g.starts[i + 1])])
        sources = {"grids_ref": (ref, g.rows(init=True) if g.ref_rows is None else g.ref_rows.to(self.device))}
        if g.ref_rows is not None:
          sources["grids_init"] = (lambda i: g.volume_rows(i, init=True), g.rows(init=True))
        for prefix, (rows_of, all_rows) in sources.items():
          metrics |= {f"{prefix}/{k}": v for k, v in self._grid_quality(rows_of).items()}
          metrics |= {f"{prefix}/stats/{k}": v for k, v in learnable_grids.param_stats(all_rows.float(), g.foot, g.pairs,
                                                                                         scale_param=g.scale_param).items()}
      metrics |= {f"grids/{k}": v for k, v in self._grid_quality(lambda i: g.volume_rows(i).detach()).items()}
      stats = learnable_grids.param_stats(g.rows().detach(), g.foot, g.pairs, init_rows=g.rows(init=True), scale_param=g.scale_param)
      metrics |= {f"grids/stats/{k}": v for k, v in stats.items()}
    for k, v in metrics.items():
      self.log(k, v, add_dataloader_idx=False)
    log.info("learn_grids at step %d: %s", step, {k: round(v, 3) for k, v in metrics.items()
                                                    if k.endswith(("/psnr", "/fg_psnr", "/lpips", "frac_degenerate")) or "/drift/" in k})

  def on_train_end(self):
    if getattr(self, "grids", None) is not None:
      self.dump_grids(os.path.join(self.output_dir, "learned_grids"))

  @torch.no_grad()
  def dump_grids(self, out_dir):
    """The grids as grid files (grid_io.write_sparse, the fitted files' format and attributes, plus learn_grids_* attributes), one
    per volume: <out_dir>/<the fitted file's name>; each also holds its raw x0 rows as the dataset x0_raw (N, CHANNELS)."""
    from grid_io import write_sparse
    os.makedirs(out_dir, exist_ok=True)
    lg = learnable_grids.learn_grids_config(self.cfg)
    for i, path in enumerate(self.grids.paths):
      with h5py.File(path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs if k not in ("layout", "layout_version", "grid_shape", "row_order", "index")}
      attrs |= {"learn_grids_init": str(lg.init), "learn_grids_learnable": bool(self.grids.learnable),
                "learn_grids_step": int(self.global_step)}
      grid, raw = self.grids.to_grid(i, attrs)
      dest = os.path.join(out_dir, os.path.basename(path))
      write_sparse(dest, grid)
      with h5py.File(dest, "a") as f:
        f.create_dataset("x0_raw", data=raw, compression="gzip", compression_opts=4)
    log.info("learn_grids: %d grids written to %s", len(self.grids.paths), out_dir)

  def _pop_val_t(self, stage):
    """The next timestep of the validation set's bag of diffusion.val_t (refilled, shuffled, when empty). The shuffles of a
    validation are seeded by diffusion.eval_seed and the set's validation seed (GSDataModule.reseed_validation: cfg.seed, the
    validation's index, the set), so at its k-th validation every run -- and a resumed one -- gives each volume the same t."""
    bags = self.__dict__.setdefault("_val_t_bags", {})
    if not bags.get(stage):
      rngs = self.__dict__.setdefault("_val_t_rngs", {})
      if stage not in rngs:
        seed = self.trainer.datamodule.validation_seeds[stage]
        rngs[stage] = np.random.default_rng([int(self.cfg.diffusion.eval_seed), int(seed)])
      bags[stage] = [float(v) for v in rngs[stage].permutation(list(self.cfg.diffusion.val_t))]
    return bags[stage].pop()

  @torch.no_grad()
  def _diffusion_val_t(self, stage, batch):
    """The rendering loss of the x0 prediction at a SHARED timestep per volume (every layer at the same t, as in the sampler), one
    t per volume from the validation set's bag of diffusion.val_t: logs loss/photom_t<t>[_l1|_dssim] over the volumes that got that
    t (t = 1: the 1-step sample, the x0 prediction from pure noise). One forward and one render for the whole batch."""
    B = batch["batch_size"]
    tv = torch.tensor([self._pop_val_t(stage) for _ in range(B)], device=self.device)   # (B,)
    image, hit, seeds, floor = self.model.inputs(batch, self.device)
    L = hit.shape[1]
    x0_hat = self.model.training_loss(image, hit, batch["x0_raw"].to(self.device), seeds=seeds, return_pred=True,
                                      t=tv[:, None].expand(B, L))[3]
    per = self._photom_per_volume(batch, self.model.decode(x0_hat, floor))
    for v in sorted(set(tv.tolist())):
      self._log_photom(stage, per, (tv == v).float(), f"loss/photom_t{v:g}")

  def _step(self, stage, batch, batch_idx, photom_key="loss/photom"):
    B = batch["batch_size"]
    _, V, *_ = batch["views"]["rgb"].shape
    H, W = batch["views"]["rgb"].shape[-2:]

    @ft.wraps(self.log)
    def _log(key, *args, **kwargs):
      name = f"{stage}/{key}" if stage == "train" else f"{stage}/{self.trainer.datamodule.validation}/{key}"
      self.log(name, *args, **kwargs, batch_size=B, add_dataloader_idx=False)

    cfg = self.cfg
    device = self.device

    source_view_idx = 0
    target_view_idx = list(range(1, V))

    # figure out which views we'll render
    render_view_idx = []

    render_source_view_idx = None
    if "source" in cfg.loss.photom["for"]:
      render_source_view_idx = len(render_view_idx)
      render_view_idx += [source_view_idx]

    render_target_view_idx = None
    if "target" in cfg.loss.photom["for"]:
      render_target_view_idx = list(range(len(render_view_idx), len(render_view_idx) + len(target_view_idx)))
      render_view_idx += target_view_idx

    need_render = (
      cfg.loss.photom.l1_weight > 0
      or cfg.loss.photom.dssim_weight > 0
      # or cfg.loss.photom.mask_weight > 0
    )

    if stage == "train":
      self.views_seen += B * len(render_view_idx)
      self.pixels_seen += B * len(render_view_idx) * H * W

    # Per-item background compositing (loss.background): train-stage only,
    # and only affects this render/loss computation -- _preview_entry
    # (panels/orbit videos) never calls model_forward, so visualizations are
    # unaffected regardless of this setting.
    bg_colors = None
    if stage == "train" and cfg.loss.background == "random":
      bg_colors = torch.rand(B, 3, device=device)
    elif stage == "train" and cfg.loss.background == "white":
      bg_colors = torch.ones(B, 3, device=device)

    # forward pass
    model_pred = model_forward(
      self.model,
      batch,
      need_render=need_render,
      render_view_idx=render_view_idx,
      bg_colors=bg_colors,
      device=device,
    )
    pred_rgb = model_pred["rgb"]

    # loss
    loss = torch.zeros((), device=device)
    out = {}

    # photometric losses
    gt_rgb = batch["views"]["rgb"][:, render_view_idx]
    gt_alpha = batch["views"]["alpha"][:, render_view_idx]
    if bg_colors is not None:
      # gt_rgb is rendered (render_objaverse.py) premultiplied over BLACK,
      # i.e. gt_rgb == true_fg_color*alpha already -- recompositing onto a
      # different background is just += bg*(1-alpha), no un-premultiply
      # needed. Same bg_colors the pred render above used, so pred/gt are
      # compared on a matching background.
      gt_rgb = gt_rgb + bg_colors[:, None, :, None, None] * (1.0 - gt_alpha)

    def _splatter_metric(metric_name, metric_full):
      """
      metric_full: (B,V,...)
      """
      _log(f"{metric_name}", metric_full.mean().detach())
      if render_source_view_idx is not None:
        _log(
          f"{metric_name}/source",
          metric_full[:,render_source_view_idx].mean().detach(),
        )
      if render_target_view_idx is not None:
        _log(
          f"{metric_name}/targets",
          metric_full[:,render_target_view_idx].mean().detach(),
        )

    RV = len(render_view_idx)

    def f(a):
      return rearrange(a, "b v ... -> (b v) ...")
    def g(a):
      return rearrange(a, "(b v) ... -> b v ...", b=B, v=RV)

    # TODO: Maybe integrate alpha into the loss function somehow? Maybe not?

    # TODO: LPIPS loss.

    ## L1 loss
    photom_loss = torch.zeros((), device=device)
    if cfg.loss.photom.l1_weight > 0:
      _photom_l1 = g((f(pred_rgb) - f(gt_rgb)).abs()) # (B,V,C,H,W)
      _splatter_metric(f"{photom_key}_l1", _photom_l1)
      photom_loss = photom_loss + cfg.loss.photom.l1_weight * _photom_l1.mean()

    ## D-SSIM Loss
    if cfg.loss.photom.dssim_weight > 0:
      _photom_dssim = g(self.dssim(f(pred_rgb), f(gt_rgb))) # (B,V,C,H,W)
      _splatter_metric(f"{photom_key}_dssim", _photom_dssim)
      photom_loss = photom_loss + cfg.loss.photom.dssim_weight * _photom_dssim.mean()

    for name, v in image_quality(pred_rgb, gt_rgb, gt_alpha).items():   # (B, V): psnr, fg_psnr, bg_l1
      _splatter_metric(f"{photom_key}_{name}", v)
    _log(photom_key, photom_loss.detach())
    loss = loss + photom_loss

    # regularization

    ## scale regularization
    scale_reg = torch.zeros((), device=device)
    if cfg.loss.scale_reg_weight > 0:
      scale_reg = pipe(
        model_pred["gauss"]["scale"],
        lambda a: rearrange(a, "b ... -> b (...)"),
        lambda a: a[a > cfg.loss.scale_reg_thresh],
        lambda a: a.mean() if a.numel() > 0 else torch.zeros((), device=device),
      )

      loss = loss + cfg.loss.scale_reg_weight * scale_reg
      _log("loss/scale_reg", scale_reg.detach())

    ## mean offset regularization
    # L2 penalty on the predicted offset itself (not the resolved position),
    # pulling Gaussians back toward their depth-peel anchor -- same term as
    # 3dgs-paper-repro's fit_3dgs.py mean_offset_reg_weight. Train stage
    # only, like there. Averaged over valid (hit) Gaussians only, so the
    # weight doesn't get diluted by the (many) empty pixels/layers.
    if (
      stage == "train"
      and "offset" in model_pred["gauss"]
      and cfg.loss.mean_offset_reg_weight > 0
    ):
      hit = batch["source"]["hit"].to(device) # (B,L,H,W)
      offset_sq = model_pred["gauss"]["offset"].pow(2).sum(dim=2) # (B,L,H,W)
      mean_offset_reg = offset_sq[hit].mean() if hit.any() else torch.zeros((), device=device)
      loss = loss + cfg.loss.mean_offset_reg_weight * mean_offset_reg
      _log("loss/mean_offset_reg", mean_offset_reg.detach())

    _log("loss", loss.detach())
    out["loss"] = loss

    return out

  def _preview_entry(self, item, with_layers=True):
    """One stage's {"gauss","scene_scale","views"} entry for
    preview_entries() below, computed from a SINGLE fixed dataset item
    (a fixed dataset item of the stage) via this model's own
    forward pass -- in true WORLD space, for module.PanelCallback's
    [GT | render | |diff|] panel and module.OrbitCallback's turntable
    video.

    The model's own Gaussians are natively predicted in the source view's
    own camera frame (see gs_dataset.py's module docstring -- there's no
    Blender world frame available to the model itself, by design, since it
    has to work from a single image with no other scene context). But THIS
    item is a real dataset row with a real recorded camera pose
    (source.pose_gl), so -- purely for this preview/orbit purpose, not
    anything the model itself relies on -- everything below is transformed
    into true Blender world space using that one known pose:
      - Gaussian means: gs_dataset._check_ground_truth_consistency's own
        reprojection formula (already proven there against real ground
        truth, ~1e-5 error).
      - Gaussian quats: gs_dataset.rotate_quats_wxyz, the inverse direction
        of what _load_ground_truth uses (that rotates world->camera; this
        is camera->world).
      - The whole supervision-set viewmats: recovered algebraically from
        that same pose and the existing source-relative viewmats
        (gs_dataset.relative_viewmats) -- world_viewmat[i] == viewmat[i] @
        inv(source_c2w_cv) -- no change to gs_dataset.py needed, target
        views' raw poses were never stored and don't need to be.
    This is what lets module.OrbitCallback stay fully generic: it never
    has to know this source-camera-frame-vs-world distinction exists.

    NOTE(andrei): Not sure if setting model to eval is the right move here,
    but gonna do it for now."""
    batch = collate_with_batch_size([item])
    device = self.device
    with set_mode(self.model, "eval"), torch.no_grad():
      gauss = self.model(self.model.inputs(batch, device))
      flat = next(flatten_gaussians(
        batch["batch_size"], gauss,
        batch["source"]["xyz_cam"].to(device), batch["source"]["hit"].to(device),
      ))
    flat["sh_degree"] = self.model.max_sh_degree

    # Unflattened (pre-flatten_gaussians) per-layer snapshot, for
    # module.ScaleLayersCallback -- `gauss` here is still the raw decoder
    # output (batch dim intact, one layer axis per Gaussian field), never
    # mutated by the flatten/world-space-transform steps below. batch dim
    # is 1 (single fixed preview item), so index it out: each field is
    # (L,C,H,W), matching flatten_gaussians' own per-item slicing.
    # (with_layers=False: left out, it is ~120 MB per object -- only ScaleLayersCallback reads it, and only
    # for the first object of a stage)
    gauss_layers = None
    if with_layers:
      gauss_layers = {k: v[0].detach() for k, v in gauss.items()}
      gauss_layers["hit"] = batch["source"]["hit"][0]

    # world <- source-camera-frame, from this item's own known real pose.
    pose_gl_np = batch["source"]["pose_gl"][0].numpy()
    c2w_cv = pose_gl_np @ OPENGL_TO_OPENCV

    means_np = flat["means"].detach().cpu().numpy()
    means_h = np.concatenate([means_np, np.ones((len(means_np), 1), np.float32)], axis=-1)
    flat["means"] = torch.from_numpy((means_h @ c2w_cv.T)[:, :3].astype(np.float32)).to(device)

    quats_np = flat["quats"].detach().cpu().numpy()
    quats_world = rotate_quats_wxyz(quats_np, c2w_cv[:3, :3])
    flat["quats"] = torch.from_numpy(quats_world.astype(np.float32)).to(device)

    v = batch["views"]
    viewmat_np = v["viewmat"][0].numpy()               # (V,4,4), source-relative
    world_viewmat = viewmat_np @ np.linalg.inv(c2w_cv)  # (V,4,4), true world-to-camera

    return {
      "mesh_index": int(item["mesh_index"]) if "mesh_index" in item else None,   # for the orbit grid's caption
      "gauss": flat,
      **({"gauss_layers": gauss_layers} if gauss_layers is not None else {}),
      # scene_scale: the source camera's own real distance from the world
      # origin -- module.OrbitCallback's orbit radius (matches
      # the deleted fit_gsplat.py's scene_scale: norm of a capture camera's
      # own world position).
      "scene_scale": float(np.linalg.norm(pose_gl_np[:3, 3])) or 1.0,
      "views": {
        "viewmat": torch.from_numpy(world_viewmat.astype(np.float32)).to(device),
        "K": v["K_image"][0].to(device),
        "width": v["rgb"].shape[-1],
        "height": v["rgb"].shape[-2],
        "gt_rgb": v["rgb"][0].to(device),
      },
    }

  def _preview_indices(self, ds):
    """Dataset indices of the preview objects, in a fixed order: ds[0] first (what the single-object preview
    has always used), then the first view of every OTHER mesh in the order the meshes first appear -- so the
    objects of a grid are distinct meshes, and any prefix of the list is the same whatever its length."""
    df = getattr(getattr(ds, "volumes", None), "df", None)
    if df is not None and {"mesh_id", "view_idx"} <= set(df.columns):
      mesh, view = df["mesh_id"].to_list(), df["view_idx"].to_list()
      best = {}
      for i, (m, v) in enumerate(zip(mesh, view)):
        if m not in best or v < best[m][0]:
          best[m] = (v, i)
      return [0] + [best[m][1] for m in dict.fromkeys(mesh) if m != mesh[0]]
    return list(range(len(ds)))

  def _preview_datasets(self):
    dm = self.trainer.datamodule
    return {stage: ds for stage, ds in {"train": dm.train_ds, **dm.val_datasets}.items() if len(ds) > 0}

  def preview_stages(self):
    """The stages that have preview objects: "train", then each val set. See module.py's comment block
    for the contract."""
    return list(self._preview_datasets())

  def preview_entries(self, stage, count=1, with_layers=False):
    """A generator of up to `count` entries of a stage -- distinct meshes (see _preview_indices), each its own
    _preview_entry() call, made when the generator reaches it and kept by nothing here: the consumer holds at
    most the one it is working on. Nothing is cached between calls. with_layers: the first entry also carries
    gauss_layers (module.ScaleLayersCallback), which is ~120 MB."""
    ds = self._preview_datasets()[stage]
    for rank, i in enumerate(self._preview_indices(ds)[:count]):
      yield self._preview_entry(ds[i], with_layers=with_layers and rank == 0)


def torch_seed(cfg):
  """The torch seed: train_seed -- the training randomness alone (weights init, noise, data order), a replicate of a run on the
  same splits -- or, when it is null or absent, seed (which the train/val splits always use)."""
  train_seed = OmegaConf.select(cfg, "train_seed", default=None)
  return int(cfg.seed if train_seed is None else train_seed)


@hydra.main(version_base=None, config_path="conf", config_name="train_gs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(torch_seed(cfg))
  torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)

  # The run's events by name (module.Events): fired by cfg.events' cadences and the scheduler's
  # trough_event, read by the callbacks with a trigger_event. Set first: everything that names an event
  # registers itself with it as it is built.
  events = Events(OmegaConf.to_container(cfg.events, resolve=True))
  events.make_current()

  datamodule = GSDataModule(cfg)
  datamodule.setup()

  with open_dict(cfg):
    # TODO: Grad accum logic
    cfg.total_steps = len(datamodule.train_dataloader()) * cfg.trainer.max_epochs

  model = GSLightningModule(cfg, datamodule.diffusion_stats, grid_init=datamodule.grid_init)

  # Stashed into cfg (not just logged) so it lands in wandb's persisted run
  # config below (OmegaConf.to_container(cfg, ...)) -- unlike the console-only
  # "Total params" line from Lightning's own ModelSummary table, this is
  # queryable after the fact. open_dict since Hydra's cfg is struct-locked by
  # default (adding a key not in the yaml schema would otherwise raise
  # ConfigAttributeError).
  with open_dict(cfg):
    cfg.model.num_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("grids."))   # (not the learnable grids)

  # A TrainClock stops the run after max_seconds of training time; a time-driven LR scheduler
  # (sched=time_wsd, module.TimeWSD) reads the same clock, which configure_optimizers hands it; so do
  # triggers with an every_n_seconds cadence (the clock then only counts, if max_seconds is null)
  clock = None
  if cfg.sched is not None and needs_clock(hydra.utils.instantiate(cfg.sched)) and cfg.max_seconds is None:
    raise SystemExit("this LR scheduler is driven by the training time: set max_seconds")
  specs = [OmegaConf.select(cb, "trigger_event") for cb in cfg.callbacks.values() if cb is not None] + list(cfg.events.values())
  seconds_cadence = any(isinstance(spec, DictConfig) and spec.get("every_n_seconds") is not None for spec in specs)
  if cfg.max_seconds is not None or seconds_cadence:   # no max_seconds: the clock only counts
    clock = TrainClock(max_seconds=cfg.max_seconds)
  model.clock = clock

  logger = False
  if cfg.wandb.mode != "disabled":
    tags: list[str] = OmegaConf.to_container(cfg.wandb.tags, resolve=True)

    # NOTE(andrei): Ensure that there's a tag to easily distinguish these runs
    # from others, and that it's the first tag that shows up.
    tag = "3dgs-model"
    with ctl.suppress(ValueError):
      tags.remove(tag)
    tags = [tag, *tags]

    resume = cfg.wandb.get("resume")
    wandb_run = wandb.init(
      project=cfg.wandb.project,
      mode=cfg.wandb.mode,
      tags=tags,
      name=cfg.wandb.name,
      id=cfg.wandb.get("id"),
      resume=resume,
      # a resumed run's config already has values: it is updated below, allowing changes
      config=None if resume else OmegaConf.to_container(cfg, resolve=True),
    )
    if resume:   # the resumed run's config takes this segment's values (e.g. the extended max_seconds, ckpt_path)
      wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    logger = ProgressWandbLogger(
      experiment=wandb_run,
      progress_fn=lambda: {
        "views_seen": model.views_seen, "pixels_seen": model.pixels_seen,
        **({} if clock is None else {"train_seconds": clock.seconds()}),
      },
      step_fn=lambda: _completed_steps(model.trainer, model),
    )

  callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
  callbacks = [events, *callbacks]   # before every consumer: see module.Events
  if clock is not None:
    callbacks = [clock, *callbacks, clock.tail()]   # first and last: see module.TrainClock

  trainer = pl.Trainer(
    **OmegaConf.to_container(cfg.trainer, resolve=True),
    logger=logger,
    callbacks=callbacks,
    plugins=[SameDirCheckpointIO()],   # checkpoints written next to their destination, not through /tmp
  )
  with timed("train"):
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
  main()
