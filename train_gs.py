#!/usr/bin/env python3
"""Train a point-cloud-anchored 3D Gaussian Splatting prediction model on
cached World Tracing decoder tokens.

Input is a wt_features.py output h5 (the (layers, patches, dim) token volume
World Tracing's decoder produced for one view, from a noised GT layered
point cloud). A per-patch linear layer (gs_decoder.PatchLinearGaussianHead)
maps each token to per-pixel per-layer Gaussian appearance/shape parameters
(opacity, scale, rotation, SH color), with the same activations/init as the
Flash3D-style decoder this script used to train. Gaussian positions are
anchored to the source view's own depth-peel point cloud (see gs_dataset.py's
dense_unproject_camera); with model.predict_mean_offset the head also
predicts a residual offset on top of that anchor (see flatten_gaussians).

For now this is overfitting only: data.item picks ONE row of the features h5
(gs_dataset.WTFeatureDataset) and the model trains on it alone.

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
        ckpt_path=/app/outputs/3dgs-model-2026-09-01_12-00-00/last.ckpt

gsplat JIT-compiles CUDA kernels on first import; see _setup_cuda_toolchain
(copied from fit_gsplat.py, which needs the same env setup).
"""

import contextlib as ctl
import functools as ft
import logging
import os
import sys

import gsplat
import h5py
import hydra
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from gs_dataset import (OPENGL_TO_OPENCV, H5Catalog, WTFeatureDataset,  # noqa: E402
                        _EmptyDataset, rotate_quats_wxyz, split_views_per_mesh)
from gs_decoder import PatchLinearGaussianHead  # noqa: E402
from module import DSSIMLoss  # noqa: E402
from util import collate_with_batch_size, pipe, set_mode, timed  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class GSModel(nn.Module):
  """Per-patch linear head over cached World Tracing decoder tokens: maps the
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

    self.head = PatchLinearGaussianHead(
      feature_dim=self.feature_dim,
      num_layers=self.num_layers,
      max_sh_degree=self.max_sh_degree,
      patch_size=int(cfg.model.patch_size),
      opacity_scale=cfg.model.opacity_scale, opacity_bias=cfg.model.opacity_bias,
      scale_scale=cfg.model.scale_scale, scale_bias=cfg.model.scale_bias,
      sh_scale=cfg.model.sh_scale, scale_lambda=cfg.model.scale_lambda,
      zero_init_last_conv=cfg.model.zero_init_last_conv,
      predict_mean_offset=cfg.model.predict_mean_offset,
    )

  def forward(self, features):
    """features: (B, L, P, D) -> dict of (B, L, C, H, W) Gaussian parameters,
    H = W = sqrt(P) * patch_size."""
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
  need_flat_gauss: bool=False,
  need_render: bool=False,
  render_view_idx = None,
  bg_colors = None,
  device,
):
  if render_view_idx is None:
    render_view_idx = slice(None)

  B = batch["batch_size"]

  out = {}
  out["gauss"] = model(batch["features"].to(device))
  _check_grid(out["gauss"], batch["source"]["xyz_cam"])

  # render 3DGS if necessary
  if need_render:
    rvi = render_view_idx

    # TODO: overrides
    # gauss_render = gauss_pred
    # gt = item.get("ground_truth")
    # if gt is not None:
    #   gauss_render = apply_ground_truth_overrides(gauss, gt, hit, predict_params, device)
    # src = item["source"]
    # hit = src["hit"].to(device)

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

      x_pred_rgb, x_pred_alpha, _ = gsplat.rasterization(
        means=x_gauss_flat["means"],
        quats=x_gauss_flat["quats"],
        scales=x_gauss_flat["scales"],
        opacities=x_gauss_flat["opacities"],
        colors=x_gauss_flat["colors"],
        viewmats=x_viewmats,
        Ks=x_Ks,
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
    if "sh_rest" in gauss:
      colors_flat = pack(
        [
          colors_flat[idx],
          rearrange(gauss["sh_rest"], "l (k c) h w -> (l h w) k c", c=3)
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
  """Train/val over cached feature volumes: both datasets hold the same
  volumes (data.item: one row, a list of rows, or null for all of them), but
  sample their supervision views from disjoint per-mesh view splits
  (data.val_fraction of each mesh's views go to val; 0 = no validation)."""

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.train_ds = None
    self.val_ds = None

  def setup(self, stage=None):
    if self.train_ds is not None:
      return
    d = self.cfg.data
    with h5py.File(d.features_h5, "r") as f:
      source_h5 = str(f.attrs.get("source_h5", f.attrs["input"]))
    views = H5Catalog(
      source_h5, H5Catalog.path(), H5Catalog.index().alias("view_idx"),
      H5Catalog.dataset("mesh_index").alias("mesh_id"),
    )
    train_views, val_views = split_views_per_mesh(views, d.val_fraction, seed=self.cfg.seed)
    rows = None if d.item is None else ([d.item] if isinstance(d.item, int) else list(d.item))
    kwargs = dict(
      rows=rows, num_layers=d.num_layers, num_views=d.num_views,
      allow_source_as_target=d.allow_source_as_target,
    )
    self.train_ds = WTFeatureDataset(d.features_h5, train_views, **kwargs)
    self.val_ds = (
      WTFeatureDataset(d.features_h5, val_views, **kwargs)
      if len(val_views) else _EmptyDataset()
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_ds,
      shuffle=True,
      collate_fn=collate_with_batch_size,
      **OmegaConf.to_container(self.cfg.loader, resolve=True),
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_ds,
      shuffle=False,
      collate_fn=collate_with_batch_size,
      num_workers=0,
      batch_size=self.cfg.loader.batch_size,
    )


class GSLightningModule(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.views_seen = 0

    with timed("build_model"):
      self.model = GSModel(cfg)
    self.dssim = DSSIMLoss(reduction="none")

    # One directory for everything this run produces (currently just
    # checkpoints, but named generically for whatever else lands here later
    # -- orbit videos/panels are wandb-only right now).
    # Defaults to the Unix timestamp at startup, so back-to-back runs never
    # share a directory unless cfg.output_dir is set explicitly.
    self.output_dir = cfg.output_dir
    os.makedirs(self.output_dir, exist_ok=True)

  def configure_optimizers(self):
    out = {}

    out["optimizer"] = hydra.utils.instantiate(self.cfg.optim)(self.model.parameters())
    if OmegaConf.select(self.cfg, "sched") is not None:
      out["lr_scheduler"] = {
        "scheduler": hydra.utils.instantiate(self.cfg.sched)(out["optimizer"]),
        "interval": "step",
      }

    return out

  def on_save_checkpoint(self, checkpoint):
    # views_seen is a plain Python int, not part of state_dict() -- persist
    # it explicitly so it keeps climbing (not resetting to 0) across resume.
    checkpoint["views_seen"] = self.views_seen

  def on_load_checkpoint(self, checkpoint):
    # .get: checkpoints written before this hook existed have no such key.
    self.views_seen = checkpoint.get("views_seen", 0)

  def training_step(self, batch, batch_idx):
    return self._step("train", batch, batch_idx)

  def validation_step(self, batch, batch_idx):
    return self._step("val", batch, batch_idx)

  def _step(self, stage, batch, batch_idx):
    B = batch["batch_size"]
    _, V, *_ = batch["views"]["rgb"].shape

    @ft.wraps(self.log)
    def _log(key, *args, **kwargs):
      self.log(f"{stage}/{key}", *args, **kwargs, batch_size=B)

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
    self.log("views_seen", self.views_seen, reduce_fx="max")

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
    pred_alpha = model_pred["alpha"]

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
      _splatter_metric("loss/photom_l1", _photom_l1)
      photom_loss = photom_loss + cfg.loss.photom.l1_weight * _photom_l1.mean()

    ## D-SSIM Loss
    if cfg.loss.photom.dssim_weight > 0:
      _photom_dssim = g(self.dssim(f(pred_rgb), f(gt_rgb))) # (B,V,C,H,W)
      _splatter_metric("loss/photom_dssim", _photom_dssim)
      photom_loss = photom_loss + cfg.loss.photom.dssim_weight * _photom_dssim.mean()

    _log("loss/photom", photom_loss.detach())
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

    # TODO: skim through this code, fix it up
    # if need_direct:
    #   raise NotImplementedError()
    # if need_direct:
    #   direct_total, direct_parts = compute_direct_loss(gauss, gt, hit, cfg.loss, device)
    #   loss = loss + direct_total
    #   for k, v in direct_parts.items():
    #     metrics[f"direct_{k}"] = v.detach()

    _log("loss", loss.detach())
    out["loss"] = loss

    # TODO: Are any of these metrics actually needed?
    # metrics["mean_opacity"] = flat["opacities"].mean().detach() if flat["opacities"].numel() else torch.zeros((), device=device)
    # metrics["mean_scale"] = flat["scales"].mean().detach() if flat["scales"].numel() else torch.zeros((), device=device)
    # metrics["frac_kept"] = torch.tensor(
    #   flat["means"].shape[0] / max(1, item["source"]["hit"].numel()), device=device)

    return out

  def _preview_entry(self, item):
    """One stage's {"gauss","scene_scale","views"} entry for
    get_preview_source() below, computed from a SINGLE fixed dataset item
    (train_ds[0], or val_ds[0] for the "val" stage) via this model's own
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
      gauss = self.model(batch["features"].to(device))
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
      "gauss": flat,
      "gauss_layers": gauss_layers,
      # scene_scale: the source camera's own real distance from the world
      # origin -- module.OrbitCallback's orbit radius (matches
      # fit_gsplat.py's own scene_scale: norm of a capture camera's own
      # world position).
      "scene_scale": float(np.linalg.norm(pose_gl_np[:3, 3])) or 1.0,
      "views": {
        "viewmat": torch.from_numpy(world_viewmat.astype(np.float32)).to(device),
        "K": v["K_image"][0].to(device),
        "width": v["rgb"].shape[-1],
        "height": v["rgb"].shape[-2],
        "gt_rgb": v["rgb"][0].to(device),
      },
    }

  def get_preview_source(self, mode):
    """{"train": entry, "val": entry-or-None} for module.PanelCallback/
    OrbitCallback -- see module.py's own comment block for the full
    contract. Unlike fit_gsplat.py's version, train and val here are NOT
    the same Gaussians rendered against different views -- the model
    predicts an entirely different Gaussian set per forward-passed item,
    so each stage gets its own full _preview_entry() call (own gauss, own
    scene_scale, own views), from a fixed dataset item (train_ds[0], or
    val_ds[0] for "val").

    mode picks both the cache granularity (epoch: once per
    self.current_epoch; step: once per self.trainer.global_step) and
    whether "val" is computed at all -- skipped (left None) in epoch mode
    even when a val split exists, since nothing pulls it there (see
    module.PanelCallback._step) -- this avoids the extra forward pass on
    every epoch-cadence tick when it's not needed."""
    key = (mode, self.current_epoch if mode == "epoch" else self.trainer.global_step)
    if getattr(self, "_preview_cache_key", None) == key:
      return self._preview_cache

    val_ds = self.trainer.datamodule.val_ds
    source = {
      "train": self._preview_entry(self.trainer.datamodule.train_ds[0]),
      "val": self._preview_entry(val_ds[0]) if mode == "step" and len(val_ds) > 0 else None,
    }
    self._preview_cache_key, self._preview_cache = key, source
    return source


@hydra.main(version_base=None, config_path="conf", config_name="train_gs")
def main(cfg: DictConfig) -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
  torch.manual_seed(int(cfg.seed))
  torch.autograd.set_detect_anomaly(cfg.torch_detect_anomaly)

  datamodule = GSDataModule(cfg)
  datamodule.setup()

  with open_dict(cfg):
    # TODO: Grad accum logic
    cfg.total_steps = len(datamodule.train_dataloader()) * cfg.trainer.max_epochs

  model = GSLightningModule(cfg)

  # Stashed into cfg (not just logged) so it lands in wandb's persisted run
  # config below (OmegaConf.to_container(cfg, ...)) -- unlike the console-only
  # "Total params" line from Lightning's own ModelSummary table, this is
  # queryable after the fact. open_dict since Hydra's cfg is struct-locked by
  # default (adding a key not in the yaml schema would otherwise raise
  # ConfigAttributeError).
  with open_dict(cfg):
    cfg.model.num_params = sum(p.numel() for p in model.parameters())

  logger = False
  if cfg.wandb.mode != "disabled":
    tags: list[str] = OmegaConf.to_container(cfg.wandb.tags, resolve=True)

    # NOTE(andrei): Ensure that there's a tag to easily distinguish these runs
    # from others, and that it's the first tag that shows up.
    tag = "3dgs-model"
    with ctl.suppress(ValueError):
      tags.remove(tag)
    tags = [tag, *tags]

    wandb_run = wandb.init(
      project=cfg.wandb.project,
      mode=cfg.wandb.mode,
      tags=tags,
      name=cfg.wandb.name,
      config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger = WandbLogger(experiment=wandb_run)

  callbacks = []
  callbacks += list(hydra.utils.instantiate(cfg.callbacks).values())

  trainer = pl.Trainer(
    **OmegaConf.to_container(cfg.trainer, resolve=True),
    logger=logger,
    callbacks=callbacks,
  )
  with timed("train"):
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
  main()
