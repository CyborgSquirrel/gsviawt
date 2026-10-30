"""Diffusion mode of train_gs.py: a diffusion model over the fitted layered 3DGS grids (World Tracing's recipe).

The data are fit_3dgs.py `gaussian_layout=layered` fits, one finalized grid .h5 per source view (see
bla/finalize_fit_dataset.py: `mesh<M>_view<V>.h5`, (H, W, 6, .) arrays, NaN where a slot has no Gaussian). Gaussian POSITIONS come
from the source view's point cloud (the depth peel, as everywhere in train_gs.py); the model generates, per slot, 14 channels
(CHANNELS), normalized per channel (x0):
    opacity (1)   logit, clamped to +-7
    scale (3)     log of the absolute size
    rotation (4)  unit quaternion in the source camera frame, sign canonicalized (w >= 0)
    sh_dc (3)     (rgb - 0.5) / SH_C0 (degree-0 colour, what the head's sh_dc means)
    offset (3)    the fitted mean minus the depth-peel position (camera frame)

WT's recipe, as the paper gives it (https://arxiv.org/abs/2606.13652): rectified flow x_t = (1 - t) x0 + t eps, the network outputs x0
(x0-parameterization), the loss is the squared L2 distance to x0 on the valid slots, invalid slots' input is replaced by the noise
(x_net = A x_t + (1 - A) eps, A = the point cloud's per-layer hit mask), timestep conditioning by AdaLN (t * 1000, sinusoidal, MLP, 6
modulation vectors per block), per-layer timesteps (diffusion_schedule.py), 20 Euler steps of v = (x_t - x0_hat) / max(t, 0.05).

Architecture otherwise as the baseline's: the frozen DINOv2 mix backbone (model/backbone=moge_mix) and a 3-block frame/ray/global
trunk; the token is lossless, the 1024 feature channels next to the raw 14 * 14 * 14 noisy values of the patch (padded to a multiple
of 64 for the heads).
"""

import contextlib
import glob
import json
import logging
import math
import os

import h5py
import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch.utils.data import Dataset

from diffusion_schedule import LayerAwareSchedule
from gs_dataset import OPENGL_TO_OPENCV, rotate_quats_wxyz
from gs_decoder import AttentionBlock, apply_rope, rope_cos_sin
from module import SH_C0

log = logging.getLogger(__name__)

FIELDS = ("opacity", "scale", "rotation", "sh_dc", "offset")
DIMS = (1, 3, 4, 3, 3)
CHANNELS = sum(DIMS)   # 14
OPACITY_LOGIT_CLAMP = 7.0
T_MIN_CLAMP = 0.05     # WT's: the x0 -> velocity conversion divides by max(t, T_MIN_CLAMP)


# ---------------------------------------------------------------------------
# grids <-> x0 channels
# ---------------------------------------------------------------------------

def load_grid(path):
  """A finalized grid file -> ({name: array (H, W, L, .)}, attrs)."""
  with h5py.File(path, "r") as f:
    grid = {k: np.asarray(f[k]) for k in (
      "gaussian_means", "gaussian_scales", "gaussian_quats", "gaussian_opacities", "gaussian_colors", "layer_valid")}
    attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items() if k != "config_json"}
  return grid, attrs


def encode_grid(grid, source):
  """The grid's Gaussians as raw (un-normalized) x0 channels, (CHANNELS, L, H, W) float32, zero at the slots without a Gaussian.
  `source`: the dataset item's "source" dict (xyz_cam (L, H, W, 3) with NaN where there is no hit, hit (L, H, W), pose_gl (4, 4))."""
  valid = grid["layer_valid"].transpose(2, 0, 1)                      # (L, H, W)
  hit = source["hit"].numpy()
  mismatch = float((valid != hit).mean())
  if mismatch > 1e-6:
    log.warning("grid layer_valid disagrees with the point cloud's hit mask on %.4f%% of the slots", 100 * mismatch)
  c2w = source["pose_gl"].numpy() @ OPENGL_TO_OPENCV
  w2c = np.linalg.inv(c2w)
  means = grid["gaussian_means"].transpose(2, 0, 1, 3)               # (L, H, W, 3), world frame
  means_cam = means @ w2c[:3, :3].T + w2c[:3, 3]
  offset = means_cam - source["xyz_cam"].numpy()

  quat = np.zeros(valid.shape + (4,), np.float32)
  quat[..., 0] = 1.0
  if valid.any():
    q = rotate_quats_wxyz(grid["gaussian_quats"].transpose(2, 0, 1, 3)[valid], c2w[:3, :3].T)   # world -> source camera
    quat[valid] = np.where(q[:, :1] < 0, -q, q)                      # q and -q are one rotation: w >= 0

  opacity = np.clip(grid["gaussian_opacities"].transpose(2, 0, 1), 1e-6, 1 - 1e-6)
  opacity = np.clip(np.log(opacity / (1 - opacity)), -OPACITY_LOGIT_CLAMP, OPACITY_LOGIT_CLAMP)
  scale = np.log(np.maximum(grid["gaussian_scales"].transpose(2, 0, 1, 3), 1e-8))
  color = (grid["gaussian_colors"].transpose(2, 0, 1, 3) - 0.5) / SH_C0

  x0 = np.concatenate([opacity[..., None], scale, quat, color, offset], axis=-1)   # (L, H, W, CHANNELS)
  x0 = np.where(valid[..., None], x0, 0.0).astype(np.float32)
  return np.ascontiguousarray(x0.transpose(3, 0, 1, 2))              # (CHANNELS, L, H, W)


def decode_x0(x0, mean, std):
  """Normalized x0 (B, L, CHANNELS, H, W) -> the Gaussian dict flatten_gaussians reads (gs_decoder.activate_gaussians' layout)."""
  raw = x0 * std[None, None, :, None, None] + mean[None, None, :, None, None]
  opacity, scale, rotation, sh_dc, offset = raw.split(DIMS, dim=2)
  return {
    "opacity": torch.sigmoid(opacity),
    "raw_scale": scale,
    "scale": torch.exp(scale),
    "rotation": F.normalize(rotation + 1e-6, dim=2),
    "sh_dc": sh_dc,
    "offset": offset,
  }


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_grid_index(grids_dir):
  """{row of the render h5 of the grid's source view: path}, and the render h5 the grids were fit from."""
  index, sources = {}, set()
  for path in sorted(glob.glob(os.path.join(grids_dir, "*.h5"))):
    with h5py.File(path, "r") as f:
      index[int(f.attrs["primary_view_row"])] = path
      sources.add(str(f.attrs["source_h5"]))
  if not index:
    raise SystemExit(f"no finalized grids (*.h5 with primary_view_row) in {grids_dir!r}")
  if len(sources) != 1:
    raise SystemExit(f"the grids in {grids_dir!r} come from several render files: {sorted(sources)}")
  return index, sources.pop()


def make_listing(grid_index, source_h5, out_path):
  """The 'features' file GSDataModule's datasets are built from, without features: one row per grid (its source view and mesh)."""
  rows = sorted(grid_index)
  with h5py.File(source_h5, "r") as f:
    meshes = f["mesh_index"][:][rows]
  with h5py.File(out_path, "w") as f:
    f.create_dataset("view_indices", data=np.array(rows, np.int64))
    f.create_dataset("mesh_index", data=np.asarray(meshes, np.int64))
    f.create_dataset("features", data=np.zeros((len(rows), 1, 1, 1), np.float16)).attrs["stored_as_uint16_bits"] = False
    f.attrs["source_h5"] = source_h5
    f.attrs["input"] = source_h5
  return out_path


class GridTargetDataset(Dataset):
  """Wraps a dataset of render items (gs_dataset.WTFeatureDataset) and adds each item's ground-truth Gaussians, "x0_raw"
  (CHANNELS, L, H, W) (see encode_grid), read from the grid file of its source view."""

  def __init__(self, inner, grid_index):
    self.inner = inner
    self.grid_index = grid_index
    self.volumes = inner.volumes   # the preview code reads the volumes' mesh/view columns
    missing = sorted(set(int(v) for v in inner.volumes.df["view_idx"].to_list()) - set(grid_index))
    if missing:
      raise SystemExit(f"source views {missing} have no grid file")

  def __len__(self):
    return len(self.inner)

  def __getitem__(self, idx):
    item = self.inner[idx]
    grid, _ = load_grid(self.grid_index[item["source_view"]])
    item["x0_raw"] = torch.from_numpy(encode_grid(grid, item["source"]))
    return item


def compute_stats(dataset, cache_path=None):
  """Per-channel mean / std of the x0 channels over the VALID slots of `dataset`'s items; cached as json (keyed by the rows)."""
  key = sorted(int(v) for v in dataset.volumes.df["view_idx"].to_list())
  if cache_path and os.path.exists(cache_path):
    with open(cache_path) as f:
      cached = json.load(f)
    if cached["rows"] == key:
      return torch.tensor(cached["mean"]), torch.tensor(cached["std"])
  total = torch.zeros(CHANNELS, dtype=torch.float64)
  total2 = torch.zeros(CHANNELS, dtype=torch.float64)
  count = 0
  for i in range(len(dataset)):
    item = dataset[i]
    x = item["x0_raw"][:, item["source"]["hit"]].double()          # (CHANNELS, valid slots)
    total += x.sum(1)
    total2 += x.square().sum(1)
    count += x.shape[1]
  mean = total / count
  std = (total2 / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
  log.info("x0 statistics over %d valid slots of %d grids: mean %s std %s", count, len(dataset), mean.numpy().round(3), std.numpy().round(3))
  if cache_path:
    with open(cache_path, "w") as f:
      json.dump({"rows": key, "count": count, "mean": mean.tolist(), "std": std.tolist()}, f)
  return mean.float(), std.float()


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t, dim=256, theta=10_000.0):
  """WT's (Wan 2.1's) timestep embedding: [cos, sin] of t * theta^(-k / (dim / 2)); t: (N,) -> (N, dim) float32."""
  half = dim // 2
  freqs = torch.outer(t.double(), torch.pow(theta, -torch.arange(half, device=t.device).double() / half))
  return torch.cat([freqs.cos(), freqs.sin()], dim=1).float()


class DiTBlock(AttentionBlock):
  """AttentionBlock with WT's DiT-style AdaLN (DecoderBlockDiT): the timestep gives, per (sample, layer), a shift / scale / gate for
  the attention and for the MLP, added to a learned per-block modulation. adaln: (B, L or 1, 6, D)."""

  def __init__(self, dim, num_heads, mode, mlp_ratio=4.0, layer_scale_init=1e-4):
    super().__init__(dim, num_heads, mode, mlp_ratio, layer_scale_init)
    self.modulation = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

  def _attend_normed(self, h, cos, sin):
    """Like AttentionBlock._attend, with the (modulated) normalization already applied."""
    n, s, d = h.shape
    q, k, v = self.qkv(h).reshape(n, s, 3, self.num_heads, d // self.num_heads).unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    if cos is not None:
      q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return self.proj(out.transpose(1, 2).reshape(n, s, d))

  def _mix(self, h, cos, sin, b, l, p, d):
    if self.mode == "frame":
      return self._attend_normed(h.reshape(b * l, p, d), cos, sin).reshape(b, l, p, d)
    if self.mode == "ray":
      return self._attend_normed(h.transpose(1, 2).reshape(b * p, l, d), None, None).reshape(b, p, l, d).transpose(1, 2)
    return self._attend_normed(h.reshape(b, l * p, d), cos.repeat(l, 1), sin.repeat(l, 1)).reshape(b, l, p, d)

  def forward(self, x, cos, sin, adaln):
    b, l, p, d = x.shape
    mod = (self.modulation + adaln.float()).unbind(2)                 # six (B, L or 1, D)
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (m[:, :, None, :] for m in mod)
    h = (self.norm1(x).float() * (1 + scale_a) + shift_a).to(x.dtype)
    x = x + (self.ls1 * self._mix(h, cos, sin, b, l, p, d) * gate_a).to(x.dtype)
    h = (self.norm2(x).float() * (1 + scale_m) + shift_m).to(x.dtype)
    return x + (self.ls2 * self.mlp(h) * gate_m).to(x.dtype)


class DiTTrunk(nn.Module):
  """gs_decoder.TransformerTrunk with AdaLN blocks: frame -> ray -> global, the per-layer FiLM before every frame block, and WT's
  timestep rule: the frame blocks get each layer's own AdaLN input, the ray and global blocks the mean over the layers."""

  MODES = ("frame", "ray", "global")

  def __init__(self, dim, num_layers, num_blocks, num_heads, mlp_ratio=4.0, layer_scale_init=1e-4, checkpoint_activations=False):
    super().__init__()
    self.checkpoint_activations = checkpoint_activations
    self.num_heads = num_heads
    self.blocks = nn.ModuleList(
      DiTBlock(dim, num_heads, self.MODES[i % 3], mlp_ratio, layer_scale_init) for i in range(num_blocks))
    self.layer_embed = nn.Embedding(num_layers, dim)
    self.film = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
    nn.init.zeros_(self.film[-1].weight)
    nn.init.zeros_(self.film[-1].bias)
    self._rope = {}

  def _rope_tables(self, grid, device):
    key = (grid, device)
    if key not in self._rope:
      cos, sin = rope_cos_sin(grid, self.blocks[0].qkv.in_features // self.num_heads)
      self._rope[key] = (cos.to(device), sin.to(device))
    return self._rope[key]

  def forward(self, x, adaln):
    _, num_layers, num_patches, _ = x.shape
    grid = int(round(num_patches ** 0.5))
    cos, sin = self._rope_tables(grid, x.device)
    gamma, beta = self.film(self.layer_embed(torch.arange(num_layers, device=x.device))).chunk(2, dim=-1)
    gamma, beta = gamma[None, :, None], beta[None, :, None]
    adaln_mean = adaln.mean(1, keepdim=True)
    for block in self.blocks:
      if block.mode == "frame":
        x = x * (1 + gamma) + beta
      a = adaln if block.mode == "frame" else adaln_mean
      if self.checkpoint_activations and torch.is_grad_enabled():
        x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, a, use_reentrant=False)
      else:
        x = block(x, cos, sin, a)
    return x


class DiffusionHead(nn.Module):
  """(image features (B, L, P, D), the network's noisy x (B, L, C, H, W), t (B, L)) -> x0_hat (B, L, C, H, W).
  Token = [normalized features | the patch's raw noisy values | zero pad]: nothing is squeezed through a projection."""

  def __init__(self, feature_dim, num_layers, channels=CHANNELS, patch_size=14, dim=None, num_blocks=3, mlp_ratio=4.0,
               layer_scale_init=1e-4, t_embed_channels=256, img_token_scale=0.3, checkpoint_activations=False):
    super().__init__()
    self.patch_size, self.channels, self.img_token_scale, self.t_embed_channels = patch_size, channels, img_token_scale, t_embed_channels
    self.noise_dim = channels * patch_size ** 2
    in_dim = feature_dim + self.noise_dim
    self.dim = dim or int(math.ceil(in_dim / 64) * 64)
    if self.dim < in_dim or self.dim % 64:
      raise ValueError(f"diffusion.dim={self.dim} must be a multiple of 64 and at least {in_dim} (the lossless token)")
    self.pad = self.dim - in_dim
    self.time_embed = nn.Sequential(nn.Linear(t_embed_channels, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
    self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, 6 * self.dim))
    self.trunk = DiTTrunk(self.dim, num_layers, num_blocks, self.dim // 64, mlp_ratio, layer_scale_init, checkpoint_activations)
    self.out = nn.Linear(self.dim, self.noise_dim)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)

  def adaln(self, t):
    """t: (B, L) in [0, 1] -> (B, L, 6, D), in fp32 (WT computes the time embedding outside autocast)."""
    b, l = t.shape
    with torch.autocast(device_type=t.device.type, enabled=False):
      emb = self.time_embed(sinusoidal_embedding(t.reshape(-1) * 1000.0, self.t_embed_channels))
      return self.time_proj(emb).reshape(b, l, 6, self.dim)

  def forward(self, feats, x_net, t):
    b, l, p, _ = feats.shape
    feats = (feats - feats.mean((1, 2, 3), keepdim=True)) / (feats.var((1, 2, 3), keepdim=True, unbiased=False) + 1e-6).sqrt()
    patches = rearrange(x_net, "b l c (h py) (w px) -> b l (h w) (c py px)", py=self.patch_size, px=self.patch_size)
    tokens = torch.cat([feats * self.img_token_scale, patches.to(feats.dtype)], dim=-1)
    if self.pad:
      tokens = F.pad(tokens, (0, self.pad))
    out = self.out(self.trunk(tokens, self.adaln(t)))
    grid = int(round(p ** 0.5))
    return rearrange(out, "b l (h w) (c py px) -> b l c (h py) (w px)", h=grid, w=grid, py=self.patch_size, px=self.patch_size)


class GSDiffusionModel(nn.Module):
  """The frozen image backbone + DiffusionHead, with the training loss and the sampler. Same surface as train_gs.GSModel for the code
  that renders (inputs(), forward() -> the Gaussian dict, max_sh_degree, backbone, head): forward() SAMPLES."""

  max_sh_degree = 0

  def __init__(self, cfg, stats):
    super().__init__()
    d = cfg.diffusion
    self.cfg_d = d
    if cfg.model.get("backbone") is None:
      raise SystemExit("diffusion mode needs an image backbone: model/backbone=moge_mix")
    self.num_layers = int(cfg.data.num_layers)
    self.patch_size = int(cfg.model.patch_size)
    self.backbone = hydra.utils.instantiate(cfg.model.backbone)
    self.head = DiffusionHead(
      int(cfg.model.feature_dim), self.num_layers, patch_size=self.patch_size, dim=d.get("dim"), num_blocks=int(d.num_blocks),
      img_token_scale=float(d.img_token_scale), checkpoint_activations=bool(d.get("checkpoint_activations", False)),
      layer_scale_init=float(cfg.model.transformer.layer_scale_init))
    mean, std = stats
    self.register_buffer("x0_mean", mean.clone().float())
    self.register_buffer("x0_std", std.clone().float())
    self.schedule = LayerAwareSchedule(self.num_layers, standard=dict(d.schedule.standard), plateau=dict(d.schedule.plateau))
    self.steps = int(d.steps)
    self.eval_seed = int(d.eval_seed)

  def inputs(self, batch, device):
    return batch["image"].to(device), batch["source"]["hit"].to(device)

  def _feats(self, image):
    feats = self.backbone(image)
    return feats.expand(-1, self.num_layers, -1, -1) if feats.shape[1] == 1 else feats

  def normalize(self, x0_raw):
    """(B, C, L, H, W) raw -> (B, L, C, H, W) normalized."""
    x0 = x0_raw.permute(0, 2, 1, 3, 4)
    return (x0 - self.x0_mean[None, None, :, None, None]) / self.x0_std[None, None, :, None, None]

  def training_loss(self, image, hit, x0_raw, generator=None):
    """The masked x0 loss. Returns (loss, parts): parts = per-field and per-t-bucket detached means."""
    feats = self._feats(image)
    x0 = self.normalize(x0_raw)
    b, l = x0.shape[:2]
    t = self.schedule.sample(b, phase=1, generator=generator).to(x0.device)          # (B, L)
    eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    t5 = t[:, :, None, None, None]
    a = hit[:, :, None].to(x0.dtype)                                                 # (B, L, 1, H, W): the validity mask
    x_net = a * ((1 - t5) * x0 + t5 * eps) + (1 - a) * eps                          # WT Eq. 7: invalid slots see the max noise
    x0_hat = self.head(feats, x_net, t)
    err = (x0_hat.float() - x0).square()                                            # (B, L, C, H, W)
    valid = a.sum().clamp_min(1)
    loss = (err * a).sum() / (valid * CHANNELS)
    with torch.no_grad():
      parts = {}
      for name, sl in zip(FIELDS, np.cumsum((0,) + DIMS)[:-1].tolist()):
        dim = DIMS[FIELDS.index(name)]
        parts[name] = ((err[:, :, sl:sl + dim] * a).sum() / (valid * dim)).detach()
      for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        in_bucket = a * ((t5 >= lo) & (t5 < hi)).to(a.dtype)
        parts[f"t{lo:g}-{min(hi, 1):g}"] = ((err * in_bucket).sum() / (in_bucket.sum().clamp_min(1) * CHANNELS)).detach()
    return loss, parts

  @torch.no_grad()
  def sample(self, image, hit, steps=None):
    """20 Euler steps of the flow from pure noise (a fixed seed: reproducible) -> the Gaussian dict. Invalid slots stay at the noise."""
    steps = steps or self.steps
    feats = self._feats(image)
    b, l = hit.shape[:2]
    gen = torch.Generator(device=image.device).manual_seed(self.eval_seed)
    eps = torch.randn((b, l, CHANNELS, *hit.shape[-2:]), device=image.device, generator=gen)
    a = hit[:, :, None].to(eps.dtype)
    x = eps
    ts = torch.linspace(1.0, 0.0, steps + 1, device=image.device)
    for i in range(steps):
      t = ts[i]
      x0_hat = self.head(feats, a * x + (1 - a) * eps, t.expand(b, l)).float()
      x = x - (t - ts[i + 1]) * (x - x0_hat) / t.clamp(min=T_MIN_CLAMP)
    return decode_x0(x, self.x0_mean, self.x0_std)

  def forward(self, inputs):
    image, hit = inputs
    return self.sample(image, hit)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAWeights:
  """Exponential moving average of a module's trainable parameters (WT: decay 0.9995 after warmup). The decay ramps up as
  min(decay, (1 + n) / (10 + n)) over the first updates. Lives on the parameters' device, fp32."""

  def __init__(self, module, decay=0.9995):
    self.decay = float(decay)
    self.updates = 0
    self.shadow = {n: p.detach().clone().float() for n, p in module.named_parameters() if p.requires_grad}

  @torch.no_grad()
  def update(self, module):
    self.updates += 1
    d = min(self.decay, (1 + self.updates) / (10 + self.updates))
    params = dict(module.named_parameters())
    for n, s in self.shadow.items():
      if s.device != params[n].device:   # the module moved to the GPU after the average was made
        s = self.shadow[n] = s.to(params[n].device)
      s.mul_(d).add_(params[n].detach().float(), alpha=1 - d)

  @contextlib.contextmanager
  def applied(self, module):
    """Run with the averaged weights in the module; the live weights come back afterwards."""
    params = dict(module.named_parameters())
    live = {n: params[n].detach().clone() for n in self.shadow}
    try:
      with torch.no_grad():
        for n, s in self.shadow.items():
          params[n].copy_(s.to(params[n].device, params[n].dtype))
      yield
    finally:
      with torch.no_grad():
        for n, v in live.items():
          params[n].copy_(v)

  def state_dict(self):
    return {"decay": self.decay, "updates": self.updates, "shadow": {n: s.cpu() for n, s in self.shadow.items()}}

  def load_state_dict(self, state):
    self.updates = int(state["updates"])
    missing = sorted(set(self.shadow) - set(state["shadow"]))
    unexpected = sorted(set(state["shadow"]) - set(self.shadow))
    if missing:   # they keep their copy of the weights the average was built from
      log.warning("EMA checkpoint has no average for %d parameters, they start from the live weights: %s", len(missing), missing)
    if unexpected:
      log.warning("EMA checkpoint has %d parameters the model no longer has, dropped: %s", len(unexpected), unexpected)
    for n, s in state["shadow"].items():
      if n in self.shadow:
        self.shadow[n].copy_(s.to(self.shadow[n].device))


class SpikeSkip:
  """WT's loss-spike filter: a loss above `factor` times the running average of the accepted losses is skipped (no
  optimizer step). A non-finite loss is always skipped and never enters the average. The filter only skips spikes once
  `warmup` losses have been accepted -- counted by the filter itself and checkpointed with it, so a chained resume keeps
  its average and does not restart the warmup. factor None: off, nothing is skipped (the average of the finite losses is
  still kept, for logging)."""

  def __init__(self, factor=4.0, decay=0.99, warmup=200):
    self.factor = None if factor is None else float(factor)
    self.decay = float(decay)
    self.warmup = int(warmup)
    self.average = None   # running average of the accepted losses
    self.accepted = 0     # losses folded into it
    self.nonfinite = 0    # non-finite losses seen (all skipped)

  def check(self, value):
    """True: skip this loss. False: it is accepted and folded into the average."""
    value = float(value)
    if not math.isfinite(value):
      self.nonfinite += 1
      return self.factor is not None   # off: the step goes ahead, but the average stays finite
    if self.factor is not None and self.accepted >= self.warmup and value > self.factor * self.average:
      return True
    self.average = value if self.average is None else self.decay * self.average + (1 - self.decay) * value
    self.accepted += 1
    return False

  def state_dict(self):
    return {"average": self.average, "accepted": self.accepted, "nonfinite": self.nonfinite}

  def load_state_dict(self, state):
    # factor/decay/warmup stay the config's, so a resumed run can change them
    self.average = state["average"]
    self.accepted = int(state["accepted"])
    self.nonfinite = int(state.get("nonfinite", 0))
