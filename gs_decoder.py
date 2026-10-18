"""Gaussian-parameter head for train_gs.py: a per-patch linear layer (optionally
preceded by a small transformer trunk) over cached World Tracing decoder tokens (wt_features.py), plus the per-field
activations/initialization it shares with the Flash3D-style decoder this
file used to hold (ported from Flash3D's models/decoder/gaussian_decoder.py).

Gaussian positions are anchored to the input point cloud
(gs_dataset.py's `dense_unproject_camera`). By default there's no
offset/xyz output at all; with predict_mean_offset=True the head also
emits a residual camera-space offset added on top of that anchor (see
train_gs.flatten_gaussians), zero-initialized so training starts exactly
at the depth-peel point cloud.

Per-pixel, per-layer parameterization (see activate_gaussians):
    opacity  = sigmoid(raw)                     in (0, 1)
    scale    = exp(raw) * scale_lambda          a MULTIPLIER on this pixel's own real
                                                 geometric footprint (depth/fx), not an
                                                 absolute value -- see GSModel.forward
                                                 in train_gs.py. An absolute scale (Flash3D's
                                                 own parameterization) has no incentive to
                                                 stay above ~1 pixel of world-space size, and
                                                 gsplat's antialiasing eps2d floor (hard-coded
                                                 ~3px minimum projected size) silently covers
                                                 for undersized Gaussians at the exact distance
                                                 they were supervised at, revealing the gap as
                                                 soon as a render's true projected size (a
                                                 different viewing distance, etc.) exceeds that
                                                 floor.
    rotation = normalize(raw_quat, dim=channel) unit quaternion, wxyz, no sign constraint
    sh_dc    = raw                              SH band-0 color coeff, unconstrained: the rendered
                                                 color is SH_C0*sh_dc + 0.5 (degree-0 SH, as gsplat
                                                 evaluates it), clamped only at render time.
    sh_rest  = raw                              unconstrained, SH band-1+ coeffs (only if max_sh_degree>0)
    offset   = raw                              unconstrained camera-space xyz residual added to the
                                                 depth-peel position (only if predict_mean_offset)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def gaussian_field_names(max_sh_degree, predict_mean_offset=False):
  """Per-Gaussian-layer output field names, in gaussian_split_dims order."""
  names = ["opacity", "scale", "rotation", "sh_dc"]
  if max_sh_degree != 0:
    names.append("sh_rest")
  if predict_mean_offset:
    names.append("offset")
  return names


def gaussian_split_dims(max_sh_degree, predict_mean_offset=False):
  """Per-Gaussian-layer output channel groups, in this fixed order:
  opacity(1), scale(3), rotation(4), sh_dc(3)[, sh_rest(3*((deg+1)^2-1))][, offset(3)].
  offset goes last so enabling it leaves every other group's channel slice
  unchanged."""
  dims = [1, 3, 4, 3]
  if max_sh_degree != 0:
    dims.append(3 * ((max_sh_degree + 1) ** 2 - 1))
  if predict_mean_offset:
    dims.append(3)
  return dims


def gaussian_init_scales_biases(max_sh_degree, opacity_scale, opacity_bias,
                                scale_scale, scale_bias, sh_scale,
                                predict_mean_offset=False):
  """Xavier-uniform (scale, bias) per output group, in gaussian_split_dims
  order. Rotation/sh_dc init scales (1.0, 5.0) are Flash3D's own hardcoded
  literals (not exposed as config there either) -- kept identical here.
  offset's scale is None: zero-init (weight and bias), not Xavier -- see
  PatchGaussianHead.__init__."""
  scales = [opacity_scale, scale_scale, 1.0, 5.0]
  biases = [opacity_bias, float(np.log(scale_bias)), 0.0, 0.0]
  if max_sh_degree != 0:
    scales.append(sh_scale)
    biases.append(0.0)
  if predict_mean_offset:
    scales.append(None)
    biases.append(0.0)
  return scales, biases


def activate_gaussians(raw, scale_lambda):
  """raw: dict field name -> (B, L, C, H, W) raw head output (the names of
  gaussian_field_names). Returns the activated Gaussian parameters, same
  layout: opacity(1), raw_scale(3), scale(3), rotation(4), sh_dc(3)[,
  sh_rest(K)][, offset(3)]."""
  # scale_lambda folded in as an ADDITIVE log-space bias (log(exp(raw)*lambda)
  # == raw + log(lambda)) rather than a separate post-exp multiply, so
  # "raw_scale" stays log(the multiplier that "scale" actually uses) --
  # matters for a log-space comparison of raw_scale against log(gt_scale);
  # without folding lambda in here that comparison would be off by a constant
  # log(scale_lambda) offset.
  raw_scale = raw["scale"] + float(np.log(scale_lambda))

  out = {
    "opacity": torch.sigmoid(raw["opacity"]),
    "raw_scale": raw_scale,
    "scale": torch.exp(raw_scale),
    # +1e-6: guards the exact-zero raw quaternion (all channels 0, e.g. under
    # zero_init_last_conv) -- F.normalize's own eps only clamps the norm
    # denominator, so a true zero vector normalizes to itself (still zero,
    # not a unit quaternion). Negligible next to any real trained value.
    "rotation": F.normalize(raw["rotation"] + 1e-6, dim=2),
    # unconstrained color: gsplat evaluates rendered color as SH_C0*sh_dc + 0.5
    # (degree-0 SH), so the raw value is used as-is (no sigmoid bounding it
    # to (0,1)).
    "sh_dc": raw["sh_dc"],
  }
  if "sh_rest" in raw:
    out["sh_rest"] = raw["sh_rest"]
  if "offset" in raw:
    out["offset"] = raw["offset"]
  return out


def rope_cos_sin(grid, head_dim, base=100.0):
  """2D RoPE tables for a row-major grid x grid patch grid: (cos, sin), each
  (grid*grid, head_dim). The first half of head_dim rotates with the patch's
  row, the second half with its column (same scheme as World Tracing's RoPE2D:
  rotate-half within each half, frequencies base^(-2i/(head_dim/2)))."""
  half = head_dim // 2
  inv_freq = 1.0 / (base ** (torch.arange(0, half, 2).float() / half))        # (half/2,)
  ang = torch.arange(grid).float()[:, None] * inv_freq[None]                  # (grid, half/2)
  ang = torch.cat([ang, ang], dim=-1)                                         # (grid, half)
  y = torch.arange(grid).repeat_interleave(grid)                              # token p = y*grid + x
  x = torch.arange(grid).repeat(grid)
  cos = torch.cat([ang.cos()[y], ang.cos()[x]], dim=-1)
  sin = torch.cat([ang.sin()[y], ang.sin()[x]], dim=-1)
  return cos, sin


def apply_rope(t, cos, sin):
  """t: (..., S, H, head_dim); cos/sin: (S, head_dim)."""
  halves = t.unflatten(-1, (2, -1))                                           # (..., 2, half)
  a, b = halves[..., : halves.shape[-1] // 2], halves[..., halves.shape[-1] // 2 :]
  rotated = torch.cat([-b, a], dim=-1).flatten(-2)                            # rotate-half in each half
  return t * cos[:, None, :] + rotated * sin[:, None, :]


class AttentionBlock(nn.Module):
  """Pre-norm transformer block over a (B, L, P, D) token volume -- a
  stripped-down version of World Tracing's head block (DecoderBlockSA): QK-norm
  attention with LayerScale'd residuals and an MLP, no timestep AdaLN. `mode`
  picks which tokens attend to each other:
    frame:  the P patches of one layer (2D RoPE)
    ray:    the L layers at one patch position (no RoPE)
    global: all L*P tokens of the volume (2D RoPE)
  layer_scale_init is small (WT's is 1e-4), so a fresh block is close to the
  identity."""

  def __init__(self, dim, num_heads, mode, mlp_ratio=4.0, layer_scale_init=1e-4):
    super().__init__()
    if mode not in ("frame", "ray", "global"):
      raise ValueError(f"mode must be frame/ray/global, got {mode!r}")
    if dim % num_heads or (dim // num_heads) % 4:
      raise ValueError(f"dim={dim} must split into num_heads={num_heads} heads of a multiple-of-4 size (2D RoPE)")
    self.mode = mode
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.norm1 = nn.LayerNorm(dim)
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.q_norm = nn.LayerNorm(head_dim)
    self.k_norm = nn.LayerNorm(head_dim)
    self.proj = nn.Linear(dim, dim)
    self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
    self.norm2 = nn.LayerNorm(dim)
    self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
    self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))

  def _attend(self, x, cos, sin):
    """x: (N, S, D) -> (N, S, D); cos/sin: (S, head_dim) or None."""
    n, s, d = x.shape
    q, k, v = self.qkv(self.norm1(x)).reshape(n, s, 3, self.num_heads, d // self.num_heads).unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    if cos is not None:
      q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return self.proj(out.transpose(1, 2).reshape(n, s, d))

  def forward(self, x, cos, sin):
    """x: (B, L, P, D); cos/sin: the (P, head_dim) RoPE tables of one layer's patch grid."""
    b, l, p, d = x.shape
    if self.mode == "frame":
      y = self._attend(x.reshape(b * l, p, d), cos, sin).reshape(b, l, p, d)
    elif self.mode == "ray":
      y = self._attend(x.transpose(1, 2).reshape(b * p, l, d), None, None).reshape(b, p, l, d).transpose(1, 2)
    else:
      y = self._attend(x.reshape(b, l * p, d), cos.repeat(l, 1), sin.repeat(l, 1)).reshape(b, l, p, d)
    x = x + self.ls1 * y
    return x + self.ls2 * self.mlp(self.norm2(x))


class TransformerTrunk(nn.Module):
  """num_blocks AttentionBlocks cycling frame -> ray -> global (World Tracing's
  "frame_ray_global" head pattern), with its per-layer FiLM: a learned layer
  embedding -> MLP -> (gamma, beta) applied as x*(1+gamma)+beta before every
  frame block (zero-initialized, so it starts as a no-op).  (B, L, P, D) in and
  out."""

  MODES = ("frame", "ray", "global")

  def __init__(self, dim, num_layers, num_blocks, num_heads=24, mlp_ratio=4.0, layer_scale_init=1e-4):
    super().__init__()
    self.num_heads = num_heads
    self.blocks = nn.ModuleList(
      AttentionBlock(dim, num_heads, self.MODES[i % 3], mlp_ratio, layer_scale_init)
      for i in range(num_blocks)
    )
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

  def forward(self, x):
    _, num_layers, num_patches, _ = x.shape
    grid = int(round(num_patches ** 0.5))
    if grid * grid != num_patches:
      raise ValueError(f"non-square patch grid: P={num_patches}")
    cos, sin = self._rope_tables(grid, x.device)
    gamma, beta = self.film(self.layer_embed(torch.arange(num_layers, device=x.device))).chunk(2, dim=-1)  # (L, D)
    gamma, beta = gamma[None, :, None], beta[None, :, None]
    for block in self.blocks:
      if block.mode == "frame":
        x = x * (1 + gamma) + beta
      x = block(x, cos, sin)
    return x


class RGBViT(nn.Module):
  """A small ViT over the source image, trained with the head: the image is cut into
  patch_size x patch_size patches (the grid of the WT tokens), each patch's pixels are
  embedded linearly, and num_blocks frame-attention AttentionBlocks (2D RoPE) mix them.
  (B, 3, H, W) in [0, 1] -> (B, 1, P, dim) tokens, one per patch, in the WT tokens' order."""

  def __init__(self, patch_size, dim, num_blocks, num_heads=16, mlp_ratio=4.0, layer_scale_init=1e-4):
    super().__init__()
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.embed = nn.Linear(3 * patch_size ** 2, dim)
    self.blocks = nn.ModuleList(
      AttentionBlock(dim, num_heads, "frame", mlp_ratio, layer_scale_init) for _ in range(num_blocks))
    self.norm = nn.LayerNorm(dim)
    self._rope = {}

  def forward(self, image):
    p = self.patch_size
    h, w = image.shape[-2:]
    if h % p or w % p:
      raise ValueError(f"image size {h}x{w} is not a multiple of the patch size {p}")
    grid = h // p
    if grid * grid != (h // p) * (w // p):
      raise ValueError(f"non-square image {h}x{w}")
    key = (grid, image.device)
    if key not in self._rope:
      cos, sin = rope_cos_sin(grid, self.blocks[0].qkv.in_features // self.num_heads)
      self._rope[key] = (cos.to(image.device), sin.to(image.device))
    cos, sin = self._rope[key]
    x = self.embed(rearrange(image * 2 - 1, "b c (h py) (w px) -> b 1 (h w) (py px c)", py=p, px=p))
    for block in self.blocks:
      x = block(x, cos, sin)
    return self.norm(x)


class PatchGaussianHead(nn.Module):
  """(B, L, P, D) World Tracing decoder tokens (wt_features.py) -> per-pixel
  Gaussian parameters for each of the L depth-peel layers, (B, L, C, H, W) per
  field, H = W = sqrt(P) * patch_size. The tokens go through an optional
  TransformerTrunk (num_blocks > 0; WT's own head does this before its
  projection) and then one linear layer per patch token.

  The same Linear is applied to every token of every layer (WT's own
  geo_proj is shared across layers too -- the tokens already carry the
  layer via WT's FiLM). Each token's outputs are laid out as
  (channel, patch_y, patch_x), so every Gaussian field's output rows are a
  contiguous slice of the weight matrix, and the per-field Xavier init
  below slices it the same way the Flash3D decoder's 1x1 conv did. Token p
  of the P = h*w patch grid is row-major (token p = h_idx*w + w_idx), the
  same order WT's own patchify/unpatchify uses.
  """

  def __init__(self, feature_dim, num_layers, max_sh_degree, patch_size=14,
               opacity_scale=1e-3, opacity_bias=0.0, scale_scale=1e-1, scale_bias=0.02,
               sh_scale=1.0, scale_lambda=0.01, zero_init_last_conv=False,
               predict_mean_offset=False, num_blocks=0, num_heads=24, mlp_ratio=4.0,
               layer_scale_init=1e-4, layer_embed_dim=None, input_proj_dim=None, input_proj_identity_init=False):
    super().__init__()
    self.num_layers = num_layers
    # With layer_embed_dim set, the features may be a single [1, P, D] map (e.g. one
    # image encoder output): it is repeated across the num_layers Gaussian layers and
    # a learnable per-layer vector is appended to every token, so the head sees
    # D + layer_embed_dim channels. (A linear head can only use it as a per-layer bias.)
    self.layer_embed_dim = layer_embed_dim
    self.layer_embed = (
      nn.Parameter(torch.randn(num_layers, layer_embed_dim)) if layer_embed_dim else None
    )
    # With input_proj_dim set, a learned Linear(feature_dim -> input_proj_dim) is applied to the
    # features first (like WT's own final_project on the concatenated encoder blocks), so a wide
    # feature (e.g. 4 DINOv2 blocks = 4096 channels) doesn't make the trunk wider.
    self.input_proj = nn.Linear(feature_dim, input_proj_dim) if input_proj_dim else None
    if self.input_proj is not None and input_proj_identity_init:
      # the first input_proj_dim input channels pass through unchanged, the rest start at zero
      if input_proj_dim > feature_dim:
        raise ValueError(f"input_proj_identity_init needs input_proj_dim <= feature_dim, got {input_proj_dim} > {feature_dim}")
      with torch.no_grad():
        self.input_proj.weight.zero_()
        self.input_proj.weight[:, :input_proj_dim].copy_(torch.eye(input_proj_dim))
        self.input_proj.bias.zero_()
    in_dim = (input_proj_dim or feature_dim) + (layer_embed_dim or 0)
    self.max_sh_degree = max_sh_degree
    self.patch_size = patch_size
    self.scale_lambda = scale_lambda
    self.predict_mean_offset = predict_mean_offset

    self.field_names = gaussian_field_names(max_sh_degree, predict_mean_offset)
    self.field_dims = gaussian_split_dims(max_sh_degree, predict_mean_offset)
    self.channels = sum(self.field_dims)
    p2 = patch_size ** 2
    self.trunk = (
      TransformerTrunk(in_dim, num_layers, num_blocks, num_heads, mlp_ratio, layer_scale_init)
      if num_blocks > 0 else None
    )
    self.proj = nn.Linear(in_dim, self.channels * p2)

    if zero_init_last_conv:
      nn.init.zeros_(self.proj.weight)
      nn.init.zeros_(self.proj.bias)
    else:
      scales, biases = gaussian_init_scales_biases(
        max_sh_degree, opacity_scale, opacity_bias, scale_scale, scale_bias, sh_scale,
        predict_mean_offset)
      start = 0
      for dim, scale, bias in zip(self.field_dims, scales, biases):
        rows = slice(start * p2, (start + dim) * p2)
        if scale is None:
          # offset: zero-init, so every Gaussian starts exactly at its
          # depth-peel position. Gradients still reach these weights through
          # the (nonzero) input features.
          nn.init.zeros_(self.proj.weight[rows])
        else:
          nn.init.xavier_uniform_(self.proj.weight[rows], scale)
        nn.init.constant_(self.proj.bias[rows], bias)
        start += dim

  def forward(self, features):
    """features: (B, L, P, D) -> dict of (B, L, C, H, W), see
    activate_gaussians."""
    if self.input_proj is not None:
      features = self.input_proj(features)
    _, num_layers, num_patches, _ = features.shape
    if self.layer_embed is not None and num_layers == 1:
      features = features.expand(-1, self.num_layers, -1, -1)
      num_layers = self.num_layers
    if num_layers != self.num_layers:
      raise ValueError(f"features have {num_layers} layers, head was built for {self.num_layers}")
    grid = int(round(num_patches ** 0.5))
    if grid * grid != num_patches:
      raise ValueError(f"non-square patch grid: P={num_patches}")
    if self.layer_embed is not None:
      embed = self.layer_embed[None, :, None, :].expand(features.shape[0], -1, num_patches, -1)
      features = torch.cat([features, embed.to(features.dtype)], dim=-1)
    if self.trunk is not None:
      features = self.trunk(features)
    x = rearrange(
      self.proj(features), "b l (h w) (c py px) -> b l c (h py) (w px)",
      h=grid, w=grid, py=self.patch_size, px=self.patch_size,
    )  # (B, L, C, H, W)
    raw = dict(zip(self.field_names, x.split(self.field_dims, dim=2)))
    return activate_gaussians(raw, self.scale_lambda)
