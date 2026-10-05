"""Gaussian-parameter head for train_gs.py: a per-patch linear layer over
cached World Tracing decoder tokens (wt_features.py), plus the per-field
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
  PatchLinearGaussianHead.__init__."""
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


class PatchLinearGaussianHead(nn.Module):
  """One linear layer per patch token: (B, L, P, D) World Tracing decoder
  tokens (wt_features.py) -> per-pixel Gaussian parameters for each of the L
  depth-peel layers, (B, L, C, H, W) per field, H = W = sqrt(P) * patch_size.

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
               predict_mean_offset=False):
    super().__init__()
    self.num_layers = num_layers
    self.max_sh_degree = max_sh_degree
    self.patch_size = patch_size
    self.scale_lambda = scale_lambda
    self.predict_mean_offset = predict_mean_offset

    self.field_names = gaussian_field_names(max_sh_degree, predict_mean_offset)
    self.field_dims = gaussian_split_dims(max_sh_degree, predict_mean_offset)
    self.channels = sum(self.field_dims)
    p2 = patch_size ** 2
    self.proj = nn.Linear(feature_dim, self.channels * p2)

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
    _, num_layers, num_patches, _ = features.shape
    if num_layers != self.num_layers:
      raise ValueError(f"features have {num_layers} layers, head was built for {self.num_layers}")
    grid = int(round(num_patches ** 0.5))
    if grid * grid != num_patches:
      raise ValueError(f"non-square patch grid: P={num_patches}")
    x = rearrange(
      self.proj(features), "b l (h w) (c py px) -> b l c (h py) (w px)",
      h=grid, w=grid, py=self.patch_size, px=self.patch_size,
    )  # (B, L, C, H, W)
    raw = dict(zip(self.field_names, x.split(self.field_dims, dim=2)))
    return activate_gaussians(raw, self.scale_lambda)
