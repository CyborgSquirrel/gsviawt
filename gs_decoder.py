"""Gaussian-parameter U-Net decoder for train_gs.py, ported from Flash3D's
models/decoder/{resnet_decoder,gaussian_decoder}.py (~/projects/flash3d).

Differences from Flash3D:
  - No offset/xyz output. Gaussian positions come straight from the input
    point cloud (gs_dataset.py's `dense_unproject_camera`), never predicted
    or refined here.
  - `num_layers` is a free hyperparameter (Flash3D hardcodes 2).

Per-pixel, per-layer parameterization (this is the "parameterization" the
plan calls out explicitly):
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
    sh_dc    = raw                              unconstrained, SH band-0 color coeff
    sh_rest  = raw                              unconstrained, SH band-1+ coeffs (only if max_sh_degree>0)
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def upsample(x, mode="nearest"):
  return F.interpolate(x, scale_factor=2, mode=mode)


class Conv3x3(nn.Module):
  def __init__(self, in_channels, out_channels, use_refl=True):
    super().__init__()
    self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
    self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

  def forward(self, x):
    return self.conv(self.pad(x))


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = Conv3x3(in_channels, out_channels)
    self.nonlin = nn.ELU(inplace=True)

  def forward(self, x):
    return self.nonlin(self.conv(x))


def gaussian_split_dims(max_sh_degree):
  """Per-Gaussian-layer output channel groups, in this fixed order:
  opacity(1), scale(3), rotation(4), sh_dc(3)[, sh_rest(3*((deg+1)^2-1))]."""
  dims = [1, 3, 4, 3]
  if max_sh_degree != 0:
    dims.append(3 * ((max_sh_degree + 1) ** 2 - 1))
  return dims


def gaussian_init_scales_biases(max_sh_degree, opacity_scale, opacity_bias,
                                scale_scale, scale_bias, sh_scale):
  """Xavier-uniform (scale, bias) per output group, in gaussian_split_dims
  order. Rotation/sh_dc init scales (1.0, 5.0) are Flash3D's own hardcoded
  literals (not exposed as config there either) -- kept identical here."""
  scales = [opacity_scale, scale_scale, 1.0, 5.0]
  biases = [opacity_bias, float(np.log(scale_bias)), 0.0, 0.0]
  if max_sh_degree != 0:
    scales.append(sh_scale)
    biases.append(0.0)
  return scales, biases


class GaussianResnetDecoder(nn.Module):
  """Flash3D-style 5-level ResNet U-Net decoder + Gaussian parameter head,
  emitting `num_layers` Gaussian layers at once via channel-slicing
  (Flash3D's `one_gauss_decoder=True` mode, generalized to `num_layers`
  layers instead of a hardcoded 2). For `one_gauss_decoder=False` (default),
  wrap `num_layers` instances of this class with `num_layers=1` each in
  `GSDecoderStack` below instead.
  """

  def __init__(self, num_ch_enc, num_layers, max_sh_degree,
              num_ch_dec=(32, 32, 64, 128, 256), upsample_mode="nearest",
              use_skips=True, opacity_scale=1e-3, opacity_bias=0.0,
              scale_scale=1e-1, scale_bias=0.02, sh_scale=1.0, scale_lambda=0.01):
    super().__init__()
    self.use_skips = use_skips
    self.upsample_mode = upsample_mode
    self.num_ch_enc = num_ch_enc
    self.num_ch_dec = np.array(num_ch_dec)
    self.max_sh_degree = max_sh_degree
    self.num_layers = num_layers
    self.scale_lambda = scale_lambda

    per_layer_dims = gaussian_split_dims(max_sh_degree)
    per_layer_scales, per_layer_biases = gaussian_init_scales_biases(
      max_sh_degree, opacity_scale, opacity_bias, scale_scale, scale_bias, sh_scale)
    self.split_dimensions = per_layer_dims * num_layers
    scale_inits = per_layer_scales * num_layers
    bias_inits = per_layer_biases * num_layers
    self.num_output_channels = sum(self.split_dimensions)

    convs = OrderedDict()
    top = len(self.num_ch_dec) - 1
    for i in range(top, -1, -1):
      num_ch_in = self.num_ch_enc[-1] if i == top else self.num_ch_dec[i + 1]
      num_ch_out = self.num_ch_dec[i]
      convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

      num_ch_in = self.num_ch_dec[i]
      if self.use_skips and i > 0:
        num_ch_in += self.num_ch_enc[i - 1]
      num_ch_out = self.num_ch_dec[i]
      convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
    self.convs = convs
    self.decoder = nn.ModuleList(list(convs.values()))
    self.out = nn.Conv2d(int(self.num_ch_dec[0]), self.num_output_channels, 1)

    start = 0
    for out_channels, scale, bias in zip(self.split_dimensions, scale_inits, bias_inits):
      nn.init.xavier_uniform_(self.out.weight[start:start + out_channels], scale)
      nn.init.constant_(self.out.bias[start:start + out_channels], bias)
      start += out_channels

  def forward(self, input_features):
    """input_features: 5 encoder feature maps, finest first / coarsest last
    (GSResnetEncoder's output). Returns a dict of (B, num_layers, C, H, W):
    opacity(1), scale(3), rotation(4), sh_dc(3)[, sh_rest(K)]."""
    x = input_features[-1]
    for i in range(len(self.num_ch_dec) - 1, -1, -1):
      x = self.convs[("upconv", i, 0)](x)
      x = upsample(x, mode=self.upsample_mode)
      if self.use_skips and i > 0:
        x = torch.cat([x, input_features[i - 1]], dim=1)
      x = self.convs[("upconv", i, 1)](x)
    x = self.out(x)  # (B, num_layers * per_layer_dims, H, W)

    per_layer_dims = gaussian_split_dims(self.max_sh_degree)
    n_fields = len(per_layer_dims)
    parts = x.split(per_layer_dims * self.num_layers, dim=1)

    field_names = ["opacity", "scale", "rotation", "sh_dc"]
    if self.max_sh_degree != 0:
      field_names.append("sh_rest")
    per_field = {name: [] for name in field_names}
    for l in range(self.num_layers):
      layer_parts = parts[l * n_fields:(l + 1) * n_fields]
      for name, raw in zip(field_names, layer_parts):
        per_field[name].append(raw)

    def stack_layers(tensors):
      # tensors: num_layers entries, each (B,C,H,W) -> (B,L,C,H,W)
      return rearrange(tensors, "l b c h w -> b l c h w")

    # scale_lambda folded in as an ADDITIVE log-space bias (log(exp(raw)*lambda)
    # == raw + log(lambda)) rather than a separate post-exp multiply, so
    # "raw_scale" stays log(the multiplier that "scale" actually uses) --
    # matters for compute_direct_loss's direct_scale_weight term, which
    # compares raw_scale directly against log(gt_scale) in log-space; without
    # folding lambda in here that comparison would be off by a constant
    # log(scale_lambda) offset.
    raw_scale = stack_layers(per_field["scale"]) + float(np.log(self.scale_lambda))

    out = {
      "opacity": torch.sigmoid(stack_layers(per_field["opacity"])),
      "raw_scale": raw_scale,
      "scale": torch.exp(raw_scale),
      "rotation": F.normalize(stack_layers(per_field["rotation"]), dim=2),
      "sh_dc": stack_layers(per_field["sh_dc"]),
    }
    if self.max_sh_degree != 0:
      out["sh_rest"] = stack_layers(per_field["sh_rest"])
    return out


class GSDecoderStack(nn.Module):
  """`num_layers` independent `GaussianResnetDecoder(num_layers=1, ...)`
  heads sharing the same encoder trunk -- Flash3D's `one_gauss_decoder=False`
  default (each layer specializes with its own decoder weights). Output
  shape matches `GaussianResnetDecoder(num_layers=L)` exactly, so callers
  don't need to know which mode is in use.
  """

  def __init__(self, num_ch_enc, num_layers, max_sh_degree, **decoder_kwargs):
    super().__init__()
    self.heads = nn.ModuleList([
      GaussianResnetDecoder(num_ch_enc, num_layers=1, max_sh_degree=max_sh_degree, **decoder_kwargs)
      for _ in range(num_layers)
    ])

  def forward(self, input_features, active_layers=None):
    """active_layers: optional iterable of layer indices to actually run
    through their own decoder head -- each head is a FULL independent 5-level
    U-Net (~9M params here, 68% of this model's total is spread across the 6
    heads), so skipping inactive ones is a real forward+backward compute/VRAM
    saving, not just cosmetic. Skipped layers get an all-zero placeholder
    (torch.zeros_like off an actually-computed layer's own output -- same
    shape/dtype/device, no grad_fn, so it costs nothing in the backward pass
    either) instead of running their head at all. None (default): every
    layer runs, identical to the pre-active_layers behavior."""
    indices = range(len(self.heads)) if active_layers is None else sorted(set(active_layers))
    computed = {i: self.heads[i](input_features) for i in indices}
    ref = next(iter(computed.values()))
    return {
      k: torch.cat([computed[i][k] if i in computed else torch.zeros_like(ref[k]) for i in range(len(self.heads))], dim=1)
      for k in ref
    }
