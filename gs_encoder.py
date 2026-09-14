"""ResNet encoder for train_gs.py, ported from Flash3D's
models/encoder/resnet_encoder.py (~/projects/flash3d).

Flash3D's encoder takes RGB(+depth) and normalizes the whole input in its
own forward(). Ours takes RGB concatenated with `num_layers` blocks of
(xyz, validity) channels from the input point cloud (see gs_dataset.py), so
normalization is channel-group-specific and happens in the caller
(train_gs.py) before this module ever sees the tensor.
"""

import numpy as np
import torch.nn as nn
import torchvision.models as tv_models

RESNETS = {
  18: (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
  50: (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V2),
}


class GSResnetEncoder(nn.Module):
  """ImageNet-pretrained ResNet trunk with a rebuilt first conv layer to
  accept `in_channels` (3 + 4*num_layers: RGB + per-layer xyz + validity).

  `bn_order`: "pre_bn" (default, matches Flash3D) captures the first skip
  feature before BatchNorm+ReLU, preserving colour/scale information for the
  decoder to recover; "monodepth" captures it after.
  """

  def __init__(self, num_layers=50, pretrained=True, bn_order="pre_bn", in_channels=3):
    super().__init__()
    if num_layers not in RESNETS:
      raise ValueError(f"{num_layers} is not a valid number of resnet layers")
    self.bn_order = bn_order
    self.num_ch_enc = np.array([64, 64, 128, 256, 512])
    if num_layers > 34:
      self.num_ch_enc[1:] *= 4

    model_ctor, weights = RESNETS[num_layers]
    self.encoder = model_ctor(weights=weights if pretrained else None)

    if in_channels != 3:
      old_conv1 = self.encoder.conv1
      self.encoder.conv1 = nn.Conv2d(
        in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride, padding=old_conv1.padding, bias=False,
      )
      nn.init.kaiming_normal_(self.encoder.conv1.weight, mode="fan_out", nonlinearity="relu")

  def forward(self, x):
    """x: (B, in_channels, H, W), already normalized per-channel-group by
    the caller. Returns the 5 Flash3D-style skip-connection feature maps
    (finest first, coarsest last)."""
    encoder = self.encoder
    features = []
    x = encoder.conv1(x)

    if self.bn_order == "pre_bn":
      features.append(x)
      x = encoder.bn1(x)
      x = encoder.relu(x)
    elif self.bn_order == "monodepth":
      x = encoder.bn1(x)
      x = encoder.relu(x)
      features.append(x)
    else:
      raise ValueError(f"unknown bn_order {self.bn_order!r}")

    features.append(encoder.layer1(encoder.maxpool(x)))
    features.append(encoder.layer2(features[-1]))
    features.append(encoder.layer3(features[-1]))
    features.append(encoder.layer4(features[-1]))
    return features
