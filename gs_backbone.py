"""Trainable image encoder in front of the Gaussian head (train_gs.py, model.backbone).

MoGeBackbone is the DINOv2 ViT-L/14 of the pretrained MoGe-1 (`Ruicheng/moge-vitl`), the
same weights and preprocessing as extractors.MoGeExtractor(weights=...), but with gradients:
the render goes in, the last block's [B, 1, P, 1024] tokens come out, and the whole
backbone is trained together with the head. Weights stay fp32; the forward runs under bf16
autocast and, in training, checkpoints each block so the activations of 24 blocks don't pile up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MoGeBackbone(nn.Module):
  PATCH_SIZE = 14

  def __init__(self, weights="Ruicheng/moge-vitl", image_size=504, checkpoint_activations=True):
    super().__init__()
    import os

    from wt._core.vendor.moge.moge_model import MoGeModel
    from wt._core.vendor.moge.utils.wrap_module import wrap_dinov2_attention_with_sdpa

    path = weights
    if not os.path.exists(path):
      from huggingface_hub import hf_hub_download

      path = hf_hub_download(repo_id=weights, filename="model.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    moge = MoGeModel(**ckpt["model_config"])
    moge.load_state_dict(ckpt["model"])
    self.backbone = moge.backbone
    for block in self.backbone.blocks:
      block.attn = wrap_dinov2_attention_with_sdpa(block.attn)
    self.register_buffer("image_mean", moge.image_mean.clone())
    self.register_buffer("image_std", moge.image_std.clone())
    del moge, ckpt
    self.image_size = int(image_size)
    if self.image_size % self.PATCH_SIZE:
      raise ValueError(f"image_size={image_size} is not a multiple of the patch size {self.PATCH_SIZE}")
    self.checkpoint_activations = bool(checkpoint_activations)
    self.out_dim = self.backbone.blocks[0].attn.qkv.in_features

  def forward(self, image):
    """image: (B, 3, S, S) RGB in [0, 1] -> (B, 1, P, out_dim) tokens, P = (S / 14)^2."""
    bb = self.backbone
    x = (image - self.image_mean) / self.image_std
    x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
      x = bb.prepare_tokens_with_masks(x)
      for block in bb.blocks:
        x = checkpoint(block, x, use_reentrant=False) if self.training and self.checkpoint_activations else block(x)
      x = bb.norm(x)[:, 1 + bb.num_register_tokens:]
    return x.float()[:, None]
