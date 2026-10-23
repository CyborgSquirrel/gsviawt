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
  trains_encoder = True   # train_gs.py: the encoder's activations compete with the renderer's for GPU memory

  def __init__(self, weights="Ruicheng/moge-vitl", image_size=504, checkpoint_activations=True, freeze_blocks=0):
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
    self.backbone = moge.backbone.float()   # the checkpoint is fp16; Adam needs fp32 weights (its eps underflows in fp16 -> NaN)
    for block in self.backbone.blocks:
      block.attn = wrap_dinov2_attention_with_sdpa(block.attn)
    self.register_buffer("image_mean", moge.image_mean.clone().float())
    self.register_buffer("image_std", moge.image_std.clone().float())
    del moge, ckpt
    self.image_size = int(image_size)
    if self.image_size % self.PATCH_SIZE:
      raise ValueError(f"image_size={image_size} is not a multiple of the patch size {self.PATCH_SIZE}")
    self.checkpoint_activations = bool(checkpoint_activations)
    self.out_dim = self.backbone.blocks[0].attn.qkv.in_features
    # freeze_blocks > 0: the patch embedding and the first freeze_blocks blocks are frozen (they
    # run without autograd, so they cost no gradient / optimizer / activation memory)
    self.freeze_blocks = int(freeze_blocks)
    if not 0 <= self.freeze_blocks <= len(self.backbone.blocks):
      raise ValueError(f"freeze_blocks={freeze_blocks} out of range for {len(self.backbone.blocks)} blocks")
    if self.freeze_blocks:
      frozen = [self.backbone.cls_token, self.backbone.pos_embed, *self.backbone.patch_embed.parameters()]
      for block in self.backbone.blocks[: self.freeze_blocks]:
        frozen += list(block.parameters())
      for p in frozen:
        p.requires_grad_(False)

  def train(self, mode=True):
    super().train(mode)
    return self

  def forward(self, image):
    """image: (B, 3, S, S) RGB in [0, 1] -> (B, 1, P, out_dim) tokens, P = (S / 14)^2."""
    bb = self.backbone
    x = (image - self.image_mean) / self.image_std
    x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
      with torch.set_grad_enabled(torch.is_grad_enabled() and self.freeze_blocks == 0):
        x = bb.prepare_tokens_with_masks(x)
        for block in bb.blocks[: self.freeze_blocks]:
          x = block(x)
      for block in bb.blocks[self.freeze_blocks:]:
        x = checkpoint(block, x, use_reentrant=False) if self.training and self.checkpoint_activations else block(x)
      x = bb.norm(x)[:, 1 + bb.num_register_tokens:]
    return x.float()[:, None]


class MoGeMixBackbone(MoGeBackbone):
  trains_encoder = False

  """Frozen MoGe-1 DINOv2 whose tokens are a learned per-channel mix of ALL its blocks: the output is
  sum_k mix[k] * norm(block_k output), mix (blocks, 1024) trained with the head (init: 1 on the last
  block, 0 elsewhere = MoGeBackbone's output). The backbone runs without autograd, so it costs no
  gradient or optimizer memory; each block's output goes through the backbone's final LayerNorm (as
  get_intermediate_layers(norm=True) does) so their scales are comparable. (B, 3, S, S) -> (B, 1, P, 1024)."""

  def __init__(self, weights="Ruicheng/moge-vitl", image_size=504):
    super().__init__(weights, image_size, checkpoint_activations=False, freeze_blocks=0)
    for p in self.backbone.parameters():
      p.requires_grad_(False)   # the whole encoder is frozen; only mix trains
    n = len(self.backbone.blocks)
    mix = torch.zeros(n, self.out_dim)
    mix[-1] = 1.0
    self.mix = nn.Parameter(mix)

  def forward(self, image):
    bb = self.backbone
    x = (image - self.image_mean) / self.image_std
    x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda), torch.no_grad():
      x = bb.prepare_tokens_with_masks(x)
      outs = []
      for block in bb.blocks:
        x = block(x)
        outs.append(bb.norm(x)[:, 1 + bb.num_register_tokens:])
      outs = torch.stack(outs, 1)   # (B, blocks, P, D)
    return (outs.float() * self.mix[None, :, None, :]).sum(1)[:, None]


class RGBPatchBackbone(nn.Module):
  """No encoder: each 14x14 patch's own RGB pixels (in [-1, 1], (py px c) order, as model.rgb_patch) are the token,
  (B, 3, S, S) -> (B, 1, P, 3 * patch_size^2). Nothing to train; the head's trunk gets 588-d tokens
  (model.feature_dim=588, no input projection) and tells the 6 layers apart by its FiLM."""

  trains_encoder = False

  def __init__(self, patch_size=14):
    super().__init__()
    self.patch_size = int(patch_size)
    self.out_dim = 3 * self.patch_size**2

  def forward(self, image):
    from einops import rearrange

    return rearrange(image * 2 - 1, "b c (h py) (w px) -> b 1 (h w) (py px c)", py=self.patch_size, px=self.patch_size)
