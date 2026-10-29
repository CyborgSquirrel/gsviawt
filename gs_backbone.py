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
  """Frozen MoGe-1 DINOv2 whose tokens are a learned per-channel mix of ALL its blocks: the output is
  sum_k mix[k] * norm(block_k output), mix (blocks, 1024) trained with the head. The backbone runs without
  autograd, so it costs no gradient or optimizer memory; each block's output goes through the backbone's
  final LayerNorm (as get_intermediate_layers(norm=True) does) so their scales are comparable.

  init: "last" (1 on the last block, 0 elsewhere = MoGeBackbone's output) or "gaussian" (N(0, 1) / blocks).
  groups: how the Gaussian layers share the mix --
    "shared":    one mix for all (mix is (blocks, 1024)); (B, 3, S, S) -> (B, 1, P, 1024), the head expands
                 the one token map to its layers;
    "first_rest": one mix for layer 0 and one shared by layers 1.., (2, blocks, 1024);
    "per_layer": one mix per layer, (num_layers, blocks, 1024);
  the last two give (B, num_layers, P, 1024), each layer its own token map."""

  GROUPS = ("shared", "first_rest", "per_layer")

  def __init__(self, weights="Ruicheng/moge-vitl", image_size=504, groups="shared", init="last", num_layers=6):
    super().__init__(weights, image_size, checkpoint_activations=False, freeze_blocks=0)
    if groups not in self.GROUPS:
      raise ValueError(f"mix groups must be one of {self.GROUPS}, got {groups!r}")
    if init not in ("last", "gaussian"):
      raise ValueError(f"mix init must be 'last' or 'gaussian', got {init!r}")
    for p in self.backbone.parameters():
      p.requires_grad_(False)   # the whole encoder is frozen; only mix trains
    n = len(self.backbone.blocks)
    num_sets = {"shared": 1, "first_rest": 2, "per_layer": num_layers}[groups]
    shape = (n, self.out_dim) if groups == "shared" else (num_sets, n, self.out_dim)
    if init == "last":
      mix = torch.zeros(shape)
      mix[..., -1, :] = 1.0
    else:
      mix = torch.randn(shape) / n
    self.mix = nn.Parameter(mix)
    # which mix each Gaussian layer reads (None: the one token map is shared)
    self.layer_sets = {"shared": None, "first_rest": [0] + [1] * (num_layers - 1), "per_layer": list(range(num_layers))}[groups]

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
    mix = self.mix if self.mix.dim() == 3 else self.mix[None]   # (sets, blocks, D)
    tokens = torch.einsum("bkpd,gkd->bgpd", outs.float(), mix)    # (B, sets, P, D)
    return tokens if self.layer_sets is None else tokens[:, self.layer_sets]


class WTMixBackbone(nn.Module):
  """Frozen World Tracing decoder (r75b: 42 blocks, width 1536) whose tokens are a learned per-channel mix of ALL its
  decoder blocks: out[l] = sum_k mix[set(l), k] * LN(block_k output[l]), mix (blocks, 1536) trained with the head.
  WT's tokens are per depth layer ((B, num_layers, P, 1536), P = (S / 14)^2), so every layer reads its own token map.

  Input regime (no geometry at inference): WT is run at flow time t=1 on pure Gaussian noise plus the RGB image, so its
  tokens depend on the image alone. The noise is fresh on every training forward; in eval it comes from a fixed seed so
  validation is reproducible. Only the decoder runs (WT's head is skipped). Each block's output goes through a
  parameter-free per-token LayerNorm before mixing (WT has no final norm of its own; the stream norms are 100-1000).

  groups / init: as MoGeMixBackbone -- shared (one mix for all layers), first_rest (one for layer 0, one for layers 1..),
  per_layer; init "last" (1 on the last block) or "gaussian" (N(0, 1) / blocks).
  The WT weights are not registered as parameters, so checkpoints hold only the mix; they are loaded again from
  `ckpt` and kept in bf16 (bf16_weights=True) on the input's device at the first forward."""

  GROUPS = ("shared", "first_rest", "per_layer")
  PATCH_SIZE = 14

  def __init__(self, config="r75b", ckpt="r75b", image_size=504, groups="shared", init="last", num_layers=6,
               bf16_weights=True, eval_seed=0):
    super().__init__()
    if groups not in self.GROUPS:
      raise ValueError(f"mix groups must be one of {self.GROUPS}, got {groups!r}")
    if init not in ("last", "gaussian"):
      raise ValueError(f"mix init must be 'last' or 'gaussian', got {init!r}")
    from wt.checkpoint import build_model_and_load_ckpt
    from wt.inference import _AC_ATTR, _bypass_activation_checkpointing

    model, wt_cfg = build_model_and_load_ckpt(config, ckpt, torch.device("cpu"))
    if int(image_size) != wt_cfg["image_size"] or int(num_layers) != wt_cfg["model_kwargs"]["num_layers"]:
      raise ValueError(f"{config} runs at image_size={wt_cfg['image_size']}, num_layers={wt_cfg['model_kwargs']['num_layers']}; "
                       f"got image_size={image_size}, num_layers={num_layers}")
    if bf16_weights:
      model = model.to(torch.bfloat16)
    model.eval().requires_grad_(False)
    self._wt = [model]   # a list: keeps the 1.7B frozen weights out of this module's parameters and state_dict
    self._bypass = _bypass_activation_checkpointing
    # the AC wrappers are unwrapped during the forward, so the hooks sit on the inner modules
    self._blocks = [getattr(b, _AC_ATTR, None) or b for b in model.net.decoder_blocks]
    self.image_size, self.num_layers, self.eval_seed = int(image_size), int(num_layers), int(eval_seed)
    self.patch_grid = self.image_size // self.PATCH_SIZE
    self.out_dim = int(wt_cfg["model_kwargs"]["decoder_embed_dim"])
    n = len(self._blocks)
    num_sets = {"shared": 1, "first_rest": 2, "per_layer": num_layers}[groups]
    shape = (n, self.out_dim) if groups == "shared" else (num_sets, n, self.out_dim)
    if init == "last":
      mix = torch.zeros(shape)
      mix[..., -1, :] = 1.0
    else:
      mix = torch.randn(shape) / n
    self.mix = nn.Parameter(mix)
    self.layer_sets = {"shared": [0] * num_layers, "first_rest": [0] + [1] * (num_layers - 1),
                       "per_layer": list(range(num_layers))}[groups]

  def train(self, mode=True):
    super().train(mode)
    self._wt[0].eval()
    return self

  def _tokens(self, image):
    """-> (B, blocks, L, P, D) LayerNormed block outputs, bf16, no grad."""
    wt = self._wt[0]
    if next(wt.parameters()).device != image.device:
      wt.to(image.device)
    B, _, S, _ = image.shape
    L, P, D, n = self.num_layers, self.patch_grid ** 2, self.out_dim, len(self._blocks)
    gen = None
    if not self.training:
      gen = torch.Generator(device=image.device).manual_seed(self.eval_seed)
    x_t = torch.randn(B, L * S * S, 3, device=image.device, generator=gen)   # pure noise: t = 1
    outs = torch.empty(B, n, L, P, D, device=image.device, dtype=torch.bfloat16)

    def hook(i):
      kind = i % 3   # frame / ray / global, see wt TransformerTrunk pattern: tokens (B*L, P, D) / (B*P, L, D) / (B, L*P, D)
      def f(_, __, out):
        out = out[0] if isinstance(out, tuple) else out
        out = out.reshape(B, P, L, D).transpose(1, 2) if kind == 1 else out.reshape(B, L, P, D)
        outs[:, i] = F.layer_norm(out.float(), (D,), eps=1e-6).to(outs.dtype)
      return f

    handles = [b.register_forward_hook(hook(i)) for i, b in enumerate(self._blocks)]
    cond = {"rgb": image[:, None].repeat(1, L, 1, 1, 1), "noise_height": S, "noise_width": S, "noise_nview": L,
            "batch_size": B, "valid_mask": torch.ones(B, L * S * S, 1, device=image.device),
            "invalid_fill_mode": "noise", "_context_only": True}
    try:
      with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=image.is_cuda), self._bypass(wt):
        wt(x_t, torch.ones(B, device=image.device), cond)
    finally:
      for h in handles:
        h.remove()
    return outs

  def forward(self, image):
    """image: (B, 3, S, S) RGB in [0, 1] (black background, as MoGeExtractor/WTExtractor) -> (B, L, P, 1536)."""
    outs = self._tokens(image)
    mix = self.mix if self.mix.dim() == 3 else self.mix[None]   # (sets, blocks, D)
    mix = mix[self.layer_sets]                                   # (L, blocks, D)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=image.is_cuda):
      tokens = torch.einsum("bklpd,lkd->blpd", outs, mix)
    return tokens.float()
