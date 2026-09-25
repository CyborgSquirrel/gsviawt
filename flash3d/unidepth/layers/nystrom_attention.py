from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Unguarded upstream, unlike every other xformers import in this vendored
# unidepth/ tree (see models/backbones/metadinov2/*.py). Confirmed against a
# scratch install: current xformers (0.0.35, the only build with wheels for
# torch 2.14/cu13) has dropped the `xformers.components` "attention
# mechanism" API this file was written against entirely -- there's no
# version of xformers that has both NystromAttention *and* runs on this
# torch/CUDA combination. Fall back to exact attention instead of chasing
# that; NystromAttention itself holds no learned parameters (just wraps the
# same q/k/v/out projections AttentionBlock already defines in a cheaper,
# approximate attention algorithm), so nothing in the checkpoint depends on
# it and exact softmax attention is a strict accuracy improvement over the
# Nystrom approximation it stands in for.
try:
    from xformers.components.attention import NystromAttention
except ImportError:
    NystromAttention = None

from .attention import AttentionBlock


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        self.attention_fn = (
            NystromAttention(num_landmarks=128, num_heads=num_heads, dropout=dropout)
            if NystromAttention is not None
            else None
        )

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        if self.attention_fn is None:
            return super().attn(
                x, attn_bias=attn_bias, context=context, pos_embed=pos_embed,
                pos_embed_context=pos_embed_context, rope=rope,
            )

        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x
