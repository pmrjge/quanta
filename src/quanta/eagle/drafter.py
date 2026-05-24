"""EAGLE-3 drafter for Kimi-K2.6 — one transformer layer that drafts tokens from the target's
fused hidden states, MLX-native.

EAGLE-3 recipe: the target's **low / mid / high** decoder-layer hidden states are concatenated
(``[B,T,3H]``) and reduced to ``H`` (``feat_proj``); combined with the (frozen) token embedding
via ``in_proj``; run through **one** pre-norm transformer layer (standard MHA + RoPE + SwiGLU);
the (frozen, shared) target LM head maps the result to next-token logits. The layer's output
hidden ``x`` is the drafter's **feature** — during multi-step drafting the target's real features
exist only for the first step, so ``x`` feeds back as the next step's feature. Training the drafter
to be accurate under that self-fed feature stream is exactly EAGLE-3's "training-time test".

Design choices for Kimi: ``feat_proj`` (3H→H) is applied **once** to the target features; the
recurrent feature is always ``H``-wide. Embedding + LM head are the target's, **frozen** and held
by the caller (not module parameters), so the optimizer trains only ``feat_proj``/``in_proj``/the
one layer/the norms (~0.8B params at full size). Attention uses standard RoPE via ``mx.fast.rope``
(base matches Kimi); the drafter is its own learned model, trained on the absolute positions it
drafts at. A per-layer KV cache (``DraftCache``) makes inference drafting incremental.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DraftCache:
    """Incremental KV for the drafter's single attention layer ([B, n_heads, S, head_dim])."""

    k: mx.array | None = None
    v: mx.array | None = None

    @property
    def offset(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        self.k = k if self.k is None else mx.concatenate([self.k, k], axis=2)
        self.v = v if self.v is None else mx.concatenate([self.v, v], axis=2)
        return self.k, self.v


class _SwiGLU(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleDrafter(nn.Module):
    def __init__(self, hidden: int = 7168, n_heads: int = 56, head_dim: int = 128,
                 intermediate: int = 14336, eps: float = 1e-6, rope_base: float = 50000.0,
                 n_feature_layers: int = 3) -> None:
        super().__init__()
        self.hidden, self.n_heads, self.head_dim = hidden, n_heads, head_dim
        self.rope_base = rope_base
        self.scale = head_dim ** -0.5
        self.n_feature_layers = n_feature_layers
        self.feat_norm = nn.RMSNorm(hidden, eps=eps)  # normalize each target hidden before fusing (per component)
        self.feat_proj = nn.Linear(n_feature_layers * hidden, hidden, bias=False)  # fuse low/mid/high
        self.in_proj = nn.Linear(2 * hidden, hidden, bias=False)                   # combine [embed, feature]
        self.input_norm = nn.RMSNorm(hidden, eps=eps)
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.post_norm = nn.RMSNorm(hidden, eps=eps)
        self.mlp = _SwiGLU(hidden, intermediate)
        self.out_norm = nn.RMSNorm(hidden, eps=eps)

    def reduce_target_features(self, feat3: mx.array) -> mx.array:
        """Fuse the concatenated low/mid/high target hidden states ``[B,T,3H]`` → ``[B,T,H]``.
        Each component is RMS-normalized first (deep residual-stream magnitudes vary widely by depth,
        so raw fusion is poorly conditioned). Applied once to the target features; the recurrent
        feature thereafter is the layer's ``x``."""
        b, t, _ = feat3.shape
        f = self.feat_norm(feat3.reshape(b, t, self.n_feature_layers, self.hidden))  # norm over H per component
        return self.feat_proj(f.reshape(b, t, self.n_feature_layers * self.hidden))

    def _attn(self, x: mx.array, offset: int, mask, cache: DraftCache | None) -> mx.array:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, t, self.n_heads * self.head_dim))

    def step(self, feature: mx.array, token_emb: mx.array, *, offset: int = 0,
             mask="causal", cache: DraftCache | None = None) -> tuple[mx.array, mx.array]:
        """One drafter pass over ``[B,T]``: feature ``[B,T,H]`` (reduced target feature, or the
        drafter's own ``x`` on later steps) + frozen ``token_emb`` ``[B,T,H]`` → ``(x, normed)``.
        ``x`` is the recurrent feature for the next step; ``normed = out_norm(x)`` feeds the (frozen,
        caller-applied) LM head for logits."""
        x = self.in_proj(mx.concatenate([token_emb, feature], axis=-1))
        x = x + self._attn(self.input_norm(x), offset, mask, cache)
        x = x + self.mlp(self.post_norm(x))
        return x, self.out_norm(x)
