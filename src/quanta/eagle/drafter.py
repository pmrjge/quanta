"""EAGLE-3 drafter for Kimi-K2.6 — one transformer layer that drafts tokens from the target's
fused hidden states, MLX-native.

EAGLE-3 recipe: the target's **low / mid / high** decoder-layer hidden states are concatenated
(``[B,T,3H]``) and reduced to ``H`` (``feat_proj``); combined with the (frozen) token embedding
via ``in_proj``; run through **one** pre-norm transformer layer (standard MHA + RoPE + SwiGLU).
The hidden ``x`` then forks into two **decoupled** outputs: ``out_norm(x)`` (head space) that the
(frozen, shared) target LM head maps to next-token logits, and ``recur_proj(recur_norm(x))`` (feature
space) that is regressed onto the next target feature and **fed back** as the next step's feature —
the target's real features exist only for the first step. Keeping the recurrent feature in its own
space (not the head's) is what lets it match the reduced target feature without corrupting the logits;
training to be accurate under that self-fed stream is exactly EAGLE-3's "training-time test".

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
                 n_feature_layers: int = 3, layerscale_init: float = 1e-4) -> None:
        super().__init__()
        self.hidden, self.n_heads, self.head_dim = hidden, n_heads, head_dim
        self.rope_base = rope_base
        self.scale = head_dim ** -0.5
        self.n_feature_layers = n_feature_layers
        self.feat_proj = nn.Linear(n_feature_layers * hidden, hidden, bias=False)  # fuse RAW low/mid/high -> H
        self.embed_in = nn.Linear(hidden, hidden, bias=False)                      # project token embedding -> H
        # Balance the front-end init to a single [(n_feature_layers+1)*H -> H] projection — the linear
        # probe that reaches 9.7%. MLX's default per-matrix init is |w| <= 1/sqrt(fan_in), which makes
        # embed_in (fan_in H) ~2x hotter than feat_proj (fan_in 3H); the embedding path then dominates x
        # at init and CE alone collapses to the marginal token. Re-init BOTH with the COMBINED fan-in so
        # neither path dominates and the sum front-end matches the probe's balanced single matrix.
        k = ((n_feature_layers + 1) * hidden) ** -0.5
        self.feat_proj.weight = mx.random.uniform(-k, k, self.feat_proj.weight.shape)
        self.embed_in.weight = mx.random.uniform(-k, k, self.embed_in.weight.shape)
        self.input_norm = nn.RMSNorm(hidden, eps=eps)
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.post_norm = nn.RMSNorm(hidden, eps=eps)
        self.mlp = _SwiGLU(hidden, intermediate)
        # LayerScale: per-channel learned gains on each residual branch, init ~0 so the block starts as
        # identity (= the raw-feature front-end) and grows only where it cuts loss. A FIXED residual
        # scale (even 0.1x) instead collapses training to the majority token — the random block output
        # drowns the small useful channels of x (whose energy sits in a few massive-activation dims).
        self.ls_attn = mx.full((hidden,), layerscale_init)
        self.ls_mlp = mx.full((hidden,), layerscale_init)
        self.out_norm = nn.RMSNorm(hidden, eps=eps)              # head-space output: the frozen LM head reads this
        self.recur_norm = nn.RMSNorm(hidden, eps=eps)            # feature-space recurrent output (its own basis,
        self.recur_proj = nn.Linear(hidden, hidden, bias=False)  # decoupled from out_norm so the two don't fight)

    def reduce_target_features(self, feat3: mx.array) -> mx.array:
        """Fuse the concatenated low/mid/high target hidden states ``[B,T,3H]`` → ``[B,T,H]`` with a
        single learned projection on the **raw** features. (An earlier version RMS-normalized each
        component first; that crushed the signal — deep residual streams carry a few massive-activation
        channels 100–1000× the rest, so the per-token RMS denominator is dominated by them and the
        useful small channels collapse. A linear probe on raw feat3 reaches ~72% train top-1 vs ~5%
        through the norm; ``feat_proj`` learns whatever per-channel conditioning the fusion needs.)
        Applied once to the target features; the recurrent feature thereafter is the drafter's
        ``recur``."""
        return self.feat_proj(feat3)

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
        """One drafter pass over ``[B,T]``: ``feature`` ``[B,T,H]`` (the reduced target feature on the
        first step, the drafter's own **recurrent** feature thereafter) + frozen ``token_emb``
        ``[B,T,H]`` → ``(recur, normed)``. Two **decoupled** outputs of the same hidden ``x``:
        ``normed = out_norm(x)`` is the head-space output the (frozen, caller-applied) LM head turns
        into logits; ``recur = recur_proj(recur_norm(x))`` is the *feature-space* output, regressed in
        training onto the next reduced target feature and self-fed as the next step's ``feature`` (same
        space as that reduced feature). Keeping them separate stops the head-space CE and the
        feature-space regression from fighting over one output (an earlier bug that held accept ~3%).
        Each residual branch is gated by a learned per-channel LayerScale (``ls_attn``/``ls_mlp``, init
        ~0) so the block starts as identity over the raw-feature front-end and only adds attn/MLP where
        it helps — without it the block collapses training to the majority token regardless of how
        small a fixed residual scale is used."""
        # Front-end = SUM of two single full-rank projections (= the 9.7% linear probe's computation),
        # NOT a re-projection of the reduced feature. The old `in_proj(concat([emb, feat_proj(feat3)]))`
        # sent the feat3->x signal through a PRODUCT of two matrices (in_proj[:,H:] . feat_proj, a rank-H
        # bottleneck); a localization sweep showed that factored front-end ALONE (no block) collapses to
        # the majority token (3.5% vs the probe's 9.7%) — optimization falls into the marginal basin.
        # `feature` is already H-wide and full-rank from feat_proj (step 1) or the self-fed recur (which
        # is regressed onto feat_proj's output, same space), so it is summed directly.
        x = self.embed_in(token_emb) + feature
        x = x + self.ls_attn * self._attn(self.input_norm(x), offset, mask, cache)
        x = x + self.ls_mlp * self.mlp(self.post_norm(x))
        return self.recur_proj(self.recur_norm(x)), self.out_norm(x)
