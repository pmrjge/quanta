"""YaRN RoPE for Kimi-K2.6 (DeepSeek-V3 convention), MLX-native.

Mirrors ``DeepseekV3YarnRotaryEmbedding`` / ``apply_rotary_pos_emb`` from the
source ``modeling_deepseek.py``. Two application paths:

* :func:`apply_rope_explicit` — the naive de-interleave + rotate-half path that
  matches HF layout exactly (default; used to prove parity).
* :func:`apply_rope_fast` — ``mx.fast.rope(traditional=True, freqs=...)``. This
  rotates consecutive pairs (interleaved layout), which differs *in layout* from
  the explicit path but is identical inside the q·k dot product (the same
  permutation hits q and k), so attention output is unchanged. Opt-in until proven.
"""

from __future__ import annotations

import math

import mlx.core as mx

from quanta.config import KimiTextConfig


def _yarn_correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, max_pos: int
) -> tuple[int, int]:
    low = math.floor(_yarn_correction_dim(beta_fast, dim, base, max_pos))
    high = math.ceil(_yarn_correction_dim(beta_slow, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_yarn_inv_freq(cfg: KimiTextConfig) -> mx.array:
    """The YaRN per-pair inverse frequencies, shape ``[qk_rope_head_dim // 2]`` (fp32)."""
    dim = cfg.qk_rope_head_dim
    base = cfg.rope.rope_theta
    factor = cfg.rope.factor

    idx = mx.arange(0, dim, 2, dtype=mx.float32)
    freq_extra = 1.0 / (base ** (idx / dim))
    freq_inter = 1.0 / (factor * (base ** (idx / dim)))

    low, high = _yarn_correction_range(
        cfg.rope.beta_fast, cfg.rope.beta_slow, dim, base, cfg.rope.original_max_position_embeddings
    )
    lo, hi = float(low), float(high)
    if lo == hi:
        hi += 0.001  # prevent singularity (HF yarn_linear_ramp_mask)
    ramp = mx.clip((mx.arange(dim // 2, dtype=mx.float32) - lo) / (hi - lo), 0.0, 1.0)
    inv_freq_mask = 1.0 - ramp
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask
    return inv_freq


def yarn_cos_sin(cfg: KimiTextConfig, positions: mx.array) -> tuple[mx.array, mx.array]:
    """Cos/sin tables for the given positions, shape ``[T, qk_rope_head_dim]`` (fp32)."""
    inv_freq = build_yarn_inv_freq(cfg)
    t = positions.astype(mx.float32)
    freqs = t[:, None] * inv_freq[None, :]
    mult = yarn_get_mscale(cfg.rope.factor, cfg.rope.mscale) / yarn_get_mscale(
        cfg.rope.factor, cfg.rope.mscale_all_dim
    )
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb) * mult, mx.sin(emb) * mult


def attention_softmax_scale(cfg: KimiTextConfig) -> float:
    """``q_head_dim**-0.5 * mscale_all_dim**2`` — the YaRN-corrected attention scale."""
    scale = cfg.q_head_dim ** -0.5
    mscale = yarn_get_mscale(cfg.rope.factor, cfg.rope.mscale_all_dim)
    return scale * mscale * mscale


def _rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rope_explicit(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """HF-exact RoPE: de-interleave consecutive pairs, then rotate-half.

    ``x``: ``[B, H, T, D]``; ``cos``/``sin``: ``[T, D]``. Output keeps HF layout.
    """
    b, h, t, d = x.shape
    x = x.reshape(b, h, t, d // 2, 2)
    x = mx.transpose(x, (0, 1, 2, 4, 3)).reshape(b, h, t, d)
    cos = cos[None, None]
    sin = sin[None, None]
    return x * cos + _rotate_half(x) * sin


def apply_rope_fast(x: mx.array, inv_freq: mx.array, offset: int = 0) -> mx.array:
    """``mx.fast.rope`` path (traditional/interleaved). Attention-equivalent to explicit.

    mlx's ``freqs`` argument is the *period* (angle = position / freqs), so we pass
    the reciprocal of our angular ``inv_freq``. With ``traditional=True`` this is
    dot-product-equivalent to :func:`apply_rope_explicit` (verified to ~1e-6).
    """
    return mx.fast.rope(
        x, dims=x.shape[-1], traditional=True, base=None, scale=1.0, offset=offset, freqs=1.0 / inv_freq
    )
