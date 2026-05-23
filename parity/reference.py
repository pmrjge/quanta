"""Dead-simple plain-``mlx.core`` reference forwards for Kimi-K2.6 decoder layers.

Independent transcription of ``modeling_deepseek.py``: explicit RMSNorm (fp32
upcast), YaRN RoPE with the de-interleave + rotate-half application, explicit MLA
attention, dense SwiGLU MLP (L0) and the sparse MoE block (L1+, with an obvious
per-token expert loop — reference clarity over speed). No fused kernels, no
nn.Module. Returns every intermediate so the first divergence is locatable.
"""

from __future__ import annotations

import math

import mlx.core as mx

from quanta.config import KimiTextConfig


def _rms(x: mx.array, w: mx.array, eps: float) -> mx.array:
    dt = x.dtype
    xf = x.astype(mx.float32)
    var = mx.mean(xf * xf, axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return w * xf.astype(dt)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _mscale(scale: float, mscale: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _yarn_inv_freq(cfg: KimiTextConfig) -> mx.array:
    dim, base, factor = cfg.qk_rope_head_dim, cfg.rope.rope_theta, cfg.rope.factor
    idx = mx.arange(0, dim, 2, dtype=mx.float32)
    freq_extra = 1.0 / (base ** (idx / dim))
    freq_inter = 1.0 / (factor * (base ** (idx / dim)))

    def corr_dim(num_rot: float) -> float:
        omax = cfg.rope.original_max_position_embeddings
        return (dim * math.log(omax / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(cfg.rope.beta_fast)), 0)
    high = min(math.ceil(corr_dim(cfg.rope.beta_slow)), dim - 1)
    lo, hi = float(low), float(high)
    if lo == hi:
        hi += 0.001
    ramp = mx.clip((mx.arange(dim // 2, dtype=mx.float32) - lo) / (hi - lo), 0.0, 1.0)
    inv_freq_mask = 1.0 - ramp
    return freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask


def _cos_sin(cfg: KimiTextConfig, positions: mx.array) -> tuple[mx.array, mx.array]:
    inv = _yarn_inv_freq(cfg)
    freqs = positions.astype(mx.float32)[:, None] * inv[None, :]
    mult = _mscale(cfg.rope.factor, cfg.rope.mscale) / _mscale(cfg.rope.factor, cfg.rope.mscale_all_dim)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb) * mult, mx.sin(emb) * mult


def _rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    b, h, t, d = x.shape
    x = mx.transpose(x.reshape(b, h, t, d // 2, 2), (0, 1, 2, 4, 3)).reshape(b, h, t, d)
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def reference_attention(
    ln1: mx.array, w: dict[str, mx.array], cfg: KimiTextConfig, positions: mx.array, dtype: mx.Dtype
) -> mx.array:
    """Explicit MLA attention; ``w`` holds dtype-cast weights. Returns ``[B, T, hidden]``."""
    heads, nope, rope, vhd, qhd = (
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.q_head_dim,
    )
    b, t, _ = ln1.shape

    q = ln1 @ w["self_attn.q_a_proj.weight"].T
    q = _rms(q, w["self_attn.q_a_layernorm.weight"], 1e-6)
    q = q @ w["self_attn.q_b_proj.weight"].T
    q = mx.transpose(q.reshape(b, t, heads, qhd), (0, 2, 1, 3))
    q_nope, q_pe = q[..., :nope], q[..., nope:]

    ckv = ln1 @ w["self_attn.kv_a_proj_with_mqa.weight"].T
    ckv, k_pe = ckv[..., : cfg.kv_lora_rank], ckv[..., cfg.kv_lora_rank :]
    k_pe = mx.transpose(k_pe.reshape(b, t, 1, rope), (0, 2, 1, 3))
    kv = _rms(ckv, w["self_attn.kv_a_layernorm.weight"], 1e-6)
    kv = kv @ w["self_attn.kv_b_proj.weight"].T
    kv = mx.transpose(kv.reshape(b, t, heads, nope + vhd), (0, 2, 1, 3))
    k_nope, value = kv[..., :nope], kv[..., nope:]

    cos, sin = _cos_sin(cfg, positions)
    cos, sin = cos.astype(dtype), sin.astype(dtype)
    q_pe = _apply_rope(q_pe, cos, sin)
    k_pe = mx.broadcast_to(_apply_rope(k_pe, cos, sin), (b, heads, t, rope))
    query = mx.concatenate([q_nope, q_pe], axis=-1)
    key = mx.concatenate([k_nope, k_pe], axis=-1)

    scale = qhd ** -0.5 * _mscale(cfg.rope.factor, cfg.rope.mscale_all_dim) ** 2
    scores = (query @ mx.transpose(key, (0, 1, 3, 2))) * scale
    i, j = mx.arange(t)[:, None], mx.arange(t)[None, :]
    scores = scores + mx.where(j <= i, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))
    attn_w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(dtype)
    attn = attn_w @ value
    attn = mx.transpose(attn, (0, 2, 1, 3)).reshape(b, t, heads * vhd)
    return attn @ w["self_attn.o_proj.weight"].T


def _router(
    xf: mx.array, w: dict[str, mx.array], cfg: KimiTextConfig
) -> tuple[mx.array, mx.array, mx.array]:
    """noaux_tc sigmoid router → (logits, topk_idx, topk_weight). All in fp32 selection."""
    topk = cfg.num_experts_per_tok
    logits = xf.astype(mx.float32) @ w["mlp.gate.weight"].astype(mx.float32).T
    scores = mx.sigmoid(logits)
    choice = scores + w["mlp.gate.e_score_correction_bias"].astype(mx.float32)[None]
    idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk]
    weights = mx.take_along_axis(scores, idx, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = weights * cfg.routed_scaling_factor
    return logits, idx, weights


def reference_layer0(
    h: mx.array,
    weights: dict[str, mx.array],
    cfg: KimiTextConfig,
    positions: mx.array,
    *,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    w = {k: v.astype(dtype) for k, v in weights.items()}
    h = h.astype(dtype)
    out: dict[str, mx.array] = {}

    ln1 = _rms(h, w["input_layernorm.weight"], cfg.rms_norm_eps)
    out["ln1"] = ln1
    attn = reference_attention(ln1, w, cfg, positions, dtype)
    out["attn"] = attn
    resid1 = h + attn
    out["resid1"] = resid1
    ln2 = _rms(resid1, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    out["ln2"] = ln2

    g = ln2 @ w["mlp.gate_proj.weight"].T
    u = ln2 @ w["mlp.up_proj.weight"].T
    mlp = (_silu(g) * u) @ w["mlp.down_proj.weight"].T
    out["mlp"] = mlp
    out["hout"] = resid1 + mlp
    return out


def reference_moe_layer(
    h: mx.array,
    weights: dict[str, mx.array],
    experts: dict[str, mx.array],
    cfg: KimiTextConfig,
    positions: mx.array,
    *,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    w = {k: v.astype(dtype) for k, v in weights.items()}
    h = h.astype(dtype)
    gate_s = experts["gate"].astype(dtype)
    up_s = experts["up"].astype(dtype)
    down_s = experts["down"].astype(dtype)
    out: dict[str, mx.array] = {}

    ln1 = _rms(h, w["input_layernorm.weight"], cfg.rms_norm_eps)
    out["ln1"] = ln1
    attn = reference_attention(ln1, w, cfg, positions, dtype)
    out["attn"] = attn
    resid1 = h + attn
    out["resid1"] = resid1
    ln2 = _rms(resid1, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    out["ln2"] = ln2

    b, t, hd = ln2.shape
    n = b * t
    xf = ln2.reshape(n, hd)
    logits, idx, tw = _router(xf, w, cfg)
    out["router_logits"], out["topk_idx"], out["topk_w"] = logits, idx, tw

    topk = cfg.num_experts_per_tok
    idx_list = idx.tolist()
    rows = []
    for ti in range(n):
        xt = xf[ti]
        acc = mx.zeros((hd,), mx.float32)
        for jj in range(topk):
            e = idx_list[ti][jj]
            g = gate_s[e] @ xt
            u = up_s[e] @ xt
            d = down_s[e] @ (_silu(g) * u)
            acc = acc + tw[ti, jj] * d.astype(mx.float32)
        rows.append(acc.astype(dtype))
    routed = mx.stack(rows).reshape(b, t, hd)
    out["routed"] = routed

    sg = xf @ w["mlp.shared_experts.gate_proj.weight"].T
    su = xf @ w["mlp.shared_experts.up_proj.weight"].T
    shared = ((_silu(sg) * su) @ w["mlp.shared_experts.down_proj.weight"].T).reshape(b, t, hd)
    out["shared"] = shared

    moe = routed + shared
    out["moe"] = moe
    out["hout"] = resid1 + moe
    return out
