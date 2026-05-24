"""Plain-MLX text-decoder modules for MiMo-V2.5 (the parity reference path).

Mirrors ``modeling_mimo_v2.py`` exactly (the spec), built on ``mlx.nn`` + ``mx.fast`` per rules 1-2:

* :class:`MiMoAttention` — hybrid full/SWA GQA. q/k/v come pre-split by the loader (the fused-qkv
  trap lives in the loader). **Partial RoPE**: only the first ``rope_dim`` dims of each head are
  rotated (GPT-NeoX half-rotation == ``mx.fast.rope(traditional=False)``); the rest pass through.
  V is scaled by ``attention_value_scale`` (0.707) before SDPA. SWA layers add a per-head
  **attention sink**: a raw bias logit is appended to the score row, softmax taken over ``k+1``,
  then the sink column dropped — so attention mass can escape (StreamingLLM-style). Because the sink
  is unsupported by ``mx.fast.scaled_dot_product_attention``, the reference uses the explicit eager
  softmax (matching HF's eager path, which also falls back from SDPA when a sink is present).
* :class:`MiMoDenseMLP` — SwiGLU ``down(silu(gate(x)) * up(x))`` (the L0 dense FFN).
* :class:`MiMoRMSNorm` — ``mx.fast.rms_norm`` (f32 reduce), matching ``MiMoV2RMSNorm``.

A short test sequence (``T <= sliding_window``) makes the SWA mask identical to the full causal
mask, so layer parity isolates the per-layer math from the windowing (windowing is tested
separately). Gated vs the HF per-layer oracle in ``parity/mimo_layer_parity.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.mimo.config import MiMoV2Config


class MiMoRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class MiMoDenseMLP(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: MiMoV2Config, intermediate: int | None = None) -> None:
        super().__init__()
        inter = intermediate if intermediate is not None else cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.gate_proj(x)
        return self.down_proj((g * mx.sigmoid(g)) * self.up_proj(x))


def causal_mask(t: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Additive [t,t] causal mask (0 on/below diagonal, -inf above)."""
    idx = mx.arange(t)
    return mx.where(idx[None, :] <= idx[:, None], mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


class MiMoAttention(nn.Module):
    def __init__(self, cfg: MiMoV2Config, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        swa = cfg.is_swa(layer_idx)
        self.swa = swa
        self.nh = cfg.attn_heads(swa)
        self.nkv = cfg.attn_kv_heads(swa)
        self.hd = cfg.attn_head_dim(swa)
        self.vhd = cfg.attn_v_head_dim(swa)
        self.rep = self.nh // self.nkv
        self.rope_dim = cfg.rope_dim(swa)
        self.scale = cfg.attn_scale(swa)
        self.v_scale = cfg.attention_value_scale
        self.theta = cfg.attn_rope_theta(swa)
        self.sliding_window = cfg.sliding_window_for(swa)
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.vhd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.nh * self.vhd, cfg.hidden_size, bias=False)
        self.has_sink = cfg.has_attn_sink(swa)
        if self.has_sink:
            self.attention_sink_bias = mx.zeros((self.nh,))

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        """Partial RoPE: rotate first rope_dim dims (neox half-rotation), pass through the rest."""
        return mx.fast.rope(x, self.rope_dim, traditional=False, base=self.theta,
                            scale=1.0, offset=offset)

    def __call__(self, x: mx.array, mask: mx.array, offset: int = 0) -> mx.array:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.nkv, self.hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.nkv, self.vhd).transpose(0, 2, 1, 3)
        v = v * self.v_scale
        q = self._rope(q, offset)
        k = self._rope(k, offset)
        if self.rep > 1:
            k = mx.repeat(k, self.rep, axis=1)
            v = mx.repeat(v, self.rep, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)).astype(mx.float32) * self.scale + mask.astype(mx.float32)
        if self.has_sink:
            sink = mx.broadcast_to(self.attention_sink_bias.astype(mx.float32)[None, :, None, None],
                                   (b, self.nh, t, 1))
            scores = mx.concatenate([scores, sink], axis=-1)
            probs = mx.softmax(scores, axis=-1)[..., :-1]
        else:
            probs = mx.softmax(scores, axis=-1)
        out = (probs.astype(v.dtype) @ v).transpose(0, 2, 1, 3).reshape(b, t, self.nh * self.vhd)
        return self.o_proj(out)


class MiMoDecoderLayer(nn.Module):
    """Pre-norm decoder block: ``x += attn(in_ln(x)); x += mlp(post_ln(x))``.

    Mixer-agnostic — ``mlp`` is :class:`MiMoDenseMLP` on L0 and the routed MoE (set externally)
    on L1+. The residual structure matches ``MiMoV2DecoderLayer``; gated end-to-end at #60.
    """

    def __init__(self, cfg: MiMoV2Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = MiMoRMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.self_attn = MiMoAttention(cfg, layer_idx)
        self.post_attention_layernorm = MiMoRMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp: nn.Module = MiMoDenseMLP(cfg)  # replaced by the routed MoE for moe layers (#59)

    def __call__(self, x: mx.array, mask: mx.array, offset: int = 0) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask, offset)
        return x + self.mlp(self.post_attention_layernorm(x))
