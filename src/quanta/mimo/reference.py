"""Streamed bf16 reference forward for MiMo-V2.5 (one text layer resident at a time).

A fully-resident bf16 forward is infeasible (47 MoE layers x ~13 GB of dequantized experts ≈ 611 GB
> RAM), so this streams: build + dequantize layer ``i``, run, ``mx.eval``, drop it, move on
(rule-8). It is assembled from the per-layer HF-gated modules (:mod:`quanta.mimo.model` /
:mod:`quanta.mimo.moe`), so it *is* the bf16 reference — the teacher-forced-ppl coherence arbiter
and the gate target for the int4 bake (#62). Text-only; vision/audio are #64/#65.

Attention masks: full-attention layers use a full causal mask; SWA layers use a sliding-window
causal mask (``0 <= i-j < sliding_window``). A test sequence longer than the window exercises the
windowing; the per-position attention math itself is gated in ``parity/mimo_layer_parity.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.loader import MiMoSourceCheckpoint
from quanta.mimo.model import MiMoDecoderLayer
from quanta.mimo.moe import MiMoMoE


def full_causal_mask(length: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    i, j = mx.arange(length)[:, None], mx.arange(length)[None, :]
    return mx.where(j <= i, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


def sliding_window_mask(length: int, window: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    i, j = mx.arange(length)[:, None], mx.arange(length)[None, :]
    keep = (j <= i) & ((i - j) < window)
    return mx.where(keep, mx.array(0.0, dtype), mx.array(-mx.inf, dtype))


def _build_layer(cfg: MiMoV2Config, ck: MiMoSourceCheckpoint, i: int,
                 dtype: mx.Dtype = mx.bfloat16) -> MiMoDecoderLayer:
    layer = MiMoDecoderLayer(cfg, i)
    nrm = ck.norm_tensors(i)
    layer.input_layernorm.weight = nrm["input_layernorm"].astype(dtype)
    layer.post_attention_layernorm.weight = nrm["post_attention_layernorm"].astype(dtype)
    aw = ck.attention_tensors(i)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(layer.self_attn, n).weight = aw[n].astype(dtype)
    if layer.self_attn.has_sink:
        layer.self_attn.attention_sink_bias = aw["attention_sink_bias"].astype(dtype)
    if cfg.is_moe(i):
        moe = MiMoMoE(cfg)
        r = ck.moe_router_tensors(i)
        moe.gate_weight = r["weight"].astype(dtype)
        moe.e_score_correction_bias = r["e_score_correction_bias"].astype(dtype)
        st = ck.expert_stacks(i)
        moe.set_experts(st["gate_proj"].astype(dtype), st["up_proj"].astype(dtype), st["down_proj"].astype(dtype))
        layer.mlp = moe
    else:
        m = ck.dense_mlp_tensors(i)
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(layer.mlp, n).weight = m[n].astype(dtype)
    return layer


def streamed_logits(cfg: MiMoV2Config, ck: MiMoSourceCheckpoint, ids: list[int],
                    dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Full streamed forward over token ids -> logits ``[1, L, vocab]``."""
    embed = ck.read("model.embed_tokens.weight").astype(dtype)
    h = embed[mx.array(ids)][None]
    mx.eval(h)
    ck.release()
    length = h.shape[1]
    full_m = full_causal_mask(length, dtype)
    swa_m = sliding_window_mask(length, cfg.sliding_window, dtype)
    for i in range(cfg.num_hidden_layers):
        layer = _build_layer(cfg, ck, i, dtype)
        h = layer(h, swa_m if cfg.is_swa(i) else full_m, offset=0)
        mx.eval(h)
        ck.release()
        del layer
    h = mx.fast.rms_norm(h, ck.read("model.norm.weight").astype(h.dtype), cfg.norm_eps)
    logits = h @ ck.read("lm_head.weight").astype(dtype).T
    mx.eval(logits)
    ck.release()
    return logits


def teacher_forced_ppl(cfg: MiMoV2Config, ck: MiMoSourceCheckpoint, ids: list[int],
                       dtype: mx.Dtype = mx.bfloat16) -> float:
    """exp(mean next-token CE) over a single streamed forward pass."""
    lp = streamed_logits(cfg, ck, ids, dtype)[0, :-1].astype(mx.float32)
    tgt = mx.array(ids[1:])
    nll = mx.mean(mx.logsumexp(lp, axis=-1) - mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0])
    return float(mx.exp(nll).item())
