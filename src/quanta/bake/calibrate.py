"""GPTQ calibration: capture per-expert input activations + the sensitivity metric.

``capture_calibration`` runs the bf16 forward and records, for each MoE layer, the
post-attention-norm input ``ln2`` ``[N, hidden]`` and the routing ``idx`` ``[N, topk]``.
Each routed expert's GPTQ input ``X`` is then ``ln2[tokens routed to it]`` — we store the
unique-token acts (not the topk-duplicated rows), so calibration is small (~hidden·N·2
bytes per layer). Memory-disciplined: capture uses only attention+norms+gate (no experts);
the residual is advanced through the experts only when there's a next layer to feed.

``activation_weighted_error`` is the DP's per-projection sensitivity: ``‖WX−ŴX‖/‖WX‖`` —
the (root of the) quantity GPTQ minimizes, far more e2e-predictive than raw recon. Here Ŵ
is RTN affine (a cheap proxy to screen int3 vs int4); GPTQ error-feedback lowers it further.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.bake.quant import dequantize_affine, quantize_affine
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.model import build_runtime_layer, load_layer_raw, load_module_weights
from quanta.modeling.attention import MLAAttention
from quanta.modeling.moe import MoEGate, SparseMoE


def _sub(weights: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}


def capture_calibration(
    ck: SourceCheckpoint, cfg: KimiTextConfig, token_ids: mx.array, *, n_layers: int | None = None
) -> list[tuple[mx.array, mx.array]]:
    """Per-MoE-layer ``(ln2 [N,hidden] bf16, idx [N,topk] int32)`` for GPTQ calibration."""
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    h = ck.embed_tokens(token_ids)[None].astype(mx.bfloat16)
    pos = mx.arange(0, h.shape[1])
    caps: list[tuple[mx.array, mx.array]] = []
    for i in range(n):
        if cfg.is_dense_layer(i):
            raw = load_layer_raw(ck, cfg, i, mx.bfloat16)
            layer = build_runtime_layer(cfg, raw)
            h = layer(h, pos, use_fast=True)
            mx.eval(h)
            del layer, raw
            ck.release()
            continue

        ne = ck.load_moe_nonexpert(i)  # attention + norms + gate + shared (no 34 GB experts)
        attn = MLAAttention(cfg)
        load_module_weights(attn, _sub(ne, "self_attn."))
        in_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        in_norm.weight = ne["input_layernorm.weight"]
        post_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        post_norm.weight = ne["post_attention_layernorm.weight"]
        gate = MoEGate(cfg)
        gate.weight = ne["mlp.gate.weight"]
        gate.e_score_correction_bias = ne["mlp.gate.e_score_correction_bias"]

        resid1 = h + attn(in_norm(h), pos, use_fast=True)
        ln2 = post_norm(resid1)
        ln2f = ln2.reshape(-1, cfg.hidden_size)
        idx, _ = gate(ln2f)
        mx.eval(ln2f, idx)
        caps.append((ln2f.astype(mx.bfloat16), idx))

        if i < n - 1:  # advance the residual through the experts to feed the next layer
            mlp = SparseMoE(cfg)
            load_module_weights(mlp.shared_experts, _sub(ne, "mlp.shared_experts."))
            mlp.gate.weight = ne["mlp.gate.weight"]
            mlp.gate.e_score_correction_bias = ne["mlp.gate.e_score_correction_bias"]
            ex = ck.load_expert_stacks(
                i, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size, dtype=mx.bfloat16
            )
            mlp.set_experts(ex["gate"], ex["up"], ex["down"])
            h = resid1 + mlp(ln2)
            mx.eval(h)
        ck.release()
    return caps


def expert_rows(ln2: mx.array, idx: mx.array, expert: int) -> mx.array:
    """Calibration input ``X`` ``[n, hidden]`` for one expert: rows routed to it (any top-k slot).

    MLX has no boolean indexing, so select via argsort of the routed mask (routed tokens sort
    first) then take the first ``count`` — vectorized integer gather.
    """
    mask = mx.any(idx == expert, axis=1).astype(mx.int32)  # [N] 1 = routed to this expert
    count = int(mx.sum(mask).item())
    rows = mx.argsort(-mask)[:count]  # indices of the routed tokens
    return ln2[rows]


def activation_weighted_error(w: mx.array, x: mx.array, bits: int, group_size: int = 128) -> float:
    """DP sensitivity ``‖WX−ŴX‖/‖WX‖`` for projection ``w`` ``[out,in]`` on inputs ``x`` ``[n,in]``.

    Ŵ is RTN affine here (cheap int3-vs-int4 screen); GPTQ error-feedback lowers it further.
    """
    wf = w.astype(mx.float32)
    wd = dequantize_affine(*quantize_affine(w, bits, group_size), bits, group_size).astype(mx.float32)
    xt = x.astype(mx.float32).T  # [in, n]
    wx = wf @ xt
    err = (wd - wf) @ xt
    return (mx.linalg.norm(err) / (mx.linalg.norm(wx) + 1e-12)).item()
