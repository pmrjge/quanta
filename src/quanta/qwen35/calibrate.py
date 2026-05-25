"""Qwen3.5-397B-A17B bake calibration: per-layer post-attention-norm activations + routing.

Mirrors the DSV4 / Nemotron calibration (:mod:`quanta.dsv4.calibrate`,
:mod:`quanta.nemotron.calibrate`) on the Qwen3.5 hybrid (Gated-DeltaNet linear + gated-GQA full)
decoder. A streamed, one-layer-resident bf16 forward advances the residual through every block; at
each layer it records the **MoE input** ``x`` ``[N, hidden]`` (``post_attention_layernorm(x)`` — the
routed experts' exact input) and the routing ``idx`` ``[N, topk]``. MoE is on **every** layer, so
every layer is captured.

The int4-g64 expert recipe is plain affine RTN over the pre-stacked stacks (``bake/quant.py``), which
needs no activations; this capture exists for (a) the activation-weighted QC gauge and (b) a future
GPTQ/AWQ expert pass — same precedent as the other models, where the capture feeds AWQ. Memory
discipline (rule-8): one block's bf16 weights resident at a time; experts are loaded only to advance
the stream and dropped before the next layer.

The forward reuses the proven naive modules (:class:`quanta.qwen35.gated_deltanet.GatedDeltaNet`,
:class:`quanta.qwen35.attention.Qwen35Attention`, :func:`quanta.qwen35.moe.qwen35_moe`) so the
capture point is numerically the bf16 reference, not a re-derivation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.qwen35.attention import Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.gated_deltanet import GatedDeltaNet
from quanta.qwen35.moe import qwen35_moe, qwen35_route


def _load_mixer(cfg: Qwen35Config, t: dict, layer_id: int) -> nn.Module:
    """Build the layer's mixer (GatedDeltaNet | Qwen35Attention) and fill it from source tensors.

    ``t`` is the loader sub-dict for the layer (``linear_attn(i)`` or ``full_attn(i)``); keys are the
    suffix sets in :mod:`quanta.qwen35.loader`.
    """
    if cfg.is_linear_attention(layer_id):
        m = GatedDeltaNet(cfg)
        m.in_proj_qkv.weight = t["in_proj_qkv.weight"]
        m.in_proj_a.weight = t["in_proj_a.weight"]
        m.in_proj_b.weight = t["in_proj_b.weight"]
        m.in_proj_z.weight = t["in_proj_z.weight"]
        m.out_proj.weight = t["out_proj.weight"]
        m.conv_weight = mx.squeeze(t["conv1d.weight"], 1)  # (C,1,K) -> (C,K)
        m.conv_bias = t.get("conv1d.bias", m.conv_bias)
        m.A_log = t["A_log"]
        m.dt_bias = t["dt_bias"]
        m.norm = t["norm.weight"]
        return m
    m = Qwen35Attention(cfg)
    m.q_proj.weight = t["q_proj.weight"]
    m.k_proj.weight = t["k_proj.weight"]
    m.v_proj.weight = t["v_proj.weight"]
    m.o_proj.weight = t["o_proj.weight"]
    m.q_norm = t["q_norm.weight"]
    m.k_norm = t["k_norm.weight"]
    return m


def _moe_params(moe: dict) -> dict:
    """Map a loader ``moe(i)`` sub-dict to the ``p`` dict :func:`qwen35_moe` consumes."""
    return {
        "gate": moe["gate"],
        "experts_gate_up": moe["experts_gate_up"],
        "experts_down": moe["experts_down"],
        "shared_gate_proj": moe["shared_gate_proj"],
        "shared_up_proj": moe["shared_up_proj"],
        "shared_down_proj": moe["shared_down_proj"],
        "shared_expert_gate": moe["shared_expert_gate"],
    }


def capture_calibration(
    ck, cfg: Qwen35Config, calib_ids: mx.array, *, n_layers: int | None = None,
) -> dict[int, tuple[mx.array, mx.array]]:
    """Per-layer ``{i: (x [N,hidden] bf16, idx [N,topk] int32)}`` for the bake.

    ``calib_ids`` is ``[S]`` (or ``[1,S]``) token ids. ``ck`` is a
    :class:`quanta.qwen35.loader.Qwen35SourceCheckpoint` (or a duck-typed artifact reader). The
    forward is the bf16 reference with one block resident at a time (rule-8).
    """
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ids = calib_ids.reshape(1, -1)
    eps = cfg.norm_eps

    h = ck.embed()[ids].astype(mx.bfloat16)  # [1, S, hidden]
    mx.eval(h)
    ck.release()

    caps: dict[int, tuple[mx.array, mx.array]] = {}
    in_norm = nn.RMSNorm(cfg.hidden_size, eps=eps)
    post_norm = nn.RMSNorm(cfg.hidden_size, eps=eps)
    for i in range(n):
        norms = ck.block_norms(i)
        in_norm.weight = norms["input_layernorm"]
        post_norm.weight = norms["post_attention_layernorm"]

        mixer_t = ck.linear_attn(i) if cfg.is_linear_attention(i) else ck.full_attn(i)
        mixer = _load_mixer(cfg, mixer_t, i)

        # mixer sub-block advances the residual (prefill: fresh state/cache)
        hn = in_norm(h)
        if cfg.is_linear_attention(i):
            y, _, _ = mixer(hn)
        else:
            y = mixer(hn, cache=None, use_fast=True, seq_hint=h.shape[1])
        h = h + y

        # MoE sub-block: capture the experts' input + routing, then advance
        moe = ck.moe(i)
        x = post_norm(h)
        xf = x.reshape(-1, cfg.hidden_size)
        idx, _ = qwen35_route(xf.astype(mx.float32), moe["gate"], cfg)
        mx.eval(xf, idx)
        caps[i] = (xf.astype(mx.bfloat16), idx.astype(mx.int32))

        if i < n - 1:  # advance the residual through the MoE to feed the next layer
            h = h + qwen35_moe(x, _moe_params(moe), cfg, sparse=True)
            mx.eval(h)
        del mixer, mixer_t, moe, norms
        ck.release()
        mx.clear_cache()
    return caps


def expert_rows(x_cap: mx.array, idx_cap: mx.array, expert: int) -> mx.array:
    """Calibration input ``X`` ``[n, hidden]`` for one expert: the rows routed to it (any top-k slot).

    Thin re-export of :func:`quanta.bake.calibrate.expert_rows` so the Qwen3.5 bake/test import it
    from one place alongside :func:`capture_calibration`.
    """
    from quanta.bake.calibrate import expert_rows as _rows

    return _rows(x_cap, idx_cap, expert)
