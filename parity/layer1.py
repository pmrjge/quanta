"""L1 (first MoE layer) parity gate: plain-mlx.core reference vs the mlx.nn runtime.

Validates the MoE block on top of the proven attention/norms: noaux_tc router
(top-8 set agreement + weights), sparse ``gather_mm`` dispatch, shared expert, and
the int4 expert dequant. Experts run in bf16 (forward-path parity; ``gather_qmm``
comes after the bake).

    uv run python -m parity.layer1
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from parity.reference import reference_moe_layer
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint
from quanta.modeling.decoder import MoEDecoderLayer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
LAYER = 1
TOKEN_IDS = [163584, 100, 500, 1024, 2048, 4096, 8192, 16000, 32000, 64000, 100000, 120000, 150000, 42, 7, 9001]
BOUNDARIES = ["ln1", "attn", "resid1", "ln2", "routed", "shared", "moe", "hout"]


def _diff(a: mx.array, b: mx.array) -> tuple[float, float]:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    abs_err = mx.max(mx.abs(a - b)).item()
    denom = mx.maximum(mx.abs(a), mx.abs(b))
    rel = mx.max(mx.abs(a - b) / mx.where(denom > 0, denom, mx.array(1.0))).item()
    return abs_err, rel


def _set_agreement(a: mx.array, b: mx.array) -> float:
    al, bl = a.tolist(), b.tolist()
    return sum(set(x) == set(y) for x, y in zip(al, bl)) / len(al)


def _load_into(layer: MoEDecoderLayer, weights: dict[str, mx.array]) -> None:
    names = {k for k, _ in tree_flatten(layer.parameters())}
    for k in weights:
        if k not in names:
            raise KeyError(f"weight key {k!r} has no matching param in the runtime module")
    layer.load_weights([(k, v) for k, v in weights.items()], strict=False)


def run(dtype: mx.Dtype) -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    ck = SourceCheckpoint(MODEL)
    nonexpert = ck.load_moe_nonexpert(LAYER)
    experts = ck.load_expert_stacks(
        LAYER, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size, dtype=dtype
    )

    ids = mx.array(TOKEN_IDS)
    h = ck.embed_tokens(ids)[None].astype(dtype)
    pos = mx.arange(h.shape[1])

    ref = reference_moe_layer(h, nonexpert, experts, cfg, pos, dtype=dtype)

    layer = MoEDecoderLayer(cfg)
    _load_into(layer, {k: v.astype(dtype) for k, v in nonexpert.items()})
    layer.mlp.set_experts(experts["gate"], experts["up"], experts["down"])

    ln1 = layer.input_layernorm(h)
    attn = layer.self_attn(ln1, pos, use_fast=False)
    resid1 = h + attn
    ln2 = layer.post_attention_layernorm(resid1)
    xf = ln2.reshape(-1, cfg.hidden_size)
    r_idx, r_w = layer.mlp.gate(xf)
    moe, routed_rt, shared_rt = layer.mlp(ln2, return_parts=True)
    hout = resid1 + moe

    layer.mlp.sort_dispatch = True
    moe_sorted = layer.mlp(ln2)
    layer.mlp.sort_dispatch = False
    mx.eval(moe_sorted)
    rt = {"ln1": ln1, "attn": attn, "resid1": resid1, "ln2": ln2,
          "routed": routed_rt, "shared": shared_rt, "moe": moe, "hout": hout}
    mx.eval(list(ref.values()), list(rt.values()), r_idx, r_w)

    name = {mx.float32: "fp32", mx.bfloat16: "bf16"}[dtype]
    print(f"\n=== L1 MoE parity ({name}) — reference vs runtime ===")
    print(f"top-8 expert set agreement: {_set_agreement(ref['topk_idx'], r_idx):.3f}")
    a, r = _diff(ref["router_logits"], (xf.astype(mx.float32) @ layer.mlp.gate.weight.astype(mx.float32).T))
    print(f"router_logits max_abs {a:.3e}")
    a, r = _diff(ref["topk_w"], r_w)
    print(f"topk_weight  max_abs {a:.3e}  max_rel {r:.3e}")
    print(f"{'op':<10}{'max_abs':>14}{'max_rel':>14}")
    for k in BOUNDARIES:
        a, r = _diff(ref[k], rt[k])
        print(f"{k:<10}{a:>14.3e}{r:>14.3e}")

    a, r = _diff(ref["moe"], moe_sorted)
    print(f"{'moe/sort':<10}{a:>14.3e}{r:>14.3e}   (#2 sorted-dispatch vs reference)")


if __name__ == "__main__":
    run(mx.bfloat16)
