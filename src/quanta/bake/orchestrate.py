"""Bake orchestration — drive the per-layer quantization into a self-contained artifact.

One layer resident at a time. Steps:
  1. capture calibration (per-MoE-layer post-norm acts + routing).
  2. PASS 1 (cheap): per routed projection, the activation-weighted RTN sensitivity at int3
     and int4 → a global DP allocates int3/int4 under the expert byte budget (target <8%).
  3. PASS 2: write the artifact — non-experts (attention, dense L0 MLP, embed/lm_head) int8,
     shared expert + norms + router bf16, routed experts GPTQ at their allocated bits, packed.

Down-proj is calibrated on the intermediate from the (bf16) gate/up output; sequential
quantized-input down calibration is a later refinement. Runnable on a slice (``n_layers``,
``expert_subset``, ``include_head``) for bounded validation; the full call is the real bake.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.bake.allocate import BPP, Projection, allocate_bits
from quanta.bake.artifact import ArtifactWriter
from quanta.bake.calibrate import activation_weighted_error, capture_calibration, expert_rows
from quanta.bake.gptq import gptq_quantize
from quanta.bake.quant import pack_affine, quantize_affine
from quanta.compressed_int4 import dequantize_packed_int4
from quanta.config import KimiTextConfig
from quanta.loader import (
    ATTENTION_SUFFIXES,
    DENSE_MLP_SUFFIXES,
    TEXT_PREFIX,
    SourceCheckpoint,
)

NORM_SUFFIXES = frozenset({
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight",
})
_EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")


def _dequant_expert(ck: SourceCheckpoint, cfg: KimiTextConfig, layer: int, e: int, proj: str) -> mx.array:
    base = f"{TEXT_PREFIX}layers.{layer}.mlp.experts.{e}.{proj}."
    out_f, in_f = ((cfg.hidden_size, cfg.moe_intermediate_size) if proj == "down_proj"
                   else (cfg.moe_intermediate_size, cfg.hidden_size))
    return dequantize_packed_int4(ck.read(base + "weight_packed"), ck.read(base + "weight_scale"),
                                  out_f, in_f, 32, mx.float32)


def _down_input(g: mx.array, u: mx.array, xe: mx.array) -> mx.array:
    """down-proj calibration input: silu(gate·x)·(up·x) from bf16 gate/up. ``[n, inter]``."""
    return (nn.silu(g @ xe.T) * (u @ xe.T)).T


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int) -> None:
    packed, scales, biases = quantize_affine(w, 8, gs)
    writer.add_quantized(key, packed, scales, biases, 8, gs)


def bake(
    source: str | Path,
    out_dir: str | Path,
    calib_ids: mx.array,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    expert_byte_budget: float | None = None,
    target: float = 0.08,
    group_size: int = 128,
) -> dict:
    cfg = KimiTextConfig.from_pretrained(source)
    ck = SourceCheckpoint(source)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts = list(range(cfg.n_routed_experts)) if expert_subset is None else list(expert_subset)
    moe_layers = [i for i in range(n) if not cfg.is_dense_layer(i)]

    caps = capture_calibration(ck, cfg, calib_ids, n_layers=n)
    cap_of = dict(zip(moe_layers, caps))

    # PASS 1 — sensitivities (cheap RTN proxy) → global DP allocation
    projs: list[Projection] = []
    for li in moe_layers:
        ln2, idx = cap_of[li]
        for e in experts:
            xe = expert_rows(ln2, idx, e)
            g, u, d = (_dequant_expert(ck, cfg, li, e, p) for p in _EXPERT_PROJS)
            xd = _down_input(g, u, xe) if xe.shape[0] > 0 else xe  # cold: skip empty matmul
            for proj, w, x in (("gate_proj", g, xe), ("up_proj", u, xe), ("down_proj", d, xd)):
                key = f"{TEXT_PREFIX}layers.{li}.mlp.experts.{e}.{proj}"
                projs.append(Projection(key, w.size,
                                        activation_weighted_error(w, x, 3, group_size),
                                        activation_weighted_error(w, x, 4, group_size)))
            ck.release()
    budget = expert_byte_budget if expert_byte_budget is not None else sum(p.params for p in projs) * BPP[4] / 8
    bits_map, total_err, used = allocate_bits(projs, budget, target)

    # PASS 2 — write the artifact
    writer = ArtifactWriter(out_dir, Path(source) / "config.json")
    if include_head:
        _write_int8(writer, f"{TEXT_PREFIX}embed_tokens", ck.read(f"{TEXT_PREFIX}embed_tokens.weight"), group_size)
        writer.add_dense(f"{TEXT_PREFIX}norm.weight", ck.read(f"{TEXT_PREFIX}norm.weight"))
        _write_int8(writer, "language_model.lm_head", ck.read("language_model.lm_head.weight"), group_size)
        ck.release()

    for i in range(n):
        pre = f"{TEXT_PREFIX}layers.{i}."
        if cfg.is_dense_layer(i):
            w = ck.load_dense_layer(i)
            for suf in ATTENTION_SUFFIXES + DENSE_MLP_SUFFIXES:
                if suf in NORM_SUFFIXES:
                    writer.add_dense(pre + suf, w[suf])
                else:
                    _write_int8(writer, pre + suf[: -len(".weight")], w[suf], group_size)
            ck.release()
            continue

        ne = ck.load_moe_nonexpert(i)
        for suf, arr in ne.items():
            if suf in NORM_SUFFIXES or suf.startswith("mlp.gate.") or suf.startswith("mlp.shared_experts."):
                writer.add_dense(pre + suf, arr)  # router + shared + norms stay bf16
            else:
                _write_int8(writer, pre + suf[: -len(".weight")], arr, group_size)  # attention int8
        ln2, idx = cap_of[i]
        for e in experts:
            xe = expert_rows(ln2, idx, e)
            g, u, d = (_dequant_expert(ck, cfg, i, e, p) for p in _EXPERT_PROJS)
            n_e = xe.shape[0]
            xd = _down_input(g, u, xe) if n_e > 0 else xe
            inputs = {"gate_proj": xe, "up_proj": xe, "down_proj": xd}
            for proj, w in (("gate_proj", g), ("up_proj", u), ("down_proj", d)):
                key = f"{pre}mlp.experts.{e}.{proj}"
                bits = bits_map[key]
                if n_e == 0:  # cold expert: GPTQ needs calibration rows → RTN fallback
                    packed, scales, biases = quantize_affine(w, bits, group_size)
                else:
                    _, codes, scales, biases = gptq_quantize(w, inputs[proj], bits, group_size=group_size)
                    packed = pack_affine(codes.astype(mx.uint32), bits)
                writer.add_quantized(key, packed, scales, biases, bits, group_size)
            ck.release()

    policy = {"experts": "int3/int4 gptq g128", "non_experts": "int8 g128",
              "shared": "bf16", "norms": "bf16", "target_error": target}
    writer.finalize(policy)
    return {"layers": n, "experts": len(experts), "alloc_error": total_err, "expert_bytes": used,
            "int4_projections": sum(v == 4 for v in bits_map.values()), "projections": len(bits_map)}
