"""Quant-vs-forward discriminator: do the baked artifact's dequantized weights match the source?

The forward is broken (0/32) and no forward-convention toggle (RoPE traditional, GDN DELTA_RULE)
moves it — which is consistent with the WEIGHTS being wrong, not the math. The slice smoke checked
dequant shapes + finiteness, never VALUES. This compares the source bf16 checkpoint against the baked
artifact for representative tensors (both lazy/mmap — only the compared tensors materialize, light):

  * dense (verbatim copy) — embed / a norm / lm_head — must be BIT-IDENTICAL (max|Δ|==0).
  * int8 affine — a linear-attn in_proj, a full-attn q_proj — rel err should be ~<1% (int8 grid).
  * int4 affine — a routed expert stack (gate_up, down) — rel err should be ~a few % (int4 grid).

If dense is exact and int8/int4 are within grid tolerance ⇒ dequant is CORRECT ⇒ the bug is the
forward math (→ build the HF layer-by-layer oracle). If any is wildly off ⇒ a bake/quant/pack bug.

    uv run python -u -m parity.qwen36_dequant_check
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.loader import Qwen35SourceCheckpoint

SRC = "/Users/pmrj/models/Qwen3.6-35B-A3B"
ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"


def _rel(a: mx.array, b: mx.array) -> tuple[float, float]:
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    maxabs = float(mx.max(mx.abs(a - b)).item())
    rel = maxabs / (float(mx.max(mx.abs(b)).item()) + 1e-8)
    return maxabs, rel


def run() -> None:
    src = Qwen35SourceCheckpoint(SRC)
    art = Qwen35Artifact(ART)

    print("=== dense (must be bit-identical) ===", flush=True)
    for name, sa, aa in (
        ("embed", src.embed(), art.embed()),
        ("final_norm", src.final_norm(), art.final_norm()),
        ("lm_head", src.lm_head(), art.lm_head()),
    ):
        m, r = _rel(sa, aa)
        print(f"  [{'OK' if m == 0.0 else 'XX'}] {name:12} max|Δ|={m:.3e} rel={r:.3e}", flush=True)

    print("=== int8 affine (rel err should be <~1%) ===", flush=True)
    s_lin, a_lin = src.linear_attn(0), art.linear_attn(0)
    for key in ("in_proj_qkv.weight", "out_proj.weight"):
        m, r = _rel(s_lin[key], a_lin[key])
        print(f"  [{'OK' if r < 0.05 else 'XX'}] L0.linear_attn.{key:18} max|Δ|={m:.3e} rel={r:.3e}", flush=True)
    s_full, a_full = src.full_attn(3), art.full_attn(3)
    for key in ("q_proj.weight", "o_proj.weight"):
        m, r = _rel(s_full[key], a_full[key])
        print(f"  [{'OK' if r < 0.05 else 'XX'}] L3.self_attn.{key:18} max|Δ|={m:.3e} rel={r:.3e}", flush=True)

    print("=== int4 affine routed experts (rel err should be ~a few %) ===", flush=True)
    s_moe, a_moe = src.moe(0), art.moe(0)
    for key in ("experts_gate_up", "experts_down"):
        m, r = _rel(s_moe[key], a_moe[key])
        print(f"  [{'OK' if r < 0.30 else 'XX'}] L0.mlp.{key:16} shape={tuple(a_moe[key].shape)} "
              f"max|Δ|={m:.3e} rel={r:.3e}", flush=True)
    # also the shared expert (int8) + router gate (dense)
    m, r = _rel(s_moe["shared_gate_proj"], a_moe["shared_gate_proj"])
    print(f"  [{'OK' if r < 0.05 else 'XX'}] L0.mlp.shared_gate_proj   max|Δ|={m:.3e} rel={r:.3e}", flush=True)
    m, r = _rel(s_moe["gate"], a_moe["gate"])
    print(f"  [{'OK' if m == 0.0 else 'XX'}] L0.mlp.gate (dense)       max|Δ|={m:.3e} rel={r:.3e}", flush=True)


if __name__ == "__main__":
    run()
