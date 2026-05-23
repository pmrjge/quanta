"""Validate the bake's affine int8 path on a real non-expert weight (bounded, ~no load).

int8 (group-128) must be near-lossless on real weights, and mx.quantized_matmul (the
resident-runtime path) must match the bf16 matmul. Shows the bits ladder (8/4/3) for
context — int8 for non-experts, int3 is the experts' GPTQ target.

    uv run python -m parity.bake_quant_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.quant import dequantize_affine, quantize_affine, recon_error
from quanta.config import KimiTextConfig
from quanta.loader import SourceCheckpoint

MODEL = "/Users/pmrj/models/Kimi-K2.6"


def run() -> None:
    KimiTextConfig.from_pretrained(MODEL)
    ck = SourceCheckpoint(MODEL)
    w = ck.load_moe_nonexpert(1)["self_attn.o_proj.weight"]  # [hidden, H*v_head_dim], real bf16
    ck.release()
    out_f, in_f = w.shape
    print(f"\n=== bake int8 quant on real weight o_proj {tuple(w.shape)} ===")

    print("recon error (rel) by bits:")
    for bits in (8, 4, 3):
        print(f"  int{bits} g128 : {recon_error(w, bits):.4%}")

    # resident-runtime path: quantized_matmul must match bf16 matmul
    wq, s, b = quantize_affine(w, 8)
    x = mx.random.normal((8, in_f)).astype(mx.bfloat16)
    y_ref = x @ w.T
    y_q = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=128, bits=8)
    rel = (mx.linalg.norm((y_ref - y_q).astype(mx.float32)) / mx.linalg.norm(y_ref.astype(mx.float32))).item()
    wd = dequantize_affine(wq, s, b, 8)
    drift = (mx.linalg.norm((w - wd).astype(mx.float32)) / mx.linalg.norm(w.astype(mx.float32))).item()
    print(f"\nint8 quantized_matmul vs bf16 : rel {rel:.4%}")
    print(f"int8 dequant round-trip       : rel {drift:.4%}")
    int8_ok = recon_error(w, 8) < 0.01 and rel < 0.02
    print(f"\nint8 near-lossless (<1% recon, <2% matmul): {int8_ok}")
    assert int8_ok, "int8 non-expert quant not near-lossless"


if __name__ == "__main__":
    run()
