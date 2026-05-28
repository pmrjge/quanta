"""int8 bake ↔ packed-runtime round-trip for InternLM2.5 (loads only N layers — small/fast).

Bakes the first ``N`` source layers at **int8 g64** to a temp dir, then confirms:
  (1) the packed runtime reads ``bits=8`` for BOTH attention and FFN back from the manifest
      (the rule-6 manifest-as-source-of-truth path, not a hardcoded width), and
  (2) the packed ``mx.quantized_matmul`` forward matches the bf16-dequant reference forward over the
      same int8 codes (kernel correctness — a wrong-bits decode would be O(1) off, not <5%).

This is the bake→runtime correctness gate; the end-to-end int8-vs-original ppl delta is the separate
``parity/internlm2_packed_ppl.py`` gate on the full artifact.

    uv run --with numpy python -m parity.internlm2_bake_roundtrip [n_layers]
"""

from __future__ import annotations

import shutil
import sys
import tempfile

import mlx.core as mx

from quanta.internlm2.bake import bake_internlm2
from quanta.internlm2.runtime import InternLM2ResidentModel

SOURCE = "/Users/pmrj/models/internlm2_5-7b-chat-1m"


def run(n_layers: int = 2) -> None:
    tmp = tempfile.mkdtemp(prefix="internlm2_int8_roundtrip_")
    try:
        summary = bake_internlm2(SOURCE, tmp, n_layers=n_layers, include_head=True,
                                 group_size=64, attn_bits=8, mlp_bits=8)
        print(f"bake counts={summary['counts']}  bytes={summary['bytes']/1e6:.1f}MB  layers={n_layers}")

        packed = InternLM2ResidentModel(tmp, packed=True, n_layers=n_layers)
        bf16 = InternLM2ResidentModel(tmp, packed=False, n_layers=n_layers)

        l0 = packed._model.layers[0]
        bits_ok = (l0.attn_bits == 8 and l0.mlp_bits == 8
                   and l0.attn_gs == 64 and l0.mlp_gs == 64)
        print(f"manifest-read layer0: attn=int{l0.attn_bits}/g{l0.attn_gs}  "
              f"mlp=int{l0.mlp_bits}/g{l0.mlp_gs}  (expect int8/g64) -> {bits_ok}")

        ids = mx.array([[1, 100, 200, 300, 400, 500, 600, 700]])
        lp = packed(ids).astype(mx.float32)
        lb = bf16(ids).astype(mx.float32)
        rel = float(mx.max(mx.abs(lp - lb)) / (mx.max(mx.abs(lb)) + 1e-6))
        finite = bool(mx.all(mx.isfinite(lp)).item())
        top1 = float(mx.mean((mx.argmax(lp[0], -1) == mx.argmax(lb[0], -1)).astype(mx.float32)).item())
        # packed (quantized_matmul) vs bf16 (dequant @ x) over the SAME int8 codes — kernel parity.
        # Gate on rel: the house qmm bound is 2e-2 (bf16_scale_g64_test); allow a touch more for the
        # 2-layer+head compounding + the rms_norm fast/explicit diff. A real wrong-bits mis-decode
        # would be O(1), not ~1.7%. top-1 is informational only here — on a 2-layer stub with random
        # (non-sentence) tokens the logits aren't peaked, so precision noise flips near-tie argmaxes;
        # the real top-1/ppl arbiter is parity/internlm2_packed_ppl.py on the full model + real prose.
        kernel_ok = rel < 2.5e-2 and finite
        print(f"packed quantized_matmul vs bf16-dequant logits: rel={rel:.4e}  "
              f"top-1 agree={top1*100:.0f}% (info)  finite={finite}")

        ok = bits_ok and kernel_ok
        print(f"\n{'PASS' if ok else 'FAIL'}")
        assert ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # own freshly-created temp scratch, not user data


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
