"""Nemotron-H resident-runtime e2e parity gate (#39).

Runs the RAM-resident quantized runtime (packed weights via gather_qmm / quantized_matmul) on
the same teacher-forced prose as the references, and checks it lands on the dequantized reference
(3.327) — confirming the gather_qmm/QuantizedLinear decode path is output-equivalent to dequant.
Also vs the bf16 reference (3.379). Same PROSE / 109 tokens / fp32-comparable head.

    uv run --with tokenizers python -m parity.nemotron_resident_ppl
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_ppl import PROSE
from quanta.nemotron.runtime import NemotronResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
BF16_PPL, DEQUANT_PPL = 3.379, 3.327  # post-U1-fix re-measure (was 5.981/5.799 under the buggy full-width norm)


def run() -> None:
    mx.set_wired_limit(int(120 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    load_min = (time.perf_counter() - t0) / 60
    ids = mx.array(tok.encode(PROSE, add_bos=False)[:192])
    t1 = time.perf_counter()
    logits, _, _ = model(ids)
    lg = logits[0, :-1].astype(mx.float32)
    targets = ids[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    print(f"\n=== Nemotron-H RESIDENT quantized runtime (tokens={ids.shape[0]}) ===")
    print(f"perplexity           : {ppl:.3f}   (dequant ref {DEQUANT_PPL:.3f} Δ {ppl - DEQUANT_PPL:+.3f}; "
          f"bf16 {BF16_PPL:.3f} Δ {100 * (ppl - BF16_PPL) / BF16_PPL:+.1f}%)")
    print(f"top-1 next-token acc : {acc:.3f}")
    print(f"load {load_min:.1f} min | forward {time.perf_counter() - t1:.1f}s")
    print("PASS" if abs(ppl - DEQUANT_PPL) / DEQUANT_PPL < 0.02 else "FAIL (resident != dequant ref)")


if __name__ == "__main__":
    run()
