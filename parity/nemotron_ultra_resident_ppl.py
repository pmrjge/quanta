"""Nemotron-Ultra U4 / M2 — full-resident e2e perplexity gate.

Loads the whole **306 GiB int4-RTN artifact RAM-resident** (:class:`NemotronResidentModel`: routed
experts as packed int4 via ``mx.gather_qmm``, dense always-on linears as int8 ``nn.QuantizedLinear``,
SSM core / norms / router / embeddings / head as bf16) and teacher-forces the **same held-out
1024-token corpus** as U3, confirming the resident forward lands on the U3 streamed-dequant RTN
reference perplexity (**3.845**).

This is the e2e closer for the packed-int4 + gather_qmm stream. M1 (``nemotron_ultra_qmoe_test``)
gated only the MoE component numerically at one layer; M2 runs the *whole* 108-layer resident model
and so additionally covers the dense **mamba/attn int8 ``QuantizedLinear`` wiring** end-to-end. The
metric (``_ppl_acc``) and corpus (``LONG_PROSE`` / ``N_TOK``) are imported verbatim from
``parity.nemotron_ultra_ppl`` so the number is directly comparable to the U3 arbiter — the only path
difference under test is resident-packed-forward (bf16 head) vs streamed-dequant-forward (fp32 head),
which must agree to the 2% gate (the Super-120B sibling ``nemotron_resident_ppl`` validates the same
bf16-vs-fp32-head equivalence).

One model resident at a time — **run solo** (~306 GiB wired, under the 490.4 GiB ceiling).

    uv run --with tokenizers python -m parity.nemotron_ultra_resident_ppl
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_ultra_ppl import LONG_PROSE, N_TOK, _ppl_acc
from quanta.nemotron.runtime import NemotronResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64"
RTN_PPL = 3.845  # U3 streamed-dequant RTN reference (parity.nemotron_ultra_ppl)


def run() -> None:
    # Pin the 306 GiB resident weight set; 400 GiB leaves headroom for forward activations while
    # staying under the 490.4 GiB working-set ceiling.
    mx.set_wired_limit(int(400 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    load_min = (time.perf_counter() - t0) / 60

    ids = mx.array(tok.encode(LONG_PROSE, add_bos=False)[:N_TOK])
    targets = ids[1:]
    t1 = time.perf_counter()
    logits, _, _ = model(ids)
    ppl, acc, _ = _ppl_acc(logits[0], targets)  # logits [1, t, vocab] -> [t, vocab]
    fwd_s = time.perf_counter() - t1

    delta = 100 * (ppl - RTN_PPL) / RTN_PPL
    print(f"\n=== Nemotron-Ultra RESIDENT quantized runtime  (tokens={ids.shape[0]}) ===")
    print(f"perplexity           : {ppl:.3f}   (U3 RTN ref {RTN_PPL:.3f}  Δ {ppl - RTN_PPL:+.3f} / {delta:+.1f}%)")
    print(f"top-1 next-token acc : {acc:.3f}")
    print(f"load {load_min:.1f} min | forward {fwd_s:.1f}s")
    print("PASS" if abs(ppl - RTN_PPL) / RTN_PPL < 0.02 else "FAIL (resident != U3 RTN ref)")


if __name__ == "__main__":
    run()
