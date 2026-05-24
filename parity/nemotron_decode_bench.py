"""Nemotron-H resident decode throughput benchmark (#41 perf).

Prefill a short prompt, then time greedy decode steps on the RAM-resident int4-g64 runtime.
Decode is op-launch bound (not bandwidth): the compiled mixer path (default) fuses each
mamba/moe layer → ~35 tok/s (vs ~30 eager). The int4-g64 *bandwidth* ceiling is ~73 tok/s —
the always-on int8 dense (shared expert + mamba in/out-proj + fc) dominates bytes/token, not the
int4 experts; closing the gap to the ceiling needs a fused mamba SSD kernel. Reports prefill
latency + steady-state decode tok/s (after warmup).

    uv run --with tokenizers python -m parity.nemotron_decode_bench [n_decode]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.nemotron.generate import attn_caches
from quanta.nemotron.runtime import NemotronResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
PROMPT = "Write a Python function that returns the n-th Fibonacci number using memoization, and explain it."
WARMUP = 8


def run() -> None:
    n_decode = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    mx.set_wired_limit(int(120 * 1024**3))
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    ids = tok.encode(PROMPT, add_bos=False)

    caches = attn_caches(model)
    t0 = time.perf_counter()
    logits, ssm, conv = model(mx.array(ids), caches=caches)  # prefill
    cur = int(mx.argmax(logits[0, -1]).item())
    prefill_s = time.perf_counter() - t0

    steps, t_start = 0, None
    for i in range(WARMUP + n_decode):
        logits, ssm, conv = model(mx.array([cur]), caches=caches, ssm=ssm, conv=conv)
        cur = int(mx.argmax(logits[0, -1]).item())
        if i == WARMUP - 1:
            t_start = time.perf_counter()  # start timing after warmup (JIT/cache warm)
        elif i >= WARMUP:
            steps += 1
    dt = time.perf_counter() - t_start
    print(f"\n=== Nemotron-H int4-g64 resident decode ({len(ids)}-tok prompt) ===")
    print(f"prefill              : {prefill_s:.2f}s ({len(ids)} tok, {len(ids) / prefill_s:.0f} tok/s)")
    print(f"decode               : {steps} steps in {dt:.2f}s → {steps / dt:.1f} tok/s")


if __name__ == "__main__":
    run()
