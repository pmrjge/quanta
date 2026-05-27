"""Nemotron-H batched-serving throughput benchmark (#147 perf — orchestrator-run only).

Sweeps ``B ∈ {1, 2, 4, 8, 16, 32}`` against the resident int4-g64 artifact; warmup 1024 prefill
tokens then 64 decode steps per stream. Reports per-stream decode tok/s + aggregate decode tok/s
+ the B/B=1 speedup factor (decode-only — prefill is measured separately for visibility). The
expected curve is bandwidth-amortization: per-stream tok/s falls slowly with B (more SDPA +
slightly bigger MoE call), aggregate tok/s grows roughly linearly until the MoE gemm becomes
compute-bound or the per-stream Mamba/Attention loop overhead catches up to the MoE call.
Target: B=32 → ~10× aggregate vs B=1 (top-22 over 512 experts gives more headroom than DSV4
top-6/256).

Decode is timed via the lower-level :meth:`step_batch` after per-stream :meth:`prefill` so the
prefill cost (single-stream chunked SSD per slot) does NOT pollute the decode tok/s — mirrors
:mod:`parity.nemotron_decode_bench`'s ``prefill_s + steady-state decode`` decomposition.

This file is **WRITE-ONLY** for the agent — it loads the real artifact and exercises the GPU at
scale. Do NOT execute from an agent. The orchestrator runs it after #147 lands:

    uv run --with tokenizers python -m parity.nemotron_batched_bench [n_decode]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
PROMPT = (
    "Write a Python function that returns the n-th Fibonacci number using memoization, "
    "and explain it. Make the explanation precise and pedagogical, building up from the "
    "naive recursion to the memoized one with concrete intermediate state at each step."
)
WARMUP_PROMPT_REPEATS = 8           # repeat the prompt to reach ~1024-token warmup as specified
WARMUP_DECODE_STEPS = 8             # decode-loop warmup (JIT/compile + cache warm) before timing
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
N_DECODE_DEFAULT = 64


def _build_long_prompt_ids(tok: NemotronTokenizer, repeats: int = WARMUP_PROMPT_REPEATS) -> list[int]:
    """Concatenate the prompt ``repeats`` times to reach a longer warmup context. Matches the
    spec's ``warmup=1024`` target while keeping the structure of the existing decode bench."""
    base = tok.encode(PROMPT, add_bos=False)
    out: list[int] = []
    for _ in range(repeats):
        out.extend(base)
    return out


def _time_batched_decode(
    model: NemotronBatchedResidentModel,
    prompt_ids: list[int],
    *,
    batch: int,
    n_decode: int,
) -> tuple[float, float, int]:
    """Prefill ``batch`` identical streams with ``prompt_ids``, then time ``n_decode`` decode steps
    across all streams (with a small warmup before the timed window). Returns
    ``(prefill_s, decode_s, total_decode_tokens)``.

    Decode timing isolates the MoE bandwidth-amortization win from the per-prompt prefill
    chunked-SSD path (which is single-stream and would otherwise inflate the reported tok/s for
    small ``n_decode``). Matches the ``prefill_s`` + ``steady-state decode tok/s`` decomposition
    of :mod:`parity.nemotron_decode_bench`."""
    # --- per-stream prefill (single-stream, one at a time — same as continuous-batching admit) -
    t0 = time.perf_counter()
    states = []
    initial_ids: list[int] = []
    prompt_arr = mx.array(prompt_ids)
    for _ in range(batch):
        state = model.make_stream_state()
        logits = model.prefill(prompt_arr, state)
        mx.eval(logits)
        initial_ids.append(int(mx.argmax(logits[0, -1]).item()))
        states.append(state)
    prefill_s = time.perf_counter() - t0

    # --- decode warmup (let JIT/compile + cache warm before we time) ----------
    next_ids = list(initial_ids)
    for _ in range(WARMUP_DECODE_STEPS):
        token_ids = [mx.array([t]) for t in next_ids]
        logits_list = model.step_batch(token_ids, states)
        next_ids = [int(mx.argmax(lg[0, -1]).item()) for lg in logits_list]

    # --- timed decode window --------------------------------------------------
    t_start = time.perf_counter()
    total = 0
    for _ in range(n_decode):
        token_ids = [mx.array([t]) for t in next_ids]
        logits_list = model.step_batch(token_ids, states)
        next_ids = [int(mx.argmax(lg[0, -1]).item()) for lg in logits_list]
        total += batch
    decode_s = time.perf_counter() - t_start
    return prefill_s, decode_s, total


def run() -> None:
    n_decode = int(sys.argv[1]) if len(sys.argv) > 1 else N_DECODE_DEFAULT
    mx.set_wired_limit(int(120 * 1024**3))
    tok = NemotronTokenizer(ART)
    prompt_ids = _build_long_prompt_ids(tok)
    print(f"\n=== Nemotron-H int4-g64 batched serving ({len(prompt_ids)}-tok prompt, "
          f"{n_decode} decode tok/stream) ===")

    # max_batch sized to the biggest B in the sweep so the wrapper does not reject anything
    model = NemotronBatchedResidentModel(ART, max_batch=max(BATCH_SIZES))

    base_decode_tps: float | None = None
    print(f"{'B':>4} | {'prefill_s':>10} | {'decode_s':>9} | {'per-stream tok/s':>17} | "
          f"{'aggregate tok/s':>16} | {'speedup vs B=1':>14}")
    print("-" * 92)
    for b in BATCH_SIZES:
        prefill_s, decode_s, total = _time_batched_decode(model, prompt_ids,
                                                          batch=b, n_decode=n_decode)
        per_stream = (total / b) / decode_s
        aggregate = total / decode_s
        if base_decode_tps is None:
            base_decode_tps = aggregate
        speedup = aggregate / base_decode_tps if base_decode_tps else 0.0
        print(f"{b:>4} | {prefill_s:>10.2f} | {decode_s:>9.2f} | {per_stream:>17.1f} | "
              f"{aggregate:>16.1f} | {speedup:>13.2f}x")


if __name__ == "__main__":
    run()
