"""DSV4-Flash batched (B>1) decode throughput benchmark — concurrent agent traces.

RUN AFTER #145 lands; orchestrator runs this — do NOT execute from an agent.

    uv run --with tokenizers python -m parity.dsv4_batched_bench

Sweeps ``B in {1, 2, 4, 8, 16, 32}`` over identical prompts (so KV cache load is uniform) on the
resident :class:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel`. For each B:

  1. Build B fresh decode caches and prefill the same warmup prompt into each (the inner
     :meth:`DSV4BatchedResidentModel.prefill` — parity-correct single-stream prefill);
  2. Decode ``GEN`` tokens per stream (lock-step), timing the steady-state ``step_batch`` calls
     after a short warmup;
  3. Report per-stream tok/s, aggregate tok/s (= B × per-stream), and peak RSS.

Aggregate scaling is the agentic-loop deployment lever: every B-fold amortization of the
always-on shared expert + the routed expert weight reads (more slots ⇒ more amortized
``mx.gather_mm`` per layer per step) raises throughput sub-linearly until ~B=32, when all 256
routed experts are saturated (every expert receives at least one slot per step).

Geometry / artifact:
  * ART = the resident int4-g64 DSV4-Flash bake (~169 GB);
  * WARMUP prefill = 1024 tokens (long enough that prompt-side bandwidth is non-trivial);
  * GEN = 64 decoded tokens per stream (steady state — 4 warmup steps not timed);
  * EOS is disabled so every stream decodes exactly GEN tokens (apples-to-apples timing).
"""

from __future__ import annotations

import resource
import sys
import time

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.decode import DSV4Cache

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
WARMUP_PROMPT_LEN = 1024
GEN = 64                          # decoded tokens per stream
WARMUP_STEPS = 4                  # steady-state ramp-up (not timed)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def _peak_rss_gib() -> float:
    """Peak resident set in GiB (Darwin/Linux portable — ru_maxrss is bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: bytes; Linux: KiB. We deploy on darwin (M3 Ultra) — but guard both.
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


def _seed_caches(model: DSV4BatchedResidentModel, prompt_ids: mx.array, n_streams: int
                 ) -> tuple[list[DSV4Cache], list[int]]:
    """Prefill ``n_streams`` fresh caches with the same prompt (single-stream prefill per cache —
    the parity-correct path). Returns the caches and the per-stream offsets ready for decode."""
    caches: list[DSV4Cache] = []
    for _ in range(n_streams):
        cache = model.make_cache()
        logits = model.prefill(prompt_ids, cache)
        mx.eval(logits)
        caches.append(cache)
    offsets = [c.offset for c in caches]
    return caches, offsets


def _bench_one(model: DSV4BatchedResidentModel, prompt_ids: mx.array, B: int) -> dict:
    """Time GEN steady-state decode steps at batch size B; return per-stream + aggregate tok/s."""
    caches, offsets = _seed_caches(model, prompt_ids, B)
    # Seed the first decode token per stream — greedy argmax of the last prefill row would normally
    # come from each stream's prefill; for a uniform-load bench we just feed a known token id.
    cur = [int(prompt_ids[-1].item())] * B
    streams_ids = [mx.array([cur[b]]) for b in range(B)]

    # warmup steps (JIT, MoE dispatch warm) — not timed
    for _ in range(WARMUP_STEPS):
        out = model.step_batch(streams_ids, caches, offsets)
        mx.eval(out)
        # greedy next-token from each stream's logits (apples-to-apples uniform decode)
        cur = [int(mx.argmax(out[b][0, -1]).item()) for b in range(B)]
        streams_ids = [mx.array([cur[b]]) for b in range(B)]
        offsets = [o + 1 for o in offsets]

    # timed loop — GEN steady-state decode steps per stream
    t0 = time.perf_counter()
    for _ in range(GEN):
        out = model.step_batch(streams_ids, caches, offsets)
        mx.eval(out)
        cur = [int(mx.argmax(out[b][0, -1]).item()) for b in range(B)]
        streams_ids = [mx.array([cur[b]]) for b in range(B)]
        offsets = [o + 1 for o in offsets]
    dt = time.perf_counter() - t0

    per_stream = GEN / dt                        # tok/s a single stream "sees" in this batch
    aggregate = B * GEN / dt                     # tok/s total wall throughput
    return {"B": B, "per_stream": per_stream, "aggregate": aggregate, "wall_s": dt,
            "peak_rss_gib": _peak_rss_gib()}


def run() -> None:
    # Pin the resident weight set (DSV4-Flash int4-g64 ≈ 169 GiB — keep MLX from paging it).
    mx.set_wired_limit(int(200 * 1024 ** 3))
    model = DSV4BatchedResidentModel(ART, max_batch=max(BATCH_SIZES), packed_experts=True)

    # Build the warmup prompt once (token-id padding is fine here — uniform across B).
    bos = model.cfg.bos_token_id
    prompt_ids = mx.array([bos] + list(range(1, WARMUP_PROMPT_LEN)))

    print(f"\n=== DSV4-Flash int4-g64 batched decode (prompt {WARMUP_PROMPT_LEN} tok, "
          f"{GEN} gen/stream) ===")
    print(f"{'B':>4}  {'per-stream tok/s':>18}  {'aggregate tok/s':>18}  "
          f"{'wall s':>10}  {'peak RSS GiB':>14}")
    for B in BATCH_SIZES:
        r = _bench_one(model, prompt_ids, B)
        print(f"{r['B']:>4}  {r['per_stream']:>18.2f}  {r['aggregate']:>18.2f}  "
              f"{r['wall_s']:>10.2f}  {r['peak_rss_gib']:>14.2f}")


if __name__ == "__main__":
    run()
