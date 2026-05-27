"""Throughput sweep for Qwen3.5 batched (B>1) decode — REAL model, RUN AFTER #147 lands.

WRITE-ONLY: this file is the orchestrator-driven bench; the agent that builds it MUST NOT run it
(loads the 398 GB baked Qwen3.5 artifact). The parity prerequisite is
``parity/qwen35_batched_test.py`` (model-free, fast, RUN AS PART OF THE GATE) — once that passes
on a tiny random-weights config, the throughput sweep is safe to run against the real artifact.

What the sweep measures:

* For each ``B in [1, 2, 4, 8, 16, 32]``:
    - Warm the runtime + caches: prefill ``warmup=1024`` tokens of a synthetic prompt PER STREAM
      (so the GDN recurrent state + GQA KV cache are populated to a realistic agentic-loop offset).
    - Run ``gen=64`` decode steps via :meth:`Qwen35BatchedResidentModel.step_batch` driving all B
      streams in lockstep (``time.perf_counter`` around ``mx.eval`` of the per-stream logits).
    - Report (i) per-step wall ms for B streams (one step), (ii) aggregate tokens/s = B*gen / total,
      (iii) per-stream tokens/s = aggregate / B, (iv) speedup vs B=1 aggregate.

The target shape (from the bandwidth model — single-stream decode is dominated by reading the
routed-expert / shared-expert weights ONCE per token):

  * aggregate tokens/s scales ~linearly with B until compute, not bandwidth, becomes the bottleneck;
  * per-stream tokens/s stays approximately flat (the cost of the shared weight read is amortized,
    not multiplied);
  * B=32 → ~10× aggregate vs B=1 (the design target).

Invocation (after the parity gate passes):

    uv run --with numpy python -m parity.qwen35_batched_bench \\
        --artifact ~/models/Qwen3.5-397B-A17B-quanta_int4 \\
        --warmup 1024 --gen 64 --batches 1,2,4,8,16,32

This script is DELIBERATELY a CLI tool (not a pytest gate) so the orchestrator can sweep without
the test harness loading the model on every run; the parity gate covers correctness, this sweep
covers the throughput claim.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx


def _parse_batch_list(s: str) -> list[int]:
    bs = [int(b.strip()) for b in s.split(",") if b.strip()]
    if not bs or any(b < 1 for b in bs):
        raise ValueError(f"--batches must be a comma-separated list of positive ints, got {s!r}")
    return bs


def _synth_prompt(length: int, vocab: int, stream_seed: int) -> list[int]:
    """Deterministic synthetic prompt token ids per stream (no tokenizer dependency for the bench).

    Uses ``mx.random`` with a per-stream seed so each stream's prompt differs (so the bench exercises
    realistic per-stream cache divergence, not the trivial identical-prompt path)."""
    mx.random.seed(stream_seed)
    ids = mx.random.randint(2, vocab - 1, (length,))   # avoid the special token range
    return [int(x.item()) for x in ids]


def _prefill_streams(model, prompts: list[list[int]]) -> list:
    """Warm all B streams' caches by prefilling each prompt once (single-stream per Design A —
    full-attention prefill needs a common per-stream offset / mask). Returns per-stream caches."""
    caches = model.make_batch_caches(len(prompts))
    for i, p in enumerate(prompts):
        model.prefill(p, caches[i])
        mx.eval(caches[i].offset)   # force the cache write to land before the next stream's prefill
    return caches


def _bench_one_batch(model, B: int, warmup: int, gen: int, vocab: int) -> dict:
    """Measure one (B, warmup, gen) point: returns timings + tokens/s for B streams.

    The decode loop is the steady-state agentic-loop hot path — per step, B streams each feed their
    last-emitted token (the bench feeds the argmax forward; sampling cost is excluded from the
    throughput number since it's not the bandwidth-bound work)."""
    prompts = [_synth_prompt(warmup, vocab, stream_seed=1000 + i) for i in range(B)]
    caches = _prefill_streams(model, prompts)
    offsets = [c.offset for c in caches]
    # warm one step (compile + JIT) so the timed window measures steady-state
    toks = [prompts[i][-1] for i in range(B)]
    per_stream = model.step_batch(toks, caches, offsets)
    mx.eval(per_stream)
    toks = [int(mx.argmax(lg[0, -1]).item()) for lg in per_stream]
    offsets = [o + 1 for o in offsets]

    # timed decode loop: B streams × gen steps
    t0 = time.perf_counter()
    for _ in range(gen):
        per_stream = model.step_batch(toks, caches, offsets)
        mx.eval(per_stream)
        toks = [int(mx.argmax(lg[0, -1]).item()) for lg in per_stream]
        offsets = [o + 1 for o in offsets]
    dt = time.perf_counter() - t0

    return {
        "B": B,
        "warmup": warmup,
        "gen": gen,
        "wall_s": dt,
        "step_ms": dt / gen * 1000.0,
        "aggregate_tps": (B * gen) / dt,
        "per_stream_tps": gen / dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3.5 batched-decode throughput sweep")
    ap.add_argument("--artifact", required=True, type=Path,
                    help="path to a baked Qwen3.5 quanta artifact (e.g. ~/models/Qwen3.5-...-quanta_int4)")
    ap.add_argument("--warmup", type=int, default=1024,
                    help="per-stream prompt length to prefill before timing (default 1024)")
    ap.add_argument("--gen", type=int, default=64,
                    help="decode steps per stream in the timed loop (default 64)")
    ap.add_argument("--batches", type=_parse_batch_list, default=[1, 2, 4, 8, 16, 32],
                    help="comma-separated batch sizes (default 1,2,4,8,16,32)")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="batched runtime's max_batch (defaults to max(--batches))")
    args = ap.parse_args()

    max_batch = args.max_batch if args.max_batch is not None else max(args.batches)
    if max_batch < max(args.batches):
        raise SystemExit(f"--max-batch={max_batch} < max(--batches)={max(args.batches)}")

    # lazy import: the batched runtime touches the artifact loader; we only want to pay that cost
    # AFTER the CLI args validate.
    from quanta.qwen35.batched_runtime import Qwen35BatchedResidentModel

    print(f"loading Qwen3.5 batched runtime from {args.artifact}  (max_batch={max_batch}) ...")
    model = Qwen35BatchedResidentModel(args.artifact, max_batch=max_batch)
    vocab = model.cfg.vocab_size
    print(f"  cfg: vocab={vocab} hidden={model.cfg.hidden_size} layers={model.num_layers} "
          f"experts={model.cfg.num_experts} top_k={model.cfg.num_experts_per_tok}")

    print("\n=== Qwen3.5 batched-decode throughput sweep ===")
    print(f"warmup={args.warmup}  gen={args.gen}  batches={args.batches}")
    print(f"{'B':>4}  {'step_ms':>10}  {'agg_tps':>10}  {'per_s_tps':>10}  {'speedup_vs_B1':>14}")
    base = None
    for B in args.batches:
        r = _bench_one_batch(model, B, args.warmup, args.gen, vocab)
        if base is None:
            base = r["aggregate_tps"]
        speedup = r["aggregate_tps"] / base
        print(f"{B:>4}  {r['step_ms']:>10.2f}  {r['aggregate_tps']:>10.1f}  "
              f"{r['per_stream_tps']:>10.2f}  {speedup:>14.2f}x")


if __name__ == "__main__":
    main()
