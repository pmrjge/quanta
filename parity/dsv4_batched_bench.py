"""DSV4-Flash batched (B>1) decode throughput benchmark — per-stream (Design-A) vs batched attention.

HEAVY: loads the resident int4-g64 DSV4-Flash bake (~180 GiB). RUN SOLO — no other model resident
(one-model-at-a-time; an over-subscribed load OOM-reboots the host). Orchestrator/standalone only:

    uv run --with tokenizers python -m parity.dsv4_batched_bench           # full B sweep
    uv run --with tokenizers python -m parity.dsv4_batched_bench 48,64      # only B=48,64

Sweeps ``B in {1,2,4,8,16,32,48,64}`` (override via argv) over identical prompts (uniform KV load) on a single resident
:class:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel`, and for each B times TWO decode paths
on the SAME weights:

  * **looped**  — ``_fused=False``: the Design-A per-stream attention loop (one ``decode_step`` per
    stream, every layer) + batched MoE — the pre-batching reference;
  * **batched** — ``_fused=True`` (the default): the per-stream attention loop collapsed into one
    batched projection + one windowed-sink SDPA across streams (``decode_step_*_batched``), per-stream
    work reduced to the bounded cache append + the window-closing compressor pool.

The **batched/looped speedup** is the attention-batching win; since the batched path still appends KV
per stream (and pools the compressor per stream), its absolute per-step time also bounds how much a
future batched/paged KV store (#153-class) could still buy — measure before building that kernel.

Memory is read with MLX's own counters (``get_active_memory`` / ``get_peak_memory``), NOT ``ru_maxrss``
(which undercounts mmap'd weights + the Metal pool ~10×); ``clear_cache`` + ``reset_peak_memory`` run
between every path/B so a peak is that configuration's true transient, not a cumulative cache high-water.

Geometry: WARMUP prefill = 256 tok (single-stream, parity-correct), GEN = 64 decoded tok/stream
(steady state; 4 warmup steps not timed), EOS disabled (every stream decodes exactly GEN).
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.decode import DSV4Cache

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
WARMUP_PROMPT_LEN = 256           # short seed: decode tok/s is MoE-dominated (~context-independent),
                                  # so a shorter seed barely moves throughput but keeps O(B) seeding cheap
GEN = 64                          # decoded tokens per stream
WARMUP_STEPS = 4                  # steady-state ramp-up (not timed)
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 48, 64)


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _seed_caches(model: DSV4BatchedResidentModel, prompt_ids: mx.array, n_streams: int
                 ) -> tuple[list[DSV4Cache], list[int]]:
    """Prefill ``n_streams`` fresh caches with the same prompt (single-stream prefill per cache — the
    parity-correct path, identical regardless of ``_fused``). Returns caches + per-stream offsets."""
    caches: list[DSV4Cache] = []
    for _ in range(n_streams):
        cache = model.make_cache()
        logits = model.prefill(prompt_ids, cache)
        mx.eval(logits)
        caches.append(cache)
    return caches, [c.offset for c in caches]


def _time_path(model: DSV4BatchedResidentModel, prompt_ids: mx.array, B: int, fused: bool) -> dict:
    """Time GEN steady-state decode steps at batch B on the chosen path (``fused`` ⇒ batched attention,
    else the per-stream Design-A loop). Fresh caches per call so the two paths never share state."""
    model._fused = fused
    mx.clear_cache()
    mx.reset_peak_memory()
    caches, offsets = _seed_caches(model, prompt_ids, B)
    cur = [int(prompt_ids[-1].item())] * B
    streams_ids = [mx.array([cur[b]]) for b in range(B)]

    for _ in range(WARMUP_STEPS):                # JIT + MoE dispatch warm (not timed)
        out = model.step_batch(streams_ids, caches, offsets)
        mx.eval(out)
        cur = [int(mx.argmax(out[b][0, -1]).item()) for b in range(B)]
        streams_ids = [mx.array([cur[b]]) for b in range(B)]
        offsets = [o + 1 for o in offsets]

    t0 = time.perf_counter()
    for _ in range(GEN):
        out = model.step_batch(streams_ids, caches, offsets)
        mx.eval(out)
        cur = [int(mx.argmax(out[b][0, -1]).item()) for b in range(B)]
        streams_ids = [mx.array([cur[b]]) for b in range(B)]
        offsets = [o + 1 for o in offsets]
    dt = time.perf_counter() - t0

    return {"per_stream": GEN / dt, "aggregate": B * GEN / dt,
            "active_gib": _gib(mx.get_active_memory()), "peak_gib": _gib(mx.get_peak_memory())}


def run(batch_sizes: tuple[int, ...] = BATCH_SIZES) -> None:
    # Pin the resident weight set (DSV4-Flash int4-g64 ≈ 180 GiB — keep MLX from paging it).
    mx.set_wired_limit(int(220 * 1024 ** 3))
    model = DSV4BatchedResidentModel(ART, max_batch=max(batch_sizes), packed_experts=True)

    bos = model.cfg.bos_token_id
    prompt_ids = mx.array([bos] + list(range(1, WARMUP_PROMPT_LEN)))

    print(f"\n=== DSV4-Flash int4-g64 batched decode (prompt {WARMUP_PROMPT_LEN} tok, "
          f"{GEN} gen/stream): looped (per-stream attn) vs batched (fused attn) ===")
    print(f"{'B':>4}  {'looped per/agg':>22}  {'batched per/agg':>22}  {'bat/loop':>9}  "
          f"{'act/peak GiB':>16}")
    for B in batch_sizes:
        lp = _time_path(model, prompt_ids, B, fused=False)
        ba = _time_path(model, prompt_ids, B, fused=True)
        ratio = ba["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        print(f"{B:>4}  {lp['per_stream']:>9.2f} /{lp['aggregate']:>11.2f}  "
              f"{ba['per_stream']:>9.2f} /{ba['aggregate']:>11.2f}  {ratio:>8.2f}x  "
              f"{ba['active_gib']:>7.1f}/{ba['peak_gib']:>7.1f}")


if __name__ == "__main__":
    import sys

    bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
    run(bs)
