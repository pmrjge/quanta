"""DSV4-Flash batched (B>1) decode throughput benchmark — per-stream (Design-A) vs batched attention.

HEAVY: loads the resident int4-g64 DSV4-Flash bake (~180 GiB). RUN SOLO — no other model resident
(one-model-at-a-time; an over-subscribed load OOM-reboots the host). Orchestrator/standalone only:

    uv run --with tokenizers python -m parity.dsv4_batched_bench           # full B sweep
    uv run --with tokenizers python -m parity.dsv4_batched_bench 48,64      # only B=48,64

Sweeps ``B in {1,2,4,8,16,32,48,64}`` (override via argv) over identical prompts (uniform KV load) on a single resident
:class:`quanta.dsv4.batched_runtime.DSV4BatchedResidentModel`, and for each B times THREE decode paths
on the SAME weights. One resident model serves all three because the batched steppers dispatch on the
cache TYPE: ``_fused`` selects attention batching, the cache type selects the KV store (#18 M4).

  * **looped**  — ``_fused=False`` + a discrete :class:`~quanta.dsv4.decode.DSV4Cache`: the Design-A
    per-stream attention loop (one ``decode_step`` per stream, every layer) + batched MoE — the
    pre-batching reference;
  * **batched** — ``_fused=True`` + a discrete ``DSV4Cache``: the per-stream attention loop collapsed
    into one batched projection + one windowed-sink SDPA across streams (``decode_step_*_batched``),
    but the KV store is still the per-stream ``_LayerCache`` — a per-stream quantize+concat ``append``
    plus a ``_pad_stack`` readback EVERY step (the pre-#18 batched path);
  * **arena** (#18) — ``_fused=True`` + an ``_ArenaCacheHandle`` (``make_cache()``, the M4 default):
    the SAME fused attention, but the per-stream KV-update IO loop is replaced by ONE scatter write +
    ONE gather read against the persistent ``max_batch``-sized batched KV arena.

Two ratios isolate the two wins on identical weights. **batched/looped** is the attention-batching win
(this bench's original measurement). **arena/batched** is THE #18 number: the per-stream KV-update IO
loop-kill in isolation — both paths share the fused attention, only the KV store differs.
**arena/looped** is the total batched-serving win. The #18 win should grow with B and with the
compressed-layer share (more per-stream pool / append / ``_pad_stack`` work removed).

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


def _seed_caches(model: DSV4BatchedResidentModel, prompt_ids: mx.array, n_streams: int,
                 *, arena: bool) -> tuple[list, list[int]]:
    """Prefill ``n_streams`` fresh caches with the same prompt (single-stream prefill per cache — the
    parity-correct path, identical regardless of ``_fused``). ``arena`` picks the cache type: an
    ``_ArenaCacheHandle`` via ``make_cache()`` (#18, leases one arena row — the default serving path)
    when True, else a discrete :class:`DSV4Cache` (the per-stream ``_LayerCache`` path the batched
    steppers dispatch to on cache type). Returns caches + per-stream offsets."""
    caches: list = []
    for _ in range(n_streams):
        cache = model.make_cache() if arena else DSV4Cache(model.num_layers)
        logits = model.prefill(prompt_ids, cache)
        mx.eval(logits)
        caches.append(cache)
    return caches, [c.offset for c in caches]


def _time_path(model: DSV4BatchedResidentModel, prompt_ids: mx.array, B: int,
               *, fused: bool, arena: bool) -> dict:
    """Time GEN steady-state decode steps at batch B on one of the three paths:

      * ``fused=False, arena=False`` — looped: Design-A per-stream attention loop (discrete cache);
      * ``fused=True,  arena=False`` — batched: fused attention, per-stream ``_LayerCache`` KV update;
      * ``fused=True,  arena=True``  — arena (#18): fused attention, ONE scatter + ONE gather KV.

    Fresh caches per call so the paths never share state; arena handles are freed (the leased row
    returned to the free-list) before returning, so the next path / B can re-lease up to ``max_batch``.
    """
    model._fused = fused
    mx.clear_cache()
    mx.reset_peak_memory()
    caches, offsets = _seed_caches(model, prompt_ids, B, arena=arena)
    try:
        cur = [int(prompt_ids[-1].item())] * B
        streams_ids = [mx.array([cur[b]]) for b in range(B)]
        # Stream-0 greedy token trace (warmup+GEN). The three paths share weights + prompt, so a
        # correct arena (int8 latent on the real head_dim=128 — exercised live here for the first time)
        # must reproduce the per-stream loop's tokens exactly (the B>=2 greedy-exact bar). run() asserts
        # looped == batched == arena, turning the throughput bench into a real-model arena correctness gate.
        toks0: list[int] = []

        for _ in range(WARMUP_STEPS):                # JIT + MoE dispatch warm (not timed)
            out = model.step_batch(streams_ids, caches, offsets)
            mx.eval(out)
            cur = [int(mx.argmax(out[b][0, -1]).item()) for b in range(B)]
            toks0.append(cur[0])
            streams_ids = [mx.array([cur[b]]) for b in range(B)]
            offsets = [o + 1 for o in offsets]

        t0 = time.perf_counter()
        for _ in range(GEN):
            out = model.step_batch(streams_ids, caches, offsets)
            mx.eval(out)
            cur = [int(mx.argmax(out[b][0, -1]).item()) for b in range(B)]
            toks0.append(cur[0])
            streams_ids = [mx.array([cur[b]]) for b in range(B)]
            offsets = [o + 1 for o in offsets]
        dt = time.perf_counter() - t0

        return {"per_stream": GEN / dt, "aggregate": B * GEN / dt, "toks": toks0,
                "active_gib": _gib(mx.get_active_memory()), "peak_gib": _gib(mx.get_peak_memory())}
    finally:
        if arena:
            for c in caches:
                model.free_cache(c)            # return each leased arena row (#18) before the next path


def run(batch_sizes: tuple[int, ...] = BATCH_SIZES) -> None:
    # Pin the resident weight set (DSV4-Flash int4-g64 ≈ 180 GiB — keep MLX from paging it).
    mx.set_wired_limit(int(220 * 1024 ** 3))
    # kv_arena defaults ON (since #18 M4) ⇒ make_cache() returns an _ArenaCacheHandle for the arena
    # path; the looped/batched paths build a discrete DSV4Cache directly (dispatch keys off the cache
    # type, not the flag — so the one resident model serves all three paths).
    model = DSV4BatchedResidentModel(ART, max_batch=max(batch_sizes), packed_experts=True)

    bos = model.cfg.bos_token_id
    prompt_ids = mx.array([bos] + list(range(1, WARMUP_PROMPT_LEN)))

    print(f"\n=== DSV4-Flash int4-g64 batched decode (prompt {WARMUP_PROMPT_LEN} tok, {GEN} gen/stream): "
          f"looped (per-stream attn) vs batched (fused attn, per-stream KV) vs arena (#18: fused attn, "
          f"scatter/gather KV) ===")
    print("aggregate tok/s (per-stream = aggregate / B). bat/loop = attn-batching win; "
          "arena/bat = #18 KV-loop-kill; arena/loop = total. GiB = arena-path active/peak. "
          "tok = looped==batched==arena greedy-exact.")
    print(f"{'B':>4}  {'looped':>9}  {'batched':>9}  {'arena':>9}  "
          f"{'bat/loop':>8}  {'arena/bat':>9}  {'arena/loop':>10}  {'arena GiB a/p':>15}  {'tok':>4}")
    all_tok_ok = True
    for B in batch_sizes:
        lp = _time_path(model, prompt_ids, B, fused=False, arena=False)  # looped: Design-A per-stream
        ba = _time_path(model, prompt_ids, B, fused=True, arena=False)   # batched: fused attn, per-stream KV
        ar = _time_path(model, prompt_ids, B, fused=True, arena=True)    # arena (#18): scatter/gather KV
        bl = ba["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        ab = ar["aggregate"] / ba["aggregate"] if ba["aggregate"] else float("nan")
        al = ar["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        tok_ok = lp["toks"] == ba["toks"] == ar["toks"]
        all_tok_ok = all_tok_ok and tok_ok
        print(f"{B:>4}  {lp['aggregate']:>9.1f}  {ba['aggregate']:>9.1f}  {ar['aggregate']:>9.1f}  "
              f"{bl:>7.2f}x  {ab:>8.2f}x  {al:>9.2f}x  "
              f"{ar['active_gib']:>6.1f}/{ar['peak_gib']:>6.1f}  {'ok' if tok_ok else 'DIFF':>4}")
        if not tok_ok:
            # First divergence vs the looped reference — surface it loudly (rule 6), don't bury a DIFF.
            ref = lp["toks"]
            for nm, got in (("batched", ba["toks"]), ("arena", ar["toks"])):
                d = next((k for k in range(min(len(ref), len(got))) if ref[k] != got[k]), None)
                if d is not None:
                    print(f"    [DIFF] {nm} diverges from looped at step {d}: "
                          f"looped={ref[d]} {nm}={got[d]}")

    print("PASS — arena greedy-exact vs per-stream loop on the real model"
          if all_tok_ok else "FAIL — arena diverged from the per-stream loop (see [DIFF] above)")
    if not all_tok_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys

    bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
    run(bs)
