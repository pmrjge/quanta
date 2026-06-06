"""Nemotron-Ultra U4 Stream-A follow-on — decode batch-scaling past B=32 to the memory ceiling.

The U4/decode-economics run (``0de52a9``) measured native multi-stream decode scaling **4.61x agg @
B=32** (peak 367 GiB) and named the one residual Stream-A lever: *"push B>32 / a batched-SSD-step tune"*.
This is the measure-first half — push the native (form-2 persistent ``BatchedMambaState``, the prod
serving path) decode sweep toward the **490.4 GiB** working-set ceiling to answer two serving questions:
(a) where does aggregate tok/s plateau, and (b) what is the max concurrent streams ``B`` the Ultra
int4-RTN backbone can serve — with the per-stream memory cost for capacity planning.

This is NOT a parity gate: ``native == fused == loop`` is already model-free-gated
(``nemotron_batched_attention_test.py``) and the economics run re-confirmed it on real Ultra weights
(``|Δ|=0``). A light ``B=1`` / ``B=4`` sanity self-validates the benched native path, then a native-only
sweep over ``B ∈ {1,8,16,32,48,64,80,96}`` with a **per-stream-memory projection guard**: before each B
it projects the next peak from the worst observed per-stream marginal and STOPS if that projection would
approach the ceiling — on the M3 Ultra an OOM is a reboot hazard, so we never launch a B that could
exceed the safe ceiling. ``max_batch`` does NOT pre-allocate (memory is composed and scales with the
*driven* B — ``batched_runtime.py:36``), so a high cap is free.

The overlap rows (``B=1/8/16/32``) reproduce the economics run; ``B=48/64/80/96`` are the new content.
The per-stream-loop baseline is omitted (it is ~B× slow and already characterized at 1.77x @ B=8 in
``0de52a9``); native is the prod serving path.

One model resident — **RUN SOLO** (~306 GiB int4-RTN backbone, no MTP sidecar; 400 GiB wired limit).

    uv run python -u -m parity.nemotron_ultra_decode_scale
"""

from __future__ import annotations

import mlx.core as mx

from parity.nemotron_batched_bench import (
    _decode_compare,
    _decode_compare_native,
    _gib,
    _prompt,
    _time_path,
)
from parity.nemotron_mtp_k_bench import ART  # the Ultra int4-RTN backbone
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel

BATCHES = (1, 8, 16, 32, 48, 64, 80, 96)
GEN = 24                  # timed decode tokens/stream (matches the economics run for the overlap rows)
SEED_LEN = 128            # short seed prompt (decode tok/s is ~context-independent here)
WIRED_GIB = 400           # pin the ~306 GiB weight set; KV/working grows in the remaining budget
WORKING_CEIL_GIB = 465.0  # skip any B whose PROJECTED peak exceeds this (≈25 GiB margin below the hard ceiling)
HARD_CEIL_GIB = 490.4     # mx.metal recommended max working set (used only for the extrapolation)
INIT_SLOPE = 2.2          # conservative initial GiB/stream guess (refined upward from measured marginals)


def _sanity(model: NemotronBatchedResidentModel) -> None:
    """Self-validate the benched native decode path on real Ultra weights (rule 4): B=1 fused==loop
    bit-exact + B=4 native(form-2)==fused bit-exact. Cheap — 4 steps each."""
    bos = model.cfg.bos_token_id
    w1, m1 = _decode_compare(model, [_prompt(11, bos)], steps=4)                       # fused==loop B=1
    wn, mn = _decode_compare_native(model, [_prompt(n, bos) for n in (9, 13, 7, 11)], steps=4)  # native==fused
    ok = w1 == 0.0 and m1 and wn == 0.0 and mn
    print(f"  parity: B=1 fused==loop |Δ|={w1:.2e}; B=4 native==fused |Δ|={wn:.2e} greedy={mn} "
          f"({'OK' if ok else 'XX'})", flush=True)
    if not ok:
        raise SystemExit("Stream-A sanity FAILED — the benched native decode path is not correct on Ultra")


def _sweep(model: NemotronBatchedResidentModel) -> None:
    """Native (form-2) decode tok/s across B, guarded against the memory ceiling; report scaling +
    per-stream cost + the extrapolated serving ceiling."""
    prompt_ids = _prompt(SEED_LEN, model.cfg.bos_token_id)
    print(f"\n  native (form-2) decode scaling ({SEED_LEN}-tok seed, {GEN} gen/stream); loop baseline "
          f"omitted (1.77x @ B=8, owned by 0de52a9):")
    print(f"  {'B':>4}  {'native per/agg tok/s':>22}  {'agg/B=1':>8}  {'GiB/stream':>10}  "
          f"{'act/peak GiB':>14}", flush=True)

    base_agg = best_agg = best_scale = 0.0
    best_b = 0
    prev_b: int | None = None
    prev_peak = 0.0
    max_slope = INIT_SLOPE                              # worst per-stream marginal seen (for the guard)
    last_per = float("nan")
    for b in BATCHES:
        if prev_b is not None:                         # project this B's peak from the worst marginal
            proj = prev_peak + max_slope * (b - prev_b)
            if proj > WORKING_CEIL_GIB:
                print(f"  {b:>4}  (skipped — projected peak {proj:.1f} GiB > ceil {WORKING_CEIL_GIB:.0f})",
                      flush=True)
                break
        mx.clear_cache()
        mx.reset_peak_memory()
        n_per, n_agg = _time_path(model, prompt_ids, b, GEN, "native")
        peak = _gib(mx.get_peak_memory())
        active = _gib(mx.get_active_memory())
        if base_agg == 0.0:
            base_agg = n_agg
        scale = n_agg / base_agg if base_agg else float("nan")
        gib_stream = (peak - active) / b               # working set per concurrent stream
        if prev_b is not None and b > prev_b:
            max_slope = max(max_slope, (peak - prev_peak) / (b - prev_b))
        print(f"  {b:>4}  {n_per:>9.2f} /{n_agg:>11.2f}  {scale:>7.2f}x  {gib_stream:>9.2f}  "
              f"{active:>5.1f}/{peak:>6.1f}", flush=True)
        prev_b, prev_peak = b, peak
        best_agg, best_b, best_scale, last_per = n_agg, b, scale, n_per

    if prev_b is not None:                             # extrapolate the memory-bounded serving ceiling
        max_b_safe = prev_b + (WORKING_CEIL_GIB - prev_peak) / max_slope
        max_b_hard = prev_b + (HARD_CEIL_GIB - prev_peak) / max_slope
        print(f"\n  per-stream working set ~{max_slope:.2f} GiB/stream (worst marginal) ⇒ extrapolated "
              f"max B ~{max_b_safe:.0f} @ {WORKING_CEIL_GIB:.0f} GiB safe / ~{max_b_hard:.0f} @ "
              f"{HARD_CEIL_GIB:.0f} GiB hard ceiling", flush=True)
        print(f"  headline: peak measured agg {best_agg:.1f} tok/s @ B={best_b} ({best_scale:.2f}x B=1); "
              f"per-stream floor {last_per:.2f} tok/s @ B={best_b}", flush=True)


def run() -> None:
    mx.set_wired_limit(int(WIRED_GIB * 1024 ** 3))
    model = NemotronBatchedResidentModel(ART, max_batch=max(BATCHES))
    assert model._fused, "batched runtime must default to the fused decode path"
    print("=== Nemotron-Ultra U4 Stream-A: decode batch-scaling to the memory ceiling (SOLO) ===")
    print(f"  artifact={ART}")
    print(f"  layers={model.num_layers}  attention={len(model._attn_globals)}  "
          f"max_batch={max(BATCHES)}  wired={WIRED_GIB} GiB  hard_ceil={HARD_CEIL_GIB} GiB")

    print("\n  parity sanity (benched native path correct on real Ultra weights):", flush=True)
    _sanity(model)
    _sweep(model)
    print("\nDONE — Stream-A scaling measured; serving ceiling printed above.", flush=True)


if __name__ == "__main__":
    run()
