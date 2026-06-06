"""Nemotron-H batched decode throughput — per-stream loop vs batched Mamba (form-1 concat / form-2
persistent) + fused GQA attention (#153).

Benchmarks the fused ``step_batch`` default (:func:`quanta.nemotron.batched_runtime.batched_decode_step_fused`:
batched q/k/v projections + per-stream ``mx.fast.rope`` + one padded ``mx.fast.scaled_dot_product_attention``
across ``B`` streams for the 8 GQA layers; Mamba stays per-stream, MoE stays the stacked single call)
against the retained per-stream :func:`batched_decode_step` reference, on the resident int4-g64 Nemotron-H
bake. This is the **first real-model exercise** of the fused attention path — its model-free parity is
gated by ``parity/nemotron_batched_attention_test.py``; here we confirm it on the actual baked weights
(the bf16 RoPE-drift hazard [[feedback-batched-rope-bf16]] only surfaces at real magnitudes across layers).

Beyond the 8 fused GQA layers, the **40 Mamba layers** are now batched across streams too (the decode is
launch/IO-bound, not FLOP-bound — the per-stream loop over 40 recurrent layers dominated, not the SDPA):
form-1 concats the per-stream ``(ssm, conv)`` each step, form-2 holds them in a persistent
``BatchedMambaState`` so there is no per-step concat. Both are gated bit-exact (Mamba has no cross-stream
reduction) by ``parity/nemotron_batched_attention_test.py``; here we confirm on the baked weights + time.

Two parts:
  A. **parity** — from identical states:
       A1. fused ``B=1`` must be **bit-exact** vs the per-stream loop — a faithful port; any Δ is a bug.
       A2. fused ``B=4`` ragged offsets must be **greedy-exact** vs the loop — batched RoPE + padded SDPA
           engage so ``|Δlogit|`` is non-zero (attention per-row reduction ULP) but argmax-stable.
       A3. native (form-2) ``B=4`` must be **bit-exact** vs fused (form-1) — identical math, only the
           recurrent-state storage differs (persistent vs reassembled).
  B. **throughput** — ``B in {1,2,4,8,16,32}``, uniform 1024-tok prefill, ``GEN=64`` decode/stream:
     per-stream + aggregate tok/s for looped / fused / native, and the headline native/looped speedup.
     Batching the 40-layer Mamba majority is the real B-scaling lever (the fused-attention-only gain was
     ~1.0–1.08×); form-2 additionally drops form-1's per-step state-concat IO.

One model only (int4-g64 ≈ 68 GB resident) — run SOLO on the M3 Ultra (the OOM-reboot hazard).

    uv run python -u -m parity.nemotron_batched_bench [n_decode]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

import quanta.nemotron.mamba_mixer as mm
from quanta.nemotron.batched_runtime import (
    NemotronBatchedResidentModel,
    batched_decode_step,
    batched_decode_step_fused,
)

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
WARMUP_PROMPT_LEN = 256           # short seed: decode tok/s ~context-independent here, keeps O(B) seeding cheap
GEN = 64                          # timed decode tokens per stream
WARMUP_STEPS = 4                  # steady-state ramp-up (JIT + KV warm) — not timed
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 48)


def _gib(nbytes: int) -> float:
    """MLX byte counter → GiB. We report MLX's own ``get_active_memory``/``get_peak_memory`` (the real
    unified working set), NOT ``resource.ru_maxrss`` — the latter misses both the mmap'd weights and
    MLX's Metal pool, so it under-reads by ~10x on this runtime."""
    return nbytes / (1024 ** 3)


def _prompt(n: int, bos: int) -> mx.array:
    """A deterministic ``bos + ramp`` prompt of length ``n`` (no tokenizer; values are irrelevant to the
    timing/parity, only the shapes + RoPE offsets matter)."""
    return mx.array([bos] + list(range(1, n)))


def _args(model: NemotronBatchedResidentModel):
    return (model.layers, model.embed_w, model.norm_f, model.lm_head_w, model.cfg.norm_eps)


def _seed(model: NemotronBatchedResidentModel, prompts: list[mx.array]) -> list:
    """Prefill each prompt into its own fresh (caches, ssm, conv) state; return the states."""
    states = []
    for p in prompts:
        state = model.make_stream_state()
        mx.eval(model.prefill(p, state))      # grows the KV + fills mamba state in place
        states.append(state)
    return states


def _decode_compare(model: NemotronBatchedResidentModel, prompts: list[mx.array], steps: int
                    ) -> tuple[float, bool]:
    """Lock-step decode ``steps`` tokens through fused vs looped from two identically-seeded state sets;
    return (worst ``|Δlogit|``, all-steps-greedy-match).

    Pins ``BATCHED_FUSED_SSD_STEP=False`` (composed SSD step on both sides) so the fused path's only
    difference from the loop is the **attention** fusion — keeping ``B=1`` bit-exact (a faithful-port
    guard). The graduated fused SSD step's real-weight greedy-exactness is gated separately in
    ``parity/nemotron_ultra_decode_step_breakdown.py`` (``_greedy_match_fused``) + model-free in
    ``parity/nemotron_batched_attention_test.py`` (``_core_grad``)."""
    f_states = _seed(model, prompts)
    l_states = _seed(model, prompts)
    f_tok = [mx.array([int(p[-1].item())]) for p in prompts]
    l_tok = [mx.array([int(p[-1].item())]) for p in prompts]
    args = _args(model)
    worst, match = 0.0, True
    saved = mm.BATCHED_FUSED_SSD_STEP
    mm.BATCHED_FUSED_SSD_STEP = False                      # composed both sides → isolate the attention fusion
    try:
        for _ in range(steps):
            fused = batched_decode_step_fused(*args, f_tok, f_states)
            loope = batched_decode_step(*args, l_tok, l_states)
            mx.eval(fused, loope)
            nf, nl = [], []
            for s in range(len(prompts)):
                fo, lo = fused[s][0, -1], loope[s][0, -1]
                worst = max(worst, float(mx.max(mx.abs(fo - lo)).item()))
                ft, lt = int(mx.argmax(fo).item()), int(mx.argmax(lo).item())
                match = match and (ft == lt)
                nf.append(mx.array([ft]))
                nl.append(mx.array([lt]))
            f_tok, l_tok = nf, nl
    finally:
        mm.BATCHED_FUSED_SSD_STEP = saved
    return worst, match


def _decode_compare_native(model: NemotronBatchedResidentModel, prompts: list[mx.array], steps: int
                           ) -> tuple[float, bool]:
    """Lock-step decode ``steps`` tokens through native (form-2 persistent state) vs fused (form-1) from
    two identically-seeded state sets; return (worst ``|Δlogit|``, greedy-match). Expected **bit-exact**
    (same batched mixer + fused SDPA; only the recurrent-state storage differs)."""
    f_states = _seed(model, prompts)
    n_states = _seed(model, prompts)
    f_tok = [mx.array([int(p[-1].item())]) for p in prompts]
    n_tok = [mx.array([int(p[-1].item())]) for p in prompts]
    args = _args(model)
    nat = model.make_batched_state(n_states)               # assemble [B,...] ssm/conv ONCE
    worst, match = 0.0, True
    for _ in range(steps):
        fused = batched_decode_step_fused(*args, f_tok, f_states)
        nativ = model.step_batch_native(n_tok, nat)
        mx.eval(fused, nativ)
        nf, nn = [], []
        for s in range(len(prompts)):
            fo, no = fused[s][0, -1], nativ[s][0, -1]
            worst = max(worst, float(mx.max(mx.abs(fo - no)).item()))
            ft, nt = int(mx.argmax(fo).item()), int(mx.argmax(no).item())
            match = match and (ft == nt)
            nf.append(mx.array([ft]))
            nn.append(mx.array([nt]))
        f_tok, n_tok = nf, nn
    return worst, match


def _parity(model: NemotronBatchedResidentModel) -> None:
    """A: real-model parity. A1 fused B=1 bit-exact vs loop; A2 fused B=4 ragged greedy-exact vs loop;
    A3 native (form-2) == fused (form-1) bit-exact (the persistent-state path on real weights)."""
    bos = model.cfg.bos_token_id

    w1, m1 = _decode_compare(model, [_prompt(11, bos)], steps=4)
    print(f"  [{'OK' if (w1 == 0.0 and m1) else 'XX'}] A1 B=1 fused==loop bit-exact   |Δlogit|={w1:.2e}  "
          "(faithful port — no batched kernels engaged)", flush=True)
    assert w1 == 0.0, (f"B=1 fused is NOT bit-exact vs the loop (|Δlogit|={w1:.2e}) — a real forward "
                       "bug, not batched-kernel ULP reorder")

    lengths = [9, 13, 7, 11]                           # heterogeneous prefill ⇒ ragged RoPE offsets
    worst, match = _decode_compare(model, [_prompt(n, bos) for n in lengths], steps=4)
    print(f"  [{'OK' if match else 'XX'}] A2 B=4 fused==loop greedy     |Δlogit|={worst:.2e}  "
          f"offsets={lengths} greedy_match={match}  (batched-kernel reduction ULP — argmax-stable)",
          flush=True)
    assert match, "real-model fused greedy tokens diverged from the per-stream loop (B=4 ragged)"

    wn, mn = _decode_compare_native(model, [_prompt(n, bos) for n in lengths], steps=4)
    print(f"  [{'OK' if (wn == 0.0 and mn) else 'XX'}] A3 B=4 native==fused bit-exact |Δlogit|={wn:.2e}  "
          f"greedy_match={mn}  (form-2 persistent state — identical math to form-1)", flush=True)
    assert wn == 0.0, (f"real-model native (form-2) != fused (form-1) (|Δlogit|={wn:.2e}) — a "
                       "persistent-batched-state bookkeeping bug, not kernel ULP")


def _time_steps(step_fn, cur: list[mx.array], b: int, n_decode: int) -> tuple[float, float]:
    """WARMUP_STEPS untimed ramp then ``n_decode`` timed steps of ``step_fn`` (greedy-feeding its own
    argmax); return (per-stream, aggregate) tok/s."""
    for _ in range(WARMUP_STEPS):
        out = step_fn(cur)
        mx.eval(out)
        cur = [mx.array([int(mx.argmax(out[s][0, -1]).item())]) for s in range(b)]
    t0 = time.perf_counter()
    for _ in range(n_decode):
        out = step_fn(cur)
        mx.eval(out)
        cur = [mx.array([int(mx.argmax(out[s][0, -1]).item())]) for s in range(b)]
    dt = time.perf_counter() - t0
    return n_decode / dt, b * n_decode / dt


def _time_path(model: NemotronBatchedResidentModel, prompt_ids: mx.array, b: int, n_decode: int,
               mode: str) -> tuple[float, float]:
    """Time decode at batch ``b`` on one path; (per-stream, agg) tok/s. ``mode``:
    ``"looped"`` (per-stream :func:`batched_decode_step`), ``"fused"`` (form-1 batched Mamba + fused
    attention via ``step_batch``), ``"native"`` (form-2 persistent :class:`BatchedMambaState` via
    ``step_batch_native``). looped/fused toggle the runtime's ``_fused`` flag through the real
    ``step_batch`` dispatch; native builds the persistent state once then steps it."""
    states = _seed(model, [prompt_ids] * b)
    cur = [mx.array([int(prompt_ids[-1].item())]) for _ in range(b)]
    if mode == "native":
        nat = model.make_batched_state(states)
        return _time_steps(lambda c: model.step_batch_native(c, nat), cur, b, n_decode)
    prev = model._fused
    model._fused = (mode == "fused")
    try:
        return _time_steps(lambda c: model.step_batch(c, states), cur, b, n_decode)
    finally:
        model._fused = prev


def run() -> None:
    n_decode = int(sys.argv[1]) if len(sys.argv) > 1 else GEN
    mx.set_wired_limit(int(120 * 1024 ** 3))
    model = NemotronBatchedResidentModel(ART, max_batch=max(BATCH_SIZES))
    assert model._fused, "batched runtime must default to the fused decode path"
    prompt_ids = _prompt(WARMUP_PROMPT_LEN, model.cfg.bos_token_id)

    print("\nA. real-model parity (fused step_batch == per-stream loop):", flush=True)
    _parity(model)

    print(f"\nB. throughput (int4-g64, prompt {WARMUP_PROMPT_LEN} tok, {n_decode} gen/stream): looped "
          "(per-stream) vs fused (form-1 batched Mamba) vs native (form-2 persistent state):", flush=True)
    print(f"{'B':>4}  {'looped per/agg':>20}  {'fused per/agg':>20}  {'native per/agg':>20}  "
          f"{'nat/loop':>9}  {'act/peak GiB':>14}", flush=True)
    for b in BATCH_SIZES:
        # Reset MLX's caching allocator + peak counter around EACH path so every row reports that B's
        # own footprint, not the cumulative high-water of the whole sweep (the leak that ratcheted a
        # prior run to ~200 GiB). clear_cache returns freed buffers to the OS, not just MLX's pool.
        peaks = []
        per_agg = {}
        for mode in ("looped", "fused", "native"):
            mx.clear_cache()
            mx.reset_peak_memory()
            per_agg[mode] = _time_path(model, prompt_ids, b, n_decode, mode)
            peaks.append(mx.get_peak_memory())
        active = mx.get_active_memory()                     # live working set after the last path
        (l_per, l_agg), (f_per, f_agg), (n_per, n_agg) = (
            per_agg["looped"], per_agg["fused"], per_agg["native"])
        spd = n_agg / l_agg if l_agg else float("nan")      # headline: form-2 native vs per-stream loop
        print(f"{b:>4}  {l_per:>8.2f} /{l_agg:>10.2f}  {f_per:>8.2f} /{f_agg:>10.2f}  "
              f"{n_per:>8.2f} /{n_agg:>10.2f}  {spd:>7.2f}x  "
              f"{_gib(active):>5.1f}/{_gib(max(peaks)):>6.1f}", flush=True)


if __name__ == "__main__":
    run()
