"""Nemotron-Ultra U4 economics — combined measure-first run for the two remaining perf streams.

The user picked the combined economics run: ONE solo ~306 GiB Ultra load that characterizes BOTH
remaining U4 streams before any kernel is built (measure-before-optimize, the project ethos). This is
NOT a parity gate — the batched decode + verify paths are already model-free-gated
(``nemotron_batched_attention_test.py``) and real-green on Super-120B; the machinery is correct. This
run measures WHERE the time goes so we can decide what (if anything) to build.

A discovery while scoping reframes MTP-M4's pessimism: the Mamba mixer + the SSD step kernel are
already batch-capable over B (``ssd_step_fused`` launches ``grid=(p,h,bn)``; the mixer runs every op
over the leading B axis). The T==1 multi-stream **decode** path (``batched_decode_step_fused`` /
``_native``) batches Mamba (one ``[B,...]`` mixer call) + attention (fused SDPA) + MoE (stacked). Only
the T>1 **verify** path (``batched_decode_step``) loops Mamba/attn per-stream. So MTP-M4's "60 layers
run per-stream" was the *verify* path — it does NOT necessarily apply to T==1 serving decode.

Part A — **multi-stream B>1 decode scaling** (Stream A). Real-Ultra decode tok/s at B∈{1,8,16,32} on the
form-2 native (persistent ``BatchedMambaState``, the prod serving path) vs the per-stream loop baseline.
Answers the open question: does the hybrid's T==1 decode amortize across B (Mamba weight reads shared ⇒
aggregate tok/s scales), or does the SSD-step compute (kernel grid scales with B) cap it? A light parity
sanity first (native==fused==loop on real Ultra weights) confirms the benched path is correct.

Part B — **T>1 verify component breakdown** (Stream B go/no-go). At B=1, decompose the per-layer-kind
compute (48 mamba / 12 attn / 48 moe / head) at T∈{1,2,3,4} using the REAL resident weights. The
decisive readout is the GROWTH Δ(T=4 − T=1) per kind: the verify forward grows ~linearly with T (eager
1.54/1.94/2.33× t_main per MTP-M3 A). If the Mamba per-token step loop (``mamba_mixer.py:148``) is the
grower ⇒ a fused multi-token SSD scan kernel is the lever (build it). If the MoE ``gather_qmm`` (more
distinct experts hit as T grows ⇒ more weight bandwidth) or the projections dominate the growth ⇒ no
kernel helps (verify must read all weights), B=1 spec is bandwidth-capped on this hybrid, and the
finding redirects fully to Stream A.

One model resident — **RUN SOLO** (~306 GiB int4-RTN backbone, no MTP sidecar; 400 GiB wired limit).

    uv run python -u -m parity.nemotron_ultra_decode_economics
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_batched_bench import (
    _decode_compare,
    _decode_compare_native,
    _gib,
    _prompt,
    _time_path,
)
from parity.nemotron_mtp_k_bench import ART  # the Ultra int4-RTN backbone
from quanta.nemotron.attention import KVCache
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel

DECODE_BATCHES = (1, 8, 16, 32)
LOOP_MAX_B = 8                # per-stream-loop baseline only at small B (it is ~B× slow + predictable;
#                              native is the prod serving path, measured at every B for the scaling curve)
GEN = 24                      # timed decode tokens per stream (enough for a stable tok/s; bounds runtime)
SEED_LEN = 128                # short seed prompt (decode tok/s is ~context-independent here)
VERIFY_TS = (1, 2, 3, 4)      # T = k+1 draft-verify widths (k = 0..3)
B_ITERS = 16                  # Part B timed iters per component
B_WARMUP = 4


def _time_call(fn, iters: int = B_ITERS, warmup: int = B_WARMUP) -> float:
    """Mean wall-clock seconds of ``fn`` (returns array(s) to eval) over ``iters`` after ``warmup``."""
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def _verify_breakdown(model: NemotronBatchedResidentModel, t: int) -> tuple[float, float, float, float]:
    """Per-layer-kind compute (ms) for a T==``t`` verify forward, isolated over the REAL resident
    weights: sum each kind's per-layer call across all its layers (48 mamba / 12 attn / 48 moe) + head.
    Mamba uses the per-token step branch (``conv_state`` populated + ``chunked_cont=False``) — exactly
    the eager verify path. Values are random (timing is value-independent); shapes + real weights matter."""
    inner = model._inner
    cfg = inner.cfg
    h = mx.random.normal((1, t, cfg.hidden_size)).astype(mx.bfloat16)
    gs = min(128, cfg.head_dim)
    conv0 = mx.zeros((1, cfg.conv_kernel - 1, cfg.mamba_conv_dim))
    mamba = [b for b in inner.layers if b.kind == "mamba"]
    attn = [b for b in inner.layers if b.kind == "attention"]
    moe = [b for b in inner.layers if b.kind == "moe"]
    mx.eval(h)

    def run_mamba():                                  # in_proj GEMM + the T-step conv/ssd loop + out_proj
        return [b(h, ssm_state=None, conv_state=conv0)[0] for b in mamba]

    def run_attn():                                   # q/k/v/o int8 GEMMs + SDPA over T queries
        return [b(h, cache=KVCache(group_size=gs))[0] for b in attn]

    def run_moe():                                    # gather_qmm over the experts the T tokens route to
        return [b(h)[0] for b in moe]

    def run_head():                                   # final RMSNorm + lm_head matmul
        hn = mx.fast.rms_norm(h, inner.norm_f.astype(h.dtype), cfg.norm_eps)
        return hn @ inner.lm_head_w.T

    return (_time_call(run_mamba) * 1e3, _time_call(run_attn) * 1e3,
            _time_call(run_moe) * 1e3, _time_call(run_head) * 1e3)


def _part_a(model: NemotronBatchedResidentModel) -> None:
    """Multi-stream B>1 decode throughput scaling: native (form-2) vs per-stream loop, B∈{1,8,16,32}."""
    bos = model.cfg.bos_token_id
    prompt_ids = _prompt(SEED_LEN, bos)

    # parity sanity (rule 4): the benched native path == the per-stream loop on REAL Ultra weights.
    w1, m1 = _decode_compare(model, [_prompt(11, bos)], steps=4)              # fused==loop B=1 bit-exact
    wn, mn = _decode_compare_native(model, [_prompt(n, bos) for n in (9, 13, 7, 11)], steps=4)  # native==fused
    print(f"  parity: B=1 fused==loop |Δ|={w1:.2e} ({'OK' if w1 == 0.0 and m1 else 'XX'}); "
          f"B=4 native==fused |Δ|={wn:.2e} greedy={mn} ({'OK' if wn == 0.0 and mn else 'XX'})", flush=True)
    if not (w1 == 0.0 and m1 and wn == 0.0 and mn):
        raise SystemExit("Part A parity sanity FAILED — the benched decode path is not correct on Ultra")

    print(f"\n  decode throughput (native form-2; loop baseline at B<={LOOP_MAX_B}; {SEED_LEN}-tok seed, "
          f"{GEN} gen/stream):")
    print(f"  {'B':>4}  {'loop per/agg':>20}  {'native per/agg':>20}  {'nat/loop':>9}  "
          f"{'agg/B=1':>8}  {'act/peak GiB':>14}", flush=True)
    base_agg = None
    for b in DECODE_BATCHES:
        mx.clear_cache()
        mx.reset_peak_memory()
        n_per, n_agg = _time_path(model, prompt_ids, b, GEN, "native")
        peak = mx.get_peak_memory()
        if b <= LOOP_MAX_B:
            mx.clear_cache()
            l_per, l_agg = _time_path(model, prompt_ids, b, GEN, "looped")
            spd = f"{n_agg / l_agg:>7.2f}x" if l_agg else f"{'nan':>8}"
            loop_col = f"{l_per:>8.2f} /{l_agg:>10.2f}"
        else:
            spd, loop_col = f"{'—':>8}", f"{'(skipped)':>20}"
        active = mx.get_active_memory()
        if base_agg is None:
            base_agg = n_agg
        scale = n_agg / base_agg if base_agg else float("nan")           # native aggregate scaling vs B=1
        print(f"  {b:>4}  {loop_col}  {n_per:>8.2f} /{n_agg:>10.2f}  {spd}  "
              f"{scale:>7.2f}x  {_gib(active):>5.1f}/{_gib(peak):>6.1f}", flush=True)


def _part_b(model: NemotronBatchedResidentModel) -> None:
    """T>1 verify component breakdown at B=1 — the Stream B go/no-go (where does the T-growth live?)."""
    print("\n  verify component breakdown (B=1, per-layer-kind compute over real weights):")
    print(f"  {'T':>3}  {'mamba(ms)':>10}  {'attn(ms)':>9}  {'moe(ms)':>9}  {'head(ms)':>9}  "
          f"{'sum(ms)':>9}  {'mamba%':>7}  {'moe%':>6}", flush=True)
    rows: dict[int, tuple[float, float, float, float]] = {}
    for t in VERIFY_TS:
        mx.clear_cache()
        tm, ta, tmo, th = _verify_breakdown(model, t)
        rows[t] = (tm, ta, tmo, th)
        s = tm + ta + tmo + th
        print(f"  {t:>3}  {tm:>10.2f}  {ta:>9.2f}  {tmo:>9.2f}  {th:>9.2f}  {s:>9.2f}  "
              f"{100 * tm / s:>6.1f}%  {100 * tmo / s:>5.1f}%", flush=True)

    t0, tmax = VERIFY_TS[0], VERIFY_TS[-1]
    dm = rows[tmax][0] - rows[t0][0]
    da = rows[tmax][1] - rows[t0][1]
    dmo = rows[tmax][2] - rows[t0][2]
    grow = {"mamba": dm, "attn": da, "moe": dmo}
    dominant = max(grow, key=grow.get)
    total_growth = dm + da + dmo
    print(f"\n  T-growth Δ(T={tmax} − T={t0}): mamba {dm:+.2f}ms  attn {da:+.2f}ms  moe {dmo:+.2f}ms  "
          f"(sum {total_growth:+.2f}ms)", flush=True)
    share = (grow[dominant] / total_growth * 100) if total_growth else float("nan")
    print(f"  dominant T-grower: {dominant.upper()} ({share:.0f}% of the growth)", flush=True)
    if dominant == "mamba":
        print("  => the per-token SSD step loop scales with T — a fused multi-token SSD scan kernel is "
              "the B=1 lever (BUILD it; gate output-equivalent to eager, then bench vs the 0.84x ceiling).",
              flush=True)
    else:
        print(f"  => {dominant.upper()} (weight-bandwidth) scales with T, not the Mamba step loop — a "
              "fused SSD kernel would NOT move B=1 spec. Verify is bandwidth-capped here; redirect to "
              "Stream A (serving throughput).", flush=True)


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    model = NemotronBatchedResidentModel(ART, max_batch=max(DECODE_BATCHES))
    assert model._fused, "batched runtime must default to the fused decode path"
    print("=== Nemotron-Ultra U4 decode economics (int4-RTN backbone, SOLO) ===")
    print(f"  artifact={ART}")
    print(f"  layers={model.num_layers}  attention={len(model._attn_globals)}  max_batch={max(DECODE_BATCHES)}")

    print("\nA. multi-stream B>1 decode scaling (Stream A):", flush=True)
    _part_a(model)

    print("\nB. T>1 verify component breakdown (Stream B go/no-go):", flush=True)
    _part_b(model)
    print("\nDONE — economics measured; decision printed above.", flush=True)


if __name__ == "__main__":
    run()
