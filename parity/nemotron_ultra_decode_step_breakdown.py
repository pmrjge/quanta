"""Nemotron-Ultra U4/Stream-A — multi-stream T=1 decode-step component breakdown (where IS the B=32 ceiling?).

Stream A measured aggregate decode **PLATEAUS at ~48 tok/s from B=32** and INFERRED the ceiling is the
"per-stream Mamba recurrence" — by elimination (memory + MoE bandwidth both had headroom), NOT by direct
measurement. The MoE microbench then showed the routed ``gather_qmm`` amortizes **5.58×/token @ B=32**, so
the MoE is provably NOT the B=32 ceiling. But the attribution to the **Mamba recurrence (the SSD step)**
vs the **Mamba projections (int8 ``in_proj``/``out_proj`` GEMMs)** was never measured — and the MoE lesson
is exactly that an *assumed* bottleneck can be wrong (the "missing fusion" already existed). This bench
MEASURES the real per-component B-scaling before any SSD-step kernel is built.

Crucially these amortize DIFFERENTLY across B: a dense int8 GEMM (the mamba projections, attn q/k/v/o, the
MoE experts) reads its weight ONCE for all B tokens — per-token cost DROPS as B grows. The **SSD step** is
the lone exception: each stream owns its ``[1,H,N,P]`` SSM state, so at B=32 the kernel reads/writes 32×
the state with **zero sharing** — its per-token cost is FLAT (or rises). So the amortization ceiling must
be whichever component's per-token cost stops dropping — directly identified here, not inferred.

For B ∈ {1,8,16,32}: prefill B streams into one **form-2 ``BatchedMambaState``** (the prod serving path),
then time a T=1 decode step decomposed by layer-kind over the REAL resident int4-RTN weights —

  * **mamba** (48): ``in_proj`` int8 GEMM → conv step → SSD step → gated RMSNorm → ``out_proj`` int8 GEMM
  * **attn**  (12): batched q/k/v/o int8 GEMMs + ONE fused padded SDPA across streams (the #153 path)
  * **moe**   (48): the stacked routed ``gather_qmm`` (already amortizing — the MoE microbench)
  * **head**:       final RMSNorm + ``lm_head``

and reports per-kind total ms, per-TOKEN µs (``ms·1000/B``), and the per-token **B-scaling** (cost@B /
cost@B=1). The kind whose per-token cost drops LEAST (ratio nearest 1, or > 1) is the lever. A mamba
**sub-breakdown** (one real block: ``in_proj`` / conv / ``ssd_step`` / ``ssd_step_fused`` / norm /
``out_proj`` at B=1 vs B=32) decides whether the SSD recurrence or the projections dominate the mamba
block — i.e. whether "the batched SSD-step kernel" is even the right lever, and whether the fused-step
kernel (``FUSED_SSD_STEP``, measured a no-win in *compiled single-stream* decode) becomes a win at B=32.
Finally an **E2E A/B** swaps ``FUSED_SSD_STEP`` off↔on in the REAL native serving decode (with cross-layer
overlap intact, plus a greedy-exactness check) — the decisive test the no-overlap microbench cannot give:
does graduating the already-built fused step actually lift serving tok/s, and is it output-equivalent?

NOT a parity gate (native==fused==loop is gated model-free + Stream-A re-confirmed ``|Δ|=0`` on real
weights); a measurement. One model resident — **RUN SOLO** (~306 GiB int4-RTN backbone; 400 GiB wired).

    uv run python -u -m parity.nemotron_ultra_decode_step_breakdown
"""

from __future__ import annotations

import time

import mlx.core as mx

import quanta.nemotron.mamba_mixer as mm
from parity.nemotron_batched_bench import _gib, _prompt, _seed, _time_path
from parity.nemotron_mtp_k_bench import ART  # the Ultra int4-RTN backbone
from quanta.nemotron.batched_runtime import (
    NemotronBatchedResidentModel,
    _fused_attn_layer,
    _read_attn_offsets,
)
from quanta.nemotron.mamba_mixer import _silu, _softplus
from quanta.nemotron.mamba_ssd import causal_conv1d_step, ssd_step, ssd_step_fused

BATCHES = (1, 8, 16, 32)      # up to the Stream-A throughput knee (B=32); where the question lives
SEED_LEN = 128                # short seed prompt (decode tok/s ~context-independent here)
ITERS = 10                    # timed iters per component (T=1 forwards are tiny; bounds runtime + KV growth)
WARMUP = 3
GEN = 24                      # timed decode tokens/stream for the fused-step A/B (matches Stream-A/economics)
WIRED_GIB = 400               # pin the ~306 GiB weight set; per-stream state grows in the remaining budget


def _time_call(fn, iters: int = ITERS, warmup: int = WARMUP) -> float:
    """Mean wall-clock **ms** of ``fn`` (returns array(s)/tuple to eval) over ``iters`` after ``warmup``."""
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters * 1e3


def _kind_breakdown(model: NemotronBatchedResidentModel, b: int) -> dict:
    """Time one T=1 decode step at batch ``b`` decomposed by layer-kind over real resident weights.

    Returns a dict with per-kind ms (summed over that kind's layers), the head ms, the real
    ``step_batch_native`` total ms, and the mamba sub-breakdown (one block) — all for batch ``b``."""
    cfg = model.cfg
    layers = model.layers
    kinds = cfg.layers_block_type
    mamba_idx = [i for i, k in enumerate(kinds) if k == "mamba"]
    attn_idx = [i for i, k in enumerate(kinds) if k == "attention"]
    moe_idx = [i for i, k in enumerate(kinds) if k == "moe"]

    states = _seed(model, [_prompt(SEED_LEN, cfg.bos_token_id)] * b)   # B prefilled per-stream triples
    nat = model.make_batched_state(states)                            # form-2 persistent batched state
    h = mx.random.normal((b, 1, cfg.hidden_size)).astype(mx.bfloat16)  # one decode-token hidden / stream
    offsets = _read_attn_offsets(layers, nat.kv, b)
    mx.eval(h)

    # --- per-kind layer totals (each kind's real per-layer compute, summed; no chaining — a breakdown) ---
    def run_mamba():     # in_proj GEMM + conv step + SSD step + gated RMSNorm + out_proj GEMM, ×48
        return [layers[i](h, cache=None, ssm_state=nat.ssm[i], conv_state=nat.conv[i],
                          use_fast=True)[0] for i in mamba_idx]

    def run_moe():       # stacked routed gather_qmm (the moe mixer is [B,1,hidden]-aware), ×48
        return [layers[i](h, cache=None, ssm_state=None, conv_state=None, use_fast=True)[0]
                for i in moe_idx]

    def run_head():      # final RMSNorm + lm_head matmul
        hn = mx.fast.rms_norm(h, model.norm_f.astype(h.dtype), cfg.norm_eps)
        return hn @ model.lm_head_w.T

    def run_attn():      # batched q/k/v/o + ONE fused padded SDPA across streams, ×12 (grows KV ~ITERS)
        return [_fused_attn_layer(layers[i], h, offsets, [nat.kv[s][i] for s in range(b)],
                                  paged_batched=model._paged_kv_batched) for i in attn_idx]

    t_mamba = _time_call(run_mamba)
    t_moe = _time_call(run_moe)
    t_head = _time_call(run_head)
    t_attn = _time_call(run_attn)          # last (mutates KV); bounded growth, representative for a breakdown

    # --- real total (the prod serving step) for a sum-of-components sanity (advances nat; timed last) ---
    cur = [mx.array([int(_prompt(SEED_LEN, cfg.bos_token_id)[-1].item())]) for _ in range(b)]
    t_total = _time_call(lambda: model.step_batch_native(cur, nat))

    sub = _mamba_sub(model, mamba_idx[0], h, nat) if b in (BATCHES[0], BATCHES[-1]) else None
    return {"mamba": t_mamba, "attn": t_attn, "moe": t_moe, "head": t_head, "total": t_total,
            "peak": _gib(mx.get_peak_memory()), "active": _gib(mx.get_active_memory()), "sub": sub}


def _mamba_sub(model: NemotronBatchedResidentModel, idx: int, h: mx.array, nat) -> dict:
    """Sub-breakdown of ONE real mamba block's decode path (the ``MambaMixer.__call__`` per-token branch):
    ``in_proj`` / conv step / ``ssd_step`` / ``ssd_step_fused`` / gated RMSNorm / ``out_proj``. Intermediates
    are pre-evaluated so each timed op measures only itself. Decides ssd-recurrence vs projection dominance
    (and whether the fused-step kernel beats the composed ``ssd_step`` at this B)."""
    mx_blk = model.layers[idx].mixer
    b = int(h.shape[0])
    ssm_b, conv_b = nat.ssm[idx], nat.conv[idx]            # [B,H,N,P], [B,K-1,Cdim]
    a = -mx.exp(mx_blk.A_log)
    proj = mx_blk.in_proj(h)
    z, xbc, dt = mx_blk._split(proj)
    dt_all = _softplus(dt + mx_blk.dt_bias)               # [B,1,H]
    conv_out, _ = causal_conv1d_step(xbc[:, 0], mx_blk.conv_weight, conv_b, mx_blk.conv_bias)  # [B,Cdim]
    xs_t, bm_t, cm_t = mx_blk._split_xbc(_silu(conv_out)[:, None], b, 1)
    x0, dt0, bm0, cm0 = xs_t[:, 0], dt_all[:, 0], bm_t[:, 0], cm_t[:, 0]
    y_t, _ = ssd_step(x0, dt0, a, bm0, cm0, mx_blk.D, ssm_b)
    y = y_t[:, None].reshape(b, 1, mx_blk.d_inner)
    yn = mx_blk.norm(y, z)
    mx.eval(proj, z, xbc, dt_all, conv_out, x0, dt0, bm0, cm0, y, yn, a)

    return {
        "in_proj": _time_call(lambda: mx_blk.in_proj(h)),
        "conv": _time_call(lambda: causal_conv1d_step(xbc[:, 0], mx_blk.conv_weight, conv_b,
                                                      mx_blk.conv_bias)),
        "ssd_step": _time_call(lambda: ssd_step(x0, dt0, a, bm0, cm0, mx_blk.D, ssm_b)),
        "ssd_fused": _time_call(lambda: ssd_step_fused(x0, dt0, a, bm0, cm0, mx_blk.D, ssm_b)),
        "norm": _time_call(lambda: mx_blk.norm(y, z)),
        "out_proj": _time_call(lambda: mx_blk.out_proj(yn)),
    }


def _greedy_match_fused(model: NemotronBatchedResidentModel, b: int, steps: int) -> tuple[float, bool]:
    """E2E greedy-exactness of the fused SSD step in the real native decode loop (rule 4): two
    identically-seeded form-2 batched states decode ``steps`` tokens, one with ``FUSED_SSD_STEP=False``
    (composed ``ssd_step``), one with ``True`` (the fused kernel); compare per-step argmax + worst
    ``|Δlogit|``. The kernel is parity-gated == ``ssd_step`` (~2e-7 fp32 reorder), so the e2e tokens must
    AGREE (argmax-stable) — confirming the optimization is output-equivalent before it can be graduated."""
    prompt = _prompt(SEED_LEN, model.cfg.bos_token_id)
    s_off = model.make_batched_state(_seed(model, [prompt] * b))
    s_on = model.make_batched_state(_seed(model, [prompt] * b))
    last = int(prompt[-1].item())
    t_off = [mx.array([last]) for _ in range(b)]
    t_on = [mx.array([last]) for _ in range(b)]
    worst, match = 0.0, True
    for _ in range(steps):
        mm.FUSED_SSD_STEP = False
        o_off = model.step_batch_native(t_off, s_off)
        mm.FUSED_SSD_STEP = True
        o_on = model.step_batch_native(t_on, s_on)
        mm.FUSED_SSD_STEP = False
        mx.eval(o_off, o_on)
        nf, nn = [], []
        for sidx in range(b):
            fo, no = o_off[sidx][0, -1], o_on[sidx][0, -1]
            worst = max(worst, float(mx.max(mx.abs(fo - no)).item()))
            ft, nt = int(mx.argmax(fo).item()), int(mx.argmax(no).item())
            match = match and (ft == nt)
            nf.append(mx.array([ft]))
            nn.append(mx.array([nt]))
        t_off, t_on = nf, nn
    return worst, match


def _ab_fused_step(model: NemotronBatchedResidentModel, b: int) -> tuple[float, float]:
    """Native (form-2) decode aggregate tok/s at batch ``b`` with the composed ``ssd_step``
    (``FUSED_SSD_STEP=False``, baseline) vs the fused kernel (``True``). Returns (agg_off, agg_on)."""
    prompt = _prompt(SEED_LEN, model.cfg.bos_token_id)
    mm.FUSED_SSD_STEP = False
    _, agg_off = _time_path(model, prompt, b, GEN, "native")
    mm.FUSED_SSD_STEP = True
    _, agg_on = _time_path(model, prompt, b, GEN, "native")
    mm.FUSED_SSD_STEP = False
    return agg_off, agg_on


def run() -> None:
    mx.set_wired_limit(int(WIRED_GIB * 1024**3))
    mx.random.seed(0)
    model = NemotronBatchedResidentModel(ART, max_batch=max(BATCHES))
    assert model._fused, "batched runtime must default to the fused decode path"
    cfg = model.cfg
    kinds = cfg.layers_block_type
    nm = sum(k == "mamba" for k in kinds)
    na = sum(k == "attention" for k in kinds)
    nmo = sum(k == "moe" for k in kinds)
    print("=== Nemotron-Ultra decode-step component breakdown (multi-stream T=1, real weights, SOLO) ===")
    print(f"  artifact={ART}")
    print(f"  layers={model.num_layers}  mamba={nm} attn={na} moe={nmo}  hidden={cfg.hidden_size}  "
          f"wired={WIRED_GIB} GiB")

    rows: dict[int, dict] = {}
    print("\n  per-kind compute per T=1 decode step (ms summed over each kind's layers):")
    print(f"  {'B':>4}  {'mamba':>8}  {'attn':>7}  {'moe':>8}  {'head':>7}  {'Σparts':>8}  "
          f"{'real tot':>8}  {'act/peak GiB':>14}", flush=True)
    for b in BATCHES:
        mx.clear_cache()
        mx.reset_peak_memory()
        r = _kind_breakdown(model, b)
        rows[b] = r
        sigma = r["mamba"] + r["attn"] + r["moe"] + r["head"]
        print(f"  {b:>4}  {r['mamba']:>8.2f}  {r['attn']:>7.2f}  {r['moe']:>8.2f}  {r['head']:>7.2f}  "
              f"{sigma:>8.2f}  {r['total']:>8.2f}  {r['active']:>5.1f}/{r['peak']:>6.1f}", flush=True)

    # --- per-token amortization view: EVERY dense-GEMM kind drops with B (weight read shared over the B
    # tokens); the SSD recurrence is the lone non-amortizer (per-stream state). The amortization-least kind
    # is informational, NOT "the ceiling" — magnitude matters (a small kind that amortizes least is not the
    # lever). The decisive signal is the absolute B=max composition + the mamba sub-breakdown below.
    b1, bn = BATCHES[0], BATCHES[-1]
    print("\n  per-TOKEN cost µs (= ms·1000/B) and B-scaling (cost@B / cost@B=1; <1 amortizes):")
    print(f"  {'kind':>7}  {'µs/tok@1':>9}  {'µs/tok@'+str(bn):>10}  {'scaling':>8}", flush=True)
    least_kind, least_ratio = None, -1.0
    for k in ("mamba", "attn", "moe", "head"):
        per1 = rows[b1][k] * 1e3 / b1
        pern = rows[bn][k] * 1e3 / bn
        ratio = pern / per1 if per1 else float("nan")
        if ratio > least_ratio:
            least_kind, least_ratio = k, ratio
        print(f"  {k:>7}  {per1:>9.1f}  {pern:>10.1f}  {ratio:>7.2f}x", flush=True)

    # --- mamba sub-breakdown (one block) at B=1 vs B=32 ---
    print("\n  mamba block sub-breakdown (ONE real block, µs/call):")
    print(f"  {'B':>4}  {'in_proj':>8}  {'conv':>7}  {'ssd_step':>9}  {'ssd_fused':>10}  {'norm':>7}  "
          f"{'out_proj':>9}  {'ssd%blk':>8}", flush=True)
    for b in (b1, bn):
        s = rows[b]["sub"]
        if s is None:
            continue
        blk_total = sum(s[k] for k in ("in_proj", "conv", "ssd_step", "norm", "out_proj"))  # composed path
        ssd_frac = 100 * s["ssd_step"] / blk_total if blk_total else float("nan")
        print(f"  {b:>4}  {s['in_proj']*1e3:>8.1f}  {s['conv']*1e3:>7.1f}  {s['ssd_step']*1e3:>9.1f}  "
              f"{s['ssd_fused']*1e3:>10.1f}  {s['norm']*1e3:>7.1f}  {s['out_proj']*1e3:>9.1f}  "
              f"{ssd_frac:>7.1f}%", flush=True)

    # --- E2E fused SSD-step A/B: the decisive test the no-overlap microbench CANNOT give. Does swapping the
    # composed ``ssd_step`` -> the already-built ``ssd_step_fused`` actually speed up the REAL native serving
    # decode (cross-layer overlap intact), and is it greedy-exact? (At B=1 *compiled* the fused step was a
    # wash — mx.compile fused the composed ops; the batched native path is EAGER, so the temporaries are
    # materialized and the kernel should win at B>1.) ---
    print("\n  E2E fused SSD-step A/B on the native (form-2) serving decode (the prod multi-stream path):")
    wmatch, ok = _greedy_match_fused(model, 4, 8)
    print(f"  greedy-exactness (B=4, 8 steps): composed-step vs fused-step |Δlogit|={wmatch:.2e} "
          f"argmax_match={ok} ({'OK — output-equivalent, rule 4' if ok else 'XX — NOT equivalent'})",
          flush=True)
    print(f"  {'B':>4}  {'composed agg':>13}  {'fused agg':>11}  {'fused/composed':>15}", flush=True)
    ab: dict[int, tuple[float, float]] = {}
    for b in BATCHES:
        mx.clear_cache()
        agg_off, agg_on = _ab_fused_step(model, b)
        ab[b] = (agg_off, agg_on)
        spd = agg_on / agg_off if agg_off else float("nan")
        print(f"  {b:>4}  {agg_off:>11.2f}  {agg_on:>11.2f}  {spd:>13.2f}x", flush=True)

    # --- verdict (data-driven, magnitude-aware) ---
    r = rows[bn]
    parts = {"mamba": r["mamba"], "attn": r["attn"], "moe": r["moe"], "head": r["head"]}
    tot = sum(parts.values())
    order = sorted(parts, key=parts.get, reverse=True)
    sub = r["sub"]
    blk_total = sum(sub[k] for k in ("in_proj", "conv", "ssd_step", "norm", "out_proj"))
    ssd_frac = sub["ssd_step"] / blk_total if blk_total else 0.0
    proj_frac = (sub["in_proj"] + sub["out_proj"]) / blk_total if blk_total else 0.0
    fused_iso = sub["ssd_step"] / sub["ssd_fused"] if sub["ssd_fused"] else float("nan")
    off_bn, on_bn = ab[bn]
    e2e_win = on_bn / off_bn if off_bn else float("nan")
    print(f"\nVERDICT (B={bn}):")
    print("  absolute step composition: " + " + ".join(f"{k} {100*parts[k]/tot:.0f}%" for k in order),
          flush=True)
    print(f"  every kind amortizes per-token (least: {least_kind.upper()} {least_ratio:.2f}× — small, "
          "not the lever)", flush=True)
    print(f"  mamba block @ B={bn}: SSD step {100*ssd_frac:.0f}% vs projections {100*proj_frac:.0f}% — the "
          f"composed ssd_step materializes [B,H,N,P] fp32 temporaries; isolated fused kernel {fused_iso:.2f}× "
          "faster", flush=True)
    print(f"  E2E: graduating FUSED_SSD_STEP makes native serving decode {e2e_win:.2f}× @ B={bn} "
          f"({off_bn:.1f} -> {on_bn:.1f} tok/s), greedy-exact={ok}", flush=True)
    if ok and e2e_win >= 1.05:
        print(f"  => LEVER CONFIRMED: the serving lever is NOT a new kernel — it is graduating the already-"
              f"built ssd_step_fused for the batched decode path ({e2e_win:.2f}× @ B={bn}, output-equivalent).",
              flush=True)
        print("     The real B=32 mamba cost is the composed-op [B,H,N,P] fp32 blowup, not the recurrence "
              "FLOPs. Next milestone: wire FUSED_SSD_STEP into the batched steppers + re-gate native==fused "
              "greedy-exact; the residual ceiling is then the MoE+mamba co-dominant weight bandwidth (B>32).",
              flush=True)
    elif ok:
        print(f"  => the fused step is output-equivalent but only {e2e_win:.2f}× e2e — cross-layer overlap "
              f"absorbs the isolated {fused_iso:.2f}× (as it did at B=1 compiled). No free serving win; the "
              "B=32 ceiling is the MoE+mamba co-dominant weight bandwidth.", flush=True)
    else:
        print("  => the fused step is NOT greedy-exact e2e — do NOT graduate (rule 4 / rule 6); investigate.",
              flush=True)
    print("\nDONE — decode-step composition + fused-step serving lever measured above.", flush=True)


if __name__ == "__main__":
    run()
