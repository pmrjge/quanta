"""Nex-N2-Pro N3 — resident packed forward + batched (B>1) serving re-gate @ 397B (SOLO).

The first N3 milestone over the shipped **int4-g64** artifact (214 GiB / 25 shards): prove the
RESIDENT serving runtime — the packed int4 routed experts (``mx.gather_qmm``) + int8 mixer
projections (``mx.quantized_matmul``) the deployment actually runs — is (a) numerically faithful to
the N2 dequant reference at full 397B scale and (b) correct through the **batched multi-stream decode**
path (the #153 hybrid loop-kill) that serving drives. This is the Super→Ultra re-gate pattern: the
``quanta.qwen35`` batched runtime is already built + model-free-gated (``qwen35_batched_test`` /
``qwen35_batched_loopkill_test``) and graduated on the **Qwen3.6-35B-A3B** keeper
(``qwen35_batched_bench``: 1.63× @ B=32, greedy-exact); Nex is the 397B sibling — same code, bigger
checkpoint, re-gate at scale.

ONE artifact load (rule 8: one decoder block materialized at a time during the streamed load) shared
across all three gates — ``Qwen35BatchedResidentModel`` wraps a ``Qwen35ResidentModel`` as
``._inner``, so the single-stream resident forward (gates 1 + 3) and the batched step (gates 2 + 3)
run on the SAME resident weights without a second load:

  1. **Resident e2e ppl (dequant-ref parity).** Teacher-force the resident packed forward
     (``model._inner(ids)`` — packed-int4 experts + int8 mixer, the served kernels) on the SAME
     645-tok held-out prose the N2 arbiter used, and compare its ppl/top-1 to the **streamed dequant
     reference** computed in-process first (``Qwen35Artifact`` + ``streamed_logits(packed=False)`` —
     the int4 codes dequantized to bf16 + run through the proven naive forward, one block resident at
     a time, then freed). Same int4 codes; the only difference is the kernel (packed fuses the dequant
     at full precision, the reference rounds each weight to bf16 first — so the reference is the
     *lossier* of the two), so the resident ppl must match within bf16-rounding (<2%) AND stay
     low-single-digit (coherent at 397B). This closes the N3 "resident e2e ppl gate" bullet.

  2. **Batched #153 loop-kill re-gate at 397B.** Drive
     :meth:`Qwen35BatchedResidentModel.step_batch` over B concurrent streams with DISTINCT prompts and
     assert **loop == loopkill greedy-exact** at every B ∈ {1,4,8,16,32} — B=1 bit-exact (the b==1
     passthroughs), B≥2 greedy-exact (the option-B packed+chunked projections keep every batched
     ``mx.quantized_matmul`` in the M≤8 bit-exact regime; only the fused padded SDPA softmax reorders).
     The REAL-model correctness gate for the hybrid loop-kill at the true 397B dims/dtypes.

  3. **Design-A equivalence (real weights).** At B=1, the batched orchestration (prefill + per-stream
     ``step_batch``) must be greedy-exact to the single-stream ``Qwen35ResidentModel`` decode at the
     same offsets — proving the batched wrapper is output-equivalent to the parity-gated single-stream
     runtime, on real weights.

Throughput (aggregate / per-stream tok/s) + resident/peak GiB are REPORTED for each B (the serving
fleet-baseline row), not asserted (hardware-variable). A per-stream-memory projection guard skips any
B whose projected peak would approach the working-set ceiling (an M3 Ultra OOM is a reboot hazard).

One model resident — **RUN SOLO** (~214 GiB int4-g64 + per-stream KV/GDN state + transients; 340 GiB
wired, well under the 490.4 GiB ceiling). Qwen3.5 serving is UNPAGED (``shim/omlx`` forces
``paged_kv=False``), so ``step_batch`` IS the prod decode hot path this gate times directly.

    uv run python -u -m parity.nex_n2_pro_batched_real                 # full op-point run (B≤32)
    uv run python -u -m parity.nex_n2_pro_batched_real 1,2 8 4 64      # cheap real smoke (tiny B/seed/gen/ppl)

# parity-gate: real-weight
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

# reuse the EXACT N2 corpus + the proven streamed reference forward + the teacher-forced metric
# (single source of truth — the resident ppl is then directly comparable to the N2 arbiter's 5.0729).
from parity.nex_n2_pro_ppl import PROSE, streamed_logits, teacher_forced
# reuse the batched-bench machinery (pure helpers, no module-load side effects).
from parity.qwen35_batched_bench import _distinct_prompt, _first_divergence, _gib
from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.batched_runtime import Qwen35BatchedResidentModel
from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Nex-N2-Pro-quanta_int4g64"   # the SHIPPED int4-g64 artifact

# --- gate geometry (op-point defaults; CLI-overridable for a cheap real smoke) ----------------------
PPL_TOK = 645            # the full N2 held-out prose (645 tok) — directly comparable to the arbiter
SEED_LEN = 32            # distinct prompt tokens/stream (own GDN recurrent state + GQA KV); decode
                         # tok/s is ~context-independent, so a short seed keeps single-stream prefill bounded
GEN = 24                 # timed decode tokens/stream (steady state)
WARMUP = 4               # JIT-warm steps (captured into the greedy trace; not timed)
BATCH_SIZES = (1, 4, 8, 16, 32)   # 1 = bit-exact anchor; 32 = cohort operating point; 4/8/16 = trend

PPL_REL_TOL = 0.02       # resident-packed vs streamed-dequant ppl: <2% = bf16-rounding, not a divergence
PPL_COHERENT = 12.0      # the resident ppl must stay low-single-digit (a forward bug blows it up by orders)
WIRED_GIB = 340          # pin the ~214 GiB weight set + per-stream state + transients (< 490.4 ceiling)
WORKING_CEIL_GIB = 465.0  # skip any B whose PROJECTED peak exceeds this (≈25 GiB margin below the hard ceiling)
INIT_SLOPE = 2.0         # conservative initial GiB/stream guess (refined upward from measured marginals)


# --- gate 1: resident packed ppl vs the streamed dequant reference ---------------------------------
def _streamed_int4_reference(ids: mx.array) -> dict:
    """The dequant reference: stream the int4 artifact one block at a time (``packed=False`` —
    dequantize each block to bf16, run the proven naive forward, free it before the next; peak ~one
    MoE block, NOT the 214 GiB whole model), teacher-force the same ids. Freed on return so only the
    resident model is held for the rest of the gate (one-model-at-a-time)."""
    t0 = time.perf_counter()
    art = Qwen35Artifact(ART)
    logits = streamed_logits(art, art.cfg, ids)         # [1,T,vocab], dequant-on-read int4
    ppl, acc, argmax = teacher_forced(logits, ids)
    dt = time.perf_counter() - t0
    del art, logits
    mx.clear_cache()
    return {"ppl": ppl, "acc": acc, "argmax": argmax, "sec": dt}


def _resident_packed_ppl(inner: Qwen35ResidentModel, ids: mx.array, ref_argmax: mx.array) -> dict:
    """Teacher-force the RESIDENT packed forward (``__call__(caches=None)`` prefill — packed-int4
    experts via ``gather_qmm`` + int8 mixer via ``quantized_matmul``, the served kernels). Same ids,
    same metric → a ppl/top-1 directly comparable to the streamed dequant reference (the codes are
    identical; only the kernel differs)."""
    t0 = time.perf_counter()
    logits = inner(ids)                                 # [1,T,vocab] (prefill regime)
    ppl, acc, argmax = teacher_forced(logits, ids)
    dt = time.perf_counter() - t0
    agree = float(mx.mean((argmax == ref_argmax).astype(mx.float32)).item())
    del logits
    mx.clear_cache()
    return {"ppl": ppl, "acc": acc, "argmax": argmax, "agree": agree, "sec": dt}


# --- gate 2: batched loop==loopkill greedy-exact + throughput --------------------------------------
def _time_path(model: Qwen35BatchedResidentModel, B: int, prompts: list[list[int]], *,
               loopkill: bool, seed_len: int, gen: int, warmup: int) -> dict:
    """Time ``gen`` steady-state decode steps at batch B on one path (``loopkill`` False=per-stream
    mixer loop / True=#153 loop-kill). Builds FRESH per-stream caches, prefills each prompt
    (single-stream Design-A prefill, always loopkill-off), warms ``warmup``, then times ``gen``.
    Flipping ONLY ``model._loopkill`` is the sole difference between the two paths, so the post-prefill
    greedy traces (warmup+gen) are directly comparable. Returns aggregate/per-stream tok/s, the
    per-stream greedy traces, and active/peak GiB."""
    model._loopkill = bool(loopkill)          # the ONLY thing that differs between paths
    mx.clear_cache()
    mx.reset_peak_memory()

    caches = model.make_batch_caches(B)
    for i, p in enumerate(prompts):
        model.prefill(p, caches[i])           # single-stream prefill (loopkill pinned off inside)
        mx.eval(caches[i].offset)             # land the cache write before the next stream's prefill
    offsets = [c.offset for c in caches]
    toks = [prompts[i][-1] for i in range(B)]  # last prompt token feeds the first decode step
    traces: list[list[int]] = [[] for _ in range(B)]

    def _step() -> None:
        nonlocal toks, offsets
        per_stream = model.step_batch(toks, caches, offsets)
        mx.eval(per_stream)
        toks = [int(mx.argmax(lg[0, -1]).item()) for lg in per_stream]
        offsets = [o + 1 for o in offsets]
        for s in range(B):
            traces[s].append(toks[s])

    for _ in range(warmup):                   # JIT warm (captured into traces; not timed)
        _step()
    t0 = time.perf_counter()
    for _ in range(gen):
        _step()
    dt = time.perf_counter() - t0
    return {"per_stream": gen / dt, "aggregate": B * gen / dt, "traces": traces,
            "active_gib": _gib(mx.get_active_memory()), "peak_gib": _gib(mx.get_peak_memory())}


# --- gate 3: B=1 batched orchestration == single-stream resident (autoregressive greedy) ------------
def _batched_b1_greedy(model: Qwen35BatchedResidentModel, prompt: list[int], n: int) -> list[int]:
    """Autoregressive greedy ``n``-token trace through the BATCHED B=1 path (prefill + ``step_batch``),
    feeding each step its own argmax (loop-kill on, as serving runs it)."""
    model._loopkill = True
    caches = model.make_batch_caches(1)
    logits = model.prefill(prompt, caches[0])         # [1,1,vocab] at the last consumed position
    mx.eval(logits)
    tok = int(mx.argmax(logits[0, -1]).item())
    out, off = [tok], len(prompt)
    for _ in range(n - 1):
        lg = model.step_batch([tok], caches, [off])[0]
        mx.eval(lg)
        tok = int(mx.argmax(lg[0, -1]).item())
        out.append(tok)
        off += 1
    return out


def _single_stream_greedy(inner: Qwen35ResidentModel, prompt: list[int], n: int) -> list[int]:
    """Autoregressive greedy ``n``-token trace through the single-stream resident decode path
    (``__call__`` with caches), feeding each step its own argmax — the parity-gated reference the
    batched B=1 path must match token-for-token."""
    cache = inner.make_caches()
    logits = inner(prompt, caches=cache, offset=0)    # prefill the prompt through the cache
    mx.eval(logits)
    tok = int(mx.argmax(logits[0, -1]).item())
    out, off = [tok], len(prompt)
    for _ in range(n - 1):
        logits = inner([tok], caches=cache, offset=off)
        mx.eval(logits)
        tok = int(mx.argmax(logits[0, -1]).item())
        out.append(tok)
        off += 1
    return out


def run(batch_sizes: tuple[int, ...] = BATCH_SIZES, seed_len: int = SEED_LEN, gen: int = GEN,
        ppl_tok: int = PPL_TOK, warmup: int = WARMUP) -> None:
    mx.set_cache_limit(8 * 1024 ** 3)
    mx.set_wired_limit(int(WIRED_GIB * 1024 ** 3))
    tok = Qwen35Tokenizer.from_pretrained(ART)            # self-contained tokenizer (no BOS)
    ids_list = tok.encode(PROSE)[:ppl_tok]
    ids = mx.array(ids_list, dtype=mx.uint32)
    vocab = None  # filled after load
    print("=== Nex-N2-Pro N3 — resident packed forward + batched serving re-gate @ 397B (SOLO) ===")
    print(f"  artifact={ART}")
    print(f"  ppl_tok={len(ids_list)}  seed_len={seed_len}  gen={gen}  warmup={warmup}  "
          f"B={list(batch_sizes)}  wired={WIRED_GIB} GiB", flush=True)

    # --- gate 1a: the streamed dequant reference (freed before the resident load) -------------------
    print("\n  [gate 1] resident packed ppl vs streamed dequant reference (same 645-tok prose):",
          flush=True)
    ref = _streamed_int4_reference(ids)
    print(f"    streamed dequant int4  : ppl {ref['ppl']:7.4f}  acc {ref['acc']:.4f}  "
          f"({ref['sec']:.0f}s)", flush=True)

    # --- load the resident batched model ONCE (wraps the single-stream Qwen35ResidentModel) ---------
    t0 = time.perf_counter()
    model = Qwen35BatchedResidentModel(ART, max_batch=max(batch_sizes))
    load_s = time.perf_counter() - t0
    vocab = int(model.cfg.vocab_size)
    bos = getattr(model.cfg, "bos_token_id", None)
    n_lin = sum(model.cfg.is_linear_attention(i) for i in range(model.num_layers))
    n_full = model.num_layers - n_lin
    assert model.packed and model.packed_experts, "served runtime must hold packed mixer + packed experts"
    assert model._loopkill, "the #153 hybrid loop-kill is graduated ON by default"
    print(f"    loaded resident {_gib(mx.get_active_memory()):.1f} GiB in {load_s / 60:.1f} min "
          f"({model.num_layers} layers = {n_lin} GDN + {n_full} GQA, packed+packed_experts)", flush=True)

    # --- gate 1b: resident packed ppl (the served kernels) ------------------------------------------
    res = _resident_packed_ppl(model._inner, ids, ref["argmax"])
    rel = abs(res["ppl"] - ref["ppl"]) / ref["ppl"]
    ppl_ok = rel < PPL_REL_TOL and res["ppl"] < PPL_COHERENT
    print(f"    resident packed int4   : ppl {res['ppl']:7.4f}  acc {res['acc']:.4f}  "
          f"agree {res['agree']:.4f}  Δppl {100 * (res['ppl'] / ref['ppl'] - 1):+.2f}%  "
          f"({res['sec']:.0f}s)  {'OK' if ppl_ok else 'FAIL'}", flush=True)
    if not ppl_ok:
        print(f"FAIL — resident packed ppl diverges from the dequant reference (rel {rel:.2%} ≥ "
              f"{PPL_REL_TOL:.0%}) or incoherent (≥ {PPL_COHERENT})", flush=True)
        raise SystemExit(1)

    # --- gate 3: B=1 batched orchestration == single-stream resident (autoregressive greedy) --------
    print("\n  [gate 3] B=1 batched (prefill+step_batch) == single-stream resident greedy (Design A):",
          flush=True)
    g_prompt = ids_list[:seed_len]
    bt = _batched_b1_greedy(model, g_prompt, gen)
    st = _single_stream_greedy(model._inner, g_prompt, gen)
    div_b1 = _first_divergence([st], [bt])
    b1_ok = div_b1 is None
    print(f"    {gen}-token greedy trace  : batched == single-stream  {'OK' if b1_ok else 'FAIL'}"
          + ("" if b1_ok else f"  [DIFF at step {div_b1[1]}: single={div_b1[2]} batched={div_b1[3]}]"),
          flush=True)
    if not b1_ok:
        print("FAIL — the batched B=1 orchestration is not greedy-exact to the single-stream runtime",
              flush=True)
        raise SystemExit(1)

    # --- gate 2: batched loop==loopkill greedy-exact + throughput sweep (memory-guarded) ------------
    print(f"\n  [gate 2] batched decode: loop (per-stream mixer) vs loopkill (#153: ONE batched "
          f"mixer/layer), {seed_len}-tok seed, {gen} gen/stream:", flush=True)
    print(f"  {'B':>4}  {'loop agg':>9}  {'loopkill agg':>12}  {'lk/loop':>8}  "
          f"{'per-stream':>10}  {'act/peak GiB':>14}  {'tok':>4}", flush=True)
    all_tok_ok = True
    ratios: dict[int, float] = {}
    prev_b: int | None = None
    prev_peak = 0.0
    max_slope = INIT_SLOPE
    for B in batch_sizes:
        if prev_b is not None:                                  # OOM guard (reboot hazard): project + skip
            proj = prev_peak + max_slope * (B - prev_b)
            if proj > WORKING_CEIL_GIB:
                print(f"  {B:>4}  (skipped — projected peak {proj:.1f} GiB > ceil "
                      f"{WORKING_CEIL_GIB:.0f})", flush=True)
                break
        prompts = [_distinct_prompt(b, vocab, seed_len, bos) for b in range(B)]
        lp = _time_path(model, B, prompts, loopkill=False, seed_len=seed_len, gen=gen, warmup=warmup)
        lk = _time_path(model, B, prompts, loopkill=True, seed_len=seed_len, gen=gen, warmup=warmup)
        ratio = lk["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        ratios[B] = ratio
        div = _first_divergence(lp["traces"], lk["traces"])
        tok_ok = div is None
        all_tok_ok = all_tok_ok and tok_ok
        print(f"  {B:>4}  {lp['aggregate']:>9.1f}  {lk['aggregate']:>12.1f}  {ratio:>7.2f}x  "
              f"{lk['per_stream']:>9.2f}  {lk['active_gib']:>5.1f}/{lk['peak_gib']:>6.1f}  "
              f"{'ok' if tok_ok else 'DIFF':>4}", flush=True)
        if div is not None:
            s, k, r, g = div
            print(f"    [DIFF] loopkill diverges from loop at stream {s} step {k}: loop={r} loopkill={g}",
                  flush=True)
        if prev_b is not None and B > prev_b:
            max_slope = max(max_slope, (lk["peak_gib"] - prev_peak) / (B - prev_b))
        prev_b, prev_peak = B, lk["peak_gib"]

    if not all_tok_ok:
        print("FAIL — the #153 hybrid loop-kill diverged from the per-stream loop at 397B (see [DIFF])",
              flush=True)
        raise SystemExit(1)

    # --- verdict ------------------------------------------------------------------------------------
    op = 32 if 32 in ratios else max(ratios)
    best = max(ratios.values())
    best_b = max(ratios, key=ratios.get)
    print(f"\nVERDICT: PASS — resident packed ppl {res['ppl']:.4f} ≈ dequant {ref['ppl']:.4f} "
          f"(Δ {100 * (res['ppl'] / ref['ppl'] - 1):+.2f}%); batched loop==loopkill greedy-exact at "
          f"every B; B=1 batched==single-stream. Serving op-point B={op}: loopkill/loop {ratios[op]:.2f}x "
          f"(best {best:.2f}x @ B={best_b}).", flush=True)
    print("PARITY-CHECKS: 3", flush=True)   # (1) resident==dequant ppl, (2) loop==loopkill, (3) B=1 Design-A


if __name__ == "__main__":
    bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
    sl = int(sys.argv[2]) if len(sys.argv) > 2 else SEED_LEN
    gn = int(sys.argv[3]) if len(sys.argv) > 3 else GEN
    pt = int(sys.argv[4]) if len(sys.argv) > 4 else PPL_TOK
    run(bs, sl, gn, pt)
