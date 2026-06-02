"""Qwen3.6 batched (B>1) UNPAGED decode throughput bench — the #153 hybrid loop-kill on the real model.

HEAVY: loads the resident int4-g64 Qwen3.6-35B-A3B bake (~19 GiB). RUN SOLO — no other model
resident (one-model-at-a-time; an over-subscribed load OOM-reboots the host). Standalone:

    uv run python -m parity.qwen35_batched_bench              # B in {1,4,8,16,32}
    uv run python -m parity.qwen35_batched_bench 1,4          # cheap harness validation
    uv run python -m parity.qwen35_batched_bench 4            # only the prod operating point

(Drives raw token-id lists straight into the runtime — no tokenizer, no extra deps.)

Drives :meth:`quanta.qwen35.batched_runtime.Qwen35BatchedResidentModel.step_batch` directly over ``B``
concurrent streams with DISTINCT prompts (each stream keeps its OWN GDN recurrent state + GQA KV — a
distinct leading token defeats any trivial identical-prompt amortization). ``step_batch`` IS the prod
decode hot path: Qwen serving is UNPAGED (``shim/omlx`` forces ``paged_kv=False``) and the
``_Qwen35BatchedSession`` wrapper just orchestrates admit/release around this same call, so timing it
directly isolates the loop-kill cleanly. For each B it times TWO decode paths on the SAME resident
weights, flipping ONLY ``model._loopkill`` (the #153 :data:`QWEN35_BATCHED_LOOPKILL_DEFAULT` flag)
between them:

  * **loop**     — ``_loopkill=False``: the per-stream mixer loop (``for s in range(B)``) inside
    :func:`~quanta.qwen35.batched_runtime.batched_decode_step` — every stream runs its GDN recurrence /
    GQA attention through its own cache one at a time (the proven pre-#153 path);
  * **loopkill** — ``_loopkill=True``: that B-stream mixer loop replaced by ONE batched mixer per layer —
    GQA via :meth:`~quanta.qwen35.attention.Qwen35Attention.decode_step_batched` (batched q/k/v/o proj +
    per-stream RoPE + shared fused padded SDPA, M1) and Gated-DeltaNet via
    :func:`~quanta.qwen35.batched_runtime._gdn_step_batched` (gather B streams' (conv,recurrent) state +
    ONE recurrence + scatter, M2 — the bigger lever: 45 of 60 layers are GDN).

Both paths run the IDENTICAL batched MoE sub-block (the routed/shared expert reads already amortize over
B via the existing ``qwen35_moe`` ``[B,1,h]→[N,h]`` reshape) — only the MIXER step differs — so
``loopkill/loop`` aggregate tok/s isolates the #153 mixer-loop-kill win. Qwen is the HYBRID case: the
loop-kill trims the per-stream Python loop on ALL 60 layers (45 GDN + 15 GQA), amortizing each mixer's
in/out projection + conv / q/k/v/o weight read ONCE across B (the bandwidth win, mirroring the MoE
expert-read amortization). Qwen's batched operating point is **B=32** (re-pinned from the original B=4
to match the cohort cap), so the win should approach InternLM2.5's (3.20×@B32) / Nemotron's (1.15×@B32);
B=1 anchors bit-exactness and the sweep shows the B-trend.

This is ALSO the real-model correctness gate for the hybrid loop-kill at the TRUE dims/dtypes (the
model-free ``qwen35_batched_loopkill_test`` used a tiny random-weights config). ``run()`` asserts EVERY
stream's greedy token trace is IDENTICAL loop-vs-loopkill — B=1 bit-exact (the b==1 paths are strict
passthroughs: GDN no-concat, GQA all-zero pad mask == mask=None) and B>=2 greedy-exact. With the CURRENT
dequantized runtime B>=2 FAILS that bar — the dense-bf16 projection-GEMM batch-M accumulation reorder is
NOT argmax-stable: it compounds over depth and flips greedy tokens (worst |Δlogit|≈1.3, three orders over
tol — NOT the [[feedback-batched-rope-bf16]] ULP-noise class). That is the real loop-kill blocker option B
fixes, surfaced loud (rule 6). Throughput is reported, not asserted (hardware-variable).

Memory is read with MLX's own counters (``get_active_memory`` / ``get_peak_memory``); ``clear_cache`` +
``reset_peak_memory`` run between every path/B so a peak is that configuration's true transient.

Geometry: prompt = 128 tok/stream (distinct per stream; single-stream Design-A prefill, loopkill-off),
GEN = 64 decoded tok/stream (steady state; WARMUP steps JIT-warm and are not timed), no EOS stop (every
stream decodes exactly WARMUP+GEN).
"""

from __future__ import annotations

import gc
import math
import time

import mlx.core as mx

from quanta.qwen35.batched_runtime import Qwen35BatchedResidentModel
from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
PROMPT_LEN = 128                  # distinct prompt tokens per stream (own GDN state + GQA KV)
GEN = 64                          # decoded tokens per stream (timed)
WARMUP_STEPS = 4                  # JIT warm (not timed)
BATCH_SIZES = (1, 4, 8, 16, 32)   # 1 = bit-exact anchor; 4 = prod operating point; 8/16/32 = trend


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _distinct_prompt(b: int, vocab: int, n: int, bos: int | None) -> list[int]:
    """A length-``n`` prompt whose leading token differs across streams (so no two streams share a
    GDN recurrent trajectory / GQA KV prefix — each stream gets its own full state, the real B-stream
    load). A large per-stream stride spreads the ids; clamp into ``[1, vocab)`` (the token VALUES only
    need to be valid + stream-distinct; decode tok/s is matmul-bandwidth dominated, ~context-independent,
    and the bench never stops on EOS so an EOS landing in a prompt is just another token)."""
    stride = (b + 1) * 1009
    head = [int(bos)] if bos is not None else []
    body = [1 + ((stride + j * 7) % (vocab - 2)) for j in range(n - len(head))]
    return head + body


def _time_path(model: Qwen35BatchedResidentModel, B: int, prompts: list[list[int]], *,
               loopkill: bool) -> dict:
    """Time GEN steady-state decode steps at batch B on one path (``loopkill`` False=loop / True=loop-
    kill). Builds FRESH per-stream caches, prefills each prompt (single-stream Design-A prefill, always
    loopkill-off), warms WARMUP_STEPS, then times GEN. Returns aggregate/per-stream tok/s, the per-stream
    greedy token traces (WARMUP+GEN), and active/peak GiB. Flipping ONLY ``model._loopkill`` is the sole
    difference between the two paths — the prefilled caches are bit-identical (deterministic prompts +
    loopkill-off prefill), so the post-prefill greedy traces are directly comparable."""
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

    for _ in range(WARMUP_STEPS):             # JIT warm (captured into traces; not timed)
        _step()

    t0 = time.perf_counter()
    for _ in range(GEN):
        _step()
    dt = time.perf_counter() - t0

    return {"per_stream": GEN / dt, "aggregate": B * GEN / dt, "traces": traces,
            "active_gib": _gib(mx.get_active_memory()), "peak_gib": _gib(mx.get_peak_memory())}


def _first_divergence(loop_traces: list[list[int]], lk_traces: list[list[int]]) -> tuple | None:
    """First (stream, step, loop_tok, loopkill_tok) where the two paths' greedy traces differ, or None
    if every stream is token-identical (rule 6 — surface the exact divergence, don't just say DIFF)."""
    for s in range(len(loop_traces)):
        ref, got = loop_traces[s], lk_traces[s]
        for k in range(min(len(ref), len(got))):
            if ref[k] != got[k]:
                return (s, k, ref[k], got[k])
    return None


def run(batch_sizes: tuple[int, ...] = BATCH_SIZES) -> None:
    max_batch = max(batch_sizes)
    # Pin the resident weight set (Qwen3.6-35B-A3B int4-g64 ≈ 19 GiB + per-stream GDN/KV state + transient).
    mx.set_wired_limit(int(80 * 1024 ** 3))
    model = Qwen35BatchedResidentModel(ART, max_batch=max_batch)

    cfg = model.cfg
    vocab = int(cfg.vocab_size)
    bos = getattr(cfg, "bos_token_id", None)
    n_lin = sum(cfg.is_linear_attention(i) for i in range(model.num_layers))
    n_full = model.num_layers - n_lin

    print(f"\n=== Qwen3.6-35B-A3B int4-g64 UNPAGED batched decode (prompt {PROMPT_LEN} tok, {GEN} gen/stream, "
          f"{model.num_layers} layers = {n_lin} GDN + {n_full} GQA): loop (per-stream mixer) vs loopkill "
          f"(#153: ONE batched mixer/layer) ===")
    print("aggregate tok/s (per-stream = aggregate / B). loopkill/loop = the #153 hybrid mixer-loop-kill "
          "win. GiB = loopkill-path active/peak. tok = loop==loopkill greedy-exact (real-model correctness).")
    print(f"{'B':>4}  {'loop':>9}  {'loopkill':>9}  {'loopkill/loop':>13}  "
          f"{'loopkill GiB a/p':>17}  {'tok':>4}")
    all_tok_ok = True
    ratios: dict[int, float] = {}
    for B in batch_sizes:
        prompts = [_distinct_prompt(b, vocab, PROMPT_LEN, bos) for b in range(B)]
        lp = _time_path(model, B, prompts, loopkill=False)   # per-stream mixer loop
        lk = _time_path(model, B, prompts, loopkill=True)    # #153 hybrid loop-kill
        ratio = lk["aggregate"] / lp["aggregate"] if lp["aggregate"] else float("nan")
        ratios[B] = ratio
        div = _first_divergence(lp["traces"], lk["traces"])
        tok_ok = div is None
        all_tok_ok = all_tok_ok and tok_ok
        print(f"{B:>4}  {lp['aggregate']:>9.1f}  {lk['aggregate']:>9.1f}  {ratio:>12.2f}x  "
              f"{lk['active_gib']:>7.1f}/{lk['peak_gib']:>7.1f}  {'ok' if tok_ok else 'DIFF':>4}")
        if div is not None:
            s, k, ref, got = div
            print(f"    [DIFF] loopkill diverges from loop at stream {s} step {k}: "
                  f"loop={ref} loopkill={got}")

    # Honest verdict (rule 6): the loop-kill MUST be token-identical to the per-stream loop (B=1 bit-
    # exact, B>=2 greedy-exact — same batched MoE, only the mixer step differs). A divergence is a real
    # #153 bug; fail loud. Throughput is reported, not asserted (hardware-variable). Graduation is judged
    # at B=32 (Qwen's re-pinned operating point), with the rest of the sweep as supporting trend.
    if not all_tok_ok:
        print("FAIL — the #153 hybrid loop-kill diverged from the per-stream mixer loop (see [DIFF] above)")
        raise SystemExit(1)
    op = 32 if 32 in ratios else max(ratios)          # prod operating point (fallback: largest swept B)
    best = max(ratios.values())
    verdict = f"PASS — loop == loopkill greedy-exact on the real model; B={op} loopkill/loop = {ratios[op]:.2f}x"
    if best > ratios[op]:
        best_b = max(ratios, key=ratios.get)
        verdict += f" (best {best:.2f}x @ B={best_b})"
    verdict += "; " + ("a win — graduate the default" if ratios[op] >= 1.0 else
                       "NOT a win at the operating point — keep default OFF")
    print(verdict)


# --- M3 (SOLO GPU): packed-experts (gather_qmm) real-model graduation gate ----------------------------
# The routed-expert RESIDENT-MEMORY lever: hold the int4 experts PACKED and dispatch via mx.gather_qmm
# instead of dequantizing them to bf16 + mx.gather_mm. They are the SAME int4 codes (only the kernel
# differs: gather_qmm fuses the dequant; the bf16 path materializes a bf16 weight via mx.dequantize THEN
# runs dense gather_mm), so the forward is greedy-exact and the teacher-forced ppl is unchanged — the ONLY
# thing that moves is resident memory (the bf16-dequant path holds the experts as bf16; packed keeps them
# int4). NB the bf16-dequant path is the LOSSIER of the two — it rounds each dequantized weight to bf16
# before the matmul; packed keeps full precision in the fused dequant, so any tiny delta is the REFERENCE's
# bf16 rounding, not packed drifting. This gate loads the real bake TWICE (packed_experts False, then
# True), ONE model resident at a time (SOLO; an over-subscribed load OOM-reboots the host), and asserts
# (a) a large resident drop, (b) greedy-exact 48-token trace, (c) teacher-forced ppl + top-1 agreement
# unchanged on REAL prose (the CLAUDE.md arbiter — NOT synthetic ids, on which a near-uniform distribution
# amplifies bf16 rounding into a misleading NLL swing). Both configs keep the graduated packed MIXER
# (option B, packed=True) — ONLY packed_experts differs, isolating the routed-expert kernel.
PPL_MAX_TOKENS = 256              # teacher-forced positions cap (the prose is shorter, so all of it is used)
GREEDY_PROMPT_LEN = 64            # real-prose prefix that seeds the greedy trace
GREEDY_GEN = 48                   # greedy tokens compared across the two configs (autoregressive top-1)

# The repo's teacher-forced ppl fixture (mirrors parity/ppl.py) — ordinary English prose, on which a
# coherent runtime is confident, so the packed-vs-bf16 ppl delta reflects bf16 rounding (<~1%), not a
# quant divergence (a real bug blows ppl up by orders, like the prior project's ~165).
PROSE = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in sugars. Inside the chloroplasts, chlorophyll "
    "absorbs sunlight, which drives the splitting of water molecules into oxygen, protons, and "
    "electrons. The oxygen is released into the atmosphere as a byproduct, while the energy "
    "captured is used to fix carbon dioxide from the air into glucose. This remarkable reaction "
    "sustains nearly all life on Earth, forming the base of the food chain and regulating the "
    "balance of oxygen and carbon dioxide in the atmosphere. Without photosynthesis, the planet "
    "would be unable to support the diversity of organisms that depend, directly or indirectly, "
    "on plants for food and breathable air."
)


def _teacher_forced(inner: Qwen35ResidentModel, ids: list[int]) -> tuple[float, float, list[int]]:
    """(mean cross-entropy, top-1 next-token accuracy vs target, per-position argmax) over a token
    sequence via the all-position prefill forward (``__call__`` with ``caches=None`` -> ``[1,T,vocab]``);
    f32 for a stable ``logsumexp``. ppl = ``exp(mean ce)``. Mirrors parity/ppl.py. The per-position argmax
    lets the gate measure packed-vs-bf16 top-1 AGREEMENT (the CLAUDE.md arbiter)."""
    arr = mx.array(ids)
    logits = inner(arr)[0].astype(mx.float32)                # [T, vocab]
    pred, tgt = logits[:-1], arr[1:]                         # position t predicts token t+1
    ce = mx.logsumexp(pred, axis=-1) - mx.take_along_axis(pred, tgt[:, None], axis=-1)[:, 0]
    am = mx.argmax(pred, axis=-1)
    acc = (am == tgt).astype(mx.float32).mean()
    return float(mx.mean(ce).item()), float(acc.item()), am.tolist()


def _greedy_trace(inner: Qwen35ResidentModel, prompt: list[int], n: int) -> list[int]:
    """Greedy-decode ``n`` tokens from ``prompt`` on a fresh cache (single stream, deterministic) via the
    cached decode path serving uses (``__call__`` with ``caches`` given). Returns the token trace to be
    diffed bf16-vs-packed (greedy-exact)."""
    cache = inner.make_caches()
    logits = inner(prompt, caches=cache, offset=0)           # prefill the prompt through the cache
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


def _measure_config(packed_experts: bool, ppl_ids: list[int], prompt: list[int]) -> dict:
    """Load ONE config of the real bake, measure resident GiB + greedy trace + teacher-forced ppl/top-1,
    then return them; the model is the only strong ref inside, so it is released on return (SOLO: only
    this one model is ever resident). Both configs keep the graduated packed mixer; ONLY ``packed_experts``
    differs, so any bf16-vs-packed difference isolates the routed-expert kernel (gather_mm dequant vs
    gather_qmm)."""
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    inner = Qwen35ResidentModel(ART, packed_experts=packed_experts)
    load_s = time.perf_counter() - t0
    resident_gib = _gib(mx.get_active_memory())              # resident weight set right after load
    ce, acc, argmax = _teacher_forced(inner, ppl_ids)
    trace = _greedy_trace(inner, prompt, GREEDY_GEN)
    return {"resident_gib": resident_gib, "peak_gib": _gib(mx.get_peak_memory()),
            "ppl": math.exp(ce), "acc": acc, "argmax": argmax, "trace": trace, "load_s": load_s}


def run_packed_experts_gate() -> None:
    """M3 (SOLO GPU): the real-model packed-experts graduation gate. Loads the int4-g64 bake twice (bf16-
    dequant experts, then packed int4 experts), ONE model resident at a time, and asserts (a) a large
    resident-memory drop, (b) greedy-exact 48-token trace, (c) teacher-forced ppl unchanged + near-total
    top-1 agreement on real prose. Green => graduate ``packed_experts=True`` as the constructor default."""
    mx.set_wired_limit(int(140 * 1024 ** 3))     # bf16-expert resident + headroom (well under 490 ceiling)
    tok = Qwen35Tokenizer.from_pretrained(ART)
    ppl_ids = tok.encode(PROSE, add_bos=True)[:PPL_MAX_TOKENS]   # add_bos no-op for Qwen3.5 (no BOS)
    prompt = ppl_ids[:GREEDY_PROMPT_LEN]
    print("\n=== Qwen3.6-35B-A3B int4-g64 M3 packed-experts gate (SOLO; one model resident at a time) ===")
    print(f"routed experts: bf16-dequant (gather_mm) vs packed int4 (gather_qmm), SAME codes. teacher-forced "
          f"on {len(ppl_ids)} real-prose tok. expect greedy-exact + ppl/top-1 unchanged; win = RESIDENT mem.")
    res: dict[bool, dict] = {}
    for pe in (False, True):                     # bf16 reference first, then packed
        res[pe] = _measure_config(pe, ppl_ids, prompt)
        mx.clear_cache()                          # fully release this model before the next loads
        gc.collect()
        mx.clear_cache()
        r, tag = res[pe], ("packed int4 (gather_qmm)" if pe else "bf16-dequant (gather_mm)")
        print(f"  packed_experts={str(pe):<5} [{tag:<24}]  resident={r['resident_gib']:6.1f} GiB  "
              f"peak={r['peak_gib']:6.1f} GiB  ppl={r['ppl']:8.4f}  top1={r['acc']:.4f}  load={r['load_s']:.1f}s")
    ref, pk = res[False], res[True]

    drop = ref["resident_gib"] - pk["resident_gib"]
    drop_ok = pk["resident_gib"] < 0.6 * ref["resident_gib"]      # expect ~0.3x; 0.6 is a loose bar
    div = _first_divergence([ref["trace"]], [pk["trace"]])
    greedy_ok = div is None
    rel_ppl = abs(pk["ppl"] - ref["ppl"]) / ref["ppl"]
    ppl_ok = rel_ppl < 0.02                                       # <2% : bf16 rounding, NOT a quant divergence
    n_pos = min(len(ref["argmax"]), len(pk["argmax"]))
    agree = sum(ref["argmax"][i] == pk["argmax"][i] for i in range(n_pos)) / max(n_pos, 1)
    agree_ok = agree >= 0.98                                      # near-total; rare near-tie flips are bf16 noise

    print(f"\n  (a) resident drop : {ref['resident_gib']:.1f} -> {pk['resident_gib']:.1f} GiB "
          f"(-{drop:.1f}, {pk['resident_gib'] / ref['resident_gib']:.2f}x)  {'OK' if drop_ok else 'FAIL'}")
    print(f"  (b) greedy-exact  : {GREEDY_GEN} tokens  {'OK' if greedy_ok else 'FAIL'}"
          + ("" if greedy_ok else f"  [DIFF at step {div[1]}: bf16={div[2]} packed={div[3]}]"))
    print(f"  (c) ppl unchanged : {ref['ppl']:.4f} -> {pk['ppl']:.4f} (rel {rel_ppl:.2%}); top-1 agree "
          f"{agree:.4f} (bf16/packed vs target {ref['acc']:.4f}/{pk['acc']:.4f})  "
          f"{'OK' if (ppl_ok and agree_ok) else 'FAIL'}")

    if not (drop_ok and greedy_ok and ppl_ok and agree_ok):
        print("FAIL — packed experts not equivalent to / not lighter than bf16 (see above); keep default OFF")
        raise SystemExit(1)
    print(f"PASS — packed experts greedy-exact + ppl/top-1 unchanged at {pk['resident_gib']:.0f} GiB "
          f"resident (vs {ref['resident_gib']:.0f} GiB bf16) -> graduate packed_experts=True")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "experts":
        run_packed_experts_gate()                # M3: packed-experts memory + greedy + NLL gate
    else:
        bs = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else BATCH_SIZES
        run(bs)
