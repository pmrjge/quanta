"""Gate: Qwen3.5 #153 hybrid loop-kill (``loopkill`` ON) == the per-stream mixer loop (``loopkill`` OFF)
— model-free, tiny tensors.

The Qwen3.5 #153 work: the serving decode step (:func:`quanta.qwen35.batched_runtime.batched_decode_step`)
replaces the per-stream mixer loop with ONE batched mixer across all ``B`` streams, for BOTH halves of
the 3:1 hybrid.

* **M1 — full-attention (GQA) layers:** batched q/k/v/o projections (each weight read ONCE for all
  ``B``), a per-stream RoPE *kernel* loop (offset + dynamic-YaRN ``inv_freq`` differ per stream — the
  bf16-drift trap, so the exact ``mx.fast.rope`` kernel is looped, never a batched reimpl), and the
  shared fused padded SDPA (:func:`quanta.modeling.batched_attention.batched_decode_attention_kv`, the
  #153 primitive InternLM2.5 / Nemotron already use). **Greedy-exact** (the padded-SDPA reorder).
* **M2 — linear-attention (Gated DeltaNet) layers** (45 of 60, the bigger lever): gather the ``B``
  streams' recurrent ``(conv,recurrent)`` state into a ``[B,...]`` batch, run ONE
  :meth:`GatedDeltaNet.__call__` recurrence (so the big in/out projections + conv weights read ONCE
  for all ``B``), scatter+commit the new state back per-stream (:func:`_gdn_step_batched`). No cross-row
  op and no positional term, so **B=1 is bit-exact** and **B>1 matches at fp tolerance** (the ~1e-7 fp32
  projection-GEMM batch-M reorder — no SDPA softmax to reorder, so tighter than the GQA half).

**Arbiter = greedy-token agreement** (the decode that actually ships): every stream must emit the SAME
argmax token with the loop-kill ON as with the proven per-stream loop OFF, at every decode step. Logits
match only up to the padded-SDPA reduction-order ULPs (the loop-kill crosses the per-stream
``mask=None`` SDPA → the batched ``mask=<pad>`` SDPA — argmax-stable fp noise, NOT a logic change; the
same equivalence class the project accepts for tiled/batched paths — see ``feedback_batched_rope_bf16``
and the InternLM2.5 batched-attention gate). ``|Δlogit|`` is reported as a soft diagnostic against
:data:`LOGIT_TOL`.

**§M0 — option-B foundational mechanism (model-free, run first).** Before any runtime is converted to
packed weights, :func:`_test_m0_batchm_packed_vs_dense` locks the matmul mechanism the packed loop-kill
rests on. The real-model bench found the dequant loop-kill is not greedy-exact at B>1 because the runtime
DEQUANTIZES the mixer projections to dense bf16, and a dense-bf16 GEMM reorders its K-reduction across
batch-M (bf16 sums are non-associative; ``feedback_batched_rope_bf16``). M0 found a sharper truth than the
option-B plan assumed: ``mx.quantized_matmul`` is batch-M bit-exact ONLY for M<=~10 (a per-row gemv kernel)
and SWITCHES to a tiled GEMM at B>=12 that reorders too (bf16 ~2.25/proj, *larger* than the dense bug) — so
a *full-batch* packed loop-kill is NOT bit-exact at the B=32 operating point. The fix locked here (user
decision): run the loop-kill projections in row-CHUNKS of <=8 (each an M<=8 ``quantized_matmul``, the
bit-exact regime), which equals the per-stream M=1 loop BIT-FOR-BIT at any B. M1/M2 build this
``_chunked_qmm`` into the packed steppers; M0 proves it model-free.

Since M3 the gate **packs** its tiny model (``loopkill ⇒ packed`` — a dense-bf16 projection reorders
across batch-M; the runtime enforces it), so every whole-model check below runs the PRODUCTION packed
config (``nn.QuantizedLinear`` mixers); §M1/§M2 additionally force multi-chunk splitting on isolated
packed mixers. Checks (whole-model B in {1} + a RAGGED B=3 with three distinct prompt lengths → ragged
offsets, the real serving case; these exercise BOTH halves of the hybrid end-to-end — the tiny cfg has
2 GDN + 1 GQA layers — plus a focused §GDN bit-exact unit):

  1. **B=1 loop-kill == per-stream loop** — greedy-exact over a multi-step decode; ``|Δlogit|`` tiny
     (B=1 differs only by the GQA SDPA ``mask=zeros`` vs ``mask=None``; GDN is bit-exact).
  2. **ragged B=3 loop-kill == per-stream loop** — greedy-exact; exercises the GQA padded SDPA over
     three different context lengths AND the batched GDN recurrence over three different states.
  3. **regression: per-stream loop (OFF) still == single-stream** — the restructure of
     ``batched_decode_step`` (the ``if loopkill / else`` split) must not perturb the OFF path; each
     ragged stream's OFF decode equals its own ``B=1`` single-stream decode (fp tolerance). The existing
     ``parity/qwen35_batched_test.py`` (default flag OFF) is the broader regression; this is a focused
     ragged-offset guard.
  4. **§GDN: batched recurrence == per-stream** — a focused unit on :func:`_gdn_step_batched` vs
     :func:`_gdn_step_through_cache` over B streams seeded with DISTINCT recurrent states (incl. a fresh
     stream → the zero-seed path): the output residual AND the committed ``(conv,recurrent)`` state +
     offset match. GDN has no cross-row op, so **B=1 is bit-exact** (a strict passthrough) and **B>1
     matches at fp tolerance** (the ~1e-7 fp32 projection-GEMM batch-M reorder — far tighter than the
     GQA half, which also reorders the SDPA softmax). Isolates the M2 unit from GQA's ULP noise (the
     whole-model tests above can only gate greedy-exact because they carry it).

Seeding is loop-kill-agnostic: streams are seeded via :meth:`Qwen35BatchedResidentModel.prefill`, which
pins ``loopkill=False`` (the single-stream contract), so ON and OFF decode from a byte-identical cache —
isolating exactly what M1 changes (the decode GQA step).

    uv run --with numpy python -m parity.qwen35_batched_loopkill_test

The real-model B-sweep (ON vs OFF throughput + greedy-exact) lands at M4 in ``parity/qwen35_batched_bench.py``;
this gate is its parity prerequisite. The default stays OFF (rule 4) until M3 flips
:data:`quanta.qwen35.batched_runtime.QWEN35_BATCHED_LOOPKILL_DEFAULT` after that bench.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from parity.qwen35_batched_test import _build_random_model, _cfg, _maxdiff, _wrap_batched
from quanta.qwen35.attention import KVCache, Qwen35Attention
from quanta.qwen35.batched_runtime import (
    QWEN35_BATCHED_LOOPKILL_DEFAULT,
    QWEN35_LOOPKILL_CHUNK,
    Qwen35BatchedResidentModel,
    _gdn_step_batched,
    _gdn_step_through_cache,
    _gqa_step_through_cache,
)
from quanta.qwen35.decode import _GDNLayerState
from quanta.qwen35.gated_deltanet import GatedDeltaNet

LOGIT_TOL = 5e-3  # bf16: per-stream RoPE (bit-exact) + padded-SDPA tiling reorder; HARD gate is greedy tokens


# --- §M0: packed-projection batch-M parity (the option-B foundational mechanism, model-free) ----------
# Decisive shape = the bench's root-cause micro-test ([B,1,4096]@[4096,12288]). M0 found a sharper truth
# than the PLAN assumed: mx.quantized_matmul is batch-M BIT-EXACT only for M<=~10 (a per-row gemv kernel);
# at M>=12 it switches to a tiled GEMM that REORDERS the K-reduction (bf16 ~2.25/proj, LARGER than the
# dense-bf16 bug it was meant to fix). So a *full-batch* packed loop-kill is NOT bit-exact at the B=32
# operating point. Fix (user decision): chunk the loop-kill projections into <=8-row slices — each an
# M<=8 quantized_matmul, the bit-exact regime — so it equals the per-stream M=1 loop bit-for-bit at ANY B.
_M0_IN, _M0_OUT = 4096, 12288
_M0_GS, _M0_BITS = 64, 4
_M0_CHUNK = 8                        # loop-kill sub-batch size; M1+ chunk the batched projections by this


def _chunked_qmm(linear, x: mx.array, chunk: int) -> mx.array:
    """Apply ``linear`` over the leading batch in row-chunks of ``<= chunk`` and concat: ``[B,1,in] ->
    [B,1,out]``. Each chunk is an ``M<=chunk`` matmul — the bit-exact regime of ``mx.quantized_matmul``
    (no K-reduction reorder) — so a chunked quantized projection equals the per-stream ``M=1`` loop
    bit-for-bit at ANY ``B``. The primitive the packed loop-kill steppers (M1/M2) build on."""
    b = int(x.shape[0])
    if b <= chunk:
        return linear(x)
    return mx.concatenate([linear(x[i:i + chunk]) for i in range(0, b, chunk)], axis=0)


def _batchm_diff(linear, in_dims: int, B: int, seed: int, *, chunk: int | None = None) -> float:
    """Worst per-row ``|Δ|`` between ONE forward over a ``[B,1,in]`` batch (full-batch if ``chunk`` is
    None, else chunked) and ``B`` separate per-stream ``[1,1,in]`` (``M=1``) forwards. Zero ⇔ the matmul
    is batch-M invariant — it does not reorder its reduction relative to the per-stream loop."""
    mx.random.seed(seed)
    x = mx.random.normal((B, 1, in_dims)).astype(mx.bfloat16)
    yb = linear(x) if chunk is None else _chunked_qmm(linear, x, chunk)
    mx.eval(yb)
    return max(_maxdiff(yb[s:s + 1], linear(x[s:s + 1])) for s in range(B))


def _test_m0_batchm_packed_vs_dense() -> bool:
    """§M0 (option-B foundational proof, model-free): locks the matmul mechanism the packed loop-kill
    rests on BEFORE the runtime is touched (rule 4 / rule 6). Builds ``nn.QuantizedLinear`` from
    ``mx.quantize`` codes (= the artifact's packed triplet) and a dense-bf16 ``nn.Linear`` from
    ``mx.dequantize`` of the SAME codes, so the ONLY difference between paths is the matmul kernel:

      * **dense-bf16 (the bug):** bit-exact only at ``B=1`` (why the bench's B=1 was greedy-exact);
        reorders the K-reduction for ``B>1`` (the |Δlogit|≈1.3 the bench rejected).
      * **quantized full-batch:** bit-exact only for ``M<=~10`` (a per-row gemv kernel); SWITCHES to a
        tiled GEMM at ``B>=12`` that reorders too. So a full-batch packed loop-kill is NOT bit-exact at
        the ``B=32`` operating point — the PLAN's premise was too strong.
      * **quantized CHUNKED (the fix):** chunking the batch into ``<=8`` rows keeps every matmul in the
        bit-exact ``M<=8`` regime, so it equals the per-stream ``M=1`` loop BIT-FOR-BIT at
        ``B∈{1,4,8,32}``. This is the mechanism M1/M2 build into the steppers.

    If a future MLX shifts the gemv→gemm threshold below 8, ``chunk_exact`` AND ``full_threshold`` both
    fail loudly — the signal to drop ``_M0_CHUNK`` (the gate self-protects the chunking decision)."""
    mx.random.seed(153)
    w = mx.random.normal((_M0_OUT, _M0_IN)).astype(mx.bfloat16)                  # [out,in]; in % gs == 0
    w_q, scales, biases = mx.quantize(w, group_size=_M0_GS, bits=_M0_BITS)

    ql = nn.QuantizedLinear(_M0_IN, _M0_OUT, bias=False, group_size=_M0_GS, bits=_M0_BITS)
    ql.weight, ql.scales, ql.biases = w_q, scales, biases                       # populate from codes (= triplets)

    deq = nn.Linear(_M0_IN, _M0_OUT, bias=False)
    deq.weight = mx.dequantize(w_q, scales, biases, group_size=_M0_GS, bits=_M0_BITS)  # SAME codes, dense bf16

    bs = (1, 4, 8, 32)
    qf = {B: _batchm_diff(ql, _M0_IN, B, 400 + B) for B in bs}                   # quantized, full batch
    qc = {B: _batchm_diff(ql, _M0_IN, B, 400 + B, chunk=_M0_CHUNK) for B in bs}  # quantized, chunked <=8
    df = {B: _batchm_diff(deq, _M0_IN, B, 400 + B) for B in bs}                  # dense bf16, full batch

    chunk_exact = all(qc[B] == 0.0 for B in bs)            # THE FIX: chunked quantized bit-exact at every B
    full_threshold = qf[8] == 0.0 and qf[32] > 0.0         # WHY: full-batch quantized exact <=8, reorders @32
    dense_bug = df[1] == 0.0 and df[4] > 0.0 and df[32] > 0.0   # THE BUG: dense bf16 reorders for B>1
    ok = chunk_exact and full_threshold and dense_bug
    qcs = " ".join(f"B{B}={qc[B]:.2e}" for B in bs)
    qfs = " ".join(f"B{B}={qf[B]:.2e}" for B in bs)
    dfs = " ".join(f"B{B}={df[B]:.2e}" for B in bs)
    print(f"  [{'OK' if ok else 'FAIL'}] §M0 batch-M: chunked-{_M0_CHUNK} quantized BIT-EXACT [{qcs}] "
          f"(fix={chunk_exact}) | full-batch quantized [{qfs}] (reorders@>=12={full_threshold}) | "
          f"dense-bf16 [{dfs}] (bug@B>1={dense_bug})")
    return ok


def _run(bm: Qwen35BatchedResidentModel, prompts: list[list[int]], n_decode: int,
         loopkill: bool) -> tuple[list[list[mx.array]], list[list[int]], list[int]]:
    """Seed ``B = len(prompts)`` streams (ragged lengths → ragged offsets) then decode ``n_decode``
    steps under the given ``loopkill`` setting, feeding each stream its own greedy argmax forward.

    Seeding uses :meth:`prefill` (``loopkill=False`` always), so ON and OFF decode from identical
    caches. Returns (per_step_logits[step][stream], per_step_tokens[step][stream], final_offsets)."""
    b = len(prompts)
    caches = [bm.make_caches() for _ in range(b)]
    last = []
    for s, p in enumerate(prompts):
        lg = bm.prefill(p, caches[s])           # [1,1,vocab]; loop-kill-agnostic (prefill pins OFF)
        mx.eval(lg)
        last.append(lg)
    offsets = [len(p) for p in prompts]
    toks = [int(mx.argmax(last[s][0, -1]).item()) for s in range(b)]

    bm._loopkill = loopkill                      # toggle the decode GQA path for this run
    step_logits: list[list[mx.array]] = []
    step_tokens: list[list[int]] = []
    for _ in range(n_decode):
        per_stream = bm.step_batch(toks, caches, list(offsets))
        mx.eval(per_stream)
        step_logits.append([per_stream[s] for s in range(b)])
        toks = [int(mx.argmax(per_stream[s][0, -1]).item()) for s in range(b)]
        step_tokens.append(list(toks))
        offsets = [o + 1 for o in offsets]
    return step_logits, step_tokens, offsets


def _compare(ref_l: list[list[mx.array]], ref_t: list[list[int]],
             tst_l: list[list[mx.array]], tst_t: list[list[int]]) -> tuple[bool, float]:
    """Greedy-token agreement (hard) + worst ``|Δlogit|`` (soft) over all steps × streams."""
    tok_match = ref_t == tst_t
    worst = max(_maxdiff(ref_l[p][s], tst_l[p][s])
                for p in range(len(ref_l)) for s in range(len(ref_l[0])))
    return tok_match, worst


def _test_path_exercised(bm: Qwen35BatchedResidentModel) -> bool:
    """The tiny cfg must contain >= 1 full-attention (GQA) AND >= 1 linear-attention (GDN) layer, else a
    loop-kill branch is never hit and the gate would be vacuously green (rule 6: never report a no-op as
    a pass). The default flag must be OFF (rule 4 — M3 flips it after the real-model bench)."""
    n_full = sum(0 if blk.is_linear else 1 for blk in bm.layers)
    n_lin = sum(1 if blk.is_linear else 0 for blk in bm.layers)
    ok = n_full >= 1 and n_lin >= 1 and QWEN35_BATCHED_LOOPKILL_DEFAULT is False
    print(f"  [{'OK' if ok else 'FAIL'}] loop-kill path exercised: {n_full} full-attn + {n_lin} "
          f"linear-attn layer(s); default flag OFF={QWEN35_BATCHED_LOOPKILL_DEFAULT is False}")
    return ok


def _test_loopkill_requires_packed() -> bool:
    """rule 4/6 enforcement: the loop-kill MUST refuse to run on a non-packed (dense-bf16) runtime — its
    batched mixer projections would reorder across batch-M (the real-model bench's |Δlogit|≈1.3 bug).
    A bf16 model wrapped ``packed=False``, then toggled ``_loopkill=True``, must RAISE on
    :meth:`step_batch` — the runtime toggle cannot bypass the guard (it is re-checked every step)."""
    cfg = _cfg()
    model = _build_random_model(cfg, seed=5)              # bf16 — deliberately NOT packed
    bm = _wrap_batched(model, max_batch=4, packed=False)
    bm._loopkill = True                                   # toggle the loop-kill on the non-packed runtime
    raised = False
    try:
        bm.step_batch([3], [bm.make_caches()], [0])
    except ValueError:
        raised = True
    print(f"  [{'OK' if raised else 'FAIL'}] loop-kill ⇒ packed enforced (step_batch raises on a "
          f"non-packed runtime): {raised}")
    return raised


def _test_b1_loopkill(bm: Qwen35BatchedResidentModel) -> bool:
    """B=1: loop-kill ON == per-stream loop OFF (greedy-exact; |Δlogit| tiny — only the SDPA mask path
    differs at B=1)."""
    prompts = [[3, 9, 15, 27, 42]]
    n_dec = 5
    ref_l, ref_t, ref_off = _run(bm, prompts, n_dec, loopkill=False)
    tst_l, tst_t, tst_off = _run(bm, prompts, n_dec, loopkill=True)
    tok_match, worst = _compare(ref_l, ref_t, tst_l, tst_t)
    ok = tok_match and worst < LOGIT_TOL and ref_off == tst_off
    print(f"  [{'OK' if ok else 'FAIL'}] B=1 loop-kill == per-stream loop  greedy={tok_match} "
          f"|Δlogit|={worst:.2e} offs={tst_off}")
    return ok


def _test_ragged_loopkill(bm: Qwen35BatchedResidentModel) -> bool:
    """Ragged B=3 (three distinct prompt lengths → ragged offsets): loop-kill ON == per-stream loop OFF
    (greedy-exact). Exercises the padded SDPA over different context lengths — the serving scenario."""
    prompts = [[3, 9, 15], [5, 19], [7, 11, 2, 8, 4]]   # lengths 3 / 2 / 5 → offsets 3 / 2 / 5
    n_dec = 5
    ref_l, ref_t, ref_off = _run(bm, prompts, n_dec, loopkill=False)
    tst_l, tst_t, tst_off = _run(bm, prompts, n_dec, loopkill=True)
    tok_match, worst = _compare(ref_l, ref_t, tst_l, tst_t)
    ok = tok_match and worst < LOGIT_TOL and ref_off == tst_off
    print(f"  [{'OK' if ok else 'FAIL'}] ragged B=3 loop-kill == per-stream loop  greedy={tok_match} "
          f"|Δlogit|={worst:.2e} offs={tst_off}")
    return ok


def _test_off_matches_single(bm: Qwen35BatchedResidentModel) -> bool:
    """Regression for the M1 ``batched_decode_step`` restructure: the OFF path (per-stream loop) must
    still equal a true single-stream decode for each ragged stream (fp tolerance). Guards against the
    new ``if loopkill / else`` split silently changing the proven per-stream path."""
    prompts = [[3, 9, 15], [5, 19], [7, 11, 2, 8, 4]]
    n_dec = 4
    bat_l, _, _ = _run(bm, prompts, n_dec, loopkill=False)         # per-stream loop inside the batch
    worst = 0.0
    for s, p in enumerate(prompts):
        sg_l, _, _ = _run(bm, [p], n_dec, loopkill=False)          # B=1 single-stream reference
        worst = max(worst, max(_maxdiff(bat_l[step][s], sg_l[step][0]) for step in range(n_dec)))
    ok = worst < LOGIT_TOL
    print(f"  [{'OK' if ok else 'FAIL'}] OFF path == single-stream (restructure regression)  "
          f"|Δ|={worst:.2e}")
    return ok


GDN_TOL = 1e-4  # loose ceiling: since M3 the shared model is PACKED, so the batched recurrence is
#                 bit-exact vs per-stream at any B (M≤chunk gemv + fixed-axis fp32 recurrence — both
#                 batch-invariant). A real state-gather/scatter/seed bug is O(0.1+); this margin clears
#                 the bit-exact 0.0 with ~1000× headroom (the bound was the bf16-era ~1e-7 reorder).


def _test_gdn_loopkill(bm: Qwen35BatchedResidentModel) -> bool:
    """§GDN: the batched Gated-DeltaNet recurrence (:func:`_gdn_step_batched`) == the per-stream
    :func:`_gdn_step_through_cache`, row-for-row — both the output residual AND the committed
    ``(conv,recurrent)`` state + offset. The GDN decode has no cross-row op (conv/recurrence/both gated
    RMSNorms act per row) and no positional term (position lives in the recurrent state), so the ONLY
    batch-size sensitivity is the projection matmul's accumulation order. Since M3 the shared ``bm`` is
    PACKED (``loopkill ⇒ packed``), so this exercises the ``b≤chunk`` full-batch passthrough (default
    chunk 8, B≤3 → one ``m()`` call) and is **bit-exact at B=1 AND B>1** (M≤chunk ``mx.quantized_matmul``
    is bit-exact vs ``M=1`` — M0 — and the fp32 recurrence reduces over fixed axes). Complements §M1,
    which forces multi-chunk splitting (chunk=2). Greedy-token-stable; isolates the GDN unit from the
    GQA SDPA softmax noise to gate it precisely.

    Exercises a real linear-attention block over ``B`` streams seeded with DISTINCT recurrent states
    (different numbers of prior decode steps → different ``(conv,recurrent)``), INCLUDING a fresh stream
    (``seeds[s] == 0`` → the ``None`` → zero-seed gather path, rule 6). ref / test caches start
    byte-identical (:meth:`_GDNLayerState._copy`) so the single compared step isolates exactly what the
    batched recurrence changes."""
    lin_idx = next((i for i, blk in enumerate(bm.layers) if blk.is_linear), None)
    assert lin_idx is not None, "tiny cfg must contain >= 1 linear-attention layer (rule 6: exercise the path)"
    blk = bm.layers[lin_idx]
    hidden = bm.cfg.hidden_size
    res: dict[int, tuple[float, float, bool]] = {}     # B -> (worst_out, worst_state, offsets_ok)
    for seeds in ([3], [2, 0, 4]):                     # B=1 (seeded) and ragged B=3 (incl. a fresh stream)
        b = len(seeds)
        mx.random.seed(202 + b)
        # seed each stream's per-stream state with seeds[s] random decode steps (fresh stays None)
        lcs_ref = [_GDNLayerState() for _ in range(b)]
        for s in range(b):
            for _ in range(seeds[s]):
                _gdn_step_through_cache(blk, lcs_ref[s], mx.random.normal((1, 1, hidden)).astype(mx.bfloat16))
        # copy the seeded state for the test path (shares immutable tensors; diverges on commit)
        lcs_test = [lcs_ref[s]._copy() for s in range(b)]
        # ONE compared decode step from the identical seeded state (same xs fed to both paths)
        xs = [mx.random.normal((1, 1, hidden)).astype(mx.bfloat16) for _ in range(b)]
        out_ref = [_gdn_step_through_cache(blk, lcs_ref[s], xs[s]) for s in range(b)]   # B × [1,1,hidden]
        x_stacked = mx.concatenate(xs, axis=0) if b > 1 else xs[0]                      # [B,1,hidden]
        h_norm = blk.input_layernorm(x_stacked)
        y = _gdn_step_batched(blk.mixer, lcs_test, h_norm)                              # [B,1,hidden]
        out_test = x_stacked + y
        mx.eval(out_ref, out_test, [lc.recurrent_state for lc in lcs_test],
                [lc.conv_state for lc in lcs_test])
        wo = max(_maxdiff(out_ref[s], out_test[s:s + 1]) for s in range(b))             # output residual
        ws = max(max(_maxdiff(lcs_ref[s].recurrent_state, lcs_test[s].recurrent_state),
                     _maxdiff(lcs_ref[s].conv_state, lcs_test[s].conv_state)) for s in range(b))
        oo = all(lcs_ref[s].offset == lcs_test[s].offset == seeds[s] + 1 for s in range(b))
        res[b] = (wo, ws, oo)
    (o1, s1, f1), (on, sn, fn) = res[1], res[3]
    b1_exact = o1 == 0.0 and s1 == 0.0 and f1                       # B=1: strict passthrough, bit-exact
    bn_ok = on < GDN_TOL and sn < GDN_TOL and fn                    # B>1: fp tolerance (GEMM reorder)
    ok = b1_exact and bn_ok
    print(f"  [{'OK' if ok else 'FAIL'}] §GDN batched recurrence == per-stream  "
          f"B=1 bit-exact={b1_exact} (out|Δ|={o1:.0e} state|Δ|={s1:.0e})  "
          f"B=3 out|Δ|={on:.2e} state|Δ|={sn:.2e} (<{GDN_TOL:.0e}) offs_ok={f1 and fn}")
    return ok


# --- §M1: packed + chunked GDN mixer (option-B M1) ------------------------------------------------
_PACK_GS, _PACK_BITS = 32, 4   # tiny-cfg GDN proj in-dims are all 32 → one affine group at g32


def _pack_gdn_block(blk, group_size: int, bits: int) -> None:
    """Swap a GDN block's five ``nn.Linear`` projections for ``nn.QuantizedLinear`` built from
    ``mx.quantize`` of the SAME weights — the model-free analogue of
    ``runtime._load_block(packed=True)`` for the GDN half (``runtime._packed_linear`` does the same
    from the artifact's packed triplet). Mutates the block in place; ``group_size`` must divide each
    projection's in-dim."""
    m = blk.mixer
    assert isinstance(m, GatedDeltaNet)
    for proj in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"):
        lin = getattr(m, proj)
        out_dims, in_dims = int(lin.weight.shape[0]), int(lin.weight.shape[1])
        wq, sc, bi = mx.quantize(lin.weight, group_size=group_size, bits=bits)
        ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=group_size, bits=bits)
        ql.weight, ql.scales, ql.biases = wq, sc, bi
        setattr(m, proj, ql)


def _test_gdn_packed_chunked() -> bool:
    """§M1: PACKED (``nn.QuantizedLinear``) GDN with the CHUNKED loop-kill == the packed GDN per-stream
    loop, **BIT-EXACT at B=1 AND B>1** — the M1 deliverable.

    The packed runtime (``_load_block(packed=True)``) holds the GDN projections as
    ``mx.quantized_matmul`` (batch-M bit-exact, rule 1) and the batched stepper applies the recurrence
    in ``≤chunk`` row-slices (so each projection matmul stays in the M0-proven bit-exact regime). Here
    the chunk is forced to **2** so the ragged ``B=3`` spans TWO chunks — exercising the
    chunk-boundary / concat / per-stream-``commit`` machinery, not just the ``b≤chunk`` passthrough.

    Because GDN has no cross-row op (conv window / gated-delta recurrence / both gated RMSNorms all act
    per row, no positional term) AND each chunked ``mx.quantized_matmul`` is the per-row gemv that is
    bit-exact vs ``M=1`` (M0 ``c503657``: bit-exact for M≤~10), the chunked packed loop-kill equals the
    per-stream loop **bit-for-bit at every B** — no fp slack (the bf16 ~1e-7 projection-reorder
    machinery is what §GDN covers; this is the *packed-runtime* equivalence the real-model bench needs).
    Cross-checks the production ``QWEN35_LOOPKILL_CHUNK == _M0_CHUNK`` so the runtime chunk IS the
    M0-validated size (rule 6). Builds its OWN fresh model + packs the first GDN block in place — the
    shared ``bm`` (and every other test) is untouched."""
    assert QWEN35_LOOPKILL_CHUNK == _M0_CHUNK, (
        f"runtime chunk {QWEN35_LOOPKILL_CHUNK} != M0-validated {_M0_CHUNK} (rule 6: keep them linked)")
    cfg = _cfg()
    model = _build_random_model(cfg, seed=11)
    lin_idx = next(i for i, blk in enumerate(model.layers) if blk.is_linear)
    blk = model.layers[lin_idx]
    _pack_gdn_block(blk, _PACK_GS, _PACK_BITS)                 # GDN projections → nn.QuantizedLinear
    hidden = cfg.hidden_size
    res: dict[int, tuple[float, float, bool]] = {}
    for seeds in ([3], [2, 0, 4]):                            # B=1 (seeded) and ragged B=3 (incl fresh)
        b = len(seeds)
        mx.random.seed(303 + b)
        lcs_ref = [_GDNLayerState() for _ in range(b)]
        for s in range(b):
            for _ in range(seeds[s]):
                _gdn_step_through_cache(blk, lcs_ref[s], mx.random.normal((1, 1, hidden)).astype(mx.bfloat16))
        lcs_test = [lcs_ref[s]._copy() for s in range(b)]
        xs = [mx.random.normal((1, 1, hidden)).astype(mx.bfloat16) for _ in range(b)]
        out_ref = [_gdn_step_through_cache(blk, lcs_ref[s], xs[s]) for s in range(b)]      # packed M=1 loop
        x_stacked = mx.concatenate(xs, axis=0) if b > 1 else xs[0]
        h_norm = blk.input_layernorm(x_stacked)
        y = _gdn_step_batched(blk.mixer, lcs_test, h_norm, chunk=2)      # packed, CHUNK=2 → 2 chunks @B=3
        out_test = x_stacked + y
        mx.eval(out_ref, out_test, [lc.recurrent_state for lc in lcs_test],
                [lc.conv_state for lc in lcs_test])
        wo = max(_maxdiff(out_ref[s], out_test[s:s + 1]) for s in range(b))
        ws = max(max(_maxdiff(lcs_ref[s].recurrent_state, lcs_test[s].recurrent_state),
                     _maxdiff(lcs_ref[s].conv_state, lcs_test[s].conv_state)) for s in range(b))
        oo = all(lcs_ref[s].offset == lcs_test[s].offset == seeds[s] + 1 for s in range(b))
        res[b] = (wo, ws, oo)
    (o1, s1, f1), (on, sn, fn) = res[1], res[3]
    b1_exact = o1 == 0.0 and s1 == 0.0 and f1                # B=1: b≤chunk passthrough, strict bit-exact
    bn_exact = on == 0.0 and sn == 0.0 and fn                # B=3: chunked packed gemv == M=1 (M0)
    ok = b1_exact and bn_exact
    print(f"  [{'OK' if ok else 'FAIL'}] §M1 packed+chunked GDN (chunk=2, gs{_PACK_GS}/int{_PACK_BITS}) "
          f"== per-stream  B=1 bit-exact={b1_exact} (out|Δ|={o1:.0e} state|Δ|={s1:.0e})  "
          f"B=3 bit-exact={bn_exact} (out|Δ|={on:.0e} state|Δ|={sn:.0e}) offs_ok={f1 and fn}  "
          f"prod_chunk={QWEN35_LOOPKILL_CHUNK}")
    return ok


# --- §M2: packed + chunked GQA mixer (option-B M2) ------------------------------------------------
GQA_STEP_TOL = 1e-3  # B>1 full-step: q/k/v/o projections are bit-exact once packed+chunked, so the ONLY
#                      divergence is the fused padded-SDPA softmax reorder (~1e-6 single layer); a real
#                      wiring bug is O(0.1+). B=1 (single-stream padded SDPA == mask=None) is ~bit-exact.


def _pack_gqa_block(blk, group_size: int, bits: int) -> None:
    """Swap a GQA block's four q/k/v/o ``nn.Linear`` projections for ``nn.QuantizedLinear`` built from
    ``mx.quantize`` of the SAME weights — the model-free analogue of ``runtime._load_block(packed=True)``
    for the GQA half. Mutates the block in place; ``group_size`` must divide each projection's in-dim."""
    m = blk.mixer
    assert isinstance(m, Qwen35Attention)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = getattr(m, proj)
        out_dims, in_dims = int(lin.weight.shape[0]), int(lin.weight.shape[1])
        wq, sc, bi = mx.quantize(lin.weight, group_size=group_size, bits=bits)
        ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=group_size, bits=bits)
        ql.weight, ql.scales, ql.biases = wq, sc, bi
        setattr(m, proj, ql)


def _test_gqa_packed_chunked() -> bool:
    """§M2: PACKED (``nn.QuantizedLinear``) GQA with the CHUNKED loop-kill == the packed GQA per-stream
    loop — the M2 deliverable. Two parts on a fresh model whose single full-attn block is packed
    (chunk forced to **2** so B=3 spans two chunks, exercising the boundary/concat machinery):

    * **A — projection chunking BIT-EXACT.** ``_project_chunked(x, 2)`` (q/k/v/gate) == per-stream
      ``_project(x[s:s+1])`` and chunked ``o_proj`` == per-stream ``o_proj``, all **== 0.0**. This is
      the core M2 change: each packed ``mx.quantized_matmul`` stays in the M≤chunk gemv regime that M0
      proved bit-exact vs ``M=1`` — so the batched projection equals the per-stream loop bit-for-bit.
    * **B — full decode step.** Chunked :meth:`decode_step_batched` vs per-stream
      :func:`_gqa_step_through_cache` over ragged seeded KV caches: **B=1 bit-exact** (single-stream
      padded SDPA == ``mask=None``) and **B>1 within ``GQA_STEP_TOL``** (the q/k/v/o projections are now
      bit-exact, so the lone divergence is the fused padded-SDPA softmax reduction order — argmax-stable
      ULP, the equivalence the project accepts for batched/tiled SDPA — see ``feedback_batched_rope_bf16``
      + the InternLM2.5 batched-attention gate). Builds its OWN model — the shared ``bm`` is untouched."""
    cfg = _cfg()
    model = _build_random_model(cfg, seed=13)
    full_idx = next(i for i, blk in enumerate(model.layers) if not blk.is_linear)
    blk = model.layers[full_idx]
    _pack_gqa_block(blk, _PACK_GS, _PACK_BITS)               # q/k/v/o → nn.QuantizedLinear
    m = blk.mixer
    hidden = cfg.hidden_size

    # --- Part A: projection chunking is bit-exact vs the per-stream M=1 loop -----------------------
    mx.random.seed(414)
    bp = 3
    x = mx.random.normal((bp, 1, hidden)).astype(mx.bfloat16)
    q_c, k_c, v_c, g_c = m._project_chunked(x, 2)            # chunked (2 chunks @B=3)
    ref = [m._project(x[s:s + 1]) for s in range(bp)]        # per-stream M=1
    q_r = mx.concatenate([r[0] for r in ref], axis=0)
    k_r = mx.concatenate([r[1] for r in ref], axis=0)
    v_r = mx.concatenate([r[2] for r in ref], axis=0)
    g_r = mx.concatenate([r[3] for r in ref], axis=0)
    out = mx.random.normal((bp, 1, m.nh * m.hd)).astype(mx.bfloat16)
    o_c = mx.concatenate([m.o_proj(out[lo:lo + 2]) for lo in range(0, bp, 2)], axis=0)   # chunked
    o_r = mx.concatenate([m.o_proj(out[s:s + 1]) for s in range(bp)], axis=0)            # per-stream
    mx.eval(q_c, k_c, v_c, g_c, q_r, k_r, v_r, g_r, o_c, o_r)
    proj_d = max(_maxdiff(q_c, q_r), _maxdiff(k_c, k_r), _maxdiff(v_c, v_r), _maxdiff(g_c, g_r))
    o_d = _maxdiff(o_c, o_r)
    proj_exact = proj_d == 0.0 and o_d == 0.0

    # --- Part B: full chunked decode step == per-stream loop (B=1 bit-exact; B>1 SDPA ULP) ---------
    res: dict[int, tuple[float, bool]] = {}
    for seeds in ([4], [3, 0, 5]):                           # B=1 and ragged B=3 (incl a fresh stream)
        b = len(seeds)
        mx.random.seed(515 + b)
        kvs_ref = [KVCache() for _ in range(b)]
        for s in range(b):
            for _ in range(seeds[s]):                        # grow each stream's KV to its ragged offset
                _gqa_step_through_cache(blk, kvs_ref[s],
                                        mx.random.normal((1, 1, hidden)).astype(mx.bfloat16), seeds[s])
        kvs_test = [kvs_ref[s]._copy() for s in range(b)]    # byte-identical seeded state for the test path
        xs = [mx.random.normal((1, 1, hidden)).astype(mx.bfloat16) for _ in range(b)]
        out_ref = [_gqa_step_through_cache(blk, kvs_ref[s], xs[s], seeds[s] + 1) for s in range(b)]
        x_stacked = mx.concatenate(xs, axis=0) if b > 1 else xs[0]
        h_norm = blk.input_layernorm(x_stacked)
        y = m.decode_step_batched(h_norm, kv_for_layer=kvs_test,
                                  offsets=[seeds[s] for s in range(b)],
                                  seq_hints=[seeds[s] + 1 for s in range(b)], chunk=2)
        out_test = x_stacked + y
        mx.eval(out_ref, out_test)
        wo = max(_maxdiff(out_ref[s], out_test[s:s + 1]) for s in range(b))
        oo = all(kvs_ref[s].offset == kvs_test[s].offset == seeds[s] + 1 for s in range(b))
        res[b] = (wo, oo)
    (o1, f1), (on, fn) = res[1], res[3]
    b1_exact = o1 < 1e-5 and f1
    bn_ulp = on < GQA_STEP_TOL and fn
    ok = proj_exact and b1_exact and bn_ulp
    print(f"  [{'OK' if ok else 'FAIL'}] §M2 packed+chunked GQA (chunk=2, gs{_PACK_GS}/int{_PACK_BITS}) "
          f"== per-stream  proj bit-exact={proj_exact} (qkvg|Δ|={proj_d:.0e} o|Δ|={o_d:.0e})  "
          f"step B=1|Δ|={o1:.2e} B=3|Δ|={on:.2e} (<{GQA_STEP_TOL:.0e}) offs_ok={f1 and fn}")
    return ok


def _pack_model(model) -> None:
    """Pack EVERY block's mixer projections (GDN ``in_proj_*``/``out_proj`` + GQA ``q/k/v/o``) to
    ``nn.QuantizedLinear`` — the model-free analogue of ``Qwen35ResidentModel(packed=True)`` for the
    whole tiny model. The #153 loop-kill REQUIRES packed (``loopkill ⇒ packed``: a dense-bf16 projection
    reorders across batch-M), so the whole-model loop-kill gate runs the PRODUCTION packed config."""
    for blk in model.layers:
        if blk.is_linear:
            _pack_gdn_block(blk, _PACK_GS, _PACK_BITS)
        else:
            _pack_gqa_block(blk, _PACK_GS, _PACK_BITS)


def run() -> None:
    cfg = _cfg()
    model = _build_random_model(cfg, seed=1)
    _pack_model(model)                                  # loop-kill ⇒ packed (run the production config)
    bm = _wrap_batched(model, max_batch=8, packed=True)
    ok = True
    print("\n=== Qwen3.5 #153 hybrid loop-kill parity (loop-kill ON == per-stream loop; tiny, model-free) ===")
    ok &= _test_m0_batchm_packed_vs_dense()
    ok &= _test_path_exercised(bm)
    ok &= _test_loopkill_requires_packed()
    ok &= _test_b1_loopkill(bm)
    ok &= _test_ragged_loopkill(bm)
    ok &= _test_off_matches_single(bm)
    ok &= _test_gdn_loopkill(bm)
    ok &= _test_gdn_packed_chunked()
    ok &= _test_gqa_packed_chunked()
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
