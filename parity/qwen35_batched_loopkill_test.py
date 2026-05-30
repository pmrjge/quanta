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

Checks (whole-model B in {1} + a RAGGED B=3 with three distinct prompt lengths → ragged offsets, the
real serving case; these exercise BOTH halves of the hybrid end-to-end — the tiny cfg has 2 GDN + 1 GQA
layers — plus a focused §GDN bit-exact unit):

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

from parity.qwen35_batched_test import _build_random_model, _cfg, _maxdiff, _wrap_batched
from quanta.qwen35.batched_runtime import (
    QWEN35_BATCHED_LOOPKILL_DEFAULT,
    Qwen35BatchedResidentModel,
    _gdn_step_batched,
    _gdn_step_through_cache,
)
from quanta.qwen35.decode import _GDNLayerState

LOGIT_TOL = 5e-3  # bf16: per-stream RoPE (bit-exact) + padded-SDPA tiling reorder; HARD gate is greedy tokens


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


GDN_TOL = 1e-4  # B>1: the fp32 projection-GEMM batch-M-dependence ([B,1,h]@W tiles over M differently
#                 than B separate [1,1,h]@W — measured ~1e-6, propagated through the fp32 recurrence to
#                 ~1e-7 here). A real state-gather/scatter/seed bug is O(0.1+); this margin separates the
#                 two by ~1000×. B=1 (no concat) is a strict passthrough → bit-exact, gated == 0 below.


def _test_gdn_loopkill(bm: Qwen35BatchedResidentModel) -> bool:
    """§GDN (M2): the batched Gated-DeltaNet recurrence (:func:`_gdn_step_batched`) == the per-stream
    :func:`_gdn_step_through_cache`, row-for-row — both the output residual AND the committed
    ``(conv,recurrent)`` state + offset. The GDN decode has no cross-row op (conv/recurrence/both gated
    RMSNorms act per row) and no positional term (position lives in the recurrent state), so the ONLY
    batch-size sensitivity is the projection GEMM's accumulation order: **B=1 is bit-exact** (the
    ``b==1`` path feeds identical ``[1,1,hidden]`` shapes — a strict passthrough), and **B>1 matches at
    fp tolerance** (``GDN_TOL``, the ~1e-7 fp32 ``[B,...]@W`` reorder — far tighter than the GQA half,
    which additionally reorders the SDPA softmax). Greedy-token-stable (the whole-model ragged check
    above confirms greedy agreement); this isolates the M2 unit from GQA's noise to gate it precisely.

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


def run() -> None:
    cfg = _cfg()
    model = _build_random_model(cfg, seed=1)
    bm = _wrap_batched(model, max_batch=8)
    ok = True
    print("\n=== Qwen3.5 #153 hybrid loop-kill parity (loop-kill ON == per-stream loop; tiny, model-free) ===")
    ok &= _test_path_exercised(bm)
    ok &= _test_b1_loopkill(bm)
    ok &= _test_ragged_loopkill(bm)
    ok &= _test_off_matches_single(bm)
    ok &= _test_gdn_loopkill(bm)
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
