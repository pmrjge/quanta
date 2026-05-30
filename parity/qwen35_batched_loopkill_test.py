"""Gate: Qwen3.5 #153 GQA loop-kill (``loopkill`` ON) == the per-stream mixer loop (``loopkill`` OFF)
— model-free, tiny tensors.

M1 of the Qwen3.5 #153 work: the serving decode step (:func:`quanta.qwen35.batched_runtime.batched_decode_step`)
replaces the per-stream GQA mixer loop on full-attention layers with ONE batched attention across all
``B`` streams — batched q/k/v/o projections (each weight read ONCE for all ``B``), a per-stream RoPE
*kernel* loop (offset + dynamic-YaRN ``inv_freq`` differ per stream — the bf16-drift trap, so the exact
``mx.fast.rope`` kernel is looped, never a batched reimpl), and the shared fused padded SDPA
(:func:`quanta.modeling.batched_attention.batched_decode_attention_kv`, the #153 primitive InternLM2.5 /
Nemotron already use). The GDN (linear-attn) half stays per-stream until M2.

**Arbiter = greedy-token agreement** (the decode that actually ships): every stream must emit the SAME
argmax token with the loop-kill ON as with the proven per-stream loop OFF, at every decode step. Logits
match only up to the padded-SDPA reduction-order ULPs (the loop-kill crosses the per-stream
``mask=None`` SDPA → the batched ``mask=<pad>`` SDPA — argmax-stable fp noise, NOT a logic change; the
same equivalence class the project accepts for tiled/batched paths — see ``feedback_batched_rope_bf16``
and the InternLM2.5 batched-attention gate). ``|Δlogit|`` is reported as a soft diagnostic against
:data:`LOGIT_TOL`.

Checks (B in {1} and a RAGGED B=3 with three distinct prompt lengths → ragged offsets, the real serving
case the padded SDPA handles):

  1. **B=1 loop-kill == per-stream loop** — greedy-exact over a multi-step decode; ``|Δlogit|`` tiny
     (B=1 differs only by the SDPA ``mask=zeros`` vs ``mask=None``).
  2. **ragged B=3 loop-kill == per-stream loop** — greedy-exact; exercises the padded SDPA over three
     different context lengths (the loop-kill's whole point).
  3. **regression: per-stream loop (OFF) still == single-stream** — the M1 restructure of
     ``batched_decode_step`` (the new ``if loopkill / else`` split) must not perturb the OFF path; each
     ragged stream's OFF decode equals its own ``B=1`` single-stream decode (fp tolerance). The existing
     ``parity/qwen35_batched_test.py`` (default flag OFF) is the broader regression; this is a focused
     ragged-offset guard.

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
)

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
    """The tiny cfg must contain >= 1 full-attention (GQA) layer, else the loop-kill is never hit and
    the gate would be vacuously green (rule 6: never report a no-op as a pass)."""
    n_full = sum(0 if blk.is_linear else 1 for blk in bm.layers)
    ok = n_full >= 1 and QWEN35_BATCHED_LOOPKILL_DEFAULT is False
    print(f"  [{'OK' if ok else 'FAIL'}] loop-kill path exercised: {n_full} full-attn layer(s); "
          f"default flag OFF={QWEN35_BATCHED_LOOPKILL_DEFAULT is False}")
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


def run() -> None:
    cfg = _cfg()
    model = _build_random_model(cfg, seed=1)
    bm = _wrap_batched(model, max_batch=8)
    ok = True
    print("\n=== Qwen3.5 #153 GQA loop-kill parity (loop-kill ON == per-stream loop; tiny, model-free) ===")
    ok &= _test_path_exercised(bm)
    ok &= _test_b1_loopkill(bm)
    ok &= _test_ragged_loopkill(bm)
    ok &= _test_off_matches_single(bm)
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
