"""Nemotron-Ultra U4 / MTP-M3 (A) — compiled T>1 spec-VERIFY graph parity gate.

MTP-M3 (the perf milestone) and its part-B counterfactual both pinned the single-stream B=1
spec-decode bottleneck to the **eager T>1 verify forward** (1.5–2.3× t_main): the compiled fused
mamba/moe graph was T==1-only, so spec's ``k+1``-token verify fell to eager. Part A is the fix —
``NemotronResidentModel.compile_verify`` (default off, rule 4) routes the T>1 verify continuation
through the SAME compiled fused mixers as the T==1 decode. ``mx.compile`` only fuses, so the compiled
verify MUST be output-equivalent to the eager verify; this gate proves it on the real 306 GiB
int4-RTN backbone before the bench (``parity/nemotron_mtp_compiled_verify_bench.py``) trusts any
speedup. The parity-first discipline (CLAUDE.md): a flagged optimization is naive until proven
output-equivalent.

The gate, per ``k in {1, 2, 3}`` (verify width ``T = k + 1``):

* prefill the SAME prompt into two independent state triples — the eager prefill is deterministic, so
  the two are bit-identical pre-verify (asserted);
* run one ``T = k + 1`` verify forward on each — **eager** (``compile_verify=False``) vs **compiled**
  (``compile_verify=True``);
* assert bit-identical across the entire verify surface:
    - the verify **logits** ``[1, k+1, vocab]``,
    - the captured **last-layer hidden** (the MTP feature),
    - the post-verify **Mamba ``(ssm, conv)`` recurrent state** per layer (the branch-3 step loop is
      the actual compile target — the one numeric risk),
    - a **follow-on ``T == 1`` decode token's logits** off each post-verify state — an end-to-end
      check that also covers the KV cache + recurrence (if any state diverged, the next token would).

One model resident — **RUN SOLO** (~306 GiB wired, under the 490.4 GiB ceiling).

    uv run --with tokenizers python -m parity.nemotron_ultra_compiled_verify_parity
"""

from __future__ import annotations

import mlx.core as mx

from parity.nemotron_mtp_k_bench import ART, N_PROMPT
from parity.nemotron_ultra_ppl import LONG_PROSE
from quanta.nemotron.runtime import NemotronResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer

K_VALUES = (1, 2, 3)
# bf16 logits are O(10) and mx.compile fuses the SAME ops in the same order → expect 0.0, a stray ULP
# at most. State magnitudes are smaller. Tight thresholds so a real divergence (a compile bug at T>1,
# not a fusion ULP) fails loud (rule 6); the actual deltas are printed regardless.
LOGIT_TOL = 1e-2
STATE_TOL = 1e-3


def _prefill(model, prompt_ids, *, k):
    """A fresh ``(caches, ssm, conv)`` prefilled on ``prompt_ids`` (eager, deterministic);
    ``max_rollback >= k + 1`` so the verify's KV fits. Evals the recurrent state before returning."""
    caches, ssm, conv = model.make_caches(max_rollback=k + 1)
    model(mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv)
    mx.eval([a for a in ssm if a is not None] + [a for a in conv if a is not None])
    return caches, ssm, conv


def _maxabs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _state_delta(xs, ys) -> float:
    """Max abs diff over the populated (mamba-layer) entries of two recurrent-state lists."""
    return max((_maxabs(xs[i], ys[i]) for i in range(len(xs)) if xs[i] is not None), default=0.0)


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    last = model.cfg.num_hidden_layers - 1
    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    # cur for the verify root (argmax of the prompt's last-position logits — a realistic verified token)
    c0, s0, v0 = model.make_caches(max_rollback=1)
    lg, _, _ = model(mx.array(prompt_ids), caches=c0, ssm=s0, conv=v0)
    cur = int(mx.argmax(lg[0, -1]).item())

    print("\n=== Nemotron-Ultra MTP-M3 (A): compiled T>1 verify-graph parity ===")
    print(f"backbone: {ART}")
    print(f"prompt {len(prompt_ids)} tok | layers {model.cfg.num_hidden_layers} | "
          f"verify root cur={cur}\n")
    print(f"  {'k':>2s} {'T':>2s}  {'prefillΔ':>9s}  {'logitsΔ':>9s}  {'hiddenΔ':>9s}  {'ssmΔ':>9s}  "
          f"{'convΔ':>9s}  {'nextΔ':>9s}  {'next-top1':>9s}  verdict")

    worst = 0.0
    all_ok = True
    for k in K_VALUES:
        t = k + 1
        # two bit-identical pre-verify states (deterministic eager prefill)
        ce, se, ve = _prefill(model, prompt_ids, k=k)
        cc, sc, vc = _prefill(model, prompt_ids, k=k)
        pf_d = max(_state_delta(se, sc), _state_delta(ve, vc))   # sanity: prefills agree pre-verify

        # verify seq = cur + k filler tokens (valid ids; values are irrelevant to a parity check —
        # both sides consume the identical sequence, only the eager-vs-compiled execution differs)
        seq = mx.array([cur, *prompt_ids[1:1 + k]])

        model.compile_verify = False
        le, caps_e = model(seq, caches=ce, ssm=se, conv=ve, capture_layers=(last,))
        model.compile_verify = True
        lc, caps_c = model(seq, caches=cc, ssm=sc, conv=vc, capture_layers=(last,))
        mx.eval(le, lc, caps_e[last], caps_c[last],
                *[a for a in se + ve + sc + vc if a is not None])

        d_log = _maxabs(le, lc)
        d_hid = _maxabs(caps_e[last], caps_c[last])
        d_ssm = _state_delta(se, sc)
        d_conv = _state_delta(ve, vc)

        # follow-on T==1 decode off each post-verify state — feed the SAME token (eager's argmax) so a
        # mismatch isolates a diverged state, not a diverged token choice. T==1 ⇒ both use the compiled
        # decode path (compile_verify is irrelevant here), so equal states ⇒ equal next logits.
        bonus = int(mx.argmax(le[0, -1]).item())
        ne, _, _ = model(mx.array([bonus]), caches=ce, ssm=se, conv=ve)
        nc, _, _ = model(mx.array([bonus]), caches=cc, ssm=sc, conv=vc)
        mx.eval(ne, nc)
        d_next = _maxabs(ne, nc)
        top1_e, top1_c = int(mx.argmax(ne[0, -1]).item()), int(mx.argmax(nc[0, -1]).item())

        ok = (d_log < LOGIT_TOL and d_hid < LOGIT_TOL and d_ssm < STATE_TOL
              and d_conv < STATE_TOL and d_next < LOGIT_TOL and top1_e == top1_c
              and pf_d < STATE_TOL)
        all_ok = all_ok and ok
        worst = max(worst, d_log, d_hid, d_next)
        print(f"  {k:>2d} {t:>2d}  {pf_d:>9.2e}  {d_log:>9.2e}  {d_hid:>9.2e}  {d_ssm:>9.2e}  "
              f"{d_conv:>9.2e}  {d_next:>9.2e}  {'==' if top1_e == top1_c else '!!':>9s}  "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    if all_ok:
        print(f"VERDICT: compiled T>1 verify == eager T>1 verify on ALL of {{logits, hidden, ssm, "
              f"conv, follow-on}} for k in {K_VALUES} (worst Δ {worst:.2e} ≤ tol). compile_verify is "
              f"output-equivalent — the bench may trust its speedup; losslessness (M2) is untouched.")
    else:
        raise SystemExit(
            f"FAIL: compiled T>1 verify diverged from eager (worst Δ {worst:.2e} > tol) — mx.compile "
            f"is NOT fuse-only on the branch-3 verify graph at T>1. Do NOT enable compile_verify; "
            f"investigate the offending mixer (rule 6, rule 4).")


if __name__ == "__main__":
    run()
