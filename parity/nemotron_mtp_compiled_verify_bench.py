"""Nemotron-Ultra U4 / MTP-M3 (A) — compiled T>1 verify-graph wall-clock bench.

The MTP-M3 part-A/part-B finding: single-stream B=1 lossless native-MTP spec-decode tops out at
**0.79× compiled greedy**, and the drafter is near-inert (part B: even a bf16 drafter is 0.79×). The
bottleneck is the **eager T>1 verify** — greedy runs the compiled fused T==1 mamba/moe graph
(88.9 ms/tok) but spec's ``T = k+1`` verify fell to eager (1.5–2.3× t_main). Part A adds
``NemotronResidentModel.compile_verify`` (default off, rule 4; gated output-equivalent in
``parity/nemotron_ultra_compiled_verify_parity.py``) which compiles that verify forward too. This
bench MEASURES whether the compiled verify crosses 1×.

Head-to-head IN ONE PROCESS (same machine / prompt / gen / baseline greedy, so the comparison is not
cross-run): for every ``draft_topk × k`` config the bench runs ``spec_generate_k`` **twice** — eager
(``compile_verify=False``, reproducing the M3 baseline) then compiled (``compile_verify=True``) — and
prints both tok/s, both ratios vs greedy, and the compiled/eager **speedup**. Because the compiled
verify is bit-identical to eager (the parity gate), ``mean_accept`` and the first-divergence vs greedy
must match between the two runs (printed as a cross-check; a mismatch would be a bug). Losslessness is
owned by M2 — the int4-RTN main model verifies every draft, so divergence from greedy is INFO only.

Economics printed: ``t_main`` (compiled T==1 decode), ``t_verify`` **eager vs compiled** for
``T = k+1``, and ``t_draft`` (the MTP head, unchanged). The compiled-verify traces for ``T ∈ {2,3,4}``
are JIT-warmed by the ``t_verify`` probes before the timed sweep (the partial-reject re-run shapes
``T ∈ {2,3}`` are subsumed).

One model resident — **RUN SOLO** (~313 GiB wired: 306 backbone + 6.56 sidecar, under 490.4 GiB).

    uv run --with tokenizers python -m parity.nemotron_mtp_compiled_verify_bench
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_mtp_k_bench import (
    ART,
    DRAFT_TOPK,
    K_VALUES,
    MTP_ART,
    N_GEN,
    N_PROMPT,
    _greedy_decode,
    _median_ms,
    _prefill_s,
)
from parity.nemotron_ultra_ppl import LONG_PROSE
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.runtime import NemotronResidentModel, build_resident_mtp
from quanta.nemotron.spec import spec_generate_k
from quanta.nemotron.tokenizer import NemotronTokenizer

INT4_BEST_RATIO = 0.79   # M3 / part-B eager-verify ceiling (draft_topk=8 k=1)


def _verify_probe(model, prompt_ids, cur0, k, *, compiled_verify) -> float:
    """Median ms of one ``T = k+1`` verify forward, warmed. With ``compiled_verify`` this also JITs the
    compiled ``T = k+1`` fused trace that the timed sweep reuses (mx.compile auto-keys per shape)."""
    model.compile_verify = compiled_verify
    vc, vs, vv = model.make_caches(max_rollback=k + 1)
    model(mx.array(prompt_ids), caches=vc, ssm=vs, conv=vv)                # prefill (eager either way)
    seq = mx.array([cur0] + [0] * k)
    mx.eval(model(seq, caches=vc, ssm=vs, conv=vv)[0])                     # warmup (JIT if compiled)
    return _median_ms(lambda: mx.eval(model(seq, caches=vc, ssm=vs, conv=vv)[0]))


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    mtp_art = NemotronArtifact(MTP_ART)
    mtp = build_resident_mtp(mtp_art, model.cfg)        # real baked int4-RTN head (default full topk)
    mtp_art.release()
    mx.clear_cache()
    embed, head = model.embed_w, model.lm_head_w
    last = model.cfg.num_hidden_layers - 1
    npt = model.cfg.num_experts_per_tok
    load_min = (time.perf_counter() - t0) / 60

    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    def _lbl(tk) -> str:
        return f"full({npt})" if tk is None else str(tk)

    # warmup: one compiled greedy decode JITs the T==1 mamba/moe mixers (t_main + the T==1 reject re-run)
    model.compile_verify = False
    _greedy_decode(model, prompt_ids, max_new=N_GEN)

    # cur0 for the probes (argmax of the prefill's last logits)
    pc, ps, pv = model.make_caches(max_rollback=1)
    plg, _, _ = model(mx.array(prompt_ids), caches=pc, ssm=ps, conv=pv)
    cur0 = int(mx.argmax(plg[0, -1]).item())

    # --- economics (median) ---
    mc, ms, mv = model.make_caches(max_rollback=1)
    model(mx.array(prompt_ids), caches=mc, ssm=ms, conv=mv)
    model.compile_verify = False
    t_main = _median_ms(lambda: mx.eval(model(mx.array([cur0]), caches=mc, ssm=ms, conv=mv)[0]))

    # t_verify eager vs compiled (the compiled probes also JIT-warm T ∈ {2,3,4} for the sweep)
    t_verify_eager = {k: _verify_probe(model, prompt_ids, cur0, k, compiled_verify=False)
                      for k in K_VALUES}
    t_verify_comp = {k: _verify_probe(model, prompt_ids, cur0, k, compiled_verify=True)
                     for k in K_VALUES}

    # t_draft (MTP head; unchanged from M3 — for the economics table)
    cap_c, cap_s, cap_v = model.make_caches(max_rollback=1)
    _lg, caps = model(mx.array(prompt_ids), caches=cap_c, ssm=cap_s, conv=cap_v,
                      capture_layers=(last,))
    prev_hidden = caps[last][-1][None, None]
    token_emb = embed[cur0][None, None].astype(prev_hidden.dtype)
    t_draft: dict = {}
    for tk in DRAFT_TOPK:
        mtp.set_draft_topk(tk)
        t_draft[tk] = _median_ms(lambda: mx.eval(mtp(prev_hidden, token_emb, head)[0]))
    mtp.set_draft_topk(None)

    # --- baseline greedy (compiled, compile_verify irrelevant at T==1) ---
    model.compile_verify = False
    greedy, g_decode_s = _greedy_decode(model, prompt_ids, max_new=N_GEN)
    g_tps = (N_GEN - 1) / g_decode_s
    pf_s = _prefill_s(model, prompt_ids)

    print("\n=== Nemotron-Ultra MTP-M3 (A): compiled T>1 verify-graph bench ===")
    print(f"backbone: {ART}")
    print(f"mtp head: {MTP_ART}")
    print(f"load {load_min:.1f} min | prompt {len(prompt_ids)} tok | gen {N_GEN} tok | "
          f"num_experts_per_tok={npt}")
    print("\neconomics (median):")
    print(f"  t_main (compiled T=1 decode)     : {t_main:7.1f} ms/tok")
    for k in K_VALUES:
        e, c = t_verify_eager[k], t_verify_comp[k]
        print(f"  t_verify T={k + 1} (k={k}): eager {e:7.1f} ({e / t_main:.2f}x) | "
              f"compiled {c:7.1f} ({c / t_main:.2f}x t_main) | verify speedup {e / c:.2f}x")
    for tk in DRAFT_TOPK:
        print(f"  t_draft draft_topk={_lbl(tk):8s}        : {t_draft[tk]:7.1f} ms")
    print(f"\ngreedy (compiled)                  : {g_decode_s:.1f}s decode ({g_tps:.1f} tok/s)  "
          f"[baseline]")

    def _spec(tk, k, flag):
        """Run spec with compile_verify=flag; return (tokens, stats, tok/s)."""
        model.compile_verify = flag
        mtp.set_draft_topk(tk)
        t1 = time.perf_counter()
        toks, stats = spec_generate_k(model, mtp, embed, head, prompt_ids,
                                      k=k, max_new=N_GEN, eos_id=None)
        dec_s = max((time.perf_counter() - t1) - pf_s, 1e-9)
        return toks, stats, (len(toks) - 1) / dec_s

    print(f"\n  {'draft_topk':10s} {'k':>2s}  {'accept':>6s}  {'tok/s(E)':>8s} {'(C)':>6s}  "
          f"{'E×':>5s} {'C×':>5s}  {'spd':>5s}  {'acc==':>5s}  {'match':>7s}")
    best_e = (0.0, "")
    best_c = (0.0, "", 0.0)
    for tk in DRAFT_TOPK:
        for k in K_VALUES:
            te_toks, te_st, e_tps = _spec(tk, k, False)
            tc_toks, tc_st, c_tps = _spec(tk, k, True)
            r_e, r_c = e_tps / g_tps, c_tps / g_tps
            spd = c_tps / max(e_tps, 1e-9)
            acc_eq = "==" if abs(te_st["mean_accept"] - tc_st["mean_accept"]) < 1e-9 else "!!"
            fd = next((i for i in range(min(len(tc_toks), len(greedy))) if tc_toks[i] != greedy[i]),
                      len(greedy))
            if e_tps > best_e[0]:
                best_e = (e_tps, f"draft_topk={_lbl(tk)} k={k}")
            if c_tps > best_c[0]:
                best_c = (c_tps, f"draft_topk={_lbl(tk)} k={k}", tc_st["mean_accept"])
            print(f"  {_lbl(tk):10s} {k:>2d}  {tc_st['mean_accept']:>4.2f}/{k + 1:<1d}  "
                  f"{e_tps:>8.1f} {c_tps:>6.1f}  {r_e:>4.2f}x {r_c:>4.2f}x  {spd:>4.2f}x  "
                  f"{acc_eq:>5s}  {fd:>3d}/{N_GEN}")

    re_best, rc_best = best_e[0] / g_tps, best_c[0] / g_tps
    print(f"\nbest eager   : {best_e[1]} -> {best_e[0]:.1f} tok/s ({re_best:.2f}x greedy)  "
          f"[reproduces M3 ≈ {INT4_BEST_RATIO:.2f}x]")
    print(f"best compiled: {best_c[1]} -> {best_c[0]:.1f} tok/s ({rc_best:.2f}x greedy, "
          f"mean_accept {best_c[2]:.2f})")
    if rc_best >= 1.0:
        print(f"VERDICT: the compiled T>1 verify graph lifts lossless B=1 spec-decode to "
              f"{rc_best:.2f}x compiled greedy — it BEATS greedy (eager verify ceilinged at "
              f"{re_best:.2f}x). Part A is the decisive B=1 lever the M3/part-B analysis predicted; "
              f"the verify speedup ({t_verify_eager[1] / t_verify_comp[1]:.2f}-"
              f"{t_verify_eager[3] / t_verify_comp[3]:.2f}x) pulled t_verify under the amortization "
              f"threshold. Losslessness holds (M2).")
    else:
        print(f"VERDICT: the compiled T>1 verify lifts B=1 spec to {rc_best:.2f}x greedy (from the "
              f"{re_best:.2f}x eager ceiling) but still <1x — the compiled verify is "
              f"{t_verify_comp[1] / t_main:.2f}-{t_verify_comp[3] / t_main:.2f}x t_main (vs eager's "
              f"{t_verify_eager[1] / t_main:.2f}-{t_verify_eager[3] / t_main:.2f}x), not yet under the "
              f"amortization threshold at this accept rate. Losslessness holds (M2); the next B=1 "
              f"levers are batched (B>1) tree-verify / a fused multi-token kernel.")


if __name__ == "__main__":
    run()
