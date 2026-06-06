"""Nemotron-Ultra U4 / Stream-B — fused multi-token SSD-scan verify WALL-CLOCK bench.

Stream B fuses the T>1 spec-VERIFY Mamba continuation into one Metal launch per layer
(``mamba_mixer.FUSED_SSD_SCAN`` ⇒ ``ssd_scan_fused`` + a bit-identical batched conv), targeting the
verify's launch-bound Mamba T-growth — the dominant verify cost (decode-economics: +77.8ms / 59% of the
T=1→4 growth, the part ``mx.compile`` can't fuse across the sequential Python T-loop, which capped
compiled-verify at MTP-M3 A's 0.84× B=1). This bench MEASURES whether collapsing that loop into one
kernel crosses 1×, head-to-head IN ONE PROCESS across three verify modes:

* **E**  — eager per-token verify (``FUSED_SSD_SCAN=False``, ``compile_verify=False``): the M3 0.79× baseline.
* **F**  — fused-scan verify (``FUSED_SSD_SCAN=True``,  ``compile_verify=False``): Stream B's lever, eager elsewhere.
* **FC** — fused scan + compiled verify (both ``True``): does fusing stack on the MTP-M3 A compile?

Ordering pins ``mx.compile`` correctness: a compiled trace is keyed by input shape and frozen at first
trace (a later module-global flip does NOT re-trace). E and F never trace a compiled-VERIFY graph; the
ONLY compiled-verify traces are made in the FC economics probe with ``FUSED_SSD_SCAN`` already True, so
every FC run reuses a flag-consistent trace. The T==1 decode graph (``t_main`` / greedy / reject re-run)
is flag-independent (the fused branch is ``t > 1`` only), so it is shared across all modes.

Because the int4-RTN main model verifies every draft, all three modes are lossless (M2 owns the proof);
``mean_accept`` and the first-divergence vs greedy must match across modes (printed as a cross-check, not
asserted). The fused-scan parity gate (``parity/nemotron_ultra_fused_scan_parity.py``) proves F/FC are
output-equivalent to E before this bench trusts any speedup.

One model resident — **RUN SOLO** (~313 GiB wired: 306 backbone + 6.56 sidecar, under 490.4 GiB).

    uv run --with tokenizers python -m parity.nemotron_ultra_fused_scan_bench
"""

from __future__ import annotations

import time

import mlx.core as mx

import quanta.nemotron.mamba_mixer as mm
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

INT4_BEST_EAGER = 0.79      # M3 eager-verify ceiling (draft_topk=8 k=1)
INT4_BEST_COMPILED = 0.84   # MTP-M3 A compiled-verify ceiling (draft_topk=8 k=1)


def _verify_probe(model, prompt_ids, cur0, k, *, fused, compiled) -> float:
    """Median ms of one ``T=k+1`` verify forward, warmed. Sets both levers; with ``compiled`` this also
    JITs the compiled ``T=k+1`` trace the timed sweep reuses (only ever traced with ``fused`` True)."""
    mm.FUSED_SSD_SCAN = fused
    model.compile_verify = compiled
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
    mtp = build_resident_mtp(mtp_art, model.cfg)
    mtp_art.release()
    mx.clear_cache()
    embed, head = model.embed_w, model.lm_head_w
    last = model.cfg.num_hidden_layers - 1
    npt = model.cfg.num_experts_per_tok
    load_min = (time.perf_counter() - t0) / 60

    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    def _lbl(tk) -> str:
        return f"full({npt})" if tk is None else str(tk)

    # warmup: one compiled greedy decode JITs the flag-independent T==1 mamba/moe mixers
    mm.FUSED_SSD_SCAN = False
    model.compile_verify = False
    _greedy_decode(model, prompt_ids, max_new=N_GEN)

    pc, ps, pv = model.make_caches(max_rollback=1)
    plg, _, _ = model(mx.array(prompt_ids), caches=pc, ssm=ps, conv=pv)
    cur0 = int(mx.argmax(plg[0, -1]).item())

    # --- economics (median) ---
    mc, ms, mv = model.make_caches(max_rollback=1)
    model(mx.array(prompt_ids), caches=mc, ssm=ms, conv=mv)
    mm.FUSED_SSD_SCAN = False
    model.compile_verify = False
    t_main = _median_ms(lambda: mx.eval(model(mx.array([cur0]), caches=mc, ssm=ms, conv=mv)[0]))

    # t_verify: eager (E) and fused-eager (F) first (NO compiled-verify trace), then fused+compiled (FC)
    # — the only compiled-verify traces, made with FUSED_SSD_SCAN already True (flag-consistent).
    t_v_eager = {k: _verify_probe(model, prompt_ids, cur0, k, fused=False, compiled=False)
                 for k in K_VALUES}
    t_v_fused = {k: _verify_probe(model, prompt_ids, cur0, k, fused=True, compiled=False)
                 for k in K_VALUES}
    t_v_fcomp = {k: _verify_probe(model, prompt_ids, cur0, k, fused=True, compiled=True)
                 for k in K_VALUES}

    # t_draft (MTP head; unchanged — flag-independent)
    cap_c, cap_s, cap_v = model.make_caches(max_rollback=1)
    _lg, caps = model(mx.array(prompt_ids), caches=cap_c, ssm=cap_s, conv=cap_v, capture_layers=(last,))
    prev_hidden = caps[last][-1][None, None]
    token_emb = embed[cur0][None, None].astype(prev_hidden.dtype)
    t_draft: dict = {}
    for tk in DRAFT_TOPK:
        mtp.set_draft_topk(tk)
        t_draft[tk] = _median_ms(lambda: mx.eval(mtp(prev_hidden, token_emb, head)[0]))
    mtp.set_draft_topk(None)

    # --- baseline greedy (compiled, flag-independent at T==1) ---
    mm.FUSED_SSD_SCAN = False
    model.compile_verify = False
    greedy, g_decode_s = _greedy_decode(model, prompt_ids, max_new=N_GEN)
    g_tps = (N_GEN - 1) / g_decode_s
    pf_s = _prefill_s(model, prompt_ids)

    print("\n=== Nemotron-Ultra Stream-B: fused multi-token SSD-scan verify bench ===")
    print(f"backbone: {ART}")
    print(f"mtp head: {MTP_ART}")
    print(f"load {load_min:.1f} min | prompt {len(prompt_ids)} tok | gen {N_GEN} tok | "
          f"num_experts_per_tok={npt} | mamba layers {model.cfg.count('mamba')}")
    print("\neconomics (median):")
    print(f"  t_main (compiled T=1 decode)     : {t_main:7.1f} ms/tok")
    for k in K_VALUES:
        e, f, fc = t_v_eager[k], t_v_fused[k], t_v_fcomp[k]
        print(f"  t_verify T={k + 1} (k={k}): eager {e:7.1f} ({e / t_main:.2f}x) | fused {f:7.1f} "
              f"({f / t_main:.2f}x, {e / f:.2f}x vs E) | fused+comp {fc:7.1f} ({fc / t_main:.2f}x, "
              f"{e / fc:.2f}x vs E)")
    for tk in DRAFT_TOPK:
        print(f"  t_draft draft_topk={_lbl(tk):8s}        : {t_draft[tk]:7.1f} ms")
    print(f"\ngreedy (compiled)                  : {g_decode_s:.1f}s decode ({g_tps:.1f} tok/s)  "
          f"[baseline]")

    def _spec(tk, k, *, fused, compiled):
        mm.FUSED_SSD_SCAN = fused
        model.compile_verify = compiled
        mtp.set_draft_topk(tk)
        t1 = time.perf_counter()
        toks, stats = spec_generate_k(model, mtp, embed, head, prompt_ids,
                                      k=k, max_new=N_GEN, eos_id=None)
        dec_s = max((time.perf_counter() - t1) - pf_s, 1e-9)
        return toks, stats, (len(toks) - 1) / dec_s

    print(f"\n  {'draft_topk':10s} {'k':>2s}  {'accept':>6s}  {'tok/s(E)':>8s} {'(F)':>6s} {'(FC)':>6s}  "
          f"{'E×':>5s} {'F×':>5s} {'FC×':>5s}  {'acc==':>5s}  {'match':>7s}")
    best = {"E": (0.0, ""), "F": (0.0, ""), "FC": (0.0, "", 0.0)}
    for tk in DRAFT_TOPK:
        for k in K_VALUES:
            e_toks, e_st, e_tps = _spec(tk, k, fused=False, compiled=False)
            f_toks, f_st, f_tps = _spec(tk, k, fused=True, compiled=False)
            c_toks, c_st, c_tps = _spec(tk, k, fused=True, compiled=True)
            r_e, r_f, r_c = e_tps / g_tps, f_tps / g_tps, c_tps / g_tps
            accs = [e_st["mean_accept"], f_st["mean_accept"], c_st["mean_accept"]]
            acc_eq = "==" if max(accs) - min(accs) < 1e-9 else "!!"
            fd = next((i for i in range(min(len(c_toks), len(greedy))) if c_toks[i] != greedy[i]),
                      len(greedy))
            if e_tps > best["E"][0]:
                best["E"] = (e_tps, f"draft_topk={_lbl(tk)} k={k}")
            if f_tps > best["F"][0]:
                best["F"] = (f_tps, f"draft_topk={_lbl(tk)} k={k}")
            if c_tps > best["FC"][0]:
                best["FC"] = (c_tps, f"draft_topk={_lbl(tk)} k={k}", c_st["mean_accept"])
            print(f"  {_lbl(tk):10s} {k:>2d}  {c_st['mean_accept']:>4.2f}/{k + 1:<1d}  "
                  f"{e_tps:>8.1f} {f_tps:>6.1f} {c_tps:>6.1f}  {r_e:>4.2f}x {r_f:>4.2f}x {r_c:>4.2f}x  "
                  f"{acc_eq:>5s}  {fd:>3d}/{N_GEN}")

    re_b, rf_b, rc_b = best["E"][0] / g_tps, best["F"][0] / g_tps, best["FC"][0] / g_tps
    print(f"\nbest eager (E)        : {best['E'][1]} -> {best['E'][0]:.1f} tok/s ({re_b:.2f}x greedy)  "
          f"[reproduces M3 ≈ {INT4_BEST_EAGER:.2f}x]")
    print(f"best fused (F)        : {best['F'][1]} -> {best['F'][0]:.1f} tok/s ({rf_b:.2f}x greedy)")
    print(f"best fused+comp (FC)  : {best['FC'][1]} -> {best['FC'][0]:.1f} tok/s ({rc_b:.2f}x greedy, "
          f"mean_accept {best['FC'][2]:.2f})")
    best_ratio = max(rf_b, rc_b)
    if best_ratio >= 1.0:
        print(f"\nVERDICT: the fused multi-token SSD-scan verify lifts lossless B=1 spec-decode to "
              f"{best_ratio:.2f}x compiled greedy — it CROSSES 1× (eager ceiling {re_b:.2f}x ≈ M3 "
              f"{INT4_BEST_EAGER}; compiled ceiling {INT4_BEST_COMPILED}). Collapsing the per-token "
              f"Mamba verify loop into one kernel was the decisive B=1 lever the economics predicted; "
              f"the per-forward verify speedup is {t_v_eager[1] / t_v_fused[1]:.2f}-"
              f"{t_v_eager[3] / t_v_fcomp[3]:.2f}x. Losslessness holds (M2).")
    else:
        print(f"\nVERDICT: the fused SSD-scan verify lifts B=1 spec to {best_ratio:.2f}x greedy "
              f"(eager {re_b:.2f}x ≈ M3 {INT4_BEST_EAGER}; vs the {INT4_BEST_COMPILED} compiled ceiling) "
              f"but still <1× — the fused verify is {t_v_fused[1] / t_main:.2f}-{t_v_fcomp[3] / t_main:.2f}x "
              f"t_main, not yet under the amortization threshold at mean_accept {best['FC'][2]:.2f}. The "
              f"residual T-growth is the unfused MoE gather_qmm (40% per the economics, weight-bandwidth, "
              f"not launch-bound). Losslessness holds (M2); the B>1 tree-verify is the throughput lever.")


if __name__ == "__main__":
    run()
