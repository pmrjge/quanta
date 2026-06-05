"""Nemotron-Ultra U4 / MTP-M3 — native-MTP spec-decode WALL-CLOCK bench (the perf milestone).

The native-MTP stream so far: M0 gated the bf16 head's structure, M1 baked the int4-RTN **sidecar**,
M2 proved the resident spec loop **lossless** (``parity/nemotron_ultra_mtp_resident_spec.py`` — spec ==
greedy up to a verified bf16 ULP near-tie, the main model verifies every draft). M3 is the **perf**
milestone: load the REAL baked head (:func:`quanta.nemotron.runtime.build_resident_mtp`) on the 306 GiB
Ultra backbone and measure the wall-clock decode speedup of lossless spec-decode vs production greedy
across the two speed levers the runtime already exposes —

* ``k`` — chained draft depth (one MTP head drafts ``k`` tokens/round; the main model verifies all
  ``k + 1`` in one forward, amortizing its cost over a longer window);
* ``draft_topk`` — the MTP moe sub-block routes through fewer experts than the main verify path (a
  lighter drafter; ``mx.gather_qmm`` over ``draft_topk`` of ``cfg.num_experts_per_tok`` experts). Pure
  speed lever — losslessness is unaffected (the main model still verifies at full topk).

Because the main model verifies every draft, the spec stream is bit-identical to greedy regardless of
head quality (**M2 owns the correctness proof**); this bench only MEASURES speed, so any divergence
from greedy (the M2 bf16 ULP near-tie at the k=1 first divergence) is reported as ``match`` info, never
asserted.

**Economics probes** (printed so a sub-1x result is actionable, not a black box):

* ``t_main``   — one main-model **compiled** ``T=1`` decode step (the greedy cost / token);
* ``t_verify`` — one main-model **eager** ``T=(k+1)`` verify forward (spec's per-round main cost; the
  compiled fused decode graph is ``T==1``-only, so the multi-token verify runs eager — a structural
  spec disadvantage at ``B=1`` that the amortization must overcome);
* ``t_draft``  — one MTP head draft step at each ``draft_topk`` (the speculative overhead / token).

One model resident at a time — **RUN SOLO** (~313 GiB wired: 306 backbone + 6.56 sidecar, under the
490.4 GiB ceiling). Single-stream ``B=1`` latency bench; the batched/tree verify (``B>1`` serving
throughput) is a separate runtime.

    uv run --with tokenizers python -m parity.nemotron_mtp_k_bench
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_ultra_ppl import LONG_PROSE
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.runtime import NemotronResidentModel, build_resident_mtp
from quanta.nemotron.spec import spec_generate_k
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64"
MTP_ART = ART + "_mtp"
N_PROMPT = 64                 # real prose prefill (matches the M2 gate — realistic Mamba state)
N_GEN = 48                    # tokens decoded by greedy AND each spec config (eos_id=None -> fixed len)
K_VALUES = (1, 2, 3)
DRAFT_TOPK = (2, 4, 8, None)  # MTP moe sub-block routing top-k; None = full (cfg.num_experts_per_tok)
PROBE_REPS = 5


def _median_ms(fn) -> float:
    """Median wall-ms of ``fn`` over ``PROBE_REPS`` calls. ``fn`` must force its own ``mx.eval`` /
    ``.item()`` so the timed region includes the GPU work (MLX is lazy)."""
    ts = []
    for _ in range(PROBE_REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _greedy_decode(model, prompt_ids, *, max_new: int) -> tuple[list[int], float]:
    """Production greedy decode: prefill (eager, T>1), then ``max_new - 1`` **compiled** ``T=1`` steps.
    Returns ``(tokens, decode_seconds)`` — decode_seconds excludes prefill (the ``cur`` from prefill is
    emitted untimed, exactly as the tok/s denominator expects)."""
    caches, ssm, conv = model.make_caches(max_rollback=1)
    logits, ssm, conv = model(mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv)
    cur = int(mx.argmax(logits[0, -1]).item())          # forces prefill eval before the decode timer
    out = [cur]
    t0 = time.perf_counter()
    while len(out) < max_new:
        logits, ssm, conv = model(mx.array([cur]), caches=caches, ssm=ssm, conv=conv)  # compiled T=1
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    return out[:max_new], time.perf_counter() - t0


def _prefill_s(model, prompt_ids) -> float:
    """Cost of a single prompt prefill (what ``spec_generate_k`` runs internally before decoding) — the
    subtrahend that turns the spec call's full wall-time into a decode-only window comparable to greedy.
    Uses the same ``max_rollback`` the spec loop's :meth:`make_caches` allocates."""
    caches, ssm, conv = model.make_caches(max_rollback=8)
    t0 = time.perf_counter()
    logits, ssm, conv = model(mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv)
    _ = int(mx.argmax(logits[0, -1]).item())            # force eval
    return time.perf_counter() - t0


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    mtp_art = NemotronArtifact(MTP_ART)
    mtp = build_resident_mtp(mtp_art, model.cfg)        # real baked head (default full topk)
    mtp_art.release()
    mx.clear_cache()
    embed, head = model.embed_w, model.lm_head_w
    last = model.cfg.num_hidden_layers - 1
    npt = model.cfg.num_experts_per_tok
    load_min = (time.perf_counter() - t0) / 60

    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    def _lbl(tk) -> str:
        return f"full({npt})" if tk is None else str(tk)

    # --- warmup: one full greedy decode JITs the compiled T=1 mamba/moe mixers (shared by every
    #     compiled step below); the eager prefill / verify / draft paths use prebuilt Metal kernels
    #     (no per-shape JIT), so one warmup hot-loads everything the sweep reuses. ---
    _greedy_decode(model, prompt_ids, max_new=N_GEN)

    # --- economics probes (median over PROBE_REPS) ---
    # t_draft: needs the captured main hidden at q + the next-token embedding (the MTP feature).
    cap_caches, cap_ssm, cap_conv = model.make_caches(max_rollback=1)
    logits, caps = model(mx.array(prompt_ids), caches=cap_caches, ssm=cap_ssm, conv=cap_conv,
                         capture_layers=(last,))
    prev_hidden = caps[last][-1][None, None]
    cur0 = int(mx.argmax(logits[0, -1]).item())
    token_emb = embed[cur0][None, None].astype(prev_hidden.dtype)
    t_draft: dict = {}
    for tk in DRAFT_TOPK:
        mtp.set_draft_topk(tk)
        t_draft[tk] = _median_ms(lambda: mx.eval(mtp(prev_hidden, token_emb, head)[0]))
    mtp.set_draft_topk(None)

    # t_main: one compiled T=1 decode step from a prefilled state.
    mc, ms, mv = model.make_caches(max_rollback=1)
    model(mx.array(prompt_ids), caches=mc, ssm=ms, conv=mv)
    t_main = _median_ms(lambda: mx.eval(model(mx.array([cur0]), caches=mc, ssm=ms, conv=mv)[0]))

    # t_verify[k]: one eager T=(k+1) verify forward from a prefilled state (token values irrelevant to
    # timing — the op count is shape-driven; later reps grow the KV a few positions, negligible).
    t_verify: dict = {}
    for k in K_VALUES:
        vc, vs, vv = model.make_caches(max_rollback=k + 1)
        model(mx.array(prompt_ids), caches=vc, ssm=vs, conv=vv)
        seq = mx.array([cur0] + [0] * k)
        t_verify[k] = _median_ms(lambda c=vc, s=vs, v=vv, q=seq: mx.eval(model(q, caches=c, ssm=s, conv=v)[0]))

    # --- baseline greedy (compiled) ---
    greedy, g_decode_s = _greedy_decode(model, prompt_ids, max_new=N_GEN)
    g_tps = (N_GEN - 1) / g_decode_s
    pf_s = _prefill_s(model, prompt_ids)

    print("\n=== Nemotron-Ultra MTP-M3 native-MTP spec-decode wall-clock bench ===")
    print(f"backbone: {ART}")
    print(f"mtp head: {MTP_ART}")
    print(f"load {load_min:.1f} min | prompt {len(prompt_ids)} tok | gen {N_GEN} tok | "
          f"num_experts_per_tok={npt}")
    print("\neconomics (median):")
    print(f"  t_main (compiled T=1 decode) : {t_main:7.1f} ms/tok")
    for k in K_VALUES:
        print(f"  t_verify T={k + 1} (k={k}) eager   : {t_verify[k]:7.1f} ms  "
              f"({t_verify[k] / t_main:.2f}x t_main)")
    for tk in DRAFT_TOPK:
        print(f"  t_draft draft_topk={_lbl(tk):8s} : {t_draft[tk]:7.1f} ms")
    print(f"\ngreedy (compiled)            : {g_decode_s:.1f}s decode ({g_tps:.1f} tok/s)  [baseline]")
    print(f"\n  {'draft_topk':12s} {'k':>2s}  {'mean_accept':>12s}  {'tok/s':>6s}  "
          f"{'vs_greedy':>9s}  {'match':>7s}")

    best = (0.0, "", 0.0)   # (tok/s, label, mean_accept)
    for tk in DRAFT_TOPK:
        mtp.set_draft_topk(tk)
        for k in K_VALUES:
            t1 = time.perf_counter()
            toks, stats = spec_generate_k(model, mtp, embed, head, prompt_ids,
                                          k=k, max_new=N_GEN, eos_id=None)
            wall = time.perf_counter() - t1
            dec_s = max(wall - pf_s, 1e-9)
            s_tps = (len(toks) - 1) / dec_s
            ratio = s_tps / g_tps
            fd = next((i for i in range(min(len(toks), len(greedy))) if toks[i] != greedy[i]),
                      len(greedy))
            if s_tps > best[0]:
                best = (s_tps, f"draft_topk={_lbl(tk)} k={k}", stats["mean_accept"])
            print(f"  {_lbl(tk):12s} {k:>2d}  {stats['mean_accept']:>6.2f}/{k + 1:<5d}  "
                  f"{s_tps:>6.1f}  {ratio:>8.2f}x  {fd:>3d}/{N_GEN}")

    best_ratio = best[0] / g_tps
    print(f"\nbest: {best[1]} -> {best[0]:.1f} tok/s ({best_ratio:.2f}x greedy, "
          f"mean_accept {best[2]:.2f} tok/round)")
    if best_ratio >= 1.0:
        print(f"VERDICT: lossless spec-decode BEATS compiled greedy at B=1 ({best_ratio:.2f}x) — "
              f"the verify amortization over the {best[2]:.2f}-token draft window wins.")
    else:
        print(f"VERDICT: single-stream spec is <1x compiled greedy at B=1 (best {best_ratio:.2f}x). "
              f"The eager T>1 verify costs ~{t_verify[1] / t_main:.1f}-{t_verify[3] / t_main:.1f}x "
              f"t_main (the compiled fused decode graph is T==1-only) and a hybrid partial-reject "
              f"round runs a 2nd main forward (the un-sliceable Mamba state re-run); together they "
              f"outweigh the {best[2]:.2f}-token/round amortization. Losslessness holds (M2). >1x "
              f"needs a compiled T>1 verify graph and/or batched (B>1) tree-verify serving (already "
              f"built; this bench is the B=1 latency baseline).")


if __name__ == "__main__":
    run()
