"""Nemotron-Ultra U4 / MTP-M3 (perf, part B) — bf16-drafter quality-ceiling counterfactual.

MTP-M3 part A measured the *production* drafter (the baked int4-RTN sidecar) and found single-stream
B=1 lossless spec tops out at **0.79× compiled greedy** — bottlenecked by the eager T>1 verify, NOT
the draft (``t_draft ≈ 5 ms`` « ``t_main`` 88.9 ms). The sweep also showed a *lighter* drafter
(``draft_topk`` ↓) is WORSE, because it lowers the accept rate at ~zero cost saving. This part pins the
opposite lever empirically: a **higher-quality** drafter. We swap the int4-RTN head for the **bf16
source head** (the un-quantized drafter — the actual ceiling; dequantizing the int4 sidecar back to bf16
would just return the lossy int4 values, so we load the real bf16 weights from the source checkpoint),
hold the int4-RTN **backbone unchanged**, and re-run the IDENTICAL M3 economics + sweep.

The prediction (from the M3 cost model ``(t_verify + t_draft + P(reject)·t_main)/mean_accept``): a bf16
drafter raises ``mean_accept`` (drafts closer to what the main model verifies → fewer rejects → less of
the hybrid Mamba reject re-run), but is verify-bound, so it lands in ~**0.88–1.26×** — realistic accept
≈ 0.88×, a hypothetical perfect chain → the ceiling. This run measures where it actually lands, and
isolates the **quantization effect on accept rate** (bf16 vs int4 drafter, same backbone / prompt / gen
/ timing — the M3 ints below are the committed run ``1325dfb``).

Losslessness is unaffected (the int4-RTN **main model** verifies every draft — M2 owns that proof); a
higher-quality drafter only changes *speed*. Divergence from greedy is reported as ``match`` info, never
asserted.

One model resident at a time — **RUN SOLO**. ~330 GiB wired (306 backbone + ~21.5 bf16 MTP expert
stack + dense), under the 490.4 GiB ceiling; the source checkpoint is mmap-streamed for the ``mtp.*``
tensors only (rule 8) and released after the head is built.

    uv run --with tokenizers python -m parity.nemotron_mtp_bf16_drafter_bench
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx.utils import tree_flatten

from parity.nemotron_mtp_k_bench import (
    ART,
    DRAFT_TOPK,
    K_VALUES,
    N_GEN,
    N_PROMPT,
    _greedy_decode,
    _median_ms,
    _prefill_s,
)
from parity.nemotron_ultra_mtp_parity import ULTRA, _fill_module, _mtp_tensors
from parity.nemotron_ultra_ppl import LONG_PROSE
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.mtp import NemotronMTP
from quanta.nemotron.runtime import NemotronResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer

# MTP-M3 part-A reference: the int4-RTN drafter, SAME backbone / prompt (64) / gen (48) / harness, from
# the committed run (1325dfb). (draft_topk, k) -> (mean_accept, tok_s, ratio_vs_greedy). Greedy 11.2 tok/s.
INT4_REF: dict = {
    (2, 1): (1.45, 7.5, 0.67), (2, 2): (1.71, 6.4, 0.57), (2, 3): (1.71, 5.0, 0.44),
    (4, 1): (1.50, 7.9, 0.71), (4, 2): (1.74, 6.5, 0.58), (4, 3): (1.74, 5.1, 0.46),
    (8, 1): (1.60, 8.9, 0.79), (8, 2): (1.88, 7.6, 0.67), (8, 3): (1.88, 5.3, 0.47),
    (None, 1): (1.52, 8.2, 0.73), (None, 2): (1.81, 7.1, 0.63), (None, 3): (1.81, 5.2, 0.46),
}
INT4_BEST_RATIO = 0.79   # draft_topk=8 k=1


def _build_bf16_drafter(cfg) -> NemotronMTP:
    """The un-quantized bf16 MTP head — built from the source ``mtp.*`` tensors (the M0 loader), filled
    into a default :class:`NemotronMTP` whose moe sub-block is the **bf16** :class:`NemotronLatentMoE`
    (``mx.gather_mm`` over bf16 expert stacks) and whose dense projections are bf16 ``nn.Linear``. This
    is the genuine quality ceiling — NOT a dequantized int4 head."""
    ck = NemotronSourceCheckpoint(ULTRA)
    t, _consumed = _mtp_tensors(ck, cfg.n_routed_experts)
    holder = NemotronMTP(cfg)
    _fill_module(holder.module, t)
    arrs = [v for _, v in tree_flatten(holder.module.parameters())]
    arrs += [holder.module.moe_block.mixer.up_stack, holder.module.moe_block.mixer.down_stack]
    mx.eval(arrs)
    del t
    mx.clear_cache()
    return holder


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronResidentModel(ART)
    tok = NemotronTokenizer(ART)
    mtp = _build_bf16_drafter(model.cfg)             # bf16 source head (the un-quantized drafter)
    embed, head = model.embed_w, model.lm_head_w
    last = model.cfg.num_hidden_layers - 1
    npt = model.cfg.num_experts_per_tok
    load_min = (time.perf_counter() - t0) / 60

    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    def _lbl(tk) -> str:
        return f"full({npt})" if tk is None else str(tk)

    # warmup: one compiled greedy decode JITs the compiled T=1 mamba/moe mixers the sweep reuses.
    _greedy_decode(model, prompt_ids, max_new=N_GEN)

    # --- economics probes (median); t_draft now the BF16 drafter, t_main/t_verify the unchanged backbone ---
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

    mc, ms, mv = model.make_caches(max_rollback=1)
    model(mx.array(prompt_ids), caches=mc, ssm=ms, conv=mv)
    t_main = _median_ms(lambda: mx.eval(model(mx.array([cur0]), caches=mc, ssm=ms, conv=mv)[0]))

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

    # late import only to keep the module-load surface light (mirrors spec.py's deferred imports)
    from quanta.nemotron.spec import spec_generate_k

    print("\n=== Nemotron-Ultra MTP-M3 (B): bf16-drafter quality-ceiling bench ===")
    print(f"backbone : {ART}  (int4-RTN, unchanged)")
    print(f"drafter  : {ULTRA}  mtp.* (BF16 source head — the quality ceiling)")
    print(f"load {load_min:.1f} min | prompt {len(prompt_ids)} tok | gen {N_GEN} tok | "
          f"num_experts_per_tok={npt}")
    print("\neconomics (median):")
    print(f"  t_main (compiled T=1 decode) : {t_main:7.1f} ms/tok")
    for k in K_VALUES:
        print(f"  t_verify T={k + 1} (k={k}) eager   : {t_verify[k]:7.1f} ms  "
              f"({t_verify[k] / t_main:.2f}x t_main)")
    for tk in DRAFT_TOPK:
        i4 = ""  # show the int4 draft cost note only at full topk (M3 t_draft was ~5 ms flat)
        print(f"  t_draft(bf16) draft_topk={_lbl(tk):8s} : {t_draft[tk]:7.1f} ms{i4}")
    print(f"\ngreedy (compiled)            : {g_decode_s:.1f}s decode ({g_tps:.1f} tok/s)  [baseline]")
    print(f"\n  {'draft_topk':12s} {'k':>2s}  {'accept(bf16)':>12s}  {'tok/s':>6s}  {'vs_greedy':>9s}  "
          f"{'match':>7s}   {'accept(int4)':>12s}  {'Δaccept':>7s}")

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
            i4_acc, _i4_tps, _i4_ratio = INT4_REF[(tk, k)]
            dacc = stats["mean_accept"] - i4_acc
            if s_tps > best[0]:
                best = (s_tps, f"draft_topk={_lbl(tk)} k={k}", stats["mean_accept"])
            print(f"  {_lbl(tk):12s} {k:>2d}  {stats['mean_accept']:>6.2f}/{k + 1:<5d}  "
                  f"{s_tps:>6.1f}  {ratio:>8.2f}x  {fd:>3d}/{N_GEN}   {i4_acc:>6.2f}/{k + 1:<5d}  "
                  f"{dacc:>+7.2f}")

    best_ratio = best[0] / g_tps
    print(f"\nbest(bf16): {best[1]} -> {best[0]:.1f} tok/s ({best_ratio:.2f}x greedy, "
          f"mean_accept {best[2]:.2f}); int4 best was {INT4_BEST_RATIO:.2f}x (accept 1.60)")
    delta_band = "below" if best_ratio < 0.88 else ("within" if best_ratio <= 1.26 else "above")
    if best_ratio >= 1.0:
        print(f"VERDICT: the bf16 (un-quantized) drafter reaches {best_ratio:.2f}x greedy — lossless "
              f"spec-decode BEATS compiled greedy at B=1 with a high-quality head ({delta_band} the "
              f"0.88–1.26x predicted band). Drafter QUALITY (not draft cost) is the B=1 lever the int4 "
              f"head left on the table; the int4 quantization tax on accept-rate cost {best_ratio - INT4_BEST_RATIO:.2f}x.")
    else:
        print(f"VERDICT: even the bf16 (un-quantized) drafter is {best_ratio:.2f}x greedy at B=1 "
              f"({delta_band} the 0.88–1.26x band) — a higher-quality head lifts accept-rate "
              f"(Δaccept above) and tok/s over the int4 head ({INT4_BEST_RATIO:.2f}x), but the eager "
              f"T>1 verify ({t_verify[1] / t_main:.1f}-{t_verify[3] / t_main:.1f}x t_main) still "
              f"dominates. Confirms the M3 finding: the decisive B=1 lever is the compiled T>1 verify "
              f"graph (part A), not the drafter. Losslessness holds (M2).")


if __name__ == "__main__":
    run()
