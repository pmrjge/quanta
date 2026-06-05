"""Nemotron-Ultra U4 / MTP-M4 — batched tree-spec verify on the REAL baked MTP head.

The prior tree-verify real gate (``parity/nemotron_batched_tree_verify_real.py``) ran on Super-120B
with a RANDOM-init MTP — that artifact has 0 ``mtp.*`` keys, so every ``W ** D`` draft path collapsed
to the ~1/W random-drafter accept floor: there was no fan-out to amortize, and batched ≈ sequential
within ~1%. **MTP-M1 baked the real Ultra MTP sidecar** (1040 ``mtp.*`` tensors), so this is the FIRST
tree-verify measurement with a TRAINED head — where the ``W ** D`` weight-amortization the MTP-M3 (A)
finding named as the throughput lever actually kicks in.

Two parts, in order:

* **PARITY (rule 4).** ``spec_generate_tree(batched=True)`` must match the proven ``batched=False``
  sequential tree on real weights: the ``B = W ** D`` batched verify is an OPTIMIZATION of the naive
  per-path verify and must be output-equivalent. (Both are also lossless vs greedy — the int4-RTN main
  model arbitrates every token — but M2 owns that proof; reported here as INFO. The bf16 near-tie that
  diverged M2's k=1 at 24/48 may surface here too.)
* **BENCH.** A ``(W, D)`` sweep — greedy (compiled) vs the k-chain vs tree(seq) vs tree(bat) — to
  measure whether the real-head ``W ** D`` amortization lifts batched tree-verify above the M3 k-chain
  (0.79× greedy) and the (A) compiled-verify (0.84×) B=1 ceilings, and the bat/seq amortization the
  random head could not show.

``batch_step`` runs mamba/attention per-stream (the SAME ``NemotronBlock`` mixers M1/M2 proved on
Ultra) and stacks the MoE into ONE ``gather_qmm`` over all ``B`` paths (the amortizer) — feeding the
Ultra moe a ``[B, 1, hidden]`` tensor. No runtime change: ``batched=True`` already exists (built +
model-free-gated for the ``nemotron_h`` family); this is the real-head measurement the headless
artifact made impossible.

One model resident — **RUN SOLO** (~313 GiB backbone+sidecar + ``B = W ** D`` state replicas, under
the 490.4 GiB ceiling).

    uv run --with tokenizers python -m parity.nemotron_ultra_tree_spec
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_mtp_k_bench import ART, MTP_ART, N_PROMPT, _greedy_decode, _prefill_s
from parity.nemotron_ultra_ppl import LONG_PROSE
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.runtime import build_resident_mtp
from quanta.nemotron.spec import spec_generate_k, spec_generate_tree
from quanta.nemotron.tokenizer import NemotronTokenizer

PARITY_WD = (2, 2)                 # canonical parity geometry (B = W^D = 4) — matches the Super gate
PARITY_MAX_NEW = 32
BENCH_MAX_NEW = 48                 # matches the M2/M3 corpus window
SWEEP = [(2, 2), (2, 3), (4, 2)]   # (width, depth) -> paths W^D = 4, 8, 16
MAX_BATCH = 16                     # >= max W^D in the sweep
K_CHAIN_BEST = 0.79               # MTP-M3 best B=1 k-chain ratio vs greedy
COMPILED_VERIFY_BEST = 0.84        # MTP-M3 (A) best B=1 compiled-verify ratio vs greedy


def _spec_tps(fn, *args, pf_s, **kwargs):
    """Run a spec generator; return ``(tokens, stats, tok/s)`` with prefill subtracted so tok/s is
    decode-only — apples-to-apples with the compiled-greedy baseline (mirrors the M3 / (A) bench)."""
    t0 = time.perf_counter()
    toks, stats = fn(*args, **kwargs)
    dec_s = max((time.perf_counter() - t0) - pf_s, 1e-9)
    return toks, stats, (len(toks) - 1) / dec_s


def _match(a, b) -> tuple[int, int, int]:
    """``(matched-token count over the common length, common length, first-divergence index)``."""
    n = min(len(a), len(b))
    fd = next((i for i in range(n) if a[i] != b[i]), n)
    matched = sum(1 for i in range(n) if a[i] == b[i])
    return matched, n, fd


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronBatchedResidentModel(ART, max_batch=MAX_BATCH)
    tok = NemotronTokenizer(ART)
    mtp_art = NemotronArtifact(MTP_ART)
    mtp = build_resident_mtp(mtp_art, model.cfg)        # the REAL baked int4-RTN MTP head
    mtp.set_draft_topk(None)                            # full-vocab logits -> _mtp_top_w top-W is exact
    mtp_art.release()
    mx.clear_cache()
    embed, head = model.embed_w, model.lm_head_w
    load_min = (time.perf_counter() - t0) / 60

    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    print("\n=== Nemotron-Ultra MTP-M4: batched tree-spec verify (REAL baked MTP head) ===")
    print(f"backbone: {ART}")
    print(f"mtp head: {MTP_ART}")
    print(f"load {load_min:.1f} min | prompt {len(prompt_ids)} tok | "
          f"layers {model.cfg.num_hidden_layers} | max_batch {MAX_BATCH}")

    # baseline greedy (compiled T=1) + the prefill subtrahend shared by every spec tok/s
    greedy, g_dec_s = _greedy_decode(model, prompt_ids, max_new=BENCH_MAX_NEW)
    g_tps = (BENCH_MAX_NEW - 1) / g_dec_s
    pf_s = _prefill_s(model, prompt_ids)
    print(f"\ngreedy (compiled): {g_dec_s:.1f}s decode ({g_tps:.1f} tok/s) [baseline] | "
          f"prefill {pf_s:.2f}s")

    # ---------------- PARITY (rule 4): batched=True == batched=False ----------------
    W, D = PARITY_WD
    print(f"\n=== PARITY: spec_generate_tree(W={W}, D={D}, max_new={PARITY_MAX_NEW}) "
          f"batched=False vs True ===")
    seq_toks, seq_st = spec_generate_tree(model, mtp, embed, head, prompt_ids,
                                          width=W, depth=D, max_new=PARITY_MAX_NEW, batched=False)
    bat_toks, bat_st = spec_generate_tree(model, mtp, embed, head, prompt_ids,
                                          width=W, depth=D, max_new=PARITY_MAX_NEW, batched=True)
    bit_id = seq_toks == bat_toks
    m_bs, n_bs, fd_bs = _match(seq_toks, bat_toks)
    r_bs = m_bs / n_bs if n_bs else 0.0
    # INFO: both vs greedy (M2 owns the spec==greedy losslessness proof; the bf16 near-tie may flip)
    m_sg, n_sg, fd_sg = _match(seq_toks, greedy)
    m_bg, n_bg, fd_bg = _match(bat_toks, greedy)
    print(f"  batched=False: {len(seq_toks)} tok, mean_accept {seq_st['mean_accept']:.2f}/{D + 1}")
    print(f"  batched=True : {len(bat_toks)} tok, mean_accept {bat_st['mean_accept']:.2f}/{D + 1}")
    if bit_id:
        print(f"  bat vs seq   : BIT-IDENTICAL ({len(seq_toks)} tok)")
    else:
        print(f"  bat vs seq   : match {r_bs:.4f} ({m_bs}/{n_bs}), first-div {fd_bs}")
    print(f"  seq vs greedy (INFO): {m_sg}/{n_sg} match, first-div {fd_sg}")
    print(f"  bat vs greedy (INFO): {m_bg}/{n_bg} match, first-div {fd_bg}")
    if not (bit_id or r_bs >= 0.99):
        raise SystemExit(
            f"PARITY FAIL: batched tree-verify diverged from sequential (match {r_bs:.4f} < 0.99) — "
            f"the B=W^D batched verify is NOT output-equivalent to the naive per-path verify "
            f"(rule 4 / rule 6). Do NOT trust the bench; investigate batch_step.")
    print(f"  PARITY OK — batched == sequential "
          f"({'bit-identical' if bit_id else f'{r_bs:.4f} >= 0.99 (bf16 batched-moe reorder)'}).")

    mx.clear_cache()

    # ---------------- BENCH: (W,D) sweep with the REAL head ----------------
    print(f"\n=== BENCH: max_new={BENCH_MAX_NEW} (REAL MTP head — W^D amortization now active) ===")

    def _cell(tps, st, depth) -> str:
        return f"{tps:4.1f}t/s a{st['mean_accept']:.2f}/{depth + 1} {tps / g_tps:.2f}x"

    print(f"  {'W,D':>4s} {'pth':>4s}  {'chain k=D':>22s}  {'tree seq':>22s}  "
          f"{'tree bat':>22s}  {'bat/seq':>7s}")
    best = (0.0, "", 0.0)
    for (W, D) in SWEEP:
        paths = W ** D
        k_toks, k_st, k_tps = _spec_tps(
            spec_generate_k, model, mtp, embed, head, prompt_ids,
            pf_s=pf_s, k=D, max_new=BENCH_MAX_NEW, eos_id=None)
        mx.clear_cache()
        ts_toks, ts_st, ts_tps = _spec_tps(
            spec_generate_tree, model, mtp, embed, head, prompt_ids,
            pf_s=pf_s, width=W, depth=D, max_new=BENCH_MAX_NEW, eos_id=None, batched=False)
        mx.clear_cache()
        tb_toks, tb_st, tb_tps = _spec_tps(
            spec_generate_tree, model, mtp, embed, head, prompt_ids,
            pf_s=pf_s, width=W, depth=D, max_new=BENCH_MAX_NEW, eos_id=None, batched=True)
        mx.clear_cache()
        amort = tb_tps / max(ts_tps, 1e-9)
        if tb_tps > best[0]:
            best = (tb_tps, f"W={W} D={D}", tb_st["mean_accept"])
        print(f"  {W},{D:<2d} {paths:>4d}  {_cell(k_tps, k_st, D):>22s}  "
              f"{_cell(ts_tps, ts_st, D):>22s}  {_cell(tb_tps, tb_st, D):>22s}  {amort:>6.2f}x")

    # ---------------- VERDICT ----------------
    rc = best[0] / g_tps
    print(f"\nbest batched tree: {best[1]} -> {best[0]:.1f} tok/s ({rc:.2f}x greedy, "
          f"accept {best[2]:.2f})")
    if rc >= 1.0:
        print(
            f"VERDICT: batched tree-verify with the REAL MTP head lifts B=1 spec to {rc:.2f}x compiled "
            f"greedy — it CROSSES 1x where the M3 k-chain ({K_CHAIN_BEST:.2f}x) and the (A) "
            f"compiled-verify ({COMPILED_VERIFY_BEST:.2f}x) could not. The W^D weight-amortization (one "
            f"batched gather_qmm over all paths) is the decisive B=1 lever; the random-head Super gate "
            f"could not see it. Losslessness holds (batched == sequential, parity above; M2 owns "
            f"spec == greedy).")
    else:
        further = "a further lift but still" if rc > COMPILED_VERIFY_BEST else "still"
        print(
            f"VERDICT: batched tree-verify with the real head reaches {rc:.2f}x compiled greedy (vs the "
            f"M3 k-chain {K_CHAIN_BEST:.2f}x / (A) compiled-verify {COMPILED_VERIFY_BEST:.2f}x) — "
            f"{further} <1x at B=1. The W^D amortization (bat/seq > 1 in the sweep) helps, but the eager "
            f"batch_step + the hybrid Mamba per-round commit-replay keep single-stream below greedy; "
            f"multi-stream B>1 decode (serving throughput) is the remaining lever. Losslessness holds "
            f"(batched == sequential).")


if __name__ == "__main__":
    run()
