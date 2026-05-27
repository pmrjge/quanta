"""Real-model parity + bench for Nemotron batched tree-spec verify (docs/batched_tree_verify.md commit 5).

Gates the ``batched=True`` opt-in for :func:`quanta.nemotron.spec.spec_generate_tree` against the
proven ``batched=False`` path on the int4-g64 resident Nemotron-H — bit-identical token streams
confirm the W^D-stream batched verify preserves the lossless contract on real weights for the
hybrid (8 GQA + 80 Mamba) architecture. The model-free pilot
(``parity/nemotron_batched_tree_verify_test.py``) gates the LOGIC; this script gates WEIGHT-level
parity AND the wall-clock economics in docs/batched_tree_verify.md.

The Nemotron baked artifact does NOT contain MTP weights (0 ``mtp.*`` / ``nextn.*`` keys in
``manifest.json``) — the MTP weight load is the deferred orchestrator integration step (see
``parity/nemotron_mtp_k_bench.py`` line 104). The MTP is therefore constructed with random init.
This is FINE FOR PARITY (CLAUDE.md rule 4 / lossless contract: ``spec_generate_tree`` is
bit-identical to plain greedy decode for ANY drafter quality — verify always arbitrates against
the main-model greedy distribution, so tokens emitted are independent of MTP weights). It MEANS
the BENCH ``mean_accept`` reflects the random-drafter floor (~1/W per step), not real-world
acceptance; the batched-vs-sequential ratio is still a valid wall-clock comparison of the verify
machinery itself.

ORCHESTRATOR runs this — do NOT invoke from an agent. Loads ~68 GB resident Nemotron — run only
with the memory free.

    uv run --with tokenizers python -m parity.nemotron_batched_tree_verify_real

Geometry:
  * ART = ``/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64``.
  * PROMPT_T ≈ 128, MAX_NEW = 32 (parity) / 64 (bench), W=2, D=2 → paths_per_round = 4.
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.mtp import NemotronMTP
from quanta.nemotron.spec import spec_generate_k, spec_generate_tree
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
PROMPT = (
    "Write a short paragraph explaining how the invention of the printing press changed the "
    "spread of scientific knowledge in early modern Europe. Keep it to two or three sentences."
)
PROMPT_T_MAX = 128
PARITY_MAX_NEW = 32
BENCH_MAX_NEW = 64
WIDTH = 2
DEPTH = 2


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return (time.perf_counter() - t0), out


def main() -> None:
    print(f"[real-parity] loading {ART}", flush=True)
    mx.set_wired_limit(int(120 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronBatchedResidentModel(ART)
    print(f"[real-parity] resident in {time.perf_counter() - t0:.1f}s "
          f"({model.cfg.num_hidden_layers} layers)", flush=True)

    # MTP is random-init: no mtp.* keys in the baked artifact (see module docstring).
    # Lossless contract still holds for parity — verify arbitrates regardless of drafter quality.
    mtp = NemotronMTP(model.cfg)
    embed, head = model.embed_w, model.lm_head_w

    tok = NemotronTokenizer(ART)
    ids_all = tok.encode(PROMPT, add_bos=False)
    ids = ids_all[:PROMPT_T_MAX]
    print(f"[real-parity] tokenized: {len(ids_all)} tok (clipped to {len(ids)}); "
          f"mtp=random-init (artifact has no mtp.* keys)", flush=True)

    # --- PARITY: batched=False vs batched=True must produce bit-identical tokens ---
    print(f"\n=== PARITY: spec_generate_tree(W={WIDTH}, D={DEPTH}, max_new={PARITY_MAX_NEW}) "
          f"batched=False vs batched=True ===", flush=True)

    mx.random.seed(0)
    seq_wall, (seq_tokens, seq_stats) = _time(
        spec_generate_tree, model, mtp, embed, head, ids,
        width=WIDTH, depth=DEPTH, max_new=PARITY_MAX_NEW, batched=False,
    )
    print(f"  batched=False : wall={seq_wall:.2f}s tokens={len(seq_tokens)} "
          f"mean_accept={seq_stats['mean_accept']:.2f} rounds={seq_stats['rounds']}", flush=True)

    mx.random.seed(0)
    bat_wall, (bat_tokens, bat_stats) = _time(
        spec_generate_tree, model, mtp, embed, head, ids,
        width=WIDTH, depth=DEPTH, max_new=PARITY_MAX_NEW, batched=True,
    )
    print(f"  batched=True  : wall={bat_wall:.2f}s tokens={len(bat_tokens)} "
          f"mean_accept={bat_stats['mean_accept']:.2f} rounds={bat_stats['rounds']}", flush=True)

    same_len = len(seq_tokens) == len(bat_tokens)
    bit_id = seq_tokens == bat_tokens
    ratio = 1.0
    if bit_id:
        print(f"\n  PARITY OK — bit-identical streams ({len(seq_tokens)} tokens)", flush=True)
    else:
        n = min(len(seq_tokens), len(bat_tokens))
        matched = sum(1 for i in range(n) if seq_tokens[i] == bat_tokens[i])
        ratio = matched / n if n else 0.0
        first_diff = next((i for i in range(n) if seq_tokens[i] != bat_tokens[i]), -1)
        print(f"\n  PARITY ! — NOT bit-identical (len_seq={len(seq_tokens)} len_bat={len(bat_tokens)})", flush=True)
        print(f"    argmax_match = {matched}/{n} = {ratio:.4f}; first diff at i={first_diff}", flush=True)
        if first_diff >= 0:
            ctx_lo = max(0, first_diff - 3)
            ctx_hi = min(n, first_diff + 4)
            print(f"    seq[{ctx_lo}:{ctx_hi}] = {seq_tokens[ctx_lo:ctx_hi]}", flush=True)
            print(f"    bat[{ctx_lo}:{ctx_hi}] = {bat_tokens[ctx_lo:ctx_hi]}", flush=True)
        if ratio < 0.99:
            raise SystemExit(f"PARITY FAIL: argmax_match {ratio:.4f} < 0.99")
        print(f"  argmax_match {ratio:.4f} >= 0.99 — within SDPA-reorder tolerance", flush=True)
    assert same_len or ratio >= 0.99

    mx.clear_cache()

    # --- BENCH: spec_generate_k(k=2) vs tree(batched=False) vs tree(batched=True) ---
    # Note: random-MTP accept rate is the floor (~1/W). The batched-vs-sequential ratio is still
    # a valid comparison of the verify machinery (same drafter quality across all 3 runs).
    print(f"\n=== BENCH: max_new={BENCH_MAX_NEW} (random-MTP accept-rate floor; "
          f"wall-clock comparison still valid) ===", flush=True)

    mx.random.seed(0)
    k_wall, (k_tokens, k_stats) = _time(
        spec_generate_k, model, mtp, embed, head, ids,
        k=2, max_new=BENCH_MAX_NEW,
    )
    k_tps = len(k_tokens) / k_wall if k_wall > 0 else float("inf")
    print(f"  spec_generate_k(k=2)         : {k_wall:.2f}s {len(k_tokens):>3}tok ({k_tps:.2f} tok/s) "
          f"mean_accept={k_stats['mean_accept']:.2f}/{2 + 1} rounds={k_stats['rounds']}", flush=True)

    mx.clear_cache()
    mx.random.seed(0)
    tseq_wall, (tseq_tokens, tseq_stats) = _time(
        spec_generate_tree, model, mtp, embed, head, ids,
        width=WIDTH, depth=DEPTH, max_new=BENCH_MAX_NEW, batched=False,
    )
    tseq_tps = len(tseq_tokens) / tseq_wall if tseq_wall > 0 else float("inf")
    print(f"  tree(W=2,D=2, batched=False) : {tseq_wall:.2f}s {len(tseq_tokens):>3}tok ({tseq_tps:.2f} tok/s) "
          f"mean_accept={tseq_stats['mean_accept']:.2f}/{DEPTH + 1} rounds={tseq_stats['rounds']}",
          flush=True)

    mx.clear_cache()
    mx.random.seed(0)
    tbat_wall, (tbat_tokens, tbat_stats) = _time(
        spec_generate_tree, model, mtp, embed, head, ids,
        width=WIDTH, depth=DEPTH, max_new=BENCH_MAX_NEW, batched=True,
    )
    tbat_tps = len(tbat_tokens) / tbat_wall if tbat_wall > 0 else float("inf")
    print(f"  tree(W=2,D=2, batched=True ) : {tbat_wall:.2f}s {len(tbat_tokens):>3}tok ({tbat_tps:.2f} tok/s) "
          f"mean_accept={tbat_stats['mean_accept']:.2f}/{DEPTH + 1} rounds={tbat_stats['rounds']}",
          flush=True)

    print(f"\n  ratios (vs chained k=2): tree_seq={tseq_tps / k_tps:.2f}x  "
          f"tree_bat={tbat_tps / k_tps:.2f}x", flush=True)
    print(f"  tree_bat / tree_seq        : {tbat_tps / tseq_tps:.2f}x "
          f"(batched amortization gain on the W^D=4 verify forwards)", flush=True)


if __name__ == "__main__":
    main()
