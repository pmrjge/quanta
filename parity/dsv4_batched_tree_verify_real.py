"""Real-model parity + bench for DSV4 batched tree-spec verify (docs/batched_tree_verify.md commit 5).

Gates the `batched=True` opt-in for `spec_generate_tree` against the proven `batched=False` path on
the int4-g64 resident DSV4 — bit-identical token streams confirm the W^D-stream batched verify
preserves the lossless contract on real weights. The model-free pilot
(`parity/dsv4_batched_tree_verify_test.py`) gates the LOGIC; this script gates the WEIGHT-level
parity AND the throughput economics in docs/batched_tree_verify.md.

ORCHESTRATOR runs this — do NOT invoke from an agent. Loads ~169 GB resident DSV4 + native MTP
(797 baked `mtp.0.*` keys) — run only with the memory free.

    uv run --with tokenizers python -m parity.dsv4_batched_tree_verify_real

Geometry:
  * ART = `/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64` (int4 AWQ experts + int8 non-experts).
  * PROMPT_T ≈ 128 (short — enough to seed real KV/Indexer/compressor state, cheap enough to time).
  * MAX_NEW = 32 (parity) / 64 (bench).
  * W=2, D=2 → paths_per_round = 4; verify rounds bounded by `max_new / mean_accept`.

The bench numbers report wall-clock; with the native MTP head loaded the accept rate reflects the
real drafter quality, so the speedup tracks the economics table in `docs/batched_tree_verify.md`
(batched should reach ≥ chained's throughput while keeping tree's higher accept rate).
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.runtime import DSV4ResidentModel
from quanta.dsv4.spec import spec_generate_k, spec_generate_tree
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
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
    t0 = time.perf_counter()
    inner = DSV4ResidentModel(ART)
    print(f"[real-parity] resident in {time.perf_counter() - t0:.1f}s "
          f"({inner.cfg.num_hidden_layers} layers, packed={inner.packed_experts}, "
          f"mtp={'on' if inner.mtp is not None else 'OFF — FATAL'})", flush=True)
    if inner.mtp is None:
        raise RuntimeError("DSV4 MTP not loaded — load_mtp defaulted off or artifact missing mtp.0.* keys")

    model = DSV4BatchedResidentModel.from_inner(inner)
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ids_all = tok.encode(PROMPT, add_bos=True)
    ids = ids_all[:PROMPT_T_MAX]
    print(f"[real-parity] tokenized: {len(ids_all)} tok (clipped to {len(ids)}; BOS-first={ids[0] == tok.bos_id})",
          flush=True)

    mtp = inner.mtp
    embed = inner.embed_w
    head = inner.lm_head_w

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
    if bit_id:
        print(f"\n  PARITY OK — bit-identical streams ({len(seq_tokens)} tokens)", flush=True)
    else:
        # fall back to argmax-match per docs/batched_tree_verify.md (SDPA / sorted-MoE may reorder)
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
        # threshold: 0.99 per the docstring tolerance (SDPA reorders ~ rare token flips)
        if ratio < 0.99:
            raise SystemExit(f"PARITY FAIL: argmax_match {ratio:.4f} < 0.99")
        print(f"  argmax_match {ratio:.4f} >= 0.99 — within SDPA-reorder tolerance", flush=True)
    assert same_len or ratio >= 0.99   # noqa: F821

    mx.clear_cache()

    # --- BENCH: spec_generate_k(k=2) vs tree(batched=False) vs tree(batched=True) ---
    print(f"\n=== BENCH: max_new={BENCH_MAX_NEW} (steady-state wall-clock + per-round breakdown) ===",
          flush=True)

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

    # Economics summary: batched vs chained ratio is the key number.
    print(f"\n  ratios (vs chained k=2): tree_seq={tseq_tps / k_tps:.2f}x  "
          f"tree_bat={tbat_tps / k_tps:.2f}x", flush=True)
    print(f"  tree_bat / tree_seq        : {tbat_tps / tseq_tps:.2f}x "
          f"(batched amortization gain on the W^D=4 verify forwards)", flush=True)


if __name__ == "__main__":
    main()
