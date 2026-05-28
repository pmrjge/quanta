"""REAL DSV4-Flash paged == discrete parity (loads the int4-g64 artifact — GPU/memory, ONE model only).

The deferred rule-4 ship gate for #152 step 4 (#175) on the actual model — the last gate before
``PAGED_KV_DEFAULT`` can flip True and close #152. A request that reuses a resident prompt prefix
(paged latent-KV blocks + the derived compressed-KV / lightning-indexer / raw-hidden ring restored from
a content-addressed boundary snapshot, suffix-only prefill) must decode **identically** to a from-scratch
full prefill + decode. The model-free ``parity/dsv4_paged_latent_test.py`` proves the snapshot/restore +
single-stream codec bit-exactly across the dense / ratio-4 / ratio-128 regimes; THIS proves the real
43-layer forward (3 attention regimes, real fp4-baked experts, real compressor + indexer) is numerically
faithful: top-1 next-token agreement over a greedy decode of a COHERENT prompt (the project's e2e
arbiter), the per-step logits max-abs diff, the prefix-reuse stats, and the recurrent-snapshot restore.

DSV4 is the biggest + riskiest keeper — loads ~180 GiB (int4-packed via gather_qmm; ~389 GiB only on
the bf16 ``packed_experts=False`` debug path). Run ALONE, never beside another large-resident
job (the OOM-reboot hazard, [[feedback-memory-safety]]).

    uv run --with regex python -m parity.dsv4_paged_real_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer
from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache
from quanta.shim.omlx import _BaseBatchedSession

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
BLOCK = 16
PREFIX_BLOCKS = 2                       # block-aligned shared prefix = 32 tokens
SUFFIX_LEN = 5
K = 10                                  # greedy decode steps to compare
PROMPT = ("The history of computing spans many centuries. Early mechanical calculators gave way to "
          "electronic machines, and then to the integrated circuit, which made modern processors "
          "possible. Today, language models run on specialized hardware and serve many users at once.")


def _argmax(row: mx.array) -> int:
    return int(mx.argmax(row.reshape(-1)).item())


def _discrete_decode(rt: DSV4BatchedResidentModel, ids: list[int], k: int):
    cache = rt.make_cache()
    row = rt.prefill(mx.array(ids), cache)[0, -1]
    mx.eval(row)
    toks, rows = [_argmax(row)], [row]
    for _ in range(k - 1):
        # DSV4 step_batch validates cache.offset == declared offset (rule 6) — pass the live offset.
        row = rt.step_batch([mx.array([toks[-1]])], [cache], [cache.offset])[0][0, -1]
        mx.eval(row)
        toks.append(_argmax(row))
        rows.append(row)
    return toks, rows


def _paged_decode(rt: DSV4BatchedResidentModel, prefix: list[int], full: list[int], k: int):
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=512,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="dsv4-paged",
                              single_stream=spec.get("single_stream", False))
    rec = RecurrentPrefixCache(block_size=BLOCK, model_name="dsv4-paged", capacity=512)
    sess = _BaseBatchedSession(runtime=rt, capacity=2, manager=mgr, rec_cache=rec)
    sess.admit(0, prefix)                            # seq A: store prefix latent blocks + derived snapshot
    row = sess.admit(1, full)                        # seq B: reuse prefix + suffix-only prefill
    mx.eval(row)
    toks, rows = [_argmax(row)], [row]
    for _ in range(k - 1):
        row = sess.step_batch({1: toks[-1]})[1]
        mx.eval(row)
        toks.append(_argmax(row))
        rows.append(row)
    # engine-level stats surface (omlx.py:738) — the smoke deferred by paged_prefix_reuse_test.py:22.
    return toks, rows, mgr.get_stats(), rec.get_stats(), sess.get_cache_stats()


def _max_abs(rows_a, rows_b) -> float:
    m = 0.0
    for a, b in zip(rows_a, rows_b, strict=True):
        m = max(m, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
    return m


def run() -> None:
    rt = DSV4BatchedResidentModel(ART, max_batch=2)
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ids = [int(t) for t in tok.encode(PROMPT, add_bos=True)]
    plen = PREFIX_BLOCKS * BLOCK
    if len(ids) < plen + SUFFIX_LEN:
        raise SystemExit(f"prompt too short: {len(ids)} tokens (need >= {plen + SUFFIX_LEN})")
    prefix = ids[:plen]
    full = ids[:plen + SUFFIX_LEN]

    discrete, d_rows = _discrete_decode(rt, full, K)
    paged, p_rows, st, rst, estats = _paged_decode(rt, prefix, full, K)

    agree = sum(a == b for a, b in zip(discrete, paged, strict=True))
    logit_max_abs = _max_abs(d_rows, p_rows)
    tokens_eq = discrete == paged
    reused = st.prefix_hit_tokens == len(prefix)
    restored = rst.snapshot_hits >= 1
    varied = len(set(discrete)) > 1                  # a real, non-degenerate continuation
    # engine.get_cache_stats() must report the same paged reuse + derived restore the caches saw.
    rec_stats = estats.get("recurrent") if isinstance(estats, dict) else None
    estats_ok = (isinstance(estats, dict)
                 and estats.get("prefix_hit_tokens") == st.prefix_hit_tokens == len(prefix)
                 and estats.get("prefix_hit_blocks") == st.prefix_hit_blocks
                 and isinstance(rec_stats, dict)
                 and rec_stats.get("snapshot_hits") == rst.snapshot_hits
                 and rec_stats.get("snapshot_stores") == rst.snapshot_stores)
    ok = tokens_eq and reused and restored and varied and estats_ok

    print(f"  prompt_tokens={len(ids)} prefix={len(prefix)} suffix={SUFFIX_LEN}")
    print(f"  discrete={discrete}")
    print(f"  paged   ={paged}")
    print(f"  [{'OK' if tokens_eq else 'FAIL'}] top-1 agreement {agree}/{K} (paged==discrete)")
    print(f"  [{'OK' if varied else 'FAIL'}] decode is varied (distinct tokens={len(set(discrete))})")
    print(f"  [info]  per-step logits max_abs(paged-discrete) = {logit_max_abs:.3e}")
    print(f"  [{'OK' if reused else 'FAIL'}] prefix reused: hit_tokens={st.prefix_hit_tokens} "
          f"(exp {len(prefix)}) hit_blocks={st.prefix_hit_blocks}")
    print(f"  [{'OK' if restored else 'FAIL'}] derived boundary restored: snapshot_hits="
          f"{rst.snapshot_hits} stores={rst.snapshot_stores}")
    e_tok = estats.get("prefix_hit_tokens") if isinstance(estats, dict) else estats
    e_blk = estats.get("prefix_hit_blocks") if isinstance(estats, dict) else None
    e_hit = rec_stats.get("snapshot_hits") if isinstance(rec_stats, dict) else None
    print(f"  [{'OK' if estats_ok else 'FAIL'}] engine.get_cache_stats(): hit_tokens={e_tok} "
          f"hit_blocks={e_blk} rec_snapshot_hits={e_hit}")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
