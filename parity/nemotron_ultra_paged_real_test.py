"""REAL Nemotron-Ultra paged == discrete parity (loads the int4-RTN backbone — GPU/memory, ONE model).

The deferred rule-4 ship gate for U4 "paged-KV on the 12 attn layers": the #152 paged contract
(``make_paged_state`` / ``prefill_paged`` / ``step_batch(paged_batched=...)`` + the #153 batched
KV loop-kill) is already built on :class:`NemotronBatchedResidentModel`, model-free-green
(``parity/paged_engine_equiv_test.py``) AND real-green on the **Super-120B** sibling
(``parity/nemotron_paged_real_test.py`` #174). The real-artifact gate on **Ultra-550B** was the one
piece ``batched_runtime.py:613`` flagged as "deferred, one model at a time" — THIS closes it.

Same proof as the Super gate, pointed at the Ultra backbone: a request that reuses a resident prompt
prefix (paged attention-KV blocks over Ultra's **12** ``attention`` layers + restored Mamba recurrent
boundary state, suffix-only prefill) must decode **identically** to a from-scratch full prefill +
decode. The contract is model-agnostic — ``paged_kv_spec`` derives ``n_layers`` from the artifact
(12 here vs the Super's 8), nothing is hardcoded — so this gate exists to prove the *real hybrid
forward at Ultra scale* is numerically faithful: int8 KV through the real GQA attention with
offset-aware causal masking + the real 48-layer Mamba SSD resumed from a boundary snapshot, arbitrated
by top-1 next-token agreement over a greedy decode (the project's e2e arbiter), the per-step logits
max-abs diff, and the prefix-reuse / recurrent-restore stats.

Loads ~306 GiB (no MTP sidecar — paged-KV is a backbone-only feature). Run ALONE — never beside
another large-resident job (the OOM-reboot hazard); the 400 GiB wired limit pins the weight set.

    uv run --with tokenizers python -m parity.nemotron_ultra_paged_real_test
"""

from __future__ import annotations

import mlx.core as mx

from parity.nemotron_mtp_k_bench import ART  # the Ultra int4-RTN backbone (not hardcoded — rule 6 style)
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer
from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache
from quanta.shim.omlx import _BaseBatchedSession

BLOCK = 16
PREFIX_BLOCKS = 2                       # block-aligned shared prefix = 32 tokens
SUFFIX_LEN = 5
K = 10                                  # greedy decode steps to compare
PROMPT = ("The history of computing spans many centuries. Early mechanical calculators gave way to "
          "electronic machines, and then to the integrated circuit, which made modern processors "
          "possible. Today, language models run on specialized hardware and serve many users at once.")


def _argmax(row: mx.array) -> int:
    return int(mx.argmax(row.reshape(-1)).item())


def _discrete_decode(rt: NemotronBatchedResidentModel, ids: list[int], k: int):
    """Baseline: a from-scratch full prefill + ``k`` greedy steps through the discrete ``KVCache``."""
    state = rt.make_stream_state()
    row = rt.prefill(mx.array(ids), state)[0, -1]
    mx.eval(row)
    toks, rows = [_argmax(row)], [row]
    for _ in range(k - 1):
        row = rt.step_batch([mx.array([toks[-1]])], [state], None)[0][0, -1]
        mx.eval(row)
        toks.append(_argmax(row))
        rows.append(row)
    return toks, rows


def _paged_decode(rt: NemotronBatchedResidentModel, prefix: list[int], full: list[int], k: int):
    """Paged: seq A stores the prefix's KV blocks + recurrent boundary; seq B reuses them and prefills
    only the suffix, then decodes ``k`` greedy steps. Drives the real serving path (``_BaseBatchedSession``
    paged mode) exactly as the Super gate does — the session orchestration is model-agnostic."""
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=512,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="nem-ultra-paged")
    rec = RecurrentPrefixCache(block_size=BLOCK, model_name="nem-ultra-paged", capacity=512)
    sess = _BaseBatchedSession(runtime=rt, capacity=2, manager=mgr, rec_cache=rec)
    sess.admit(0, prefix)                            # seq A: store prefix KV blocks + recurrent boundary
    row = sess.admit(1, full)                        # seq B: reuse prefix + suffix-only prefill
    mx.eval(row)
    toks, rows = [_argmax(row)], [row]
    for _ in range(k - 1):
        row = sess.step_batch({1: toks[-1]})[1]
        mx.eval(row)
        toks.append(_argmax(row))
        rows.append(row)
    return toks, rows, mgr.get_stats(), rec.get_stats(), sess.get_cache_stats()


def _max_abs(rows_a, rows_b) -> float:
    m = 0.0
    for a, b in zip(rows_a, rows_b, strict=True):
        m = max(m, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
    return m


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))           # pin the 306 GiB weight set (Ultra needs this; Super didn't)
    rt = NemotronBatchedResidentModel(ART, max_batch=2)
    tok = NemotronTokenizer(ART)
    ids = [int(t) for t in tok.encode(PROMPT)]
    plen = PREFIX_BLOCKS * BLOCK
    if len(ids) < plen + SUFFIX_LEN:
        raise SystemExit(f"prompt too short: {len(ids)} tokens (need >= {plen + SUFFIX_LEN})")
    prefix = ids[:plen]
    full = ids[:plen + SUFFIX_LEN]

    n_attn = len(rt._attn_globals)                   # the paged (attention) layer count — 12 for Ultra
    print("=== Nemotron-Ultra paged==discrete real gate (int4-RTN backbone) ===")
    print(f"  artifact={ART}")
    print(f"  layers={rt.num_layers} attention(paged)={n_attn} block={BLOCK}")

    discrete, d_rows = _discrete_decode(rt, full, K)
    paged, p_rows, st, rst, estats = _paged_decode(rt, prefix, full, K)

    agree = sum(a == b for a, b in zip(discrete, paged, strict=True))
    logit_max_abs = _max_abs(d_rows, p_rows)
    tokens_eq = discrete == paged
    reused = st.prefix_hit_tokens == len(prefix)
    restored = rst.snapshot_hits >= 1
    varied = len(set(discrete)) > 1                  # a real, non-degenerate continuation
    # engine.get_cache_stats() must report the same paged reuse + recurrent restore the caches saw.
    rec_stats = estats.get("recurrent") if isinstance(estats, dict) else None
    estats_ok = (isinstance(estats, dict)
                 and estats.get("prefix_hit_tokens") == st.prefix_hit_tokens == len(prefix)
                 and estats.get("prefix_hit_blocks") == st.prefix_hit_blocks
                 and isinstance(rec_stats, dict)
                 and rec_stats.get("snapshot_hits") == rst.snapshot_hits
                 and rec_stats.get("snapshot_stores") == rst.snapshot_stores)
    # rule 6: the paged spec MUST be the 12-attention-layer count derived from the artifact, not a stale 8.
    nlayers_ok = st.num_layers == n_attn == sum(1 for k in rt.cfg.layers_block_type if k == "attention")
    ok = tokens_eq and reused and restored and varied and estats_ok and nlayers_ok

    print(f"  prompt_tokens={len(ids)} prefix={len(prefix)} suffix={SUFFIX_LEN}")
    print(f"  discrete={discrete}")
    print(f"  paged   ={paged}")
    print(f"  [{'OK' if tokens_eq else 'FAIL'}] top-1 agreement {agree}/{K} (paged==discrete)")
    print(f"  [{'OK' if nlayers_ok else 'FAIL'}] paged manager covers {st.num_layers} attention layers "
          f"(exp {n_attn})")
    print(f"  [{'OK' if varied else 'FAIL'}] decode is varied (distinct tokens={len(set(discrete))})")
    print(f"  [info]  per-step logits max_abs(paged-discrete) = {logit_max_abs:.3e}")
    print(f"  [{'OK' if reused else 'FAIL'}] prefix reused: hit_tokens={st.prefix_hit_tokens} "
          f"(exp {len(prefix)}) hit_blocks={st.prefix_hit_blocks}")
    print(f"  [{'OK' if restored else 'FAIL'}] recurrent boundary restored: snapshot_hits="
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
