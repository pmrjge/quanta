"""REAL Nemotron paged == discrete parity (loads the int4-g64 artifact — GPU/memory, ONE model only).

The deferred rule-4 ship gate for #152 step 3 on the actual model: a request that reuses a resident
prompt prefix (paged attention-KV blocks + restored Mamba recurrent boundary state, suffix-only
prefill) must decode **identically** to a from-scratch full prefill + decode. The model-free
``parity/paged_engine_equiv_test.py`` proves the session orchestration bit-exactly on varied tokens;
THIS proves the real hybrid forward (int8 KV through the real GQA attention with offset-aware causal
masking + the real Mamba SSD resumed from a boundary snapshot) is numerically faithful — top-1
next-token agreement over a greedy decode of a COHERENT prompt (the project's e2e arbiter), the
per-step logits max-abs diff, and the prefix-reuse / recurrent-restore stats.

Loads ~68 GB. Run ALONE — never beside another large-resident job (the OOM-reboot hazard).

    uv run --with tokenizers python -m parity.nemotron_paged_real_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.tokenizer import NemotronTokenizer
from quanta.paged import PagedKVCacheManager, RecurrentPrefixCache
from quanta.shim.omlx import _BaseBatchedSession

NEM_ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
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
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=512,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="nem-paged")
    rec = RecurrentPrefixCache(block_size=BLOCK, model_name="nem-paged", capacity=512)
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
    return toks, rows, mgr.get_stats(), rec.get_stats()


def _max_abs(rows_a, rows_b) -> float:
    m = 0.0
    for a, b in zip(rows_a, rows_b, strict=True):
        m = max(m, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
    return m


def run() -> None:
    rt = NemotronBatchedResidentModel(NEM_ART, max_batch=2)
    tok = NemotronTokenizer(NEM_ART)
    ids = [int(t) for t in tok.encode(PROMPT)]
    plen = PREFIX_BLOCKS * BLOCK
    if len(ids) < plen + SUFFIX_LEN:
        raise SystemExit(f"prompt too short: {len(ids)} tokens (need >= {plen + SUFFIX_LEN})")
    prefix = ids[:plen]
    full = ids[:plen + SUFFIX_LEN]

    discrete, d_rows = _discrete_decode(rt, full, K)
    paged, p_rows, st, rst = _paged_decode(rt, prefix, full, K)

    agree = sum(a == b for a, b in zip(discrete, paged, strict=True))
    logit_max_abs = _max_abs(d_rows, p_rows)
    tokens_eq = discrete == paged
    reused = st.prefix_hit_tokens == len(prefix)
    restored = rst.snapshot_hits >= 1
    varied = len(set(discrete)) > 1                  # a real, non-degenerate continuation
    ok = tokens_eq and reused and restored and varied

    print(f"  prompt_tokens={len(ids)} prefix={len(prefix)} suffix={SUFFIX_LEN}")
    print(f"  discrete={discrete}")
    print(f"  paged   ={paged}")
    print(f"  [{'OK' if tokens_eq else 'FAIL'}] top-1 agreement {agree}/{K} (paged==discrete)")
    print(f"  [{'OK' if varied else 'FAIL'}] decode is varied (distinct tokens={len(set(discrete))})")
    print(f"  [info]  per-step logits max_abs(paged-discrete) = {logit_max_abs:.3e}")
    print(f"  [{'OK' if reused else 'FAIL'}] prefix reused: hit_tokens={st.prefix_hit_tokens} "
          f"(exp {len(prefix)}) hit_blocks={st.prefix_hit_blocks}")
    print(f"  [{'OK' if restored else 'FAIL'}] recurrent boundary restored: snapshot_hits="
          f"{rst.snapshot_hits} stores={rst.snapshot_stores}")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
