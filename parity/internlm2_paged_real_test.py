"""REAL InternLM2.5 paged == discrete parity (loads the int8-g64 artifact — GPU/memory, ONE model only).

The deferred rule-4 ship gate for #152 step 5 (#176) on the actual model: a request that reuses a
resident prompt prefix (paged attention-KV blocks, suffix-only prefill) must decode **identically** to a
from-scratch full prefill + decode. The model-free ``parity/internlm2_paged_test.py`` proves the session
orchestration bit-exactly on a tiny bf16 model; THIS proves the real dense-GQA forward — int8-g64 KV
through the real attention with offset-aware causal masking + dynamic-NTK RoPE — is numerically faithful:
top-1 next-token agreement over a greedy decode of a COHERENT prompt (the project's e2e arbiter), the
per-step logits max-abs diff, and the prefix-reuse stats. InternLM2.5 has NO recurrent state, so (unlike
Nemotron) there is no boundary snapshot to restore — pure paged attention-KV reuse.

Loads ~6 GB. Run ALONE — never beside another large-resident job (the OOM-reboot hazard).

    uv run --with tokenizers python -m parity.internlm2_paged_real_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.internlm2.batched_runtime import InternLM2BatchedResidentModel
from quanta.internlm2.tokenizer import InternLM2Tokenizer
from quanta.paged import PagedKVCacheManager
from quanta.shim.omlx import _BaseBatchedSession

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
BLOCK = 16
PREFIX_BLOCKS = 2                       # block-aligned shared prefix = 32 tokens
SUFFIX_LEN = 5
K = 10                                  # greedy decode steps to compare
PROMPT = ("The history of computing spans many centuries. Early mechanical calculators gave way to "
          "electronic machines, and then to the integrated circuit, which made modern processors "
          "possible. Today, language models run on specialized hardware and serve many users at once.")


def _argmax(row: mx.array) -> int:
    return int(mx.argmax(row.reshape(-1)).item())


def _discrete_decode(rt: InternLM2BatchedResidentModel, ids: list[int], k: int):
    cache = rt.new_cache()
    row = rt.prefill(mx.array(ids), cache)[0, -1]
    mx.eval(row)
    toks, rows = [_argmax(row)], [row]
    for _ in range(k - 1):
        row = rt.step_batch([mx.array([toks[-1]])], [cache], None)[0][0, -1]
        mx.eval(row)
        toks.append(_argmax(row))
        rows.append(row)
    return toks, rows


def _paged_decode(rt: InternLM2BatchedResidentModel, prefix: list[int], full: list[int], k: int):
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=512,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="ilm2-paged")
    # dense model: NO RecurrentPrefixCache (has_recurrent_state=False) — pure attention-KV reuse.
    sess = _BaseBatchedSession(runtime=rt, capacity=2, manager=mgr)
    sess.admit(0, prefix)                            # seq A: store prefix KV blocks
    row = sess.admit(1, full)                        # seq B: reuse prefix + suffix-only prefill
    mx.eval(row)
    toks, rows = [_argmax(row)], [row]
    for _ in range(k - 1):
        row = sess.step_batch({1: toks[-1]})[1]
        mx.eval(row)
        toks.append(_argmax(row))
        rows.append(row)
    # engine-level stats surface (omlx.py:738) — the smoke deferred by paged_prefix_reuse_test.py:22.
    return toks, rows, mgr.get_stats(), sess.get_cache_stats()


def _max_abs(rows_a, rows_b) -> float:
    m = 0.0
    for a, b in zip(rows_a, rows_b, strict=True):
        m = max(m, float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()))
    return m


def run() -> None:
    rt = InternLM2BatchedResidentModel(ART, max_batch=2)
    tok = InternLM2Tokenizer.from_pretrained(ART)
    ids = [int(t) for t in tok.encode(PROMPT, add_bos=True)]
    plen = PREFIX_BLOCKS * BLOCK
    if len(ids) < plen + SUFFIX_LEN:
        raise SystemExit(f"prompt too short: {len(ids)} tokens (need >= {plen + SUFFIX_LEN})")
    prefix = ids[:plen]
    full = ids[:plen + SUFFIX_LEN]

    discrete, d_rows = _discrete_decode(rt, full, K)
    paged, p_rows, st, estats = _paged_decode(rt, prefix, full, K)

    agree = sum(a == b for a, b in zip(discrete, paged, strict=True))
    logit_max_abs = _max_abs(d_rows, p_rows)
    tokens_eq = discrete == paged
    reused = st.prefix_hit_tokens == len(prefix)
    varied = len(set(discrete)) > 1                  # a real, non-degenerate continuation
    # engine.get_cache_stats() must report the same reuse the manager saw (the deferred smoke).
    estats_ok = (isinstance(estats, dict)
                 and estats.get("prefix_hit_tokens") == st.prefix_hit_tokens == len(prefix)
                 and estats.get("prefix_hit_blocks") == st.prefix_hit_blocks)
    ok = tokens_eq and reused and varied and estats_ok

    print(f"  prompt_tokens={len(ids)} prefix={len(prefix)} suffix={SUFFIX_LEN}")
    print(f"  discrete={discrete}")
    print(f"  paged   ={paged}")
    print(f"  [{'OK' if tokens_eq else 'FAIL'}] top-1 agreement {agree}/{K} (paged==discrete)")
    print(f"  [{'OK' if varied else 'FAIL'}] decode is varied (distinct tokens={len(set(discrete))})")
    print(f"  [info]  per-step logits max_abs(paged-discrete) = {logit_max_abs:.3e}")
    print(f"  [{'OK' if reused else 'FAIL'}] prefix reused: hit_tokens={st.prefix_hit_tokens} "
          f"(exp {len(prefix)}) hit_blocks={st.prefix_hit_blocks}")
    e_tok = estats.get("prefix_hit_tokens") if isinstance(estats, dict) else estats
    e_blk = estats.get("prefix_hit_blocks") if isinstance(estats, dict) else None
    print(f"  [{'OK' if estats_ok else 'FAIL'}] engine.get_cache_stats(): "
          f"hit_tokens={e_tok} hit_blocks={e_blk}")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
