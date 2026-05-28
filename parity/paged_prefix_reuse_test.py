"""Paged KV prefix-reuse parity (model-free, tiny synthetic tensors) — #152 step 1, gate (b).

The whole point of the paged cache: a second request (or a later agentic turn) that shares a prompt
prefix re-references the resident prefix blocks instead of recomputing them, and the result is
**identical** to a one-shot prefill of the full sequence. Two checks:

1. **Concurrent reuse == one-shot.** Sequence A stores a prompt's KV (its full blocks get content-
   hashed). Sequence B ``match_prefix``-es the same prompt + a suffix: it reuses A's full prefix
   blocks (ref-count++), writes only the suffix, and its gathered stream is bit-identical to a discrete
   :class:`~quanta.nemotron.attention.KVCache` fed the whole prompt+suffix stream. Prefix-hit stats
   reflect the reuse, and A is unaffected (its own gather still matches).

2. **Cross-turn reuse survives free.** After A is freed, its full prefix blocks stay resident (ref 0 but
   still hashed). A later sequence B re-hits them (resurrected from the LRU free list) and reproduces
   the same KV — the multi-turn agentic win (turn N+1's prompt = turn N's text + more).

Model-free: a few KB of random tensors, fixed seed, runs in ms on CPU — SAFE alongside a live GPU job.

    uv run python -m parity.paged_prefix_reuse_test

deferred (run later on GPU): a real serving prefix-reuse decode (shared system prompt across requests)
must match the no-reuse decode token-for-token, and ``engine.get_cache_stats()`` must report the hits.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.nemotron.attention import KVCache
from quanta.paged.paged_kv_cache import PagedKVCacheManager

N_KV = 2
HEAD_DIM = 128
GROUP_SIZE = 64
BLOCK_SIZE = 4
NUM_LAYERS = 2


def _stream(seed: int, n: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    k = mx.random.normal((1, N_KV, n, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal((1, N_KV, n, HEAD_DIM)).astype(mx.bfloat16)
    return k, v


def _discrete_final(k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
    cache = KVCache(quantized=True, group_size=GROUP_SIZE)
    kf = vf = None
    for t in range(k.shape[2]):
        kf, vf = cache.update(k[:, :, t:t + 1], v[:, :, t:t + 1])
    mx.eval(kf, vf)
    return kf, vf


def _mgr() -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=NUM_LAYERS, block_size=BLOCK_SIZE, max_blocks=64,
                               group_size=GROUP_SIZE, quantized=True, model_name="prefix-test")


def _write_run(mgr: PagedKVCacheManager, seq, ids: list[int], k: mx.array, v: mx.array, start: int) -> None:
    """Append tokens ``ids`` (k/v slice [start:start+len(ids)]) one step at a time, all layers."""
    for j, tid in enumerate(ids):
        t = start + j
        mgr.advance(seq, [tid])
        for L in range(NUM_LAYERS):
            mgr.view(seq, L).update(k[:, :, t:t + 1], v[:, :, t:t + 1])
        mgr.commit(seq)


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _check_concurrent() -> bool:
    P, S = 8, 3                       # prompt = 2 full blocks (block_size 4); suffix = 3
    prompt_ids = list(range(100, 100 + P))
    suffix_ids = list(range(900, 900 + S))
    k, v = _stream(seed=0, n=P + S)

    mgr = _mgr()
    seqA = mgr.new_sequence()
    _write_run(mgr, seqA, prompt_ids, k, v, start=0)         # A stores the prompt; full blocks hashed

    seqB = mgr.new_sequence()
    n_match = mgr.match_prefix(seqB, prompt_ids + suffix_ids)  # B reuses A's 2 full prefix blocks
    _write_run(mgr, seqB, suffix_ids, k, v, start=P)          # B writes only the suffix

    bk, bv = mgr.gather(seqB, 0)
    ak, av = mgr.gather(seqA, 0)
    dk, dv = _discrete_final(k, v)                            # one-shot over the whole P+S stream
    dpk, dpv = _discrete_final(k[:, :, :P], v[:, :, :P])      # A's reference (prompt only)
    mx.eval(bk, bv, ak, av)

    st = mgr.get_stats()
    match_ok = n_match == P
    reuse_eq = _max_abs(dk, bk) == 0.0 and _max_abs(dv, bv) == 0.0
    a_intact = _max_abs(dpk, ak) == 0.0 and _max_abs(dpv, av) == 0.0
    stats_ok = st.prefix_hit_blocks == (P // BLOCK_SIZE) and st.prefix_hit_tokens == P
    ok = match_ok and reuse_eq and a_intact and stats_ok
    print(f"(b1) concurrent: n_match={n_match}(exp {P}) reuse==one-shot={reuse_eq} A_intact={a_intact} "
          f"hit_blocks={st.prefix_hit_blocks} hit_tokens={st.prefix_hit_tokens} -> {'PASS' if ok else 'FAIL'}")
    return ok


def _check_cross_turn() -> bool:
    P, S = 8, 3
    prompt_ids = list(range(100, 100 + P))
    suffix_ids = list(range(900, 900 + S))
    k, v = _stream(seed=0, n=P + S)

    mgr = _mgr()
    seqA = mgr.new_sequence()
    _write_run(mgr, seqA, prompt_ids, k, v, start=0)
    mgr.free(seqA)                                            # turn ends; prefix blocks stay resident

    seqB = mgr.new_sequence()
    n_match = mgr.match_prefix(seqB, prompt_ids + suffix_ids)  # later turn re-hits the idle blocks
    _write_run(mgr, seqB, suffix_ids, k, v, start=P)
    bk, bv = mgr.gather(seqB, 0)
    dk, dv = _discrete_final(k, v)
    mx.eval(bk, bv)

    match_ok = n_match == P
    eq = _max_abs(dk, bk) == 0.0 and _max_abs(dv, bv) == 0.0
    ok = match_ok and eq
    print(f"(b2) cross-turn (after free): n_match={n_match}(exp {P}) reuse==one-shot={eq} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def run() -> None:
    ok = True
    ok = _check_concurrent() and ok
    ok = _check_cross_turn() and ok
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
