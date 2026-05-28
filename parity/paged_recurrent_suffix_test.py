"""Recurrent-prefix suffix-compute parity (model-free, tiny tensors) — #152 step 3, the hybrid enabler.

The hard part of paging a HYBRID model (Nemotron Mamba / Qwen3.5 GatedDeltaNet): a recurrent layer's
state at position ``n`` depends on every token ``0..n-1``, so you cannot skip a shared prefix during
prefill the way you can for pure attention. The escape hatch (this gate proves it): the recurrent
state at a block boundary is a **deterministic function of the prefix tokens**, so snapshotting it at
that boundary and restoring it lets a later request resume the recurrence at the boundary and process
**only the suffix** — bit-identically to running the whole sequence.

A faithful tiny stand-in for the recurrence: a gated accumulator ``state = decay*state + E[token]``
(order- and history-dependent, deterministic — the essential property of Mamba/GDN state). Two checks:

1. **resume == full.** Run the recurrence over the whole sequence (capturing the state at each block
   boundary). Then: a first sequence stores its prefix-boundary snapshots into a real
   :class:`~quanta.paged.recurrent_cache.RecurrentPrefixCache`; a second sequence sharing that prefix
   restores the boundary state by content hash (``lookup_at`` at the boundary the paged KV matched)
   and runs ONLY the suffix. Its final state is bit-identical (max_abs == 0) to the one-shot run.

2. **cache plumbing.** Cross-turn reuse survives a freed sequence (snapshot keyed by token-chain hash,
   not by sequence); ``match`` returns the deepest stored boundary; a miss returns ``(0, None)``; the
   LRU evicts the least-recently-used boundary past ``capacity``; ``model_name`` salts the hash (a
   different model never reuses another's snapshot).

Model-free: a few floats, fixed seed, ms on CPU — SAFE alongside a live GPU job.

    uv run python -m parity.paged_recurrent_suffix_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.paged.recurrent_cache import RecurrentPrefixCache

D = 8
BLOCK = 4
DECAY = 0.9


def _embed(seed: int, vocab: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((vocab, D)).astype(mx.float32)


def _run_from(emb: mx.array, ids: list[int], state: mx.array, base: int,
              snaps: dict[int, mx.array]) -> tuple[mx.array, dict[int, mx.array]]:
    """Gated-accumulator recurrence ``state = decay*state + E[token]`` from ``state`` over ``ids``
    (whose first token sits at absolute position ``base``); returns (final_state, snaps) where snaps
    records the state after each token landing on a BLOCK boundary (the snapshot-able positions)."""
    for j, t in enumerate(ids):
        state = DECAY * state + emb[t]
        abs_pos = base + j + 1  # position AFTER consuming this token
        if abs_pos % BLOCK == 0:
            snaps[abs_pos] = state
    mx.eval(state)
    return state, snaps


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a - b)).item())


def _check_resume_equals_full() -> bool:
    vocab = 64
    emb = _embed(0, vocab)
    prompt = list(range(10, 10 + 8))      # 8 tokens = 2 full blocks (BLOCK=4)
    suffix = list(range(40, 40 + 5))      # 5-token suffix (crosses one more boundary at pos 12)
    full_ids = prompt + suffix
    zero = mx.zeros((D,), dtype=mx.float32)

    # one-shot reference over the whole sequence
    full_state, full_snaps = _run_from(emb, full_ids, zero, 0, {})

    # sequence A: process the prompt, store its per-boundary snapshots (boundaries at pos 4, 8)
    _a_state, a_snaps = _run_from(emb, prompt, zero, 0, {})
    cache = RecurrentPrefixCache(block_size=BLOCK, model_name="rec-test", capacity=64)
    n_full_prompt = len(prompt) // BLOCK
    payloads = [a_snaps[(bi + 1) * BLOCK] for bi in range(n_full_prompt)]  # state after block bi
    cache.store(prompt, payloads)

    # sequence B (shares the prompt as prefix): the paged KV matched n=8 tokens (2 blocks); restore the
    # recurrent boundary state at exactly that boundary and run ONLY the suffix.
    n = len(prompt)                       # what match_prefix would return (2 * BLOCK)
    restored = cache.lookup_at(full_ids, n)
    if restored is None:
        print("(1) resume==full: FAIL (no snapshot at matched boundary)")
        return False
    b_state, _ = _run_from(emb, suffix, restored, n, {})

    eq = _max_abs(b_state, full_state) == 0.0
    # the restored boundary state must equal the one-shot state at that boundary too
    boundary_eq = _max_abs(restored, full_snaps[n]) == 0.0
    ok = eq and boundary_eq
    print(f"(1) resume==full: suffix_final==oneshot={eq} boundary_restore_exact={boundary_eq} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def _check_cache_plumbing() -> bool:
    vocab = 64
    emb = _embed(1, vocab)
    zero = mx.zeros((D,), dtype=mx.float32)
    ok = True

    # cross-turn: store from one prompt, retrieve after that "sequence" is gone (hash-keyed, not seq).
    prompt = list(range(5, 5 + 12))       # 3 full blocks
    _s, snaps = _run_from(emb, prompt, zero, 0, {})
    cache = RecurrentPrefixCache(block_size=BLOCK, model_name="m", capacity=64)
    cache.store(prompt, [snaps[(bi + 1) * BLOCK] for bi in range(3)])

    # match() returns the deepest stored boundary (here the full 3-block prefix, 12 tokens)
    n_match, payload = cache.match(prompt + [99, 98])  # longer query, shares the 12-token prefix
    deepest_ok = n_match == 12 and payload is not None and _max_abs(payload, snaps[12]) == 0.0

    # lookup_at a shallower boundary (4 tokens) also hits (stored every boundary)
    shallow = cache.lookup_at(prompt, 4)
    shallow_ok = shallow is not None and _max_abs(shallow, snaps[4]) == 0.0

    # a miss (unknown prefix) returns (0, None)
    miss_n, miss_p = cache.match([1000, 1001, 1002, 1003])
    miss_ok = miss_n == 0 and miss_p is None

    # model_name salts the hash: a differently-named cache never reuses this one's snapshot
    other = RecurrentPrefixCache(block_size=BLOCK, model_name="other", capacity=64)
    other.store(prompt, [snaps[4]])
    cross_n, _ = cache.match(prompt)  # cache still only knows its own store
    salt_ok = cross_n == 12  # unaffected by `other`

    # LRU eviction past capacity: capacity=1 keeps only the most-recently-stored boundary
    lru = RecurrentPrefixCache(block_size=BLOCK, model_name="m", capacity=1)
    p1 = list(range(0, 4))
    p2 = list(range(100, 104))
    _s1, sn1 = _run_from(emb, p1, zero, 0, {})
    _s2, sn2 = _run_from(emb, p2, zero, 0, {})
    lru.store(p1, [sn1[4]])
    lru.store(p2, [sn2[4]])               # evicts p1 (LRU)
    evicted = lru.lookup_at(p1, 4) is None
    kept = lru.lookup_at(p2, 4) is not None
    lru_ok = evicted and kept and lru.get_stats().evictions == 1

    ok = deepest_ok and shallow_ok and miss_ok and salt_ok and lru_ok
    print(f"(2) plumbing: deepest_match={deepest_ok} shallow_hit={shallow_ok} miss={miss_ok} "
          f"name_salt={salt_ok} lru_evict={lru_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run() -> None:
    ok = True
    ok = _check_resume_equals_full() and ok
    ok = _check_cache_plumbing() and ok
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
