"""Model-free gate: PagedDSV4Cache.replicate(B) — the cache half of tree-spec verify over paged KV (#158-160).

DSV4's paged decode cache (``quanta.dsv4.decode.paged_cache``) now satisfies the batched tree-spec
verify contract: ``replicate(B)`` forks the underlying sequence B ways (sequence-level copy-on-write via
``PagedKVCacheManager.replicate`` — ONE fork clones all layers together) and rebuilds B paged bundles,
each carrying its source layer's per-stream derived state by structural sharing. The discrete
``DSV4Cache._copy`` is the WRONG hook for paged (it builds a non-paged ``_LayerCache``, dropping the
paged view) and now fails loud on the paged subclass.

This drives the REAL paged single-stream latent lifecycle (advance -> append_kv -> commit -> gather) —
no model, no checkpoint, no GPU — and proves:

  A. **COW isolation** — ``replicate(B)`` returns B distinct paged caches at the prefix offset; each
     replica's gathered latent == prefix + ITS OWN divergent tail (prefix blocks COW-shared, the
     mid-block tail write clones), siblings differ on the tail, and the original (un-replicated) cache
     is bit-identical to before replicate (read-only prefix).
  B. **derived-state sharing** — per-stream compressed-KV / indexer-KV / raw-hidden ring is shared into
     each replica by reference and diverges losslessly on append (MLX immutability), exactly as the
     discrete ``_layer_shallow_copy``.
  C. **fail-loud** — ``PagedDSV4Cache._copy`` raises (rule 6): the per-layer copy is the wrong primitive
     for paged latent; the paged path replicates by forking the sequence.

    uv run python -m parity.dsv4_paged_replicate_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.decode import PagedDSV4Cache, paged_cache
from quanta.paged import PagedKVCacheManager

HEAD_DIM = 128       # one int8-g128 group per token -> exercises the REAL latent quant round-trip
N_LAYERS = 2
BLOCK = 4
GROUP = 128


def _mgr() -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=N_LAYERS, block_size=BLOCK, max_blocks=128,
                               group_size=GROUP, bits=8, quantized=True, single_stream=True,
                               model_name="paged-replicate-test")


def _latent(seed: int) -> list[mx.array]:
    """One token's latent per layer: a list of ``[1, 1, HEAD_DIM]`` (the single-stream codec input)."""
    return [mx.random.normal((1, 1, HEAD_DIM), key=mx.random.key(seed * 100 + L)) for L in range(N_LAYERS)]


def _write_tok(mgr: PagedKVCacheManager, cache: PagedDSV4Cache, tok_id: int, latent: list[mx.array]) -> None:
    seq = cache._seq
    mgr.advance(seq, [tok_id])
    for L in range(N_LAYERS):
        cache.layers[L].append_kv(latent[L])
    mgr.commit(seq)


def _gather(cache: PagedDSV4Cache, layer: int = 0) -> mx.array:
    kv = cache.layers[layer].kv          # gather prefix blocks + suffix -> bf16 latent [1, n_written, C]
    mx.eval(kv)
    return kv


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_replicate_cow_isolation() -> None:
    mgr = _mgr()
    cacheA = paged_cache(mgr, mgr.new_sequence(), N_LAYERS, quantized=True, group_size=GROUP)
    # prefix: 6 tokens -> 1 full block(4) + partial tail(2). A divergent write at pos 6 lands mid the
    # shared tail block, so each replica must COW-clone it (the strong isolation path).
    P, B = 6, 3
    for t in range(P):
        _write_tok(mgr, cacheA, 1000 + t, _latent(t))
    assert cacheA.offset == P, f"prefix offset {cacheA.offset} != {P}"
    prefix_kv = _gather(cacheA)

    reps = cacheA.replicate(B)
    assert len(reps) == B and all(isinstance(r, PagedDSV4Cache) for r in reps), "replicate must return B paged caches"
    assert all(r.offset == P for r in reps), "replicas must start at the prefix offset"
    assert len({r._seq.seq_id for r in reps}) == B and all(r._seq.seq_id != cacheA._seq.seq_id for r in reps), \
        "replicas must be distinct forked sequences (not the parent)"

    tails = [_latent(500 + j) for j in range(B)]
    for j, r in enumerate(reps):
        _write_tok(mgr, r, 9100 + j, tails[j])     # each replica appends its OWN divergent tail at pos P

    gathered, each_ok = [], True
    for r in reps:
        rk = _gather(r)
        gathered.append(rk)
        if rk.shape[1] != P + 1 or _maxabs(rk[:, :P], prefix_kv) != 0.0:
            each_ok = False                        # prefix portion must be bit-identical (COW shared)
    diverged = all(_maxabs(gathered[i][:, P:], gathered[i + 1][:, P:]) > 0.0 for i in range(B - 1))
    parent_ok = (cacheA.offset == P and _maxabs(_gather(cacheA), prefix_kv) == 0.0)

    assert each_ok and diverged and parent_ok, f"each_ok={each_ok} diverged={diverged} parent_ok={parent_ok}"
    print(f"A replicate(B={B}) COW-isolated: prefix bit-shared, tails diverge, parent read-only  ok")


def test_derived_state_shared_then_diverges() -> None:
    mgr = _mgr()
    cacheA = paged_cache(mgr, mgr.new_sequence(), N_LAYERS, quantized=True, group_size=GROUP)
    for t in range(4):
        _write_tok(mgr, cacheA, 2000 + t, _latent(t))
    ikv0 = mx.random.normal((1, 3, 8), key=mx.random.key(77))      # seed per-stream derived state on the
    ring0 = mx.random.normal((1, 4, HEAD_DIM), key=mx.random.key(78))  # source, as the model forward would
    cacheA.layers[0].ikv = ikv0
    cacheA.layers[0].ring = ring0

    reps = cacheA.replicate(2)
    shared = all(r.layers[0].ikv is ikv0 and r.layers[0].ring is ring0 for r in reps)  # shared by reference
    # diverge losslessly: grow replica 0's ring; the source + replica 1 stay the original object.
    reps[0].layers[0].ring = mx.concatenate([reps[0].layers[0].ring, mx.zeros((1, 1, HEAD_DIM))], axis=1)
    indep = (cacheA.layers[0].ring is ring0 and reps[1].layers[0].ring is ring0
             and reps[0].layers[0].ring.shape[1] == ring0.shape[1] + 1)

    assert shared and indep, f"shared={shared} indep={indep}"
    print("B per-stream derived state (ikv/ring) shared by ref into replicas, diverges losslessly  ok")


def test_copy_fails_loud() -> None:
    mgr = _mgr()
    cacheA = paged_cache(mgr, mgr.new_sequence(), N_LAYERS, quantized=True, group_size=GROUP)
    _write_tok(mgr, cacheA, 3000, _latent(0))
    raised = False
    try:
        cacheA._copy()
    except NotImplementedError as e:
        raised = "replicate" in str(e).lower() or "fork" in str(e).lower()
    assert raised, "PagedDSV4Cache._copy must fail loud (replicate via sequence fork is the paged primitive)"
    print("C PagedDSV4Cache._copy fails loud (replicate via sequence fork is the paged primitive)  ok")


def run() -> None:
    test_replicate_cow_isolation()
    test_derived_state_shared_then_diverges()
    test_copy_fails_loud()
    print("PASS — PagedDSV4Cache.replicate(B) satisfies the tree-spec verify cache contract over paged KV "
          "(COW-isolated latent, shared derived state, fail-loud _copy)")


if __name__ == "__main__":
    run()
