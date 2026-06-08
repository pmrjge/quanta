"""Paged KV core parity (model-free, tiny synthetic tensors) — #152 step 1, gate (a)+(c).

The paged cache must be a behavior-exact substitute for the discrete per-stream cache before it can
ever back serving. Two checks here:

(a) **Bit-identical to discrete.** Driving the same per-token k/v stream through
    :class:`quanta.nemotron.attention.KVCache` (the discrete int8/bf16 reference) and through
    :class:`quanta.paged.PagedKVCacheManager` must yield the **same** ``(k_full, v_full)`` — max_abs
    == 0, not just "close". This holds because int8 packing is on ``head_dim`` (last axis) while blocks
    cut the seq axis, so a block always stores whole per-token quant records (no group split). Checked
    for ``quantized=True`` and ``quantized=False``, over 2 layers (per-layer independence).

(c) **Copy-on-write isolation.** Fork a sequence whose tail block is partial+shared, append a *different*
    token to each branch: each branch's stream must equal a discrete cache fed that branch's own tokens,
    and the cache must record >=1 COW clone. Proves divergent shared-prefix branches don't corrupt each
    other.

Model-free: a few KB of random tensors, fixed seed, runs in ms on CPU — SAFE alongside a live GPU job.

    uv run python -m parity.paged_cache_test

deferred (run later on GPU): teacher-forced ppl with the paged path ON vs the discrete path on a real
artifact must be bit-identical (the rule-4 ship gate); written here, not run during the model-free build.
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
    """A deterministic per-token k/v stream: k,v are [1, N_KV, n, HEAD_DIM] bf16."""
    mx.random.seed(seed)
    k = mx.random.normal((1, N_KV, n, HEAD_DIM)).astype(mx.bfloat16)
    v = mx.random.normal((1, N_KV, n, HEAD_DIM)).astype(mx.bfloat16)
    return k, v


def _discrete_final(k: mx.array, v: mx.array, *, quantized: bool) -> tuple[mx.array, mx.array]:
    cache = KVCache(quantized=quantized, group_size=GROUP_SIZE)
    kf = vf = None
    for t in range(k.shape[2]):
        kf, vf = cache.update(k[:, :, t:t + 1], v[:, :, t:t + 1])
        mx.eval(kf, vf)
    return kf, vf


def _paged_manager(*, quantized: bool) -> PagedKVCacheManager:
    return PagedKVCacheManager(num_layers=NUM_LAYERS, block_size=BLOCK_SIZE, max_blocks=64,
                               group_size=GROUP_SIZE, quantized=quantized, model_name="paged-test")


def _paged_final(mgr: PagedKVCacheManager, seq, k: mx.array, v: mx.array, *, layer: int) -> tuple[mx.array, mx.array]:
    """Drive one decode-style step per token: advance (one id) -> per-layer write -> commit -> gather."""
    kf = vf = None
    for t in range(k.shape[2]):
        mgr.advance(seq, [1000 + t])
        for L in range(NUM_LAYERS):
            mgr.view(seq, L).update(k[:, :, t:t + 1], v[:, :, t:t + 1])
        mgr.commit(seq)
        kf, vf = mgr.gather(seq, layer)
        mx.eval(kf, vf)
    return kf, vf


def _max_abs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _check_identical(quantized: bool) -> bool:
    T = 10  # 2 full blocks (block_size 4) + a partial tail of 2
    k, v = _stream(seed=0, n=T)
    dk, dv = _discrete_final(k, v, quantized=quantized)
    mgr = _paged_manager(quantized=quantized)
    seq = mgr.new_sequence()
    pk, pv = _paged_final(mgr, seq, k, v, layer=0)

    shape_ok = (pk.shape == dk.shape == (1, N_KV, T, HEAD_DIM))
    off_ok = (mgr.view(seq, 0).offset == T)
    dk_max = _max_abs(dk, pk)
    dv_max = _max_abs(dv, pv)
    ok = shape_ok and off_ok and dk_max == 0.0 and dv_max == 0.0
    tag = "int8" if quantized else "bf16"
    print(f"(a) {tag}: shape_ok={shape_ok} offset_ok={off_ok} "
          f"k_max_abs={dk_max:.3e} v_max_abs={dv_max:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def _check_cow() -> bool:
    # seqA: 6 tokens => 1 full block + a partial tail of 2 (block_size 4). Fork shares the partial tail.
    base_n = 6
    k, v = _stream(seed=1, n=base_n)
    xk, xv = _stream(seed=2, n=2)  # two distinct divergent tokens (one per branch)

    mgr = _paged_manager(quantized=True)
    seqA = mgr.new_sequence()
    _paged_final(mgr, seqA, k, v, layer=0)

    seqB = mgr.fork(seqA)
    cow_before = mgr.get_stats().cow_copies

    # A appends divergent token 0; B appends divergent token 1 — both into the shared partial tail block.
    mgr.advance(seqA, [9000])
    for L in range(NUM_LAYERS):
        mgr.view(seqA, L).update(xk[:, :, 0:1], xv[:, :, 0:1])
    mgr.commit(seqA)
    mgr.advance(seqB, [9001])
    for L in range(NUM_LAYERS):
        mgr.view(seqB, L).update(xk[:, :, 1:2], xv[:, :, 1:2])
    mgr.commit(seqB)

    ak, av = mgr.gather(seqA, 0)
    bk, bv = mgr.gather(seqB, 0)
    mx.eval(ak, av, bk, bv)

    # references: A == discrete(base + xtoken0); B == discrete(base + xtoken1)
    refA_k = mx.concatenate([k, xk[:, :, 0:1]], axis=2)
    refA_v = mx.concatenate([v, xv[:, :, 0:1]], axis=2)
    refB_k = mx.concatenate([k, xk[:, :, 1:2]], axis=2)
    refB_v = mx.concatenate([v, xv[:, :, 1:2]], axis=2)
    dAk, dAv = _discrete_final(refA_k, refA_v, quantized=True)
    dBk, dBv = _discrete_final(refB_k, refB_v, quantized=True)

    cow = mgr.get_stats().cow_copies - cow_before
    a_ok = _max_abs(dAk, ak) == 0.0 and _max_abs(dAv, av) == 0.0
    b_ok = _max_abs(dBk, bk) == 0.0 and _max_abs(dBv, bv) == 0.0
    indep = _max_abs(ak, bk) > 0.0  # the branches actually diverged
    ok = a_ok and b_ok and indep and cow >= 1
    print(f"(c) cow: branchA_exact={a_ok} branchB_exact={b_ok} diverged={indep} "
          f"cow_copies={cow} -> {'PASS' if ok else 'FAIL'}")
    return ok


def _check_replicate() -> bool:
    # B-way sequence-level replicate (the paged analog of DSV4Cache.replicate(B)). seqA: 6 tokens =>
    # 1 full block + a partial tail of 2 (block_size 4). replicate(B) forks B COW siblings sharing the
    # partial tail; each writes a DISTINCT divergent token into it, so each must COW-clone the tail.
    base_n, B = 6, 3
    k, v = _stream(seed=1, n=base_n)
    xk, xv = _stream(seed=3, n=B)  # B distinct divergent tokens (one per branch)

    mgr = _paged_manager(quantized=True)
    seqA = mgr.new_sequence()
    _paged_final(mgr, seqA, k, v, layer=0)

    cow_before = mgr.get_stats().cow_copies
    branches = mgr.replicate(seqA, B)
    branches_ok = (len(branches) == B and all(br.seq_id != seqA.seq_id for br in branches)
                   and len({br.seq_id for br in branches}) == B)

    for j, br in enumerate(branches):  # each branch appends its own divergent token into the shared tail
        mgr.advance(br, [9100 + j])
        for L in range(NUM_LAYERS):
            mgr.view(br, L).update(xk[:, :, j:j + 1], xv[:, :, j:j + 1])
        mgr.commit(br)

    cow = mgr.get_stats().cow_copies - cow_before
    gathered = []
    each_exact = True
    for j, br in enumerate(branches):  # branch j == discrete(base + its own divergent token)
        bk, bv = mgr.gather(br, 0)
        mx.eval(bk, bv)
        gathered.append(bk)
        ref_k = mx.concatenate([k, xk[:, :, j:j + 1]], axis=2)
        ref_v = mx.concatenate([v, xv[:, :, j:j + 1]], axis=2)
        dref_k, dref_v = _discrete_final(ref_k, ref_v, quantized=True)
        if _max_abs(dref_k, bk) != 0.0 or _max_abs(dref_v, bv) != 0.0:
            each_exact = False
    diverged = all(_max_abs(gathered[i], gathered[i + 1]) > 0.0 for i in range(B - 1))

    pk, _pv = mgr.gather(seqA, 0)  # parent prefix still readable + unchanged length (read-only)
    mx.eval(pk)
    parent_ok = (pk.shape[2] == base_n)

    ok = branches_ok and each_exact and diverged and parent_ok and cow >= B
    print(f"(d) replicate(B={B}): branches_ok={branches_ok} each_exact={each_exact} diverged={diverged} "
          f"parent_ok={parent_ok} cow_copies={cow} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run() -> None:
    ok = True
    ok = _check_identical(quantized=True) and ok
    ok = _check_identical(quantized=False) and ok
    ok = _check_cow() and ok
    ok = _check_replicate() and ok
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
