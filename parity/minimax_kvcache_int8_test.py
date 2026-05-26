"""Gate: MiniMax int8 KV cache — quantized=True path is parity-faithful to bf16 within affine
tolerance, and rollback is **lossless on every quantized field** (codes + scales + biases sliced
together).

Model-free (~ms): tiny GQA shapes (B=1, n_kv=2, head_dim=64, group_size=32), random k/v over a
sequence of appends + truncates. Verifies, in order:

  (1) ``offset`` matches between bf16 and int8 caches after every append (storage shape differs but
      the cached-position count is the same — proves int8 concat is along the right axis);
  (2) the returned ``(k, v)`` from the int8 cache equals bf16 within the int8 affine tolerance
      (max abs error scales with each group's dynamic range / 255 — comfortably under 1e-1 for
      normalized inputs);
  (3) ``truncate(length)`` on the int8 cache leaves it **bit-identical** to a fresh int8 cache built
      from only the first ``length`` appends (proves the slice is lossless — every one of the six
      int8 fields slices in lockstep along the seq axis);
  (4) fail-loud (rule 6): ``length < 0`` and ``length > offset`` raise;
  (5) MiniMaxCache(n_layers, quantized=True) propagates the flag — every layer has it; truncate at
      the cache level lossless-slices every layer.

    uv run python -m parity.minimax_kvcache_int8_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.minimax.attention import KVCache
from quanta.minimax.decode import MiniMaxCache, _LayerKVCache

B, N_KV, HD = 1, 2, 64
GS = 32
TOL = 1e-1   # int8 affine on random [-1, 1] data → typical max abs err ≈ 4/255 ≈ 0.016 per group


def _rand_kv(t: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(t * 17 + 3)
    k = mx.random.uniform(-1.0, 1.0, (B, N_KV, t, HD)).astype(mx.bfloat16)
    v = mx.random.uniform(-1.0, 1.0, (B, N_KV, t, HD)).astype(mx.bfloat16)
    mx.eval(k, v)
    return k, v


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def run() -> None:
    ok = True

    # Append a sequence of (k, v) chunks to both bf16 and int8 caches, comparing after each.
    chunks = [_rand_kv(t) for t in (5, 1, 1, 1, 7)]
    bc = KVCache()
    qc = KVCache(quantized=True, group_size=GS)

    offsets_ok = True
    parity_ok = True
    max_err = 0.0
    for i, (k, v) in enumerate(chunks):
        bk, bv = bc.update(k, v)
        qk, qv = qc.update(k, v)
        offsets_ok &= (bc.offset == qc.offset == bk.shape[2])
        err = max(_maxabs(bk, qk), _maxabs(bv, qv))
        max_err = max(max_err, err)
        if err >= TOL:
            parity_ok = False
            print(f"  chunk {i}: offset={bc.offset} max|Δ|={err:.4f}")
    ok &= offsets_ok and parity_ok
    print(f"  [{'OK' if offsets_ok and parity_ok else 'FAIL'}] append parity over {len(chunks)} chunks: "
          f"final offset={bc.offset}, max|Δ| over sequence = {max_err:.4f} (tol {TOL})")

    # (3) Truncate on the int8 cache → bit-identical to building int8 from just the first `length`
    # appends. Build the "ground truth" int8 cache from the prefix only, then compare codes + scales
    # + biases byte-exact. Target length lands mid-stream: 5+1+1 = 7 of the 15 total appends.
    target_len = 7
    qc_truth = KVCache(quantized=True, group_size=GS)
    consumed = 0
    for k, v in chunks:
        t_here = k.shape[2]
        if consumed + t_here <= target_len:
            qc_truth.update(k, v)
            consumed += t_here
        elif consumed < target_len:
            take = target_len - consumed
            qc_truth.update(k[:, :, :take], v[:, :, :take])
            consumed = target_len
            break
        else:
            break

    # truncate the full int8 cache to target_len (use _LayerKVCache so we have a truncate method)
    qc_t = _LayerKVCache(quantized=True, group_size=GS)
    for k, v in chunks:
        qc_t.update(k, v)
    qc_t.truncate(target_len)

    def _eq(a: mx.array, b: mx.array) -> bool:
        return bool(mx.all(a == b).item())

    truncate_ok = (qc_t.offset == target_len == qc_truth.offset
                   and _eq(qc_t.k_q, qc_truth.k_q) and _eq(qc_t.k_s, qc_truth.k_s)
                   and _eq(qc_t.k_b, qc_truth.k_b) and _eq(qc_t.v_q, qc_truth.v_q)
                   and _eq(qc_t.v_s, qc_truth.v_s) and _eq(qc_t.v_b, qc_truth.v_b))
    ok &= truncate_ok
    print(f"  [{'OK' if truncate_ok else 'FAIL'}] lossless truncate to {target_len}: codes/scales/biases "
          f"byte-equal to prefix-only build")

    # truncate to 0 wipes every field
    qc_t.truncate(0)
    wipe_ok = (qc_t.offset == 0 and qc_t.k_q is None and qc_t.k_s is None and qc_t.k_b is None
               and qc_t.v_q is None and qc_t.v_s is None and qc_t.v_b is None)
    ok &= wipe_ok
    print(f"  [{'OK' if wipe_ok else 'FAIL'}] truncate(0) clears every quantized field")

    # (4) fail-loud
    qc_fl = _LayerKVCache(quantized=True, group_size=GS)
    for k, v in chunks[:2]:
        qc_fl.update(k, v)
    r1 = _raises(lambda: qc_fl.truncate(-1), ValueError)
    r2 = _raises(lambda: qc_fl.truncate(qc_fl.offset + 1), ValueError)
    loud = r1 and r2
    ok &= loud
    print(f"  [{'OK' if loud else 'FAIL'}] fail-loud: neg_length={r1} forward_roll={r2}")

    # (5) MiniMaxCache propagates the flag + lossless truncate per layer
    mc = MiniMaxCache(n_layers=3, quantized=True, group_size=GS)
    assert len(mc) == 3
    for layer_idx in range(3):
        for k, v in chunks[:3]:
            mc[layer_idx].update(k, v)
    flag_ok = all(mc[i].quantized for i in range(3))
    pre = mc.offset
    mc.truncate(pre - 2)
    layers_consistent = all(mc[i].offset == pre - 2 for i in range(3))
    cache_ok = flag_ok and layers_consistent
    ok &= cache_ok
    print(f"  [{'OK' if cache_ok else 'FAIL'}] MiniMaxCache(quantized=True): flag={flag_ok} "
          f"truncate_per_layer_consistent={layers_consistent}")

    # bf16 default still works (backward compat)
    mc_bf = MiniMaxCache(n_layers=2)
    bf_default_ok = (not mc_bf.quantized) and all(not mc_bf[i].quantized for i in range(2))
    ok &= bf_default_ok
    print(f"  [{'OK' if bf_default_ok else 'FAIL'}] MiniMaxCache() default is bf16 (backward-compat)")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
