"""Gate: Qwen3.5 int8 KV cache — ``KVCache(quantized=True)`` on the 15 full-attention layers (#114
1M-context target). Same pattern as MiniMax (GQA ``[B, n_kv, S, head_dim]``), wired into
``Qwen35Cache`` so the hybrid (KV on full layers, recurrent state on linear) still rolls back
losslessly per layer.

Model-free (~ms): tiny GQA shapes (B=1, n_kv=2, head_dim=64, group_size=32), random k/v over a
sequence of appends + truncates. Verifies:

  (1) ``offset`` matches between bf16 and int8 caches after every append;
  (2) the returned ``(k, v)`` from the int8 cache equals bf16 within int8 affine tolerance;
  (3) ``_kv_truncate`` on int8 leaves the cache **bit-identical** to a fresh int8 cache built from
      the prefix only (every quantized field slices in lockstep along the seq axis);
  (4) ``Qwen35Cache(quantized=True)`` applies int8 only to full-attn layers — linear-attn layers
      stay as plain recurrent state (no benefit from int8 on an O(1) state);
  (5) bf16 default backward-compat.

    uv run python -m parity.qwen35_kvcache_int8_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import KVCache
from quanta.qwen35.decode import Qwen35Cache, _GDNLayerState, _kv_truncate

B, N_KV, HD = 1, 2, 64
GS = 32
TOL = 1e-1


def _rand_kv(t: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed * 19 + 5)
    k = mx.random.uniform(-1.0, 1.0, (B, N_KV, t, HD)).astype(mx.bfloat16)
    v = mx.random.uniform(-1.0, 1.0, (B, N_KV, t, HD)).astype(mx.bfloat16)
    mx.eval(k, v)
    return k, v


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _eq(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(a == b).item())


def run() -> None:
    ok = True

    chunks = [_rand_kv(t, s) for s, t in enumerate([5, 1, 1, 1, 4])]
    bc = KVCache()
    qc = KVCache(quantized=True, group_size=GS)
    offsets_ok = parity_ok = True
    max_err = 0.0
    for k, v in chunks:
        bk, bv = bc.update(k, v)
        qk, qv = qc.update(k, v)
        offsets_ok &= (bc.offset == qc.offset == bk.shape[2])
        e = max(_maxabs(bk, qk), _maxabs(bv, qv))
        max_err = max(max_err, e)
        if e >= TOL:
            parity_ok = False
    ok &= offsets_ok and parity_ok
    print(f"  [{'OK' if offsets_ok and parity_ok else 'FAIL'}] append parity over {len(chunks)} "
          f"chunks: final offset={bc.offset} max|Δ|={max_err:.4f} (tol {TOL})")

    # (3) lossless int8 truncate
    target_len = 7  # 5+1+1
    qc_full = KVCache(quantized=True, group_size=GS)
    for k, v in chunks:
        qc_full.update(k, v)
    _kv_truncate(qc_full, target_len)

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

    trunc_ok = (qc_full.offset == target_len == qc_truth.offset
                and _eq(qc_full.k_q, qc_truth.k_q) and _eq(qc_full.k_s, qc_truth.k_s)
                and _eq(qc_full.k_b, qc_truth.k_b) and _eq(qc_full.v_q, qc_truth.v_q)
                and _eq(qc_full.v_s, qc_truth.v_s) and _eq(qc_full.v_b, qc_truth.v_b))
    ok &= trunc_ok
    print(f"  [{'OK' if trunc_ok else 'FAIL'}] _kv_truncate({target_len}) lossless: quantized trio "
          f"byte-equal to prefix-only build")

    # truncate(<=0) clears every field
    _kv_truncate(qc_full, 0)
    wipe_ok = (qc_full.offset == 0 and qc_full.k_q is None and qc_full.k_s is None
               and qc_full.k_b is None and qc_full.v_q is None and qc_full.v_s is None
               and qc_full.v_b is None)
    ok &= wipe_ok
    print(f"  [{'OK' if wipe_ok else 'FAIL'}] _kv_truncate(0) clears every quantized field")

    # (4) Qwen35Cache(quantized=True) on a mixed hybrid: int8 applies only to full-attn layers
    # alternate linear and full attention (3 of each = 6 layers)
    def is_lin(i: int) -> bool:
        return i % 2 == 0  # even = linear, odd = full

    qc_hybrid = Qwen35Cache(n_layers=6, layer_is_linear=is_lin, quantized=True, group_size=GS)
    hybrid_ok = True
    for i in range(6):
        lc = qc_hybrid[i]
        if is_lin(i):
            hybrid_ok &= isinstance(lc, _GDNLayerState)  # recurrent state — no quantized flag
        else:
            hybrid_ok &= (isinstance(lc, KVCache) and lc.quantized and lc.group_size == GS)
    ok &= hybrid_ok
    print(f"  [{'OK' if hybrid_ok else 'FAIL'}] Qwen35Cache(quantized=True): full-attn layers get "
          f"int8, linear-attn layers stay recurrent")

    # (5) bf16 default backward-compat
    qc_bf = Qwen35Cache(n_layers=4, layer_is_linear=lambda i: False)
    bf_ok = (not qc_bf.quantized) and all(not qc_bf[i].quantized for i in range(4))
    ok &= bf_ok
    print(f"  [{'OK' if bf_ok else 'FAIL'}] Qwen35Cache() default bf16 (backward-compat)")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
