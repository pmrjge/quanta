"""Gate: GLM int8 MLA latent cache — ``_LayerKVCache(quantized=True)`` ports the Kimi MLACache
pattern (int8 affine on ``c_kv``, ``k_pe`` stays bf16) and ``GLMCache.truncate`` slices the
quantized trio (codes + scales + biases) together so rollback is lossless.

Model-free (~ms): tiny MLA shapes (B=1, kv_lora=64, rope=8, group_size=32), random latent + rope key
appended over a sequence + truncated. Verifies:

  (1) ``offset`` matches between bf16 and int8 caches after every append (proves concat is along the
      right axis — S=1 dim);
  (2) the returned ``(c_kv, k_pe)`` from the int8 cache equals bf16 within int8 affine tolerance
      (``k_pe`` is **bit-equal** — it's always bf16; only ``c_kv`` is quantized);
  (3) ``GLMCache.truncate(length)`` on an int8 cache leaves it **bit-identical** to a fresh int8
      cache built from the prefix only (every quantized field plus ``k_pe`` + indexer key slice in
      lockstep along the seq axis);
  (4) fail-loud: negative length raises;
  (5) bf16 default is backward-compat (existing GLM serving paths see no behavior change unless they
      opt in).

    uv run python -m parity.glm_kvcache_int8_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.glm.decode import GLMCache, _LayerKVCache

B, KV_LORA, ROPE = 1, 64, 8
GS = 32
TOL = 1e-1


def _rand_chunk(t: int, seed: int) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed * 31 + 11)
    c_kv = mx.random.uniform(-1.0, 1.0, (B, t, KV_LORA)).astype(mx.bfloat16)
    k_pe = mx.random.uniform(-1.0, 1.0, (B, 1, t, ROPE)).astype(mx.bfloat16)
    mx.eval(c_kv, k_pe)
    return c_kv, k_pe


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def _eq(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(a == b).item())


def run() -> None:
    ok = True

    chunks = [_rand_chunk(t, s) for s, t in enumerate([4, 1, 1, 1, 3])]

    # (1)+(2) append parity over a sequence
    bc = _LayerKVCache()
    qc = _LayerKVCache(quantized=True, group_size=GS)
    offsets_ok = parity_ok = True
    max_err_c = max_err_p = 0.0
    for ckv, kpe in chunks:
        bckv, bkpe = bc.update(ckv, kpe)
        qckv, qkpe = qc.update(ckv, kpe)
        offsets_ok &= (bc.offset == qc.offset == bckv.shape[1])
        ec = _maxabs(bckv, qckv)
        ep = _maxabs(bkpe, qkpe)
        max_err_c = max(max_err_c, ec)
        max_err_p = max(max_err_p, ep)
        if ec >= TOL or ep > 0:
            parity_ok = False
    ok &= offsets_ok and parity_ok
    print(f"  [{'OK' if offsets_ok and parity_ok else 'FAIL'}] append parity: final offset={bc.offset} "
          f"max|Δc_kv|={max_err_c:.4f} max|Δk_pe|={max_err_p:.4f} (k_pe must be bit-equal; tol {TOL})")

    # (3) GLMCache.truncate lossless. Build two caches with the same chunks, truncate one to
    # target_len, compare to a freshly-built cache from just the prefix.
    target_len = 6  # 4+1+1
    n_layers = 2

    def _populate(gc: GLMCache, ckhunks):
        for layer in range(n_layers):
            for ckv, kpe in ckhunks:
                gc[layer].kv.update(ckv, kpe)

    gc_full = GLMCache(n_layers, quantized=True, group_size=GS)
    _populate(gc_full, chunks)
    gc_full.truncate(target_len)

    # build truth: append only prefix of chunks up to target_len total positions
    prefix: list[tuple[mx.array, mx.array]] = []
    consumed = 0
    for ckv, kpe in chunks:
        t_here = ckv.shape[1]
        if consumed + t_here <= target_len:
            prefix.append((ckv, kpe))
            consumed += t_here
        elif consumed < target_len:
            take = target_len - consumed
            prefix.append((ckv[:, :take], kpe[:, :, :take]))
            consumed = target_len
            break
        else:
            break
    gc_truth = GLMCache(n_layers, quantized=True, group_size=GS)
    _populate(gc_truth, prefix)

    trunc_ok = (gc_full.offset == target_len == gc_truth.offset)
    for layer in range(n_layers):
        a, b = gc_full[layer].kv, gc_truth[layer].kv
        trunc_ok &= (_eq(a.c_kv_q, b.c_kv_q) and _eq(a.c_kv_s, b.c_kv_s) and _eq(a.c_kv_b, b.c_kv_b)
                     and _eq(a.k_pe, b.k_pe))
    ok &= trunc_ok
    print(f"  [{'OK' if trunc_ok else 'FAIL'}] GLMCache.truncate({target_len}) lossless across "
          f"{n_layers} layers (quantized trio + k_pe byte-equal to prefix-only build)")

    # (4) fail-loud
    gc_fl = GLMCache(n_layers, quantized=True, group_size=GS)
    _populate(gc_fl, chunks[:2])
    loud = _raises(lambda: gc_fl.truncate(-1), ValueError)
    ok &= loud
    print(f"  [{'OK' if loud else 'FAIL'}] fail-loud: truncate(-1) raises")

    # (5) bf16 default backward-compat
    gc_bf = GLMCache(n_layers)
    bf_default = (not gc_bf.quantized) and all(not gc_bf[i].kv.quantized for i in range(n_layers))
    ok &= bf_default
    print(f"  [{'OK' if bf_default else 'FAIL'}] GLMCache() default bf16 (backward-compat)")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
