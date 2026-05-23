"""GPTQ for routed experts — error-feedback affine quantization with a Woodbury inverse.

GPTQ minimizes the layer-wise ``‖WX − ŴX‖²`` (X = calibration activations ``[n, in]``),
whose curvature is the exact Hessian ``H = XᵀX``. We quantize input-columns left-to-right;
after each column we push its rounding error onto the not-yet-quantized columns weighted by
``H⁻¹`` (Optimal Brain Surgeon), so the activation-weighted error is far below round-to-nearest.

Inverse via **Woodbury**, not Cholesky-of-H: under top-8 routing most experts see ``n ≪ in``
rows, so ``H = δI + XᵀX`` is diagonal-plus-low-rank and
``H⁻¹ = (1/δ)I − (1/δ²)Xᵀ(I + XXᵀ/δ)⁻¹X`` inverts only the small ``[n,n]`` Gram (``O(n³)``)
instead of the ``[in,in]`` Hessian. The ordered OBS update then reads its coefficients from the
upper-Cholesky ``R`` of ``H⁻¹`` (``RᵀR = H⁻¹``). The only sequential work is the bounded
per-block column loop (CLAUDE.md-sanctioned); between blocks one batched GEMM propagates the
accumulated error to all trailing columns.

Codes are MLX-affine (``ŵ = q·scale + bias``, ``q ∈ [0, 2^bits−1]``, group-128 along ``in``),
so the GPTQ result packs into the same layout ``mx.quantized_matmul`` / ``mx.gather_qmm`` consume.
"""

from __future__ import annotations

import mlx.core as mx


def woodbury_inverse(x: mx.array, delta: float) -> mx.array:
    """``(δI + XᵀX)⁻¹`` via Woodbury — inverts the ``[n,n]`` Gram, not ``[in,in]``. ``x`` is ``[n,in]``."""
    n, in_ = x.shape
    xf = x.astype(mx.float32)
    gram = xf @ xf.T  # [n,n]
    m = mx.eye(n) + gram / delta
    with mx.stream(mx.cpu):  # MLX has no GPU Cholesky
        m_inv = mx.linalg.cholesky_inv(mx.linalg.cholesky(m))
    mx.eval(m_inv)
    h_inv = -(xf.T @ (m_inv @ xf)) / (delta * delta)  # [in,in], low-rank part
    return h_inv + (1.0 / delta) * mx.eye(in_)


def _group_params(wg: mx.array, maxq: int) -> tuple[mx.array, mx.array]:
    """Per-row affine scale/bias for a group of columns ``wg`` ``[out, g]`` → ``([out], [out])``."""
    lo = mx.min(wg, axis=1)
    hi = mx.max(wg, axis=1)
    scale = (hi - lo) / maxq
    scale = mx.where(scale <= 0, mx.array(1.0), scale)  # degenerate group → unit scale
    return scale, lo


def gptq_quantize(
    w: mx.array, x: mx.array, bits: int, *, group_size: int = 128, damp: float = 0.01
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """GPTQ-quantize ``w`` ``[out, in]`` on calibration ``x`` ``[n, in]``.

    Returns ``(w_hat, codes, scales, biases)``: dequantized weight (for the loss/QC gauge), the
    integer codes ``[out, in]``, and per-group ``scales``/``biases`` ``[out, n_groups]``.
    Block size is tied to ``group_size`` so each block is exactly one quant group.
    """
    out, in_ = w.shape
    maxq = (1 << bits) - 1
    wf = w.astype(mx.float32)
    xf = x.astype(mx.float32)

    delta = max(damp * (mx.sum(xf * xf) / in_).item(), 1e-8)  # δ = damp·mean(diag H)
    h_inv = woodbury_inverse(xf, delta)
    with mx.stream(mx.cpu):  # MLX has no GPU Cholesky
        r = mx.linalg.cholesky(h_inv).T  # upper R, RᵀR = H⁻¹
    mx.eval(r)

    work = mx.array(wf)  # working copy, mutated by error feedback
    n_groups = (in_ + group_size - 1) // group_size
    scales = mx.zeros((out, n_groups))
    biases = mx.zeros((out, n_groups))
    codes = mx.zeros((out, in_), dtype=mx.int32)
    w_hat = mx.zeros((out, in_))

    for i0 in range(0, in_, group_size):  # one block == one group
        i1 = min(i0 + group_size, in_)
        g = i0 // group_size
        sc, bi = _group_params(work[:, i0:i1], maxq)
        scales[:, g] = sc
        biases[:, g] = bi
        err = mx.zeros((out, i1 - i0))
        for j in range(i1 - i0):  # bounded (<=group_size) sequential column loop
            col = i0 + j
            wc = work[:, col]
            q = mx.clip(mx.round((wc - bi) / sc), 0, maxq)
            wq = q * sc + bi
            e = (wc - wq) / r[col, col]
            if col + 1 < i1:  # propagate within the block
                work[:, col + 1 : i1] = work[:, col + 1 : i1] - e[:, None] * r[col, col + 1 : i1][None, :]
            err[:, j] = e
            codes[:, col] = q.astype(mx.int32)
            w_hat[:, col] = wq
        if i1 < in_:  # one batched GEMM: push the block's error onto all trailing columns
            work[:, i1:] = work[:, i1:] - err @ r[i0:i1, i1:]
        mx.eval(work, codes, scales, biases, w_hat)  # bound graph growth per block
    return w_hat, codes, scales, biases


def gptq_quantize_batch(
    ws: mx.array, xs: list[mx.array], bits: int, *, group_size: int = 128, damp: float = 0.01
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched GPTQ over a chunk of ``E`` experts (same ``bits``) → ``(codes, scales, biases)``.

    Same algorithm as :func:`gptq_quantize`, but the expensive ordered column loop runs **once**
    for all ``E`` experts with batched ``[E,…]`` ops (and one batched trailing GEMM per block),
    so the Python/launch overhead is shared — the cross-expert speedup that makes the full bake
    feasible. ``ws`` is ``[E, out, in]``; ``xs`` is a per-expert list of ``[n_e, in]`` (the
    Hessian inverse is still per-expert, but ``E`` is small and the column loop is the cost).
    """
    e, out, in_ = ws.shape
    maxq = (1 << bits) - 1
    rs = []
    for i in range(e):  # per-expert inverse (small E); the column loop below is the shared cost
        xf = xs[i].astype(mx.float32)
        delta = max(damp * (mx.sum(xf * xf) / in_).item(), 1e-8)
        with mx.stream(mx.cpu):
            rs.append(mx.linalg.cholesky(woodbury_inverse(xf, delta)).T)
    r = mx.stack(rs)  # [E, in, in], upper RᵀR = H⁻¹ per expert
    mx.eval(r)

    work = ws.astype(mx.float32)
    n_groups = (in_ + group_size - 1) // group_size
    scales = mx.zeros((e, out, n_groups))
    biases = mx.zeros((e, out, n_groups))
    codes = mx.zeros((e, out, in_), dtype=mx.int32)

    for i0 in range(0, in_, group_size):
        i1 = min(i0 + group_size, in_)
        g = i0 // group_size
        lo = mx.min(work[:, :, i0:i1], axis=2)
        hi = mx.max(work[:, :, i0:i1], axis=2)
        sc = mx.where(hi - lo <= 0, mx.array(1.0), (hi - lo) / maxq)  # [E,out]
        scales[:, :, g] = sc
        biases[:, :, g] = lo
        err = mx.zeros((e, out, i1 - i0))
        for j in range(i1 - i0):  # bounded shared column loop (batched across E)
            col = i0 + j
            wc = work[:, :, col]
            q = mx.clip(mx.round((wc - lo) / sc), 0, maxq)
            wq = q * sc + lo
            de = (wc - wq) / r[:, col, col][:, None]  # [E,out]
            if col + 1 < i1:
                work[:, :, col + 1 : i1] = work[:, :, col + 1 : i1] - de[:, :, None] * r[:, col, col + 1 : i1][:, None, :]
            err[:, :, j] = de
            codes[:, :, col] = q.astype(mx.int32)
        if i1 < in_:
            work[:, :, i1:] = work[:, :, i1:] - err @ r[:, i0:i1, i1:]  # [E,out,blk]@[E,blk,rest]
        mx.eval(work, codes, scales, biases)
    return codes, scales, biases
