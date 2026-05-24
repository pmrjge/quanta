"""Mamba-2 SSD (state-space duality) core for Nemotron-H — MLX, parity-gated.

Mamba-2's simplification is that the state transition ``A`` is a **scalar per head**, which
lets the whole SSM be recast as matmuls. Per head, state ``S ∈ (N, P)`` (N=ssm_state,
P=head_dim); ``B``/``C`` are shared within each of ``G`` groups (``H/G`` heads per group):

    discretize:  a_t = dt_t * A          (dt_t > 0, A < 0  ->  decay in (0,1])
    recurrence:  S_t = exp(a_t) * S_{t-1} + dt_t * (B_t ⊗ x_t)
    output:      y_t = C_t · S_t + D * x_t

Three entry points, all numerically equivalent (gated in parity/mamba_ssd_test.py):

* :func:`ssd_sequential` — the dead-simple O(L) reference (python loop over time). The oracle.
* :func:`ssd_chunked`    — the SSD prefill: segment-sum decay -> diagonal block + per-chunk
  states + a bounded scan over chunks + off-diagonal. All batched matmuls except the
  ``nc``-length chunk scan (bounded; state carries across token-blocks for long context).
* :func:`ssd_step`       — the O(1)-state decode step (one token, vectorized over heads).

Plus causal depthwise conv1d (prefill + a rolling decode state). These are the SSM/conv
primitives; the surrounding mixer (in_proj split, silu, gated RMSNorm, out_proj) wraps them.

No fused Metal kernel is needed for prefill (it's GEMM-bound by construction); a
``mx.fast.metal_kernel`` fusion of the decode step is a later, measured-only optimization.
"""

from __future__ import annotations

import mlx.core as mx


def ssd_sequential(x, dt, A, B, C, D, state_in=None):
    """Naive O(L) recurrence — the parity oracle. Returns (y, state_out).

    x: (Bn,L,H,P)  dt: (Bn,L,H)  A: (H,)  B,C: (Bn,L,G,N)  D: (H,)  state_in: (Bn,H,N,P)|None
    """
    bn, length, h, p = x.shape
    g, n = B.shape[-2], B.shape[-1]
    rep = h // g
    bf = mx.repeat(B, rep, axis=2)  # group -> head: (Bn,L,H,N)
    cf = mx.repeat(C, rep, axis=2)
    s = mx.zeros((bn, h, n, p), dtype=x.dtype) if state_in is None else state_in
    ys = []
    for t in range(length):  # reference only: explicit time loop (never the hot path)
        da = mx.exp(dt[:, t, :] * A)  # (Bn,H)
        upd = dt[:, t, :, None, None] * (bf[:, t, :, :, None] * x[:, t, :, None, :])  # (Bn,H,N,P)
        s = da[:, :, None, None] * s + upd
        y = mx.sum(cf[:, t, :, :, None] * s, axis=2) + D[None, :, None] * x[:, t, :, :]  # (Bn,H,P)
        ys.append(y)
    return mx.stack(ys, axis=1), s


def ssd_step(x_t, dt_t, A, B_t, C_t, D, state):
    """One decode step (O(1) state), vectorized over heads. Returns (y_t, state).

    x_t: (Bn,H,P)  dt_t: (Bn,H)  B_t,C_t: (Bn,G,N)  state: (Bn,H,N,P)

    Scan in fp32 (bf16 is unstable for the recurrence); inputs upcast, output cast back.
    """
    out_dtype = x_t.dtype
    x_t, dt_t, A = x_t.astype(mx.float32), dt_t.astype(mx.float32), A.astype(mx.float32)
    B_t, C_t, D, state = (B_t.astype(mx.float32), C_t.astype(mx.float32),
                          D.astype(mx.float32), state.astype(mx.float32))
    h = x_t.shape[1]
    g = B_t.shape[-2]
    rep = h // g
    bf = mx.repeat(B_t, rep, axis=1)  # (Bn,H,N)
    cf = mx.repeat(C_t, rep, axis=1)
    da = mx.exp(dt_t * A)  # (Bn,H)
    upd = dt_t[:, :, None, None] * (bf[:, :, :, None] * x_t[:, :, None, :])  # (Bn,H,N,P)
    state = da[:, :, None, None] * state + upd
    y = mx.sum(cf[:, :, :, None] * state, axis=2) + D[None, :, None] * x_t  # (Bn,H,P)
    return y.astype(out_dtype), state


_SSD_STEP_SRC = r"""
    uint pp = thread_position_in_grid.x;   // head_dim index p
    uint hh = thread_position_in_grid.y;   // head index h
    uint bb = thread_position_in_grid.z;   // batch index
    const int H = dims[0], P = dims[1], N = dims[2], rep = dims[3], BN = dims[4];
    if ((int)pp >= P || (int)hh >= H || (int)bb >= BN) return;
    const int G = H / rep;
    const int g = (int)hh / rep;
    const float dtv = dt[bb * H + hh];
    const float da = exp(dtv * A[hh]);
    const float xv = x[(bb * H + hh) * P + (int)pp];
    const int sbase = ((bb * H + hh) * N) * P + (int)pp;  // state[bb,hh,0,pp], stride P over n
    const int bc = (bb * G + g) * N;                       // B/C[bb,g,0]
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float s = da * state[sbase + n * P] + dtv * B[bc + n] * xv;
        new_state[sbase + n * P] = s;
        acc += C[bc + n] * s;
    }
    y[(bb * H + hh) * P + (int)pp] = acc + D[hh] * xv;
"""

_ssd_step_kernel = mx.fast.metal_kernel(
    name="ssd_step",
    input_names=["x", "dt", "A", "B", "C", "D", "state", "dims"],
    output_names=["y", "new_state"],
    source=_SSD_STEP_SRC,
)


def ssd_step_fused(x_t, dt_t, A, B_t, C_t, D, state):
    """Fused single-kernel decode step — output-equivalent to :func:`ssd_step`, one Metal launch.

    Same signature/semantics; one thread per ``(batch, head, head_dim)`` loops over the state
    dim ``N`` (updates ``state[.,h,:,p]`` and accumulates ``y[.,h,p]``). fp32 scan (the recurrence
    is bf16-unstable); inputs upcast, output cast back. Replaces ~8 composed ops × 40 mamba layers
    with one launch — the decode op-launch-bound fix (#41)."""
    out_dtype = x_t.dtype
    x_t, dt_t, A = x_t.astype(mx.float32), dt_t.astype(mx.float32), A.astype(mx.float32)
    B_t, C_t = B_t.astype(mx.float32), C_t.astype(mx.float32)
    D, state = D.astype(mx.float32), state.astype(mx.float32)
    bn, h, p = x_t.shape
    g, n = B_t.shape[-2], B_t.shape[-1]
    dims = mx.array([h, p, n, h // g, bn], dtype=mx.int32)
    y, new_state = _ssd_step_kernel(
        inputs=[x_t, dt_t, A, B_t, C_t, D, state, dims],
        grid=(p, h, bn),
        threadgroup=(min(p, 256), 1, 1),
        output_shapes=[(bn, h, p), (bn, h, n, p)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y.astype(out_dtype), new_state


def ssd_chunked(x, dt, A, B, C, D, chunk_size, state_in=None):
    """SSD prefill via matmuls — output-equivalent to :func:`ssd_sequential`. Returns (y, state).

    Splits the sequence into ``chunk_size`` chunks: intra-chunk via a segment-sum decay matrix
    (the attention dual), inter-chunk via a bounded scan over the carried state (the SSM dual).
    ``state_in`` carries across token-blocks so long-context prefill stays bounded-memory.

    The scan (cumsum / exp / state recurrence) runs in fp32 — bf16 is numerically unstable for
    the SSM duality (standard for Mamba); inputs are upcast and the output cast back.
    """
    out_dtype = x.dtype
    x, dt, A = x.astype(mx.float32), dt.astype(mx.float32), A.astype(mx.float32)
    B, C, D = B.astype(mx.float32), C.astype(mx.float32), D.astype(mx.float32)
    if state_in is not None:
        state_in = state_in.astype(mx.float32)
    bn, length, h, p = x.shape
    g, n = B.shape[-2], B.shape[-1]
    rep = h // g
    q = chunk_size
    if length % q != 0:
        raise ValueError(f"length {length} not divisible by chunk_size {q}")
    nc = length // q
    bf = mx.repeat(B, rep, axis=2)  # (Bn,L,H,N)
    cf = mx.repeat(C, rep, axis=2)

    def chunk(t, last):  # (Bn,L,H,last) -> (Bn,nc,H,Q,last)
        return mx.transpose(t.reshape(bn, nc, q, h, last), (0, 1, 3, 2, 4))

    xt = chunk(x, p)            # (Bn,nc,H,Q,P)
    bt = chunk(bf, n)           # (Bn,nc,H,Q,N)
    ct = chunk(cf, n)           # (Bn,nc,H,Q,N)
    dtc = mx.transpose(dt.reshape(bn, nc, q, h), (0, 1, 3, 2))  # (Bn,nc,H,Q)
    a = dtc * A.reshape(1, 1, h, 1)            # (Bn,nc,H,Q)
    acum = mx.cumsum(a, axis=-1)               # inclusive cumsum

    # intra-chunk (diagonal) block
    cb = ct @ mx.swapaxes(bt, -1, -2)          # (Bn,nc,H,Q,Q): C_i . B_j
    diff = acum[..., :, None] - acum[..., None, :]  # ā_i - ā_j
    tri = mx.tril(mx.ones((q, q), dtype=mx.bool_))   # keep i>=j (causal, intra-chunk)
    decay = mx.exp(mx.where(tri, diff, -mx.inf))     # mask BEFORE exp: upper triangle -> 0, no overflow
    m = cb * decay * dtc[..., None, :]          # dt_j on columns (mask already folded into decay)
    y_diag = m @ xt                             # (Bn,nc,H,Q,P)

    # per-chunk end state and total chunk decay
    decay_end = mx.exp(acum[..., -1:] - acum) * dtc          # (Bn,nc,H,Q)
    state_c = mx.swapaxes(bt * decay_end[..., None], -1, -2) @ xt  # (Bn,nc,H,N,P)
    da_chunk = mx.exp(acum[..., -1])            # (Bn,nc,H)

    # inter-chunk: bounded scan over chunks carrying the state (off-diagonal contribution)
    s = mx.zeros((bn, h, n, p), dtype=x.dtype) if state_in is None else state_in
    offs = []
    for c in range(nc):  # bounded: nc = block_len / chunk_size
        c_state = ct[:, c] @ s                  # (Bn,H,Q,P)
        offs.append(mx.exp(acum[:, c])[..., None] * c_state)
        s = da_chunk[:, c][..., None, None] * s + state_c[:, c]
    y_off = mx.stack(offs, axis=1)              # (Bn,nc,H,Q,P)

    y = y_diag + y_off + D.reshape(1, 1, h, 1, 1) * xt
    y = mx.transpose(y, (0, 1, 3, 2, 4)).reshape(bn, length, h, p)
    return y.astype(out_dtype), s


def causal_conv1d(u, weight, bias=None):
    """Causal depthwise conv (prefill). u: (Bn,L,C), weight: (C,K), bias: (C,)|None -> (Bn,L,C).

    Windowed sum over the bounded kernel (K≈4); ``mx.conv1d(groups=C)`` is the production swap.
    """
    length = u.shape[1]
    k = weight.shape[-1]
    up = mx.pad(u, [(0, 0), (k - 1, 0), (0, 0)])  # left-pad K-1 (causal)
    y = sum(up[:, i:i + length, :] * weight[:, i] for i in range(k))  # bounded K loop
    return y if bias is None else y + bias


def causal_conv1d_step(u_t, weight, conv_state, bias=None):
    """One decode step of the causal conv. u_t: (Bn,C), conv_state: (Bn,K-1,C) -> (y, new_state)."""
    window = mx.concatenate([conv_state, u_t[:, None, :]], axis=1)  # (Bn,K,C)
    y = mx.sum(window * mx.swapaxes(weight, 0, 1)[None], axis=1)     # (Bn,C)
    if bias is not None:
        y = y + bias
    return y, window[:, 1:, :]
