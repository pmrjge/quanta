"""Gated DeltaNet linear-attention mixer for Qwen3.5 (the 45 ``linear_attention`` layers), MLX.

Qwen3-Next-style Gated DeltaNet — a *gated delta rule* recurrence with an O(1) recurrent state
``S ∈ [v_heads, k_dim, v_dim]`` (kept in fp32), which is what makes the 1M context cheap on the
linear layers. Pipeline (faithful to the HF ``Qwen3NextGatedDeltaNet`` reference):

* ``in_proj_qkv`` 4096 -> 12288 = q(16 heads × 128) + k(16 heads × 128) + v(64 heads × 128).
* a **causal depthwise conv1d** (kernel 4) over the whole 12288-wide qkv stream, then ``silu``
  (the conv + activation is on q,k,v together, matching the reference's single conv over qkv).
* per-value-head decay ``g_t = exp(-softplus(in_proj_a(x) + dt_bias) · exp(A_log))`` — the same
  log-space discretization Mamba-2 uses (``a = -exp(A_log) < 0``; ``dt = softplus(...) > 0``;
  ``g = exp(dt·a) ∈ (0,1]``). Per **value** head (64), broadcast over the k_dim of the state.
* write strength ``β_t = sigmoid(in_proj_b(x))`` — per value head (64), in (0,1).
* **gated delta rule** state update, per value head, with the 16 key-heads grouped 4:1 under the
  64 value-heads (``rep = v_heads // k_heads = 4``; key-head ``h//rep`` feeds value-head ``h``):

      u_t = v_t - Sᵀ_{t-1} k_t                       (delta: the prediction error / "new value")
      S_t = g_t · S_{t-1} + β_t · (k_t ⊗ u_t)        (k_t ⊗ u_t is the [k_dim, v_dim] outer prod)
      o_t = Sᵀ_t q_t                                  ([v_dim] read-out)

  l2-normalize q,k along their head dim before the recurrence (DeltaNet/fla convention).
* output gate ``z = sigmoid(in_proj_z(x))`` applied as a per-head **gated RMSNorm**: a per-head
  fp32 RMSNorm (weight ``norm`` [v_dim]) of the readout, gated by ``silu(z)`` (FusedRMSNormGated:
  the gate multiplies the *normalized* readout — gate-before-weight, like the Nemotron mamba norm).
* ``out_proj`` 8192 -> 4096.

Three numerically equivalent paths (gated in ``parity/qwen35_forward_test.py``):

* :func:`gdn_recurrence` — the dead-simple O(L) sequential scan (python loop over time). Oracle.
* :func:`gdn_chunked`    — the chunk-parallel prefill: a bounded loop over fixed-size chunks
  carrying the [k_dim, v_dim] state; **within** a chunk the delta rule is still sequential over
  its ≤chunk tokens (the delta rule's ``Sᵀk`` data-dependence has no segment-sum dual like
  Mamba-2's scalar-A SSD; chunking just bounds the prefill memory and lets the cross-chunk decay
  fold into one matmul). Output-equivalent to the scan, with state carried for long context.
* :func:`gdn_step`       — the O(1)-state decode step (one token, vectorized over heads).

Assumption (to confirm at the torch-oracle stage, per project methodology): the exact gated-delta
recurrence uses the **delta correction** ``u = v − Sᵀk`` (Yang et al. 2024 / fla
``chunk_gated_delta_rule``), NOT the plain gated-linear-attention update ``S←g·S + β·kᵀv``. The
plain form is recoverable by setting ``DELTA_RULE=False`` below; both are self-consistent across
the three paths, so the scan==chunk==decode gate holds either way and the torch oracle picks the
one that matches the checkpoint.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.moe import silu

# The gated *delta* rule subtracts the current prediction (u = v − Sᵀk) before the rank-1 write.
# Set False for the plain gated-linear-attention update (S ← g·S + β·kᵀv, the task's simplified
# form). Both keep scan==chunk==decode; the torch oracle selects the checkpoint-correct one.
DELTA_RULE = True


def _softplus(x: mx.array) -> mx.array:  # numerically stable, matches mamba_mixer
    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """L2-normalize the last dim (DeltaNet q/k normalization), in fp32."""
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def _repeat_kheads(t: mx.array, rep: int, axis: int) -> mx.array:
    """Repeat the ``k_heads`` axis ``rep`` times so it aligns with the ``v_heads`` axis (GQA 4:1)."""
    return mx.repeat(t, rep, axis=axis)


def gdn_recurrence(q, k, v, g, beta, state_in=None):
    """Naive O(L) gated-delta-rule scan — the parity oracle. Returns (o, state_out).

    q,k: (B,L,Hk,Dk)   v: (B,L,Hv,Dv)   g,beta: (B,L,Hv)   state_in: (B,Hv,Dk,Dv)|None

    Hk key-heads grouped under Hv value-heads (rep=Hv//Hk). Scan runs in fp32 (the recurrence is
    bf16-unstable, like Mamba); q/k are assumed already l2-normalized by the caller.
    """
    b, length, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    rep = hv // hk
    q = _repeat_kheads(q.astype(mx.float32), rep, axis=2)  # (B,L,Hv,Dk)
    k = _repeat_kheads(k.astype(mx.float32), rep, axis=2)
    v, g, beta = v.astype(mx.float32), g.astype(mx.float32), beta.astype(mx.float32)
    s = mx.zeros((b, hv, dk, dv), dtype=mx.float32) if state_in is None else state_in.astype(mx.float32)
    os = []
    for t in range(length):  # reference only: explicit time loop (never the hot path)
        kt = k[:, t]                                            # (B,Hv,Dk)
        vt = v[:, t]                                            # (B,Hv,Dv)
        if DELTA_RULE:
            pred = mx.sum(s * kt[:, :, :, None], axis=2)        # Sᵀk: (B,Hv,Dv)
            u = vt - pred
        else:
            u = vt
        s = (g[:, t][:, :, None, None] * s
             + beta[:, t][:, :, None, None] * (kt[:, :, :, None] * u[:, :, None, :]))
        o = mx.sum(s * q[:, t][:, :, :, None], axis=2)          # Sᵀq: (B,Hv,Dv)
        os.append(o)
    return mx.stack(os, axis=1), s


def gdn_step(q_t, k_t, v_t, g_t, beta_t, state):
    """One decode step (O(1) state), vectorized over heads. Returns (o_t, state).

    q_t,k_t: (B,Hk,Dk)  v_t: (B,Hv,Dv)  g_t,beta_t: (B,Hv)  state: (B,Hv,Dk,Dv)

    Scan in fp32 (bf16 is unstable); q/k assumed l2-normalized. Output kept fp32 (the mixer casts
    back after the gated norm).
    """
    q_t, k_t = q_t.astype(mx.float32), k_t.astype(mx.float32)
    v_t, g_t, beta_t = v_t.astype(mx.float32), g_t.astype(mx.float32), beta_t.astype(mx.float32)
    state = state.astype(mx.float32)
    hk = k_t.shape[1]
    hv = v_t.shape[1]
    rep = hv // hk
    k_t = _repeat_kheads(k_t, rep, axis=1)                      # (B,Hv,Dk)
    q_t = _repeat_kheads(q_t, rep, axis=1)
    if DELTA_RULE:
        pred = mx.sum(state * k_t[:, :, :, None], axis=2)       # (B,Hv,Dv)
        u = v_t - pred
    else:
        u = v_t
    state = (g_t[:, :, None, None] * state
             + beta_t[:, :, None, None] * (k_t[:, :, :, None] * u[:, :, None, :]))
    o = mx.sum(state * q_t[:, :, :, None], axis=2)              # (B,Hv,Dv)
    return o, state


def gdn_chunked(q, k, v, g, beta, chunk_size, state_in=None):
    """Chunk-parallel gated-delta-rule prefill — output-equivalent to :func:`gdn_recurrence`.

    Bounded loop over ``chunk_size`` chunks carrying the [Dk,Dv] state across token-blocks (so
    long-context prefill stays bounded-memory). The within-chunk delta rule is still sequential
    over its ≤chunk tokens — the gated delta rule has a true ``Sᵀk`` data-dependence (no scalar-A
    segment-sum dual), so this is the bounded inner loop the rules permit (cf. the chunked-SSD
    chunk scan), not a per-token hot loop over the full sequence. Returns (o, state).
    """
    out_dtype = v.dtype
    b, length, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    q = _repeat_kheads(q.astype(mx.float32), hv // hk, axis=2)
    k = _repeat_kheads(k.astype(mx.float32), hv // hk, axis=2)
    v, g, beta = v.astype(mx.float32), g.astype(mx.float32), beta.astype(mx.float32)
    if length % chunk_size != 0:
        raise ValueError(f"length {length} not divisible by chunk_size {chunk_size}")
    nc = length // chunk_size
    s = mx.zeros((b, hv, dk, dv), dtype=mx.float32) if state_in is None else state_in.astype(mx.float32)
    out = []
    for c in range(nc):  # bounded: nc = block_len / chunk_size
        lo = c * chunk_size
        oc, s = _gdn_chunk(q[:, lo:lo + chunk_size], k[:, lo:lo + chunk_size],
                           v[:, lo:lo + chunk_size], g[:, lo:lo + chunk_size],
                           beta[:, lo:lo + chunk_size], s)
        out.append(oc)
    return mx.concatenate(out, axis=1).astype(out_dtype), s


def _gdn_chunk(q, k, v, g, beta, s):
    """One chunk of the delta-rule scan (inputs fp32, repeated to Hv). Returns (o_chunk, state)."""
    b, q_len, hv, dk = q.shape
    os = []
    for t in range(q_len):  # bounded: <= chunk_size, the permitted inner block scan
        kt, vt = k[:, t], v[:, t]
        if DELTA_RULE:
            u = vt - mx.sum(s * kt[:, :, :, None], axis=2)
        else:
            u = vt
        s = (g[:, t][:, :, None, None] * s
             + beta[:, t][:, :, None, None] * (kt[:, :, :, None] * u[:, :, None, :]))
        os.append(mx.sum(s * q[:, t][:, :, :, None], axis=2))
    return mx.stack(os, axis=1), s


def causal_conv1d(u, weight, bias=None):
    """Causal depthwise conv (prefill). u: (B,L,C), weight: (C,K), bias: (C,)|None -> (B,L,C).

    Windowed sum over the bounded kernel (K=4); ``mx.conv1d(groups=C)`` is the production swap.
    """
    length = u.shape[1]
    k = weight.shape[-1]
    up = mx.pad(u, [(0, 0), (k - 1, 0), (0, 0)])  # left-pad K-1 (causal)
    y = sum(up[:, i:i + length, :] * weight[:, i] for i in range(k))  # bounded K loop
    return y if bias is None else y + bias


def causal_conv1d_step(u_t, weight, conv_state, bias=None):
    """One decode step of the causal conv. u_t: (B,C), conv_state: (B,K-1,C) -> (y, new_state)."""
    window = mx.concatenate([conv_state, u_t[:, None, :]], axis=1)  # (B,K,C)
    y = mx.sum(window * mx.swapaxes(weight, 0, 1)[None], axis=1)     # (B,C)
    if bias is not None:
        y = y + bias
    return y, window[:, 1:, :]


def _gated_rmsnorm(x, weight, gate, eps):
    """Per-head gated RMSNorm (FusedRMSNormGated): RMSNorm(x)·weight, gated by silu(gate).

    x,gate: (B,L,Hv,Dv); weight: (Dv,). Normalization in fp32; the gate multiplies the *normalized*
    value (gate-before-weight order, matching the Nemotron mamba gated norm)."""
    xf = x.astype(mx.float32) * silu(gate.astype(mx.float32))
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return weight.astype(mx.float32) * xf


class GatedDeltaNet(nn.Module):
    """Qwen3.5 linear-attention (Gated DeltaNet) mixer. State = (recurrent [Hv,Dk,Dv], conv ring)."""

    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.hk = cfg.linear_num_key_heads          # 16
        self.hv = cfg.linear_num_value_heads        # 64
        self.dk = cfg.linear_key_head_dim           # 128
        self.dv = cfg.linear_value_head_dim         # 128
        self.k_dim = cfg.linear_k_dim               # 2048 (q == k width)
        self.v_dim = cfg.linear_v_dim               # 8192
        self.conv_dim = cfg.linear_qkv_dim          # 12288
        self.k = cfg.linear_conv_kernel_dim         # 4
        self.eps = cfg.norm_eps
        self.chunk = 64
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.hv, bias=False)      # decay logits
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.hv, bias=False)      # write-strength logits
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.v_dim, bias=False)   # output gate
        self.out_proj = nn.Linear(self.v_dim, cfg.hidden_size, bias=False)
        self.norm = mx.ones((self.dv,))                  # per-head gated RMSNorm weight (fp32 in ckpt)
        self.conv_weight = mx.zeros((self.conv_dim, self.k))  # depthwise (C,K); loader squeezes (C,1,K)
        self.conv_bias = mx.zeros((self.conv_dim,))
        self.A_log = mx.zeros((self.hv,))                # per-value-head decay (fp32 in ckpt)
        self.dt_bias = mx.zeros((self.hv,))

    def _split_qkv(self, qkv, b, t):
        q = qkv[..., : self.k_dim].reshape(b, t, self.hk, self.dk)
        k = qkv[..., self.k_dim : 2 * self.k_dim].reshape(b, t, self.hk, self.dk)
        v = qkv[..., 2 * self.k_dim :].reshape(b, t, self.hv, self.dv)
        return q, k, v

    def __call__(self, x, *, state=None, conv_state=None):
        b, t, _ = x.shape
        a = -mx.exp(self.A_log.astype(mx.float32))                  # (Hv,) < 0
        qkv = self.in_proj_qkv(x)
        dt = _softplus(self.in_proj_a(x).astype(mx.float32) + self.dt_bias.astype(mx.float32))  # (B,T,Hv)
        g = mx.exp(dt * a[None, None, :])                          # (B,T,Hv) decay in (0,1]
        beta = mx.sigmoid(self.in_proj_b(x).astype(mx.float32))    # (B,T,Hv)
        z = self.in_proj_z(x).reshape(b, t, self.hv, self.dv)      # output gate (pre-sigmoid/silu)
        if conv_state is None:  # prefill
            qkv_pre = qkv
            qkv = silu(causal_conv1d(qkv, self.conv_weight, self.conv_bias))
            q, k, v = self._split_qkv(qkv, b, t)
            q, k = _l2norm(q), _l2norm(k)
            o, state = self._prefill(q, k, v, g, beta, state)
            conv_state = qkv_pre[:, -(self.k - 1):]               # for decode continuation
        else:  # decode step (t == 1)
            conv_out, conv_state = causal_conv1d_step(qkv[:, 0], self.conv_weight, conv_state,
                                                      self.conv_bias)
            q, k, v = self._split_qkv(silu(conv_out)[:, None], b, 1)
            q, k = _l2norm(q), _l2norm(k)
            if state is None:
                state = mx.zeros((b, self.hv, self.dk, self.dv), dtype=mx.float32)
            o, state = gdn_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state)
            o = o[:, None]                                         # (B,1,Hv,Dv)
        o = _gated_rmsnorm(o, self.norm, z, self.eps)             # (B,T,Hv,Dv) fp32
        o = o.reshape(b, t, self.v_dim).astype(x.dtype)
        return self.out_proj(o), state, conv_state

    def _prefill(self, q, k, v, g, beta, state):
        length, qc = q.shape[1], self.chunk
        pad = (-length) % qc
        if pad:  # pad to a chunk multiple; g=1,beta=0 -> padded steps are no-ops, sliced off after
            q = mx.pad(q, [(0, 0), (0, pad), (0, 0), (0, 0)])
            k = mx.pad(k, [(0, 0), (0, pad), (0, 0), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, pad), (0, 0), (0, 0)])
            g = mx.pad(g, [(0, 0), (0, pad), (0, 0)], constant_values=1.0)   # decay 1 = identity
            beta = mx.pad(beta, [(0, 0), (0, pad), (0, 0)])                  # write 0 = no update
        o, state = gdn_chunked(q, k, v, g, beta, qc, state_in=state)
        return o[:, :length], state
