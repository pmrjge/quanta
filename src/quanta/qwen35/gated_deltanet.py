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
* :func:`gdn_chunked`    — the chunk-carrying prefill: a bounded loop over fixed-size chunks
  carrying the [k_dim, v_dim] state; **within** a chunk the delta rule is still sequential over
  its ≤chunk tokens (chunking just bounds the prefill memory and lets the cross-chunk decay
  fold into one matmul). Output-equivalent to the scan, with state carried for long context.
* :func:`gdn_chunked_wy` — the chunk-PARALLEL prefill (WY/UT representation, the fla /
  HF ``torch_chunk_gated_delta_rule`` algorithm): the within-chunk delta rule is folded into
  batched matmuls over ALL chunks at once via the UT transform ``T = (I − tril(diag(β)KKᵀ⊙Γ))⁻¹``
  (forward substitution — a bounded loop over the ≤chunk ROWS, run ONCE for the whole sequence,
  not per chunk), then a bounded cross-chunk state-carry loop of ~6 matmuls per chunk. This is
  what makes 100K–1M-token prefill feasible (the sequential within-chunk scan is O(L) tiny
  kernel launches per layer; WY is O(L/C) matmuls). Output-equivalent to the scan (fp32
  reassociation only — NOT bit-exact); takes the **log** decay (``dt·a``) so extreme decays
  never round through ``exp`` → 0 → ``log`` (the cumulative-decay differences stay finite).
* :func:`gdn_step`       — the O(1)-state decode step (one token, vectorized over heads).

Chunked prefill **continuation** (long-context driver): :func:`causal_conv1d` takes an optional
``state`` (the prior K-1 pre-activation rows) replacing the zero left-pad — bit-exact to the
full-sequence conv split at any boundary — and :meth:`GatedDeltaNet.__call__` treats
``conv_state`` given with ``T>1`` as a *prefill continuation* (previously an invalid input that
silently took token 0 only). Both gated in ``parity/qwen35_prefill_chunked_test.py``.

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
        s = g[:, t][:, :, None, None] * s                       # decay the state FIRST (gated delta rule)
        if DELTA_RULE:
            pred = mx.sum(s * kt[:, :, :, None], axis=2)        # (decayed S)ᵀk: (B,Hv,Dv)
            u = vt - pred
        else:
            u = vt
        s = s + beta[:, t][:, :, None, None] * (kt[:, :, :, None] * u[:, :, None, :])  # add write
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
    state = g_t[:, :, None, None] * state                      # decay the state FIRST (gated delta rule)
    if DELTA_RULE:
        pred = mx.sum(state * k_t[:, :, :, None], axis=2)       # (decayed S)ᵀk: (B,Hv,Dv)
        u = v_t - pred
    else:
        u = v_t
    state = state + beta_t[:, :, None, None] * (k_t[:, :, :, None] * u[:, :, None, :])  # add write
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
        s = g[:, t][:, :, None, None] * s                       # decay the state FIRST (gated delta rule)
        if DELTA_RULE:
            u = vt - mx.sum(s * kt[:, :, :, None], axis=2)      # (decayed S)ᵀk
        else:
            u = vt
        s = s + beta[:, t][:, :, None, None] * (kt[:, :, :, None] * u[:, :, None, :])  # add write
        os.append(mx.sum(s * q[:, t][:, :, :, None], axis=2))
    return mx.stack(os, axis=1), s


def gdn_chunked_wy(q, k, v, log_g, beta, chunk_size=64, state_in=None):
    """Chunk-PARALLEL gated-delta-rule prefill (WY/UT representation) — output-equivalent to
    :func:`gdn_recurrence` / :func:`gdn_chunked` up to fp32 reassociation. Returns (o, state).

    q,k: (B,L,Hk,Dk) — q already l2-normalized AND scaled by ``Dk^-0.5``, k l2-normalized (the
    mixer's convention; the HF reference applies the scale internally, here the caller does).
    v: (B,L,Hv,Dv).  log_g, beta: (B,L,Hv) — **log** decay ``dt·a`` (NOT the exponentiated ``g``;
    log-space keeps the cumulative-decay differences finite under extreme decay) and write
    strength.  state_in: (B,Hv,Dk,Dv)|None.

    Port of the HF/fla ``torch_chunk_gated_delta_rule`` (the N1-gated reference): per chunk of
    ``chunk_size`` tokens, the sequential delta rule is folded into matmuls via the UT transform
    ``T = (I − strictly_lower(diag(β) K Kᵀ ⊙ Γ))⁻¹`` (forward substitution over the ≤chunk rows —
    a bounded loop run ONCE over ALL chunks batched, the rules' permitted inner block scan), giving
    the pre-state "new values" ``U = T (β⊙V)`` and decay-folded keys ``W = T (β⊙K⊙exp(Γ))``; then a
    bounded cross-chunk loop carries the [Dk,Dv] state with ~6 matmuls per chunk. Ragged lengths
    pad internally (log-decay 0 = identity, β 0 = no write — pad steps are provable no-ops on both
    the outputs and the carried state). Memory is O(L/C·C²) per head for the decay/UT matrices —
    the long-context driver bounds L per call (its ``chunk_tokens``), never the full sequence.
    """
    out_dtype = v.dtype
    b, length, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    rep = hv // hk
    c = int(chunk_size)
    # fp32, k-heads repeated under v-heads, head-major [B,Hv,L,D] (the reference layout)
    q = mx.transpose(_repeat_kheads(q.astype(mx.float32), rep, axis=2), (0, 2, 1, 3))
    k = mx.transpose(_repeat_kheads(k.astype(mx.float32), rep, axis=2), (0, 2, 1, 3))
    v = mx.transpose(v.astype(mx.float32), (0, 2, 1, 3))
    lg = mx.transpose(log_g.astype(mx.float32), (0, 2, 1))                 # [B,Hv,L]
    bt = mx.transpose(beta.astype(mx.float32), (0, 2, 1))
    pad = (-length) % c
    if pad:  # pad to a chunk multiple: log-decay 0 (identity) + beta 0 (no write) ⇒ no-op steps
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)])
        lg = mx.pad(lg, [(0, 0), (0, 0), (0, pad)])
        bt = mx.pad(bt, [(0, 0), (0, 0), (0, pad)])
    nc = (length + pad) // c
    qc = q.reshape(b, hv, nc, c, dk)
    kc = k.reshape(b, hv, nc, c, dk)
    vc = v.reshape(b, hv, nc, c, dv)
    lgc = mx.cumsum(lg.reshape(b, hv, nc, c), axis=-1)     # within-chunk cumulative log decay
    btc = bt.reshape(b, hv, nc, c)
    k_beta = kc * btc[..., None]
    v_beta = vc * btc[..., None]
    # within-chunk pairwise decay Γ[i,j] = exp(lg_i − lg_j), lower-triangular (diag = 1); the
    # exp is taken on the tril'd difference (upper entries exp(0)=1) then tril'd again — the
    # reference's exact two-tril construction.
    diff = lgc[..., :, None] - lgc[..., None, :]
    decay = mx.tril(mx.exp(mx.tril(diff)))                                 # [B,Hv,NC,C,C]
    strict = mx.tril(mx.ones((c, c), dtype=mx.float32), k=-1)              # zero diag + upper
    attn = -((k_beta @ mx.swapaxes(kc, -1, -2)) * decay) * strict
    # UT transform: forward substitution inverting (I − strictly-lower) — sequential over the
    # bounded ≤c rows, batched over ALL chunks at once (slice-assignment; MLX arrays are
    # functionally updated so the RHS reads the pre-assignment rows).
    for i in range(1, c):
        row = attn[..., i, :i]
        attn[..., i, :i] = row + mx.sum(row[..., None] * attn[..., :i, :i], axis=-2)
    attn = attn + mx.eye(c, dtype=mx.float32)
    u_all = attn @ v_beta                                  # T(β⊙V): pre-state "new values"
    w_all = attn @ (k_beta * mx.exp(lgc)[..., None])       # T(β⊙K⊙exp(Γcum)): decay-folded keys
    s = (mx.zeros((b, hv, dk, dv), dtype=mx.float32) if state_in is None
         else state_in.astype(mx.float32))
    outs = []
    for ci in range(nc):  # bounded cross-chunk state carry (~6 matmuls per chunk)
        q_i, k_i, lg_i = qc[:, :, ci], kc[:, :, ci], lgc[:, :, ci]
        attn_i = (q_i @ mx.swapaxes(k_i, -1, -2)) * decay[:, :, ci]
        v_new = u_all[:, :, ci] - w_all[:, :, ci] @ s                       # delta vs carried state
        o_i = (q_i * mx.exp(lg_i)[..., None]) @ s + attn_i @ v_new
        g_last = lg_i[..., -1]
        s = (s * mx.exp(g_last)[..., None, None]
             + mx.swapaxes(k_i * mx.exp(g_last[..., None] - lg_i)[..., None], -1, -2) @ v_new)
        outs.append(o_i)
    o = mx.concatenate(outs, axis=2)[:, :, :length]                         # [B,Hv,L,Dv]
    return mx.transpose(o, (0, 2, 1, 3)).astype(out_dtype), s


def causal_conv1d(u, weight, bias=None, state=None):
    """Causal depthwise conv (prefill). u: (B,L,C), weight: (C,K), bias: (C,)|None -> (B,L,C).

    Windowed sum over the bounded kernel (K=4); ``mx.conv1d(groups=C)`` is the production swap.
    ``state`` (B,K-1,C) — the prior K-1 *pre-activation* rows — replaces the zero left-pad for
    chunked-prefill continuation: bit-exact to the full-sequence conv split at that boundary
    (identical terms, identical bounded-K summation order). ``None`` = the historical zero pad.
    """
    length = u.shape[1]
    k = weight.shape[-1]
    if state is None:
        up = mx.pad(u, [(0, 0), (k - 1, 0), (0, 0)])  # left-pad K-1 (causal)
    else:
        if state.ndim != 3 or state.shape[1] != k - 1 or state.shape[2] != u.shape[2]:
            raise ValueError(f"causal_conv1d state must be [B,{k - 1},{u.shape[2]}] "
                             f"(got {tuple(state.shape)}) — rule 6")
        up = mx.concatenate([state.astype(u.dtype), u], axis=1)
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
    """Per-head gated RMSNorm — Qwen3.5 ``Qwen3_5MoeRMSNormGated``: ``silu(gate)·weight·RMSNorm(x)``.

    x,gate: (B,L,Hv,Dv); weight: (Dv,). The RMSNorm normalizes ``x`` **alone** (fp32), then the
    weight multiplies, then the silu-gate multiplies — gate AFTER the norm, NOT before. The gate must
    not enter the RMS reduction (it is per-element, and RMS is nonlinear), so the gate-before-norm
    order (Nemotron's mamba norm) is WRONG here. Matches the HF reference forward (normalize → weight
    → ``* silu(gate)``)."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)   # RMSNorm of x ALONE
    xf = weight.astype(mx.float32) * xf                                  # weight
    return xf * silu(gate.astype(mx.float32))                            # then silu-gate


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

    def __call__(self, x, *, state=None, conv_state=None, wy=False):
        b, t, _ = x.shape
        a = -mx.exp(self.A_log.astype(mx.float32))                  # (Hv,) < 0
        qkv = self.in_proj_qkv(x)
        dt = _softplus(self.in_proj_a(x).astype(mx.float32) + self.dt_bias.astype(mx.float32))  # (B,T,Hv)
        log_g = dt * a[None, None, :]                              # (B,T,Hv) log decay (≤ 0)
        g = mx.exp(log_g)                                          # (B,T,Hv) decay in (0,1]
        beta = mx.sigmoid(self.in_proj_b(x).astype(mx.float32))    # (B,T,Hv)
        z = self.in_proj_z(x).reshape(b, t, self.hv, self.dv)      # output gate (pre-sigmoid/silu)
        if conv_state is None or t > 1:  # prefill — fresh, or chunked CONTINUATION (conv_state given)
            qkv_pre = qkv
            qkv = silu(causal_conv1d(qkv, self.conv_weight, self.conv_bias, state=conv_state))
            q, k, v = self._split_qkv(qkv, b, t)
            q, k = _l2norm(q) * (self.dk ** -0.5), _l2norm(k)    # HF scales q by 1/√dk for the readout
            o, state = self._prefill(q, k, v, g, beta, state, log_g=log_g, wy=wy)
            # next conv window: last K-1 PRE-conv rows, left-extended by the prior window (or
            # zeros) when this block is shorter than K-1 (the t<K-1 edge previously produced a
            # silently wrong-shaped window).
            if t >= self.k - 1:
                conv_state = qkv_pre[:, -(self.k - 1):]
            else:
                prev = (conv_state if conv_state is not None
                        else mx.zeros((b, self.k - 1, self.conv_dim), dtype=qkv_pre.dtype))
                conv_state = mx.concatenate([prev.astype(qkv_pre.dtype), qkv_pre],
                                            axis=1)[:, -(self.k - 1):]
        else:  # decode step (t == 1)
            conv_out, conv_state = causal_conv1d_step(qkv[:, 0], self.conv_weight, conv_state,
                                                      self.conv_bias)
            q, k, v = self._split_qkv(silu(conv_out)[:, None], b, 1)
            q, k = _l2norm(q) * (self.dk ** -0.5), _l2norm(k)    # HF scales q by 1/√dk for the readout
            if state is None:
                state = mx.zeros((b, self.hv, self.dk, self.dv), dtype=mx.float32)
            o, state = gdn_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state)
            o = o[:, None]                                         # (B,1,Hv,Dv)
        o = _gated_rmsnorm(o, self.norm, z, self.eps)             # (B,T,Hv,Dv) fp32
        o = o.reshape(b, t, self.v_dim).astype(x.dtype)
        return self.out_proj(o), state, conv_state

    def _prefill(self, q, k, v, g, beta, state, *, log_g=None, wy=False):
        if wy:  # chunk-parallel WY/UT path (long-context driver) — needs the LOG decay
            if log_g is None:
                raise ValueError("wy prefill needs log_g (the pre-exp decay) — rule 6")
            return gdn_chunked_wy(q, k, v, log_g, beta, self.chunk, state_in=state)
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
