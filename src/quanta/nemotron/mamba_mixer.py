"""Mamba-2 mixer block for Nemotron-H (nn.Module wrapping the SSD kernel).

Pipeline (matches NemotronH/Mamba-2): ``in_proj`` -> split [z, xBC, dt] -> causal conv1d on
xBC + silu -> split [x, B, C] -> SSD -> gated RMSNorm with ``z`` -> ``out_proj``. Prefill uses
the chunked SSD (padding the sequence up to a chunk multiple with dt=0 no-op steps); decode
uses the O(1) recurrence + a rolling conv state. The Mamba "cache" is ``(ssm_state, conv_state)``.

A=-exp(A_log); dt=softplus(dt_proj + dt_bias). The SSM core params (A_log/D/dt_bias/conv) are
loaded in bf16/fp32 and never quantized.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.mamba_ssd import (
    causal_conv1d,
    causal_conv1d_step,
    ssd_chunked,
    ssd_step,
    ssd_step_fused,
)

# Opt-in fused one-launch decode kernel. Parity-exact (== ssd_step, ~2e-7), BUT measured NOT a
# win in the compiled decode path (mx.compile already fuses the composed SSD ops: ~34 vs ~35
# tok/s), so default off per rule-4 (optimizations default to the proven path until a measured win).
FUSED_SSD_STEP = False


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _softplus(x: mx.array) -> mx.array:  # numerically stable
    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


class MambaRMSNormGated(nn.Module):
    """Mamba-2 gated RMSNorm (Zamba2 / NemotronH ``Zamba2RMSNormGated``): gate the SSD output by
    ``silu(z)``, then RMS-normalize **within each of ``n_groups`` channel groups** — the variance is
    over ``d_inner // n_groups`` channels, **not** the full ``d_inner``.

    The reference applies the group structure (``group_size = d_inner // n_groups``, ``group_count =
    n_groups``) because the SSD itself is grouped (B/C share ``n_groups`` groups). A plain full-width
    ``nn.RMSNorm`` here is *self-consistent* (prefill==decode) but silently diverges from the
    transformers mixer by ~40% — caught by ``parity/nemotron_ultra_layer_parity.py`` (the old
    ``nemotron_layers_test`` only checked self-consistency, never a numeric reference). The per-group
    normalization uses the fused ``mx.fast.rms_norm`` over the group axis (weight applied after, like
    the reference: ``self.weight * normalize_per_group(y * silu(z))``)."""

    def __init__(self, d_inner: int, n_groups: int, eps: float) -> None:
        super().__init__()
        if d_inner % n_groups != 0:
            raise ValueError(f"d_inner {d_inner} not divisible by n_groups {n_groups}")
        self.weight = mx.ones((d_inner,))
        self.n_groups = n_groups
        self.group_size = d_inner // n_groups
        self.eps = eps

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        b, t, d = y.shape
        h = (y.astype(mx.float32) * _silu(z.astype(mx.float32))).reshape(b, t, self.n_groups, self.group_size)
        ones = mx.ones((self.group_size,), dtype=mx.float32)  # per-group RMS, weight applied after
        h = mx.fast.rms_norm(h, ones, self.eps).reshape(b, t, d).astype(y.dtype)
        return self.weight * h


class MambaMixer(nn.Module):
    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.h, self.p = cfg.mamba_num_heads, cfg.mamba_head_dim
        self.g, self.n = cfg.mamba_n_groups, cfg.ssm_state_size
        self.k, self.chunk = cfg.conv_kernel, cfg.chunk_size
        self.d_inner, self.conv_dim = cfg.mamba_d_inner, cfg.mamba_conv_dim
        self.in_proj = nn.Linear(cfg.hidden_size, cfg.mamba_in_proj_dim, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.hidden_size, bias=False)
        self.norm = MambaRMSNormGated(self.d_inner, cfg.mamba_n_groups, cfg.norm_eps)  # group-wise
        self.conv_weight = mx.zeros((self.conv_dim, self.k))  # depthwise (C,K); loader squeezes (C,1,K)
        self.conv_bias = mx.zeros((self.conv_dim,))
        self.A_log = mx.zeros((self.h,))
        self.D = mx.ones((self.h,))
        self.dt_bias = mx.zeros((self.h,))

    def _split(self, proj):
        z = proj[..., : self.d_inner]
        xbc = proj[..., self.d_inner : self.d_inner + self.conv_dim]
        dt = proj[..., self.d_inner + self.conv_dim :]
        return z, xbc, dt

    def _split_xbc(self, xbc, b, t):
        gn = self.g * self.n
        x = xbc[..., : self.d_inner].reshape(b, t, self.h, self.p)
        bm = xbc[..., self.d_inner : self.d_inner + gn].reshape(b, t, self.g, self.n)
        cm = xbc[..., self.d_inner + gn :].reshape(b, t, self.g, self.n)
        return x, bm, cm

    def __call__(self, x, *, state=None, conv_state=None, chunked_cont=False):
        b, t, _ = x.shape
        a = -mx.exp(self.A_log)
        z, xbc, dt = self._split(self.in_proj(x))
        if conv_state is None:  # fresh prefill (chunked SSD)
            xbc_pre = xbc
            xbc = _silu(causal_conv1d(xbc, self.conv_weight, self.conv_bias))
            xs, bm, cm = self._split_xbc(xbc, b, t)
            dt = _softplus(dt + self.dt_bias)
            y, state = self._prefill(xs, dt, a, bm, cm, state)
            conv_state = xbc_pre[:, -(self.k - 1) :]   # for decode continuation
        elif chunked_cont:
            # Prefill CONTINUATION via the **chunked SSD** (resumed from ``(state, conv_state)``) — the
            # #152 paged-suffix path. Unlike the per-token branch below (which matches ``batch_step``),
            # this matches the FRESH chunked-prefill numerics so a suffix prefilled on top of a reused
            # prefix is output-equivalent to a one-shot prefill of prefix+suffix (rule 4). The restored
            # ``conv_state`` (the raw last ``k-1`` xBC of the prefix) is prepended so the depthwise
            # conv1d sees the correct left context for the first suffix positions; the prepended window
            # is sliced off after the conv and is NOT fed to the SSD (it is already folded into
            # ``state``). The SSD resumes via ``ssd_chunked(state_in=state)`` (same kernel as fresh
            # prefill). Gated on the real artifact in ``parity/nemotron_paged_real_test.py``.
            if state is None:
                state = mx.zeros((b, self.h, self.n, self.p), dtype=x.dtype)
            cw = conv_state.shape[1]                          # == k-1 (raw prefix-tail window)
            xbc_ext = mx.concatenate([conv_state, xbc], axis=1)        # [b, (k-1)+t, conv_dim]
            xbc_conv = _silu(causal_conv1d(xbc_ext, self.conv_weight, self.conv_bias))[:, cw:]
            xs, bm, cm = self._split_xbc(xbc_conv, b, t)
            dt = _softplus(dt + self.dt_bias)
            y, state = self._prefill(xs, dt, a, bm, cm, state)
            conv_state = xbc_ext[:, -(self.k - 1) :]          # raw window for the next continuation
        else:
            # Mid-stream continuation: use the **same per-token step ops** as the t==1
            # decode path for every t in [0..T-1]. Chunked SSD with ``state_in`` is
            # mathematically equivalent but the chunk-major reductions diverge from the
            # per-token step ops in bf16 (~7-bit mantissa) — measured ~22% argmax-match
            # against :meth:`quanta.nemotron.batched_runtime.NemotronBatchedResidentModel.batch_step`
            # (which always steps per-token), see commit 5 of docs/batched_tree_verify.md.
            # batch_step calls ``blk(...)`` with t=1 per replica → this branch matches it
            # by construction by using the exact same ``causal_conv1d_step`` + ``ssd_step``
            # ops in the same order, just T times. The Python loop is bounded by ``depth+1``
            # (W^D verify chain length, typically 2–3) — the kind of bounded per-token
            # spec-verify loop CLAUDE.md rule 3 explicitly permits (the unbounded forbidden
            # form is over tokens-of-the-output or experts-per-token, neither apply here).
            if state is None:
                state = mx.zeros((b, self.h, self.n, self.p), dtype=x.dtype)
            dt_all = _softplus(dt + self.dt_bias)
            step_fn = ssd_step_fused if FUSED_SSD_STEP else ssd_step
            ys: list[mx.array] = []
            for ti in range(t):
                conv_out, conv_state = causal_conv1d_step(
                    xbc[:, ti], self.conv_weight, conv_state, self.conv_bias,
                )
                xs_t, bm_t, cm_t = self._split_xbc(_silu(conv_out)[:, None], b, 1)
                y_t, state = step_fn(
                    xs_t[:, 0], dt_all[:, ti], a, bm_t[:, 0], cm_t[:, 0], self.D, state,
                )
                ys.append(y_t[:, None])
            y = ys[0] if t == 1 else mx.concatenate(ys, axis=1)
        y = y.reshape(b, t, self.d_inner)
        y = self.norm(y, z)  # group-wise gated RMSNorm (gate=silu(z) applied inside, per n_groups)
        return self.out_proj(y), state, conv_state

    def _prefill(self, xs, dt, a, bm, cm, state):
        length, q = xs.shape[1], self.chunk
        pad = (-length) % q
        if pad:  # pad to a chunk multiple; dt=0 -> padded steps are no-ops, sliced off after
            xs = mx.pad(xs, [(0, 0), (0, pad), (0, 0), (0, 0)])
            bm = mx.pad(bm, [(0, 0), (0, pad), (0, 0), (0, 0)])
            cm = mx.pad(cm, [(0, 0), (0, pad), (0, 0), (0, 0)])
            dt = mx.pad(dt, [(0, 0), (0, pad), (0, 0)])
        y, state = ssd_chunked(xs, dt, a, bm, cm, self.D, q, state_in=state)
        return y[:, :length], state
