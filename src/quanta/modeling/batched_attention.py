"""Batched ragged-length decode attention — Approach 1 for vectorizing the per-stream attention
loop in the ``B > 1`` decode runtimes (#152 throughput follow-on).

The batched decode runtimes step ``B`` concurrent streams. Today attention runs in a **Python loop
over streams** — DSV4 "Design-A" (per-stream attention + batched MoE), and the InternLM2 / Nemotron
``step_batch`` loops that run the whole inner model per stream. Each stream contributes ONE decode
query and a *ragged* number of cached keys/values (its own context length ``L_b``). This module
collapses that loop into a single fused :func:`mx.fast.scaled_dot_product_attention` over all ``B``
streams:

  * pad the per-stream keys/values to ``L_max = max_b L_b`` (zero tail);
  * add a per-stream additive mask ``[B, 1, 1, L_max]`` that sends padding columns (``j >= L_b``) to
    a large negative → zero softmax weight;
  * one SDPA over ``q [B, H, 1, D]`` × ``K, V [B, H, L_max, D]``.

Because the masked tail contributes ~zero to the softmax, each stream's output is **mathematically
identical** to a single-stream SDPA over exactly its own ``L_b`` keys; the only difference from the
per-stream loop is the SDPA reduction tiling (``L_max`` vs ``L_b``) → argmax-stable fp ULP noise,
not a logic change (the same pattern as the chunked-prefill parity note). This is the **pure-MLX**
realization of batched decode attention — :func:`mx.fast.scaled_dot_product_attention` only consumes
a dense ``[B, H, L, D]`` tensor, so a "varlen / paged" form that avoids the padding waste needs a
custom Metal kernel (the #153-class follow-up). For shared-prefix agentic streams the lengths are
close, so ``L_max ≈ L_b`` and the padding waste is small.

Rules honored: one fused ``mx.fast`` SDPA (rule 2); the only loop is the **bounded coarse per-stream
stack at the IO boundary** (``B ≤ max_batch``), never over tokens/heads/hidden (rule 3); a loud
pre-alloc guard on the padded-KV bytes (memory-safety — fail before allocating, never OOM the host).
"""

from __future__ import annotations

import mlx.core as mx

# Fail-loud ceiling on the transient padded-K+V tensor (bytes): 2·B·n_kv·L_max·D·itemsize. Decode is
# ``Lq = 1`` so the SDPA scores ``[B, H, 1, L_max]`` are linear in L_max (no O(T²) blow-up), but the
# B×L_max padding itself can still be large when one stream is far longer than the rest. Guard it so a
# pathological L_max fails loud instead of OOM-rebooting the host (a prior bench did exactly that).
_PADDED_KV_BYTES_CAP = 64 * 1024 ** 3  # 64 GiB transient ceiling


def _itemsize(a: mx.array) -> int:
    """Bytes per element of an mx.array (``nbytes / size``; size==0 ⇒ fall back to 4)."""
    n = int(a.size)
    return int(a.nbytes) // n if n else 4


def _guard_padded_kv(b: int, n_kv: int, l_max: int, d: int, itemsize: int) -> None:
    est = 2 * b * n_kv * l_max * d * itemsize
    if est > _PADDED_KV_BYTES_CAP:
        raise ValueError(
            f"batched_decode_attention: padded K+V would be ~{est / 1024 ** 3:.1f} GiB "
            f"(B={b}, n_kv={n_kv}, L_max={l_max}, D={d}) > cap "
            f"{_PADDED_KV_BYTES_CAP / 1024 ** 3:.0f} GiB — refusing to allocate (memory-safety). "
            "Streams differ too much in length for the pad+mask path; keep per-stream or use a "
            "varlen kernel (#153-class).")


def decode_pad_mask(lengths: list[int], l_max: int, dtype: mx.Dtype) -> mx.array:
    """Additive SDPA mask ``[B, 1, 1, L_max]``: ``0`` for valid key columns (``j < L_b``), large
    negative (≈ ``-inf`` → zero softmax weight) for the padded tail (``j >= L_b``).

    A large finite negative (not ``float('-inf')``) is used so the kernel never produces ``NaN`` even
    if a future caller passes a fully-padded row; every real decode stream has ``L_b >= 1`` so at
    least one column is always valid here.
    """
    cols = mx.arange(l_max)[None, :]                      # [1, L_max]
    lens = mx.array(lengths, dtype=mx.int32)[:, None]      # [B, 1]
    valid = cols < lens                                    # [B, L_max] bool
    neg = mx.array(-1e9, dtype=dtype)
    mask = mx.where(valid, mx.array(0.0, dtype=dtype), neg)  # [B, L_max]
    return mask[:, None, None, :]                          # [B, 1, 1, L_max]


def _pad_seq(a: mx.array, l_max: int) -> mx.array:
    """Pad ``[n_kv, L_b, D]`` to ``[n_kv, L_max, D]`` with a zero tail along the seq axis."""
    pad = l_max - int(a.shape[1])
    if pad == 0:
        return a
    return mx.pad(a, [(0, 0), (0, pad), (0, 0)])


def batched_decode_attention(
    q_list: list[mx.array],
    k_list: list[mx.array],
    v_list: list[mx.array],
    *,
    scale: float,
    n_rep: int = 1,
) -> mx.array:
    """One fused SDPA across ``B`` decode streams of ragged KV length → ``[B, H, 1, D]``.

    Args:
      q_list[b]: ``[H, 1, D]`` (or ``[1, H, 1, D]``) — stream ``b``'s single decode query, RoPE
        already applied by the caller (mirrors the per-stream path so parity is exact).
      k_list[b], v_list[b]: ``[n_kv, L_b, D]`` (or ``[1, n_kv, L_b, D]``) — stream ``b``'s cached
        keys/values at its own ragged length ``L_b`` (RoPE already applied to keys).
      scale: softmax scale (the model's ``attn_scale``), passed straight to the SDPA.
      n_rep: GQA repeat factor (``n_heads // n_kv``); keys/values are repeated to the ``H`` query
        heads exactly as the per-stream path does (``mx.repeat(..., axis=1)``), so the batched output
        is the same code on the same tensors.

    Row ``b`` of the result equals a single-stream
    ``mx.fast.scaled_dot_product_attention(q_b, repeat(k_b), repeat(v_b), scale, mask=None)`` over
    exactly ``L_b`` keys (the padded tail is masked to zero weight), up to SDPA reduction-order ULPs.
    """
    b = len(q_list)
    if b == 0:
        raise ValueError("batched_decode_attention: empty stream list")
    if len(k_list) != b or len(v_list) != b:
        raise ValueError(
            f"batched_decode_attention: ragged list lengths q={b} k={len(k_list)} v={len(v_list)}")

    def _drop_batch(a: mx.array, ndim: int) -> mx.array:
        """Strip a leading singleton batch axis if the caller passed ``[1, ...]``."""
        return a[0] if a.ndim == ndim + 1 and a.shape[0] == 1 else a

    q_list = [_drop_batch(q, 3) for q in q_list]           # -> [H, 1, D]
    k_list = [_drop_batch(k, 3) for k in k_list]           # -> [n_kv, L_b, D]
    v_list = [_drop_batch(v, 3) for v in v_list]

    lengths = [int(k.shape[1]) for k in k_list]
    l_max = max(lengths)
    n_kv = int(k_list[0].shape[0])
    d = int(k_list[0].shape[-1])
    _guard_padded_kv(b, n_kv, l_max, d, _itemsize(k_list[0]))

    q = mx.stack(q_list, axis=0)                           # [B, H, 1, D]
    k = mx.stack([_pad_seq(kk, l_max) for kk in k_list], axis=0)   # [B, n_kv, L_max, D]
    v = mx.stack([_pad_seq(vv, l_max) for vv in v_list], axis=0)
    if n_rep > 1:                                          # GQA: kv head -> its query group
        k = mx.repeat(k, n_rep, axis=1)                    # [B, H, L_max, D]
        v = mx.repeat(v, n_rep, axis=1)

    mask = decode_pad_mask(lengths, l_max, q.dtype)        # [B, 1, 1, L_max]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
