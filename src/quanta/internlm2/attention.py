"""GQA attention for InternLM2.5-7B-Chat-1M, MLX-native.

InternLM2 (vs Qwen2 / Qwen3 / Qwen3-Next) specifics — every divergence from the Qwen2.5 attention
path :mod:`quanta.qwen25.attention` mirrors:

* **NO biases.** Every projection is ``nn.Linear(bias=False)`` — InternLM2 sets ``config.bias=False``.
* **No QK-norm, no output gate.** Like Qwen2.
* **Llama-style ``rotate_half`` RoPE** (not Qwen2's interleaved-pair form). MLX exposes this as
  ``mx.fast.rope(..., traditional=False)`` — the default — which splits the head_dim in half and
  rotates ``(x[..., :d/2], x[..., d/2:])`` by ``(cos, sin)``.
* **Dynamic-NTK long-context.** Below ``cfg.max_position_embeddings`` (262144) the base is
  ``cfg.rope_theta=5e7``; above it the InternLM2 NTK formula rescales the base per the *current*
  total sequence length:

      ``base · ((factor · seq_len / max_pos) − (factor − 1)) ^ (dim / (dim − 2))``

  The base is recomputed *per forward pass* (it depends only on the running seq_len, not on
  per-position frequencies), then handed to ``mx.fast.rope(..., base=…)``. This is exactly the
  HF ``InternLM2DynamicNTKScalingRotaryEmbedding`` behavior at ``modeling_internlm2.py:144-159``.

Two equivalent paths (the standard quanta gate):

* fast (default): ``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention`` (tiled — never
  materializes a T×T score matrix, memory-safe at the 1M context).
* naive: explicit RoPE + manual softmax — the short-sequence parity reference.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
from quanta.internlm2.config import InternLM2Config


class KVCache:
    """Plain GQA KV cache: ``[B, n_kv, S, head_dim]`` k/v growing along the seq axis.

    Two storage modes (mirror :class:`quanta.qwen25.attention.KVCache`):

    * ``quantized=False``: bf16 verbatim — parity reference / short-context decode path.
    * ``quantized=True`` (default for InternLM2.5-1M): per-token, per-group affine int-``bits``
      over ``head_dim`` via :mod:`quanta.cache_quant`. ``update`` dequantizes the full cache for
      the SDPA return so the attention path is unchanged. ``bits`` is wired through to
      ``mx.quantize`` (MLX supports 2/3/4/6/8) — InternLM2.5-7B defaults to **int8 g64** (~8.5
      bpp vs bf16's 16 bpp): conservative, matching the safer default chosen for the 7B Qwen2.5
      drop-in. 7B @ 1M ctx, kv_heads=8, head_dim=128, layers=32 → int8 ≈ 32 GB.
    """

    def __init__(self, *, quantized: bool = False, group_size: int = 64,
                 bits: int = 8) -> None:
        self.quantized = quantized
        self.group_size = group_size
        self.bits = bits
        # bf16 mode
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        # int<bits> mode (codes + per-group scales/biases)
        self.k_q: mx.array | None = None
        self.k_s: mx.array | None = None
        self.k_b: mx.array | None = None
        self.v_q: mx.array | None = None
        self.v_s: mx.array | None = None
        self.v_b: mx.array | None = None

    @property
    def offset(self) -> int:
        if self.quantized:
            return 0 if self.k_q is None else self.k_q.shape[2]
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if not self.quantized:
            if self.k is None:
                self.k, self.v = k, v
            else:
                self.k = mx.concatenate([self.k, k], axis=2)
                self.v = mx.concatenate([self.v, v], axis=2)
            return self.k, self.v
        k_qn, k_sn, k_bn = quantize_last_axis(k, self.group_size, bits=self.bits)
        v_qn, v_sn, v_bn = quantize_last_axis(v, self.group_size, bits=self.bits)
        if self.k_q is None:
            self.k_q, self.k_s, self.k_b = k_qn, k_sn, k_bn
            self.v_q, self.v_s, self.v_b = v_qn, v_sn, v_bn
        else:
            self.k_q = mx.concatenate([self.k_q, k_qn], axis=2)
            self.k_s = mx.concatenate([self.k_s, k_sn], axis=2)
            self.k_b = mx.concatenate([self.k_b, k_bn], axis=2)
            self.v_q = mx.concatenate([self.v_q, v_qn], axis=2)
            self.v_s = mx.concatenate([self.v_s, v_sn], axis=2)
            self.v_b = mx.concatenate([self.v_b, v_bn], axis=2)
        k_full = dequantize_last_axis(self.k_q, self.k_s, self.k_b, self.group_size,
                                       dtype=k.dtype, bits=self.bits)
        v_full = dequantize_last_axis(self.v_q, self.v_s, self.v_b, self.group_size,
                                       dtype=v.dtype, bits=self.bits)
        return k_full, v_full


def _rope_fast(x: mx.array, base: float, offset: int) -> mx.array:
    """Llama-style ``rotate_half`` RoPE on the full last dim via ``mx.fast.rope``.

    ``traditional=False`` selects the rotate-half variant InternLM2 uses (vs Qwen2's
    interleaved-pair form). ``base`` is the *effective* base — caller passes the NTK-scaled
    value when ``seq_len > max_position_embeddings``.
    """
    return mx.fast.rope(x, dims=x.shape[-1], traditional=False, base=base, scale=1.0, offset=offset)


def _rope_explicit(x: mx.array, base: float, offset: int) -> mx.array:
    """Explicit ``rotate_half`` RoPE on the full last dim (parity reference for :func:`_rope_fast`).

    ``x`` ``[B, H, T, D]`` with even ``D``. Computes ``inv_freq = 1 / base ** (idx / D)`` with
    ``idx`` even-indexed (0, 2, …, D-2); ``cos/sin`` are duplicated across the two halves of D
    (``emb = cat(freqs, freqs, dim=-1)`` per HF), and the rotation is

        ``q_embed = q * cos + rotate_half(q) * sin``

    with ``rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2], dim=-1)``.
    """
    b, h, t, d = x.shape
    half = d // 2
    idx = mx.arange(0, d, 2, dtype=mx.float32)
    inv_freq = 1.0 / (base ** (idx / d))                                   # [D/2]
    pos = (mx.arange(t, dtype=mx.float32) + offset)[:, None]               # [T, 1]
    ang = pos * inv_freq[None, :]                                          # [T, D/2]
    cos_half = mx.cos(ang)                                                 # [T, D/2]
    sin_half = mx.sin(ang)                                                 # [T, D/2]
    cos = mx.concatenate([cos_half, cos_half], axis=-1)[None, None]        # [1, 1, T, D]
    sin = mx.concatenate([sin_half, sin_half], axis=-1)[None, None]
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def batched_rope_fast(x: mx.array, offsets: list[int], bases: list[float]) -> mx.array:
    """Per-stream RoPE for ``B`` decode streams at **heterogeneous** absolute positions — bit-identical
    to the looped single-stream :func:`_rope_fast`.

    ``x`` ``[B, H, T, D]`` (decode ⇒ ``T == 1``); ``offsets[b]`` is the absolute position of
    ``x[b, :, 0]`` for stream ``b`` and ``bases[b]`` its dynamic-NTK base (``cfg.ntk_base(seq_len_b)``,
    per-stream because streams sit at different lengths). :func:`mx.fast.rope` takes a *scalar*
    ``offset``/``base``, so the heterogeneous-offset batch is applied stream-by-stream over the
    **bounded** stream axis (``B ≤ max_batch`` — a coarse IO-level loop, rule 3; never over
    tokens/heads/hidden) and concatenated back to ``[B, H, T, D]``.

    Critically this calls the SAME ``mx.fast.rope`` kernel the per-stream decode reference
    (:func:`_rope_fast`) uses, so the batched decode path is **bit-exact** with the loop at every dtype.
    A hand-rolled fp32-then-cast rotate-half form (the obvious "vectorized" shortcut) matches in fp32 but
    its bf16 ULP drift on large-magnitude q **compounds across 32 layers and flips greedy tokens** on the
    real model — caught by ``parity/internlm2_batched_bench`` (model-free gates with tiny random init and
    ≤2 layers do not surface it).
    """
    rotated = [mx.fast.rope(x[s:s + 1], dims=x.shape[-1], traditional=False,
                            base=float(bases[s]), scale=1.0, offset=int(offsets[s]))
               for s in range(x.shape[0])]                                # B × [1, H, T, D]
    return mx.concatenate(rotated, axis=0)                                # [B, H, T, D]


def _causal_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal additive mask (query j at abs pos kv_len-q_len+j)."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


class InternLM2Attention(nn.Module):
    """GQA + Llama-style RoPE + dynamic-NTK attention for InternLM2.5-7B-Chat-1M.

    The four projections (``wq``/``wk``/``wv``/``wo``) live as separate ``nn.Linear`` modules —
    the source's fused ``wqkv`` was deinterleaved at load time by
    :func:`quanta.internlm2.loader._split_wqkv`, so the hot path has no fused-qkv branch.
    """

    def __init__(self, cfg: InternLM2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads          # 32
        self.nkv = cfg.num_key_value_heads         # 8
        self.hd = cfg.head_dim                     # 128
        self.rep = cfg.n_rep                       # 4
        self.scale = cfg.attn_scale
        bias = bool(cfg.attention_bias)            # False for InternLM2.5
        self.wq = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=bias)
        self.wk = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=bias)
        self.wv = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=bias)
        self.wo = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=bias)

    def _project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """q,k,v -> [B, H, T, D] / [B, n_kv, T, D]."""
        b, t, _ = x.shape
        q = self.wq(x).reshape(b, t, self.nh, self.hd)
        k = self.wk(x).reshape(b, t, self.nkv, self.hd)
        v = self.wv(x).reshape(b, t, self.nkv, self.hd)
        q = mx.transpose(q, (0, 2, 1, 3))                       # [B, H, T, D]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v

    def __call__(self, x: mx.array, *, cache: KVCache | None = None, use_fast: bool = True,
                 seq_hint: int | None = None) -> mx.array:
        b, t, _ = x.shape
        offset = cache.offset if cache is not None else 0
        seq_len = seq_hint if seq_hint is not None else (offset + t)
        # NTK base depends on the *current* total seq len, NOT per-position — recompute per fwd.
        base = self.cfg.ntk_base(seq_len)

        q, k, v = self._project(x)
        q = (_rope_fast if use_fast else _rope_explicit)(q, base, offset)
        k = (_rope_fast if use_fast else _rope_explicit)(k, base, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]
        kr = mx.repeat(k, self.rep, axis=1)                     # GQA: kv head -> its query group
        vr = mx.repeat(v, self.rep, axis=1)
        if use_fast:
            mask = "causal" if t > 1 else None
            out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale, mask=mask)
        else:
            scores = (q @ mx.swapaxes(kr, -1, -2)) * self.scale + _causal_mask(t, kv_len, q.dtype)
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = w @ vr
        out = mx.transpose(out, (0, 2, 1, 3))                   # [B, T, H, D]
        out = out.reshape(b, t, self.nh * self.hd)
        return self.wo(out)
