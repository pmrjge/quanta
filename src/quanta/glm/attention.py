"""Multi-head Latent Attention (MLA) for GLM-5.1 (``glm_moe_dsa``), MLX-native.

GLM-5.1 is DeepSeek-V3.2-style MLA, so this mirrors :class:`quanta.modeling.attention.MLAAttention`
(the Kimi MLA) and :mod:`quanta.dsv4.attention`, grounded in :mod:`quanta.glm.config` /
:mod:`quanta.glm.loader`:

* **q**: ``q_a_proj -> q_a_layernorm (weighted RMSNorm over ``q_lora_rank``) -> q_b_proj`` reshaped to
  ``[B,T,H,qk_head_dim]``, split into ``q_nope`` (``qk_nope_head_dim=192``) and ``q_pe``
  (``qk_rope_head_dim=64``).
* **kv**: ``kv_a_proj_with_mqa`` → ``c_kv`` (``kv_lora_rank=512``) ++ ``k_pe`` (``qk_rope_head_dim=64``,
  a *single* shared MQA key-rope head); ``c_kv = kv_a_layernorm(c_kv)``; ``kv_b_proj(c_kv)`` →
  ``[B,T,H, qk_nope_head_dim + v_head_dim]`` split into ``k_nope`` and ``value`` (``v_head_dim=256``).
* **RoPE** is **partial** (only the last ``qk_rope_head_dim`` dims of ``q_pe``/``k_pe``) and
  **interleaved** (``rope_interleave`` — consecutive complex pairs), with **no YaRN**
  (``rope_type="default"``); ``softmax_scale = qk_head_dim**-0.5`` (no ``mscale``).
* **attention**: causal ``softmax(QKᵀ·scale + mask)V`` over ``query = [q_nope|q_pe]`` and
  ``key = [k_nope|k_pe]`` (``k_pe`` broadcast across heads); output ``[B,T,H,v_head_dim]`` → ``o_proj``.

Two paths share identical projections / RoPE / scale and are kept output-equivalent (rule 4):

* ``naive`` (default): explicit interleaved-complex RoPE + explicit masked softmax — the parity
  reference, fine only at small T.
* ``fast`` (``use_fast=True``): ``mx.fast.rope(traditional=True)`` + ``mx.fast.scaled_dot_product_attention``
  (value zero-padded ``v_head_dim``→``qk_head_dim`` for a uniform SDPA head dim, then sliced back).

The decode stepper (:meth:`MLAAttention.step`) is incremental and numerically equal to the prefill path
evaluated at the same absolute positions (the #83 ``incremental == prefill`` gate); it threads a small
layer KV cache (a ``.update(c_kv, k_pe) -> (full_c_kv, full_k_pe)`` protocol, e.g.
:class:`quanta.glm.model.GLMDecodeCache`'s per-layer KV cache or the parity test's stub). All of the
above is gated model-free (tiny random weights) in ``parity/glm_forward_test.py`` /
``parity/glm_attn_test.py``; the formula-vs-real-model check is the deferred torch/ppl oracle.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.glm.config import GLMConfig

# DeepSeek-V3 q_a / kv_a sub-layernorm eps (HF ``DeepseekV3RMSNorm`` default), independent of
# ``rms_norm_eps`` which is the block input/post-attn norm eps.
_SUBNORM_EPS = 1e-6


# --- RoPE (interleaved-complex, partial, no YaRN) ----------------------------
def build_inv_freq(cfg: GLMConfig) -> mx.array:
    """Per-pair inverse frequencies for the rotated sub-head, shape ``[qk_rope_head_dim // 2]`` (fp32).

    Plain (non-YaRN) RoPE: ``1 / theta**(2i/rope_dim)`` — ``rope_type="default"`` so there is no
    frequency interpolation/scaling."""
    dim = cfg.qk_rope_head_dim
    idx = mx.arange(0, dim, 2, dtype=mx.float32)
    return 1.0 / (cfg.rope_theta ** (idx / dim))


def rope_cos_sin(cfg: GLMConfig, positions: mx.array) -> tuple[mx.array, mx.array]:
    """``(cos, sin)`` tables of shape ``[T, qk_rope_head_dim // 2]`` (fp32) for the given absolute
    ``positions`` ``[T]`` (interleaved layout: one angle per consecutive complex pair)."""
    inv = build_inv_freq(cfg)
    ang = positions.astype(mx.float32)[:, None] * inv[None, :]   # [T, rope/2]
    return mx.cos(ang), mx.sin(ang)


def _apply_rope_interleaved(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the last dim of ``x`` (size ``rd``) as ``rd/2`` *interleaved* complex pairs
    ``(x0,x1),(x2,x3),…``. ``x``: ``[B,H,T,rd]``; ``cos``/``sin``: ``[T, rd/2]``."""
    *lead, rd = x.shape
    xr = x.reshape(*lead, rd // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    c, s = cos[None, None], sin[None, None]                      # broadcast over [B,H]
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return mx.stack([o0, o1], axis=-1).reshape(*lead, rd)


def rope_naive(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Explicit interleaved RoPE over a full ``[B,H,T,rope_dim]`` tensor (the rotated sub-head)."""
    return _apply_rope_interleaved(x, cos, sin)


def rope_fast(x: mx.array, inv_freq: mx.array, offset: int = 0) -> mx.array:
    """``mx.fast.rope`` (traditional/interleaved) over ``[B,H,T,rope_dim]`` — verified
    output-equivalent to :func:`rope_naive` (mlx ``freqs`` is the period, so pass ``1/inv_freq``)."""
    return mx.fast.rope(x, dims=x.shape[-1], traditional=True, base=None, scale=1.0,
                        offset=offset, freqs=1.0 / inv_freq)


def _causal_additive_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal mask: query ``j`` (abs pos ``kv_len-q_len+j``) attends keys ``i`` ≤ that pos."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


class MLAAttention(nn.Module):
    """GLM-5.1 MLA block (low-rank q/kv, partial interleaved RoPE, optional DSA indexer mask)."""

    def __init__(self, cfg: GLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.nope = cfg.qk_nope_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.qk_head_dim = cfg.qk_head_dim
        self.scale = cfg.softmax_scale

        bias = cfg.attention_bias
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=bias)
        self.q_a_layernorm = nn.RMSNorm(cfg.q_lora_rank, eps=_SUBNORM_EPS)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(cfg.hidden_size, cfg.kv_lora_rank + self.rope, bias=bias)
        self.kv_a_layernorm = nn.RMSNorm(cfg.kv_lora_rank, eps=_SUBNORM_EPS)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, self.num_heads * (self.nope + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, cfg.hidden_size, bias=bias)

    # ---- shared projection ---------------------------------------------------
    def _project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Project ``x`` ``[B,T,dim]`` to ``(q_latent, q_nope, q_pe, c_kv, k_pe)`` — returns the
        q-latent (fed to the indexer), per-head ``q_nope``/``q_pe`` ``[B,H,T,*]`` and the shared MQA
        latent ``c_kv`` ``[B,T,kv_lora]`` / rope key ``k_pe`` ``[B,1,T,rope]`` (RoPE not yet applied)."""
        b, t, _ = x.shape
        h, nope, rope, kv_lora = self.num_heads, self.nope, self.rope, self.cfg.kv_lora_rank
        q_latent = self.q_a_layernorm(self.q_a_proj(x))                  # [B,T,q_lora]
        q = self.q_b_proj(q_latent).reshape(b, t, h, self.qk_head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))                               # [B,H,T,qk_head_dim]
        q_nope, q_pe = q[..., :nope], q[..., nope:]
        ckv = self.kv_a_proj_with_mqa(x)
        c_kv, k_pe = ckv[..., :kv_lora], ckv[..., kv_lora:]
        c_kv = self.kv_a_layernorm(c_kv)                                # [B,T,kv_lora]
        k_pe = mx.transpose(k_pe.reshape(b, t, 1, rope), (0, 2, 1, 3))  # [B,1,T,rope]
        return q_latent, q_nope, q_pe, c_kv, k_pe

    def _expand_kv(self, c_kv: mx.array) -> tuple[mx.array, mx.array]:
        """Up-project the latent to per-head ``(k_nope, value)`` ``[B,H,S,*]`` via ``kv_b_proj``."""
        b, s, _ = c_kv.shape
        h, nope, vhd = self.num_heads, self.nope, self.v_head_dim
        kv = self.kv_b_proj(c_kv).reshape(b, s, h, nope + vhd)
        kv = mx.transpose(kv, (0, 2, 1, 3))                            # [B,H,S,nope+vhd]
        return kv[..., :nope], kv[..., nope:]

    def _attend(self, q_nope: mx.array, q_pe: mx.array, c_kv: mx.array, k_pe: mx.array,
                mask: mx.array, use_fast: bool) -> mx.array:
        """Core SDPA: assemble query/key/value from the latent + roped rope-heads, attend under
        ``mask`` (additive ``[Tq,Tkv]`` or broadcastable), return ``[B,T,H*v_head_dim]``."""
        b, h, t, _ = q_nope.shape
        kv_len = c_kv.shape[1]
        k_nope, value = self._expand_kv(c_kv)
        k_pe = mx.broadcast_to(k_pe, (b, h, kv_len, self.rope))        # share the MQA rope key
        query = mx.concatenate([q_nope, q_pe], axis=-1)                # [B,H,T,qk_head_dim]
        key = mx.concatenate([k_nope, k_pe], axis=-1)                  # [B,H,S,qk_head_dim]
        if use_fast:
            vpad = mx.concatenate(
                [value, mx.zeros((b, h, kv_len, self.qk_head_dim - self.v_head_dim), value.dtype)], -1)
            # SDPA requires the additive mask to promote to the output dtype, so match query dtype
            # (a wider fp32 mask would raise on bf16 inputs); ±inf and 0 are exact in bf16.
            out = mx.fast.scaled_dot_product_attention(
                query, key, vpad, scale=self.scale, mask=mask.astype(query.dtype))
            out = out[..., :self.v_head_dim]
        else:
            scores = (query @ mx.swapaxes(key, -1, -2)) * self.scale   # [B,H,T,S]
            w = mx.softmax((scores + mask).astype(mx.float32), axis=-1).astype(query.dtype)
            out = w @ value                                            # [B,H,T,v_head_dim]
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, h * self.v_head_dim)
        return out

    def __call__(self, x: mx.array, positions: mx.array, *, use_fast: bool = False,
                 index_mask: mx.array | None = None) -> mx.array:
        """Full-sequence (prefill) MLA. ``x``: ``[B,T,dim]``; ``positions``: absolute pos ``[T]``.

        ``index_mask`` (optional, from the DSA indexer): an additive ``[B,T,Tkv]`` mask (0 keep /
        ``-inf`` drop) ANDed with the causal mask — when the indexer keeps all causal tokens this is a
        no-op and the result equals plain causal MLA (the #84 keep-all == dense gate). Returns
        ``[B,T,dim]``."""
        b, t, _ = x.shape
        _, q_nope, q_pe, c_kv, k_pe = self._project(x)
        inv = build_inv_freq(self.cfg)
        if use_fast:
            off = int(positions[0].item()) if t > 0 else 0
            q_pe = rope_fast(q_pe, inv, offset=off)
            k_pe = rope_fast(k_pe, inv, offset=off)
        else:
            cos, sin = rope_cos_sin(self.cfg, positions)
            cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
            q_pe = rope_naive(q_pe, cos, sin)
            k_pe = rope_naive(k_pe, cos, sin)
        mask = _causal_additive_mask(t, t, mx.float32)                 # [T,T]
        if index_mask is not None:                                     # [B,T,Tkv] additive
            mask = mask[None] + index_mask
            mask = mask[:, None]                                       # [B,1,T,Tkv] broadcast over heads
        out = self._attend(q_nope, q_pe, c_kv, k_pe, mask, use_fast)
        return self.o_proj(out)

    def step(self, x_t: mx.array, cache, offset: int, *, use_fast: bool = False,
             index_mask: mx.array | None = None) -> mx.array:
        """One decode token at absolute position ``offset`` — incremental, equal to the prefill path at
        that position. ``x_t``: ``[B,1,dim]``; ``cache``: a layer KV cache with
        ``update(c_kv, k_pe) -> (c_kv_all, k_pe_all)`` (append the new token, return the full streams).

        ``index_mask`` (optional, from the DSA indexer): an additive ``[B,1,S]`` mask (0 keep / ``-inf``
        drop) ANDed with the (trivial) single-query causal mask — keep-all ⇒ plain causal decode.
        Returns ``[B,1,dim]``."""
        _, q_nope, q_pe, c_kv, k_pe = self._project(x_t)
        inv = build_inv_freq(self.cfg)
        if use_fast:
            q_pe = rope_fast(q_pe, inv, offset=offset)
            k_pe = rope_fast(k_pe, inv, offset=offset)
        else:
            pos = mx.array([offset])
            cos, sin = rope_cos_sin(self.cfg, pos)
            cos, sin = cos.astype(x_t.dtype), sin.astype(x_t.dtype)
            q_pe = rope_naive(q_pe, cos, sin)
            k_pe = rope_naive(k_pe, cos, sin)
        c_kv, k_pe = cache.update(c_kv, k_pe)                          # append → full [B,S,*]
        kv_len = c_kv.shape[1]
        mask = _causal_additive_mask(1, kv_len, mx.float32)           # [1,S] (single query, all causal)
        if index_mask is not None:                                    # [B,1,S] additive
            mask = mask[None] + index_mask                            # [B,1,S]
            mask = mask[:, None]                                      # [B,1,1,S] broadcast over heads
        out = self._attend(q_nope, q_pe, c_kv, k_pe, mask, use_fast)
        return self.o_proj(out)
