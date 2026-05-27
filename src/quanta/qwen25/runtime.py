"""Resident Qwen2.5-14B-Instruct-1M model — packed-weight path via ``mx.quantized_matmul``.

The packed path holds the artifact's affine-quantized matmul weights *in their on-disk packed
layout* and runs them through ``mx.quantized_matmul`` directly — dequantization happens inside
the kernel, never materializing a bf16 weight tensor. Resident footprint matches the artifact:

* attention int8 g64:  ~67 MB/layer × 48  ≈ 3.2 GB
* FFN int4 g64:        ~119 MB/layer × 48 ≈ 5.7 GB
* embed + lm_head bf16:                   ≈ 3.1 GB
* norms / biases bf16:                     ~negligible
* **Total ≈ 12 GB resident** (vs ~28 GB for the bf16-dequantized parity path).

The bf16 path is kept under ``packed=False`` as the parity reference / fallback. Both paths
share :class:`~quanta.qwen25.decode.Qwen25Cache`, the same RoPE/SDPA, and the same
:class:`Qwen25Cache` state machine — only the matmul kernel differs.

API mirrors the other resident models in this repo so the oMLX shim wires it uniformly:

* ``__init__(art_dir, *, packed=True, n_layers=None, quantized_kv=False)`` — load packed by default.
* ``__call__(token_ids, *, cache=None, seq_hint=None)`` → ``[B, T, vocab]`` logits.
* ``.cfg`` — :class:`~quanta.qwen25.config.Qwen25Config` (DCA fields, eos, vocab, …).
* ``new_cache(quantized=…)`` — fresh :class:`~quanta.qwen25.decode.Qwen25Cache` per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.qwen25.artifact import Qwen25Artifact
from quanta.qwen25.config import Qwen25Config
from quanta.qwen25.decode import Qwen25Cache
from quanta.qwen25.model import Qwen25Model

# Mirrors the bake's per-tensor-kind bit width. If the bake's policy changes, this dispatch
# must be updated in lock-step — read off ``art.manifest`` to verify at load time (rule-6).
_ATTN_BITS = 8
_MLP_BITS = 4
_GROUP_SIZE = 64


@dataclass(frozen=False)
class _PackedLayer:
    """All per-layer tensors a packed forward needs, kept in artifact layout (no dequant).

    Attention has Qwen2's q/k/v additive biases (``*_bias``, bf16) **separate** from the affine
    quant zero-point biases (``*_wbias``, the per-group scalar each ``mx.dequantize`` group needs).
    ``o_proj`` has no additive bias. FFN projections have no additive biases.
    """

    input_norm: mx.array            # [H] bf16 RMSNorm weight
    post_norm: mx.array             # [H] bf16 RMSNorm weight
    # Attention — int8 g64
    q_packed: mx.array
    q_scale: mx.array
    q_wbias: mx.array
    q_bias: mx.array | None
    k_packed: mx.array
    k_scale: mx.array
    k_wbias: mx.array
    k_bias: mx.array | None
    v_packed: mx.array
    v_scale: mx.array
    v_wbias: mx.array
    v_bias: mx.array | None
    o_packed: mx.array
    o_scale: mx.array
    o_wbias: mx.array
    # FFN — int4 g64
    gate_packed: mx.array
    gate_scale: mx.array
    gate_wbias: mx.array
    up_packed: mx.array
    up_scale: mx.array
    up_wbias: mx.array
    down_packed: mx.array
    down_scale: mx.array
    down_wbias: mx.array


def _load_quant_triplet(art: Qwen25Artifact, base: str) -> tuple[mx.array, mx.array, mx.array]:
    """Load a packed affine weight's three siblings: ``.weight_packed`` / ``.weight_scale`` /
    ``.weight_bias`` — verbatim from the artifact, no dequant."""
    return (art.raw(base),
            art.get(base + ".weight_scale"),
            art.get(base + ".weight_bias"))


def _load_packed_layer(art: Qwen25Artifact, cfg: Qwen25Config, i: int) -> _PackedLayer:
    """Stream layer ``i``'s packed tensors out of the artifact (rule-8 — one layer resident)."""
    p = f"model.layers.{i}."
    out: dict = {
        "input_norm": art.read(p + "input_layernorm.weight"),
        "post_norm": art.read(p + "post_attention_layernorm.weight"),
    }
    for name, key, has_add_bias in (
        ("q", "self_attn.q_proj", cfg.attention_bias),
        ("k", "self_attn.k_proj", cfg.attention_bias),
        ("v", "self_attn.v_proj", cfg.attention_bias),
        ("o", "self_attn.o_proj", False),
    ):
        packed, scale, wbias = _load_quant_triplet(art, p + key)
        out[f"{name}_packed"] = packed
        out[f"{name}_scale"] = scale
        out[f"{name}_wbias"] = wbias
        if name in ("q", "k", "v"):
            out[f"{name}_bias"] = (art.get(p + key + ".bias").astype(mx.bfloat16)
                                    if has_add_bias else None)
    for name, key in (("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")):
        packed, scale, wbias = _load_quant_triplet(art, p + key)
        out[f"{name}_packed"] = packed
        out[f"{name}_scale"] = scale
        out[f"{name}_wbias"] = wbias
    return _PackedLayer(**out)


def _qmm(x: mx.array, packed: mx.array, scale: mx.array, wbias: mx.array,
         bits: int, group_size: int = _GROUP_SIZE) -> mx.array:
    """Quantized matmul: ``x @ W.T`` where ``W`` is affine-packed (dequant fused in the kernel).

    Equivalent to ``x @ mx.dequantize(packed, scale, wbias, group_size, bits).T`` but never
    materializes the bf16 weight — that's the entire point of the packed runtime.
    """
    return mx.quantized_matmul(x, packed, scale, wbias,
                                transpose=True, group_size=group_size, bits=bits)


def _rmsnorm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """Weighted RMSNorm over the last dim, computed in fp32 (matches ``nn.RMSNorm``)."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (w.astype(mx.float32) * xf).astype(x.dtype)


def _packed_attention(x: mx.array, layer: _PackedLayer, cfg: Qwen25Config,
                      cache, *, abs_pos_start: int, dca_at_decode: bool,
                      use_fast: bool = True) -> mx.array:
    """GQA attention forward (Qwen2 form: QKV biases ON, no QK-norm, full RoPE) over packed weights.

    Two modes — switched by ``dca_at_decode``:

    * **Standard (prefill / short-context decode).** Q and K are both rotated with **intra-chunk
      positions** (``abs_pos_start % chunk_size``); the cache stores intra-rotated K, so the K
      already in the cache is in the same frame. When DCA is configured (``cfg.use_dca``), the
      attention is sliced to the **same-chunk** portion of the cache only (``[chunk_q * chunk_size,
      (chunk_q+1) * chunk_size)``), so a chunk-N prefill never attends to chunks 0..N-1 during
      prefill — that long-range integration is deferred to the DCA-at-decode path below. When DCA
      is OFF (``cfg.use_dca = False``), the full cache is in chunk 0 and the slice is a no-op.

    * **DCA at decode (``dca_at_decode=True``).** T=1 and the cache spans multiple chunks. Q gets
      **two** rotations (intra + successor); scores are computed for both and merged per
      cache position by chunk regime:

        - cache-pos in **same chunk** as Q  → use ``scores_intra``
        - cache-pos in an **earlier chunk** → use ``scores_succ``

      This is the Qwen2.5-1M training-free DCA at the decode step — gives the summary generator
      access to the full 1M cache with the trained-window-bounded relative positions the model
      learned. Memory: ``O(T_q × N_cache)`` score matrix — fine for T_q=1 even at N_cache=1M
      (~4MB per head per layer; one layer's worth at a time).
    """
    b, t, _ = x.shape
    q = _qmm(x, layer.q_packed, layer.q_scale, layer.q_wbias, _ATTN_BITS)
    k = _qmm(x, layer.k_packed, layer.k_scale, layer.k_wbias, _ATTN_BITS)
    v = _qmm(x, layer.v_packed, layer.v_scale, layer.v_wbias, _ATTN_BITS)
    if layer.q_bias is not None:                        # Qwen2 q/k/v additive biases
        q = q + layer.q_bias
        k = k + layer.k_bias
        v = v + layer.v_bias

    q = q.reshape(b, t, cfg.num_attention_heads, cfg.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 2, 1, 3)

    # K always rotated with intra-chunk positions, so multi-chunk caches stay in the trained frame.
    intra_pos_start = (abs_pos_start % cfg.dca_chunk_size) if cfg.use_dca else abs_pos_start
    k_intra = mx.fast.rope(k, dims=cfg.head_dim, traditional=True, base=cfg.rope_theta,
                            scale=1.0, offset=intra_pos_start)
    if cache is not None:
        k_full, v_full = cache.update(k_intra, v)
    else:
        k_full, v_full = k_intra, v

    if not dca_at_decode:
        # Standard path: Q rotated with intra positions; attend only to same-chunk slice.
        q_intra = mx.fast.rope(q, dims=cfg.head_dim, traditional=True, base=cfg.rope_theta,
                                scale=1.0, offset=intra_pos_start)
        if cfg.use_dca:
            chunk_q = abs_pos_start // cfg.dca_chunk_size
            cs = cfg.dca_chunk_size
            chunk_start = chunk_q * cs
            chunk_end = min(k_full.shape[-2], (chunk_q + 1) * cs)
            k_slice = k_full[..., chunk_start:chunk_end, :]
            v_slice = v_full[..., chunk_start:chunk_end, :]
        else:
            k_slice, v_slice = k_full, v_full
        kr = mx.repeat(k_slice, cfg.n_rep, axis=1)      # GQA: kv head -> its query group
        vr = mx.repeat(v_slice, cfg.n_rep, axis=1)
        mask = "causal" if t > 1 else None
        out = mx.fast.scaled_dot_product_attention(q_intra, kr, vr, scale=cfg.attn_scale, mask=mask)
    else:
        # DCA decode path: dual Q rotation, regime select over the full cache.
        succ_pos_start = intra_pos_start + cfg.dca_chunk_size
        q_intra = mx.fast.rope(q, dims=cfg.head_dim, traditional=True, base=cfg.rope_theta,
                                scale=1.0, offset=intra_pos_start)
        q_succ = mx.fast.rope(q, dims=cfg.head_dim, traditional=True, base=cfg.rope_theta,
                               scale=1.0, offset=succ_pos_start)
        kr = mx.repeat(k_full, cfg.n_rep, axis=1)
        vr = mx.repeat(v_full, cfg.n_rep, axis=1)
        scores_intra = mx.matmul(q_intra, mx.swapaxes(kr, -1, -2)) * cfg.attn_scale
        scores_succ = mx.matmul(q_succ, mx.swapaxes(kr, -1, -2)) * cfg.attn_scale
        chunk_q = abs_pos_start // cfg.dca_chunk_size
        n_cache = kr.shape[-2]
        cache_chunks = mx.arange(n_cache) // cfg.dca_chunk_size
        same_chunk = (cache_chunks == chunk_q)
        scores = mx.where(same_chunk[None, None, None, :], scores_intra, scores_succ)
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = weights @ vr

    out = out.transpose(0, 2, 1, 3).reshape(b, t, cfg.q_dim)
    return _qmm(out, layer.o_packed, layer.o_scale, layer.o_wbias, _ATTN_BITS)


def _packed_ffn(x: mx.array, layer: _PackedLayer) -> mx.array:
    """SwiGLU FFN over packed weights: ``down(silu(gate(x)) * up(x))``."""
    gate = _qmm(x, layer.gate_packed, layer.gate_scale, layer.gate_wbias, _MLP_BITS)
    up = _qmm(x, layer.up_packed, layer.up_scale, layer.up_wbias, _MLP_BITS)
    intermediate = nn.silu(gate) * up
    return _qmm(intermediate, layer.down_packed, layer.down_scale, layer.down_wbias, _MLP_BITS)


class _PackedModel:
    """Packed-weight resident forward — never materializes a bf16 matmul weight."""

    def __init__(self, art: Qwen25Artifact, cfg: Qwen25Config, *, n_layers: int | None = None):
        self.cfg = cfg
        self.n_layers = cfg.num_hidden_layers if n_layers is None else n_layers
        self.embed = art.embed()                                            # bf16 [V, H]
        self.final_norm = art.final_norm()                                  # bf16 [H]
        self.lm_head = art.lm_head() if not cfg.tie_word_embeddings else None
        self.layers: list[_PackedLayer] = []
        for i in range(self.n_layers):
            self.layers.append(_load_packed_layer(art, cfg, i))
            art.release()                                                   # drop shard handles (rule-8)
            mx.clear_cache()

    def __call__(self, token_ids: mx.array, *, caches=None, use_fast: bool = True,
                 abs_pos_start: int | None = None) -> mx.array:
        """Forward ``[B, T]`` token ids → ``[B, T, vocab]`` logits.

        ``abs_pos_start`` is the absolute position of ``token_ids[..., 0]`` in the full sequence.
        Defaults to ``caches[0].offset`` (= where the new tokens land in the cache), but the caller
        can override for chunked DCA prefill — e.g. ``abs_pos_start = chunk_idx * chunk_size`` so the
        intra-chunk RoPE offset resets to 0 at each chunk boundary, even though ``cache.offset`` has
        grown to ``chunk_idx * chunk_size``.

        DCA is auto-engaged at decode (T=1) once ``abs_pos_start`` crosses into chunk 1+ — the cache
        already holds intra-rotated K from prior chunks (via this same code path), so the DCA
        decode score-mix is exactly correct.
        """
        x = mx.take(self.embed, token_ids, axis=0)                          # [B, T, H]
        t = x.shape[1]
        if abs_pos_start is None:
            if caches is not None and caches[0] is not None:
                abs_pos_start = caches[0].offset
            else:
                abs_pos_start = 0
        dca_at_decode = (self.cfg.use_dca and t == 1
                         and abs_pos_start >= self.cfg.dca_chunk_size)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            # Attention residual
            n = _rmsnorm(x, layer.input_norm, self.cfg.norm_eps)
            x = x + _packed_attention(n, layer, self.cfg, cache,
                                       abs_pos_start=abs_pos_start,
                                       dca_at_decode=dca_at_decode,
                                       use_fast=use_fast)
            # FFN residual
            n = _rmsnorm(x, layer.post_norm, self.cfg.norm_eps)
            x = x + _packed_ffn(n, layer)
        x = _rmsnorm(x, self.final_norm, self.cfg.norm_eps)
        head = self.embed if self.cfg.tie_word_embeddings else self.lm_head
        return x @ head.T


class Qwen25ResidentModel:
    """Bf16-resident or packed-resident Qwen2.5-14B-Instruct-1M, loaded from a baked quanta artifact.

    ``packed=True`` (default) uses ``mx.quantized_matmul`` and holds the matmul weights in their
    on-disk packed layout (~12 GB resident). ``packed=False`` dequantizes to bf16 once at load
    (~28 GB resident) and runs through ``nn.Linear`` — the parity reference path.
    """

    def __init__(self, art_dir: str | Path, *, packed: bool = True, n_layers: int | None = None,
                 quantized_kv: bool = False, kv_group_size: int = 64) -> None:
        self.art = Qwen25Artifact(art_dir)
        self.cfg: Qwen25Config = self.art.cfg
        self.quantized_kv = quantized_kv
        self.kv_group_size = kv_group_size
        self.packed = packed

        if packed:
            self._model = _PackedModel(self.art, self.cfg, n_layers=n_layers)
        else:
            m = Qwen25Model(self.cfg, n_layers=n_layers)
            m.load_from(self.art)
            self._model = m
        self.art.release()
        mx.clear_cache()

    @property
    def n_layers(self) -> int:
        return self._model.n_layers

    # ``num_layers`` is the canonical attribute name the oMLX shim (RuntimeLike Protocol) reads.
    @property
    def num_layers(self) -> int:
        return self._model.n_layers

    def new_cache(self, *, quantized: bool | None = None,
                  group_size: int | None = None) -> Qwen25Cache:
        """Allocate a fresh decode cache. Defaults follow the runtime's construction flags."""
        return Qwen25Cache(
            self.cfg,
            quantized=self.quantized_kv if quantized is None else quantized,
            group_size=self.kv_group_size if group_size is None else group_size,
        )

    # Alias for the oMLX shim's ``_SingleTokenStepper``, which calls ``runtime.make_caches()``
    # uniformly across model classes (Qwen3.5 / GLM / MiniMax all expose this name).
    def make_caches(self) -> Qwen25Cache:
        return self.new_cache()

    def __call__(self, token_ids: mx.array, *,
                 cache: Qwen25Cache | None = None,
                 caches: Qwen25Cache | list | None = None,
                 offset: int | None = None,
                 use_fast: bool = True) -> mx.array:
        """Forward ``[B, T]`` token ids → ``[B, T, vocab]`` logits.

        Cache kwargs (accepts both forms — the oMLX ``_SingleTokenStepper`` passes ``caches=``):

        * ``cache=Qwen25Cache``     — quanta-native singular name.
        * ``caches=Qwen25Cache``    — shim-compat alias (a single Qwen25Cache, *not* a list).
        * ``caches=[KVCache, ...]`` — a raw per-layer list (rare; if the caller already has one).

        ``offset`` (shim-compat): absolute position of the first new token. When set, it overrides
        ``cache.offset`` for DCA RoPE-offset computation — used for chunked DCA prefill where
        successive chunks each pass ``offset = chunk_idx * chunk_size`` so the intra-chunk RoPE
        offset is the chunk-local position rather than the absolute one.
        """
        # Normalize input shape: the oMLX shim's ``_SingleTokenStepper`` passes a 1-D ``[T]`` array
        # (one or more tokens to ingest at the running offset); the in-repo ``generate`` already
        # 2-D-ifies. Either way, the downstream forward expects ``[B, T]`` — promote to 2-D here so
        # both call sites share one code path.
        if token_ids.ndim == 1:
            token_ids = token_ids[None]

        # Unify cache kwargs.
        cache_obj = cache if cache is not None else caches
        if cache_obj is None:
            cache_list: list | None = None
        elif isinstance(cache_obj, list):
            cache_list = cache_obj
        else:
            cache_list = cache_obj.as_list()

        # abs_pos_start: explicit ``offset`` wins, else fall back to cache.offset, else 0.
        if offset is not None:
            abs_pos_start = int(offset)
        elif cache_obj is not None and not isinstance(cache_obj, list):
            abs_pos_start = cache_obj.offset
        elif cache_list is not None and cache_list and cache_list[0] is not None:
            abs_pos_start = cache_list[0].offset
        else:
            abs_pos_start = 0

        return self._model(token_ids, caches=cache_list, use_fast=use_fast,
                            abs_pos_start=abs_pos_start)
