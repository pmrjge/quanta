"""Resident InternLM2.5-7B-Chat-1M model — packed-weight path via ``mx.quantized_matmul``.

The packed path holds the artifact's affine-quantized matmul weights *in their on-disk packed
layout* and runs them through ``mx.quantized_matmul`` directly — dequantization happens inside
the kernel, never materializing a bf16 weight tensor. Resident footprint matches the artifact:

* attention int8 g64:  ~57 MB/layer × 32 ≈ 1.8 GB
* FFN int4 g64:        ~89 MB/layer × 32 ≈ 2.9 GB
* embed + output bf16:                    ≈ 1.5 GB
* norms bf16:                             ~negligible
* **Total ≈ 6.2 GB resident** (vs ~14 GB for the bf16-dequantized parity path).

The bf16 path is kept under ``packed=False`` as the parity reference / fallback. Both paths
share :class:`~quanta.internlm2.decode.InternLM2Cache`, the same Llama-style ``rotate_half``
RoPE + dynamic-NTK base scaling, and the same KV cache state machine — only the matmul kernel
differs.

API mirrors the other resident models in this repo so the oMLX shim wires it uniformly:

* ``__init__(art_dir, *, packed=True, n_layers=None, quantized_kv=True, kv_bits=8)`` — packed
  weights + int8 g64 KV (InternLM2.5-7B default; conservative — matches Qwen2.5-7B's safer
  default for the 1M context).
* ``__call__(token_ids, *, cache=None, offset=None, ...)`` → ``[B, T, vocab]`` logits.
* ``.cfg`` — :class:`~quanta.internlm2.config.InternLM2Config` (NTK fields, eos, vocab, …).
* ``new_cache(quantized=…)`` — fresh :class:`~quanta.internlm2.decode.InternLM2Cache` per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.internlm2.artifact import InternLM2Artifact
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.decode import InternLM2Cache
from quanta.internlm2.model import InternLM2Model

# Default bake policy — documentation + the model-free lock-step gate. The packed runtime no longer
# trusts these constants for decode: it reads each weight's actual ``(bits, group_size)`` from the
# artifact manifest in ``_load_packed_layer`` (rule-6 — the baked manifest is the single source of
# truth, so a differently-baked artifact, e.g. int8-everywhere, decodes correctly with no code change).
_ATTN_BITS = 8
_MLP_BITS = 4
_GROUP_SIZE = 64


@dataclass(frozen=False)
class _PackedLayer:
    """All per-layer tensors a packed forward needs, kept in artifact layout (no dequant).

    InternLM2.5 has no additive biases (``cfg.attention_bias=False`` and the FFN ``w*`` projections
    are bias-free too), so we only carry the per-affine-group zero-point biases that
    ``mx.dequantize`` needs (the ``*_wbias`` triplet sibling).
    """

    attn_norm: mx.array             # [H] bf16 RMSNorm weight (pre-attention)
    ffn_norm: mx.array              # [H] bf16 RMSNorm weight (pre-FFN)
    # Attention — int8 g64
    q_packed: mx.array
    q_scale: mx.array
    q_wbias: mx.array
    k_packed: mx.array
    k_scale: mx.array
    k_wbias: mx.array
    v_packed: mx.array
    v_scale: mx.array
    v_wbias: mx.array
    o_packed: mx.array
    o_scale: mx.array
    o_wbias: mx.array
    # FFN — int4 g64
    w1_packed: mx.array      # gate
    w1_scale: mx.array
    w1_wbias: mx.array
    w3_packed: mx.array      # up
    w3_scale: mx.array
    w3_wbias: mx.array
    w2_packed: mx.array      # down
    w2_scale: mx.array
    w2_wbias: mx.array
    # Per-kind affine quant params, read from the artifact manifest (NOT hardcoded — rule-6). All
    # attention projections share (attn_bits, attn_gs); all FFN projections share (mlp_bits, mlp_gs).
    attn_bits: int
    attn_gs: int
    mlp_bits: int
    mlp_gs: int


def _load_quant_triplet(art: InternLM2Artifact, base: str
                        ) -> tuple[mx.array, mx.array, mx.array, int, int]:
    """Load a packed affine weight's three siblings (``.weight_packed`` / ``.weight_scale`` /
    ``.weight_bias`` — verbatim, no dequant) plus its ``(bits, group_size)`` from the manifest.

    The decode width travels with the artifact: rule-6 — the baked manifest is the single source of
    truth, never a hardcoded width that could silently mis-decode a differently-baked artifact."""
    meta = art.manifest[base]
    return (art.raw(base),
            art.get(base + ".weight_scale"),
            art.get(base + ".weight_bias"),
            int(meta["bits"]), int(meta["group_size"]))


def _load_packed_layer(art: InternLM2Artifact, cfg: InternLM2Config, i: int) -> _PackedLayer:
    """Stream layer ``i``'s packed tensors out of the artifact (rule-8 — one layer resident)."""
    del cfg  # currently unused; here for symmetry with qwen25 / future bias variants
    p = f"model.layers.{i}."
    out: dict = {
        "attn_norm": art.read(p + "attention_norm.weight"),
        "ffn_norm": art.read(p + "ffn_norm.weight"),
    }
    attn_bg: tuple[int, int] | None = None
    for name, key in (
        ("q", "attention.wq"),
        ("k", "attention.wk"),
        ("v", "attention.wv"),
        ("o", "attention.wo"),
    ):
        packed, scale, wbias, bits, gs = _load_quant_triplet(art, p + key)
        out[f"{name}_packed"] = packed
        out[f"{name}_scale"] = scale
        out[f"{name}_wbias"] = wbias
        if attn_bg is None:
            attn_bg = (bits, gs)
        elif (bits, gs) != attn_bg:
            raise ValueError(f"{p}{key}: attn quant ({bits},{gs}) != {attn_bg} — non-uniform bake (rule-6)")
    # InternLM2 FFN naming: w1=gate, w3=up, w2=down. The runtime fields keep the same names.
    mlp_bg: tuple[int, int] | None = None
    for name, key in (
        ("w1", "feed_forward.w1"),
        ("w3", "feed_forward.w3"),
        ("w2", "feed_forward.w2"),
    ):
        packed, scale, wbias, bits, gs = _load_quant_triplet(art, p + key)
        out[f"{name}_packed"] = packed
        out[f"{name}_scale"] = scale
        out[f"{name}_wbias"] = wbias
        if mlp_bg is None:
            mlp_bg = (bits, gs)
        elif (bits, gs) != mlp_bg:
            raise ValueError(f"{p}{key}: mlp quant ({bits},{gs}) != {mlp_bg} — non-uniform bake (rule-6)")
    out["attn_bits"], out["attn_gs"] = attn_bg
    out["mlp_bits"], out["mlp_gs"] = mlp_bg
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


def _packed_attention(x: mx.array, layer: _PackedLayer, cfg: InternLM2Config,
                      cache, *, abs_pos_start: int, seq_len: int) -> mx.array:
    """GQA attention forward (Llama-style ``rotate_half`` + dynamic-NTK base) over packed weights.

    No biases (InternLM2.5 sets every ``bias=False``). The RoPE base is rescaled per the InternLM2
    dynamic-NTK formula whenever the *current total* sequence length exceeds
    ``cfg.max_position_embeddings``; the rescaled base depends only on ``seq_len`` (not on
    per-position frequencies), so a single ``mx.fast.rope`` call with the effective base handles
    the whole pass.
    """
    b, t, _ = x.shape
    q = _qmm(x, layer.q_packed, layer.q_scale, layer.q_wbias, layer.attn_bits, layer.attn_gs)
    k = _qmm(x, layer.k_packed, layer.k_scale, layer.k_wbias, layer.attn_bits, layer.attn_gs)
    v = _qmm(x, layer.v_packed, layer.v_scale, layer.v_wbias, layer.attn_bits, layer.attn_gs)

    q = q.reshape(b, t, cfg.num_attention_heads, cfg.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 2, 1, 3)

    base = cfg.ntk_base(seq_len)
    q = mx.fast.rope(q, dims=cfg.head_dim, traditional=False, base=base,
                      scale=1.0, offset=abs_pos_start)
    k = mx.fast.rope(k, dims=cfg.head_dim, traditional=False, base=base,
                      scale=1.0, offset=abs_pos_start)
    if cache is not None:
        k, v = cache.update(k, v)

    kr = mx.repeat(k, cfg.n_rep, axis=1)                     # GQA: kv head -> its query group
    vr = mx.repeat(v, cfg.n_rep, axis=1)
    mask = "causal" if t > 1 else None
    out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=cfg.attn_scale, mask=mask)

    out = out.transpose(0, 2, 1, 3).reshape(b, t, cfg.q_dim)
    return _qmm(out, layer.o_packed, layer.o_scale, layer.o_wbias, layer.attn_bits, layer.attn_gs)


def _packed_ffn(x: mx.array, layer: _PackedLayer) -> mx.array:
    """SwiGLU FFN over packed weights: ``w2(silu(w1(x)) * w3(x))`` — InternLM2 w1/w3/w2 naming."""
    gate = _qmm(x, layer.w1_packed, layer.w1_scale, layer.w1_wbias, layer.mlp_bits, layer.mlp_gs)
    up = _qmm(x, layer.w3_packed, layer.w3_scale, layer.w3_wbias, layer.mlp_bits, layer.mlp_gs)
    intermediate = nn.silu(gate) * up
    return _qmm(intermediate, layer.w2_packed, layer.w2_scale, layer.w2_wbias,
                layer.mlp_bits, layer.mlp_gs)


class _PackedModel:
    """Packed-weight resident forward — never materializes a bf16 matmul weight."""

    def __init__(self, art: InternLM2Artifact, cfg: InternLM2Config, *,
                 n_layers: int | None = None):
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

    def __call__(self, token_ids: mx.array, *, caches=None,
                 abs_pos_start: int | None = None,
                 last_only: bool = False) -> mx.array:
        """Forward ``[B, T]`` token ids → ``[B, T, vocab]`` (or ``[B, 1, vocab]`` if ``last_only``).

        ``abs_pos_start`` is the absolute position of ``token_ids[..., 0]`` in the full sequence.
        Defaults to ``caches[0].offset`` (= where the new tokens land in the cache), but the caller
        can override for chunked prefill — e.g. ``abs_pos_start = chunk_idx * chunk_size`` so RoPE
        sees the correct absolute position even though the cache grew incrementally.

        ``last_only`` slices the residual stream to its **last position only** before the output
        matmul — essential for long-prompt prefill: at T=262144 with vocab=92544, the full
        ``[B, T, V]`` materialization is ~48 GB transient, vs ~180 KB for the last row alone. The
        caller (generate-loop / chunked prefill) only ever needs the last position's logits during
        prefill anyway, so this is purely a memory-safety win, not a semantic change.

        Dynamic-NTK is auto-engaged when ``cache.offset + T > cfg.max_position_embeddings``: the
        RoPE base is rescaled per InternLM2's formula and applied uniformly across every layer.
        """
        x = mx.take(self.embed, token_ids, axis=0)                          # [B, T, H]
        t = x.shape[1]
        if abs_pos_start is None:
            if caches is not None and caches[0] is not None:
                abs_pos_start = caches[0].offset
            else:
                abs_pos_start = 0
        seq_len = abs_pos_start + t
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            n = _rmsnorm(x, layer.attn_norm, self.cfg.norm_eps)
            x = x + _packed_attention(n, layer, self.cfg, cache,
                                       abs_pos_start=abs_pos_start, seq_len=seq_len)
            n = _rmsnorm(x, layer.ffn_norm, self.cfg.norm_eps)
            x = x + _packed_ffn(n, layer)
        x = _rmsnorm(x, self.final_norm, self.cfg.norm_eps)
        if last_only:
            x = x[:, -1:, :]                                                # [B, 1, H]
        head = self.embed if self.cfg.tie_word_embeddings else self.lm_head
        return x @ head.T

    def decode_batched(self, stream_tokens: list, caches: list, offsets: list[int], *,
                       paged_batched: bool = False) -> mx.array:
        """One batched decode step across ``B`` streams over packed weights (Approach-1 attention).

        The packed analogue of :meth:`InternLM2Model.decode_batched`: ``stream_tokens[b]`` is stream
        ``b``'s new token, ``caches[b]`` its per-layer cache list, ``offsets[b]`` the abs position. All
        ``_qmm`` projections / FFN / output head batch over ``B``; the only per-stream work is the
        bounded KV-cache update feeding one fused SDPA via
        :func:`~quanta.modeling.batched_attention.batched_decode_attention_kv`. Returns ``[B, 1, vocab]``
        — the batched equivalent of looping :meth:`__call__` per stream. ``paged_batched`` (#153
        loop-kill): with paged views, that helper swaps the per-stream ``.update()`` loop for ONE
        ``write_batched`` + ONE ``gather_batched`` (bit-exact; rule-4 flag).
        """
        from quanta.modeling.batched_attention import batched_decode_attention_kv

        from quanta.internlm2.attention import batched_rope_fast
        from quanta.internlm2.model import _stack_decode_tokens

        cfg = self.cfg
        ids = _stack_decode_tokens(stream_tokens)                       # [B, 1] int
        b = int(ids.shape[0])
        clists = [c if isinstance(c, list) else c.as_list() for c in caches]
        bases = [cfg.ntk_base(int(o) + 1) for o in offsets]
        x = mx.take(self.embed, ids, axis=0)                            # [B, 1, H]
        for i, layer in enumerate(self.layers):
            n = _rmsnorm(x, layer.attn_norm, cfg.norm_eps)
            q = _qmm(n, layer.q_packed, layer.q_scale, layer.q_wbias, layer.attn_bits, layer.attn_gs)
            k = _qmm(n, layer.k_packed, layer.k_scale, layer.k_wbias, layer.attn_bits, layer.attn_gs)
            v = _qmm(n, layer.v_packed, layer.v_scale, layer.v_wbias, layer.attn_bits, layer.attn_gs)
            q = mx.transpose(q.reshape(b, 1, cfg.num_attention_heads, cfg.head_dim), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(b, 1, cfg.num_key_value_heads, cfg.head_dim), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(b, 1, cfg.num_key_value_heads, cfg.head_dim), (0, 2, 1, 3))
            q = batched_rope_fast(q, offsets, bases)
            k = batched_rope_fast(k, offsets, bases)
            out = batched_decode_attention_kv(q, k, v, [clists[s][i] for s in range(b)],
                                              scale=cfg.attn_scale, n_rep=cfg.n_rep,
                                              paged_batched=paged_batched)
            out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, 1, cfg.q_dim)
            x = x + _qmm(out, layer.o_packed, layer.o_scale, layer.o_wbias, layer.attn_bits, layer.attn_gs)
            n = _rmsnorm(x, layer.ffn_norm, cfg.norm_eps)
            x = x + _packed_ffn(n, layer)
        x = _rmsnorm(x, self.final_norm, cfg.norm_eps)
        head = self.embed if cfg.tie_word_embeddings else self.lm_head
        return x @ head.T


class InternLM2ResidentModel:
    """Bf16-resident or packed-resident InternLM2.5-7B-Chat-1M, loaded from a baked quanta artifact.

    ``packed=True`` (default) uses ``mx.quantized_matmul`` and holds the matmul weights in their
    on-disk packed layout (~6.2 GB resident). ``packed=False`` dequantizes to bf16 once at load
    (~14 GB resident) and runs through ``nn.Linear`` — the parity reference path.
    """

    def __init__(self, art_dir: str | Path, *, packed: bool = True, n_layers: int | None = None,
                 quantized_kv: bool = True, kv_group_size: int = 64, kv_bits: int = 8) -> None:
        self.art = InternLM2Artifact(art_dir)
        self.cfg: InternLM2Config = self.art.cfg
        self.quantized_kv = quantized_kv
        self.kv_group_size = kv_group_size
        self.kv_bits = kv_bits
        self.packed = packed

        if packed:
            self._model = _PackedModel(self.art, self.cfg, n_layers=n_layers)
        else:
            m = InternLM2Model(self.cfg, n_layers=n_layers)
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
                  group_size: int | None = None,
                  bits: int | None = None) -> InternLM2Cache:
        """Allocate a fresh decode cache. Defaults follow the runtime's construction flags."""
        return InternLM2Cache(
            self.cfg,
            quantized=self.quantized_kv if quantized is None else quantized,
            group_size=self.kv_group_size if group_size is None else group_size,
            bits=self.kv_bits if bits is None else bits,
        )

    # Alias for the oMLX shim's ``_SingleTokenStepper``, which calls ``runtime.make_caches()``
    # uniformly across model classes (Qwen3.5 / GLM / MiniMax / Qwen2.5 all expose this name).
    def make_caches(self) -> InternLM2Cache:
        return self.new_cache()

    def __call__(self, token_ids: mx.array, *,
                 cache: InternLM2Cache | None = None,
                 caches: InternLM2Cache | list | None = None,
                 offset: int | None = None,
                 use_fast: bool = True,
                 last_only: bool = False) -> mx.array:
        """Forward ``[B, T]`` token ids → ``[B, T, vocab]`` logits.

        Cache kwargs (accepts both forms — the oMLX ``_SingleTokenStepper`` passes ``caches=``):

        * ``cache=InternLM2Cache``     — quanta-native singular name.
        * ``caches=InternLM2Cache``    — shim-compat alias (a single InternLM2Cache, *not* a list).
        * ``caches=[KVCache, ...]``    — a raw per-layer list (rare; if the caller already has one).

        ``offset`` (shim-compat): absolute position of the first new token. When set, it overrides
        ``cache.offset`` for the RoPE position / NTK base — used for chunked prefill where each
        chunk passes ``offset = chunk_idx * chunk_size``.

        ``use_fast`` is plumbed for signature symmetry with the bf16 path; the packed path always
        uses ``mx.fast.rope`` + ``mx.fast.scaled_dot_product_attention``.
        """
        del use_fast  # packed runtime is always fast
        # Normalize input shape: the oMLX shim's ``_SingleTokenStepper`` passes a 1-D ``[T]`` array
        # (one or more tokens to ingest at the running offset); the in-repo ``generate`` already
        # 2-D-ifies. Promote to 2-D so both call sites share one code path.
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

        if isinstance(self._model, _PackedModel):
            return self._model(token_ids, caches=cache_list,
                                abs_pos_start=abs_pos_start, last_only=last_only)
        # bf16 reference path
        return self._model(token_ids, caches=cache_list, use_fast=True,
                            abs_pos_start=abs_pos_start, last_only=last_only)

    def decode_batched(self, stream_tokens: list, caches: list, offsets: list[int], *,
                       paged_batched: bool = False) -> mx.array:
        """Batched single-step decode across ``B`` streams → ``[B, 1, vocab]`` (Approach-1 attention).

        Delegates to the active inner forward's ``decode_batched`` (``_PackedModel`` in prod, the bf16
        :class:`~quanta.internlm2.model.InternLM2Model` under ``packed=False``). ``stream_tokens[b]`` is
        stream ``b``'s new token, ``caches[b]`` its per-layer cache list (or an
        :class:`~quanta.internlm2.decode.InternLM2Cache`), ``offsets[b]`` the abs position. The batched
        equivalent of calling :meth:`__call__` once per stream — the win is one fused SDPA + one batched
        matmul per layer instead of ``B`` looped ones. ``paged_batched`` (#153 loop-kill) is threaded to
        the inner ``decode_batched`` (ONE ``write_batched`` + ONE ``gather_batched`` when caches are paged)."""
        return self._model.decode_batched(stream_tokens, caches, offsets, paged_batched=paged_batched)
