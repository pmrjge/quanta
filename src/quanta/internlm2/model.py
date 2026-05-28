"""Naive bf16 reference forward for InternLM2.5-7B-Chat-1M.

The float baseline that the resident quantized runtime must match in teacher-forced perplexity and
top-1 logit agreement. Built with plain ``mlx.nn`` modules + ``mx.fast`` primitives (rules 1+2); no
fused absorb, no quantization, no sparse paths — just the obviously-correct forward.

Layer (InternLM2-style, dense, all 32 layers):

    h = x + Attention(RMSNorm(x))                # GQA + Llama-style RoPE + dynamic-NTK
    y = h + SwiGLU(RMSNorm(h))                   # w2(silu(w1(h)) * w3(h))

Note the **InternLM2 module naming**: ``attention.wq/wk/wv/wo``, ``feed_forward.w1/w3/w2``,
``attention_norm`` (pre-attn), ``ffn_norm`` (pre-FFN). These names round-trip 1:1 with the
source ``model.layers.{i}.attention.*`` / ``model.layers.{i}.feed_forward.*`` /
``attention_norm`` / ``ffn_norm`` keys, so the loader's :meth:`block_norms` / :meth:`attention`
/ :meth:`mlp` dict keys plug straight into the module attributes.

Loaded from either :class:`~quanta.internlm2.loader.InternLM2SourceCheckpoint` (bf16 source — the
fused ``wqkv`` was split at load time) or :class:`~quanta.internlm2.artifact.InternLM2Artifact`
(dequantized baked) — both expose the same suffix-keyed dicts.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from quanta.internlm2.attention import InternLM2Attention, KVCache
from quanta.internlm2.config import InternLM2Config


def _stack_decode_tokens(stream_tokens: list) -> mx.array:
    """Stack ``B`` per-stream single decode tokens (``int`` / ``[1]`` / ``[1,1]`` mx.array) → ``[B, 1]``
    int ids. Used by the batched-decode path (one new token per stream per step)."""
    rows = []
    for tok in stream_tokens:
        a = tok if isinstance(tok, mx.array) else mx.array([int(tok)])
        rows.append(int(a.reshape(-1)[0].item()))
    return mx.array(rows, dtype=mx.int32)[:, None]


class _SwiGLU(nn.Module):
    """InternLM2 SwiGLU FFN: ``w2(silu(w1(x)) * w3(x))`` — w1=gate, w3=up, w2=down. No bias."""

    def __init__(self, cfg: InternLM2Config) -> None:
        super().__init__()
        bias = bool(cfg.attention_bias)  # InternLM2's single ``bias`` field governs FFN too
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.w3 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=bias)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: InternLM2Config) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.attention = InternLM2Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.feed_forward = _SwiGLU(cfg)

    def __call__(self, x: mx.array, *, cache: KVCache | None = None, use_fast: bool = True,
                 seq_hint: int | None = None) -> mx.array:
        h = x + self.attention(self.attention_norm(x), cache=cache, use_fast=use_fast,
                                seq_hint=seq_hint)
        return h + self.feed_forward(self.ffn_norm(h))


def _load_decoder_layer(layer: _DecoderLayer, source: Any, i: int,
                        dtype: mx.Dtype = mx.bfloat16) -> None:
    """Stream layer ``i``'s weights from a duck-typed ``source`` into ``layer`` (no release).

    Shared by :meth:`InternLM2Model.load_from` (whole-model load) and :func:`internlm2_logits`
    (one-layer-resident streaming ppl) so both paths use a single loading convention — InternLM2
    source key naming (``wq/wk/wv/wo``, ``w1/w3/w2``, ``attention_norm``/``ffn_norm``). The caller
    owns ``source.release()`` + ``mx.clear_cache()`` (rule-8 memory discipline).
    """
    ln = source.block_norms(i)
    layer.attention_norm.weight = ln["attention_norm"].astype(dtype)
    layer.ffn_norm.weight = ln["ffn_norm"].astype(dtype)
    attn = source.attention(i)
    layer.attention.wq.weight = attn["wq.weight"].astype(dtype)
    layer.attention.wk.weight = attn["wk.weight"].astype(dtype)
    layer.attention.wv.weight = attn["wv.weight"].astype(dtype)
    layer.attention.wo.weight = attn["wo.weight"].astype(dtype)
    mlp = source.mlp(i)
    layer.feed_forward.w1.weight = mlp["w1.weight"].astype(dtype)
    layer.feed_forward.w3.weight = mlp["w3.weight"].astype(dtype)
    layer.feed_forward.w2.weight = mlp["w2.weight"].astype(dtype)


class InternLM2Model(nn.Module):
    """Dense 32-layer InternLM2.5 reference forward (parity baseline).

    Constructed empty; load weights via :meth:`load_from` (a loader or artifact) which streams them
    in one layer at a time. For full-model parity, drive the whole 32-layer stack; for bounded
    validation, drive a slice (``n_layers``).
    """

    def __init__(self, cfg: InternLM2Config, *, n_layers: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.num_hidden_layers if n_layers is None else n_layers
        # InternLM2 names the embedding ``model.tok_embeddings`` (kept here as ``tok_embeddings``).
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [_DecoderLayer(cfg) for _ in range(self.n_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        if not cfg.tie_word_embeddings:
            # InternLM2 names the output projection bare ``output`` (top-level, not ``lm_head``).
            self.output = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array, *, caches: list[KVCache] | None = None,
                 use_fast: bool = True, seq_hint: int | None = None,
                 abs_pos_start: int | None = None, last_only: bool = False) -> mx.array:
        """Forward ``[B, T]`` token ids → logits ``[B, T, vocab]``.

        ``caches`` (one ``KVCache`` per layer) enables incremental decode; pass ``None`` for prefill.
        ``seq_hint`` lets the model see the full sequence length (for dynamic-NTK base scaling) when
        the forward is chunked — defaults to ``cache.offset + T`` per layer.

        ``abs_pos_start`` / ``last_only`` mirror the packed path's signature so this bf16 reference is
        callable from :class:`~quanta.internlm2.runtime.InternLM2ResidentModel`.
        ``abs_pos_start`` is mapped onto ``seq_hint`` (the bf16 :class:`~quanta.internlm2.attention.InternLM2Attention`
        uses absolute positions via ``cache.offset`` already — the override is only for chunked
        prefill).  ``last_only`` slices the residual to its last row before the output head,
        matching the packed path.
        """
        if seq_hint is None and abs_pos_start is not None and caches is None:
            seq_hint = abs_pos_start + token_ids.shape[-1]
        x = self.tok_embeddings(token_ids)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, cache=cache, use_fast=use_fast, seq_hint=seq_hint)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:, :]
        if self.cfg.tie_word_embeddings:
            return x @ self.tok_embeddings.weight.T
        return self.output(x)

    def decode_batched(self, stream_tokens: list, caches: list, offsets: list[int]) -> mx.array:
        """One batched decode step across ``B`` streams (Approach-1 vectorized attention).

        Replaces the per-stream ``step_batch`` loop: ``stream_tokens[b]`` is stream ``b``'s single new
        token (``int`` / ``[1]`` / ``[1,1]``), ``caches[b]`` its per-layer ``[KVCache]*n_layers`` list,
        ``offsets[b]`` the new token's absolute position. Projections, FFN and the output head batch
        over ``B`` trivially (per-token ops); the only per-stream work is the **bounded** KV-cache
        update (bookkeeping, not compute) feeding one fused :func:`batched_decode_attention`. Returns
        ``[B, 1, vocab]`` — equal to looping :meth:`__call__` per stream up to RoPE/SDPA tiling ULPs.
        """
        from quanta.modeling.batched_attention import batched_decode_attention

        from quanta.internlm2.attention import batched_rope_fast

        cfg = self.cfg
        ids = _stack_decode_tokens(stream_tokens)                    # [B, 1] int
        b = int(ids.shape[0])
        clists = [c if isinstance(c, list) else c.as_list() for c in caches]
        bases = [cfg.ntk_base(int(o) + 1) for o in offsets]
        x = self.tok_embeddings(ids)                                 # [B, 1, H]
        for i, layer in enumerate(self.layers):
            att = layer.attention
            n = layer.attention_norm(x)
            q = mx.transpose(att.wq(n).reshape(b, 1, att.nh, att.hd), (0, 2, 1, 3))   # [B,H,1,D]
            k = mx.transpose(att.wk(n).reshape(b, 1, att.nkv, att.hd), (0, 2, 1, 3))  # [B,nkv,1,D]
            v = mx.transpose(att.wv(n).reshape(b, 1, att.nkv, att.hd), (0, 2, 1, 3))
            q = batched_rope_fast(q, offsets, bases)
            k = batched_rope_fast(k, offsets, bases)
            qs, ks, vs = [], [], []
            for s in range(b):                                       # bounded stream loop (cache I/O)
                kf, vf = clists[s][i].update(k[s:s + 1], v[s:s + 1])  # [1, nkv, L_s, D]
                qs.append(q[s])
                ks.append(kf)
                vs.append(vf)
            out = batched_decode_attention(qs, ks, vs, scale=att.scale, n_rep=att.rep)  # [B,H,1,D]
            out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, 1, att.nh * att.hd)
            x = x + att.wo(out)
            x = x + layer.feed_forward(layer.ffn_norm(x))
        x = self.norm(x)
        if cfg.tie_word_embeddings:
            return x @ self.tok_embeddings.weight.T
        return self.output(x)

    # --- weight loading --------------------------------------------------------
    def load_from(self, source: Any) -> None:
        """Stream weights from a duck-typed source (``InternLM2SourceCheckpoint`` or ``InternLM2Artifact``).

        Both expose the same ``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` /
        ``attention`` / ``mlp`` surface, so this is a single load path that works for both bf16
        source and dequantized baked. Per rule-8 the source releases its shard handles per layer.
        """
        self.tok_embeddings.weight = source.embed().astype(mx.bfloat16)
        self.norm.weight = source.final_norm().astype(mx.bfloat16)
        if not self.cfg.tie_word_embeddings:
            self.output.weight = source.lm_head().astype(mx.bfloat16)

        for i in range(self.n_layers):
            _load_decoder_layer(self.layers[i], source, i, mx.bfloat16)
            if hasattr(source, "release"):
                source.release()
            mx.clear_cache()


def internlm2_logits(source: Any, ids: mx.array, cfg: InternLM2Config,
                     dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Full bf16 forward over ``ids`` ``[B, T]`` → logits ``[B, T, vocab]`` (fp32), **one layer
    resident at a time** (rule-8).

    Streams each decoder layer's weights out of ``source`` (an
    :class:`~quanta.internlm2.loader.InternLM2SourceCheckpoint` or
    :class:`~quanta.internlm2.artifact.InternLM2Artifact`), runs it, evals, and releases the shard
    handles before loading the next — so peak resident weight is ~one layer, not the whole 7B model.
    The decoder blocks run in ``dtype`` (bf16, matching the runtime); the final norm + output head
    are computed in fp32 for ppl precision. ``use_fast=True`` (mx.fast RoPE/SDPA) — equivalence to
    the naive path is gated model-free in ``parity/internlm2_forward_test.py``.
    """
    if ids.ndim == 1:
        ids = ids[None]
    emb = source.embed().astype(dtype)                          # [V, H]
    h = emb[ids]                                                # [B, T, H]
    del emb
    if hasattr(source, "release"):
        source.release()
    mx.clear_cache()
    for i in range(cfg.num_hidden_layers):
        layer = _DecoderLayer(cfg)
        _load_decoder_layer(layer, source, i, dtype)
        h = layer(h, use_fast=True)
        mx.eval(h)
        del layer
        if hasattr(source, "release"):
            source.release()
        mx.clear_cache()
    h = mx.fast.rms_norm(h.astype(mx.float32), source.final_norm().astype(mx.float32), cfg.norm_eps)
    head = source.lm_head().astype(mx.float32)                  # [V, H] (untied for InternLM2.5)
    return h @ head.T


def teacher_forced_ppl(source: Any, ids: mx.array, cfg: InternLM2Config,
                       dtype: mx.Dtype = mx.bfloat16) -> float:
    """Mean teacher-forced perplexity of ``ids`` ``[1, S]`` (next-token CE over positions 0..S-2).

    The e2e arbiter (CLAUDE.md methodology): on clean prose with the correct BOS, a parity-correct
    bf16 forward gives low single-digit ppl; a localized forward bug yields catastrophic ppl (the
    Kimi lesson — ~165 with a bug vs ~3.3 fixed). Heavy: streams the real ~7B source.
    """
    logits = internlm2_logits(source, ids, cfg, dtype).astype(mx.float32)[0]   # [S, vocab]
    tgt = ids[0, 1:]
    lse = mx.logsumexp(logits[:-1], axis=-1)
    tok = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
    return float(mx.exp(mx.mean(lse - tok)).item())
