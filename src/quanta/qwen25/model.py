"""Naive bf16 reference forward for Qwen2.5-14B-Instruct-1M.

The float baseline that the resident quantized runtime must match in teacher-forced perplexity and
top-1 logit agreement. Built with plain ``mlx.nn`` modules + ``mx.fast`` primitives (rules 1+2); no
fused absorb, no quantization, no sparse paths — just the obviously-correct forward.

Layer (Qwen2-style, dense, all 48 layers):

    h = x + Attention(RMSNorm(x))                # GQA + QKV biases + full RoPE
    y = h + SwiGLU(RMSNorm(h))                   # silu(gate(h)) * up(h)  ->  down

Mirrors :mod:`quanta.qwen35.model` minus everything Qwen2.5 lacks (MoE, hybrid SSM, MTP, QK-norm,
output gate, partial RoPE, mRoPE). Loaded from either :class:`~quanta.qwen25.loader.Qwen25SourceCheckpoint`
(bf16 source) or :class:`~quanta.qwen25.artifact.Qwen25Artifact` (dequantized baked) — both expose
the same suffix-keyed dicts.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from quanta.qwen25.attention import KVCache, Qwen25Attention
from quanta.qwen25.config import Qwen25Config


class _SwiGLU(nn.Module):
    """Standard SwiGLU FFN: ``silu(gate(x)) * up(x) -> down``. No bias."""

    def __init__(self, cfg: Qwen25Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen25Config) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.self_attn = Qwen25Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = _SwiGLU(cfg)

    def __call__(self, x: mx.array, *, cache: KVCache | None = None, use_fast: bool = True,
                 seq_hint: int | None = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), cache=cache, use_fast=use_fast,
                                seq_hint=seq_hint)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen25Model(nn.Module):
    """Dense 48-layer Qwen2.5 reference forward (parity baseline).

    Constructed empty; load weights via :meth:`load_from` (a loader or artifact) which streams them
    in one layer at a time. For full-model parity, drive the whole 48-layer stack; for bounded
    validation, drive a slice (``n_layers``).
    """

    def __init__(self, cfg: Qwen25Config, *, n_layers: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.num_hidden_layers if n_layers is None else n_layers
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [_DecoderLayer(cfg) for _ in range(self.n_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array, *, caches: list[KVCache] | None = None,
                 use_fast: bool = True, seq_hint: int | None = None,
                 abs_pos_start: int | None = None, last_only: bool = False) -> mx.array:
        """Forward ``[B, T]`` token ids → logits ``[B, T, vocab]``.

        ``caches`` (one ``KVCache`` per layer) enables incremental decode; pass ``None`` for prefill.
        ``seq_hint`` lets the model see the full sequence length (for DCA / future scaling) when the
        forward is chunked — defaults to ``cache.offset + T`` per layer.

        ``abs_pos_start`` / ``last_only`` mirror the packed path's signature so this bf16 reference is
        callable from :class:`~quanta.qwen25.runtime.Qwen25ResidentModel`. ``abs_pos_start`` is mapped
        onto ``seq_hint`` (the bf16 :class:`~quanta.qwen25.attention.Qwen25Attention` uses absolute
        positions via ``cache.offset`` already — chunked DCA prefill needs the packed path).
        ``last_only`` slices the residual to its last row before lm_head, matching the packed path.
        """
        if seq_hint is None and abs_pos_start is not None and caches is None:
            # Approximation only — chunked DCA prefill requires the packed path.
            seq_hint = abs_pos_start + token_ids.shape[-1]
        x = self.embed_tokens(token_ids)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, cache=cache, use_fast=use_fast, seq_hint=seq_hint)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:, :]
        if self.cfg.tie_word_embeddings:
            return x @ self.embed_tokens.weight.T
        return self.lm_head(x)

    # --- weight loading --------------------------------------------------------
    def load_from(self, source: Any) -> None:
        """Stream weights from a duck-typed source (``Qwen25SourceCheckpoint`` or ``Qwen25Artifact``).

        Both expose the same ``embed`` / ``final_norm`` / ``lm_head`` / ``block_norms`` /
        ``attention`` / ``mlp`` surface, so this is a single load path that works for both bf16 source
        and dequantized baked. Per rule-8 the source releases its shard handles per layer.
        """
        self.embed_tokens.weight = source.embed().astype(mx.bfloat16)
        self.norm.weight = source.final_norm().astype(mx.bfloat16)
        if not self.cfg.tie_word_embeddings:
            self.lm_head.weight = source.lm_head().astype(mx.bfloat16)

        for i in range(self.n_layers):
            layer = self.layers[i]
            ln = source.block_norms(i)
            layer.input_layernorm.weight = ln["input_layernorm"].astype(mx.bfloat16)
            layer.post_attention_layernorm.weight = ln["post_attention_layernorm"].astype(mx.bfloat16)

            attn = source.attention(i)
            layer.self_attn.q_proj.weight = attn["q_proj.weight"].astype(mx.bfloat16)
            layer.self_attn.k_proj.weight = attn["k_proj.weight"].astype(mx.bfloat16)
            layer.self_attn.v_proj.weight = attn["v_proj.weight"].astype(mx.bfloat16)
            layer.self_attn.o_proj.weight = attn["o_proj.weight"].astype(mx.bfloat16)
            if self.cfg.attention_bias:
                layer.self_attn.q_proj.bias = attn["q_proj.bias"].astype(mx.bfloat16)
                layer.self_attn.k_proj.bias = attn["k_proj.bias"].astype(mx.bfloat16)
                layer.self_attn.v_proj.bias = attn["v_proj.bias"].astype(mx.bfloat16)

            mlp = source.mlp(i)
            layer.mlp.gate_proj.weight = mlp["gate_proj.weight"].astype(mx.bfloat16)
            layer.mlp.up_proj.weight = mlp["up_proj.weight"].astype(mx.bfloat16)
            layer.mlp.down_proj.weight = mlp["down_proj.weight"].astype(mx.bfloat16)

            if hasattr(source, "release"):
                source.release()
            mx.clear_cache()
