"""Streamed bf16 source loader for InternLM2.5-7B-Chat-1M (``internlm2``).

The checkpoint is **plain bf16** (no ``quantization_config``), ~14 GB across 8 shards, so — like the
GLM/Qwen2.5/Qwen3.5 loaders and unlike the Kimi (int4) / DSV4 (fp4/fp8) loaders — there is **no
dequant**. Accessors stream the needed tensors from the sharded safetensors via
``model.safetensors.index.json``; ``mx.load`` memory-maps the shard and only the requested tensor
materializes on ``eval``. Per-kind accessors hand back **one layer's** params at a time so a consumer
(the bf16 reference forward or the bake) never holds more than a layer resident (rule 8).

The accessor surface mirrors :class:`quanta.qwen25.loader.Qwen25SourceCheckpoint`: ``embed`` /
``final_norm`` / ``lm_head`` plus per-layer ``block_norms`` / ``attention`` / ``mlp``. The single
non-trivial difference is the **wqkv split**: InternLM2 ships attention as a fused weight
``model.layers.{i}.attention.wqkv.weight`` of shape
``[(num_heads + 2·num_kv_heads) · head_dim, hidden_size]`` (``[6144, 4096]`` for the 7B). Its rows
are laid out *per-kv-head*: each block of ``(num_key_value_groups + 2) · head_dim`` rows covers one
kv-head's q-heads (slots ``[0..num_key_value_groups)``), its k-head (slot ``-2``), and its v-head
(slot ``-1``). :func:`_split_wqkv` deinterleaves this once and returns the three standard
``wq``/``wk``/``wv`` projections, preserving the per-kv-head query ordering so a standard GQA
``mx.repeat`` over the kv axis pairs each q-head with its kv-head exactly as the source model
does.

Tensors come back in their **native dtype** verbatim (all bf16 in this checkpoint); never silently
downcast (rule 6). The source key namespace is preserved: ``model.tok_embeddings``,
``model.norm``, ``output`` (untied lm_head), and per layer ``attention.{wq,wk,wv,wo}``,
``feed_forward.{w1,w2,w3}``, ``attention_norm``, ``ffn_norm`` — different from the Llama-style
namespace Qwen2.5 uses but kept verbatim so the artifact reader's keys round-trip 1:1.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.internlm2.config import InternLM2Config

# --- top-level text tensors (InternLM2 has its own naming — *not* Llama/Qwen) -----------------------
MODEL_PREFIX = "model."
EMBED_KEY = MODEL_PREFIX + "tok_embeddings.weight"
FINAL_NORM_KEY = MODEL_PREFIX + "norm.weight"
LM_HEAD_KEY = "output.weight"  # bare top-level; NOT lm_head.weight, NOT model.lm_head

# --- per-kind suffix sets (after wqkv split — declarative; bake's policy partition mirrors these) ----
# Attention weights (matmul) — int8-quantizable, 2-D ``[out, in]``. Synthesized from wqkv split for
# wq/wk/wv; loaded verbatim for wo.
ATTN_WEIGHT_SUFFIXES: tuple[str, ...] = (
    "wq.weight",
    "wk.weight",
    "wv.weight",
    "wo.weight",
)
# SwiGLU FFN — int4-quantizable (dominates byte count; bf16 source tolerates int4 g64 well).
# InternLM2 naming: w1 = gate, w3 = up, w2 = down.
MLP_SUFFIXES: tuple[str, ...] = (
    "w1.weight",
    "w3.weight",
    "w2.weight",
)


def _split_wqkv(wqkv: mx.array, cfg: InternLM2Config) -> tuple[mx.array, mx.array, mx.array]:
    """Deinterleave the fused ``wqkv`` ``[(H + 2·H_kv)·D, hidden]`` into ``(wq, wk, wv)``.

    Source layout (row-major):

        wqkv.reshape(num_kv_heads, num_key_value_groups + 2, head_dim, hidden_size)

    Within each per-kv-head row block (``gs = num_key_value_groups + 2`` slots × ``head_dim`` rows):

        - slots ``[0 .. num_key_value_groups)`` → q-heads of this kv-head
        - slot ``-2`` → k-head of this kv-head
        - slot ``-1`` → v-head of this kv-head

    Returns:

        wq → ``[num_attention_heads · head_dim, hidden_size]`` (= ``[4096, 4096]`` for 7B)
        wk → ``[num_key_value_heads   · head_dim, hidden_size]`` (= ``[1024, 4096]``)
        wv → ``[num_key_value_heads   · head_dim, hidden_size]``

    The per-kv-head interleaved q ordering is **preserved**: q-heads ``[0..rep)`` belong to
    kv-head 0, ``[rep..2·rep)`` to kv-head 1, etc. — so a standard GQA ``mx.repeat`` over the kv
    axis pairs each q-head with the correct kv-head at attention time.
    """
    n_kv = cfg.num_key_value_heads
    rep = cfg.num_key_value_groups
    gs = rep + 2
    hd = cfg.head_dim
    hidden = cfg.hidden_size
    expected = (n_kv * gs * hd, hidden)
    if tuple(wqkv.shape) != expected:
        raise ValueError(f"wqkv shape {tuple(wqkv.shape)} != expected {expected}")
    grouped = wqkv.reshape(n_kv, gs, hd, hidden)                # [n_kv, gs, hd, hidden]
    wq = grouped[:, :rep, :, :].reshape(n_kv * rep * hd, hidden)
    wk = grouped[:, rep,   :, :].reshape(n_kv * hd, hidden)
    wv = grouped[:, rep + 1, :, :].reshape(n_kv * hd, hidden)
    return wq, wk, wv


class InternLM2SourceCheckpoint:
    """Lazy, sharded reader for an InternLM2.5-7B-Chat-1M bf16 checkpoint directory."""

    def __init__(self, model_dir: str | Path, cfg: InternLM2Config | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg if cfg is not None else InternLM2Config.from_pretrained(self.dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self._wm: dict[str, str] = index["weight_map"]
        self._cache_file: str | None = None      # single-entry shard mmap cache
        self._cache: dict[str, mx.array] = {}

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    # ---- tensor access ------------------------------------------------------
    def _tensor(self, key: str) -> mx.array:
        """The tensor for ``key``, streamed from its shard (fails loud if absent — rule 6)."""
        shard = self._wm.get(key)
        if shard is None:
            raise KeyError(f"{key!r} not in InternLM2 weight_map ({self.dir})")
        if shard != self._cache_file:
            self._cache = mx.load(str(self.dir / shard))
            self._cache_file = shard
        return self._cache[key]

    def has(self, key: str) -> bool:
        return key in self._wm

    def release(self) -> None:
        """Drop the current shard handle so its mmap can be released."""
        self._cache = {}
        self._cache_file = None

    # ---- top-level ----------------------------------------------------------
    def embed(self) -> mx.array:
        return self._tensor(EMBED_KEY)

    def final_norm(self) -> mx.array:
        return self._tensor(FINAL_NORM_KEY)

    def lm_head(self) -> mx.array:
        """Output projection — separate from the embedding for InternLM2.5 (``tie_word_embeddings=False``).

        Lives at the bare top-level key ``output.weight`` (NOT ``lm_head.weight``).
        """
        key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        return self._tensor(key)

    # ---- per-layer kinds ----------------------------------------------------
    def block_norms(self, i: int) -> dict[str, mx.array]:
        """The two RMSNorms on every layer — InternLM2 naming.

        Returns ``{"attention_norm": …, "ffn_norm": …}`` — pre-attn and pre-FFN, the same
        positions as Llama's ``input_layernorm`` / ``post_attention_layernorm``.
        """
        p = f"{MODEL_PREFIX}layers.{i}."
        return {
            "attention_norm": self._tensor(p + "attention_norm.weight"),
            "ffn_norm": self._tensor(p + "ffn_norm.weight"),
        }

    def attention(self, i: int) -> dict[str, mx.array]:
        """GQA attention tensors for layer ``i``: wq / wk / wv (split from fused wqkv) + wo.

        InternLM2 ships ``wqkv`` (a single fused weight); we deinterleave once via
        :func:`_split_wqkv` and return the three standard projections so the bake / artifact /
        runtime see a uniform GQA layout. No biases (``cfg.attention_bias=False`` for the 7B).
        """
        p = f"{MODEL_PREFIX}layers.{i}.attention."
        wqkv = self._tensor(p + "wqkv.weight")
        wq, wk, wv = _split_wqkv(wqkv, self.cfg)
        out: dict[str, mx.array] = {
            "wq.weight": wq,
            "wk.weight": wk,
            "wv.weight": wv,
            "wo.weight": self._tensor(p + "wo.weight"),
        }
        # InternLM2.5 ships no biases (cfg.attention_bias=False). If a future variant adds them,
        # they would live at p+"wqkv.bias" + p+"wo.bias" — caller can extend. Not synthesized here.
        return out

    def mlp(self, i: int) -> dict[str, mx.array]:
        """SwiGLU FFN tensors for layer ``i``: w1 (gate) / w3 (up) / w2 (down) — InternLM2 naming."""
        p = f"{MODEL_PREFIX}layers.{i}.feed_forward."
        return {s: self._tensor(p + s) for s in MLP_SUFFIXES}
