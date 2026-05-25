"""MiniMax-M2.7 full decoder forward (MLX) — embed -> 62 pre-norm blocks -> final norm -> logits.

Standard DeepSeek/Mixtral-style pre-norm decoder (every layer identical: all MoE, all full GQA
softmax):

* ``h = embed[ids]``;
* each block: ``h = h + attn(input_layernorm(h))`` then ``h = h + moe(post_attention_layernorm(h))``;
* ``logits = (rmsnorm_final(h) @ lm_head.T)``.

The block is an ``mlx.nn`` module so it composes ``nn.RMSNorm`` + :class:`MiniMaxAttention` (which
carries the GQA projections + per-layer QK-norm) and holds the MoE router/expert weights as raw
stacks dispatched by :func:`quanta.minimax.moe.minimax_moe`. There is **no shared expert**
(``shared_intermediate_size == 0``) — the MoE output is the routed sum only.

Two consumers share the block math:

* :class:`MiniMaxModel` — a random-init module for the model-free gates (prefill==decode, per-layer
  naive==optimized) and, eventually, the resident runtime.
* :func:`minimax_logits` — the **streamed** teacher-forced reference: it materializes one layer's
  bf16 weights from :class:`quanta.minimax.loader.MiniMaxSourceCheckpoint`, runs the block, frees the
  shard mmaps, and moves on (rule 8: ≤1 layer's expert stacks resident — the memory peak). The heavy
  real-weight invocation is **deferred to a GPU session** and is documented (not run) in
  :mod:`parity.minimax_forward_test`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.minimax.attention import MiniMaxAttention
from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.moe import minimax_moe


class MiniMaxBlock(nn.Module):
    """One MiniMax decoder layer: ``h + attn(input_ln(h))`` then ``h + moe(post_attn_ln(h))``.

    The attention sub-module owns q/k/v/o + q_norm/k_norm; the MoE weights live here as raw arrays
    (``gate``/``e_score_correction_bias`` + the ``w1/w3/w2`` expert stacks) so the sparse gather
    dispatch in :func:`minimax_moe` can run without per-expert parameter plumbing.
    """

    def __init__(self, cfg: MiniMaxConfig, layer_id: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        h, inter, e = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_local_experts
        self.input_layernorm = nn.RMSNorm(h, eps=cfg.norm_eps)
        self.self_attn = MiniMaxAttention(cfg, layer_id)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=cfg.norm_eps)
        # Router + routed expert stacks (Mixtral naming: w1=gate, w3=up, w2=down). Raw arrays so the
        # sparse gather_mm path consumes them directly; the loader / set_moe fills real weights.
        self.gate_weight = mx.zeros((e, h))
        self.e_score_correction_bias = mx.zeros((e,))
        self.w1 = mx.zeros((e, inter, h))    # gate
        self.w3 = mx.zeros((e, inter, h))    # up
        self.w2 = mx.zeros((e, h, inter))    # down

    def set_moe(self, gate: mx.array, bias: mx.array, w1: mx.array, w3: mx.array, w2: mx.array) -> None:
        self.gate_weight, self.e_score_correction_bias = gate, bias
        self.w1, self.w3, self.w2 = w1, w3, w2

    def _router(self) -> dict:
        return {"weight": self.gate_weight, "e_score_correction_bias": self.e_score_correction_bias}

    def _experts(self) -> dict:
        return {"w1": self.w1, "w3": self.w3, "w2": self.w2}

    def __call__(self, x, *, offset=0, cache=None, use_fast=True):
        x = x + self.self_attn(self.input_layernorm(x), offset=offset, cache=cache, use_fast=use_fast)
        x = x + minimax_moe(self.post_attention_layernorm(x), self._router(), self._experts(), self.cfg)
        return x


class MiniMaxModel(nn.Module):
    """Assembled MiniMax-M2.7 decoder (random-init): embed -> blocks -> final norm -> lm_head."""

    def __init__(self, cfg: MiniMaxConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [MiniMaxBlock(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, token_ids, *, caches=None, use_fast=True):
        """Logits for ``token_ids`` ``[t]`` -> ``[1, t, vocab]``. ``caches`` (per-layer KV caches)
        grow in place; ``None`` ⇒ fresh prefill. Pass the prior caches (one token at a time) to
        continue decode."""
        h = self.embed_tokens(token_ids)[None]                     # [1, t, hidden]
        n = len(self.layers)
        caches = caches if caches is not None else [None] * n
        for i, blk in enumerate(self.layers):
            h = blk(h, cache=caches[i], use_fast=use_fast)
        return self.lm_head(self.norm(h))


# --- streamed teacher-forced reference (one layer resident; heavy load deferred) ----------------
def load_block(block: MiniMaxBlock, ck, cfg: MiniMaxConfig, i: int, dtype: mx.Dtype = mx.bfloat16) -> None:
    """Fill one built block from the source checkpoint (attention + QK-norms + block norms + MoE).

    ``ck.attention(i)`` already materializes the dequantized bf16 q/k/v/o + q_norm/k_norm; ``ck.moe(i)``
    returns ``{"router": {...}, "experts": {w1,w3,w2:[E,*]}}``. Direct attribute assignment (the
    block mixes ``nn.Linear``/``nn.RMSNorm`` with raw expert stacks). Shard mmaps are released by the
    loader after each accessor — keep ≤1 layer resident (rule 8)."""
    a = ck.attention(i)
    m = block.self_attn
    m.q_proj.weight = a["q_proj"].astype(dtype)
    m.k_proj.weight = a["k_proj"].astype(dtype)
    m.v_proj.weight = a["v_proj"].astype(dtype)
    m.o_proj.weight = a["o_proj"].astype(dtype)
    m.q_norm.weight = a["q_norm"].astype(dtype)
    m.k_norm.weight = a["k_norm"].astype(dtype)
    norms = ck.block_norms(i)
    block.input_layernorm.weight = norms["input_layernorm"].astype(dtype)
    block.post_attention_layernorm.weight = norms["post_attention_layernorm"].astype(dtype)
    mo = ck.moe(i)
    r, ex = mo["router"], mo["experts"]
    block.set_moe(r["weight"], r["e_score_correction_bias"],
                  ex["w1"].astype(dtype), ex["w3"].astype(dtype), ex["w2"].astype(dtype))


def minimax_logits(ck, ids: mx.array, cfg: MiniMaxConfig, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Teacher-forced logits ``[1,S,vocab]`` for token ids ``[1,S]`` (streamed, ≤1 layer resident).

    Deferred to a GPU session — never run against real tensors in the model-free gate.
    """
    h = ck.embed()[ids[0]].astype(dtype)[None]                     # [1, S, hidden]
    for i in range(cfg.num_hidden_layers):
        blk = MiniMaxBlock(cfg, i)
        load_block(blk, ck, cfg, i, dtype)
        h = blk(h, use_fast=True)
        mx.eval(h)
        ck.release()
    h = mx.fast.rms_norm(h.astype(mx.float32), ck.final_norm().astype(mx.float32), cfg.norm_eps)
    return h @ ck.lm_head().astype(mx.float32).T


def teacher_forced_ppl(ck, ids: mx.array, cfg: MiniMaxConfig, dtype: mx.Dtype = mx.bfloat16) -> float:
    """Mean teacher-forced perplexity of ``ids`` ``[1,S]`` (next-token CE over positions 0..S-2).

    Deferred — heavy (real weights + memory). Documented, not run, in the model-free gate.
    """
    logits = minimax_logits(ck, ids, cfg, dtype).astype(mx.float32)[0]    # [S, vocab]
    tgt = ids[0, 1:]
    lse = mx.logsumexp(logits[:-1], axis=-1)
    tok = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
    return float(mx.exp(mx.mean(lse - tok)).item())
