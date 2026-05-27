"""Qwen3.5-397B-A17B decoder assembly: embed -> [hybrid blocks] -> final norm -> lm_head, MLX.

Each of the 60 layers is a **two-residual pre-norm block** (DeepSeek/Qwen-MoE layout):

    x = x + mixer(input_layernorm(x))                 # mixer = GatedDeltaNet (linear) | Qwen35Attention (full)
    x = x + moe(post_attention_layernorm(x))          # MoE on *every* layer (512 experts top-10 + shared)

The mixer kind for layer ``i`` follows the config schedule (:meth:`Qwen35Config.is_linear_attention`
/ :meth:`~Qwen35Config.is_full_attention`): 45 Gated-DeltaNet *linear* layers + 15 gated-GQA *full*
layers (3:1). The model threads per-layer decode state: a growing KV cache on full-attention layers,
the ``(recurrent_state, conv_state)`` recurrence on linear layers, and nothing on the (stateless) MoE
sublayer. Weights are random-init here — enough to run and gate prefill==decode; the streamed loader
(``quanta.qwen35.loader``) filling these modules one layer at a time is a separate step (rule-8: a
single text layer resident).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.qwen35.attention import KVCache, Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.gated_deltanet import GatedDeltaNet
from quanta.qwen35.moe import qwen35_moe


class Qwen35MoEModule(nn.Module):
    """nn.Module wrapper over :func:`qwen35_moe` holding the router + pre-stacked expert + shared
    params, so a block owns its MoE weights (filled by the loader; random-init for the gate)."""

    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        h, e, inter = cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size
        si = cfg.shared_expert_intermediate_size
        self.gate = mx.zeros((e, h))
        self.experts_gate_up = mx.zeros((e, cfg.moe_gate_up_out, h))   # [E, 2*inter, h]
        self.experts_down = mx.zeros((e, h, inter))                    # [E, h, inter]
        self.shared_gate_proj = mx.zeros((si, h))
        self.shared_up_proj = mx.zeros((si, h))
        self.shared_down_proj = mx.zeros((h, si))
        self.shared_expert_gate = mx.zeros((1, h))
        self.token_chunk = 8192

    def set_experts(self, gate_up: mx.array, down: mx.array) -> None:
        self.experts_gate_up, self.experts_down = gate_up, down

    def _params(self) -> dict:
        return {
            "gate": self.gate,
            "experts_gate_up": self.experts_gate_up,
            "experts_down": self.experts_down,
            "shared_gate_proj": self.shared_gate_proj,
            "shared_up_proj": self.shared_up_proj,
            "shared_down_proj": self.shared_down_proj,
            "shared_expert_gate": self.shared_expert_gate,
        }

    def __call__(self, x, *, sparse: bool = True, topk_override: int | None = None):
        return qwen35_moe(x, self._params(), self.cfg, sparse=sparse,
                          token_chunk=self.token_chunk, topk_override=topk_override)


class Qwen35Block(nn.Module):
    """One decoder layer: ``x + mixer(in_norm(x))`` then ``x + moe(post_norm(x))``.

    Returns ``(out, recurrent_state, conv_state)`` — the linear-attention recurrence state passes
    through functionally; full-attention mutates its KV cache in place; for full layers the linear
    state passthroughs are ``None``."""

    def __init__(self, cfg: Qwen35Config, layer_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_linear = cfg.is_linear_attention(layer_id)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mixer = GatedDeltaNet(cfg) if self.is_linear else Qwen35Attention(cfg)
        self.mlp = Qwen35MoEModule(cfg)

    def __call__(self, x, *, cache=None, state=None, conv_state=None, use_fast=True,
                 seq_hint=None, sparse=True, topk_override: int | None = None):
        """Forward one block. ``topk_override`` is forwarded to the MoE so the MTP draft head can
        run a lighter MoE (top-1 / top-2) without changing the main-model path (which always uses
        ``cfg.num_experts_per_tok``). Lossless: the main model verifies every drafted token, and
        only the drafter's routing changes."""
        h = self.input_layernorm(x)
        if self.is_linear:
            y, state, conv_state = self.mixer(h, state=state, conv_state=conv_state)
        else:
            y = self.mixer(h, cache=cache, use_fast=use_fast, seq_hint=seq_hint)
        x = x + y
        x = x + self.mlp(self.post_attention_layernorm(x), sparse=sparse,
                         topk_override=topk_override)
        return x, state, conv_state


class Qwen35Model(nn.Module):
    def __init__(self, cfg: Qwen35Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Qwen35Block(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def make_state(self) -> tuple[list, list, list]:
        """Fresh per-layer decode state: KV caches (full layers) + zero recurrent/conv (linear)."""
        n = self.cfg.num_hidden_layers
        caches: list = [None] * n
        state: list = [None] * n
        conv: list = [None] * n
        for i, blk in enumerate(self.layers):
            if blk.is_linear:
                m = blk.mixer
                state[i] = mx.zeros((1, m.hv, m.dk, m.dv), dtype=mx.float32)
                conv[i] = mx.zeros((1, m.k - 1, m.conv_dim))
            else:
                caches[i] = KVCache()
        return caches, state, conv

    def __call__(self, token_ids, *, caches=None, state=None, conv=None, use_fast=True,
                 seq_hint=None, sparse=True):
        """Logits for ``token_ids`` ``[t]`` -> ``[1, t, vocab]`` plus updated ``(state, conv)`` lists.

        ``caches`` (per-full-layer KV caches) grow in place. All of ``caches``/``state``/``conv``
        ``None`` ⇒ a fresh prefill (chunked linear-attention, KV from offset 0); pass the prior
        state to continue (one token at a time for decode). ``seq_hint`` is the total sequence
        length for the dynamic-YaRN factor (so chunked prefill matches single-shot); defaults to the
        prefill length / cache-aware decode length."""
        h = self.embed_tokens(token_ids)[None]  # [1, t, hidden]
        n = len(self.layers)
        caches = caches if caches is not None else [None] * n
        state = state if state is not None else [None] * n
        conv = conv if conv is not None else [None] * n
        for i, blk in enumerate(self.layers):
            h, state[i], conv[i] = blk(h, cache=caches[i], state=state[i], conv_state=conv[i],
                                       use_fast=use_fast, seq_hint=seq_hint, sparse=sparse)
        return self.lm_head(self.norm(h)), state, conv
