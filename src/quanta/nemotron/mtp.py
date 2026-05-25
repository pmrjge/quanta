"""Nemotron-H native MTP (multi-token-prediction) draft head — the 1-head draft forward (MLX).

Grounded from the source checkpoint index (NVIDIA-Nemotron-3-Super-120B-A12B):
``num_nextn_predict_layers == 1`` — **one** native MTP head — but the head is materialized as **two
stacked mixer sub-blocks** under ``mtp.layers.0`` / ``mtp.layers.1`` (DeepSeek-V3-style):

* ``mtp.layers.0`` — the *fusion* + an **attention** mixer sub-block. Keys: ``enorm.weight`` /
  ``hnorm.weight`` (RMSNorms over the token-embedding and the previous hidden state), ``eh_proj.weight``
  (``[hidden, 2*hidden]`` — fuses ``concat([enorm(embed), hnorm(hidden)])`` → hidden), the attention
  ``mixer.{q,k,v,o}_proj.weight`` (a :class:`quanta.nemotron.attention.NemotronAttention`), and a
  per-sub-block pre-norm ``norm.weight``.
* ``mtp.layers.1`` — a **latent-MoE** mixer sub-block. Keys: the router ``mixer.gate.weight`` +
  ``mixer.gate.e_score_correction_bias``, ``mixer.fc1_latent_proj`` / ``mixer.fc2_latent_proj``, the
  routed ``mixer.experts.{e}.{up,down}_proj.weight`` and ``mixer.shared_experts.{up,down}_proj.weight``
  (a :class:`quanta.nemotron.moe.NemotronLatentMoE`), a per-sub-block pre-norm ``norm.weight``, and the
  head's ``final_layernorm.weight`` (the RMSNorm before the shared LM head).

So one MTP head = ``fuse(embed, prev_hidden) → attn-subblock → moe-subblock → final_layernorm → head``,
predicting the token at ``p+2`` from the running hidden at ``p`` plus the embedding of the token at
``p+1`` — exactly the EAGLE-style "self-speculation" feature, but using the checkpoint's own trained
head. One head ⇒ each spec round drafts **one** token (see :mod:`quanta.nemotron.spec`).

Each sub-block reuses the runtime's own :class:`quanta.nemotron.model.NemotronBlock` (which is precisely
``x + mixer(norm(x))`` with the right pre-norm), so the attention/MoE mixers, their RoPE/softmax, the
sparse top-k dispatch and the relu^2 latent experts are the *same* code paths as the main decoder — no
re-implementation (CLAUDE.md rule 1). The forward is pure; assembling the param dict from the ``mtp.*``
tensors is the loader/runtime's job. Gated structurally (model-free) in
``parity/nemotron_mtp_spec_test.py``; the real teacher-forced accept-rate is deferred (task #40).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.model import NemotronBlock


class NemotronMTPModule(nn.Module):
    """One MTP head: fuse(embed, prev_hidden) → attn sub-block → moe sub-block → final norm.

    Holds the fusion params (``enorm`` / ``hnorm`` / ``eh_proj``), the two stacked
    :class:`NemotronBlock` sub-blocks (an ``attention`` block typed from ``mtp.layers.0`` and a ``moe``
    block typed from ``mtp.layers.1``), and the head's ``final_layernorm``. The forward returns
    ``(logits, new_hidden)`` where ``new_hidden`` is the running residual stream *before* the final
    norm (the state a subsequent MTP module would chain on, mirroring DeepSeek-V3 sequential MTP), and
    ``logits`` is ``final_layernorm`` → shared LM head. The embedding/head matrices are the target
    model's (frozen, caller-held) — passed per call, never owned here (the EAGLE frozen-embed/head
    convention)."""

    def __init__(self, cfg: NemotronHConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.enorm = nn.RMSNorm(h, eps=cfg.norm_eps)
        self.hnorm = nn.RMSNorm(h, eps=cfg.norm_eps)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)         # concat([enorm(e), hnorm(hid)]) -> hidden
        self.attn_block = NemotronBlock(cfg, "attention")      # mtp.layers.0.{norm, mixer.*}
        self.moe_block = NemotronBlock(cfg, "moe")             # mtp.layers.1.{norm, mixer.*}
        self.final_layernorm = nn.RMSNorm(h, eps=cfg.norm_eps)

    def _fuse(self, prev_hidden: mx.array, token_emb: mx.array) -> mx.array:
        """Fuse the previous hidden ``[B,T,hidden]`` and the next-token embedding ``[B,T,hidden]`` into
        a fresh hidden ``[B,T,hidden]``: ``eh_proj(concat([enorm(embed), hnorm(prev_hidden)]))``."""
        e = self.enorm(token_emb.astype(prev_hidden.dtype))
        hid = self.hnorm(prev_hidden)
        return self.eh_proj(mx.concatenate([e, hid], axis=-1))

    def __call__(self, prev_hidden: mx.array, token_emb: mx.array, head: mx.array, *,
                 offset: int = 0, use_fast: bool = True) -> tuple[mx.array, mx.array]:
        """One MTP draft step. ``prev_hidden`` / ``token_emb``: ``[B,T,hidden]`` (the main model's hidden
        at the predicting positions, and the embedding of the next token); ``head``: the shared LM-head
        matrix ``[vocab, hidden]``. Returns ``(logits [B,T,vocab], new_hidden [B,T,hidden])``.

        The attention sub-block runs stateless (``cache=None``) over the given window — for a single
        drafted token (``T == 1``) that is a length-1 causal self-attention, and the main model verifies
        every draft, so the window/offset convention is a *speed* lever, never a correctness one
        (CLAUDE.md rule 4; same convention as :func:`quanta.dsv4.mtp.mtp_forward`)."""
        x = self._fuse(prev_hidden, token_emb)
        x, _, _ = self.attn_block(x, cache=None, use_fast=use_fast)   # x + attn(norm(x))
        x, _, _ = self.moe_block(x)                                   # x + moe(norm(x))  (stateless)
        normed = self.final_layernorm(x)
        logits = normed @ head.T.astype(normed.dtype)
        return logits, x


class NemotronMTP:
    """Holder over the native MTP head — the drafter the spec decoder consumes.

    Nemotron-H ships ``num_nextn_predict_layers == 1`` head (two sub-blocks), so this holds a single
    :class:`NemotronMTPModule`. A ``modules`` list is kept (length == ``num_nextn_predict_layers``) so
    the surface generalizes to a multi-head checkpoint without changing the spec loop; the loader fills
    each module's params. The call surface is uniform with the dsv4/glm drafters —
    ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)`` — so the spec loop and the model-free
    test stub mirror it. Holds no state across calls; the shared ``embed``/``head`` are the target
    model's and are threaded in by the caller."""

    def __init__(self, cfg: NemotronHConfig) -> None:
        self.cfg = cfg
        n = max(1, int(getattr(cfg, "num_nextn_predict_layers", 1)))
        self.modules: list[NemotronMTPModule] = [NemotronMTPModule(cfg) for _ in range(n)]

    @property
    def module(self) -> NemotronMTPModule:
        """The single MTP head (Nemotron ships exactly one)."""
        return self.modules[0]

    def __call__(self, prev_hidden: mx.array, token_emb: mx.array, head: mx.array, *,
                 offset: int = 0, use_fast: bool = True) -> tuple[mx.array, mx.array]:
        return self.module(prev_hidden, token_emb, head, offset=offset, use_fast=use_fast)
