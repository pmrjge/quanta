"""Qwen3.5-397B-A17B native MTP (multi-token-prediction) head — the 1-head draft forward (MLX).

Qwen3.5 ships a single native MTP module (``mtp_num_hidden_layers == 1``, ``qwen3_next_mtp``). It
predicts the token at position ``p+2`` from the embedding of the *just-emitted* token ``p+1`` plus the
main model's last hidden state at position ``p`` — the EAGLE-style "self-speculation" feature, but using
the checkpoint's own trained head instead of a learned drafter.

Faithful MLX port of the reference ``Qwen3NextMTP`` forward:

    e = pre_fc_norm_embedding(embed[next_ids])     # [B,T,hidden]  normalize the next-token embedding
    x = pre_fc_norm_hidden(prev_hidden)            # [B,T,hidden]  normalize the main model's hidden
    x = fc(concat[e, x])                           # [B,T,hidden]  fuse the two [.,hidden] -> [.,hidden]
    x = Block.forward(x, ...)                       # the inherited mtp.layers.0 decoder block (full-attn+MoE)
    logits = head(norm(x))                          # final RMSNorm + the (shared) LM head

``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden`` / ``norm`` are weighted RMSNorms over ``hidden``;
``fc`` is a bias-free linear over the concatenated ``[2*hidden]`` (``x @ fc.T``). The fused result is a
plain residual ``[B,T,hidden]`` (Qwen3.5 has no Hyper-Connections — unlike DSV4) fed straight into the
inherited :class:`quanta.qwen35.model.Qwen35Block` (always **full-attention**), then read out via the
MTP ``norm`` + the shared LM head — mirroring the main model's tail.

The inherited block is reused verbatim (the SAME ``Qwen35Block`` forward prefill uses), so the MTP head
adds no new attention / MoE math; it only assembles the combine + readout around it. The MTP block's
routed experts ship **per-expert** in the source (un-fused), so the runtime fuses gate+up into the
``[E, 2*inter, hidden]`` stack the block's MoE consumes (see :func:`build_mtp_block`).

This module is the pure forward, gated against the reference in ``parity/qwen35_mtp_spec_test.py``
(structurally, model-free) and — deferred — the real teacher-forced check.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.model import Qwen35Block


def _rms_w(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """Weighted RMSNorm over the last dim (fp32 reduction), matching the model's norms."""
    return mx.fast.rms_norm(x.astype(mx.float32), w.astype(mx.float32), eps).astype(x.dtype)


def mtp_combine(prev_hidden: mx.array, next_ids: mx.array, embed: mx.array, p: dict,
                cfg: Qwen35Config) -> mx.array:
    """MTP input combine → a fresh residual stream ``[B,T,hidden]``.

    ``prev_hidden``: the main model's hidden ``[B,T,hidden]`` at the predicting positions; ``next_ids``:
    the next-token ids ``[B,T]`` (the tokens whose embeddings seed the prediction); ``embed``: the shared
    embedding table ``[vocab,hidden]``. Mirrors the reference:
    ``fc(concat[ pre_fc_norm_embedding(embed[next_ids]), pre_fc_norm_hidden(prev_hidden) ])``.
    """
    eps = cfg.norm_eps
    e = _rms_w(embed[next_ids].astype(prev_hidden.dtype), p["pre_fc_norm_embedding"], eps)  # [B,T,hidden]
    x = _rms_w(prev_hidden, p["pre_fc_norm_hidden"], eps)                                   # [B,T,hidden]
    cat = mx.concatenate([e, x], axis=-1)                                                   # [B,T,2*hidden]
    return cat @ p["fc"].T.astype(cat.dtype)                                                # [B,T,hidden]


def mtp_forward(prev_hidden: mx.array, next_ids: mx.array, embed: mx.array, head: mx.array,
                p: dict, cfg: Qwen35Config, block: Qwen35Block) -> mx.array:
    """One MTP head's next-token logits ``[B,T,vocab]``.

    ``prev_hidden``: the main model's hidden ``[B,T,hidden]`` (before the final norm) at the predicting
    positions; ``next_ids``: the next-token ids ``[B,T]``; ``embed``/``head``: the shared embedding
    ``[vocab,hidden]`` and LM-head ``[vocab,hidden]`` matrices; ``p``: the combine/readout param dict
    (``fc`` / ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden`` / ``norm``); ``block``: the inherited
    full-attn+MoE decoder block (a :class:`Qwen35Block`). Faithful to the reference ``Qwen3NextMTP``.

    The inherited block is run over the absolute positions ``[0, T)`` of the given window (the reference's
    teacher-forced convention; the block here does not thread a decode offset). The MTP head only changes
    *which* token is drafted, never correctness — the main model verifies every draft (see
    :mod:`quanta.qwen35.spec`) — so the window convention is a speed lever, not a correctness one.
    """
    h = mtp_combine(prev_hidden, next_ids, embed, p, cfg)             # [B,T,hidden]
    h, _, _ = block(h, cache=None, state=None, conv_state=None)       # inherited full-attn + MoE block
    h = _rms_w(h, p["norm"], cfg.norm_eps)                            # MTP final norm
    return h @ head.T.astype(h.dtype)


def build_mtp_block(art, cfg: Qwen35Config) -> tuple[Qwen35Block, dict]:
    """Assemble the MTP head from a :class:`quanta.qwen35.artifact.Qwen35Artifact`.

    Returns ``(block, p)``: a runnable full-attention :class:`Qwen35Block` (its routed experts fused
    gate+up from the per-expert MTP stacks) plus the combine/readout param dict
    (``fc`` / ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden`` / ``norm``). The MTP block types as a
    full-attention layer regardless of the main schedule, so we force ``layer_types`` to a full slot.
    """
    t = art.mtp(0)
    # the MTP decoder block is ALWAYS full attention — pick any full-attention layer id for typing
    full_id = next(i for i in range(cfg.num_hidden_layers) if cfg.is_full_attention(i))
    blk = Qwen35Block(cfg, full_id)
    assert isinstance(blk.mixer, Qwen35Attention)
    blk.input_layernorm.weight = t["input_layernorm"]
    blk.post_attention_layernorm.weight = t["post_attention_layernorm"]
    attn = t["attention"]
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(blk.mixer, proj).weight = attn[f"{proj}.weight"]
    blk.mixer.q_norm = attn["q_norm.weight"]
    blk.mixer.k_norm = attn["k_norm.weight"]
    moe = t["moe"]
    blk.mlp.gate = moe["gate"]
    # MTP source stores experts un-fused (separate gate/up); fuse to [E, 2*inter, hidden] for the block.
    gate_up = mx.concatenate([moe["experts_gate_proj"], moe["experts_up_proj"]], axis=1)
    blk.mlp.set_experts(gate_up, moe["experts_down_proj"])
    blk.mlp.shared_gate_proj = moe["shared_gate_proj"]
    blk.mlp.shared_up_proj = moe["shared_up_proj"]
    blk.mlp.shared_down_proj = moe["shared_down_proj"]
    blk.mlp.shared_expert_gate = moe["shared_expert_gate"]
    p = {k: t[k] for k in ("fc", "pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm")}
    return blk, p


class MTPHead:
    """Callable wrapper over the native MTP block — the drafter the spec decoder consumes.

    Closes over the assembled inherited ``block`` + the combine/readout ``p`` dict + the config so the
    spec loop can draft with a uniform ``mtp(prev_hidden, next_ids, embed, head) -> logits`` call (the
    duck-typed surface a test stub mirrors). The wrapper holds no state across calls; the shared
    ``embed``/``head`` are passed in per call (they are the main model's, not MTP-owned).
    """

    def __init__(self, block: Qwen35Block, p: dict, cfg: Qwen35Config) -> None:
        self.block = block
        self.p = p
        self.cfg = cfg

    @classmethod
    def from_artifact(cls, art, cfg: Qwen35Config) -> "MTPHead":
        """Build directly from a :class:`Qwen35Artifact` (assembles + evals the MTP block)."""
        blk, p = build_mtp_block(art, cfg)
        mx.eval([v for v in p.values()]
                + [blk.mlp.experts_gate_up, blk.mlp.experts_down])
        return cls(blk, p, cfg)

    def __call__(self, prev_hidden: mx.array, next_ids: mx.array, embed: mx.array,
                 head: mx.array) -> mx.array:
        return mtp_forward(prev_hidden, next_ids, embed, head, self.p, self.cfg, self.block)
