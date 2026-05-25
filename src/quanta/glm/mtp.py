"""GLM-5.1 native MTP (multi-token-prediction) head — the 1-head draft forward (MLX).

GLM-5.1 ships a single native MTP block (``num_nextn_predict_layers == 1``) at source layer
``num_hidden_layers`` (DeepSeek-V3 nextn style). It predicts the token at position ``p+2`` from the main
model's last hidden state at position ``p`` plus the embedding of the *just-emitted* token ``p+1`` — the
EAGLE-style self-speculation feature, using the checkpoint's own trained head instead of a learned drafter.

Faithful to the DeepSeek-V3 ``DeepseekV3MTP``/``nextn`` forward (concat-fuse, like
:class:`quanta.nemotron.mtp.NemotronMTPModule`; GLM has **no** Hyper-Connections, so the residual is a
plain ``[B,T,dim]`` stream, unlike the DSV4 additive ``e_proj + h_proj``):

    e = enorm(embed[next_ids])                       # [B,T,d]   normalize the next-token embedding
    h = hnorm(prev_hidden)                           # [B,T,d]   normalize the main model's hidden
    x = eh_proj(concat([e, h], axis=-1))             # [B,T,d]   fuse into a fresh residual stream
    x = GLMDecoderLayer.forward(x, ...)              # the MTP layer's inherited decoder block
    logits = (rmsnorm(x, shared_head.norm)) @ head.T # readout through the shared-head norm + LM head

``enorm`` / ``hnorm`` / ``shared_head.norm`` are weighted RMSNorms over ``d``; ``eh_proj`` is a bias-free
``[d, 2d]`` linear (``concat`` then project). The inherited block is a full GLM MoE decoder block
(:class:`quanta.glm.model.GLMDecoderLayer`) — the *same* attention / DSA indexer / sparse MoE code the
main decoder uses (rule 1, no reimplementation).

:func:`build_mtp_layer` assembles the inherited :class:`GLMDecoderLayer` from the loader/artifact
``mtp()`` dict (the same per-layer wiring as :func:`quanta.glm.model.load_block`); :func:`mtp_forward`
is the pure forward and :class:`MTPHead` the callable the spec loop consumes with the exact signature
``mtp(prev_hidden, next_ids, embed, head) -> logits`` that :mod:`quanta.glm.spec` already calls. Mirrors
:mod:`quanta.dsv4.mtp`. Gated structurally model-free in ``parity/glm_mtp_spec_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.glm.config import GLMConfig
from quanta.glm.model import GLMDecoderLayer, _rms_w  # reuse the exact block RMSNorm (single source)


def build_mtp_layer(p: dict, cfg: GLMConfig, dtype: mx.Dtype = mx.bfloat16) -> GLMDecoderLayer:
    """Build the inherited GLM decoder block of the MTP head from the loader/artifact ``mtp()`` dict.

    ``p`` is the dict :meth:`quanta.glm.loader.GLMSourceCheckpoint.mtp` / :meth:`quanta.glm.artifact.GLMArtifact.mtp`
    return: the combine params (``enorm`` / ``hnorm`` / ``eh_proj`` / ``shared_head_norm``) plus the
    inherited ``attention`` (incl. ``indexer``) / ``router`` / ``shared`` / ``experts`` and the block
    norms. The MTP block is a full MoE layer, so it types as the MTP layer id (``cfg.mtp_layer_id``,
    which is ``>= first_k_dense_replace`` ⇒ a :class:`quanta.glm.moe.SparseMoE` mlp). Mirrors the per-layer
    wiring of :func:`quanta.glm.model.load_block` (no reimplementation)."""
    layer = GLMDecoderLayer(cfg, cfg.mtp_layer_id)
    a = p["attention"]
    ix = a["indexer"]
    upd: dict[str, mx.array] = {
        "input_layernorm.weight": p["input_layernorm"],
        "post_attention_layernorm.weight": p["post_attention_layernorm"],
        "self_attn.q_a_proj.weight": a["q_a_proj"],
        "self_attn.q_a_layernorm.weight": a["q_a_layernorm"],
        "self_attn.q_b_proj.weight": a["q_b_proj"],
        "self_attn.kv_a_proj_with_mqa.weight": a["kv_a_proj_with_mqa"],
        "self_attn.kv_a_layernorm.weight": a["kv_a_layernorm"],
        "self_attn.kv_b_proj.weight": a["kv_b_proj"],
        "self_attn.o_proj.weight": a["o_proj"],
        "indexer.wq_b.weight": ix["wq_b"],
        "indexer.wk.weight": ix["wk"],
        "indexer.weights_proj.weight": ix["weights_proj"],
        "indexer.k_norm.weight": ix["k_norm_weight"],
        "indexer.k_norm.bias": ix["k_norm_bias"],
    }
    layer.load_weights([(k, v.astype(dtype)) for k, v in upd.items()], strict=False)
    r = p["router"]
    sh = p["shared"]
    es = p["experts"]
    layer.mlp.gate.weight = r["weight"].astype(dtype)
    layer.mlp.gate.e_score_correction_bias = r["e_score_correction_bias"].astype(mx.float32)
    layer.mlp.load_weights([
        ("shared_gate.weight", sh["gate_proj"].astype(dtype)),
        ("shared_up.weight", sh["up_proj"].astype(dtype)),
        ("shared_down.weight", sh["down_proj"].astype(dtype)),
    ], strict=False)
    layer.mlp.set_experts(es["gate_proj"].astype(dtype), es["up_proj"].astype(dtype),
                          es["down_proj"].astype(dtype))
    return layer


def mtp_combine(prev_hidden: mx.array, next_ids: mx.array, embed: mx.array, p: dict,
                cfg: GLMConfig) -> mx.array:
    """MTP input fuse → a fresh residual stream ``[B,T,dim]`` (DeepSeek-V3 nextn concat convention).

    ``prev_hidden``: the main model's hidden ``[B,T,dim]`` at the predicting positions; ``next_ids``:
    the next-token ids ``[B,T]`` (the tokens whose embeddings seed the prediction); ``embed``: the
    shared embedding table ``[vocab,dim]``. Computes ``eh_proj(concat([enorm(embed[next_ids]),
    hnorm(prev_hidden)]))`` (no bias)."""
    eps = cfg.rms_norm_eps
    e = _rms_w(embed[next_ids].astype(prev_hidden.dtype), p["enorm"], eps)   # [B,T,dim]
    h = _rms_w(prev_hidden, p["hnorm"], eps)                                 # [B,T,dim]
    fused = mx.concatenate([e, h], axis=-1)                                  # [B,T,2*dim]
    return fused @ p["eh_proj"].T.astype(fused.dtype)                        # [B,T,dim]


def mtp_forward(prev_hidden: mx.array, next_ids: mx.array, embed: mx.array, head: mx.array,
                p: dict, layer: GLMDecoderLayer, cfg: GLMConfig, *, use_fast: bool = False,
                use_indexer: bool = True) -> mx.array:
    """One MTP head's next-token logits ``[B,T,vocab]``.

    ``prev_hidden``: the main model's hidden stream ``[B,T,dim]`` (before the final norm) at the
    predicting positions; ``next_ids``: the next-token ids ``[B,T]``; ``embed`` / ``head``: the shared
    embedding ``[vocab,dim]`` and LM-head ``[vocab,dim]`` matrices; ``p``: the combine/readout params
    (``enorm`` / ``hnorm`` / ``eh_proj`` / ``shared_head_norm``); ``layer``: the inherited GLM decoder
    block (:func:`build_mtp_layer`). Faithful to the DeepSeek-V3 nextn forward.

    The inherited block runs over the absolute positions ``[0, T)`` of the given window (the reference's
    teacher-forced convention; this prefill block does not thread a decode offset). The MTP head only
    changes *which* token is drafted, never correctness — the main model verifies every draft (see
    :mod:`quanta.glm.spec`) — so the window convention is a speed lever, not a correctness one (rule 4)."""
    t = next_ids.shape[1]
    h = mtp_combine(prev_hidden, next_ids, embed, p, cfg)                    # [B,T,dim]
    positions = mx.arange(t)
    h = layer(h, positions, use_fast=use_fast, use_indexer=use_indexer)     # inherited decoder block
    h = _rms_w(h, p["shared_head_norm"], cfg.rms_norm_eps)
    return h @ head.T.astype(h.dtype)


class MTPHead:
    """Callable wrapper over the native MTP block — the drafter the spec decoder consumes.

    Closes over the combine/readout params ``p`` and the assembled inherited block ``layer`` so the spec
    loop drafts with a uniform ``mtp(prev_hidden, next_ids, embed, head) -> logits`` call (the exact
    surface :mod:`quanta.glm.spec` calls and the model-free test stub mirrors). Holds no state across
    calls; the shared ``embed`` / ``head`` are the main model's (not MTP-owned) and are passed in per
    call (the EAGLE frozen-embed/head convention)."""

    def __init__(self, p: dict, layer: GLMDecoderLayer, cfg: GLMConfig, *,
                 use_fast: bool = False, use_indexer: bool = True) -> None:
        self.p = p
        self.layer = layer
        self.cfg = cfg
        self.use_fast = use_fast
        self.use_indexer = use_indexer

    def __call__(self, prev_hidden: mx.array, next_ids: mx.array, embed: mx.array,
                 head: mx.array) -> mx.array:
        return mtp_forward(prev_hidden, mx.array(next_ids).reshape(1, -1), embed, head, self.p,
                           self.layer, self.cfg, use_fast=self.use_fast, use_indexer=self.use_indexer)
