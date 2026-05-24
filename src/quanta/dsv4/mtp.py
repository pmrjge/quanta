"""DeepSeek-V4-Flash native MTP (multi-token-prediction) head — the 1-head draft forward (MLX).

DSV4 ships a single native MTP block (``num_nextn_predict_layers == 1``). It predicts the token at
position ``p+1`` from the embedding of the *just-emitted* token ``p+1`` plus the main model's last
hidden state at position ``p`` — exactly the EAGLE-style "self-speculation" feature, but using the
checkpoint's own trained head instead of a learned drafter.

Faithful MLX port of the reference ``model.py`` ``MTPBlock.forward`` (DeepSeek-V4-Flash inference):

    e = enorm(embed[next_ids])                 # [B,T,d]   normalize the next-token embedding
    x = hnorm(prev_hidden)                      # [B,T,hc,d] normalize the main model's HC residual
    x = e_proj(e)[:, :, None, :] + h_proj(x)    # [B,T,hc,d] combine into a fresh HC residual stream
    x = Block.forward(x, ...)                   # the MTP layer's inherited decoder block (dsv4_block)
    logits = head(x) = lm_head(rmsnorm_w(hc_head(x), norm))

``enorm``/``hnorm``/``norm`` are weighted RMSNorms over ``d``; ``e_proj``/``h_proj`` are bias-free
linears (``x @ W.T``, matching the reference ``Linear(bias=False)``). The combine adds the (broadcast)
embedding projection across the ``hc`` copies, so the MTP input is again an HC residual ``[B,T,hc,d]``
fed straight into the inherited :func:`quanta.dsv4.model.dsv4_block`. The final reduction reuses
:func:`quanta.dsv4.hyper.hc_head` + the weighted RMSNorm + the (shared) LM head, mirroring
:func:`quanta.dsv4.model.dsv4_logits`'s tail.

Params are a ``p`` dict mirroring the loader/artifact field names (see
:meth:`quanta.dsv4.loader.DeepSeekV4SourceCheckpoint.mtp` for ``e_proj``/``h_proj``/``enorm``/``hnorm``/
``norm``/``hc_head_{fn,base,scale}``) **plus** the inherited Block params that
:func:`quanta.dsv4.model.dsv4_block` consumes (``attn``/``router``/``experts``/``shared``/
``attn_norm``/``ffn_norm``/``hc_attn_*``/``hc_ffn_*``). Assembling that dict from the ``mtp.0.*``
tensors is the loader/runtime's job; this module is the pure forward, gated against the reference in
``parity/dsv4_mtp_spec_test.py`` (structurally, model-free) and — deferred — the real teacher-forced
check for task #78.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.attention import _rms_w
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.hyper import hc_head
from quanta.dsv4.model import dsv4_block


def mtp_layer_id(cfg: DeepSeekV4Config, j: int = 0) -> int:
    """The compress_ratios index of MTP head ``j``: the ``j``-th block past the main decoder.

    The reference appends ``MTPBlock(args.n_layers + layer_id, args)`` for each MTP head, so head
    ``j`` types as layer ``num_hidden_layers + j`` (its attention regime / RoPE come from
    ``compress_ratios[num_hidden_layers + j]``)."""
    lid = cfg.num_hidden_layers + j
    if not 0 <= lid < len(cfg.compress_ratios):
        raise IndexError(
            f"MTP head {j} types as layer {lid} but compress_ratios has length "
            f"{len(cfg.compress_ratios)} (cannot type the MTP block; refusing to guess)"
        )
    return lid


def mtp_combine(prev_hidden: mx.array, next_ids: mx.array, embed: mx.array, p: dict,
                cfg: DeepSeekV4Config) -> mx.array:
    """MTP input combine → a fresh HC residual stream ``[B,T,hc,d]``.

    ``prev_hidden``: the main model's HC residual ``[B,T,hc,d]`` at the predicting positions;
    ``next_ids``: the next-token ids ``[B,T]`` (the tokens whose embeddings seed the prediction);
    ``embed``: the shared embedding table ``[vocab,d]``. Mirrors ``MTPBlock.forward``:
    ``e_proj(enorm(embed[next_ids]))`` broadcast over ``hc`` + ``h_proj(hnorm(prev_hidden))``."""
    eps = cfg.norm_eps
    e = _rms_w(embed[next_ids].astype(prev_hidden.dtype), p["enorm"], eps)   # [B,T,d]
    x = _rms_w(prev_hidden, p["hnorm"], eps)                                 # [B,T,hc,d]
    e_proj = (e @ p["e_proj"].T.astype(e.dtype))[:, :, None, :]              # [B,T,1,d] -> bcast hc
    h_proj = x @ p["h_proj"].T.astype(x.dtype)                              # [B,T,hc,d]
    return e_proj + h_proj


def mtp_forward(prev_hidden: mx.array, next_ids: mx.array, embed: mx.array, head: mx.array,
                p: dict, cfg: DeepSeekV4Config, j: int = 0) -> mx.array:
    """One MTP head's next-token logits ``[B,T,vocab]``.

    ``prev_hidden``: the main model's HC residual stream ``[B,T,hc,d]`` (the hidden *before* the final
    HC-head reduction) at the predicting positions; ``next_ids``: the next-token ids ``[B,T]``;
    ``embed``/``head``: the shared embedding ``[vocab,d]`` and LM-head ``[vocab,d]`` matrices; ``p``:
    the MTP block param dict (combine/head params + the inherited Block params consumed by
    :func:`dsv4_block`). Faithful to the reference ``MTPBlock.forward``.

    The inherited block is run via :func:`dsv4_block` over the absolute positions ``[0, T)`` of the
    given window (the reference's teacher-forced convention; ``dsv4_block`` does not thread a decode
    offset). The MTP head only changes *which* token is drafted, never correctness — the main model
    verifies every draft (see :mod:`quanta.dsv4.spec`) — so the window convention is a speed lever,
    not a correctness one."""
    lid = mtp_layer_id(cfg, j)
    h = mtp_combine(prev_hidden, next_ids, embed, p, cfg)             # [B,T,hc,d]
    h = dsv4_block(h, p, cfg, lid, next_ids)                          # inherited decoder block
    hh = hc_head(h, p["hc_head_fn"], p["hc_head_scale"], p["hc_head_base"],
                 cfg.hc_mult, cfg.norm_eps, cfg.hc_eps)               # [B,T,d]
    hh = _rms_w(hh, p["norm"], cfg.norm_eps)
    return hh @ head.T.astype(hh.dtype)


class MTPHead:
    """Callable wrapper over the native MTP block — the drafter the spec decoder consumes.

    Closes over the assembled MTP param dict ``p`` and the config so the spec loop can draft with a
    uniform ``mtp(prev_hidden, next_ids, embed, head) -> logits`` call (the duck-typed surface a test
    stub mirrors). The wrapper holds no state across calls; the shared ``embed``/``head`` are passed
    in per call (they are the main model's, not MTP-owned), matching :class:`quanta.eagle.spec`'s
    frozen-embed/head convention."""

    def __init__(self, p: dict, cfg: DeepSeekV4Config, j: int = 0) -> None:
        self.p = p
        self.cfg = cfg
        self.j = j

    def __call__(self, prev_hidden: mx.array, next_ids: mx.array, embed: mx.array,
                 head: mx.array) -> mx.array:
        return mtp_forward(prev_hidden, next_ids, embed, head, self.p, self.cfg, self.j)
