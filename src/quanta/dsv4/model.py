"""DeepSeek-V4 full decoder forward (MLX, streamed) — embed -> HC -> 43 blocks -> HC-head -> logits.

Assembles the validated components into the reference forward (faithful to ``model.py`` ``Block.forward``
/ ``Transformer.forward``):

* ``h = hc_expand(embed[ids])`` — the residual stream carries ``hc_mult`` copies ``[B,S,hc,dim]``.
* each block: ``hc_pre`` (reduce copies) -> ``attn_norm`` -> attention (dense for ratio-0 layers,
  compressed+indexer otherwise) -> ``hc_post`` (expand); then ``hc_pre`` -> ``ffn_norm`` -> MoE
  (hash/sqrtsoftplus) -> ``hc_post``.
* ``logits = (rmsnorm(hc_head(h)) @ head.T)``.

Weights stream one layer at a time via :class:`quanta.dsv4.loader.DeepSeekV4SourceCheckpoint`
(rule-8: ≤1 layer resident — each layer's bf16 expert stacks are the memory peak, freed before the
next). This is the bf16/f32 reference forward; the int4/int8 resident runtime is task #77. Gated
block-by-block (and final head) against the authors' real code in ``parity/dsv4_forward_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.attention import _rms_w, attention_dense
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.hyper import hc_expand, hc_head, hc_post, hc_pre
from quanta.dsv4.indexer import attention_compressed
from quanta.dsv4.moe import dsv4_moe


def dsv4_block(h: mx.array, p: dict, cfg: DeepSeekV4Config, layer_id: int,
               ids: mx.array, *, topk_override: int | None = None) -> mx.array:
    """One decoder block on the HC residual stream ``h`` ``[B,S,hc,dim] -> [B,S,hc,dim]``.

    ``topk_override`` (optional) is forwarded to :func:`dsv4_moe` so the MTP draft head can run
    a lighter MoE (top-1 / top-2) without changing the main-model path (which always uses
    ``cfg.num_experts_per_tok``). Lossless: the main model verifies every drafted token, and
    only the drafter's routing changes."""
    eps, hc, iters, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
    attn = attention_compressed if cfg.has_compressor(layer_id) else attention_dense

    res = h
    x, post, comb = hc_pre(h, p["hc_attn_fn"], p["hc_attn_scale"], p["hc_attn_base"], hc, iters, eps, heps)
    x = _rms_w(x, p["attn_norm"], eps)
    x = attn(x, p["attn"], cfg, layer_id)
    h = hc_post(x, res, post, comb)

    res = h
    x, post, comb = hc_pre(h, p["hc_ffn_fn"], p["hc_ffn_scale"], p["hc_ffn_base"], hc, iters, eps, heps)
    x = _rms_w(x, p["ffn_norm"], eps)
    x = dsv4_moe(x, p["router"], p["experts"], p["shared"], cfg, layer_id, ids,
                 topk_override=topk_override)
    h = hc_post(x, res, post, comb)
    return h


def load_block_params(ck, cfg: DeepSeekV4Config, layer_id: int, dtype: mx.Dtype = mx.bfloat16,
                      *, packed_experts: bool = False) -> dict:
    """Materialize one block's params (attention + router + expert stacks + shared + norms + HC).

    With ``packed_experts=True`` (the resident decode path, #141) the routed experts are loaded
    as int4 packed dicts (``{packed, scale, bias, awq_scale, group_size, bits}``) instead of
    bf16 ``[E,*,*]`` stacks. Attention / shared / norms / HC stay bf16/f32 — they're small.
    ``dsv4_moe`` auto-detects the packed shape and routes through ``mx.gather_qmm``."""
    def cast(d):
        return {k: (cast(v) if isinstance(v, dict) else v.astype(dtype)) for k, v in d.items()}

    if packed_experts and hasattr(ck, "expert_stacks_packed"):
        experts = ck.expert_stacks_packed(layer_id)
    else:
        experts = {k: v.astype(dtype) for k, v in ck.expert_stacks(layer_id).items()}

    p = {"attn": cast(ck.attention(layer_id)),
         "router": ck.moe_router(layer_id),                          # gate.weight/bias bf16, tid2eid int
         "experts": experts,
         "shared": cast(ck.shared_expert(layer_id))}
    p.update({k: v.astype(dtype) for k, v in ck.block_norms(layer_id).items()})
    p.update({k: v.astype(mx.float32) for k, v in ck.block_hc(layer_id).items()})   # HC params stay f32
    return p


def dsv4_logits(ck, ids: mx.array, cfg: DeepSeekV4Config, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Teacher-forced logits ``[B,S,vocab]`` for token ids ``[B,S]`` (streamed, one layer resident)."""
    h = hc_expand(ck.embed()[ids].astype(dtype), cfg.hc_mult)        # [B,S,hc,dim]
    for layer_id in range(cfg.num_hidden_layers):
        p = load_block_params(ck, cfg, layer_id, dtype)
        h = dsv4_block(h, p, cfg, layer_id, ids)
        mx.eval(h)
        ck.release()
    fhc = ck.final_hc()
    hh = hc_head(h, fhc["fn"], fhc["scale"], fhc["base"], cfg.hc_mult, cfg.norm_eps, cfg.hc_eps)
    hh = _rms_w(hh, ck.final_norm(), cfg.norm_eps)
    return hh @ ck.head().T.astype(hh.dtype)


def teacher_forced_ppl(ck, ids: mx.array, cfg: DeepSeekV4Config, dtype: mx.Dtype = mx.bfloat16) -> float:
    """Mean teacher-forced perplexity of ``ids`` ``[1,S]`` (next-token CE over positions 0..S-2)."""
    logits = dsv4_logits(ck, ids, cfg, dtype).astype(mx.float32)[0]   # [S, vocab]
    tgt = ids[0, 1:]
    lse = mx.logsumexp(logits[:-1], axis=-1)
    tok = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
    return float(mx.exp(mx.mean(lse - tok)).item())
