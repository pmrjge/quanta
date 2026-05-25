"""GLM-5.1 (``glm_moe_dsa``) bake → a self-contained int4(AWQ)/int8/bf16 artifact (parity-first).

Streamed, one layer resident at a time (rule 8), mirroring the DSV4 bake (:mod:`quanta.dsv4.bake`) on
GLM's **plain pre-norm residual** stack (GLM has no Hyper-Connections). Per-tensor scheme follows the
project quant policy and this task's recipe:

* **routed experts** (the stacked ``gate_proj`` / ``up_proj`` / ``down_proj``) → **int4 AWQ g64**. The
  bf16 source has sub-int4-grid headroom, so AWQ's per-input-channel scale ``s`` buys back
  activation-weighted error vs plain RTN. ``gate_proj``/``up_proj`` calibrate on the expert's routed
  post-attention-norm rows; ``down_proj`` on the SwiGLU intermediate ``silu(gate·x)·(up·x)`` of those
  rows (no clamp — GLM has no ``swiglu_limit``). Cold experts (no routed rows) and
  ``expert_method="rtn"`` store plain int4 with ``s=1`` (identity scale → one runtime path: always
  divide the input by a stored ``s``);
* **non-expert matmuls** — MLA low-rank q/kv projections (``q_a_proj``/``q_b_proj``/
  ``kv_a_proj_with_mqa``/``kv_b_proj``), ``o_proj``, the DSA indexer (``wq_b``/``wk``/
  ``weights_proj``), the dense-FFN layers' ``gate_proj``/``up_proj``/``down_proj``, the always-on shared
  expert, and (if untied) ``lm_head`` → **int8** affine g``group_size``;
* **control tensors** — every norm (``input_layernorm`` / ``post_attention_layernorm`` / the MLA q/kv
  sub-norms / the indexer ``k_norm.{weight,bias}``), the router ``gate`` weight and its
  ``e_score_correction_bias``, the final norm, and the embeddings → **dense** verbatim. They are stored
  in their native dtype (never silently downcast — rule 6); the manifest records the true dtype and the
  runtime casts as needed.

Two streamed passes: (1) capture per-MoE-layer post-attention-norm activations + routing for AWQ
(:func:`quanta.glm.calibrate.capture_calibration`); (2) write the artifact (one layer's bf16 source
resident at a time, released before the next). The native MTP block (one full decoder layer at index
``num_hidden_layers``) is baked like a decoder layer. Runnable on a slice (``n_layers``,
``expert_subset``) for bounded validation; the full call is the real bake.

DEFERRED (needs the ~1.5 TB bf16 checkpoint + GPU; do NOT run in a model-free session)::

    from quanta.glm.bake import bake_glm
    bake_glm("/Users/pmrj/models/GLM-5.1", "/Users/pmrj/models/GLM-5.1-quanta_int4",
             calib_ids, group_size=64, expert_method="awq", scale_dtype=mx.bfloat16)
    # then teacher-forced ppl over the resident int4/int8 runtime vs the bf16 reference
    # (quanta.glm.model.glm_teacher_forced_ppl) — the only arbiter (see CLAUDE.md Settled Findings).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.awq import awq_quantize
from quanta.bake.calibrate import expert_rows
from quanta.bake.quant import quantize_affine
from quanta.glm.calibrate import capture_calibration
from quanta.glm.config import GLMConfig
from quanta.glm.loader import GLMSourceCheckpoint

EMBED, NORMF, HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
_EXPERT_BITS = 4
_INT8_BITS = 8

# The three SwiGLU projections of one expert / dense MLP / shared expert (loader stack order).
_EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")
# Non-expert MLA + indexer matmuls within the attention sub-dict (→ int8). Anything in the sub-dict
# not listed here is a control tensor (→ dense); a key in neither fails loud (rule 6).
_ATTN_MATMULS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
_ATTN_CONTROL = ("q_a_layernorm", "kv_a_layernorm")
_INDEXER_MATMULS = ("wq_b", "wk", "weights_proj")
# indexer LayerNorm carries a weight AND a bias; the loader names them k_norm_weight / k_norm_bias.
_INDEXER_CONTROL = {"k_norm_weight": "k_norm.weight", "k_norm_bias": "k_norm.bias"}


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                scale_dtype: mx.Dtype | None) -> None:
    """int8 affine-quantize a 2-D weight ``[out,in]`` and add it under ``key`` (no ``.weight``)."""
    writer.add_quantized(key, *quantize_affine(w, _INT8_BITS, gs, scale_dtype=scale_dtype),
                         _INT8_BITS, gs)


def _swiglu_inter(xe: mx.array, w_gate: mx.array, w_up: mx.array) -> mx.array:
    """down-proj calibration input: ``silu(gate·x) · (up·x)`` → ``[n, inter]`` (GLM has no clamp).

    Matches :meth:`quanta.glm.moe.SparseMoE._routed_chunk` (``nn.silu(g) * u``)."""
    xf = xe.astype(mx.float32)
    g = xf @ w_gate.astype(mx.float32).T
    u = xf @ w_up.astype(mx.float32).T
    return nn.silu(g) * u


def _bake_expert(writer: ArtifactWriter, base: str, w_gate: mx.array, w_up: mx.array,
                 w_down: mx.array, xe: mx.array | None, gs: int, method: str,
                 scale_dtype: mx.Dtype | None) -> int:
    """AWQ (or RTN) int4 the three SwiGLU expert matrices under ``base``
    (``.gate_proj``/``.up_proj``/``.down_proj``).

    Returns the routed-row count (0 = cold/RTN). ``gate_proj``/``up_proj`` calibrate on the expert's
    routed rows ``xe`` ``[n, hidden]``; ``down_proj`` on ``silu(g)·u`` of those rows. Cold experts and
    ``method='rtn'`` store plain int4 with ``s=1`` (identity scale) so the runtime always divides the
    input by a stored ``s`` (one path)."""
    if method == "awq" and xe is not None and xe.shape[0] > 0:
        for proj, w in (("gate_proj", w_gate), ("up_proj", w_up)):
            s, p, sc, b = awq_quantize(w, xe, _EXPERT_BITS, gs)
            writer.add_awq_quantized(f"{base}.{proj}", p, sc, b, s.astype(mx.bfloat16),
                                     _EXPERT_BITS, gs)
        inter = _swiglu_inter(xe, w_gate, w_up)  # [n, inter] down-proj input
        s, p, sc, b = awq_quantize(w_down, inter, _EXPERT_BITS, gs)
        writer.add_awq_quantized(f"{base}.down_proj", p, sc, b, s.astype(mx.bfloat16),
                                 _EXPERT_BITS, gs)
        return int(xe.shape[0])
    for proj, w in (("gate_proj", w_gate), ("up_proj", w_up), ("down_proj", w_down)):  # RTN / cold
        p, sc, b = quantize_affine(w, _EXPERT_BITS, gs, scale_dtype=scale_dtype)
        ones = mx.ones((w.shape[1],), dtype=mx.bfloat16)
        writer.add_awq_quantized(f"{base}.{proj}", p, sc, b, ones, _EXPERT_BITS, gs)
    return 0


def _bake_attention(writer: ArtifactWriter, ap: str, attn: dict, gs: int,
                    scale_dtype: mx.Dtype | None) -> None:
    """Attention sub-block: MLA q/kv/o matmuls int8, q/kv sub-norms dense, + the DSA indexer.

    ``ap`` is the artifact prefix ``layers.{i}.self_attn.``. The indexer is a nested sub-dict
    (``attn['indexer']``); its matmuls go int8 and its LayerNorm weight+bias go dense (rule 6: every
    key is dispatched, none silently dropped)."""
    for name, arr in attn.items():
        if name == "indexer":
            continue
        if name in _ATTN_MATMULS:
            _write_int8(writer, f"{ap}{name}", arr, gs, scale_dtype)
        elif name in _ATTN_CONTROL:
            writer.add_dense(f"{ap}{name}.weight", arr)
        else:
            raise ValueError(f"{ap}{name}: no quant policy (not an MLA matmul or sub-norm)")

    idx = attn["indexer"]
    ip = f"{ap}indexer."
    for name in _INDEXER_MATMULS:
        _write_int8(writer, f"{ip}{name}", idx[name], gs, scale_dtype)
    for src, dst in _INDEXER_CONTROL.items():  # k_norm weight + bias (LayerNorm) → dense
        writer.add_dense(f"{ip}{dst}", idx[src])


def _bake_router(writer: ArtifactWriter, rp: str, router: dict) -> None:
    """Router: gate weight + ``e_score_correction_bias`` → **dense** (bf16/f32 control; never int).

    ``rp`` is ``layers.{i}.mlp.gate.``. The gate steers selection (a tiny ``[E,hidden]`` matmul) and
    the correction bias steers it further; both are precision-sensitive control tensors (policy)."""
    writer.add_dense(f"{rp}weight", router["weight"])
    writer.add_dense(f"{rp}e_score_correction_bias", router["e_score_correction_bias"])


def _bake_shared(writer: ArtifactWriter, sp: str, shared: dict, gs: int,
                 scale_dtype: mx.Dtype | None) -> None:
    """Shared expert (always-on SwiGLU MLP) → int8 affine (per recipe). ``sp`` is the artifact prefix
    ``layers.{i}.mlp.shared_experts.``; keys are ``gate_proj``/``up_proj``/``down_proj``."""
    for proj in _EXPERT_PROJ:
        _write_int8(writer, f"{sp}{proj}", shared[proj], gs, scale_dtype)


def _bake_dense_mlp(writer: ArtifactWriter, mp: str, mlp: dict, gs: int,
                    scale_dtype: mx.Dtype | None) -> None:
    """Dense FFN (``first_k_dense_replace`` layers) → int8 affine. ``mp`` is ``layers.{i}.mlp.``."""
    for proj in _EXPERT_PROJ:
        _write_int8(writer, f"{mp}{proj}", mlp[proj], gs, scale_dtype)


def _bake_moe(writer: ArtifactWriter, i: int, ck: GLMSourceCheckpoint, cfg: GLMConfig,
              experts: list[int], caps: dict, gs: int, method: str,
              scale_dtype: mx.Dtype | None) -> int:
    """MoE FFN at layer ``i``: router dense, shared int8, routed experts int4-AWQ. Returns warm count.

    The expert stacks (``[E, out, in]``) are this layer's memory peak — loaded, quantized per expert,
    and dropped before returning (rule 8)."""
    _bake_router(writer, f"layers.{i}.mlp.gate.", ck.moe_router(i))
    _bake_shared(writer, f"layers.{i}.mlp.shared_experts.", ck.shared_expert(i), gs, scale_dtype)

    es = ck.expert_stacks(i)                       # {gate_proj,up_proj:[E,inter,hidden], down_proj:[E,hidden,inter]}
    x_cap, idx_cap = caps.get(i, (None, None))
    warm = 0
    for e in experts:
        base = f"layers.{i}.mlp.experts.{e}"
        xe = expert_rows(x_cap, idx_cap, e) if (method == "awq" and x_cap is not None) else None
        warm += int(_bake_expert(writer, base, es["gate_proj"][e], es["up_proj"][e],
                                 es["down_proj"][e], xe, gs, method, scale_dtype) > 0)
    del es
    return warm


def _bake_block(writer: ArtifactWriter, i: int, ck: GLMSourceCheckpoint, cfg: GLMConfig,
                experts: list[int], caps: dict, gs: int, method: str,
                scale_dtype: mx.Dtype | None) -> int:
    """One decoder block (attention + norms + dense-or-MoE FFN). Returns warm-expert count (0 if dense)."""
    _bake_attention(writer, f"layers.{i}.self_attn.", ck.attention(i), gs, scale_dtype)

    norms = ck.block_norms(i)
    writer.add_dense(f"layers.{i}.input_layernorm.weight", norms["input_layernorm"])
    writer.add_dense(f"layers.{i}.post_attention_layernorm.weight", norms["post_attention_layernorm"])

    if cfg.is_dense_layer(i):
        _bake_dense_mlp(writer, f"layers.{i}.mlp.", ck.dense_mlp(i), gs, scale_dtype)
        return 0
    return _bake_moe(writer, i, ck, cfg, experts, caps, gs, method, scale_dtype)


def _bake_mtp(writer: ArtifactWriter, ck: GLMSourceCheckpoint, cfg: GLMConfig, gs: int,
              method: str, scale_dtype: mx.Dtype | None) -> int:
    """Native MTP block (one full MoE decoder layer at index ``num_hidden_layers``) + its embed/hidden
    combine (``eh_proj`` int8; ``enorm``/``hnorm``/``shared_head.norm`` dense). Returns warm count."""
    i = cfg.mtp_layer_id
    t = ck.mtp(0)
    p = f"layers.{i}."

    _bake_attention(writer, f"{p}self_attn.", t["attention"], gs, scale_dtype)
    for name in ("enorm", "hnorm", "input_layernorm", "post_attention_layernorm"):
        writer.add_dense(f"{p}{name}.weight", t[name])
    writer.add_dense(f"{p}shared_head.norm.weight", t["shared_head_norm"])
    _write_int8(writer, f"{p}eh_proj", t["eh_proj"], gs, scale_dtype)

    _bake_router(writer, f"{p}mlp.gate.", t["router"])
    _bake_shared(writer, f"{p}mlp.shared_experts.", t["shared"], gs, scale_dtype)
    es = t["experts"]
    warm = 0
    for e in range(cfg.n_routed_experts):  # MTP experts always RTN (no calibration capture for it)
        warm += int(_bake_expert(writer, f"{p}mlp.experts.{e}", es["gate_proj"][e], es["up_proj"][e],
                                 es["down_proj"][e], None, gs, "rtn", scale_dtype) > 0)
    del t, es
    return warm


def bake_glm(
    source: str | Path,
    out_dir: str | Path,
    calib_ids: mx.array,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    include_mtp: bool = True,
    group_size: int = 64,
    expert_method: str = "awq",
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the GLM-5.1 bf16 source into a self-contained int4(AWQ g64)/int8/bf16 artifact at ``out_dir``.

    Returns a summary ``dict`` (bytes written, per-kind counts, warm experts). ``n_layers`` /
    ``expert_subset`` slice the bake for bounded validation; ``include_head`` toggles embed/norm/head,
    ``include_mtp`` the native MTP block (only when ``include_head`` and the whole stack is baked).
    """
    assert expert_method in ("awq", "rtn"), f"expert_method must be 'awq'|'rtn', got {expert_method!r}"
    cfg = GLMConfig.from_pretrained(source)
    ck = GLMSourceCheckpoint(source, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts = list(range(cfg.n_routed_experts)) if expert_subset is None else list(expert_subset)

    caps = capture_calibration(ck, cfg, calib_ids, n_layers=n) if expert_method == "awq" else {}

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")

    if include_head:
        writer.add_dense(EMBED, ck.embed())          # bf16 (logit-sensitive; policy)
        writer.add_dense(NORMF, ck.final_norm())
        if not cfg.tie_word_embeddings:
            _write_int8(writer, "lm_head", ck.lm_head(), group_size, scale_dtype)
        ck.release()

    warm = 0
    for i in range(n):
        warm += _bake_block(writer, i, ck, cfg, experts, caps, group_size, expert_method, scale_dtype)
        ck.release()
        mx.clear_cache()

    full = n_layers is None  # the MTP block belongs to the full stack only
    if include_head and include_mtp and full and cfg.num_nextn_predict_layers > 0:
        warm += _bake_mtp(writer, ck, cfg, group_size, expert_method, scale_dtype)
        ck.release()
        mx.clear_cache()

    counts = {"int8": 0, "dense": 0, "expert_int4": 0}
    for entry in writer.manifest.values():  # tally written kinds for the summary
        fmt = entry["format"]
        if fmt == "affine_packed":
            counts["int8"] += 1
        elif fmt == "awq_packed":
            counts["expert_int4"] += 1
        else:
            counts["dense"] += 1

    scale_tag = "bf16" if scale_dtype == mx.bfloat16 else "fp32"
    policy = {"experts": f"int4 {expert_method} g{group_size}",
              "non_experts": f"int8 g{group_size}",
              "norms_router_shared_embed_control": "bf16/f32", "scales": scale_tag}
    writer.finalize(policy)  # flushes shards + writes index/config/manifest

    out = Path(out_dir)
    total_bytes = sum(p.stat().st_size for p in out.glob("model-*.safetensors"))  # exact, on-disk
    return {"layers": n, "experts_per_layer": len(experts), "warm_experts": warm,
            "expert_method": expert_method,
            "mtp_baked": include_head and include_mtp and full and cfg.num_nextn_predict_layers > 0,
            "counts": counts, "bytes": total_bytes}
