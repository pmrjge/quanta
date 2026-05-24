"""DeepSeek-V4-Flash bake → a self-contained int4(AWQ)/int8/bf16 artifact (parity-first).

Streamed, one layer resident at a time (rule-8), mirroring the Nemotron bake
(:mod:`quanta.nemotron.bake`). Per-tensor scheme follows the project quant policy:

* **routed experts** (the stacked ``w1`` gate / ``w3`` up / ``w2`` down) → **int4 AWQ** g128. The
  fp4 source dequantizes to bf16 with sub-int4-grid headroom, so AWQ's per-input-channel scale ``s``
  buys back activation-weighted error vs plain RTN. ``w1``/``w3`` calibrate on the expert's routed
  post-FFN-norm rows; ``w2`` on the SwiGLU intermediate of those rows. Cold experts (no routed rows)
  and ``expert_method="rtn"`` store plain int4 with ``s=1`` (one runtime path: always divide by ``s``);
* **non-expert matmuls** (attention ``wq_a/wq_b/wkv/wo_a/wo_b``, compressor ``wkv/wgate``, indexer
  ``wq_b/weights_proj``, MoE router gate, MTP ``e_proj/h_proj``, LM head) → **int8** affine g128;
* **control tensors** (all norms, Hyper-Connection ``fn/base/scale``, ``attn_sink``, ``ape``, router
  ``bias``/``tid2eid``, shared expert, embeddings) → **dense** verbatim. They are stored in their
  native dtype (bf16 norms/embed, f32 HC/sink/ape) — never silently downcast (rule-6); the manifest
  records the true dtype and the runtime casts as needed.

Two streamed passes: (1) capture per-layer post-FFN-norm activations + routing for AWQ; (2) write the
artifact (one block's bf16 source resident at a time, released before the next). The native MTP block
is baked like a decoder layer (matmuls int8, control dense). Runnable on a slice (``n_layers``,
``expert_subset``) for bounded validation; the full call is the real bake.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.awq import awq_quantize
from quanta.bake.calibrate import expert_rows
from quanta.bake.quant import quantize_affine
from quanta.dsv4.calibrate import capture_calibration
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from quanta.dsv4.moe import silu

EMBED, NORMF, HEAD = "embed.weight", "norm.weight", "head.weight"
_EXPERT_BITS = 4
_INT8_BITS = 8

# Non-expert matmuls (→ int8) within each loader sub-dict, keyed by the loader's short name and the
# source-name infix used to build the artifact key. Anything in a sub-dict not listed here is a
# control tensor (→ dense). Fail-loud coverage is enforced in `_write_sub` (rule-6).
_ATTN_MATMULS = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
_ATTN_CONTROL = ("q_norm", "kv_norm", "attn_sink")
_COMPRESSOR_MATMULS = ("wkv", "wgate")
_COMPRESSOR_CONTROL = ("ape", "norm")
_INDEXER_MATMULS = ("wq_b", "weights_proj")


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                scale_dtype: mx.Dtype | None) -> None:
    """int8 affine-quantize a 2-D weight ``[out,in]`` and add it under ``key``."""
    writer.add_quantized(key, *quantize_affine(w, _INT8_BITS, gs, scale_dtype=scale_dtype), _INT8_BITS, gs)


def _swiglu_inter(xe: mx.array, w_gate: mx.array, w_up: mx.array, limit: float) -> mx.array:
    """down-proj (``w2``) calibration input: ``silu(clamp(gate·x)) · clamp(up·x)`` → ``[n, inter]``.

    Matches :func:`quanta.dsv4.moe._swiglu_stack`: clamp gate above ``limit`` and ``up`` to
    ``[-limit, limit]`` when ``limit > 0``."""
    xf = xe.astype(mx.float32)
    g = xf @ w_gate.astype(mx.float32).T
    u = xf @ w_up.astype(mx.float32).T
    if limit > 0:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    return silu(g) * u


def _bake_expert(writer: ArtifactWriter, base: str, w_gate: mx.array, w_up: mx.array,
                 w_down: mx.array, xe: mx.array | None, gs: int, method: str,
                 limit: float, scale_dtype: mx.Dtype | None) -> int:
    """AWQ (or RTN) int4 the three SwiGLU expert matrices under ``base`` (``.w1/.w3/.w2``).

    Returns the routed-row count (0 = cold/RTN). ``w1``/``w3`` calibrate on the expert's routed rows
    ``xe`` ``[n, dim]``; ``w2`` on ``silu(g)·u`` of those rows. Cold experts and ``method='rtn'`` use
    plain int4 with ``s=1`` (identity scale) so the runtime always divides by a stored ``s`` (one path).
    """
    if method == "awq" and xe is not None and xe.shape[0] > 0:
        for proj, w in (("w1", w_gate), ("w3", w_up)):
            s, p, sc, b = awq_quantize(w, xe, _EXPERT_BITS, gs)
            writer.add_awq_quantized(f"{base}.{proj}", p, sc, b, s.astype(mx.bfloat16), _EXPERT_BITS, gs)
        inter = _swiglu_inter(xe, w_gate, w_up, limit)  # [n, inter] down-proj input
        s, p, sc, b = awq_quantize(w_down, inter, _EXPERT_BITS, gs)
        writer.add_awq_quantized(f"{base}.w2", p, sc, b, s.astype(mx.bfloat16), _EXPERT_BITS, gs)
        return int(xe.shape[0])
    for proj, w in (("w1", w_gate), ("w3", w_up), ("w2", w_down)):  # RTN / cold → s=1 (plain int4)
        p, sc, b = quantize_affine(w, _EXPERT_BITS, gs, scale_dtype=scale_dtype)
        ones = mx.ones((w.shape[1],), dtype=mx.bfloat16)
        writer.add_awq_quantized(f"{base}.{proj}", p, sc, b, ones, _EXPERT_BITS, gs)
    return 0


def _write_sub(writer: ArtifactWriter, prefix: str, sub: dict, matmuls: tuple[str, ...],
               control: tuple[str, ...], gs: int, scale_dtype: mx.Dtype | None) -> None:
    """Write one loader sub-dict: ``matmuls`` → int8, ``control`` → dense. Fail loud on any key that
    is neither (rule-6: refuse to bake a tensor with no policy). Nested dicts (compressor/indexer) are
    dispatched by the caller, so they must not appear in ``sub``."""
    for name, arr in sub.items():
        if isinstance(arr, dict):
            raise ValueError(f"{prefix}{name}: nested dict must be dispatched by the caller, not _write_sub")
        if name in matmuls:
            _write_int8(writer, f"{prefix}{name}", arr, gs, scale_dtype)
        elif name in control:
            # control tensor stored verbatim (native dtype) — never silently downcast (rule-6)
            writer.add_dense(_dense_name(prefix, name), arr)
        else:
            raise ValueError(f"{prefix}{name}: no quant policy (not in matmuls or control)")


def _dense_name(prefix: str, name: str) -> str:
    """Source tensor name for a dense control tensor: weight tensors get ``.weight``, bare params
    (``attn_sink``, ``ape``) do not — matching the source checkpoint layout the runtime reads."""
    bare = {"attn_sink", "ape"}
    return f"{prefix}{name}" if name in bare else f"{prefix}{name}.weight"


def _bake_attention(writer: ArtifactWriter, ap: str, attn: dict, cfg: DeepSeekV4Config,
                    layer_id: int, gs: int, scale_dtype: mx.Dtype | None) -> None:
    """Attention sub-block: q/kv/o matmuls int8, norms+sink dense, + compressor/indexer when present."""
    flat = {k: v for k, v in attn.items() if not isinstance(v, dict)}
    _write_sub(writer, ap, flat, _ATTN_MATMULS, _ATTN_CONTROL, gs, scale_dtype)
    if "compressor" in attn:
        _write_sub(writer, f"{ap}compressor.", attn["compressor"],
                   _COMPRESSOR_MATMULS, _COMPRESSOR_CONTROL, gs, scale_dtype)
    if "indexer" in attn:
        idx = attn["indexer"]
        _write_int8(writer, f"{ap}indexer.wq_b", idx["wq_b"], gs, scale_dtype)
        _write_int8(writer, f"{ap}indexer.weights_proj", idx["weights_proj"], gs, scale_dtype)
        _write_sub(writer, f"{ap}indexer.compressor.", idx["compressor"],
                   _COMPRESSOR_MATMULS, _COMPRESSOR_CONTROL, gs, scale_dtype)


def _bake_mtp(writer: ArtifactWriter, ck: DeepSeekV4SourceCheckpoint, cfg: DeepSeekV4Config,
              j: int, gs: int, scale_dtype: mx.Dtype | None) -> None:
    """MTP block: ``e_proj``/``h_proj`` int8; norms + ``hc_head_*`` dense."""
    t = ck.mtp(j)
    p = f"mtp.{j}."
    for name in ("e_proj", "h_proj"):
        _write_int8(writer, f"{p}{name}", t[name], gs, scale_dtype)
    for name in ("enorm", "hnorm", "norm"):
        writer.add_dense(f"{p}{name}.weight", t[name])
    for k in ("fn", "base", "scale"):
        writer.add_dense(f"{p}hc_head_{k}", t[f"hc_head_{k}"])
    del t


def bake_dsv4(
    source: str | Path,
    out_dir: str | Path,
    calib_ids: mx.array,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    group_size: int = 128,
    expert_method: str = "awq",
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the DSV4-Flash source into a self-contained int4(AWQ)/int8/bf16 artifact at ``out_dir``.

    Returns a summary ``dict`` (bytes written, per-kind counts, warm experts). ``n_layers`` /
    ``expert_subset`` slice the bake for bounded validation; ``include_head`` toggles embed/norm/head.
    """
    assert expert_method in ("awq", "rtn"), f"expert_method must be 'awq'|'rtn', got {expert_method!r}"
    cfg = DeepSeekV4Config.from_pretrained(source)
    ck = DeepSeekV4SourceCheckpoint(source, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts = list(range(cfg.n_routed_experts)) if expert_subset is None else list(expert_subset)
    limit = cfg.swiglu_limit

    caps = capture_calibration(ck, cfg, calib_ids, n_layers=n) if expert_method == "awq" else {}

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")
    counts = {"int8": 0, "dense": 0, "expert_int4": 0}

    if include_head:
        writer.add_dense(EMBED, ck.embed())          # bf16 (logit-sensitive; policy)
        writer.add_dense(NORMF, ck.final_norm())
        fhc = ck.final_hc()
        for k in ("fn", "base", "scale"):
            writer.add_dense(f"hc_head_{k}", fhc[k])
        if not cfg.tie_word_embeddings:
            _write_int8(writer, "head", ck.head(), group_size, scale_dtype)
        ck.release()

    warm = 0
    for i in range(n):
        ap = f"layers.{i}.attn."
        _bake_attention(writer, ap, ck.attention(i), cfg, i, group_size, scale_dtype)

        norms = ck.block_norms(i)
        writer.add_dense(f"layers.{i}.attn_norm.weight", norms["attn_norm"])
        writer.add_dense(f"layers.{i}.ffn_norm.weight", norms["ffn_norm"])
        for k, v in ck.block_hc(i).items():                    # hc_attn_*/hc_ffn_* (f32, dense)
            writer.add_dense(f"layers.{i}.{k}", v)

        router = ck.moe_router(i)                              # gate matmul int8; bias/tid2eid dense
        _write_int8(writer, f"layers.{i}.ffn.gate", router["weight"], group_size, scale_dtype)
        if "tid2eid" in router:
            writer.add_dense(f"layers.{i}.ffn.gate.tid2eid", router["tid2eid"])
        elif "bias" in router:
            writer.add_dense(f"layers.{i}.ffn.gate.bias", router["bias"])

        shared = ck.shared_expert(i)                           # always-on, bf16 (never quantized)
        for proj in ("w1", "w2", "w3"):
            writer.add_dense(f"layers.{i}.ffn.shared_experts.{proj}.weight", shared[proj])

        es = ck.expert_stacks(i, cfg.n_routed_experts)         # {w1,w3:[E,inter,dim], w2:[E,dim,inter]}
        x_cap, idx_cap = caps.get(i, (None, None))
        for e in experts:
            base = f"layers.{i}.ffn.experts.{e}"
            xe = expert_rows(x_cap, idx_cap, e) if (expert_method == "awq" and x_cap is not None) else None
            warm += int(_bake_expert(writer, base, es["w1"][e], es["w3"][e], es["w2"][e],
                                     xe, group_size, expert_method, limit, scale_dtype) > 0)
        del es, shared, router, norms
        ck.release()
        mx.clear_cache()

    if include_head and cfg.n_mtp_layers > 0:  # native MTP head (baked like a decoder layer)
        for j in range(cfg.n_mtp_layers):
            _bake_mtp(writer, ck, cfg, j, group_size, scale_dtype)
            ck.release()
            mx.clear_cache()

    for entry in writer.manifest.values():  # tally written kinds for the summary
        fmt = entry["format"]
        if fmt == "affine_packed":
            counts["int8"] += 1
        elif fmt == "awq_packed":
            counts["expert_int4"] += 1
        else:
            counts["dense"] += 1

    scale_tag = "bf16" if scale_dtype == mx.bfloat16 else "fp32"
    policy = {"experts": f"int4 {expert_method} g{group_size}", "non_experts": f"int8 g{group_size}",
              "shared_norms_hc_control": "bf16/f32", "scales": scale_tag}
    writer.finalize(policy)  # flushes shards + writes index/config/manifest

    out = Path(out_dir)
    total_bytes = sum(p.stat().st_size for p in out.glob("model-*.safetensors"))  # exact, on-disk
    return {"layers": n, "experts_per_layer": len(experts), "warm_experts": warm,
            "expert_method": expert_method, "mtp_layers": cfg.n_mtp_layers if include_head else 0,
            "counts": counts, "bytes": total_bytes}
