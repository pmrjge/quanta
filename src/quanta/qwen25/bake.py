"""Qwen2.5-14B-Instruct-1M (``qwen2``) bake → a self-contained int8/int4/bf16 artifact (parity-first).

Streamed, one layer resident at a time (rule-8). Qwen2.5 is **dense** (no MoE), so the bake is the
simplest of the quanta targets: no GPTQ expert allocation, no calibration capture, no MTP, no SSM.
Plain affine RTN over the matmul weights, with the per-tensor policy below.

**Per-tensor quant policy (applied to every layer 0..47):**

* **Attention matmul weights** (``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``) → **int8 affine,
  group_size 64**. The attention path is the long-context numerical bottleneck (the 1M-context KV
  cache is what dominates resident memory, not weights — see project memory ``project_flash_sdpa_longctx``),
  so the matmul precision stays at int8 (~0.8% recon, ~lossless per #44).
* **FFN matmul weights** (``gate_proj`` / ``up_proj`` / ``down_proj``) → **int4 affine, group_size 64**.
  FFN dominates the weight byte count (3 projections × hidden×inter = ~212M params/layer vs ~73M for
  attention); int4 g64 RTN from a bf16 source is settled (Nemotron / Qwen3.5 / GLM ship it).
* **Attention biases** (``q_proj.bias`` / ``k_proj.bias`` / ``v_proj.bias`` — Qwen2 specific,
  ``o_proj`` has no bias) → **bf16, dense**. Tiny (= ``q_dim`` or ``kv_dim``), and quantizing a
  1-D bias has no benefit.
* **RMSNorms** (``input_layernorm`` / ``post_attention_layernorm`` / ``model.norm``) → **bf16, dense**.
* **Embeddings** (``embed_tokens``, ``lm_head``) → **bf16, dense** — logit-sensitive (mirrors
  Qwen3.5 / Nemotron / DSV4). Untied for Qwen2.5 (``tie_word_embeddings=false``), so both are baked.

The dual-chunk-attention long-context policy travels with the artifact's ``config.json`` (copied
verbatim from the source by :class:`~quanta.bake.artifact.ArtifactWriter`; the
``dual_chunk_attention_config`` block is part of it). No extra ``quanta_long_context`` injection —
unlike Qwen3.5, DCA is a *source-native* policy already encoded in the config; the runtime reads it
straight off :class:`~quanta.qwen25.config.Qwen25Config`.

Estimated resident weight footprint (full 48-layer bake):

* attention int8 g64:  ~73 MB/layer × 48  ≈ 3.5 GB
* FFN int4 g64:        ~113 MB/layer × 48 ≈ 5.4 GB
* embed + lm_head bf16: 152064 × 5120 × 2 × 2 ≈ 3.0 GB
* norms / biases bf16:  ~negligible
* **Total ≈ 12 GB resident** — vs 28 GB bf16 source.

Runnable on a slice (``n_layers``) for bounded validation; the full call is the real bake.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.quant import quantize_affine
from quanta.qwen25.config import Qwen25Config
from quanta.qwen25.loader import (
    ATTN_BIAS_SUFFIXES,
    ATTN_WEIGHT_SUFFIXES,
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    MLP_SUFFIXES,
    MODEL_PREFIX,
    Qwen25SourceCheckpoint,
)

_ATTN_BITS = 8   # int8 affine for q/k/v/o (numerical headroom for 1M attention)
_MLP_BITS = 4    # int4 affine for SwiGLU FFN (byte-budget dominator; bf16 source tolerates int4 g64)


def _write_quant(writer: ArtifactWriter, key: str, w: mx.array, bits: int, gs: int,
                 scale_dtype: mx.Dtype | None) -> None:
    """Affine-quantize a 2-D weight ``[out, in]`` at ``bits`` / ``group_size`` and add it.

    ``key`` is the base (without ``.weight``); the artifact writer expands it to
    ``{base}.weight_packed`` / ``.weight_scale`` / ``.weight_bias``.
    """
    writer.add_quantized(key, *quantize_affine(w, bits, gs, scale_dtype=scale_dtype), bits, gs)


def _bake_attention(writer: ArtifactWriter, prefix: str, attn: dict[str, mx.array], gs: int,
                    scale_dtype: mx.Dtype | None) -> None:
    """Bake one self-attn block: int8 q/k/v/o weights, bf16 q/k/v biases.

    ``prefix`` is ``layers.{i}.self_attn.`` (already includes the model-namespace prefix).
    """
    for suffix in ATTN_WEIGHT_SUFFIXES:
        _write_quant(writer, prefix + suffix[: -len(".weight")], attn[suffix], _ATTN_BITS, gs,
                     scale_dtype)
    for suffix in ATTN_BIAS_SUFFIXES:
        if suffix in attn:                                     # absent if attention_bias=False
            writer.add_dense(prefix + suffix, attn[suffix])    # native dtype; never downcast (rule-6)


def _bake_mlp(writer: ArtifactWriter, prefix: str, mlp: dict[str, mx.array], gs: int,
              scale_dtype: mx.Dtype | None) -> None:
    """Bake one SwiGLU FFN block: int4 gate_proj / up_proj / down_proj. ``prefix`` is ``layers.{i}.mlp.``."""
    for suffix in MLP_SUFFIXES:
        _write_quant(writer, prefix + suffix[: -len(".weight")], mlp[suffix], _MLP_BITS, gs,
                     scale_dtype)


def bake_qwen25(
    source: str | Path,
    out_dir: str | Path,
    *,
    n_layers: int | None = None,
    include_head: bool = True,
    group_size: int = 64,
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the Qwen2.5-14B-Instruct-1M bf16 source into a self-contained int8/int4/bf16 artifact.

    Args:
      source: directory of the source HF checkpoint (``~/models/Qwen2.5-14B-Instruct-1M``).
      out_dir: destination for the baked artifact (typically
               ``~/models/Qwen2.5-14B-Instruct-1M-quanta_int4g64``).
      n_layers: bake only the first ``n_layers`` (default = all 48). For bounded validation.
      include_head: bake ``embed_tokens`` / ``model.norm`` / ``lm_head``. Default True.
      group_size: affine group size for both int8 and int4 quant (default 64 per #45/#115).
      scale_dtype: if ``mx.bfloat16``, store scales/biases bf16 instead of fp32 (halves overhead,
                   ~lossless per #44). Default None = fp32 scales.

    Returns:
      Summary ``dict`` with per-kind tensor counts, layer count, and artifact byte size.
    """
    cfg = Qwen25Config.from_pretrained(source)
    ck = Qwen25SourceCheckpoint(source, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")

    if include_head:
        writer.add_dense(EMBED_KEY, ck.embed())                 # token table → bf16 (logit-sensitive)
        writer.add_dense(FINAL_NORM_KEY, ck.final_norm())       # final RMSNorm → bf16
        if not cfg.tie_word_embeddings:
            writer.add_dense(LM_HEAD_KEY, ck.lm_head())         # output head → bf16
        ck.release()
        mx.clear_cache()

    for i in range(n):
        lp = f"{MODEL_PREFIX}layers.{i}."

        norms = ck.block_norms(i)
        writer.add_dense(lp + "input_layernorm.weight", norms["input_layernorm"])
        writer.add_dense(lp + "post_attention_layernorm.weight", norms["post_attention_layernorm"])

        _bake_attention(writer, lp + "self_attn.", ck.attention(i), group_size, scale_dtype)
        _bake_mlp(writer, lp + "mlp.", ck.mlp(i), group_size, scale_dtype)

        del norms
        ck.release()
        mx.clear_cache()        # drop MLX buffer-cache high-water between layers (rule-8)

    counts = {"int8_attn": 0, "int4_mlp": 0, "dense": 0}
    for entry in writer.manifest.values():
        fmt, bits = entry["format"], entry.get("bits")
        if fmt == "affine_packed":
            counts["int4_mlp" if bits == _MLP_BITS else "int8_attn"] += 1
        else:
            counts["dense"] += 1

    scale_tag = "bf16" if scale_dtype == mx.bfloat16 else "fp32"
    policy = {
        "attention_weights": f"int{_ATTN_BITS} affine g{group_size}",
        "mlp_weights": f"int{_MLP_BITS} affine g{group_size}",
        "biases_norms_embed_head": "bf16",
        "scales": scale_tag,
        "long_context": {
            "method": "dual_chunk_attention",
            "chunk_size": cfg.dca_chunk_size,
            "local_size": cfg.dca_local_size,
            "original_max_position_embeddings": cfg.dca_original_max,
            "max_position_embeddings": cfg.max_position_embeddings,
        },
    }
    writer.finalize(policy)

    out = Path(out_dir)
    total_bytes = sum(p.stat().st_size for p in out.glob("model-*.safetensors"))
    return {
        "layers": n,
        "counts": counts,
        "bytes": total_bytes,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
    }
