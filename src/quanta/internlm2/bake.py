"""InternLM2.5-7B-Chat-1M (``internlm2``) bake → a self-contained int8/int4/bf16 artifact.

Streamed, one layer resident at a time (rule-8). InternLM2.5 is **dense** (no MoE), so the bake is
as simple as the Qwen2.5 case: no GPTQ expert allocation, no calibration capture, no MTP, no SSM.
Plain affine RTN over the matmul weights, with the per-tensor policy below.

The one InternLM2-specific twist is the **wqkv split**: the source ships a single fused weight
``model.layers.{i}.attention.wqkv.weight``, but the loader deinterleaves it at ``attention(i)``
into three standard ``wq``/``wk``/``wv`` projections — by the time the bake sees them they're
already split, so the bake quantizes three separate weights and writes them under three separate
keys (``attention.wq.weight_packed``, ``attention.wk.weight_packed``, ``attention.wv.weight_packed``).
The artifact never holds the fused ``wqkv`` again — the runtime sees three plain GQA projections.

**Per-tensor quant policy (applied to every layer 0..31):**

* **Attention matmul weights** (``wq`` / ``wk`` / ``wv`` / ``wo``) → **int8 affine, group_size 64**.
  The 1M-context KV cache dominates resident memory, not the attention weights — keep the
  matmul precision at int8 (~0.8% recon, ~lossless per #44).
* **FFN matmul weights** (``w1`` / ``w3`` / ``w2``) → **int4 affine, group_size 64**.
  FFN dominates the weight byte count (3 projections × hidden×inter = ~176M params/layer vs ~50M
  for attention); int4 g64 RTN from a bf16 source is settled (Nemotron / Qwen3.5 / GLM / Qwen2.5
  all ship it).
* **No biases** (``cfg.attention_bias=False`` for InternLM2.5 — the source's single ``bias`` field
  governs every projection and is ``False``). Nothing to bake for the bias kinds.
* **RMSNorms** (``attention_norm`` / ``ffn_norm`` / ``model.norm``) → **bf16, dense**.
* **Embeddings** (``model.tok_embeddings``, ``output``) → **bf16, dense** — logit-sensitive
  (mirrors Qwen2.5 / Qwen3.5 / Nemotron / DSV4). Untied for InternLM2.5
  (``tie_word_embeddings=False``), so both are baked under their respective source keys.

The dynamic-NTK long-context policy travels with the artifact's ``config.json`` (copied verbatim
from the source by :class:`~quanta.bake.artifact.ArtifactWriter`; the ``rope_scaling`` block is
part of it). No extra ``quanta_long_context`` injection — like DCA for Qwen2.5-1M, NTK is a
*source-native* policy already encoded in the config; the runtime reads it straight off
:class:`~quanta.internlm2.config.InternLM2Config`.

Estimated resident weight footprint (full 32-layer bake):

* attention int8 g64:  ~57 MB/layer × 32 ≈ 1.8 GB
* FFN int4 g64:        ~89 MB/layer × 32 ≈ 2.9 GB
* embed + output bf16: 92544 × 4096 × 2 × 2 ≈ 1.5 GB
* norms bf16:          ~negligible
* **Total ≈ 6.2 GB resident** — vs ~14 GB bf16 source.

Runnable on a slice (``n_layers``) for bounded validation; the full call is the real bake.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.quant import quantize_affine
from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.loader import (
    ATTN_WEIGHT_SUFFIXES,
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    MLP_SUFFIXES,
    MODEL_PREFIX,
    InternLM2SourceCheckpoint,
)

_ATTN_BITS = 8   # int8 affine for wq/wk/wv/wo (numerical headroom for 1M attention)
_MLP_BITS = 4    # int4 affine for SwiGLU FFN (byte-budget dominator; bf16 source tolerates int4 g64)

# Sidecar files copied verbatim from the source checkpoint so the artifact is self-contained
# (rule: no external references). Tokenizer files are required for serve; generation_config /
# chat_template are optional but ship if present in the source.
_SIDECAR_FILES = (
    "tokenizer.model",                # required: SentencePiece protobuf
    "tokenizer_config.json",          # required: added tokens + chat template + token policy
    "special_tokens_map.json",        # recommended: canonical bos/eos/unk/pad strings
    "generation_config.json",         # recommended: sampling defaults + full eos set
    "tokenization_internlm2.py",      # optional: HF tokenizer fallback (we use SP directly)
    "tokenization_internlm2_fast.py", # optional: HF fast-tokenizer fallback
    "configuration_internlm2.py",     # optional: HF config class
    "modeling_internlm2.py",          # optional: HF reference forward
)


def _copy_sidecars(source: Path, dest: Path) -> list[str]:
    """Copy every present sidecar from ``source`` to ``dest``; return the list copied."""
    copied: list[str] = []
    for name in _SIDECAR_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def _write_quant(writer: ArtifactWriter, key: str, w: mx.array, bits: int, gs: int,
                 scale_dtype: mx.Dtype | None) -> None:
    """Affine-quantize a 2-D weight ``[out, in]`` at ``bits`` / ``group_size`` and add it.

    ``key`` is the base (without ``.weight``); the artifact writer expands it to
    ``{base}.weight_packed`` / ``.weight_scale`` / ``.weight_bias``.
    """
    writer.add_quantized(key, *quantize_affine(w, bits, gs, scale_dtype=scale_dtype), bits, gs)


def _bake_attention(writer: ArtifactWriter, prefix: str, attn: dict[str, mx.array], gs: int,
                    scale_dtype: mx.Dtype | None, *, bits: int = _ATTN_BITS) -> None:
    """Bake one self-attention block: int-``bits`` wq/wk/wv/wo weights (already split by the loader).

    ``prefix`` is ``layers.{i}.attention.`` (already includes the model-namespace prefix). No
    biases are baked — InternLM2.5 has ``bias=False`` for every projection.
    """
    for suffix in ATTN_WEIGHT_SUFFIXES:
        _write_quant(writer, prefix + suffix[: -len(".weight")], attn[suffix], bits, gs,
                     scale_dtype)


def _bake_mlp(writer: ArtifactWriter, prefix: str, mlp: dict[str, mx.array], gs: int,
              scale_dtype: mx.Dtype | None, *, bits: int = _MLP_BITS) -> None:
    """Bake one SwiGLU FFN block: int-``bits`` w1 / w3 / w2. ``prefix`` is ``layers.{i}.feed_forward.``."""
    for suffix in MLP_SUFFIXES:
        _write_quant(writer, prefix + suffix[: -len(".weight")], mlp[suffix], bits, gs,
                     scale_dtype)


def bake_internlm2(
    source: str | Path,
    out_dir: str | Path,
    *,
    n_layers: int | None = None,
    include_head: bool = True,
    group_size: int = 64,
    attn_bits: int = _ATTN_BITS,
    mlp_bits: int = _MLP_BITS,
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the InternLM2.5-7B-Chat-1M bf16 source into a self-contained int/bf16 artifact.

    Args:
      source: directory of the source HF checkpoint (``~/models/internlm2_5-7b-chat-1m``).
      out_dir: destination for the baked artifact (e.g.
               ``~/models/internlm2_5-7b-chat-1m-quanta_int8g64``).
      n_layers: bake only the first ``n_layers`` (default = all 32). For bounded validation.
      include_head: bake ``tok_embeddings`` / ``model.norm`` / ``output``. Default True.
      group_size: affine group size for the int quant (default 64 per #45/#115).
      attn_bits: bit width for the attention wq/wk/wv/wo weights (default 8). The runtime reads the
                 actual width back from the manifest, so this is the single source of truth.
      mlp_bits: bit width for the SwiGLU FFN w1/w3/w2 weights (default 4; pass 8 for an int8 artifact
                — a 7B fits comfortably resident at int8, ~lossless).
      scale_dtype: if ``mx.bfloat16``, store scales/biases bf16 instead of fp32 (halves overhead,
                   ~lossless per #44). Default None = fp32 scales.

    Returns:
      Summary ``dict`` with per-kind tensor counts, layer count, and artifact byte size.
    """
    src_dir = Path(source)
    cfg = InternLM2Config.from_pretrained(src_dir)
    ck = InternLM2SourceCheckpoint(src_dir, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers

    writer = ArtifactWriter(out_dir, src_dir / "config.json")
    # Self-contained artifact: tokenizer / generation_config / chat_template / reference modules
    # ride along so serve can load the bundle with no reference back to the source checkpoint.
    sidecars = _copy_sidecars(src_dir, writer.dir)

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
        writer.add_dense(lp + "attention_norm.weight", norms["attention_norm"])
        writer.add_dense(lp + "ffn_norm.weight", norms["ffn_norm"])

        _bake_attention(writer, lp + "attention.", ck.attention(i), group_size, scale_dtype,
                        bits=attn_bits)
        _bake_mlp(writer, lp + "feed_forward.", ck.mlp(i), group_size, scale_dtype, bits=mlp_bits)

        del norms
        ck.release()
        mx.clear_cache()        # drop MLX buffer-cache high-water between layers (rule-8)

    # Classify by key (robust whether or not attn_bits == mlp_bits — bits alone can't tell them apart
    # for an int8-everywhere bake).
    counts = {"attn_quant": 0, "mlp_quant": 0, "dense": 0}
    for key, entry in writer.manifest.items():
        if entry["format"] == "affine_packed":
            counts["mlp_quant" if "feed_forward" in key else "attn_quant"] += 1
        else:
            counts["dense"] += 1

    scale_tag = "bf16" if scale_dtype is not None and scale_dtype == mx.bfloat16 else "fp32"
    policy = {
        "attention_weights": f"int{attn_bits} affine g{group_size}",
        "mlp_weights": f"int{mlp_bits} affine g{group_size}",
        "biases_norms_embed_head": "bf16",
        "scales": scale_tag,
        "long_context": {
            "method": "dynamic_ntk",
            "rope_theta": cfg.rope_theta,
            "scaling_type": cfg.rope_scaling_type,
            "scaling_factor": cfg.rope_scaling_factor,
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
        "sidecars": sidecars,
    }
