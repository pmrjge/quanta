"""Qwen3.5-397B-A17B (``qwen3_5_moe``) bake → a self-contained int4/int8/bf16 artifact (parity-first).

Streamed, one layer resident at a time (rule-8), mirroring the DSV4 (pre-stacked int4 experts +
:class:`~quanta.bake.artifact.ArtifactWriter`) and Nemotron (recurrent-SSM control kept **bf16**)
bakes. Per-tensor scheme follows the project quant policy + the #115 recipe:

* **routed experts** — the **pre-stacked** ``mlp.experts.gate_up_proj`` ``[E, 2*moe_inter, hidden]``
  (fused gate+up) and ``mlp.experts.down_proj`` ``[E, hidden, moe_inter]`` — → **int4 affine,
  group_size 64** (``bake/quant.py``). The bf16 source has the sub-int4-grid headroom (settled), and
  the stacks are quantized **as 3-D tensors in one shot** (``mx.quantize`` groups over the trailing
  ``in`` dim), keeping them in the exact ``[E, out, in]`` layout ``mx.gather_qmm`` decodes — no
  per-expert python loop (rule-3);
* **non-experts** → **int8 affine**: gated-GQA ``q/k/v/o_proj`` (the 15 full layers), Gated-DeltaNet
  ``in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj`` (the 45 linear layers), and the shared
  expert (``shared_{gate,up,down}_proj``);
* **bf16, NOT quantized** (control): Gated-DeltaNet **SSM control** ``A_log`` / ``dt_bias`` /
  ``conv1d`` (depthwise conv) / ``norm`` (per-head gated RMSNorm), **all RMSNorms**
  (input/post-attention + per-head q/k norm + final norm), the router ``gate``, the
  ``shared_expert_gate`` sigmoid scalar, and the ``embed_tokens`` / ``lm_head`` token tables.
  Stored verbatim in their native dtype (most bf16; ``A_log`` / DeltaNet ``norm`` are f32 in the
  checkpoint) — never silently downcast (rule-6).

The native **MTP** block (``mtp.*``) is baked like a decoder layer: its routed experts → int4 g64,
its full-attn projections + shared expert → int8, and its norms + the ``fc`` embed/hidden fusion
(``fc`` / ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden`` / ``norm``) → bf16. NB the MTP block
stores experts **per-expert** in the source, so the loader hands them back as **separate** stacks
(``experts_gate_proj`` / ``experts_up_proj`` / ``experts_down_proj``), each baked int4 g64.

**Dynamic-YaRN 1M policy:** the resident artifact must serve 1M context by default, so the bake
writes the quanta long-context policy (``max_context`` / ``yarn_factor`` / ``yarn_original_max`` /
``yarn_dynamic``, read straight off :class:`~quanta.qwen35.config.Qwen35Config`) into the artifact's
``config.json`` (a ``quanta_long_context`` block + ``max_position_embeddings`` raised to the target)
so the resident runtime defaults to the 1M policy without a code change.

Runnable on a slice (``n_layers``, ``expert_subset``) for bounded validation; the full call is the
real bake.

    # DEFERRED — the real bake is GPU+memory-heavy (~775 GB bf16 source); run in a future GPU slot,
    # NOT here. (then teacher-forced ppl over the resident int4/int8 runtime vs the bf16 reference.)
    # from quanta.qwen35.bake import bake_qwen35
    # bake_qwen35("/Users/pmrj/models/Qwen3.5-397B-A17B",
    #             "/Users/pmrj/models/Qwen3.5-397B-A17B-quanta_int4", calib_ids,
    #             group_size=64, scale_dtype=mx.bfloat16)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.quant import quantize_affine
from quanta.qwen35.calibrate import capture_calibration
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.loader import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    FULL_ATTN_SUFFIXES,
    LINEAR_ATTN_SUFFIXES,
    LM_HEAD_KEY,
    LM_PREFIX,
    SHARED_EXPERT_PROJS,
    Qwen35SourceCheckpoint,
)

_EXPERT_BITS = 4
_INT8_BITS = 8

# --- per-kind suffix policy (mirror the loader's enumeration; classify each, fail loud on a miss) ---
# Linear-attention (Gated DeltaNet): which suffixes are int8 matmuls vs bf16 SSM control.
_LINEAR_INT8 = ("in_proj_qkv.weight", "in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight",
                "out_proj.weight")
_LINEAR_BF16 = ("conv1d.weight", "A_log", "dt_bias", "norm.weight")  # SSM control → bf16
# Full attention (gated GQA): q/k/v/o are int8; the per-head q/k RMSNorm is bf16.
_FULL_INT8 = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
_FULL_BF16 = ("q_norm.weight", "k_norm.weight")

# Fail loud at import if our int8/bf16 partition does not EXACTLY tile the loader's enumeration
# (rule-6: no suffix without a policy, no policy for a suffix the loader never reads) — drift-proofing.
if set(_LINEAR_INT8) | set(_LINEAR_BF16) != set(LINEAR_ATTN_SUFFIXES):
    raise AssertionError(f"linear-attn suffix policy {sorted(set(_LINEAR_INT8) | set(_LINEAR_BF16))} "
                         f"!= loader enumeration {sorted(LINEAR_ATTN_SUFFIXES)}")
if set(_FULL_INT8) | set(_FULL_BF16) != set(FULL_ATTN_SUFFIXES):
    raise AssertionError(f"full-attn suffix policy {sorted(set(_FULL_INT8) | set(_FULL_BF16))} "
                         f"!= loader enumeration {sorted(FULL_ATTN_SUFFIXES)}")


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                scale_dtype: mx.Dtype | None) -> None:
    """int8 affine-quantize a 2-D weight ``[out,in]`` and add it under ``key`` (no ``.weight``)."""
    writer.add_quantized(key, *quantize_affine(w, _INT8_BITS, gs, scale_dtype=scale_dtype),
                         _INT8_BITS, gs)


def _write_expert_stack(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                        scale_dtype: mx.Dtype | None) -> None:
    """int4 affine-quantize a **pre-stacked 3-D** expert tensor ``[E, out, in]`` in one shot.

    ``mx.quantize`` groups over the trailing ``in`` dim, so the 3-D stack stays in the ``[E, out, in]``
    layout the runtime's ``mx.gather_qmm`` consumes — packed codes ``[E, out, in*bits/32]`` + scales /
    biases ``[E, out, in/gs]``. Stored under ``key`` as a standard ``affine_packed`` entry.
    """
    writer.add_quantized(key, *quantize_affine(w, _EXPERT_BITS, gs, scale_dtype=scale_dtype),
                         _EXPERT_BITS, gs)


def _write_suffix_sub(writer: ArtifactWriter, prefix: str, sub: dict, int8: tuple[str, ...],
                      bf16: tuple[str, ...], gs: int, scale_dtype: mx.Dtype | None) -> None:
    """Write a loader suffix-keyed sub-dict: ``int8`` suffixes → int8, ``bf16`` → dense verbatim.

    Fail loud on any suffix that is in neither set (rule-6 — refuse to bake a tensor with no policy).
    Matmul keys drop the trailing ``.weight`` (so the runtime reads ``{base}.weight_packed``); dense
    control keys are stored at their full source key verbatim.
    """
    for suffix, arr in sub.items():
        if suffix in int8:
            _write_int8(writer, prefix + suffix[: -len(".weight")], arr, gs, scale_dtype)
        elif suffix in bf16:
            writer.add_dense(prefix + suffix, arr)  # native dtype; never downcast (rule-6)
        else:
            raise ValueError(f"{prefix}{suffix}: no quant policy (not in int8 or bf16 suffix set)")


def _bake_moe_block(writer: ArtifactWriter, prefix: str, moe: dict, gs: int,
                    scale_dtype: mx.Dtype | None) -> None:
    """Bake one MoE block (main-decoder layout): router gate + shared-gate bf16, shared expert int8,
    fused pre-stacked routed experts int4 g64. ``prefix`` is e.g. ``layers.{i}.mlp.``."""
    writer.add_dense(prefix + "gate.weight", moe["gate"])                       # router → bf16
    writer.add_dense(prefix + "shared_expert_gate.weight", moe["shared_expert_gate"])  # sigmoid → bf16
    for proj in SHARED_EXPERT_PROJS:                                            # shared expert → int8
        _write_int8(writer, f"{prefix}shared_expert.{proj}", moe[f"shared_{proj}"], gs, scale_dtype)
    # fused, pre-stacked routed experts → int4 g64 (3-D, gather_qmm-ready)
    _write_expert_stack(writer, prefix + "experts.gate_up_proj", moe["experts_gate_up"],
                        gs, scale_dtype)
    _write_expert_stack(writer, prefix + "experts.down_proj", moe["experts_down"], gs, scale_dtype)


def _bake_mtp(writer: ArtifactWriter, ck: Qwen35SourceCheckpoint, cfg: Qwen35Config,
              j: int, gs: int, scale_dtype: mx.Dtype | None) -> None:
    """Bake the native MTP block like a decoder layer: fc-fusion + norms bf16; full-attn int8;
    shared expert int8; per-expert routed stacks int4 g64."""
    t = ck.mtp(j)
    p = f"mtp.{j}."
    # fc embed/hidden fusion + its pre-norms + the block's final norm → bf16
    writer.add_dense(p + "fc.weight", t["fc"])
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        writer.add_dense(p + f"{name}.weight", t[name])
    # the inherited full-attn + MoE decoder block (always full-attention)
    writer.add_dense(p + "input_layernorm.weight", t["input_layernorm"])
    writer.add_dense(p + "post_attention_layernorm.weight", t["post_attention_layernorm"])
    _write_suffix_sub(writer, p + "self_attn.", t["attention"], _FULL_INT8, _FULL_BF16, gs, scale_dtype)
    moe = t["moe"]
    mp = p + "mlp."
    writer.add_dense(mp + "gate.weight", moe["gate"])
    writer.add_dense(mp + "shared_expert_gate.weight", moe["shared_expert_gate"])
    for proj in SHARED_EXPERT_PROJS:
        _write_int8(writer, f"{mp}shared_expert.{proj}", moe[f"shared_{proj}"], gs, scale_dtype)
        # per-expert stacks (the MTP source stores experts un-fused) → int4 g64
        _write_expert_stack(writer, f"{mp}experts.{proj}", moe[f"experts_{proj}"], gs, scale_dtype)
    del t, moe


def _bake_long_context(out_dir: Path, cfg: Qwen35Config) -> None:
    """Inject the quanta dynamic-YaRN 1M policy into the **already-written** artifact ``config.json``.

    ``ArtifactWriter.finalize`` copies the source ``config.json`` verbatim (+ ``quantization_config``);
    here we add a self-describing ``quanta_long_context`` block carrying the baked policy fields read
    off :class:`Qwen35Config` (``max_context`` / ``yarn_factor`` / ``yarn_original_max`` /
    ``yarn_dynamic``), at the top level and mirrored into the nested ``text_config``, so the resident
    runtime serves the 1M target by default.

    Crucially it does **NOT** touch ``max_position_embeddings``: :meth:`Qwen35Config.from_pretrained`
    derives ``yarn_original_max`` (the dynamic-YaRN baseline below which no scaling is applied) FROM
    ``max_position_embeddings``, so raising it to ``max_context`` would set ``yarn_original_max`` to
    1M and make ``effective_yarn_factor`` return 1.0 for every sequence — silently disabling dynamic
    YaRN. The native window stays native (262144); the 1M reach comes from ``yarn_factor`` (=4) ×
    that native baseline (≈1.05M ≥ the 1.01M target), exactly as ``effective_yarn_factor`` computes.
    ``config.json`` is the artifact's own config, NOT the source's — this is not a source mutation.
    """
    cfg_path = out_dir / "config.json"
    conf = json.loads(cfg_path.read_text())
    policy = {
        "max_context": cfg.max_context,
        "yarn_factor": cfg.yarn_factor,
        "yarn_original_max": cfg.yarn_original_max,
        "yarn_dynamic": cfg.yarn_dynamic,
    }
    conf["quanta_long_context"] = policy
    tc = conf.get("text_config")
    if isinstance(tc, dict):
        tc["quanta_long_context"] = policy
    cfg_path.write_text(json.dumps(conf, indent=2))


def bake_qwen35(
    source: str | Path,
    out_dir: str | Path,
    calib_ids: mx.array,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    include_mtp: bool = True,
    group_size: int = 64,
    capture_acts: bool = False,
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the Qwen3.5-397B-A17B bf16 source into a self-contained int4/int8/bf16 artifact.

    Returns a summary ``dict`` (per-kind counts, layers, bytes). ``n_layers`` / ``expert_subset``
    slice the bake for bounded validation; ``include_head`` toggles embed/norm/head; ``include_mtp``
    toggles the native MTP block. ``capture_acts`` runs the streamed calibration forward (post-norm
    acts + routing) for the QC gauge / a future GPTQ pass — off by default since the int4 recipe is
    plain affine RTN over the stacks. ``group_size`` is the routed-expert (and non-expert) group
    (64 per #115). The dynamic-YaRN 1M policy is baked into ``config.json`` at finalize.
    """
    cfg = Qwen35Config.from_pretrained(source)
    ck = Qwen35SourceCheckpoint(source, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts_sel = None if expert_subset is None else list(expert_subset)

    caps = capture_calibration(ck, cfg, calib_ids, n_layers=n) if capture_acts else {}

    writer = ArtifactWriter(out_dir, Path(source) / "config.json")

    if include_head:
        writer.add_dense(EMBED_KEY, ck.embed())                 # token table → bf16 (logit-sensitive)
        writer.add_dense(FINAL_NORM_KEY, ck.final_norm())       # final RMSNorm → bf16
        if not cfg.tie_word_embeddings:
            writer.add_dense(LM_HEAD_KEY, ck.lm_head())         # output head → bf16
        ck.release()

    for i in range(n):
        lp = f"{LM_PREFIX}layers.{i}."
        norms = ck.block_norms(i)
        writer.add_dense(lp + "input_layernorm.weight", norms["input_layernorm"])
        writer.add_dense(lp + "post_attention_layernorm.weight", norms["post_attention_layernorm"])

        if cfg.is_linear_attention(i):
            _write_suffix_sub(writer, lp + "linear_attn.", ck.linear_attn(i),
                              _LINEAR_INT8, _LINEAR_BF16, group_size, scale_dtype)
        else:
            _write_suffix_sub(writer, lp + "self_attn.", ck.full_attn(i),
                              _FULL_INT8, _FULL_BF16, group_size, scale_dtype)

        moe = ck.moe(i)
        if experts_sel is not None:  # bounded validation: subset the pre-stacked experts
            moe = dict(moe)
            moe["experts_gate_up"] = moe["experts_gate_up"][experts_sel]
            moe["experts_down"] = moe["experts_down"][experts_sel]
        _bake_moe_block(writer, lp + "mlp.", moe, group_size, scale_dtype)

        del norms, moe
        ck.release()
        mx.clear_cache()

    if include_head and include_mtp and cfg.num_mtp_modules > 0:
        for j in range(cfg.num_mtp_modules):
            _bake_mtp(writer, ck, cfg, j, group_size, scale_dtype)
            ck.release()
            mx.clear_cache()

    counts = {"int8": 0, "expert_int4": 0, "dense": 0}
    for entry in writer.manifest.values():
        fmt, bits = entry["format"], entry.get("bits")
        if fmt == "affine_packed":
            counts["expert_int4" if bits == _EXPERT_BITS else "int8"] += 1
        else:
            counts["dense"] += 1

    scale_tag = "bf16" if scale_dtype == mx.bfloat16 else "fp32"
    policy = {
        "experts": f"int4 affine g{group_size}",
        "non_experts": f"int8 affine g{group_size}",
        "ssm_control_norms_router_shared_gate_head": "bf16/f32",
        "scales": scale_tag,
        "long_context": {"max_context": cfg.max_context, "yarn_factor": cfg.yarn_factor,
                         "yarn_original_max": cfg.yarn_original_max, "yarn_dynamic": cfg.yarn_dynamic},
    }
    writer.finalize(policy)  # flushes shards + writes index/config/manifest
    _bake_long_context(Path(out_dir), cfg)  # bake the dynamic-YaRN 1M policy into config.json

    out = Path(out_dir)
    total_bytes = sum(p.stat().st_size for p in out.glob("model-*.safetensors"))
    return {
        "layers": n,
        "experts_per_layer": (cfg.num_experts if experts_sel is None else len(experts_sel)),
        "mtp_layers": (cfg.num_mtp_modules if (include_head and include_mtp) else 0),
        "captured_layers": len(caps),
        "counts": counts,
        "bytes": total_bytes,
    }
