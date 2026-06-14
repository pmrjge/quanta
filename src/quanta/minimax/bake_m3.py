"""MiniMax-M3-VL bake → a self-contained int6/int8/bf16 artifact (parity-first, full VL).

Streamed, one text layer resident at a time (rule 8), mirroring the Qwen3.5/Nex bake
(:mod:`quanta.qwen35.bake`) — M3's text backbone is structurally the qwen35 sibling (Gemma
``(1+w)`` norms, partial RoPE, per-head QK-norm, GQA, sigmoid-noaux MoE) — with three M3 deltas:
**per-expert source experts pre-stacked at load time**, a **trained block-sparse indexer kept bf16**,
and **no YaRN** (M3 is natively 1M, so the artifact inherits the source 1M window verbatim — no
dynamic-YaRN policy to bake). Per-tensor scheme follows :mod:`quanta.minimax.quant_policy_m3`:

* **routed experts** — the **pre-stacked** ``block_sparse_moe.experts.gate_up_proj``
  ``[E, 2*moe_inter, hidden]`` (fused w1=gate over w3=up) and ``experts.down_proj``
  ``[E, hidden, moe_inter]`` (w2) → **int6 affine g64** (``expert_bits=6`` for margin; the user's
  decision — skip int4). Quantized **as 3-D stacks in one shot** (``mx.quantize`` groups over the
  trailing ``in`` dim), keeping the exact ``[E, out, in]`` layout ``mx.gather_qmm`` decodes — no
  per-expert python loop (rule 3). The per-expert→stacked pack is :meth:`loader_m3.moe`'s job.
* **non-experts** → **int8 affine**: GQA ``q/k/v/o_proj`` (every layer), the dense-FFN
  ``mlp.{gate,up,down}_proj`` (layers 0–2), and the shared expert
  ``block_sparse_moe.shared_experts.{gate,up,down}_proj``.
* **bf16/f32, NOT quantized** (control): every RMSNorm (input/post-attention + per-head q/k norm +
  final norm), the **router** ``gate`` + ``e_score_correction_bias`` (kept **f32** — routing
  precision), the **trained sparse-attention indexer** ``index_{q,k}_proj`` + ``index_{q,k}_norm``
  (kept bf16 to protect block selection), and the ``embed_tokens`` / ``lm_head`` token tables. Stored
  verbatim in the native source dtype — never silently downcast (rule 6).

**Full VL (the user's standing decision):** the whole vision tower (``vision_tower.*``), the
``multi_modal_projector.*`` and ``patch_merge_mlp.*`` — 523 tensors, ~1.6 GiB — are copied **dense
bf16 verbatim** so the baked bundle is a COMPLETE self-contained VL model (the ViT *forward* lands in
the vision track; the *weights* live in the artifact from M2 on). The vision pass streams shard by
shard (the vision tensors span only 2 of the 59 shards) so it never holds more than one shard.

The text decoder is read via :class:`quanta.minimax.loader_m3.MiniMaxM3SourceCheckpoint` (which
refuses vision keys — the ViT is a separate read path); the vision pass uses its own minimal
shard-grouped reader (:func:`_bake_vision`) so the text loader's text-only contract stays clean.

Runnable on a slice (``n_layers``, ``expert_subset``, ``include_vision``) for bounded validation; the
full call is the real bake. **Data-free** (plain affine RTN over the stacks — bf16 source has the
sub-int6-grid headroom; settled finding). **Run SOLO** (one model resident; OOM/reboot hazard).

    # the real full bake is GPU+memory-heavy (~809.5 GiB bf16 source, ~330 GiB int6 out, multi-hour);
    # run it SOLO via parity/run_bake_minimax_m3_int6g64, then the M2b teacher-forced ppl arbiter.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx

from quanta.bake.artifact import ArtifactWriter
from quanta.bake.quant import quantize_affine
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.loader_m3 import (
    ATTN_SUFFIXES,
    DENSE_MLP_PROJS,
    SHARED_EXPERT_PROJS,
    SPARSE_INDEX_SUFFIXES,
    MiniMaxM3SourceCheckpoint,
)
from quanta.minimax.quant_policy_m3 import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    LM_PREFIX,
    VISION_PREFIXES,
)

_EXPERT_BITS = 6   # the user's decision: routed experts int6 g64 for margin (skip int4)
_INT8_BITS = 8

# Attention suffix partition (mirror loader_m3.ATTN_SUFFIXES; classify each, fail loud on a miss):
# q/k/v/o are int8 matmuls; the per-head q/k RMSNorm is bf16 (the Gemma (1+w) fold is applied at
# LOAD time by the forward — the bake stores RAW norm weights verbatim).
_ATTN_INT8 = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
_ATTN_BF16 = ("q_norm.weight", "k_norm.weight")
if set(_ATTN_INT8) | set(_ATTN_BF16) != set(ATTN_SUFFIXES):
    raise AssertionError(f"attn suffix policy {sorted(set(_ATTN_INT8) | set(_ATTN_BF16))} != loader "
                         f"enumeration {sorted(ATTN_SUFFIXES)}")

# Source metadata copied verbatim so the bundle is self-contained + servable (text AND vision):
# the tokenizer tables, the chat template, the authoritative generation_config (eos 200020), and —
# for the VL track — the image/video preprocessor config. ``config.json`` / ``manifest.json`` / the
# index are written by the ArtifactWriter; these are the rest. Each copied only if present (rule 6).
_METADATA_SIDECARS: tuple[str, ...] = (
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",   # VL image processor (the vision track reimplements the logic)
)


def _write_int8(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                scale_dtype: mx.Dtype | None) -> None:
    """int8 affine-quantize a 2-D weight ``[out,in]`` and add it under ``key`` (the key WITHOUT a
    trailing ``.weight`` — the runtime reads ``{key}.weight_packed``)."""
    writer.add_quantized(key, *quantize_affine(w, _INT8_BITS, gs, scale_dtype=scale_dtype),
                         _INT8_BITS, gs)


def _write_expert_stack(writer: ArtifactWriter, key: str, w: mx.array, gs: int,
                        scale_dtype: mx.Dtype | None, bits: int = _EXPERT_BITS) -> None:
    """int``bits`` affine-quantize a **pre-stacked 3-D** expert tensor ``[E, out, in]`` in one shot.

    ``mx.quantize`` groups over the trailing ``in`` dim, so the stack stays in the ``[E, out, in]``
    layout the runtime's ``mx.gather_qmm`` consumes — packed codes ``[E, out, in*bits/32]`` + scales /
    biases ``[E, out, in/gs]``. The manifest records ``bits`` so the resident loader decodes at the
    baked width — never a hardcoded default (rule 6). MLX affine supports {2,3,4,6,8}; M3 ships int6.
    """
    writer.add_quantized(key, *quantize_affine(w, bits, gs, scale_dtype=scale_dtype), bits, gs)


def _bake_attention(writer: ArtifactWriter, prefix: str, attn: dict, ck: MiniMaxM3SourceCheckpoint,
                    i: int, cfg: MiniMaxM3Config, gs: int, scale_dtype: mx.Dtype | None) -> None:
    """One layer's attention: q/k/v/o int8, per-head q/k norm bf16, and (on sparse layers 3–59) the
    trained block-sparse indexer ``index_{q,k}_proj/norm`` bf16. Fail loud on any unclassified suffix
    (rule 6). ``prefix`` is e.g. ``language_model.model.layers.{i}.self_attn.``."""
    for suffix, arr in attn.items():
        if suffix in _ATTN_INT8:
            _write_int8(writer, prefix + suffix[: -len(".weight")], arr, gs, scale_dtype)
        elif suffix in _ATTN_BF16:
            writer.add_dense(prefix + suffix, arr)          # raw norm; (1+w) folded at load (rule 6)
        else:
            raise ValueError(f"{prefix}{suffix}: no quant policy (not in int8 or bf16 attn set)")
    if cfg.is_sparse_attention_layer(i):                    # trained indexer → bf16 verbatim
        for suffix, arr in ck.sparse_index(i).items():
            if suffix not in SPARSE_INDEX_SUFFIXES:
                raise ValueError(f"{prefix}{suffix}: unexpected sparse-index tensor")
            writer.add_dense(prefix + suffix, arr)


def _bake_moe_block(writer: ArtifactWriter, prefix: str, moe: dict, gs: int,
                    scale_dtype: mx.Dtype | None, expert_bits: int = _EXPERT_BITS) -> None:
    """One MoE block (layers 3–59): router ``gate`` + ``e_score_correction_bias`` f32 dense, shared
    expert int8, fused pre-stacked routed experts int``expert_bits`` g64. ``prefix`` is e.g.
    ``language_model.model.layers.{i}.block_sparse_moe.``."""
    writer.add_dense(prefix + "gate.weight", moe["gate"])                       # router → f32 dense
    writer.add_dense(prefix + "e_score_correction_bias", moe["e_score_correction_bias"])  # f32 dense
    for proj in SHARED_EXPERT_PROJS:                                            # shared expert → int8
        _write_int8(writer, f"{prefix}shared_experts.{proj}", moe[f"shared_{proj}"], gs, scale_dtype)
    # fused, pre-stacked routed experts → int{expert_bits} g64 (3-D, gather_qmm-ready)
    _write_expert_stack(writer, prefix + "experts.gate_up_proj", moe["experts_gate_up"],
                        gs, scale_dtype, expert_bits)
    _write_expert_stack(writer, prefix + "experts.down_proj", moe["experts_down"], gs, scale_dtype,
                        expert_bits)


def _bake_vision(writer: ArtifactWriter, source: Path) -> int:
    """Copy the whole vision tower / projector / patch-merge **dense bf16 verbatim** into the artifact
    (full-VL build). Streams shard by shard (the 523 vision tensors span only 2 of the 59 shards) so
    no more than one source shard is resident — the text loader stays text-only; this is the vision
    read path. Returns the number of vision tensors copied (rule-6 coverage of the VL track)."""
    wm = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in wm.items():
        if key.startswith(VISION_PREFIXES):
            by_shard[shard].append(key)
    n = 0
    for shard in sorted(by_shard):
        blob = mx.load(str(source / shard))
        for key in by_shard[shard]:
            writer.add_dense(key, blob[key])               # native dtype verbatim (rule 6)
            n += 1
        del blob
        mx.clear_cache()
    return n


def _assert_native_1m_context(out_dir: Path, cfg: MiniMaxM3Config) -> None:
    """M3 is natively 1M (no YaRN). The ArtifactWriter copies the source ``config.json`` verbatim, so
    the artifact already declares the 1M window — assert it (rule 6: the artifact MUST be a first-class
    1M model) and stamp a self-describing ``quanta_long_context`` marker (``yarn_dynamic=False`` — M3
    has no rope scaling) onto both the top level and the nested ``text_config``."""
    cfg_path = out_dir / "config.json"
    conf = json.loads(cfg_path.read_text())
    tc = conf.get("text_config", conf)
    mpe = tc.get("max_position_embeddings", conf.get("max_position_embeddings"))
    if int(mpe or 0) != cfg.max_position_embeddings:
        raise AssertionError(f"artifact config max_position_embeddings={mpe} != native "
                             f"{cfg.max_position_embeddings} (M3 is natively 1M; refusing to ship)")
    marker = {"max_context": cfg.max_position_embeddings, "yarn_dynamic": False,
              "native_long_context": True}
    conf["quanta_long_context"] = marker
    if isinstance(conf.get("text_config"), dict):
        conf["text_config"]["quanta_long_context"] = marker
    cfg_path.write_text(json.dumps(conf, indent=2))


def _copy_metadata_sidecars(source: Path, out_dir: Path, cfg: MiniMaxM3Config) -> None:
    """Copy the source tokenizer / generation / VL-preprocessor metadata (:data:`_METADATA_SIDECARS`)
    into the artifact so the baked bundle is self-contained and servable. M3 ships a correct
    ``generation_config.json`` (eos 200020), so it is copied; SYNTHESIZE one only if the source ships
    none (rule 6: a re-opened artifact must never fall back to a wrong eos)."""
    for name in _METADATA_SIDECARS:
        src = source / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)
    gen_path = out_dir / "generation_config.json"
    if not gen_path.exists():
        eos = cfg.eos_token_ids
        synthesized: dict = {
            "eos_token_id": list(eos) if len(eos) != 1 else int(eos[0]),
            "bos_token_id": cfg.bos_token_id,
            "_quanta_note": "source shipped no generation_config.json; eos resolved from config (rule 6)",
        }
        gen_path.write_text(json.dumps(synthesized, indent=2))


# A self-contained, servable VL artifact MUST contain these; a bake that drops any is not standalone
# and fails loud (rule 6). ``manifest``/index/config are written by the ArtifactWriter;
# ``generation_config.json`` (eos 200020) + ``tokenizer_config.json`` + the VL ``preprocessor_config``
# are placed by :func:`_copy_metadata_sidecars`.
_REQUIRED_ARTIFACT_FILES: tuple[str, ...] = (
    "config.json", "manifest.json", "model.safetensors.index.json",
    "generation_config.json", "tokenizer_config.json",
)
# Substrings that betray a NON-self-contained reference in the artifact json metadata.
_LEAK_MARKERS: tuple[str, ...] = ("/Users/", "/home/", ".cache", "huggingface", "/blobs/",
                                  "/snapshots/")


def _audit_self_contained(out_dir: Path, source: Path) -> dict:
    """Fail loud (rule 6) unless the baked artifact is FULLY self-contained inside ``out_dir``: no
    symlinks, the required sidecars + a tokenizer table present, no path/cache leak in the json
    metadata (incl. the source's own absolute path), and a relative ``weight_map`` whose every shard
    exists. Mirrors :func:`quanta.qwen35.bake._audit_self_contained`. Returns a summary dict."""
    out_dir, source = Path(out_dir), Path(source)
    links = [str(p) for p in out_dir.rglob("*") if p.is_symlink()]
    if links:
        raise AssertionError(f"artifact NOT self-contained: {len(links)} symlink(s), e.g. {links[0]}")

    missing = [f for f in _REQUIRED_ARTIFACT_FILES if not (out_dir / f).exists()]
    if missing:
        raise AssertionError(f"artifact missing required sidecar(s) (not servable): {missing}")
    if not ((out_dir / "tokenizer.json").exists() or (out_dir / "vocab.json").exists()):
        raise AssertionError("artifact has no tokenizer table (tokenizer.json or vocab.json)")

    markers = (*_LEAK_MARKERS, str(source.resolve()))
    leaks = {name: hits for name in ("config.json", "manifest.json",
                                     "model.safetensors.index.json", "generation_config.json")
             if (hits := [m for m in markers if m in (out_dir / name).read_text()])}
    if leaks:
        raise AssertionError(f"artifact json leaks external refs (not self-contained): {leaks}")

    wmap = json.loads((out_dir / "model.safetensors.index.json").read_text())["weight_map"]
    nonrel = sorted({v for v in wmap.values() if "/" in v})
    if nonrel:
        raise AssertionError(f"artifact weight_map has non-relative shard refs: {nonrel[:3]}")
    shards = sorted(set(wmap.values()))
    absent = [s for s in shards if not (out_dir / s).exists()]
    if absent:
        raise AssertionError(f"artifact weight_map references absent shard(s): {absent[:3]}")

    return {"symlinks": 0, "sidecars": "present", "leaks": "none",
            "shards": len(shards), "weight_map_entries": len(wmap)}


def bake_minimax_m3(
    source: str | Path,
    out_dir: str | Path,
    *,
    n_layers: int | None = None,
    expert_subset: Iterable[int] | None = None,
    include_head: bool = True,
    include_vision: bool = True,
    group_size: int = 64,
    expert_bits: int = _EXPERT_BITS,
    scale_dtype: mx.Dtype | None = None,
) -> dict:
    """Bake the MiniMax-M3-VL bf16 source into a self-contained int6/int8/bf16 artifact (full VL).

    Returns a summary ``dict`` (per-kind counts, layers, vision tensors, bytes, self-containment
    audit). ``n_layers`` / ``expert_subset`` slice the bake for bounded validation; ``include_head``
    toggles embed/norm/head; ``include_vision`` toggles the vision-tower passthrough (the full bake
    keeps it — the user's full-VL decision). ``expert_bits`` is the routed-expert width (6 — the user's
    int6 margin decision). Data-free RTN. Streamed one text layer resident at a time (rule 8); the
    artifact is asserted to declare the native 1M window and :func:`_audit_self_contained` then fails
    loud unless the folder is fully standalone (rule 6)."""
    cfg = MiniMaxM3Config.from_pretrained(source)
    ck = MiniMaxM3SourceCheckpoint(source, cfg)
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    experts_sel = None if expert_subset is None else list(expert_subset)

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

        _bake_attention(writer, lp + "self_attn.", ck.attention(i), ck, i, cfg, group_size,
                        scale_dtype)

        if cfg.is_moe_layer(i):
            moe = ck.moe(i)
            if experts_sel is not None:  # bounded validation: subset the pre-stacked experts
                moe = dict(moe)
                moe["experts_gate_up"] = moe["experts_gate_up"][experts_sel]
                moe["experts_down"] = moe["experts_down"][experts_sel]
            _bake_moe_block(writer, lp + "block_sparse_moe.", moe, group_size, scale_dtype,
                            expert_bits)
            del moe
        else:
            for proj in DENSE_MLP_PROJS:                        # dense FFN (layers 0–2) → int8
                _write_int8(writer, lp + f"mlp.{proj}", ck.dense_mlp(i)[proj], group_size, scale_dtype)

        del norms
        ck.release()
        mx.clear_cache()

    vision_tensors = _bake_vision(writer, Path(source)) if include_vision else 0

    # Scheme counts: int8 vs int{expert_bits} affine are distinguished by bits (the only non-int8
    # affine entries are the routed-expert stacks); everything else is dense (norms/router/indexer/
    # head/vision).
    counts = {"int8": 0, "expert_int": 0, "dense": 0}
    for entry in writer.manifest.values():
        if entry["format"] == "affine_packed":
            counts["expert_int" if entry.get("bits") == expert_bits else "int8"] += 1
        else:
            counts["dense"] += 1

    scale_tag = "bf16" if scale_dtype == mx.bfloat16 else "fp32"
    policy = {
        "experts": f"int{expert_bits} affine g{group_size}",
        "non_experts": f"int8 affine g{group_size}",
        "norms_router_indexer_head_vision": "bf16/f32",
        "scales": scale_tag,
        "full_vl": bool(include_vision),
        "native_context": cfg.max_position_embeddings,
    }
    writer.finalize(policy)  # flushes shards + writes index/config/manifest
    _assert_native_1m_context(Path(out_dir), cfg)              # M3 is natively 1M; declare + assert it
    _copy_metadata_sidecars(Path(source), Path(out_dir), cfg)  # tokenizer + generation + VL preproc
    audit = _audit_self_contained(Path(out_dir), Path(source))  # rule 6: fail loud if not standalone

    out = Path(out_dir)
    total_bytes = sum(p.stat().st_size for p in out.glob("model-*.safetensors"))
    return {
        "layers": n,
        "experts_per_layer": (cfg.num_local_experts if experts_sel is None else len(experts_sel)),
        "vision_tensors": vision_tensors,
        "expert_bits": expert_bits,
        "counts": counts,
        "bytes": total_bytes,
        "self_contained": audit,
    }
