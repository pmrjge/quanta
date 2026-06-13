"""Model-free gate for the MiniMax-M3-VL M0 config logic: nested text+vision parsing, eos
derivation, the MTP-presence refine, per-layer dense/MoE + sparse-attention typing, and the quant
policy's rule-6 coverage over a SYNTHETIC index — all on temp-dir checkpoints (no weights, no real
model). Complements the real-path ``parity/minimax_m3_fit_test.py`` (which needs the 796 GB
checkpoint); this runs anywhere, in the model-free sweep.

Covers:

* **nested parse + eos.** A minimal ``minimax_m3_vl`` config (text_config + vision_config +
  multimodal wrapper) parses; eos comes from ``generation_config.json`` (the M3 case), falling
  back to ``text_config.eos_token_id`` when absent (rule 6).
* **MTP refine.** A config that DECLARES ``num_mtp_modules`` but whose index ships no ``mtp.*``
  weight refines the effective count to 0 (M3); an index that DOES carry ``mtp.*`` keeps it.
* **per-layer typing.** ``moe_layer_freq`` / ``sparse_attention_freq`` drive ``is_dense_layer`` /
  ``is_moe_layer`` / ``is_sparse_attention_layer``; a length mismatch fails loud (refuse to guess).
* **rule-6 coverage.** :func:`coverage` over a synthetic index (text keys + a few vision keys)
  reports no missing/extra and classifies experts→expert_int, attn/shared/dense-ffn→int8,
  norms/router/indexer/embed/vision→dense; :func:`project_resident` orders int4<int6<bf16.

    uv run python -m parity.minimax_m3_config_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.quant_policy_m3 import (
    DENSE,
    EXPERT_INT,
    INT8,
    coverage,
    expected_keymap,
    project_resident,
)

_N = 0  # PARITY-CHECKS counter


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _text_config() -> dict:
    """A minimal-but-VALID minimax_m3_vl text_config: 5 layers (2 dense + 3 MoE), M3-shaped where it
    matters (gemma norm, partial RoPE 0.5, per-head qk-norm, sigmoid+bias, clamped swiglu, trained
    sparse indexer on the MoE layers, declared MTP)."""
    return {
        "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 32,
        "dense_intermediate_size": 96, "shared_intermediate_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "rotary_dim": 8, "partial_rotary_factor": 0.5, "rope_theta": 5e6,
        "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
        "attention_output_gate": False,
        "num_local_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "use_routing_bias": True, "routed_scaling_factor": 2.0,
        "moe_layer_freq": [0, 0, 1, 1, 1],
        "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "num_mtp_modules": 7, "num_nextn_predict_layers": 1,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576,
        "tie_word_embeddings": False,
        "eos_token_id": 200020,  # used only when generation_config is absent
        "sparse_attention_config": {
            "use_sparse_attention": True, "sparse_index_dim": 128, "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16, "sparse_block_size": 128, "sparse_score_type": "max",
            "sparse_init_block": 0, "sparse_local_block": 1,
            "sparse_attention_freq": [0, 0, 1, 1, 1],
        },
    }


def _vision_config() -> dict:
    return {"hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
            "intermediate_size": 64, "patch_size": 14, "image_size": 224, "projection_dim": 64,
            "rope_theta": 10000.0, "rope_mode": "3d", "layer_norm_eps": 1e-5, "hidden_act": "gelu",
            "model_type": "clip_vision_model", "num_channels": 3,
            "img_token_compression_config": {"spatial_merge_size": 2, "temporal_patch_size": 2}}


def _index(cfg: MiniMaxM3Config, *, with_mtp: bool, with_vision: bool) -> dict:
    """Build a synthetic weight_map whose key set EXACTLY equals expected_keymap (+ optional vision /
    mtp), so coverage() can be exercised without real shards."""
    wm = {k: "model-00001.safetensors" for k in expected_keymap(cfg)}
    if with_vision:
        wm["vision_tower.vision_model.embeddings.patch_embedding.weight"] = "v.safetensors"
        wm["vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"] = "v.safetensors"
        wm["multi_modal_projector.linear_1.weight"] = "v.safetensors"
        wm["patch_merge_mlp.linear_1.weight"] = "v.safetensors"
    if with_mtp:
        wm["language_model.model.mtp.0.fc.weight"] = "model-00001.safetensors"
    return wm


def _write(d: Path, *, with_gen: bool, with_vision: bool = True, gen_eos=None) -> None:
    cfg = {"model_type": "minimax_m3_vl", "text_config": _text_config(),
           "image_token_index": 200025, "video_token_index": 200026, "image_seq_length": 576,
           "vision_feature_layer": -1, "vision_feature_select_strategy": "full",
           "projector_hidden_act": "gelu", "multimodal_projector_bias": True}
    if with_vision:
        cfg["vision_config"] = _vision_config()
    (d / "config.json").write_text(json.dumps(cfg))
    if with_gen:
        (d / "generation_config.json").write_text(json.dumps(
            {"bos_token_id": 200019, "eos_token_id": gen_eos if gen_eos is not None else 200020}))


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # --- Case A: M3-like source (generation_config present, no mtp weights) ------------------- #
        a_dir = root / "m3_like"
        a_dir.mkdir()
        _write(a_dir, with_gen=True)
        # parse WITHOUT an index first to see the declared MTP, then add the index for the refine
        a0 = MiniMaxM3Config.from_pretrained(a_dir)
        _ck(a0.num_mtp_modules_declared == 7, "A: declared MTP read from config")
        # now drop in an index WITHOUT mtp weights → effective MTP must refine to 0
        (a_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": _index(a0, with_mtp=False, with_vision=True)}))
        a = MiniMaxM3Config.from_pretrained(a_dir)
        _ck(a.eos_token_ids == (200020,), f"A: eos from generation_config {a.eos_token_ids}")
        _ck(a.bos_token_id == 200019, "A: bos from generation_config")
        _ck(a.num_mtp_modules == 0, f"A: MTP refine to 0 (no weights), got {a.num_mtp_modules}")
        _ck(a.max_position_embeddings == 1048576, "A: 1M native window")
        _ck((a.q_dim, a.kv_dim, a.n_rep) == (128, 32, 4), f"A: GQA geom {a.q_dim}/{a.kv_dim}/{a.n_rep}")
        _ck(a.partial_rotary_factor == 0.5 and a.use_gemma_norm and a.hidden_act == "swigluoai",
            "A: M3 attention/activation flags")
        _ck(a.has_shared_expert and a.routed_scaling_factor == 2.0 and a.scoring_func == "sigmoid",
            "A: MoE routing flags")
        _ck(a.vision is not None and a.vision.num_hidden_layers == 3, "A: vision sub-config parsed")

        # per-layer typing
        dense = [i for i in range(a.num_hidden_layers) if a.is_dense_layer(i)]
        moe = [i for i in range(a.num_hidden_layers) if a.is_moe_layer(i)]
        sparse = [i for i in range(a.num_hidden_layers) if a.is_sparse_attention_layer(i)]
        _ck(dense == [0, 1] and moe == [2, 3, 4], f"A: layer split {dense}/{moe}")
        _ck(sparse == [2, 3, 4], f"A: sparse-attn layers {sparse} (== MoE layers here)")

        # rule-6 coverage + scheme assignment
        wm = json.loads((a_dir / "model.safetensors.index.json").read_text())["weight_map"]
        cov = coverage(list(wm), a)
        _ck(not cov["missing"] and not cov["extra"], f"A: coverage miss/extra {cov['missing'][:3]} "
            f"{cov['extra'][:3]}")
        km = cov["keymap"]
        _ck(km["language_model.model.layers.2.block_sparse_moe.experts.0.w1.weight"] == EXPERT_INT,
            "A: routed expert -> expert_int")
        _ck(km["language_model.model.layers.2.block_sparse_moe.shared_experts.gate_proj.weight"]
            == INT8, "A: shared expert -> int8")
        _ck(km["language_model.model.layers.2.self_attn.q_proj.weight"] == INT8, "A: attn -> int8")
        _ck(km["language_model.model.layers.0.mlp.gate_proj.weight"] == INT8, "A: dense ffn -> int8")
        _ck(km["language_model.model.layers.2.self_attn.index_q_proj.weight"] == DENSE,
            "A: trained indexer -> dense (bf16)")
        _ck(km["language_model.model.layers.2.block_sparse_moe.gate.weight"] == DENSE,
            "A: router gate -> dense")
        _ck(km["vision_tower.vision_model.embeddings.patch_embedding.weight"] == DENSE,
            "A: vision -> dense")
        _ck(len(cov["vision"]) == 4, f"A: 4 vision keys classified, got {len(cov['vision'])}")

        # projection ordering (synthetic shapes: all bf16 numel from a flat size table)
        sizes = {k: ("BF16", 4096) for k in wm}
        p4 = project_resident(sizes, km, expert_bits=4)
        p6 = project_resident(sizes, km, expert_bits=6)
        _ck(p4["mix_gib"] < p6["mix_gib"] < p6["bf16_gib"], "A: mix order int4<int6<bf16")

        # --- Case B: index WITH mtp weights → effective MTP kept --------------------------------- #
        b_dir = root / "m3_with_mtp"
        b_dir.mkdir()
        _write(b_dir, with_gen=True)
        b0 = MiniMaxM3Config.from_pretrained(b_dir)
        (b_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": _index(b0, with_mtp=True, with_vision=False)}))
        b = MiniMaxM3Config.from_pretrained(b_dir)
        _ck(b.num_mtp_modules == 7, f"B: MTP present must be kept, got {b.num_mtp_modules}")

        # --- Case C: no generation_config → eos falls back to text_config ------------------------ #
        c_dir = root / "m3_no_gen"
        c_dir.mkdir()
        _write(c_dir, with_gen=False)
        c = MiniMaxM3Config.from_pretrained(c_dir)
        _ck(c.eos_token_ids == (200020,), f"C: eos fallback to text_config, got {c.eos_token_ids}")

        # --- Case D: malformed schedule fails loud (refuse to guess) ----------------------------- #
        d_dir = root / "m3_bad"
        d_dir.mkdir()
        bad = {"model_type": "minimax_m3_vl", "text_config": {**_text_config(),
               "moe_layer_freq": [0, 0, 1]}}  # length 3 != 5 layers
        (d_dir / "config.json").write_text(json.dumps(bad))
        raised = False
        try:
            MiniMaxM3Config.from_pretrained(d_dir)
        except ValueError:
            raised = True
        _ck(raised, "D: moe_layer_freq length mismatch must raise (refuse to guess)")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — MiniMax-M3-VL M0 config/policy logic: nested parse, eos, MTP refine, per-layer "
          f"typing, rule-6 coverage ({_N} checks).")


if __name__ == "__main__":
    run()
