"""Model-free gate for the Qwen3.5 N0 config/bake logic: ChatML eos derivation, the 1M-context
artifact round-trip, and the MTP-presence refine — all on SYNTHETIC temp-dir checkpoints (no weights,
no real model). Complements the real-path ``parity/nex_n2_pro_fit_test.py`` (which needs the 739 GB
Nex checkpoint); this runs anywhere, in the model-free sweep.

Covers the three N0 fixes:

* **eos (rule 6).** A Nex-like source (no ``generation_config.json``, ``config.json`` eos = lone
  ``<|endoftext|>``) derives the real ChatML stop set ``{<|im_end|>, <|endoftext|>}`` from the
  tokenizer; a 35B-like source (WITH ``generation_config.json``) trusts it unchanged.
* **1M round-trip.** The bake writes standard YaRN + a ``quanta_long_context`` block + raises
  ``max_position_embeddings`` to the 1M target; re-opening the artifact reads back the 1M served
  window with the dynamic-YaRN baseline (262144) intact — and synthesizes a correct
  ``generation_config.json`` when the source lacked one.
* **MTP refine.** A config that DECLARES an MTP head but whose index ships no ``mtp.*`` weight refines
  ``num_mtp_modules`` to 0 (Nex); an index that DOES carry ``mtp.*`` keeps it (the base model).

    uv run python -m parity.qwen35_config_eos_yarn_test
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from quanta.qwen35.bake import _bake_long_context, _copy_metadata_sidecars
from quanta.qwen35.config import Qwen35Config

_N = 0  # PARITY-CHECKS counter


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _text_config() -> dict:
    """A minimal-but-VALID qwen3_5_moe text_config (4 layers: 3 linear + 1 full), Nex-shaped where it
    matters (head_dim 256, partial rotary 0.25, mrope [11,11,10] summing to rotary_dim//2=32)."""
    return {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 1000, "hidden_size": 256, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
        "attn_output_gate": True, "partial_rotary_factor": 0.25,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": 0.25, "rope_theta": 1e7, "rope_type": "default"},
        "linear_num_key_heads": 16, "linear_num_value_heads": 64,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64, "norm_topk_prob": True,
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 262144,
        "eos_token_id": 248044,  # Nex-like: lone <|endoftext|>, NOT the chat turn-ender
        "tie_word_embeddings": False,
    }


def _write_checkpoint(d: Path, *, with_gen: bool, with_mtp_weights: bool,
                      gen_eos: list[int] | None = None) -> None:
    (d / "config.json").write_text(json.dumps({"text_config": _text_config(),
                                               "model_type": "qwen3_5_moe"}))
    (d / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 248045, "content": "<|im_start|>", "special": True},
        {"id": 248046, "content": "<|im_end|>", "special": True},
        {"id": 248044, "content": "<|endoftext|>", "special": True},
    ]}))
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "add_bos_token": False}))
    wm = {"lm_head.weight": "model-00001.safetensors",
          "model.language_model.embed_tokens.weight": "model-00001.safetensors"}
    if with_mtp_weights:
        wm["mtp.fc.weight"] = "model-00001.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    if with_gen:
        (d / "generation_config.json").write_text(json.dumps(
            {"eos_token_id": gen_eos if gen_eos is not None else [248046, 248044]}))


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # --- Case A: Nex-like source (no generation_config, no mtp weights) ---------------------- #
        src = root / "nex_like"
        src.mkdir()
        _write_checkpoint(src, with_gen=False, with_mtp_weights=False)
        a = Qwen35Config.from_pretrained(src)
        _ck(a.eos_token_ids == (248046, 248044), f"A: chat stop set {a.eos_token_ids}")
        _ck(a.eos_token_id == 248046, "A: primary eos must be <|im_end|>")
        _ck(a.num_mtp_modules == 0, f"A: MTP must refine to 0 (no weights), got {a.num_mtp_modules}")
        _ck(a.max_position_embeddings == 262144, "A: source native window preserved")
        _ck(a.yarn_original_max == 262144 and a.yarn_dynamic, "A: dynamic-YaRN baseline native")
        _ck(a.effective_yarn_factor(8192) == 1.0, "A: no YaRN tax below the native window")
        _ck(a.effective_yarn_factor(1_010_000) > 3.8, "A: YaRN ramps past the native window")

        # --- Case B: base-like source (generation_config present, mtp weights present) ----------- #
        base = root / "base_like"
        base.mkdir()
        _write_checkpoint(base, with_gen=True, with_mtp_weights=True, gen_eos=[111, 222])
        b = Qwen35Config.from_pretrained(base)
        _ck(b.eos_token_ids == (111, 222), f"B: must trust generation_config eos, got {b.eos_token_ids}")
        _ck(b.num_mtp_modules == 1, f"B: MTP present must be kept, got {b.num_mtp_modules}")

        # --- Case C: the 1M bake round-trip (the user's explicit requirement) --------------------- #
        art = root / "artifact"
        art.mkdir()
        shutil.copyfile(src / "config.json", art / "config.json")  # ArtifactWriter.finalize analog
        _bake_long_context(art, a)                                  # write 1M YaRN into the config
        _copy_metadata_sidecars(src, art, a)                        # synthesize generation_config
        # the baked artifact ships no mtp weights ⇒ its index has none ⇒ reopen refines to 0
        (art / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "lm_head.weight": "model-00001.safetensors"}}))

        conf = json.loads((art / "config.json").read_text())
        _ck(conf["max_position_embeddings"] == a.max_context, "C: config must DECLARE the 1M window")
        rp = conf["text_config"]["rope_parameters"]
        _ck(rp["rope_type"] == "yarn" and rp["factor"] == 4.0
            and rp["original_max_position_embeddings"] == 262144, f"C: standard YaRN block {rp}")
        _ck(conf["text_config"]["rope_scaling"]["rope_type"] == "yarn", "C: HF rope_scaling mirror")
        _ck(rp["mrope_section"] == [11, 11, 10], "C: mRoPE preserved alongside YaRN")
        gen = json.loads((art / "generation_config.json").read_text())
        _ck(gen["eos_token_id"] == [248046, 248044], f"C: synthesized eos {gen.get('eos_token_id')}")

        c = Qwen35Config.from_pretrained(art)
        _ck(c.max_position_embeddings == 1_010_000, f"C: served window {c.max_position_embeddings}")
        _ck(c.yarn_original_max == 262144, f"C: YaRN baseline stays native, got {c.yarn_original_max}")
        _ck(c.max_context == 1_010_000 and c.yarn_factor == 4.0 and c.yarn_dynamic,
            "C: long-context policy read back")
        _ck(c.effective_yarn_factor(8192) == 1.0, "C: dynamic YaRN STILL off below native (not broken)")
        _ck(c.effective_yarn_factor(1_010_000) > 3.8, "C: dynamic YaRN ramps at 1M")
        _ck(c.eos_token_ids == (248046, 248044), f"C: artifact eos round-trips {c.eos_token_ids}")
        _ck(c.num_mtp_modules == 0, "C: artifact MTP absent")

    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — Qwen3.5 N0 config/bake logic: eos derivation, 1M round-trip, MTP refine ({_N} checks).")


if __name__ == "__main__":
    run()
