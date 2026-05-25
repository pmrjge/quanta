"""Gate: GLM-5.1 (``glm_moe_dsa``) config dataclass — model-free, ~0 GB.

Verifies :class:`quanta.glm.config.GLMConfig` parses the checkpoint config and that its derived dims
match the **empirically-confirmed** tensor shapes from the real safetensors headers (so the loader /
forward build to the right shapes), plus that the bake target (int4-AWQ g64 experts + int8 rest) fits
under the 490.4 GiB ceiling. Self-contained: the config values are inlined (the real ``config.json``
contents), so no checkpoint is read.

    uv run python -m parity.glm_config_test
"""

from __future__ import annotations

from quanta.glm.config import GLMConfig

# The real GLM-5.1 config.json (verbatim subset), incl. the transformers ``text_config``/unknown-key
# and ``rope_parameters`` shapes ``from_dict`` must tolerate.
RAW = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "vocab_size": 154880, "hidden_size": 6144, "intermediate_size": 12288,
    "num_hidden_layers": 78, "num_attention_heads": 64, "num_key_value_heads": 64,
    "rms_norm_eps": 1e-5, "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
    "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
    "qk_head_dim": 256, "v_head_dim": 256, "head_dim": 64,
    "index_head_dim": 128, "index_n_heads": 32, "index_topk": 2048, "indexer_rope_interleave": True,
    "n_routed_experts": 256, "num_experts_per_tok": 8, "n_shared_experts": 1,
    "moe_intermediate_size": 2048, "first_k_dense_replace": 3, "moe_layer_freq": 1,
    "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
    "routed_scaling_factor": 2.5, "norm_topk_prob": True,
    "num_nextn_predict_layers": 1, "rope_interleave": True, "max_position_embeddings": 202752,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "eos_token_id": [154820, 154827, 154829], "pad_token_id": 154820,
    "some_future_unknown_key": "ignored",
}

GiB = 1024 ** 3


def run() -> None:
    ok = True
    cfg = GLMConfig.from_dict(RAW)

    # (0) core fields parsed (incl. rope_theta lifted out of rope_parameters, eos tuple-ized)
    good = (cfg.hidden_size == 6144 and cfg.num_hidden_layers == 78 and cfg.vocab_size == 154880
            and cfg.rope_theta == 1000000.0 and cfg.eos_token_id == (154820, 154827, 154829)
            and cfg.model_type == "glm_moe_dsa" and not cfg.tie_word_embeddings)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] parse core: layers={cfg.num_hidden_layers} d={cfg.hidden_size} "
          f"theta={cfg.rope_theta:g} eos={cfg.eos_token_id}")

    # (1) derived dims match the confirmed safetensors shapes (out_features of each projection)
    h, nh = cfg.hidden_size, cfg.num_attention_heads
    checks = {
        "q_a_proj out (q_lora)": (cfg.q_lora_rank, 2048),
        "q_b_proj out (nh*qk)": (nh * cfg.qk_head_dim, 16384),
        "kv_a out (kv_lora+rope)": (cfg.kv_lora_rank + cfg.qk_rope_head_dim, 576),
        "kv_b out (nh*(nope+v))": (nh * (cfg.qk_nope_head_dim + cfg.v_head_dim), 28672),
        "o_proj in (nh*v)": (nh * cfg.v_head_dim, 16384),
        "indexer wq_b out (ih*idim)": (cfg.index_n_heads * cfg.index_head_dim, 4096),
        "indexer wk out (idim)": (cfg.index_head_dim, 128),
        "indexer weights_proj out (ih)": (cfg.index_n_heads, 32),
        "router gate out (experts)": (cfg.n_routed_experts, 256),
        "eh_proj in (2*d)": (2 * h, 12288),
    }
    bad = {k: v for k, (v, exp) in checks.items() if v != exp}
    good = not bad
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] derived dims vs confirmed shapes" + (f"  MISMATCH {bad}" if bad else ""))

    # (2) layer regime + MTP + softmax scale
    dense = [i for i in range(cfg.num_hidden_layers) if cfg.is_dense_layer(i)]
    good = (dense == [0, 1, 2] and cfg.is_moe_layer(3) and cfg.mtp_layer_id == 78
            and abs(cfg.softmax_scale - 1.0 / 16.0) < 1e-12)
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] regime: dense={dense} mtp_layer={cfg.mtp_layer_id} "
          f"softmax_scale={cfg.softmax_scale:.5f}")

    # (3) bake target fits: int4-AWQ g64 experts (4.5 bpp) + int8 rest (8.25 bpp) < 490.4 GiB
    n_moe = cfg.num_hidden_layers - cfg.first_k_dense_replace + cfg.num_nextn_predict_layers  # +MTP MoE
    exp_params = cfg.n_routed_experts * 3 * cfg.moe_intermediate_size * cfg.hidden_size * n_moe
    exp_gib = exp_params * 4.5 / 8 / GiB
    good = 380 < exp_gib < 390 and exp_gib + 25 < 490.4  # experts ~385 GiB, +~19 rest, big headroom
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] bake fit: experts(g64)={exp_gib:.1f} GiB over {n_moe} MoE layers "
          f"(+int8 rest ~19 GiB, ceiling 490.4)")

    # (4) fail-loud on an inconsistent config (rule 6)
    try:
        GLMConfig.from_dict({**RAW, "qk_head_dim": 999})
        good = False
    except ValueError:
        good = True
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] inconsistent qk_head_dim -> ValueError")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
