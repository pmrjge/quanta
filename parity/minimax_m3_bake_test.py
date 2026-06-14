"""Model-free gate for the MiniMax-M3-VL bake (int4 + int6) — a tiny synthetic checkpoint, no real weights.

Builds a few-KB synthetic M3-VL checkpoint on disk (nested config + one safetensors shard + index +
tokenizer/generation stubs), runs the REAL :func:`quanta.minimax.bake_m3.bake_minimax_m3` entry point
over it (the same code path the 809 GiB bake takes — dense L0–2, MoE L3 with the trained sparse
indexer, vision passthrough, native-1M assert, self-contained audit), and round-trips the result back
through BOTH :class:`quanta.minimax.loader_m3.MiniMaxM3SourceCheckpoint` (the bf16 source reader) and
:class:`quanta.minimax.artifact_m3.MiniMaxM3Artifact` (the dequant-on-read baked reader). Catches
stub-vs-real interface rot in the bake + artifact surface without touching the real model.

Checks (rel = ‖Δ‖ / ‖ref‖):

  (a) the artifact dequant of every quantized tensor is **bit-identical** to the RTN round-trip of the
      source tensor (int8 attention/dense-FFN/shared, int4/int6 routed-expert stacks) — proves the bake
      quantizes exactly what the reader dequantizes, at the manifest-recorded width;
  (b) the **F32 router gate + e_score_correction_bias survive as F32** (read via ``get`` not ``read``)
      — bit-identical to the source, NOT bf16-downcast (a downcast could flip a top-k tie ⇒ a
      different expert; this is the M3-specific precision invariant);
  (c) dense norms / embed / head round-trip bf16-exact; the manifest schemes are right (experts
      affine_packed bits=4/6, non-experts bits=8, norms/router/indexer/vision dense); ``raw`` returns
      3-D expert codes and refuses a dense key; the text reader refuses a vision key (the ViT weights
      ARE baked, reached via the vision track);
  (d) the artifact is self-contained (the bake's own audit ran) and its config declares the native 1M
      window + the ``quanta_long_context`` marker; the scheme counts match (2 int6 stacks, the int8
      projections, the dense control + vision).

Group size 8 (divides the synthetic in-dims 32/24/16); the real bake uses 64.

    uv run python -m parity.minimax_m3_bake_test
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.quant import quantize_affine
from quanta.minimax.artifact_m3 import MiniMaxM3Artifact
from quanta.minimax.bake_m3 import bake_minimax_m3
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.loader_m3 import MiniMaxM3SourceCheckpoint

GS = 32  # MLX affine group size (32/64/128 supported); divides every synthetic in-dim. real bake = 64
BITS_INT8 = 8  # non-expert width (fixed); the routed-expert width is swept: int4 (shipped) + int6 (retired)

# tiny dims — every QUANTIZED in-dim (hidden, q_dim, dense/shared/moe inter) is a multiple of GS=32
VOCAB, H, HD = 48, 64, 16
NH, NKV, ROT = 4, 2, 8
DENSE_INTER, MOE_INTER, SHARED_INTER = 64, 32, 32
E, TOPK = 4, 2
IDX_HEADS, IDX_DIM = 2, 8
N_LAYERS = 4  # L0–2 dense, L3 MoE+sparse
NATIVE_CTX = 1_048_576

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _rel(a: mx.array, b: mx.array) -> float:
    return float((mx.linalg.norm((a - b).astype(mx.float32))
                  / (mx.linalg.norm(b.astype(mx.float32)) + 1e-12)).item())


def _bf(shape, k) -> mx.array:
    return (mx.random.normal(shape, key=k) * 0.1).astype(mx.bfloat16)


def _build_checkpoint() -> str:
    """Write a tiny but ``from_pretrained``-valid synthetic MiniMax-M3-VL checkpoint to a tempdir."""
    d = tempfile.mkdtemp(suffix="_m3_bake_gate")
    p = Path(d)
    tc = {
        "vocab_size": VOCAB, "hidden_size": H, "intermediate_size": MOE_INTER,
        "dense_intermediate_size": DENSE_INTER, "shared_intermediate_size": SHARED_INTER,
        "num_hidden_layers": N_LAYERS, "num_attention_heads": NH, "num_key_value_heads": NKV,
        "head_dim": HD, "rotary_dim": ROT, "use_qk_norm": True, "qk_norm_type": "per_head",
        "use_gemma_norm": True, "attention_output_gate": False, "rope_theta": 5e6,
        "num_local_experts": E, "num_experts_per_tok": TOPK, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "use_routing_bias": True, "norm_topk_prob": True,
        "routed_scaling_factor": 2.0, "moe_layer_freq": [0, 0, 0, 1],
        "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "sparse_attention_config": {
            "use_sparse_attention": True, "sparse_attention_freq": [0, 0, 0, 1],
            "sparse_topk_blocks": 2, "sparse_block_size": 4, "sparse_num_index_heads": IDX_HEADS,
            "sparse_index_dim": IDX_DIM, "sparse_init_block": 0, "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
        "num_mtp_modules": 7, "rms_norm_eps": 1e-6, "max_position_embeddings": NATIVE_CTX,
        "bos_token_id": 200019, "eos_token_id": 200020, "tie_word_embeddings": False,
    }
    vc = {"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
          "intermediate_size": 32, "patch_size": 14, "image_size": 56, "projection_dim": H}
    conf = {"model_type": "minimax_m3_vl", "text_config": tc, "vision_config": vc,
            "image_token_index": 200025, "video_token_index": 200026, "tie_word_embeddings": False,
            "max_position_embeddings": NATIVE_CTX}
    (p / "config.json").write_text(json.dumps(conf))
    # servable sidecars so the bake's metadata copy + audit pass
    (p / "generation_config.json").write_text(json.dumps({"eos_token_id": 200020,
                                                          "bos_token_id": 200019}))
    (p / "tokenizer_config.json").write_text(json.dumps({"model_type": "minimax_m3_vl"}))
    (p / "tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {"vocab": {}}}))
    (p / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "MiniMax"}))

    ks = mx.random.split(mx.random.key(0), 256)
    c = iter(range(256))
    t: dict[str, mx.array] = {}
    t["language_model.model.embed_tokens.weight"] = _bf((VOCAB, H), ks[next(c)])
    t["language_model.model.norm.weight"] = _bf((H,), ks[next(c)])
    t["language_model.lm_head.weight"] = _bf((VOCAB, H), ks[next(c)])
    for i in range(N_LAYERS):
        lp = f"language_model.model.layers.{i}."
        t[lp + "input_layernorm.weight"] = _bf((H,), ks[next(c)])
        t[lp + "post_attention_layernorm.weight"] = _bf((H,), ks[next(c)])
        sp = lp + "self_attn."
        t[sp + "q_proj.weight"] = _bf((NH * HD, H), ks[next(c)])
        t[sp + "k_proj.weight"] = _bf((NKV * HD, H), ks[next(c)])
        t[sp + "v_proj.weight"] = _bf((NKV * HD, H), ks[next(c)])
        t[sp + "o_proj.weight"] = _bf((H, NH * HD), ks[next(c)])
        t[sp + "q_norm.weight"] = _bf((HD,), ks[next(c)])
        t[sp + "k_norm.weight"] = _bf((HD,), ks[next(c)])
        is_moe = i == 3
        if is_moe:  # trained block-sparse indexer (sparse layers only)
            t[sp + "index_q_proj.weight"] = _bf((IDX_HEADS * IDX_DIM, H), ks[next(c)])
            t[sp + "index_k_proj.weight"] = _bf((IDX_DIM, H), ks[next(c)])
            t[sp + "index_q_norm.weight"] = _bf((IDX_DIM,), ks[next(c)])
            t[sp + "index_k_norm.weight"] = _bf((IDX_DIM,), ks[next(c)])
            mp = lp + "block_sparse_moe."
            t[mp + "gate.weight"] = (mx.random.normal((E, H), key=ks[next(c)])).astype(mx.float32)
            t[mp + "e_score_correction_bias"] = (mx.random.normal((E,), key=ks[next(c)])
                                                 ).astype(mx.float32)
            t[mp + "shared_experts.gate_proj.weight"] = _bf((SHARED_INTER, H), ks[next(c)])
            t[mp + "shared_experts.up_proj.weight"] = _bf((SHARED_INTER, H), ks[next(c)])
            t[mp + "shared_experts.down_proj.weight"] = _bf((H, SHARED_INTER), ks[next(c)])
            for e in range(E):
                ep = mp + f"experts.{e}."
                t[ep + "w1.weight"] = _bf((MOE_INTER, H), ks[next(c)])
                t[ep + "w3.weight"] = _bf((MOE_INTER, H), ks[next(c)])
                t[ep + "w2.weight"] = _bf((H, MOE_INTER), ks[next(c)])
        else:  # dense FFN
            t[lp + "mlp.gate_proj.weight"] = _bf((DENSE_INTER, H), ks[next(c)])
            t[lp + "mlp.up_proj.weight"] = _bf((DENSE_INTER, H), ks[next(c)])
            t[lp + "mlp.down_proj.weight"] = _bf((H, DENSE_INTER), ks[next(c)])
    # a few vision tensors (full-VL passthrough) — dense bf16 verbatim, prefix-keyed
    t["vision_tower.vision_model.pre_layrnorm.weight"] = _bf((16,), ks[next(c)])
    t["vision_tower.vision_model.pre_layrnorm.bias"] = _bf((16,), ks[next(c)])
    t["multi_modal_projector.linear_1.weight"] = _bf((H, 16), ks[next(c)])
    t["multi_modal_projector.linear_1.bias"] = _bf((H,), ks[next(c)])
    t["patch_merge_mlp.linear_1.weight"] = _bf((16, 16), ks[next(c)])

    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(p / shard), t)
    (p / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in t}}))
    return d


def _rtn(w: mx.array, bits: int) -> mx.array:
    """The exact RTN round-trip the bake applies (bf16 scales): dequant(quantize(w))."""
    pq, sc, b = quantize_affine(w, bits, GS, scale_dtype=mx.bfloat16)
    return mx.dequantize(pq, sc, b, group_size=GS, bits=bits).astype(mx.bfloat16)


def _check_bake(src: str, cfg: MiniMaxM3Config, bits: int) -> None:
    """Bake the synthetic checkpoint at routed-expert width ``bits`` and round-trip it through BOTH
    readers (the same checks for any width — the bake/artifact path is bits-agnostic, the manifest
    carries the width). Cleans up its own ``out`` dir."""
    out = f"{src}_int{bits}g64"
    try:
        stats = bake_minimax_m3(src, out, group_size=GS, expert_bits=bits, scale_dtype=mx.bfloat16)
        ck = MiniMaxM3SourceCheckpoint(src, cfg)          # bf16 source reader
        art = MiniMaxM3Artifact(out)                       # dequant-on-read baked reader
        man = json.loads((Path(out) / "manifest.json").read_text())["tensors"]
        L_DENSE, L_MOE = 0, 3

        # (a) quantized tensors dequant bit-identical to the source RTN round-trip ----------------
        a_src = ck.attention(L_MOE)
        a_art = art.attention(L_MOE)
        attn_err = max(_rel(a_art[s], _rtn(a_src[s], BITS_INT8))
                       for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))
        dm_src, dm_art = ck.dense_mlp(L_DENSE), art.dense_mlp(L_DENSE)
        dense_err = max(_rel(dm_art[pj], _rtn(dm_src[pj], BITS_INT8)) for pj in dm_src)
        m_src, m_art = ck.moe(L_MOE), art.moe(L_MOE)
        eg_err = _rel(m_art["experts_gate_up"], _rtn(m_src["experts_gate_up"], bits))
        ed_err = _rel(m_art["experts_down"], _rtn(m_src["experts_down"], bits))
        shared_err = max(_rel(m_art[f"shared_{pj}"], _rtn(m_src[f"shared_{pj}"], BITS_INT8))
                         for pj in ("gate_proj", "up_proj", "down_proj"))
        _ck(attn_err < 1e-6, f"int8 attn dequant != source RTN: {attn_err:.2e}")
        _ck(dense_err < 1e-6, f"int8 dense-FFN dequant != source RTN: {dense_err:.2e}")
        _ck(eg_err < 1e-6 and ed_err < 1e-6, f"int{bits} expert dequant != source RTN: {eg_err:.2e}/{ed_err:.2e}")
        _ck(shared_err < 1e-6, f"int8 shared dequant != source RTN: {shared_err:.2e}")

        # (b) F32 router gate + bias survive as F32, bit-identical (NOT bf16-downcast) -------------
        gate_f32 = m_art["gate"].dtype == mx.float32 and m_src["gate"].dtype == mx.float32
        bias_f32 = m_art["e_score_correction_bias"].dtype == mx.float32
        gate_exact = _rel(m_art["gate"], m_src["gate"]) == 0.0
        bias_exact = _rel(m_art["e_score_correction_bias"], m_src["e_score_correction_bias"]) == 0.0
        _ck(gate_f32 and bias_f32, "router gate/bias not F32 in artifact (downcast — routing precision lost)")
        _ck(gate_exact and bias_exact, "router gate/bias not bit-identical to source (F32 not preserved)")

        # (c) dense round-trip + manifest schemes + raw/refusals ----------------------------------
        n_src, n_art = ck.block_norms(L_MOE), art.block_norms(L_MOE)
        norm_exact = all(_rel(n_art[k], n_src[k].astype(mx.bfloat16)) == 0.0 for k in n_src)
        embed_exact = _rel(art.embed(), ck.embed().astype(mx.bfloat16)) == 0.0
        ix_src, ix_art = ck.sparse_index(L_MOE), art.sparse_index(L_MOE)
        index_exact = all(_rel(ix_art[k], ix_src[k].astype(mx.bfloat16)) == 0.0 for k in ix_src)
        _ck(norm_exact and embed_exact and index_exact, "dense norms/embed/indexer did not round-trip exact")

        mp = "language_model.model.layers.3.block_sparse_moe."
        sp = "language_model.model.layers.3.self_attn."
        schemes_ok = (
            man[mp + "experts.gate_up_proj"]["format"] == "affine_packed"
            and man[mp + "experts.gate_up_proj"]["bits"] == bits
            and man[mp + "experts.down_proj"]["bits"] == bits
            and man[sp + "q_proj"]["format"] == "affine_packed" and man[sp + "q_proj"]["bits"] == BITS_INT8
            and man[mp + "gate.weight"]["format"] == "dense"
            and man[mp + "gate.weight"]["dtype"] == "float32"
            and man[mp + "e_score_correction_bias"]["dtype"] == "float32"
            and man[sp + "q_norm.weight"]["format"] == "dense"
            and man[sp + "index_q_proj.weight"]["format"] == "dense"
            and man["vision_tower.vision_model.pre_layrnorm.weight"]["format"] == "dense"
            and man["multi_modal_projector.linear_1.bias"]["format"] == "dense"
        )
        _ck(schemes_ok, "manifest schemes wrong (expert/int8/dense partition)")

        raw_ok = art.raw(mp + "experts.gate_up_proj").ndim == 3
        try:
            art.raw("language_model.model.norm.weight")
            dense_refuse = False
        except ValueError:
            dense_refuse = True
        try:
            art.read("vision_tower.vision_model.pre_layrnorm.weight")
            vision_refuse = False
        except KeyError:
            vision_refuse = True
        _ck(raw_ok and dense_refuse and vision_refuse,
            f"raw/refusal contract broken: raw3d={raw_ok} dense_refuse={dense_refuse} vis_refuse={vision_refuse}")

        # (d) self-contained audit + native-1M config + scheme counts -----------------------------
        conf = json.loads((Path(out) / "config.json").read_text())
        tcc = conf.get("text_config", conf)
        ctx_ok = (tcc.get("max_position_embeddings") == NATIVE_CTX
                  and conf.get("quanta_long_context", {}).get("max_context") == NATIVE_CTX
                  and conf["quanta_long_context"]["yarn_dynamic"] is False)
        _ck(ctx_ok, f"artifact config does not declare native 1M + marker: {conf.get('quanta_long_context')}")
        audit_ok = stats["self_contained"]["leaks"] == "none" and stats["self_contained"]["symlinks"] == 0
        counts_ok = (stats["counts"]["expert_int"] == 2          # 1 MoE layer × {gate_up, down}
                     and stats["vision_tensors"] == 5
                     and stats["expert_bits"] == bits)
        _ck(audit_ok, f"self-contained audit not clean: {stats['self_contained']}")
        _ck(counts_ok, f"scheme/vision counts wrong: {stats['counts']} vis={stats['vision_tensors']}")

        print(f"\n=== MiniMax-M3-VL bake gate (model-free, tiny synthetic) — int{bits} experts ===")
        print(f"(a) int8 attn {attn_err:.1e} / dense {dense_err:.1e} / shared {shared_err:.1e}; "
              f"int{bits} experts {eg_err:.1e}/{ed_err:.1e}  (all == source RTN)")
        print(f"(b) router gate+bias F32-preserved bit-exact: {gate_exact and bias_exact}")
        print("(c) dense round-trip exact; manifest schemes ok; raw 3-D + dense/vision refusals ok")
        print(f"(d) self-contained {audit_ok}; native 1M + marker {ctx_ok}; counts {stats['counts']} "
              f"vision {stats['vision_tensors']}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def run() -> None:
    mx.random.seed(0)
    src = _build_checkpoint()
    try:
        cfg = MiniMaxM3Config.from_pretrained(src)
        for bits in (4, 6):  # int4 = the shipped width going forward; int6 = the retired arm, still gated
            _check_bake(src, cfg, bits)
        print(f"\nPARITY-CHECKS: {_N}")
        print("PASS — M3 bake: int4/int6 experts + int8 non-experts dequant == source RTN; F32 router "
              "preserved; dense/vision verbatim; native-1M self-contained VL artifact round-trips.")
    finally:
        shutil.rmtree(src, ignore_errors=True)


if __name__ == "__main__":
    run()
