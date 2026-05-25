"""Gate: GLM-5.1 streamed bf16 loader — model-free, tiny synthetic checkpoint (~KB), ~0 GB.

Synthesizes a 2-layer (1 dense + 1 MoE) + 1-MTP GLM checkpoint with the **real key schema** but
toy dims, then round-trips every :class:`quanta.glm.loader.GLMSourceCheckpoint` accessor: shapes
match the confirmed checkpoint layout, ``expert_stacks`` stacks the routed experts in id order
``[E,out,in]``, the MTP block exposes its full param set, and a missing key fails loud (rule 6).
No real weights are touched.

    uv run --with numpy python -m parity.glm_loader_test
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.glm.config import GLMConfig
from quanta.glm.loader import GLMSourceCheckpoint

CFG = {
    "model_type": "glm_moe_dsa", "vocab_size": 10, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
    "q_lora_rank": 4, "kv_lora_rank": 4, "qk_nope_head_dim": 4, "qk_rope_head_dim": 2,
    "qk_head_dim": 6, "v_head_dim": 6,
    "index_head_dim": 3, "index_n_heads": 2, "index_topk": 4,
    "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 4,
    "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "eos_token_id": [9], "pad_token_id": 9, "tie_word_embeddings": False,
}


def _w(*shape, fill=0.0):
    return (mx.ones(shape, dtype=mx.bfloat16) * fill) if fill else mx.zeros(shape, dtype=mx.bfloat16)


def _attn_block(t: dict, p: str, c: GLMConfig) -> None:
    nh, h = c.num_attention_heads, c.hidden_size
    t[p + "self_attn.q_a_proj.weight"] = _w(c.q_lora_rank, h)
    t[p + "self_attn.q_a_layernorm.weight"] = _w(c.q_lora_rank)
    t[p + "self_attn.q_b_proj.weight"] = _w(nh * c.qk_head_dim, c.q_lora_rank)
    t[p + "self_attn.kv_a_proj_with_mqa.weight"] = _w(c.kv_lora_rank + c.qk_rope_head_dim, h)
    t[p + "self_attn.kv_a_layernorm.weight"] = _w(c.kv_lora_rank)
    t[p + "self_attn.kv_b_proj.weight"] = _w(nh * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank)
    t[p + "self_attn.o_proj.weight"] = _w(h, nh * c.v_head_dim)
    ip = p + "self_attn.indexer."
    t[ip + "wq_b.weight"] = _w(c.index_n_heads * c.index_head_dim, c.q_lora_rank)
    t[ip + "wk.weight"] = _w(c.index_head_dim, h)
    t[ip + "weights_proj.weight"] = _w(c.index_n_heads, h)
    t[ip + "k_norm.weight"] = _w(c.index_head_dim)
    t[ip + "k_norm.bias"] = _w(c.index_head_dim)


def _build(d: Path) -> GLMConfig:
    c = GLMConfig.from_dict(CFG)
    h, mi = c.hidden_size, c.moe_intermediate_size
    t: dict[str, mx.array] = {
        "model.embed_tokens.weight": _w(c.vocab_size, h),
        "model.norm.weight": _w(h),
        "lm_head.weight": _w(c.vocab_size, h),
    }
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = _w(h)
        t[p + "post_attention_layernorm.weight"] = _w(h)
        _attn_block(t, p, c)
        if c.is_dense_layer(i):
            t[p + "mlp.gate_proj.weight"] = _w(c.intermediate_size, h)
            t[p + "mlp.up_proj.weight"] = _w(c.intermediate_size, h)
            t[p + "mlp.down_proj.weight"] = _w(h, c.intermediate_size)
        else:
            t[p + "mlp.gate.weight"] = _w(c.n_routed_experts, h)
            t[p + "mlp.gate.e_score_correction_bias"] = mx.zeros((c.n_routed_experts,), dtype=mx.float32)
            for j in range(c.n_routed_experts):
                ep = f"{p}mlp.experts.{j}."
                t[ep + "gate_proj.weight"] = _w(mi, h, fill=float(j))  # fill==id -> verify stack order
                t[ep + "up_proj.weight"] = _w(mi, h)
                t[ep + "down_proj.weight"] = _w(h, mi)
            for proj, sh in (("gate_proj", (mi, h)), ("up_proj", (mi, h)), ("down_proj", (h, mi))):
                t[f"{p}mlp.shared_experts.{proj}.weight"] = _w(*sh)
    # MTP block at layer num_hidden_layers
    mp = f"model.layers.{c.mtp_layer_id}."
    t[mp + "enorm.weight"] = _w(h)
    t[mp + "hnorm.weight"] = _w(h)
    t[mp + "eh_proj.weight"] = _w(h, 2 * h)
    t[mp + "shared_head.norm.weight"] = _w(h)
    t[mp + "input_layernorm.weight"] = _w(h)
    t[mp + "post_attention_layernorm.weight"] = _w(h)
    _attn_block(t, mp, c)
    t[mp + "mlp.gate.weight"] = _w(c.n_routed_experts, h)
    t[mp + "mlp.gate.e_score_correction_bias"] = mx.zeros((c.n_routed_experts,), dtype=mx.float32)
    for j in range(c.n_routed_experts):
        ep = f"{mp}mlp.experts.{j}."
        t[ep + "gate_proj.weight"] = _w(mi, h, fill=float(j))
        t[ep + "up_proj.weight"] = _w(mi, h)
        t[ep + "down_proj.weight"] = _w(h, mi)
    for proj, sh in (("gate_proj", (mi, h)), ("up_proj", (mi, h)), ("down_proj", (h, mi))):
        t[f"{mp}mlp.shared_experts.{proj}.weight"] = _w(*sh)

    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), t)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))
    (d / "config.json").write_text(json.dumps(CFG))
    return c


def run() -> None:
    ok = True
    d = Path(tempfile.mkdtemp(prefix="glm_loader_"))
    try:
        c = _build(d)
        ck = GLMSourceCheckpoint(d)
        h, mi, E = c.hidden_size, c.moe_intermediate_size, c.n_routed_experts

        good = (tuple(ck.embed().shape) == (c.vocab_size, h)
                and tuple(ck.final_norm().shape) == (h,)
                and tuple(ck.lm_head().shape) == (c.vocab_size, h) and ck.num_layers == 2)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] top-level: embed/norm/lm_head shapes")

        a = ck.attention(1)
        good = (tuple(a["q_b_proj"].shape) == (c.num_attention_heads * c.qk_head_dim, c.q_lora_rank)
                and tuple(a["kv_a_proj_with_mqa"].shape) == (c.kv_lora_rank + c.qk_rope_head_dim, h)
                and tuple(a["kv_b_proj"].shape) == (c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank)
                and tuple(a["o_proj"].shape) == (h, c.num_attention_heads * c.v_head_dim)
                and tuple(a["indexer"]["wq_b"].shape) == (c.index_n_heads * c.index_head_dim, c.q_lora_rank)
                and tuple(a["indexer"]["wk"].shape) == (c.index_head_dim, h)
                and tuple(a["indexer"]["k_norm_bias"].shape) == (c.index_head_dim,))
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] attention + DSA indexer shapes")

        r = ck.moe_router(1)
        es = ck.expert_stacks(1)
        order_ok = all(abs(float(es["gate_proj"][j].mean().item()) - j) < 1e-3 for j in range(E))
        good = (tuple(r["weight"].shape) == (E, h) and r["e_score_correction_bias"].dtype == mx.float32
                and tuple(es["gate_proj"].shape) == (E, mi, h)
                and tuple(es["down_proj"].shape) == (E, h, mi) and order_ok)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] router + expert_stacks {tuple(es['gate_proj'].shape)} id-order={order_ok}")

        dm = ck.dense_mlp(0)
        good = (tuple(dm["gate_proj"].shape) == (c.intermediate_size, h)
                and tuple(dm["down_proj"].shape) == (h, c.intermediate_size))
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] dense_mlp (L0) shapes")

        m = ck.mtp()
        good = (tuple(m["eh_proj"].shape) == (h, 2 * h) and tuple(m["enorm"].shape) == (h,)
                and tuple(m["shared_head_norm"].shape) == (h,)
                and tuple(m["experts"]["gate_proj"].shape) == (E, mi, h)
                and "indexer" in m["attention"])
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] mtp block: eh_proj+norms+attention+experts")

        try:
            ck.mtp(1)
            mtp_ok = False
        except IndexError:
            mtp_ok = True
        try:
            ck._tensor("model.layers.0.nonexistent.weight")
            miss_ok = False
        except KeyError:
            miss_ok = True
        good = mtp_ok and miss_ok
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] fail-loud: mtp(1)->IndexError missing-key->KeyError")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
