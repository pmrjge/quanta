"""Gate: GLM-5.1 (``glm_moe_dsa``) assembled forward — model-free, tiny random weights, ~0 GB.

The capstone parity gate for tasks #83–#86. A tiny random GLM (small hidden / few heads / few experts /
4 layers = 1 dense + 3 MoE, with a randomized :class:`quanta.glm.model.GLMResidentModel`) is checked
end to end — every gate is parity-first (an optimized path equals its naive reference), so a divergence
is a forward-math bug:

* **#84 DSA indexer keep-all == dense.** With ``index_topk >= T`` the Lightning-Indexer keeps every
  causal token, so the indexer-masked MLA is **bit-identical** to plain causal MLA. With ``index_topk <
  T`` the selection actually *bites* (output differs), proving the top-k is real, not a no-op. Both
  also checked naive-vs-fast.
* **#83 attention + #85 MoE** are exercised inside the assembled block here and gated finer in
  ``parity/glm_attn_test.py`` / ``parity/glm_moe_test.py``.
* **#86 full 78-layer forward (here: the tiny analogue).**
  - the assembled forward is **finite** and the right shape ``[1,T,vocab]``;
  - **per-layer naive == fast**: each decoder block's optimized (``mx.fast.*``) path equals its naive
    path on the same residual (so the speed path is output-equivalent, layer by layer);
  - **incremental == prefill**: the resident model's KV-cached single-token decode equals the
    full-sequence prefill (and ``capture_layers`` returns the post-block hidden the MTP/spec path uses);
  - the **streamed bf16 reference forward** (``glm_logits``, one layer resident — rule 8) runs through
    the real :class:`quanta.glm.loader.GLMSourceCheckpoint` accessors on a tiny synthetic checkpoint
    (random weights) and equals the resident model's prefill, proving the loader→forward wiring.

The real bf16 teacher-forced-ppl gate (the true arbiter) needs the full checkpoint + GPU and is
DEFERRED — see :func:`quanta.glm.model.glm_teacher_forced_ppl` and that module's docstring. Here every
tensor is a few KB; no checkpoint weights are loaded.

    uv run --with numpy python -m parity.glm_forward_test
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import mlx.core as mx

from quanta.glm.attention import MLAAttention
from quanta.glm.config import GLMConfig
from quanta.glm.indexer import LightningIndexer
from quanta.glm.model import GLMResidentModel, glm_logits
from quanta.glm.loader import GLMSourceCheckpoint

CFG = {
    "model_type": "glm_moe_dsa", "vocab_size": 40, "hidden_size": 16, "intermediate_size": 24,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
    "q_lora_rank": 8, "kv_lora_rank": 6, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
    "qk_head_dim": 8, "v_head_dim": 8,
    "index_head_dim": 6, "index_n_heads": 2, "index_topk": 3,
    "n_routed_experts": 5, "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 6,
    "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "eos_token_id": [39], "pad_token_id": 39, "tie_word_embeddings": False,
}


def _randomize_module(module, rng: np.random.Generator, scale: float = 0.5) -> None:
    """Fill every leaf param of an ``nn.Module`` with small reproducible random values."""
    flat: dict[str, mx.array] = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            items = node.items()
        elif isinstance(node, (list, tuple)):
            items = enumerate(node)
        else:
            flat[prefix.rstrip(".")] = node
            return
        for k, v in items:
            walk(f"{prefix}{k}.", v)

    walk("", module.parameters())
    upd = [(k, mx.array((rng.standard_normal(v.shape) * scale).astype(np.float32))) for k, v in flat.items()]
    module.load_weights(upd, strict=False)


def _copy_params(src, dst) -> None:
    """Copy ``src``'s parameters into ``dst`` (same architecture / shapes)."""
    dst.update(src.parameters())


def _build_resident(cfg: GLMConfig, rng: np.random.Generator, *, use_fast: bool, use_indexer: bool):
    model = GLMResidentModel(cfg, use_fast=use_fast, use_indexer=use_indexer)
    _randomize_module(model, rng, scale=0.3)
    # randomize the routed expert stacks too (parameters() init leaves them as the zeros buffers)
    def r(*s):
        return mx.array((rng.standard_normal(s) * 0.2).astype(np.float32))

    for layer in model.layers:
        if hasattr(layer.mlp, "set_experts"):
            e, inter, h = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
            layer.mlp.set_experts(r(e, inter, h), r(e, inter, h), r(e, h, inter))
    return model


def _build_synthetic_ckpt(d: Path, cfg: GLMConfig, model: GLMResidentModel) -> None:
    """Write a tiny synthetic checkpoint (real GLM key schema) whose weights are the resident model's,
    so the streamed ``glm_logits`` can be diffed against the resident prefill (loader→forward wiring)."""
    h, mi, e = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts

    # store fp32 so the streamed-vs-resident diff isolates the loader→forward *wiring* from a bf16
    # round-trip (the real checkpoint is bf16; ``glm_logits`` is run here at dtype=fp32 to match).
    def W(arr):
        return mx.array(np.array(arr)).astype(mx.float32)

    t: dict[str, mx.array] = {
        "model.embed_tokens.weight": W(model.embed_tokens.weight),
        "model.norm.weight": W(model.norm.weight),
        "lm_head.weight": W(model.lm_head.weight),
    }
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = W(layer.input_layernorm.weight)
        t[p + "post_attention_layernorm.weight"] = W(layer.post_attention_layernorm.weight)
        sa = layer.self_attn
        t[p + "self_attn.q_a_proj.weight"] = W(sa.q_a_proj.weight)
        t[p + "self_attn.q_a_layernorm.weight"] = W(sa.q_a_layernorm.weight)
        t[p + "self_attn.q_b_proj.weight"] = W(sa.q_b_proj.weight)
        t[p + "self_attn.kv_a_proj_with_mqa.weight"] = W(sa.kv_a_proj_with_mqa.weight)
        t[p + "self_attn.kv_a_layernorm.weight"] = W(sa.kv_a_layernorm.weight)
        t[p + "self_attn.kv_b_proj.weight"] = W(sa.kv_b_proj.weight)
        t[p + "self_attn.o_proj.weight"] = W(sa.o_proj.weight)
        ix = layer.indexer
        ip = p + "self_attn.indexer."
        t[ip + "wq_b.weight"] = W(ix.wq_b.weight)
        t[ip + "wk.weight"] = W(ix.wk.weight)
        t[ip + "weights_proj.weight"] = W(ix.weights_proj.weight)
        t[ip + "k_norm.weight"] = W(ix.k_norm.weight)
        t[ip + "k_norm.bias"] = W(ix.k_norm.bias)
        if cfg.is_dense_layer(i):
            t[p + "mlp.gate_proj.weight"] = W(layer.mlp.gate_proj.weight)
            t[p + "mlp.up_proj.weight"] = W(layer.mlp.up_proj.weight)
            t[p + "mlp.down_proj.weight"] = W(layer.mlp.down_proj.weight)
        else:
            t[p + "mlp.gate.weight"] = W(layer.mlp.gate.weight)
            t[p + "mlp.gate.e_score_correction_bias"] = mx.array(
                np.array(layer.mlp.gate.e_score_correction_bias)).astype(mx.float32)
            gs, us, ds = layer.mlp.gate_stack, layer.mlp.up_stack, layer.mlp.down_stack
            for j in range(e):
                ep = f"{p}mlp.experts.{j}."
                t[ep + "gate_proj.weight"] = W(gs[j])
                t[ep + "up_proj.weight"] = W(us[j])
                t[ep + "down_proj.weight"] = W(ds[j])
            t[f"{p}mlp.shared_experts.gate_proj.weight"] = W(layer.mlp.shared_gate.weight)
            t[f"{p}mlp.shared_experts.up_proj.weight"] = W(layer.mlp.shared_up.weight)
            t[f"{p}mlp.shared_experts.down_proj.weight"] = W(layer.mlp.shared_down.weight)
    # a minimal MTP block so the loader's schema is complete (not exercised by this gate)
    mp = f"model.layers.{cfg.mtp_layer_id}."
    for nm, sh in (("enorm.weight", (h,)), ("hnorm.weight", (h,)), ("eh_proj.weight", (h, 2 * h)),
                   ("shared_head.norm.weight", (h,)), ("input_layernorm.weight", (h,)),
                   ("post_attention_layernorm.weight", (h,))):
        t[mp + nm] = mx.zeros(sh, dtype=mx.bfloat16)
    nh = cfg.num_attention_heads
    for nm, sh in (("self_attn.q_a_proj.weight", (cfg.q_lora_rank, h)),
                   ("self_attn.q_a_layernorm.weight", (cfg.q_lora_rank,)),
                   ("self_attn.q_b_proj.weight", (nh * cfg.qk_head_dim, cfg.q_lora_rank)),
                   ("self_attn.kv_a_proj_with_mqa.weight", (cfg.kv_lora_rank + cfg.qk_rope_head_dim, h)),
                   ("self_attn.kv_a_layernorm.weight", (cfg.kv_lora_rank,)),
                   ("self_attn.kv_b_proj.weight", (nh * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank)),
                   ("self_attn.o_proj.weight", (h, nh * cfg.v_head_dim)),
                   ("self_attn.indexer.wq_b.weight", (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)),
                   ("self_attn.indexer.wk.weight", (cfg.index_head_dim, h)),
                   ("self_attn.indexer.weights_proj.weight", (cfg.index_n_heads, h)),
                   ("self_attn.indexer.k_norm.weight", (cfg.index_head_dim,)),
                   ("self_attn.indexer.k_norm.bias", (cfg.index_head_dim,))):
        t[mp + nm] = mx.zeros(sh, dtype=mx.bfloat16)
    t[mp + "mlp.gate.weight"] = mx.zeros((e, h), dtype=mx.bfloat16)
    t[mp + "mlp.gate.e_score_correction_bias"] = mx.zeros((e,), dtype=mx.float32)
    for j in range(e):
        ep = f"{mp}mlp.experts.{j}."
        t[ep + "gate_proj.weight"] = mx.zeros((mi, h), dtype=mx.bfloat16)
        t[ep + "up_proj.weight"] = mx.zeros((mi, h), dtype=mx.bfloat16)
        t[ep + "down_proj.weight"] = mx.zeros((h, mi), dtype=mx.bfloat16)
    for proj, sh in (("gate_proj", (mi, h)), ("up_proj", (mi, h)), ("down_proj", (h, mi))):
        t[f"{mp}mlp.shared_experts.{proj}.weight"] = mx.zeros(sh, dtype=mx.bfloat16)

    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), t)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))
    (d / "config.json").write_text(json.dumps(CFG))


def _maxabs(a, b) -> float:
    return float(np.max(np.abs(np.array((a - b).astype(mx.float32)))))


def run() -> None:
    cfg = GLMConfig.from_dict(CFG)
    rng = np.random.default_rng(0)
    B, T = 1, 6
    x = mx.array((rng.standard_normal((B, T, cfg.hidden_size)) * 0.5).astype(np.float32))
    pos = mx.arange(T)
    ok = True

    # === (#84) DSA indexer keep-all == dense, and bites when topk < T =========================
    attn = MLAAttention(cfg)
    _randomize_module(attn, rng, scale=0.5)
    q_latent = attn.q_a_layernorm(attn.q_a_proj(x))
    o_dense = attn(x, pos, use_fast=False)

    cfg_keep = dataclasses.replace(cfg, index_topk=999)            # topk >= T -> keep all causal
    ix_keep = LightningIndexer(cfg_keep)
    _randomize_module(ix_keep, rng, scale=0.5)
    m_keep_n = ix_keep.select_mask(x, q_latent, pos, use_fast=False)
    m_keep_f = ix_keep.select_mask(x, q_latent, pos, use_fast=True)
    o_keep = attn(x, pos, use_fast=False, index_mask=m_keep_n)
    d_keep = _maxabs(o_dense, o_keep)
    finite_mask = np.array(mx.where(mx.isinf(m_keep_n), 0.0, m_keep_n).astype(mx.float32))
    keep_is_causal = bool(np.all((finite_mask == 0.0)))          # every causal entry kept (0), none -inf
    good = d_keep < 1e-5 and keep_is_causal
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] indexer keep-all == dense  |Δ|={d_keep:.2e} causal_mask={keep_is_causal}")
    d_keep_nf = _maxabs(mx.where(mx.isinf(m_keep_n), mx.array(0.0), m_keep_n),
                        mx.where(mx.isinf(m_keep_f), mx.array(0.0), m_keep_f))
    good = d_keep_nf < 1e-5
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] indexer mask naive == fast (keep-all)  |Δ|={d_keep_nf:.2e}")

    ix_bite = LightningIndexer(cfg)                               # topk = 3 < T = 6 -> selection bites
    _randomize_module(ix_bite, rng, scale=0.5)
    m_bite = ix_bite.select_mask(x, q_latent, pos, use_fast=False)
    o_bite = attn(x, pos, use_fast=False, index_mask=m_bite)
    n_dropped = int(np.sum(np.array(mx.isinf(m_bite))))          # some causal tokens dropped
    d_bite = _maxabs(o_dense, o_bite)
    good = d_bite > 1e-3 and n_dropped > 0
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] indexer top-k BITES (topk<T): drops {n_dropped} keys, "
          f"|Δ| vs dense={d_bite:.2e}")

    # === (#86) assembled forward: finite, per-layer naive==fast, incremental==prefill =========
    model_n = _build_resident(cfg_keep, np.random.default_rng(7), use_fast=False, use_indexer=True)
    model_f = _build_resident(cfg_keep, np.random.default_rng(7), use_fast=True, use_indexer=True)
    _copy_params(model_n, model_f)                               # identical weights, naive vs fast
    for ln, lf in zip(model_n.layers, model_f.layers):           # incl. the non-parameter expert stacks
        if hasattr(ln.mlp, "set_experts"):
            lf.mlp.set_experts(ln.mlp.gate_stack, ln.mlp.up_stack, ln.mlp.down_stack)

    ids = mx.array([3, 7, 1, 9, 2, 5])
    log_n = model_n(ids, offset=0)
    log_f = model_f(ids, offset=0)
    finite = bool(mx.all(mx.isfinite(log_n)).item()) and bool(mx.all(mx.isfinite(log_f)).item())
    good = log_n.shape == (1, T, cfg.vocab_size) and finite
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] assembled forward finite & shaped {tuple(log_n.shape)}")

    # per-layer naive == fast (each block's optimized path equals its naive path on the same residual)
    h_n = model_n.embed_tokens(ids.reshape(1, -1))
    worst = 0.0
    for ln, lf in zip(model_n.layers, model_f.layers):
        o_ln = ln(h_n, pos, use_fast=False, use_indexer=True)
        o_lf = lf(h_n, pos, use_fast=True, use_indexer=True)
        worst = max(worst, _maxabs(o_ln, o_lf))
        h_n = o_ln
    good = worst < 1e-3
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] per-layer naive == fast (worst over {cfg.num_hidden_layers} layers)  |Δ|={worst:.2e}")

    # incremental (KV-cached, single-token) == prefill, + capture_layers
    cache = model_n.make_caches()
    cols = [model_n(mx.array([int(ids[k].item())]), caches=cache, offset=k) for k in range(T)]
    log_inc = mx.concatenate(cols, axis=1)
    d_inc = _maxabs(log_n, log_inc)
    last = cfg.num_hidden_layers - 1
    _, caps = model_n(ids, offset=0, capture_layers=(last,))
    cap_ok = last in caps and tuple(caps[last].shape) == (T, cfg.hidden_size)
    good = d_inc < 1e-4 and cap_ok and cache.offset == T
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] incremental == prefill  |Δ|={d_inc:.2e}  capture={cap_ok} offset={cache.offset}")

    # === (#86) streamed bf16 reference forward (loader → forward) == resident prefill =========
    d = Path(tempfile.mkdtemp(prefix="glm_fwd_"))
    try:
        _build_synthetic_ckpt(d, cfg_keep, model_n)
        ck = GLMSourceCheckpoint(d)
        log_stream = glm_logits(ck, ids.reshape(1, -1), cfg_keep, dtype=mx.float32,
                                use_fast=False, use_indexer=True)
        rel = _maxabs(log_n, log_stream) / float(np.max(np.abs(np.array(log_n.astype(mx.float32)))))
        good = log_stream.shape == (1, T, cfg.vocab_size) and rel < 5e-3
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] streamed glm_logits (loader, 1-layer-resident) == resident prefill  rel={rel:.2e}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
