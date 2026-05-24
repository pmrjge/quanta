"""Per-layer parity: plain-mlx MiMo decoder modules == HF ``modeling_mimo_v2`` (the text oracle).

The full HF model is not a usable oracle here (fp8 matmul needs CUDA; full bf16 dequant > RAM), so we
gate **one layer at a time** against HF's own module classes (loaded via the dynamic-module machinery)
fed our dequantized bf16 weights — memory-safe (rule-8) and it validates the subtle bits at once:
value_scale (0.707) ordering, partial-RoPE half-rotation over ``rope_dim`` dims, the SWA per-head
attention-sink concat-softmax-drop, and GQA. Everything runs in **float32** so the diff isolates
spec-correctness from bf16 noise. Short sequence (T<=sliding_window) ⇒ the SWA mask equals the full
causal mask, so this isolates the per-layer math from windowing (windowing tested separately).

    uv run --with torch --with numpy python -m parity.mimo_layer_parity
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn

warnings.filterwarnings("ignore")
from transformers import AutoConfig  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402

from quanta.mimo.config import MiMoV2Config  # noqa: E402
from quanta.mimo.loader import MiMoSourceCheckpoint  # noqa: E402
from quanta.mimo.model import MiMoAttention, MiMoDecoderLayer, MiMoDenseMLP, causal_mask  # noqa: E402

ART = "/Users/pmrj/models/MiMo-V2.5"
T = 8


def _t(a: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(a.astype(mx.float32)))


def _mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def _rel(ref: torch.Tensor, got: mx.array) -> float:
    r = np.array(got.astype(mx.float32))
    b = ref.detach().cpu().float().numpy()
    return float(np.linalg.norm(r - b) / (np.linalg.norm(b) + 1e-30))


def _set_linear(mod: MiMoAttention | MiMoDenseMLP, name: str, w: mx.array) -> None:
    getattr(mod, name).weight = w.astype(mx.float32)


def run() -> None:
    cfg = MiMoV2Config.from_pretrained(ART)
    ck = MiMoSourceCheckpoint(ART, cfg)
    hf_cfg = AutoConfig.from_pretrained(ART, trust_remote_code=True)
    hf_cfg._attn_implementation = "eager"
    HFAttn = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2Attention", ART)
    HFRot = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2RotaryEmbedding", ART)
    HFMLP = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2MLP", ART)

    mx.random.seed(0)
    ok = True

    def attn_case(layer_idx: int) -> None:
        nonlocal ok
        swa = cfg.is_swa(layer_idx)
        w = ck.attention_tensors(layer_idx)
        ck.release()
        x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)

        # --- mine ---
        a = MiMoAttention(cfg, layer_idx)
        for n, k in (("q_proj", "q_proj"), ("k_proj", "k_proj"), ("v_proj", "v_proj"), ("o_proj", "o_proj")):
            _set_linear(a, n, w[k])
        if a.has_sink:
            a.attention_sink_bias = w["attention_sink_bias"].astype(mx.float32)
        got = a(x, causal_mask(T), offset=0)

        # --- HF oracle ---
        with torch.no_grad():
            hf = HFAttn(hf_cfg, swa, layer_idx, projection_layout="fused_qkv")
            hf.qkv_proj.weight = tnn.Parameter(_t(mx.concatenate([w["q_proj"], w["k_proj"], w["v_proj"]], axis=0)))
            hf.o_proj.weight = tnn.Parameter(_t(w["o_proj"]))
            if hf.attention_sink_bias is not None:
                hf.attention_sink_bias = tnn.Parameter(_t(w["attention_sink_bias"]))
            hf = hf.float().eval()
            rot = HFRot(hf_cfg, swa).float()
            pos = torch.arange(T)[None]
            cos, sin = rot(_t(x), pos)
            m = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None]
            ref = hf(_t(x), position_embeddings=(cos, sin), attention_mask=m,
                     cache_position=torch.arange(T), position_ids=pos)[0]

        rel = _rel(ref, got)
        good = rel < 2e-4
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] L{layer_idx} attn ({'SWA+sink' if swa else 'full'}): "
              f"rel={rel:.2e} sink={a.has_sink}")

    print("=== MiMo per-layer parity: plain-mlx vs HF modeling_mimo_v2 (f32) ===")
    attn_case(0)   # full attention, no sink
    attn_case(1)   # SWA, with sink

    # dense MLP (L0)
    m = ck.dense_mlp_tensors(0)
    ck.release()
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    mlp = MiMoDenseMLP(cfg)
    for n in ("gate_proj", "up_proj", "down_proj"):
        _set_linear(mlp, n, m[n])
    got = mlp(x)
    with torch.no_grad():
        hfm = HFMLP(hf_cfg)
        hfm.gate_proj.weight = tnn.Parameter(_t(m["gate_proj"]))
        hfm.up_proj.weight = tnn.Parameter(_t(m["up_proj"]))
        hfm.down_proj.weight = tnn.Parameter(_t(m["down_proj"]))
        ref = hfm.float().eval()(_t(x))
    rel = _rel(ref, got)
    good = rel < 2e-4
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] L0 dense MLP: rel={rel:.2e}")

    # full decoder layer (L0: norm -> attn -> residual -> norm -> dense MLP -> residual)
    HFLayer = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2DecoderLayer", ART)
    nrm = ck.norm_tensors(0)
    aw = ck.attention_tensors(0)
    mw = ck.dense_mlp_tensors(0)
    ck.release()
    x = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.float32)
    mine = MiMoDecoderLayer(cfg, 0)
    mine.input_layernorm.weight = nrm["input_layernorm"].astype(mx.float32)
    mine.post_attention_layernorm.weight = nrm["post_attention_layernorm"].astype(mx.float32)
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _set_linear(mine.self_attn, n, aw[n])
    for n in ("gate_proj", "up_proj", "down_proj"):
        _set_linear(mine.mlp, n, mw[n])
    got = mine(x, causal_mask(T), offset=0)
    with torch.no_grad():
        layer = HFLayer(hf_cfg, layer_idx=0, attention_projection_layout="fused_qkv")
        layer.input_layernorm.weight = tnn.Parameter(_t(nrm["input_layernorm"]))
        layer.post_attention_layernorm.weight = tnn.Parameter(_t(nrm["post_attention_layernorm"]))
        layer.self_attn.qkv_proj.weight = tnn.Parameter(_t(mx.concatenate([aw["q_proj"], aw["k_proj"], aw["v_proj"]], axis=0)))
        layer.self_attn.o_proj.weight = tnn.Parameter(_t(aw["o_proj"]))
        layer.mlp.gate_proj.weight = tnn.Parameter(_t(mw["gate_proj"]))
        layer.mlp.up_proj.weight = tnn.Parameter(_t(mw["up_proj"]))
        layer.mlp.down_proj.weight = tnn.Parameter(_t(mw["down_proj"]))
        layer = layer.float().eval()
        rot = HFRot(hf_cfg, False).float()
        pos = torch.arange(T)[None]
        cos, sin = rot(_t(x), pos)
        m = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None]
        ref = layer(_t(x), attention_mask=m, position_ids=pos, position_embeddings=(cos, sin),
                    cache_position=torch.arange(T))
        ref = ref[0] if isinstance(ref, tuple) else ref
    rel = _rel(ref, got)
    good = rel < 2e-4
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] L0 full decoder layer: rel={rel:.2e}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
