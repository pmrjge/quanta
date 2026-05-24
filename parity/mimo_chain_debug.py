"""Localize the e2e bug: mine-vs-HF decoder forward, layer-by-layer, in lockstep (f32, L>window).

f32 and bf16 full forwards are both garbage (ppl ~90k) although every component matches HF in
isolation — so a structural/chain bug appears only in the full stack. This feeds the SAME input to
my decoder layer and HF's, compares outputs, then advances using HF's output as ground truth, so the
first layer that diverges (given correct input) is the culprit. Real weights, real masks/rope,
L=202 so sliding-window attention is exercised.

    uv run --with torch --with numpy --with tokenizers python -m parity.mimo_chain_debug [N]
"""

from __future__ import annotations

import sys
import warnings

import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn
from tokenizers import Tokenizer

warnings.filterwarnings("ignore")
from transformers import AutoConfig  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask  # noqa: E402

from quanta.mimo.config import MiMoV2Config  # noqa: E402
from quanta.mimo.loader import MiMoSourceCheckpoint  # noqa: E402
from quanta.mimo.reference import _build_layer, full_causal_mask, sliding_window_mask  # noqa: E402

ART = "/Users/pmrj/models/MiMo-V2.5"
PROSE = ("The history of writing traces the development of expressing language by systems of "
         "markings. In the history of how writing systems have evolved, more complete writing "
         "systems were preceded by proto-writing, systems of ideographic or early mnemonic symbols. "
         "True writing, in which the content of a linguistic utterance is encoded so that another "
         "reader can reconstruct the exact utterance written down, is a later development, "
         "distinguished from proto-writing which avoids encoding grammatical words and affixes. One "
         "of the earliest forms of written expression is cuneiform, which emerged in the ancient "
         "Near East and was used for thousands of years before being gradually replaced by "
         "alphabetic scripts. The invention of writing transformed human societies, enabling laws, "
         "the administration of complex states, the preservation of literature, and the "
         "accumulation of knowledge across many generations of people living in distant places.")


def _t(a: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(a.astype(mx.float32)))


def _mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def _rel(a: mx.array, b: torch.Tensor) -> float:
    r = np.array(a.astype(mx.float32))
    bb = b.detach().cpu().float().numpy()
    return float(np.linalg.norm(r - bb) / (np.linalg.norm(bb) + 1e-30))


def _build_hf_layer(HFLayer, hf_cfg, ck, cfg, i):
    layer = HFLayer(hf_cfg, layer_idx=i, attention_projection_layout="fused_qkv")
    nrm = ck.norm_tensors(i)
    aw = ck.attention_tensors(i)
    layer.input_layernorm.weight = tnn.Parameter(_t(nrm["input_layernorm"]))
    layer.post_attention_layernorm.weight = tnn.Parameter(_t(nrm["post_attention_layernorm"]))
    layer.self_attn.qkv_proj.weight = tnn.Parameter(_t(mx.concatenate([aw["q_proj"], aw["k_proj"], aw["v_proj"]], 0)))
    layer.self_attn.o_proj.weight = tnn.Parameter(_t(aw["o_proj"]))
    if layer.self_attn.attention_sink_bias is not None:
        layer.self_attn.attention_sink_bias = tnn.Parameter(_t(aw["attention_sink_bias"]))
    if cfg.is_moe(i):
        r = ck.moe_router_tensors(i)
        st = ck.expert_stacks(i)
        layer.mlp.gate.weight = tnn.Parameter(_t(r["weight"]))
        layer.mlp.gate.e_score_correction_bias = tnn.Parameter(_t(r["e_score_correction_bias"]))
        for e in range(cfg.n_routed_experts):
            layer.mlp.experts[e].gate_proj.weight = tnn.Parameter(_t(st["gate_proj"][e]))
            layer.mlp.experts[e].up_proj.weight = tnn.Parameter(_t(st["up_proj"][e]))
            layer.mlp.experts[e].down_proj.weight = tnn.Parameter(_t(st["down_proj"][e]))
    else:
        m = ck.dense_mlp_tensors(i)
        layer.mlp.gate_proj.weight = tnn.Parameter(_t(m["gate_proj"]))
        layer.mlp.up_proj.weight = tnn.Parameter(_t(m["up_proj"]))
        layer.mlp.down_proj.weight = tnn.Parameter(_t(m["down_proj"]))
    return layer.float().eval()


def run() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cfg = MiMoV2Config.from_pretrained(ART)
    ck = MiMoSourceCheckpoint(ART, cfg)
    tok = Tokenizer.from_file(f"{ART}/tokenizer.json")
    ids = [cfg.bos_token_id] + tok.encode(PROSE).ids
    L = len(ids)
    hf_cfg = AutoConfig.from_pretrained(ART, trust_remote_code=True)
    hf_cfg._attn_implementation = "eager"
    HFLayer = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2DecoderLayer", ART)
    HFRot = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2RotaryEmbedding", ART)

    embed = ck.read("model.embed_tokens.weight").astype(mx.float32)
    h = embed[mx.array(ids)][None]
    mx.eval(h)
    ck.release()
    fm = full_causal_mask(L, mx.float32)
    sm = sliding_window_mask(L, cfg.sliding_window, mx.float32)

    pid = torch.arange(L)[None]
    emb_t = torch.zeros(1, L, 8)
    full_hf = create_causal_mask(hf_cfg, emb_t, None, None, pid)
    swa_hf = create_sliding_window_causal_mask(hf_cfg, emb_t, None, None, pid)
    rot_full = HFRot(hf_cfg, False).float()
    rot_swa = HFRot(hf_cfg, True).float()

    def _rms(a: mx.array) -> float:
        f = a.astype(mx.float32)
        return float(mx.sqrt(mx.mean(f * f)).item())

    h_hf = h
    h_mine = mx.array(h)  # my accumulated trajectory (feeds my own outputs forward)
    print(f"=== chain diff (L={L}, f32): per-layer (on HF input) vs accumulated (on my input) ===", flush=True)
    for i in range(n):
        swa = cfg.is_swa(i)
        ht = _t(h_hf)
        mine = _build_layer(cfg, ck, i, mx.float32)
        my_on_hf = mine(h_hf, sm if swa else fm, offset=0)        # per-layer correctness
        my_on_mine = mine(h_mine, sm if swa else fm, offset=0)    # accumulated trajectory
        del mine
        with torch.no_grad():
            hf = _build_hf_layer(HFLayer, hf_cfg, ck, cfg, i)
            cos, sin = (rot_swa if swa else rot_full)(ht, pid)
            ref = hf(ht, attention_mask=(swa_hf if swa else full_hf), position_embeddings=(cos, sin),
                     position_ids=pid, cache_position=torch.arange(L))
            ref = ref[0] if isinstance(ref, tuple) else ref
            del hf
        kind = ("swa" if swa else "full") + ("+moe" if cfg.is_moe(i) else "+dense")
        print(f"  L{i:2d} {kind:9s}: per-layer={_rel(my_on_hf, ref):.2e}  accum={_rel(my_on_mine, ref):.2e}  "
              f"rms(mine)={_rms(my_on_mine):.2f} rms(hf)={_rms(_mx(ref)):.2f}", flush=True)
        h_hf = _mx(ref)        # HF ground-truth trajectory
        h_mine = my_on_mine    # my accumulated trajectory
        ck.release()


if __name__ == "__main__":
    run()
