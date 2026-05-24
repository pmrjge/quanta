"""Parity: plain-mlx MiMoMoE == HF MiMoV2MoE (noaux_tc top-k sparse routing + SwiGLU experts).

Uses a small synthetic config (16 experts, top-8, ``n_group=topk_group=1`` as in MiMo-V2.5) with
random weights so the full routing + gather-dispatch + expert math is exercised without loading the
real 256 experts (those run end-to-end at #60). float32, so the diff isolates spec-correctness.

    uv run --with torch --with numpy python -m parity.mimo_moe_parity
"""

from __future__ import annotations

import dataclasses
import warnings

import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn

warnings.filterwarnings("ignore")
from transformers import AutoConfig  # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402

from quanta.mimo.config import MiMoV2Config  # noqa: E402
from quanta.mimo.moe import MiMoMoE  # noqa: E402

ART = "/Users/pmrj/models/MiMo-V2.5"
E, H, INTER, TOPK, T = 16, 128, 64, 8, 6


def _t(a: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(a.astype(mx.float32)))


def _rel(ref: torch.Tensor, got: mx.array) -> float:
    r = np.array(got.astype(mx.float32))
    b = ref.detach().cpu().float().numpy()
    return float(np.linalg.norm(r - b) / (np.linalg.norm(b) + 1e-30))


def run() -> None:
    overrides = dict(n_routed_experts=E, hidden_size=H, moe_intermediate_size=INTER,
                     num_experts_per_tok=TOPK, n_group=1, topk_group=1,
                     norm_topk_prob=True, routed_scaling_factor=1.0)
    cfg = dataclasses.replace(MiMoV2Config.from_pretrained(ART), **overrides)
    hf_cfg = AutoConfig.from_pretrained(ART, trust_remote_code=True)
    for k, v in {**overrides, "scoring_func": "sigmoid", "topk_method": "noaux_tc"}.items():
        setattr(hf_cfg, k, v)
    HFMoE = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2MoE", ART)

    mx.random.seed(0)
    gate_w = mx.random.normal((E, H)) * 0.05
    ebias = mx.random.normal((E,)) * 0.1
    gs = mx.random.normal((E, INTER, H)) * 0.05
    us = mx.random.normal((E, INTER, H)) * 0.05
    ds = mx.random.normal((E, H, INTER)) * 0.05
    x = mx.random.normal((1, T, H))

    mine = MiMoMoE(cfg)
    mine.gate_weight = gate_w
    mine.e_score_correction_bias = ebias
    mine.set_experts(gs, us, ds)
    got = mine(x)

    with torch.no_grad():
        hf = HFMoE(hf_cfg)
        hf.gate.weight = tnn.Parameter(_t(gate_w))
        hf.gate.e_score_correction_bias = tnn.Parameter(_t(ebias))
        for i in range(E):
            hf.experts[i].gate_proj.weight = tnn.Parameter(_t(gs[i]))
            hf.experts[i].up_proj.weight = tnn.Parameter(_t(us[i]))
            hf.experts[i].down_proj.weight = tnn.Parameter(_t(ds[i]))
        ref = hf.float().eval()(_t(x))

    rel = _rel(ref, got)
    good = rel < 2e-4
    print("=== MiMo MoE parity: plain-mlx vs HF MiMoV2MoE (f32) ===")
    print(f"  [{'OK' if good else 'FAIL'}] MoE E={E} topk={TOPK} n_group=1: rel={rel:.2e}")
    print("PASS" if good else "FAIL")


if __name__ == "__main__":
    run()
