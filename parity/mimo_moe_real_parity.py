"""Decisive check: REAL-weights MoE parity vs HF + checkpoint integrity (is the model broken?).

The synthetic MoE gate used 16 random experts; this loads L1's actual 256 fp8 experts + real router
into both ``MiMoMoE`` and HF ``MiMoV2MoE`` and diffs them (f32). If this PASSES, every component is
validated on real weights -> my streamed forward equals what HF would produce, so the e2e garbage
implicates the checkpoint/model, not my code. Also prints integrity stats (mean/std/NaN/zeros) for
the router + a few real expert stacks.

    uv run --with torch --with numpy python -m parity.mimo_moe_real_parity
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
from quanta.mimo.moe import MiMoMoE  # noqa: E402

ART = "/Users/pmrj/models/MiMo-V2.5"
LAYER, T = 1, 4


def _t(a: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(a.astype(mx.float32)))


def _stat(name: str, a: mx.array) -> None:
    f = a.astype(mx.float32)
    print(f"  {name:22s} shape={tuple(a.shape)} mean={float(mx.mean(f).item()):+.4f} "
          f"std={float(mx.std(f).item()):.4f} nan={int(mx.sum(~mx.isfinite(f)).item())} "
          f"zeros%={100 * float(mx.mean((f == 0).astype(mx.float32)).item()):.1f}")


def run() -> None:
    cfg = MiMoV2Config.from_pretrained(ART)
    ck = MiMoSourceCheckpoint(ART, cfg)
    r = ck.moe_router_tensors(LAYER)
    st = ck.expert_stacks(LAYER)  # all 256 experts, bf16
    ck.release()

    print(f"=== integrity: L{LAYER} router + experts ===")
    _stat("router.weight", r["weight"])
    _stat("e_score_corr_bias", r["e_score_correction_bias"])
    for proj in ("gate_proj", "up_proj", "down_proj"):
        _stat(f"experts.{proj}", st[proj])
        _stat(f"  expert0.{proj}", st[proj][0])
        _stat(f"  expert255.{proj}", st[proj][255])

    gf = {k: st[k].astype(mx.float32) for k in st}
    mx.random.seed(0)
    x = mx.random.normal((1, T, cfg.hidden_size))

    mine = MiMoMoE(cfg)
    mine.gate_weight = r["weight"].astype(mx.float32)
    mine.e_score_correction_bias = r["e_score_correction_bias"].astype(mx.float32)
    mine.set_experts(gf["gate_proj"], gf["up_proj"], gf["down_proj"])
    got = mine(x)

    print("=== real-weights MoE parity vs HF MiMoV2MoE (f32, 256 experts) ===", flush=True)
    hf_cfg = AutoConfig.from_pretrained(ART, trust_remote_code=True)
    HFMoE = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2MoE", ART)
    with torch.no_grad():
        hf = HFMoE(hf_cfg)
        hf.gate.weight = tnn.Parameter(_t(r["weight"]))
        hf.gate.e_score_correction_bias = tnn.Parameter(_t(r["e_score_correction_bias"]))
        for i in range(cfg.n_routed_experts):
            hf.experts[i].gate_proj.weight = tnn.Parameter(_t(st["gate_proj"][i]))
            hf.experts[i].up_proj.weight = tnn.Parameter(_t(st["up_proj"][i]))
            hf.experts[i].down_proj.weight = tnn.Parameter(_t(st["down_proj"][i]))
        ref = hf.float().eval()(_t(x))

    rr = np.array(got.astype(mx.float32))
    bb = ref.detach().cpu().float().numpy()
    rel = float(np.linalg.norm(rr - bb) / (np.linalg.norm(bb) + 1e-30))
    # routing agreement
    print(f"  rel err: {rel:.2e}  ->  {'OK (MoE correct on real weights)' if rel < 2e-4 else 'FAIL (my MoE bug)'}")
    print("PASS" if rel < 2e-4 else "FAIL")


if __name__ == "__main__":
    run()
