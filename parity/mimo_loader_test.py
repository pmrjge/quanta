"""Smoke test: MiMo-V2.5 streamed source loader (fp8→bf16, fused-qkv split). Pure MLX, no torch.

Checks that per-layer text tensors load and dequantize to finite bf16 with the right shapes, and
that the fused-qkv split lands at the exact per-layer-type offsets (full vs SWA). Bounded to a few
experts so it stays memory-light.

    uv run python -m parity.mimo_loader_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.loader import MiMoSourceCheckpoint

ART = "/Users/pmrj/models/MiMo-V2.5"


def _finite(x: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(x.astype(mx.float32))).item())


def run() -> None:
    cfg = MiMoV2Config.from_pretrained(ART)
    ck = MiMoSourceCheckpoint(ART, cfg)
    ok = True

    for li in (0, 1):  # 0 = full-attn + dense MLP, 1 = SWA + MoE
        swa = cfg.is_swa(li)
        q, k, v = cfg.qkv_sizes(swa)
        a = ck.attention_tensors(li)
        shp = {n: tuple(a[n].shape) for n in a}
        split_ok = (shp["q_proj"] == (q, cfg.hidden_size) and shp["k_proj"] == (k, cfg.hidden_size)
                    and shp["v_proj"] == (v, cfg.hidden_size)
                    and shp["o_proj"] == (cfg.hidden_size, cfg.o_in_features(swa)))
        sink_ok = (("attention_sink_bias" in a) == cfg.has_attn_sink(swa))
        fin = all(_finite(t) for t in a.values())
        norms = ck.norm_tensors(li)
        norm_ok = all(tuple(norms[n].shape) == (cfg.hidden_size,) for n in norms)
        ok = ok and split_ok and sink_ok and fin and norm_ok
        print(f"[{'OK' if split_ok and sink_ok and fin and norm_ok else 'FAIL'}] "
              f"L{li} {'SWA ' if swa else 'full'} attn: q/k/v/o={shp['q_proj']}/{shp['k_proj']}/"
              f"{shp['v_proj']}/{shp['o_proj']} sink={'attention_sink_bias' in a} finite={fin}")
        ck.release()

    # dense MLP (L0)
    m = ck.dense_mlp_tensors(0)
    dense_ok = (tuple(m["gate_proj"].shape) == (cfg.intermediate_size, cfg.hidden_size)
                and tuple(m["down_proj"].shape) == (cfg.hidden_size, cfg.intermediate_size)
                and all(_finite(t) for t in m.values()))
    ok = ok and dense_ok
    print(f"[{'OK' if dense_ok else 'FAIL'}] L0 dense MLP: gate={tuple(m['gate_proj'].shape)} "
          f"down={tuple(m['down_proj'].shape)} finite={all(_finite(t) for t in m.values())}")
    ck.release()

    # MoE router + a few experts (L1)
    r = ck.moe_router_tensors(1)
    router_ok = (tuple(r["weight"].shape) == (cfg.n_routed_experts, cfg.hidden_size)
                 and "e_score_correction_bias" in r
                 and tuple(r["e_score_correction_bias"].shape) == (cfg.n_routed_experts,))
    ne = 4
    st = ck.expert_stacks(1, n_experts=ne)
    mi, h = cfg.moe_intermediate_size, cfg.hidden_size
    expert_ok = (tuple(st["gate_proj"].shape) == (ne, mi, h) and tuple(st["up_proj"].shape) == (ne, mi, h)
                 and tuple(st["down_proj"].shape) == (ne, h, mi) and all(_finite(t) for t in st.values()))
    ok = ok and router_ok and expert_ok
    print(f"[{'OK' if router_ok else 'FAIL'}] L1 router: gate={tuple(r['weight'].shape)} "
          f"bias={'e_score_correction_bias' in r}")
    print(f"[{'OK' if expert_ok else 'FAIL'}] L1 experts[:{ne}]: gate={tuple(st['gate_proj'].shape)} "
          f"down={tuple(st['down_proj'].shape)} finite={all(_finite(t) for t in st.values())}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
