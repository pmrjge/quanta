"""Smoke + shape gate for the DeepSeek-V4 streamed loader (mmap raw fp8/fp4 reader).

Loads representatives of all three attention regimes (pure-SW L0, indexer L2, compressor L3), both
router kinds (hash L0 -> tid2eid, scored L3 -> bias), the shared expert, a few routed fp4 experts,
HC params, and the MTP extras — asserting every shape against the config geometry and that all
dequantized values are finite. Deliberately loads only a handful of experts so it stays light while
other jobs run.

    uv run --with numpy python -m parity.dsv4_loader_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def _finite(a: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(a.astype(mx.float32))).item())


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    H, hd, nh = cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads
    ok = True

    def check(tag, cond, extra=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'OK' if cond else 'FAIL'}] {tag}{(' ' + extra) if extra else ''}")

    # --- attention regimes -----------------------------------------------------
    print("=== attention (3 regimes) ===")
    for i in (0, 2, 3):
        a = ck.attention(i)
        shapes_ok = (
            a["wq_a"].shape == (cfg.q_lora_rank, H)
            and a["wq_b"].shape == (nh * hd, cfg.q_lora_rank)
            and a["wkv"].shape == (hd, H)
            and a["wo_a"].shape == (cfg.o_groups * cfg.o_lora_rank, nh * hd // cfg.o_groups)
            and a["wo_b"].shape == (H, cfg.o_groups * cfg.o_lora_rank)
            and a["attn_sink"].shape == (nh,)
        )
        fin = all(_finite(a[k]) for k in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"))
        has_comp = "compressor" in a
        has_idx = "indexer" in a
        regime_ok = has_comp == cfg.has_compressor(i) and has_idx == cfg.has_indexer(i)
        check(f"L{i} shapes+finite", shapes_ok and fin)
        check(f"L{i} regime comp={has_comp} idx={has_idx}", regime_ok)
        if has_comp:
            c = a["compressor"]
            coff = cfg.compressor_coff(i)
            cok = c["wkv"].shape == (coff * hd, H) and c["norm"].shape == (hd,) \
                and c["ape"].shape == (cfg.compress_ratio(i), coff * hd)
            check(f"L{i} compressor shapes", cok,
                  f"wkv{tuple(c['wkv'].shape)} ape{tuple(c['ape'].shape)}")
        if has_idx:
            x = a["indexer"]
            ihd, inh = cfg.index_head_dim, cfg.index_n_heads
            xok = x["wq_b"].shape == (inh * ihd, cfg.q_lora_rank) \
                and x["weights_proj"].shape == (inh, H) \
                and x["compressor"]["wkv"].shape == (2 * ihd, H)
            check(f"L{i} indexer shapes", xok,
                  f"wq_b{tuple(x['wq_b'].shape)} wproj{tuple(x['weights_proj'].shape)}")
        ck.release()

    # --- routers (hash vs scored) ---------------------------------------------
    print("=== MoE router (hash vs scored) ===")
    r0 = ck.moe_router(0)
    check("L0 hash: gate.weight + tid2eid",
          r0["weight"].shape == (cfg.n_routed_experts, H)
          and "tid2eid" in r0 and r0["tid2eid"].shape == (cfg.vocab_size, cfg.num_experts_per_tok)
          and "bias" not in r0,
          f"tid2eid{tuple(r0['tid2eid'].shape)}")
    r3 = ck.moe_router(3)
    check("L3 scored: gate.weight + bias",
          r3["weight"].shape == (cfg.n_routed_experts, H)
          and "bias" in r3 and r3["bias"].shape == (cfg.n_routed_experts,)
          and "tid2eid" not in r3)
    ck.release()

    # --- experts (fp4) + shared (fp8) -----------------------------------------
    print("=== experts (fp4) + shared (fp8) ===")
    mi = cfg.moe_intermediate_size
    se = ck.shared_expert(2)
    check("L2 shared expert shapes+finite",
          se["w1"].shape == (mi, H) and se["w2"].shape == (H, mi) and se["w3"].shape == (mi, H)
          and all(_finite(se[k]) for k in se))
    st = ck.expert_stacks(2, n_experts=4)
    check("L2 routed expert stacks [E,out,in]+finite",
          st["w1"].shape == (4, mi, H) and st["w2"].shape == (4, H, mi) and st["w3"].shape == (4, mi, H)
          and all(_finite(st[k]) for k in st),
          f"w1{tuple(st['w1'].shape)} w2{tuple(st['w2'].shape)}")
    ck.release()

    # --- HC / norms / top-level / MTP -----------------------------------------
    print("=== HC params, norms, top-level, MTP ===")
    hc = ck.block_hc(2)
    check("L2 hc_attn/ffn fn[mix_hc,hc*dim] base[mix_hc] scale[3]",
          hc["hc_attn_fn"].shape == (cfg.mix_hc, cfg.hc_mult * H)
          and hc["hc_attn_base"].shape == (cfg.mix_hc,) and hc["hc_attn_scale"].shape == (3,)
          and hc["hc_ffn_fn"].shape == (cfg.mix_hc, cfg.hc_mult * H))
    nm = ck.block_norms(2)
    check("L2 norms", nm["attn_norm"].shape == (H,) and nm["ffn_norm"].shape == (H,))
    fhc = ck.final_hc()
    check("top-level embed/head/norm/hc_head",
          ck.embed().shape == (cfg.vocab_size, H) and ck.head().shape == (cfg.vocab_size, H)
          and ck.final_norm().shape == (H,)
          and fhc["fn"].shape == (cfg.hc_mult, cfg.hc_mult * H) and fhc["scale"].shape == (1,))
    m = ck.mtp(0)
    check("MTP extras (e_proj/h_proj/enorm/hnorm/norm/hc_head)",
          m["e_proj"].shape == (H, H) and m["h_proj"].shape == (H, H)
          and m["enorm"].shape == (H,) and m["norm"].shape == (H,)
          and m["hc_head_fn"].shape == (cfg.hc_mult, cfg.hc_mult * H))
    ck.release()

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
