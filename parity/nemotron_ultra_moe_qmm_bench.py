"""Nemotron-Ultra — MoE gather_qmm batch-scaling microbench (does the routed FFN amortize at B>1?).

Stream B pinned the residual B=1 spec-verify cost to the **MoE `gather_qmm`** (40%, weight-bandwidth):
at B=1 each token reads its 22 routed experts' int4 weights to process ONE token — bandwidth-bound, low
arithmetic intensity, the part a Mamba-scan kernel can't touch. The serving lever is the OPPOSITE regime:
at B>1 the sorted `gather_qmm` reads each *touched* expert's weights ONCE for every token routed to it,
so as B grows and tokens share experts, the per-token weight bandwidth DROPS — the "fused gather_qmm"
amortization the other models (DSV4 `_swiglu_stack_packed`, qwen35) rely on. This microbench MEASURES
that amortization directly on ONE real Ultra MoE layer (rule 8 — one block resident), so we can see the
real B=32 gain (and whether sorted dispatch helps) before deciding any deeper fusion is warranted.

Per N = B·T routed tokens (N ∈ a sweep through the serving regime), times the real
`NemotronQuantizedMoE.__call__` (int4 experts via `mx.gather_qmm` + int8 dense fc1/fc2/shared), median
over reps, **sorted vs unsorted dispatch**, and reports per-token µs + the amortization vs N=1. Random
hidden ⇒ near-worst-case expert overlap (natural-language routing shares MORE, so real text amortizes
further — DSV4 #143). Not a parity gate (the routed-MoE is already gated bit-exact in
`parity/nemotron_ultra_qmoe_test.py`); a measurement.

One MoE layer resident — light, but **RUN SOLO** to keep the timing clean (~6 GiB packed int4 stack).

    uv run python -u -m parity.nemotron_ultra_moe_qmm_bench
"""

from __future__ import annotations

import mlx.core as mx

from parity.nemotron_mtp_k_bench import ART, _median_ms
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.runtime import build_resident_block

# B·T routed-token counts spanning the serving regime: B=1 (single-stream decode) → B=32 (the Stream-A
# throughput knee) → B=64; plus a verify-ish 22 (one stream, k≈22 tree) and 128 (headroom).
N_VALUES = (1, 8, 16, 22, 32, 64, 128)
WIRED_GIB = 40


def _bench(moe, n: int, hidden: int, *, sorted_dispatch: bool) -> float:
    moe.sort_dispatch = sorted_dispatch
    x = mx.random.normal([1, n, hidden]).astype(mx.bfloat16)
    mx.eval(moe(x))                                   # warm
    return _median_ms(lambda: mx.eval(moe(x)))


def run() -> None:
    mx.set_wired_limit(int(WIRED_GIB * 1024**3))
    mx.random.seed(0)
    cfg = NemotronHConfig.from_pretrained(ART)
    moe_idx = cfg.layers_block_type.index("moe")
    art = NemotronArtifact(ART)
    blk = build_resident_block(art, cfg, moe_idx)
    art.release()
    mx.clear_cache()
    moe = blk.mixer
    hidden = cfg.hidden_size

    print("\n=== Nemotron-Ultra MoE gather_qmm batch-scaling microbench (one real layer) ===")
    print(f"backbone: {ART}")
    print(f"moe layer {moe_idx} | {cfg.n_routed_experts} experts top-{cfg.num_experts_per_tok} | "
          f"latent {cfg.moe_latent_size} inter {cfg.moe_intermediate_size} | hidden {hidden}\n")
    print(f"  {'N=B·T':>6s}  {'sorted ms':>10s}  {'µs/tok':>8s}  {'unsort ms':>10s}  {'µs/tok':>8s}  "
          f"{'sort×':>6s}  {'tok-amort':>9s}")

    base_sorted = None
    for n in N_VALUES:
        t_s = _bench(moe, n, hidden, sorted_dispatch=True)
        t_u = _bench(moe, n, hidden, sorted_dispatch=False)
        us_s, us_u = t_s / n * 1e3, t_u / n * 1e3
        if base_sorted is None:
            base_sorted = us_s
        sort_speedup = t_u / max(t_s, 1e-9)
        amort = base_sorted / us_s            # per-token speedup vs N=1 (the bandwidth amortization)
        print(f"  {n:>6d}  {t_s:>10.3f}  {us_s:>8.1f}  {t_u:>10.3f}  {us_u:>8.1f}  {sort_speedup:>5.2f}x  "
              f"{amort:>8.2f}x")

    print()
    moe.sort_dispatch = True
    us1 = base_sorted
    t32 = _bench(moe, 32, hidden, sorted_dispatch=True)
    us32 = t32 / 32 * 1e3
    print(f"HEADLINE: per-token MoE cost {us1:.1f} µs @ B=1 → {us32:.1f} µs @ B=32 "
          f"({us1 / us32:.2f}× cheaper/token) — the gather_qmm weight-bandwidth amortization that the "
          f"B=1 spec verify (Stream B) could NOT exploit. Real-text routing shares more experts than this "
          f"random-hidden lower bound, so serving amortizes further.")


if __name__ == "__main__":
    run()
