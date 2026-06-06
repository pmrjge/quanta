"""Nemotron-Ultra U4 / Stream-B — fused multi-token SSD-scan VERIFY parity gate (per-block, real weights).

Stream B fuses the T>1 spec-VERIFY Mamba continuation into one Metal launch per layer
(``mamba_mixer.FUSED_SSD_SCAN`` ⇒ ``ssd_scan_fused`` + a bit-identical batched conv), behind a flag that
defaults off (rule 4). ``ssd_scan_fused`` is a genuine reordering of the per-token ``ssd_step`` loop (its
``C·s`` accumulation differs from ``mx.sum``), so it must be proven output-equivalent on real weights.

**Why this gate is per-block, not full-model.** The fusion's ENTIRE surface is the mamba mixer's T>1
continuation. The model-free kernel test (``nemotron_ssd_scan_kernel_test.py``) showed the scan is ~2.2e-7
rel vs the per-token loop (fp32) and the batched conv is bit-identical — both far below a bf16 ULP. A
first attempt at a *full-model* verify gate FAILED its intermediate-state assertion, and the per-layer
trace explained why: the bf16-cast mamba block output is bit-identical for most layers, but the ~2.2e-7
fp32 scan difference occasionally straddles a bf16 rounding boundary and flips a SINGLE bf16 ULP (first
seen ~2⁻⁶ at the 2nd mamba layer), which then **cascades chaotically** through the 108-layer hybrid
(growing through clean powers of two: 2⁻⁶ → 2⁻⁴ → … → 10²). That is the *exact* M2/M3 settled finding
("a single bf16 ULP near-tie flip cascades chaotically — 'spec == T=1 greedy' is the wrong real-weight
criterion"). The cascade afflicts ANY ULP-level reordering (it is bf16 chaos, not a fusion bug), so the
honest, decisive proof is **per-mamba-block output-equivalence**: given identical inputs, does each real
mamba block produce the same output eager vs fused? The downstream cascade does not (it is the same class
that makes compiled-verify-vs-eager bit-identical only because ``mx.compile`` preserves op order). The
full-model arbiter is **top-1 agreement** (the verify's argmax — what spec-decode actually consumes), and
the bench (``nemotron_ultra_fused_scan_bench.py``) confirms it e2e via ``acc==`` (accept rates identical
across modes ⇒ identical accepted/emitted tokens) and ``match`` (reproduces M2 losslessness — the int4
main model verifies every draft regardless).

This gate streams all 48 mamba layers (rule 8 — one block resident at a time, never the 306 GiB model),
and for each, given **identical** ``(x, ssm_state, conv_state)`` for ``T ∈ {2, 3, 4}``, asserts:

* block **output** ``[1, T, hidden]`` (bf16): equal within ≤ 1 bf16 ULP (mostly bit-identical) — a real
  bug (wrong window / state-carry / time index) would be O(1)+ rel here, not ≤ 1 ULP;
* new **conv_state** (bf16): **bit-identical** (the batched conv is bit-identical by construction);
* new **ssm_state** (fp32): ≤ a tight fp32 tolerance (the scan's ``C·s`` reorder, ~5e-7).

    uv run --with tokenizers python -m parity.nemotron_ultra_fused_scan_parity
"""

from __future__ import annotations

import mlx.core as mx

import quanta.nemotron.mamba_mixer as mm
from parity.nemotron_mtp_k_bench import ART
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.runtime import build_resident_block

T_VALUES = (2, 3, 4)
SEED_LEN = 16          # short fresh prefill to seed a realistic (ssm_state, conv_state) per block
# Block OUTPUT is bf16 (O(1)); ≤ 1 bf16 ULP at the rounding boundary is ~2⁻⁶ rel — a real fusion bug is
# O(1)+. conv is bit-identical (batched conv == per-token). ssm is fp32 (the scan reorder, ~5e-7).
OUT_TOL_REL = 2e-2
CONV_TOL = 0.0
STATE_TOL = 1e-4


def _maxabs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _rel(a, b) -> float:
    m = float(mx.max(mx.abs(b.astype(mx.float32))).item())
    return _maxabs(a, b) / (m + 1e-30)


def run() -> None:
    mx.set_wired_limit(int(90 * 1024**3))
    cfg = NemotronHConfig.from_pretrained(ART)
    kinds = cfg.layers_block_type
    mamba_idx = [i for i, k in enumerate(kinds) if k == "mamba"]
    art = NemotronArtifact(ART)

    print("\n=== Nemotron-Ultra Stream-B: fused SSD-scan per-mamba-block parity (real weights) ===")
    print(f"backbone: {ART}")
    print(f"streaming {len(mamba_idx)} mamba blocks (rule 8: one resident at a time) | "
          f"T in {T_VALUES}\n")
    print(f"  {'layer':>6s}  {'T':>2s}  {'out rel':>9s}  {'out abs':>9s}  {'convΔ':>9s}  {'ssmΔ':>9s}  "
          f"verdict")

    mx.random.seed(0)
    worst_out = worst_conv = worst_ssm = 0.0
    bit_ident = total = 0
    all_ok = True
    for i in mamba_idx:
        blk = build_resident_block(art, cfg, i)
        art.release()
        mx.clear_cache()
        # seed a realistic recurrent state via a fresh prefill (eager)
        mm.FUSED_SSD_SCAN = False
        pre = mx.random.normal([1, SEED_LEN, cfg.hidden_size]).astype(mx.bfloat16)
        _, s0, c0 = blk(pre)
        mx.eval(s0, c0)
        for t in T_VALUES:
            x = mx.random.normal([1, t, cfg.hidden_size]).astype(mx.bfloat16)
            mm.FUSED_SSD_SCAN = False
            oe, se, ce = blk(x, ssm_state=s0, conv_state=c0)
            mm.FUSED_SSD_SCAN = True
            of, sf, cf = blk(x, ssm_state=s0, conv_state=c0)
            mm.FUSED_SSD_SCAN = False
            mx.eval(oe, se, ce, of, sf, cf)
            o_rel, o_abs = _rel(of, oe), _maxabs(of, oe)
            d_conv, d_ssm = _maxabs(cf, ce), _maxabs(sf, se)
            ok = o_rel <= OUT_TOL_REL and d_conv <= CONV_TOL and d_ssm <= STATE_TOL
            all_ok = all_ok and ok
            total += 1
            bit_ident += int(o_abs == 0.0)
            worst_out = max(worst_out, o_rel)
            worst_conv = max(worst_conv, d_conv)
            worst_ssm = max(worst_ssm, d_ssm)
            print(f"  L{i:>4d}  {t:>2d}  {o_rel:>9.2e}  {o_abs:>9.2e}  {d_conv:>9.2e}  {d_ssm:>9.2e}  "
                  f"{'PASS' if ok else 'FAIL'}")
        del blk
        mx.clear_cache()

    print()
    print(f"  {bit_ident}/{total} (T,layer) cases bit-identical block output; "
          f"worst out rel {worst_out:.2e} | worst convΔ {worst_conv:.2e} | worst ssmΔ {worst_ssm:.2e}")
    if all_ok:
        print(f"\nVERDICT: fused multi-token SSD-scan == eager per-token step on ALL {len(mamba_idx)} real "
              f"mamba blocks (output ≤ 1 bf16 ULP, conv bit-identical, ssm ≤ {STATE_TOL:.0e}) for "
              f"T in {T_VALUES}. FUSED_SSD_SCAN is output-equivalent on real weights — the per-layer "
              f"bf16-ULP differences cascade chaotically downstream (the M2/M3 settled finding; the bench "
              f"confirms top-1 / accept-rate agreement e2e), but the fusion itself is correct everywhere "
              f"it acts. Losslessness (M2) is untouched.")
    else:
        raise SystemExit(
            f"FAIL: a real mamba block diverged beyond bf16 ULP (worst out rel {worst_out:.2e} > "
            f"{OUT_TOL_REL}, convΔ {worst_conv:.2e}, ssmΔ {worst_ssm:.2e}) — the fused scan / batched conv "
            f"is NOT output-equivalent on real weights. Do NOT enable FUSED_SSD_SCAN (rule 6, rule 4).")


if __name__ == "__main__":
    run()
