"""Nemotron-Ultra U4 / MTP-M1 — baked MTP draft-head sidecar recon vs the bf16 source head.

The second milestone of the native-MTP self-speculative decode stream (#40): after MTP-M0 gated the
bf16 head's *structural assembly*, this gates that the **baked int4-RTN sidecar**
(`parity/run_bake_nemotron_ultra_mtp_int4rtn_g64.py` → `...-quanta_int4rtn_g64_mtp`) dequantizes
**faithfully** — the same head, just int4-RTN experts + int8 dense + bf16 core.

Three gates, strongest first:

1. **Coverage + format (rule 6).** Every source ``mtp.*`` tensor (1040) is present in the sidecar at
   the format its policy dictates — ``dense`` bf16 (norms/router/bias), ``affine_packed`` int8
   (``eh_proj``/qkvo/fc1/fc2/shared), ``awq_packed`` int4 (the 512 routed experts) — checked against
   :func:`quant_policy.classify`. No orphan, no wrong-bits silent fallback.
2. **Bit-exact faithfulness.** For ``eh_proj`` (int8) and a sample of experts (int4 RTN), an
   *independent* in-script ``quantize_affine`` of the source weight reproduces the baked
   packed/scale/bias **bit-for-bit** (and the RTN ``awq_scale`` is exactly ones, i.e. s=1). Proves
   the bake wrote the policy's quantization, not something merely close.
3. **Recon forward.** The baked-dequant head and the bf16-source head run through the *identical*
   :class:`NemotronMTPModule` forward (M0 proved that forward == an independent reference), so the
   only difference is the quantization — an honest recon. The bf16 router is unchanged on both sides,
   so the top-22 routing is identical and the delta is purely int4-expert + int8-dense quant error.

What it does NOT gate: the head's *functional* accept rate — that's MTP-M2 (real spec decode). M1 is
parity-first: the sidecar must dequantize faithfully BEFORE it's wired into the resident spec loop.
Layer-streamed (rule 8): the two 21.5 GiB expert stacks load **sequentially** (build ref, eval, free,
then build baked), so the peak is one head. Run solo.

    uv run python -m parity.nemotron_ultra_mtp_bake_parity
"""

from __future__ import annotations

import mlx.core as mx

from quanta.bake.quant import quantize_affine
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.mtp import NemotronMTPModule
from quanta.nemotron.quant_policy import classify
from parity.nemotron_ultra_mtp_parity import HEADV, T, _fill_module, _mtp_tensors, _rel

SRC = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
MTP_ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64_mtp"


def _head_forward(t: dict[str, mx.array], cfg: NemotronHConfig, prev_hidden: mx.array,
                  token_emb: mx.array, head: mx.array) -> tuple[mx.array, mx.array]:
    """Run a filled :class:`NemotronMTPModule` forward (the path M0 gated == inline reference).

    Used on BOTH sides — bf16-source ``t`` and baked-dequant ``t`` — so the only difference between
    the two outputs is the quantization. Returns ``(logits, new_hidden)``; the module + its 21.5 GiB
    expert stack are local and freed when this returns (sequential rule-8 residency)."""
    mtp = NemotronMTPModule(cfg)
    _fill_module(mtp, t)
    return mtp(prev_hidden, token_emb, head, use_fast=False, return_hidden=True)


def _coverage_format(art: NemotronArtifact, source_mtp: list[str]) -> tuple[bool, list[str], dict]:
    """Every source ``mtp.*`` tensor present at its policy-dictated format/bits (rule 6)."""
    bad: list[str] = []
    counts = {"bf16": 0, "int8_affine": 0, "int4_gptq": 0}
    for key in source_mtp:
        sch = classify(key)
        counts[sch.kind] += 1
        if sch.kind == "bf16":
            ok = art.manifest.get(key, {}).get("format") == "dense"
        else:
            base = key[: -len(".weight")]
            m = art.manifest.get(base, {})
            want_fmt, want_bits = ("affine_packed", 8) if sch.kind == "int8_affine" else ("awq_packed", 4)
            ok = m.get("format") == want_fmt and m.get("bits") == want_bits
        if not ok:
            bad.append(key)
    return (not bad), bad, counts


def _exact_match(ck: NemotronSourceCheckpoint, art: NemotronArtifact, n: int) -> bool:
    """Independent RTN quant of the source == baked codes, bit-for-bit (eh_proj int8 + sample experts)."""
    def eq(a: mx.array, b: mx.array) -> bool:
        return bool(mx.array_equal(a, b))

    # int8 eh_proj (dense always-on)
    ek, gsd = "mtp.layers.0.eh_proj", int(art.manifest["mtp.layers.0.eh_proj"]["group_size"])
    p, s, b = quantize_affine(ck.read(ek + ".weight"), 8, gsd, scale_dtype=mx.bfloat16)
    ok = (eq(p, art.raw(ek + ".weight_packed")) and eq(s, art.raw(ek + ".weight_scale"))
          and eq(b, art.raw(ek + ".weight_bias")))
    ck.release()

    # int4 RTN experts (sample): codes + s=1 ones (awq_scale)
    gse = int(art.manifest["mtp.layers.1.mixer.experts.0.up_proj"]["group_size"])
    for e in (0, n // 2, n - 1):
        for proj in ("up_proj", "down_proj"):
            base = f"mtp.layers.1.mixer.experts.{e}.{proj}"
            w = ck.read(base + ".weight")
            p, s, b = quantize_affine(w, 4, gse, scale_dtype=mx.bfloat16)
            ones = mx.ones((w.shape[1],), dtype=mx.bfloat16)
            ok = ok and eq(p, art.raw(base + ".weight_packed")) and eq(s, art.raw(base + ".weight_scale"))
            ok = ok and eq(b, art.raw(base + ".weight_bias")) and eq(ones, art.raw(base + ".awq_scale"))
            ck.release()
    return bool(ok)


def run() -> None:
    mx.random.seed(0)
    cfg = NemotronHConfig.from_pretrained(SRC)
    n = cfg.n_routed_experts
    art = NemotronArtifact(MTP_ART)
    ck = NemotronSourceCheckpoint(SRC)
    source_mtp = sorted(k for k in ck.weight_map if k.startswith("mtp."))

    cover_ok, bad, counts = _coverage_format(art, source_mtp)
    exact_ok = _exact_match(ck, art, n)

    # Recon forward — identical module, bf16-source weights vs baked-dequant weights (sequential).
    prev_hidden = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.bfloat16)
    token_emb = mx.random.normal((1, T, cfg.hidden_size)).astype(mx.bfloat16)
    head = mx.random.normal((HEADV, cfg.hidden_size)).astype(mx.bfloat16)

    r_logits, r_hidden = _head_forward(_mtp_tensors(ck, n)[0], cfg, prev_hidden, token_emb, head)
    mx.eval(r_logits, r_hidden)
    mx.clear_cache()
    b_logits, b_hidden = _head_forward(_mtp_tensors(art, n)[0], cfg, prev_hidden, token_emb, head)
    mx.eval(b_logits, b_hidden)

    d_logits, d_hidden = _rel(b_logits, r_logits), _rel(b_hidden, r_hidden)
    agree = float(mx.mean((mx.argmax(b_logits, -1) == mx.argmax(r_logits, -1)).astype(mx.float32)))
    recon_ok = d_logits < 0.10 and d_hidden < 0.10
    ok = cover_ok and exact_ok and recon_ok

    print("\n=== Nemotron-Ultra MTP-M1 (baked int4-RTN sidecar recon vs bf16 source head) ===")
    print(f"sidecar: {MTP_ART}")
    print(f"mtp tensors covered : {len(source_mtp)}/1040  "
          f"(bf16 {counts['bf16']} / int8 {counts['int8_affine']} / int4 {counts['int4_gptq']})  "
          f"format+coverage {'PASS' if cover_ok else f'FAIL {bad[:5]}'}")
    print(f"bit-exact faithfulness (indep RTN quant == baked codes; eh_proj + experts 0/{n // 2}/{n - 1}) "
          f": {'PASS' if exact_ok else 'FAIL'}")
    print(f"recon forward (baked-dequant vs bf16 head, T={T}; routing identical, bf16 router):")
    print(f"  logits      Δ rel {d_logits:.2e}   top-1 agree {agree:.3f}")
    print(f"  new_hidden  Δ rel {d_hidden:.2e}   (pre-final-norm residual, the chained-draft feature)")
    print("PASS" if ok else "FAIL")
    assert ok, (f"MTP-M1 bake recon failed: cover={cover_ok} exact={exact_ok} "
                f"d_logits={d_logits:.2e} d_hidden={d_hidden:.2e}")


if __name__ == "__main__":
    run()
