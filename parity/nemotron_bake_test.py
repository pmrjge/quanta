"""Nemotron AWQ bake orchestrator — bounded validation (one layer resident, no full bake).

(1) Policy coverage: ``classify()`` maps every representative tensor (mamba / attention / moe +
    experts + globals) with no unmapped raise (rule-6).
(2) AWQ on real Nemotron latent acts: for the hottest routed expert, ``(x/s)·dequant(W·diag s)ᵀ``
    reconstructs ``x·Wᵀ`` within int4 tolerance (the per-expert ``s`` contract is correct), and its
    activation-weighted error is no worse than plain RTN.
(3) Slice bake (2 layers, 8 experts): completes bounded, and the manifest tags experts
    ``awq_packed``, dense projections ``affine_packed``, SSM core ``dense``.

    uv run --with tokenizers python -m parity.nemotron_bake_test
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx

from quanta.bake.awq import awq_quantize
from quanta.bake.calibrate import expert_rows
from quanta.bake.quant import quantize_affine
from quanta.nemotron.bake import bake_nemotron
from quanta.nemotron.calibrate import capture_calibration
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.quant_policy import classify
from quanta.nemotron.tokenizer import NemotronTokenizer
from parity.nemotron_ppl import EMBED, HEAD, MODEL, NORMF, PROSE

GS, BITS, NTOK = 128, 4, 48


def _coverage(cfg: NemotronHConfig) -> bool:
    names = [EMBED, NORMF, HEAD, "backbone.layers.0.norm.weight"]
    names += [f"backbone.layers.0.mixer.{s}" for s in
              ("in_proj.weight", "out_proj.weight", "conv1d.weight", "conv1d.bias",
               "A_log", "D", "dt_bias", "norm.weight")]
    names += [f"backbone.layers.1.mixer.{s}" for s in
              ("gate.weight", "gate.e_score_correction_bias", "fc1_latent_proj.weight",
               "fc2_latent_proj.weight", "shared_experts.up_proj.weight", "shared_experts.down_proj.weight",
               "experts.0.up_proj.weight", "experts.0.down_proj.weight")]
    ai = cfg.hybrid_override_pattern.index("*")
    names += [f"backbone.layers.{ai}.mixer.{s}" for s in
              ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")]
    try:
        kinds = {n: classify(n).kind for n in names}
    except ValueError as e:
        print(f"  unmapped tensor: {e}")
        return False
    return (kinds["backbone.layers.1.mixer.experts.0.up_proj.weight"] == "int4_gptq"
            and kinds["backbone.layers.1.mixer.fc1_latent_proj.weight"] == "int8_affine"
            and kinds["backbone.layers.0.mixer.A_log"] == "bf16" and kinds[EMBED] == "bf16")


def _awq_real_acts(cfg: NemotronHConfig, ck: NemotronSourceCheckpoint) -> tuple:
    tok = NemotronTokenizer(MODEL)
    ids = mx.array(tok.encode(PROSE, add_bos=False)[:NTOK])
    caps = capture_calibration(ck, cfg, ids, n_layers=2)  # layer 1 is the first moe (pattern M,E,...)
    latent, idx = caps[1]
    counts = mx.sum(mx.any(idx[:, :, None] == mx.arange(cfg.n_routed_experts)[None, None, :], axis=1), axis=0)
    e = int(mx.argmax(counts).item())
    xe = expert_rows(latent, idx, e)  # [n, latent]
    w = ck.expert_stacks(1, cfg.n_routed_experts)["up"][e]  # [inter, latent]
    xf, wf = xe.astype(mx.float32), w.astype(mx.float32)
    yref = xf @ wf.T

    s, p, sc, b = awq_quantize(w, xe, BITS, GS)
    wq = mx.dequantize(p, sc, b, group_size=GS, bits=BITS).astype(mx.float32)
    yawq = (xf / s.astype(mx.float32)[None, :]) @ wq.T  # runtime applies x/s
    awq_err = float(mx.mean((yawq - yref) ** 2))

    pr, scr, br = quantize_affine(w, BITS, GS)
    wr = mx.dequantize(pr, scr, br, group_size=GS, bits=BITS).astype(mx.float32)
    rtn_err = float(mx.mean((xf @ wr.T - yref) ** 2))

    rt = awq_err / (float(mx.mean(yref ** 2)) + 1e-9)  # relative round-trip error (s contract)
    return e, int(xe.shape[0]), rt, awq_err, rtn_err


def _slice_bake(cfg: NemotronHConfig) -> tuple:
    out = tempfile.mkdtemp(suffix="_nemo_slice")
    tok = NemotronTokenizer(MODEL)
    ids = mx.array(tok.encode(PROSE, add_bos=False)[:NTOK])
    stats = bake_nemotron(MODEL, out, ids, n_layers=2, expert_subset=range(8), include_head=False,
                          group_size=GS, expert_method="awq", scale_dtype=mx.bfloat16)
    man = json.loads((Path(out) / "manifest.json").read_text())["tensors"]

    def fmt(k: str) -> str:
        return man[k]["format"]

    ok = (fmt("backbone.layers.1.mixer.experts.0.up_proj") == "awq_packed"
          and fmt("backbone.layers.1.mixer.fc1_latent_proj") == "affine_packed"
          and fmt("backbone.layers.0.mixer.A_log") == "dense")
    shutil.rmtree(out, ignore_errors=True)
    return stats, ok


def run() -> None:
    cfg = NemotronHConfig.from_pretrained(MODEL)
    ck = NemotronSourceCheckpoint(MODEL)

    cov = _coverage(cfg)
    e, n, rt, awq_err, rtn_err = _awq_real_acts(cfg, ck)
    stats, fmt_ok = _slice_bake(cfg)

    rt_ok = rt < 0.10  # int4 act-weighted round-trip
    helps_ok = awq_err <= rtn_err * 1.02  # AWQ no worse than RTN on real latent salience

    print("\n=== Nemotron AWQ bake orchestrator ===")
    print(f"policy coverage (rule-6)                 : {cov}")
    print(f"AWQ round-trip expert {e} (n={n} rows)   : rt {rt:.4f}<0.10 -> {rt_ok}")
    print(f"AWQ <= RTN act-weighted                  : {helps_ok}  (AWQ {awq_err:.4f} / RTN {rtn_err:.4f})")
    print(f"slice bake manifest formats              : {fmt_ok}  {stats}")
    assert all([cov, rt_ok, helps_ok, fmt_ok])
    print("Nemotron AWQ bake OK (policy covers all; s-contract reconstructs; manifest well-formed)")


if __name__ == "__main__":
    run()
