"""Nemotron-3-Ultra AWQ slice diagnostic (U2 de-risk): does finding #38's relu² down-proj AWQ
degeneracy reproduce at Ultra scale — BEFORE committing the multi-hour bake?

The user chose AWQ experts (reversing decision #1's "not AWQ"). Finding #38 measured AWQ at +75%
e2e ppl on Super-120B (a *bf16* source, like Ultra) because each routed expert's **down-projection**
reads ``relu2(up·latent)`` — non-negative and sparse — so AWQ's per-input-channel scale
``s = (mean|x|/max)^α`` collapses on the many near-zero channels (clamped at ``eps=1e-6``), and the
runtime's folded ``1/s`` then amplifies any calibration→inference activation mismatch on exactly
those channels. Plain int4 RTN (``s=1``) has no such failure mode (lossless e2e, +0.1% g128 / −2.5%
g64). The failure is architectural (relu² activation), NOT source precision — so Ultra being bf16
does not rescue it. This test checks whether the mechanism is actually present at Ultra's wider scale.

Method — stream ONLY Ultra layers 0 (mamba) → 1 (the first ``moe`` in ``layers_block_type``),
capturing the real routed latent + routing the way :mod:`quanta.nemotron.calibrate` does, but
materializing NO routed experts (gate + fc1 only; mamba layer-0 forward is the faithful block, the
only ~1 GiB resident — rule 8). Then, per warm expert (sampled), run a **held-out** test: fit the
AWQ scale on 70% of the expert's routed rows and measure the activation-weighted reconstruction
error on the held-out 30% for BOTH AWQ and RTN. If AWQ generalizes *worse* than RTN on the down-proj
(ratio > 1) with degenerate scales, finding #38's mechanism is present → recommend RTN. If AWQ
generalizes as well or better, the full AWQ bake is viable. (Diagnostic, not a pass/fail gate — it
reports numbers for the bake-method decision; the e2e arbiter remains U3 teacher-forced ppl.)

    uv run --with tokenizers python -m parity.nemotron_ultra_awq_slice_test
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from quanta.bake.awq import _recon, awq_scale
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.model import NemotronBlock, load_block
from quanta.nemotron.moe import relu2
from quanta.nemotron.tokenizer import NemotronTokenizer

ULTRA = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
REPO = Path("/Users/pmrj/Environment/agentic_ai/finally_quanta")
BITS, GS = 4, 64          # the _int4g64 expert target (g64: more scale overhead, better e2e per #38)
CALIB_TOKENS = 4096       # ~176 rows/expert avg over 512 experts top-22 → robust held-out split
FIT_FRAC = 0.7
MIN_ROWS = 64             # only experts with enough rows for a meaningful fit/eval split
MAX_EXPERTS = 24
MOE_LAYER = 1             # first 'moe' in layers_block_type (verified: [mamba, moe, ...])
EMBED = "backbone.embeddings.weight"


def _calib_ids(tok: NemotronTokenizer) -> mx.array:
    """Agentic-domain calibration (project code + instruction docs), truncated, no BOS — mirrors
    parity.run_bake_nemotron._calib_corpus (with the current repo path)."""
    files = (sorted(REPO.glob("src/quanta/**/*.py")) + sorted(REPO.glob("parity/*.py"))
             + [REPO / "INITIAL_PROMPT.md", REPO / "CLAUDE.md"])
    text = "\n\n".join(p.read_text() for p in files if p.exists())
    return mx.array(tok.encode(text, add_bos=False)[:CALIB_TOKENS])


def _capture(ck: NemotronSourceCheckpoint, cfg: NemotronHConfig,
             ids: mx.array) -> tuple[mx.array, mx.array]:
    """Stream layers 0 (mamba) → 1 (moe); return (latent [N,lat] bf16, idx [N,topk] int32) for the
    first MoE layer — the routed experts' input + routing, exactly as the bake's AWQ calibration
    sees them, but with NO 21.5 GiB expert stack materialized (gate + fc1 only)."""
    h = ck.read(EMBED)[ids][None].astype(mx.bfloat16)
    mx.eval(h)
    ck.release()
    blk0 = NemotronBlock(cfg, "mamba")           # faithful mamba forward (~1 GiB, the only big load)
    load_block(blk0, ck, cfg, 0)
    h, _, _ = blk0(h)
    mx.eval(h)
    del blk0
    ck.release()
    mx.clear_cache()

    t = ck.moe_nonexpert_tensors(MOE_LAYER)      # gate + fc1/fc2 + shared + norm — NO experts
    norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
    norm.weight = t["layer_norm"]
    fc1 = nn.Linear(cfg.hidden_size, cfg.moe_latent_size, bias=False)
    fc1.weight = t["fc1_latent_proj.weight"]
    hf = norm(h).reshape(-1, cfg.hidden_size)
    # routing — mirrors NemotronLatentMoE._route (sigmoid scores + correction-bias top-k; U1-exact)
    logits = hf.astype(mx.float32) @ t["gate.weight"].astype(mx.float32).T
    choice = mx.sigmoid(logits) + t["gate.e_score_correction_bias"].astype(mx.float32)[None]
    topk = cfg.num_experts_per_tok
    idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    latent = fc1(hf)
    mx.eval(latent, idx)
    ck.release()
    mx.clear_cache()
    return latent.astype(mx.bfloat16), idx


def _err(w: mx.array, x_eval: mx.array, s: mx.array) -> float:
    """Held-out activation-weighted recon error ``‖(x/s)·dq(W·s)ᵀ − xWᵀ‖ / ‖xWᵀ‖`` (``s=1`` → RTN).
    Uses AWQ's exact recon (``_recon`` = affine quantize→dequantize), so this measures precisely
    what the runtime computes with the calibration-fitted scale on UNSEEN rows."""
    xf, wf = x_eval.astype(mx.float32), w.astype(mx.float32)
    y = xf @ wf.T
    yq = (xf / s[None, :]) @ _recon(wf * s[None, :], BITS, GS).T
    return float((mx.linalg.norm(yq - y) / (mx.linalg.norm(y) + 1e-12)).item())


def run() -> None:
    mx.set_cache_limit(8 * 1024**3)
    cfg = NemotronHConfig.from_pretrained(ULTRA)
    assert cfg.layers_block_type[MOE_LAYER] == "moe", "MOE_LAYER is not a moe layer"
    tok = NemotronTokenizer(ULTRA)
    ids = _calib_ids(tok)
    ck = NemotronSourceCheckpoint(ULTRA)
    latent, idx = _capture(ck, cfg, ids)
    n_tok = latent.shape[0]

    idx_np = np.asarray(idx)
    counts = np.bincount(idx_np.reshape(-1), minlength=cfg.n_routed_experts)
    warm = np.where(counts >= MIN_ROWS)[0]
    if len(warm) == 0:
        raise SystemExit(f"no warm experts (>= {MIN_ROWS} rows) among {n_tok} tokens; raise CALIB_TOKENS")
    pick = np.unique(warm[np.linspace(0, len(warm) - 1, min(MAX_EXPERTS, len(warm))).astype(int)])

    rows = []
    for e in pick:                                # bounded sample loop (IO/accounting; rule 3 OK)
        e = int(e)
        rid = np.where(np.any(idx_np == e, axis=1))[0]
        nfit = int(len(rid) * FIT_FRAC)
        fit, ev = mx.array(rid[:nfit]), mx.array(rid[nfit:])
        xf_fit, xf_ev = latent[fit], latent[ev]
        up = ck.read(ck.expert_key(MOE_LAYER, e, "up_proj"))      # [inter, lat]
        down = ck.read(ck.expert_key(MOE_LAYER, e, "down_proj"))  # [lat, inter]
        ones_up, ones_dn = mx.ones((up.shape[1],)), mx.ones((down.shape[1],))
        # up-proj (input = dense latent): AWQ vs RTN on held-out rows
        s_up = awq_scale(up, xf_fit, BITS, GS)
        eu_awq, eu_rtn = _err(up, xf_ev, s_up), _err(up, xf_ev, ones_up)
        # down-proj (input = relu2(up·latent), sparse): AWQ vs RTN on held-out rows
        din_fit = relu2(xf_fit.astype(mx.float32) @ up.astype(mx.float32).T)
        din_ev = relu2(xf_ev.astype(mx.float32) @ up.astype(mx.float32).T)
        s_dn = awq_scale(down, din_fit, BITS, GS)
        ed_awq, ed_rtn = _err(down, din_ev, s_dn), _err(down, din_ev, ones_dn)
        # down-proj degeneracy: relu² channel sparsity (near-zero) + AWQ scale dynamic range
        a = mx.mean(mx.abs(din_fit), axis=0)
        a = a / mx.max(a)
        sparsity = float(mx.mean((a < 1e-3).astype(mx.float32)).item())
        s_rng = float((mx.max(s_dn) / mx.maximum(mx.min(s_dn), 1e-12)).item())
        rows.append((e, len(rid), eu_awq, eu_rtn, ed_awq, ed_rtn, sparsity, s_rng))
    ck.release()

    arr = np.array([r[2:] for r in rows])         # [E, 6]
    mu = arr.mean(axis=0)
    up_ratio, dn_ratio = mu[0] / mu[1], mu[2] / mu[3]
    dn_worse = int((arr[:, 2] / arr[:, 3] > 1.0).sum())

    print("\n=== Nemotron-Ultra AWQ slice diagnostic (layer 1, first MoE) ===")
    print(f"calib tokens {n_tok} | warm experts (>= {MIN_ROWS} rows) {len(warm)} | sampled {len(pick)}")
    print(f"{'exp':>5} {'rows':>5} {'up_awq':>8} {'up_rtn':>8} {'dn_awq':>8} {'dn_rtn':>8} "
          f"{'dn_spars':>9} {'dn_srng':>11}")
    for (e, nr, eua, eur, eda, edr, sp, sr) in rows:
        print(f"{e:>5} {nr:>5} {eua:>8.4f} {eur:>8.4f} {eda:>8.4f} {edr:>8.4f} {sp:>9.2%} {sr:>11.1f}")
    print("-" * 78)
    print(f"mean up-proj   held-out err : AWQ {mu[0]:.4f} vs RTN {mu[1]:.4f}  -> ratio {up_ratio:.3f} "
          f"({'AWQ helps' if up_ratio < 1 else 'AWQ HURTS'})")
    print(f"mean down-proj held-out err : AWQ {mu[2]:.4f} vs RTN {mu[3]:.4f}  -> ratio {dn_ratio:.3f} "
          f"({'AWQ helps' if dn_ratio < 1 else 'AWQ HURTS'})")
    print(f"down-proj relu² channel sparsity (near-zero): mean {mu[4]:.2%}")
    print(f"down-proj AWQ scale dynamic range (max/min) : mean {mu[5]:.1f}")
    print(f"down-proj experts where AWQ worse than RTN  : {dn_worse}/{len(rows)}")
    print()
    if dn_ratio > 1.0:
        print(f"VERDICT: AWQ DEGRADES the relu² down-proj at Ultra scale (held-out ratio "
              f"{dn_ratio:.3f} > 1) — finding #38's mechanism is PRESENT. Recommend plain int4 RTN "
              f"(e2e-lossless) for the experts; AWQ would risk the +75% e2e regression.")
    else:
        print(f"VERDICT: AWQ generalizes on the down-proj at Ultra scale (held-out ratio "
              f"{dn_ratio:.3f} <= 1) — finding #38 does NOT reproduce here; the full AWQ bake is "
              f"viable, with U3 teacher-forced ppl as the final arbiter (RTN the known-good fallback).")


if __name__ == "__main__":
    run()
