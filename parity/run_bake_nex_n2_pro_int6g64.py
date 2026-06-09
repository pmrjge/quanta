"""Nex-N2-Pro (Qwen3.5-397B-A17B) int6-g64 RTN bake → ~/models/Nex-N2-Pro-quanta_int6g64.

The N2 **int6 safety-net arm** — the sibling of ``run_bake_nex_n2_pro_int4g64`` with the routed
experts at **int6** (``expert_bits=6``) instead of int4. Everything else is identical:

* **routed experts** (pre-stacked ``mlp.experts.gate_up_proj`` / ``down_proj``, all 60 MoE layers,
  512 experts each) → **int6 affine g64** (bf16 scales), quantized as 3-D stacks in one shot
  (``gather_qmm``-ready, no per-expert loop — rule 3);
* **non-experts** → **int8 affine g64** (gated-GQA q/k/v/o, Gated-DeltaNet in/out projections, shared
  expert) — UNCHANGED from the int4 arm (only the routed-expert width moves);
* **bf16/f32 control** (never quantized — rule 6): SSM control, every RMSNorm, the router ``gate``,
  the ``shared_expert_gate`` sigmoid, ``embed_tokens`` / ``lm_head``.

**Why int6 too.** int4-RTN was ~lossless (+0.3% ppl) on the bf16-source Nemotron-Ultra, so int4 is
the strong default — but the N2 bits decision is settled by the **e2e-ppl arbiter**
(``parity/nex_n2_pro_ppl.py``: bf16 vs int4 vs int6 head-to-head), not by assumption. This arm gives
that arbiter its int6 datapoint; if int4 regresses on Nex, int6 (≈304 GiB, still well under the 490.4
GiB ceiling) ships instead. MLX affine supports {2,3,4,6,8}, and ``gather_qmm`` decodes int6 at the
manifest-recorded width (``Qwen35Artifact`` reads ``bits`` from the manifest — never a hardcoded 4).

**Data-free.** Plain affine RTN over the stacks (``capture_acts=False``); the ``calib_ids`` is an
unused 1-token dummy.

**Self-contained (rule 6).** ``bake_qwen35`` bakes the **1M dynamic-YaRN** policy into ``config.json``
(``max_position_embeddings`` 1,010,000 + standard HF YaRN), **synthesizes** ``generation_config.json``
(Nex ships none; the ChatML two-eos stop set ``{<|im_end|>, <|endoftext|>}``), copies the tokenizer,
and then ``_audit_self_contained`` **fails loud** unless the artifact folder is fully standalone (no
symlinks, no external refs, relative weight_map, all shards present). ``include_mtp=False`` (Nex ships
zero ``mtp.*`` weights).

Streamed one layer resident at a time (rule 8): the per-MoE 512-expert bf16 stack is the peak; the
739 GiB whole model is never loaded. Expected resident **int6-g64 ≈ 304 GiB** (N0 projection, < the
490.4 GiB ceiling). **Run SOLO** (one model resident at a time — OOM/reboot hazard otherwise).

    uv run python -m parity.run_bake_nex_n2_pro_int6g64            # the real full bake
    uv run python -m parity.run_bake_nex_n2_pro_int6g64 --smoke    # tiny slice (2 layers, 8 experts)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from quanta.qwen35.bake import bake_qwen35

NEX = "/Users/pmrj/models/Nex-N2-Pro"
OUT = "/Users/pmrj/models/Nex-N2-Pro-quanta_int6g64"
GROUP_SIZE = 64       # the _int6g64 group (bf16 scales) — matches the N0 fit (304 GiB)
EXPERT_BITS = 6       # the safety-net arm: routed experts int6 (vs the int4 default)
CEILING_GIB = 490.4   # M3 Ultra recommended-max working set; the resident mix MUST fit


def run(smoke: bool = False) -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    ids = mx.array([0], dtype=mx.uint32)  # unused (data-free RTN; capture_acts=False) — dummy
    out = OUT + "_smoke" if smoke else OUT
    kw = dict(n_layers=2, expert_subset=range(8)) if smoke else {}
    print(f"{'SMOKE ' if smoke else ''}Nex-N2-Pro int6-RTN g{GROUP_SIZE} experts, int8 dense, "
          f"bf16 SSM/norms/head; MTP excluded; 1M-YaRN baked -> {out}", flush=True)
    t0 = time.perf_counter()
    stats = bake_qwen35(NEX, out, ids, include_head=True, include_mtp=False,
                        group_size=GROUP_SIZE, expert_bits=EXPERT_BITS, scale_dtype=mx.bfloat16, **kw)
    dt = time.perf_counter() - t0
    gib = stats["bytes"] / 2**30
    print(f"NEX-N2-PRO int6-g64 BAKE DONE in {dt / 3600:.2f}h ({dt / 60:.1f} min) | {gib:.1f} GiB "
          f"on disk\n{stats}", flush=True)
    # the bake's own audit already failed loud if not self-contained; surface it for the log
    print(f"SELF-CONTAINED — {stats['self_contained']}", flush=True)
    # sanity: the artifact's config must declare the 1M window (the user's explicit requirement)
    conf = json.loads((Path(out) / "config.json").read_text())
    mpe = conf.get("max_position_embeddings")
    assert mpe == 1_010_000, f"artifact config must declare 1M context, got max_position_embeddings={mpe}"
    # sanity: the int6 mix must fit RAM-resident under the M3 Ultra ceiling (skip on a sliced smoke)
    if not smoke:
        assert gib < CEILING_GIB, f"int6 mix {gib:.1f} GiB exceeds the {CEILING_GIB} GiB ceiling"
        assert stats["expert_bits"] == EXPERT_BITS and stats["counts"]["expert_int4"] == 120, \
            f"expected 120 int6 expert stacks @ bits={EXPERT_BITS}, got {stats['counts']}"
    print(f"VERIFIED — 1M context (max_position_embeddings={mpe}); resident {gib:.1f} GiB "
          f"< {CEILING_GIB} GiB ceiling", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv[1:])
