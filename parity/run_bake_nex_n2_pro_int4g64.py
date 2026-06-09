"""Nex-N2-Pro (Qwen3.5-397B-A17B) int4-g64 RTN bake → ~/models/Nex-N2-Pro-quanta_int4g64.

The N2 int4 arm. Nex-N2-Pro is the post-trained **Qwen3.5-397B-A17B** (``qwen3_5_moe``), so the
in-tree :func:`quanta.qwen35.bake.bake_qwen35` targets it directly (the Qwen3.6-35B-A3B keeper is the
35B sibling baked the same way). Quant mix (project policy, #115 recipe):

* **routed experts** (pre-stacked ``mlp.experts.gate_up_proj`` / ``down_proj``, all 60 MoE layers,
  512 experts each) → **int4 affine g64** (bf16 scales), quantized as 3-D stacks in one shot
  (``gather_qmm``-ready, no per-expert loop — rule 3);
* **non-experts** → **int8 affine g64**: gated-GQA q/k/v/o (the 15 full layers), Gated-DeltaNet
  in/out projections (the 45 linear layers), and the shared expert;
* **bf16/f32 control** (never quantized — rule 6): SSM control (``A_log`` / ``dt_bias`` / ``conv1d`` /
  per-head ``norm``), every RMSNorm, the router ``gate``, the ``shared_expert_gate`` sigmoid, and the
  ``embed_tokens`` / ``lm_head`` token tables.

**Data-free.** int4 is plain affine **RTN** over the stacks (``capture_acts=False``) and int8 dense is
plain affine — no calibration forward, so the ``calib_ids`` below is an unused 1-token dummy (kept to
satisfy the positional arg; the int4-RTN-was-~lossless-e2e-on-bf16-source finding from Nemotron-Ultra
makes RTN the default arm, with int6 the safety net — N2's bits decision is the e2e-ppl arbiter).

**1M context is baked into ``config.json``** (N0): :func:`bake_qwen35` writes standard HF YaRN +
``max_position_embeddings`` 1,010,000 + a ``quanta_long_context`` block, and **synthesizes a correct
``generation_config.json``** (Nex ships none; the ChatML stop set ``{<|im_end|>, <|endoftext|>}`` is
derived from the tokenizer) + copies ``tokenizer.json`` / ``tokenizer_config.json`` /
``chat_template.jinja`` — so the artifact is a self-contained, first-class 1M model.

**``include_mtp=False``** — Nex declares ``mtp_num_hidden_layers=1`` but ships ZERO ``mtp.*`` weights
(N0 finding; ``from_pretrained`` refines ``num_mtp_modules→0``), so native-MTP spec-decode is N/A.

Streamed one layer resident at a time (rule 8): the per-MoE 512-expert bf16 stack is the peak; the
739 GiB whole model is never loaded. Expected resident **int4-g64 ≈ 214 GiB** (N0 projection, < the
490.4 GiB ceiling). **Run SOLO** (one model resident at a time — OOM/reboot hazard otherwise).

    uv run python -m parity.run_bake_nex_n2_pro_int4g64            # the real full bake
    uv run python -m parity.run_bake_nex_n2_pro_int4g64 --smoke    # tiny slice (2 layers, 8 experts)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

from quanta.qwen35.bake import bake_qwen35

NEX = "/Users/pmrj/models/Nex-N2-Pro"
OUT = "/Users/pmrj/models/Nex-N2-Pro-quanta_int4g64"
GROUP_SIZE = 64  # the _int4g64 expert (and non-expert) target, bf16 scales — matches the N0 fit (214 GiB)


def run(smoke: bool = False) -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    ids = mx.array([0], dtype=mx.uint32)  # unused (data-free RTN; capture_acts=False) — dummy
    out = OUT + "_smoke" if smoke else OUT
    # slice for the smoke: first 2 layers (L0 linear-attn + ... ; both MoE) with 8 experts each
    kw = dict(n_layers=2, expert_subset=range(8)) if smoke else {}
    print(f"{'SMOKE ' if smoke else ''}Nex-N2-Pro int4-RTN g{GROUP_SIZE} experts, int8 dense, "
          f"bf16 SSM/norms/head; MTP excluded; 1M-YaRN baked -> {out}", flush=True)
    t0 = time.perf_counter()
    stats = bake_qwen35(NEX, out, ids, include_head=True, include_mtp=False,
                        group_size=GROUP_SIZE, scale_dtype=mx.bfloat16, **kw)
    dt = time.perf_counter() - t0
    gib = stats["bytes"] / 2**30
    print(f"NEX-N2-PRO int4-g64 BAKE DONE in {dt / 3600:.2f}h ({dt / 60:.1f} min) | {gib:.1f} GiB "
          f"on disk\n{stats}", flush=True)
    # sanity: the artifact's config must declare the 1M window (the user's explicit requirement)
    import json
    conf = json.loads((Path(out) / "config.json").read_text())
    mpe = conf.get("max_position_embeddings")
    assert mpe == 1_010_000, f"artifact config must declare 1M context, got max_position_embeddings={mpe}"
    print(f"VERIFIED — artifact config declares 1M context (max_position_embeddings={mpe})", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv[1:])
