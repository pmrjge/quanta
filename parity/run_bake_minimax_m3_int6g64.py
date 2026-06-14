"""MiniMax-M3-VL int6-g64 RTN bake → ~/models/MiniMax-M3-quanta_int6g64 (full VL).

The M2 bake (the user's decisions: **full VL** + **int6-g64 for margin**, skip int4):

* **routed experts** (pre-stacked ``block_sparse_moe.experts.{gate_up,down}_proj``, all 57 MoE layers
  3–59, 128 experts each) → **int6 affine g64** (bf16 scales), quantized as 3-D stacks in one shot
  (``gather_qmm``-ready, no per-expert loop — rule 3);
* **non-experts** → **int8 affine g64**: GQA q/k/v/o, the dense-FFN gate/up/down (layers 0–2), the
  shared expert;
* **bf16/f32 control** (never quantized — rule 6): every RMSNorm, the router ``gate`` +
  ``e_score_correction_bias`` (f32), the trained sparse indexer (``index_{q,k}_proj/norm``, bf16),
  ``embed_tokens`` / ``lm_head``;
* **full VL**: the whole vision tower + projector + patch-merge → dense bf16 verbatim (523 tensors).

**Data-free** plain affine RTN over the stacks (bf16 source has the sub-int6-grid headroom — settled).
**Self-contained (rule 6):** ``bake_minimax_m3`` asserts the artifact declares the native 1M window,
copies the tokenizer + the VL preprocessor + the authoritative ``generation_config.json`` (eos
200020), and ``_audit_self_contained`` **fails loud** unless the folder is fully standalone.

Streamed one text layer resident at a time (rule 8): the per-MoE 128-expert bf16 stack (~14.5 GiB) is
the peak; the 809.5 GiB whole model is never loaded. Expected resident **int6-g64 ≈ 329.6 GiB** (M0
projection, < the 490.4 GiB ceiling). **Run SOLO** (one model resident — OOM/reboot hazard).

    uv run python -m parity.run_bake_minimax_m3_int6g64           # the real full bake (multi-hour, SOLO)
    uv run python -m parity.run_bake_minimax_m3_int6g64 --smoke   # tiny slice (4 layers, 8 experts, no ViT)

# parity-gate: real-weight
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from quanta.minimax.bake_m3 import bake_minimax_m3
from quanta.minimax.config_m3 import MiniMaxM3Config

MINIMAX = "/Users/pmrj/models/MiniMax-M3"
OUT = "/Users/pmrj/models/MiniMax-M3-quanta_int6g64"
GROUP_SIZE = 64       # the _int6g64 group (bf16 scales) — matches the M0 fit (329.6 GiB)
EXPERT_BITS = 6       # the user's decision: routed experts int6 (skip int4) for margin
CEILING_GIB = 490.4   # M3 Ultra recommended-max working set; the resident mix MUST fit
NATIVE_CTX = 1_048_576


def run(smoke: bool = False) -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    out = OUT + "_smoke" if smoke else OUT
    kw = dict(n_layers=4, expert_subset=range(8), include_vision=False) if smoke else {}
    print(f"{'SMOKE ' if smoke else ''}MiniMax-M3-VL int6-RTN g{GROUP_SIZE} experts, int8 dense, "
          f"bf16 norms/router/indexer/head{'' if smoke else ' + full ViT'}; 1M native -> {out}",
          flush=True)
    cfg = MiniMaxM3Config.from_pretrained(MINIMAX)
    n_moe = sum(1 for i in range(cfg.num_hidden_layers) if cfg.is_moe_layer(i))
    t0 = time.perf_counter()
    stats = bake_minimax_m3(MINIMAX, out, group_size=GROUP_SIZE, expert_bits=EXPERT_BITS,
                            scale_dtype=mx.bfloat16, **kw)
    dt = time.perf_counter() - t0
    gib = stats["bytes"] / 2**30
    print(f"MINIMAX-M3 int6-g64 BAKE DONE in {dt / 3600:.2f}h ({dt / 60:.1f} min) | {gib:.1f} GiB "
          f"on disk\n{stats}", flush=True)
    print(f"SELF-CONTAINED — {stats['self_contained']}", flush=True)
    # the artifact's config must declare the native 1M window (the user's explicit requirement)
    conf = json.loads((Path(out) / "config.json").read_text())
    tc = conf.get("text_config", conf)
    mpe = tc.get("max_position_embeddings", conf.get("max_position_embeddings"))
    assert mpe == NATIVE_CTX, f"artifact config must declare 1M context, got max_position_embeddings={mpe}"
    if not smoke:
        assert gib < CEILING_GIB, f"int6 mix {gib:.1f} GiB exceeds the {CEILING_GIB} GiB ceiling"
        assert stats["expert_bits"] == EXPERT_BITS and stats["counts"]["expert_int"] == 2 * n_moe, \
            f"expected {2 * n_moe} int6 expert stacks ({n_moe} MoE layers × 2), got {stats['counts']}"
        assert stats["vision_tensors"] == 523, \
            f"full VL must bake 523 vision tensors, got {stats['vision_tensors']}"
    print(f"VERIFIED — 1M context (max_position_embeddings={mpe}); "
          f"{'SMOKE slice ok' if smoke else f'resident {gib:.1f} GiB < {CEILING_GIB} GiB ceiling, full VL'}",
          flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv[1:])
