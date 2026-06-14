"""MiniMax-M3-VL M3-1: real-weight resident serving re-gate @ 397B (SOLO).

Validates :class:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel` on the REAL int6-g64 artifact at
full scale — the resident batched-serving foundation. Two forwards over the *same* held-out prose,
**sequentially** so only one is ever resident:

  1. **resident packed runtime** — ``MiniMaxM3ResidentModel(INT6, packed_experts=True)`` loads all 60
     text layers RAM-resident (routed experts held packed int6 → ``mx.gather_qmm``; the int8 mixer
     dequantized to bf16), ~325 GiB, pinned with ``set_wired_limit``. One prefill over the prompt.
     Freed before step 2.
  2. **streamed reference** — the M1/M2-gated one-layer-resident forward
     (:func:`parity.minimax_m3_ppl.streamed_logits` over the dequant-on-read
     :class:`~quanta.minimax.artifact_m3.MiniMaxM3Artifact`, ``gather_mm``, ~14.5 GiB peak) on the
     SAME int6 codes — the proven float baseline (M2b: ppl 5.00 on this artifact).

The resident ``gather_qmm`` path must MATCH the streamed ``gather_mm`` reference on the same codes:
a tight teacher-forced **Δppl** (the arbiter — CLAUDE.md methodology #4; the resident runtime ships
the M2b int6 quality) and high **top-1 agreement**. The two are NOT bit-identical: fused
``gather_qmm`` dequantizes the int6 codes at full precision and matmuls, while the streamed reference
rounds the dequantized weight to bf16 first (``MiniMaxM3Artifact._dequant``) — so the resident path
is actually the MORE precise one, and the few top-1 flips are low-confidence bf16 near-ties (the
settled secondary-signal finding: top-1 is a loose floor, ppl is the gate). This is the 397B-scale
analogue of the model-free ``parity/minimax_m3_runtime_test.py`` gate.

    uv run python -m parity.minimax_m3_runtime_real            # full re-gate (all 60 layers, SOLO)
    uv run python -m parity.minimax_m3_runtime_real 4 64       # n_layers, n_tok (bounded code smoke)

# parity-gate: real-weight
"""

from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx

from parity.minimax_m3_ppl import PROSE, streamed_logits, teacher_forced
from quanta.minimax.artifact_m3 import MiniMaxM3Artifact
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel
from quanta.minimax.tokenizer import MiniMaxTokenizer

SRC = "/Users/pmrj/models/MiniMax-M3"
INT6 = "/Users/pmrj/models/MiniMax-M3-quanta_int6g64"
N_TOK = 256          # a parity re-gate, not a ppl measurement — fewer tokens than the M2b arbiter
DPPL_CEILING = 1.0   # THE gate: same int6 codes, two dequant precisions ⇒ ppl Δ is sub-percent
AGREE_FLOOR = 0.90   # loose secondary signal — bf16 near-ties flip top-1 (the Nemotron-Ultra rule)
PPL_CEILING = 30.0   # the resident runtime must ship a healthy 397B ppl (the M2b int6 value ~5.0)

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _set_wired() -> None:
    """Pin the resident weight set (best-effort; the deployment target holds the model RAM-resident)."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        rec = int(info.get("max_recommended_working_set_size", 0))
        if rec > 0:
            mx.set_wired_limit(rec)
    except Exception:  # noqa: BLE001 — wired-limit is an optimization, never fail the gate on it
        pass


def run(n_layers: int | None = None, n_tok: int = N_TOK) -> None:
    full = n_layers is None
    mx.set_cache_limit(8 * 1024**3)
    _set_wired()
    cfg = MiniMaxM3Config.from_pretrained(SRC)
    # build the tokenizer directly (MiniMaxM3Config duck-types bos/eos; the BPE reads only
    # tokenizer.json); add_bos_token absent ⇒ raw encode (the M2b arbiter precedent).
    tok = MiniMaxTokenizer(os.path.join(SRC, "tokenizer.json"), cfg)
    ids_list = tok.encode(PROSE)[:n_tok]
    ids = mx.array(ids_list, dtype=mx.uint32)
    print(f"=== MiniMax-M3-VL M3-1 resident serving re-gate — {len(ids_list)} tok, "
          f"{'all 60' if full else n_layers} layers (SOLO) ===", flush=True)

    # 1) resident packed runtime (gather_qmm) — load all layers, one prefill, then FREE (rule 8/one model)
    t0 = time.perf_counter()
    model = MiniMaxM3ResidentModel(INT6, n_layers=n_layers, packed_experts=True)
    t_load = time.perf_counter() - t0
    logits_res = model(ids)                      # [1,T,vocab] prefill
    mx.eval(logits_res)
    ppl_res, acc_res, argmax_res = teacher_forced(logits_res, ids)
    n_built = model.num_layers
    del model
    mx.clear_cache()
    print(f"  [resident ] {n_built}L packed gather_qmm — ppl {ppl_res:7.4f}  acc {acc_res:.4f}  "
          f"(load {t_load:.0f}s)", flush=True)

    # 2) streamed reference (gather_mm, dequant-on-read) — the M1/M2-gated float baseline, SAME codes
    art = MiniMaxM3Artifact(INT6)
    t1 = time.perf_counter()
    logits_ref = streamed_logits(art, art.cfg, ids, n_layers=n_layers)
    ppl_ref, acc_ref, argmax_ref = teacher_forced(logits_ref, ids)
    dt_ref = time.perf_counter() - t1
    del art
    mx.clear_cache()

    agree = float(mx.mean((argmax_res == argmax_ref).astype(mx.float32)).item())
    rel = float((mx.linalg.norm((logits_res - logits_ref).astype(mx.float32))
                 / (mx.linalg.norm(logits_ref.astype(mx.float32)) + 1e-9)).item())
    dppl = 100.0 * (ppl_res / ppl_ref - 1.0) if ppl_ref > 0 else float("inf")
    print(f"  [streamed ] {n_built}L gather_mm ref    — ppl {ppl_ref:7.4f}  acc {acc_ref:.4f}  "
          f"({dt_ref:.0f}s)", flush=True)
    print(f"  resident vs streamed: top-1 agree {agree:.4f} | logit rel {rel:.2e} | Δppl {dppl:+.3f}%",
          flush=True)

    _ck(math.isfinite(ppl_res) and math.isfinite(ppl_ref), "non-finite ppl from a forward")
    _ck(agree >= AGREE_FLOOR,
        f"resident gather_qmm != streamed gather_mm: top-1 agree {agree:.4f} < {AGREE_FLOOR}")
    _ck(abs(dppl) < DPPL_CEILING,
        f"resident ppl {ppl_res:.4f} drifts from streamed {ppl_ref:.4f}: Δ {dppl:+.3f}% "
        f"(ceiling {DPPL_CEILING}%)")
    if full:
        _ck(ppl_res < PPL_CEILING,
            f"resident ppl {ppl_res:.4f} >= {PPL_CEILING}: the served runtime is not coherent "
            f"(expected ~5.0, the M2b int6 value)")
        print(f"\nVERDICT: resident M3 serving runtime VALIDATED @ 397B — packed-int6 gather_qmm == "
              f"the M1/M2 streamed reference (agree {agree:.4f}, Δppl {dppl:+.3f}%); ships the M2b "
              f"int6 quality (ppl {ppl_res:.3f}).", flush=True)
    else:
        print(f"\nSMOKE ok — resident path ran ({n_built} layers); agree {agree:.4f}, Δppl "
              f"{dppl:+.3f}% (numbers not meaningful on a partial model).", flush=True)
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else N_TOK
    run(nl, nt)
