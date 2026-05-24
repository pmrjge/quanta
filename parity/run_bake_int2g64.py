"""Bake: int2 gate/up + int4 down at group-64, bf16 scales → Kimi-K2.6-quanta_int2g64.

The "lighter middleground": same bits as int2g4 (gate/up int2, down int4) but finer groups
(g128→g64) for lower quant error, paid for by **bf16 scales** so the per-group overhead halves
(int2-g64 = 2.5 bpp vs int2-g64-fp32 3.0). Lands ~388 GiB — same size as the fp32-g128 int2g4
artifact but a finer grid, so it should beat int2g4's 4.687 ppl. Separate output dir; the
validated int3 (3.279) and int2g4 (4.687) artifacts are untouched. RTN (coding method doesn't
move e2e on the int4 source — settled).

    uv run --with tiktoken python -m parity.run_bake_int2g64
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import mlx.core as mx

from quanta.bake.orchestrate import bake
from quanta.config import KimiTextConfig
from quanta.tokenizer import KimiTokenizer
from parity.run_bake import MODEL, _agentic_corpus

OUT = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = _agentic_corpus(tok)
    print(f"calibration tokens: {ids.shape[0]} | scheme: gate/up int2, down int4, g64, bf16 scales (RTN)",
          flush=True)
    t0 = time.perf_counter()
    stats = bake(MODEL, OUT, ids, include_head=True, expert_method="rtn",
                 group_size=64, scale_dtype=mx.bfloat16,
                 fixed_expert_bits={"gate_proj": 2, "up_proj": 2, "down_proj": 4})
    for fn in ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(Path(MODEL) / fn, Path(OUT) / fn)
    print(f"INT2-g64 BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
