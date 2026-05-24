"""Experiment: int2 gate/up + int4 down (RTN) → Kimi-K2.6-quanta_int2g4.

Forces the aggressive scheme to test whether the int3-floor (ppl 3.279) survives a 2-bit
gate/up — the size lever that frees ~93 GiB (388 GiB total). Separate output dir so the
validated int3 artifact is untouched. DWQ wouldn't change this e2e (coding method doesn't move
ppl), so RTN is the honest, fast probe.

    uv run --with tiktoken python -m parity.run_bake_int2
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

OUT = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g4"


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = _agentic_corpus(tok)
    print(f"calibration tokens: {ids.shape[0]} | scheme: gate/up int2, down int4 (RTN)", flush=True)
    t0 = time.perf_counter()
    stats = bake(MODEL, OUT, ids, include_head=True, expert_method="rtn",
                 fixed_expert_bits={"gate_proj": 2, "up_proj": 2, "down_proj": 4})
    for fn in ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(Path(MODEL) / fn, Path(OUT) / fn)
    print(f"INT2 BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
