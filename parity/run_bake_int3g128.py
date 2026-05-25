"""Bake: uniform int3 gate/up/down at group-128, bf16 scales → Kimi-K2.6-quanta_int3g128.

Raises the int2g64 quality bottleneck (gate/up int2) to int3 *everywhere*. Uniform int3 g128 lands
~398 GiB — only ~+10 GiB over int2g64 (388 GiB): raising gate/up int2→int3 (+49) is nearly offset by
dropping down int4→int3 (−39), so killing the 2-bit gate/up bottleneck is "almost free". RTN coding
(settled: scale-only methods don't move e2e on Kimi's int4 source — only the bit count does; AWQ/DWQ/
GPTQ are e2e-equivalent here and RTN is ~10× faster). bf16 scales halve the per-group overhead
(int3-g128 = 3.25 bpp). Separate output dir; int2g64 untouched. This base replaces int2g64 for both
serving and EAGLE-3 feature capture (its features have a higher next-token ceiling than int2g64's).

    uv run --with tiktoken python -m parity.run_bake_int3g128
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

OUT = "/Users/pmrj/models/Kimi-K2.6-quanta_int3g128"


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = _agentic_corpus(tok)
    print(f"calibration tokens: {ids.shape[0]} | scheme: gate/up/down int3, g128, bf16 scales (RTN)",
          flush=True)
    t0 = time.perf_counter()
    stats = bake(MODEL, OUT, ids, include_head=True, expert_method="rtn",
                 group_size=128, scale_dtype=mx.bfloat16,
                 fixed_expert_bits={"gate_proj": 3, "up_proj": 3, "down_proj": 3})
    for fn in ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja"):  # self-contained tokenizer
        shutil.copy(Path(MODEL) / fn, Path(OUT) / fn)
    print(f"INT3-g128 BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
