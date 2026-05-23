"""Full bake driver → ~/models/Kimi-K2.6-quanta_int3 (background, ~hours).

Assembles a calibration corpus, runs the full 61-layer bake (int3/int4 GPTQ experts, int8
non-experts, bf16 shared/norms), and copies the tokenizer into the artifact for
self-containment. expert_byte_budget is the real ~490 GiB-minus-non-experts ceiling so the DP
produces a genuine int3/int4 mix (all-int4 would be ~539 GB and not fit).

    uv run --with tiktoken python -m parity.run_bake
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import mlx.core as mx

from quanta.bake.orchestrate import bake
from quanta.config import KimiTextConfig
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
OUT = "/Users/pmrj/models/Kimi-K2.6-quanta_int3"
REPO = Path("/Users/pmrj/Environment/quant/finally_quanta")
EXPERT_BUDGET = 470e9  # ~490 GiB ceiling minus int8 non-experts + bf16 shared
CALIB_TOKENS = 8192


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    from parity.ppl_long import LONG_TEXT  # science prose; domain-matched to the ppl eval

    parts = [LONG_TEXT]
    for f in ("INITIAL_PROMPT.md", "CLAUDE.md"):
        p = REPO / f
        if p.exists():
            parts.append(p.read_text())
    ids = mx.array(tok.encode("\n\n".join(parts), add_bos=True)[:CALIB_TOKENS])
    print(f"calibration tokens: {ids.shape[0]}", flush=True)

    t0 = time.perf_counter()
    stats = bake(MODEL, OUT, ids, include_head=True, expert_byte_budget=EXPERT_BUDGET, target=0.08)
    shutil.copy(Path(MODEL) / "tiktoken.model", Path(OUT) / "tiktoken.model")  # self-contained tokenizer
    print(f"BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
