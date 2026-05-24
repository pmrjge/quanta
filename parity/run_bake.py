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


def _agentic_corpus(tok: KimiTokenizer) -> mx.array:
    """Agentic-domain calibration: the project's code + instruction docs — representative of an
    agentic loop (reading/writing code, following instructions) — to a full 8192 tokens."""
    files = (sorted(REPO.glob("src/quanta/**/*.py")) + sorted(REPO.glob("parity/*.py"))
             + [REPO / "INITIAL_PROMPT.md", REPO / "CLAUDE.md"])
    text = "\n\n".join(p.read_text() for p in files if p.exists())
    return mx.array(tok.encode(text, add_bos=True)[:CALIB_TOKENS])


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    cfg = KimiTextConfig.from_pretrained(MODEL)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)
    ids = _agentic_corpus(tok)
    print(f"calibration tokens: {ids.shape[0]} (agentic: code + docs)", flush=True)

    t0 = time.perf_counter()
    stats = bake(MODEL, OUT, ids, include_head=True, expert_byte_budget=EXPERT_BUDGET, target=0.08,
                 expert_method="rtn")  # scale-only RTN: e2e-equivalent to GPTQ on the int4 source, ~10x faster
    for fn in ("tiktoken.model", "tokenizer_config.json", "chat_template.jinja"):  # self-contained tokenizer + chat
        shutil.copy(Path(MODEL) / fn, Path(OUT) / fn)
    print(f"BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
