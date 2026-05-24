"""Full Nemotron-H bake → ~/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4awq (background).

The quant_policy mix: int4 **AWQ** routed experts (the bf16 source has the sub-grid headroom
Kimi's int4 source lacked), int8 affine dense (mamba in/out-proj, attention q/k/v/o, latent
fc1/fc2, shared expert), bf16 SSM core + every norm + router + embeddings/head. Well under the
490 GiB ceiling (bf16 is ~247 GiB; the mix is far smaller) — this is a decode-bandwidth play,
not a fit constraint. AWQ calibrates on ~2048 tokens of agentic prose+code. Self-contained:
the HF tokenizer is copied in.

    uv run --with tokenizers python -m parity.run_bake_nemotron
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import mlx.core as mx

from quanta.nemotron.bake import bake_nemotron
from quanta.nemotron.tokenizer import NemotronTokenizer

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
OUT = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4awq"
REPO = Path("/Users/pmrj/Environment/quant/finally_quanta")
CALIB_TOKENS = 2048
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def _calib_corpus(tok: NemotronTokenizer) -> mx.array:
    """Agentic-domain calibration (project code + instruction docs), to CALIB_TOKENS."""
    files = (sorted(REPO.glob("src/quanta/**/*.py")) + sorted(REPO.glob("parity/*.py"))
             + [REPO / "INITIAL_PROMPT.md", REPO / "CLAUDE.md"])
    text = "\n\n".join(p.read_text() for p in files if p.exists())
    return mx.array(tok.encode(text, add_bos=False)[:CALIB_TOKENS])  # Nemotron uses no BOS


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    tok = NemotronTokenizer(MODEL)
    ids = _calib_corpus(tok)
    print(f"calibration tokens: {ids.shape[0]} | int4-AWQ experts, int8 dense, bf16 SSM/norms/head",
          flush=True)
    t0 = time.perf_counter()
    stats = bake_nemotron(MODEL, OUT, ids, include_head=True, group_size=128,
                          expert_method="awq", scale_dtype=mx.bfloat16)
    for fn in _TOKENIZER_FILES:
        src = Path(MODEL) / fn
        if src.exists():
            shutil.copy(src, Path(OUT) / fn)
    print(f"NEMOTRON BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
