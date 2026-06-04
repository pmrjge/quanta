"""Full Nemotron-3-Ultra-550B bake → ~/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4awq_g64.

The U2 deliverable. quant_policy mix: int4 **AWQ g64** routed relu² experts (the U2 slice de-risk
`parity/nemotron_ultra_awq_slice_test.py` cleared AWQ at Ultra scale — finding #38's down-proj
collapse does NOT reproduce: AWQ helps up-proj 0.806 / ties down-proj 0.984, the α-grid rejects the
degenerate scales), int8 affine dense (mamba in/out-proj, attention q/k/v/o, latent fc1/fc2, shared
expert), bf16 SSM core + every norm + router + embeddings/head. RTN (`expert_method="rtn"`) is the
known-good fallback if U3 teacher-forced ppl regresses. Streamed one layer resident at a time
(rule 8): the per-MoE expert stack (~21.5 GiB bf16) is the peak; the 1023 GiB whole model is never
loaded. Self-contained: the HF tokenizer is copied in.

Memory: the int4-AWQ g64 mix is ~290 GiB resident (U0-measured) — far under the 490.4 GiB ceiling;
this is a decode-bandwidth play, not a fit constraint. AWQ calibrates on ~4K agentic prose+code
tokens (matching the de-risk slice). **Hours; run solo (OOM hazard — one model resident at a time).**

    uv run --with tokenizers python -m parity.run_bake_nemotron_ultra_int4awq_g64           # full
    uv run --with tokenizers python -m parity.run_bake_nemotron_ultra_int4awq_g64 --smoke   # tiny slice
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

from quanta.nemotron.bake import bake_nemotron
from quanta.nemotron.tokenizer import NemotronTokenizer

ULTRA = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
OUT = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4awq_g64"
REPO = Path("/Users/pmrj/Environment/agentic_ai/finally_quanta")
CALIB_TOKENS = 4096          # matches the de-risk slice (~176 rows/expert avg → robust AWQ scales)
GROUP_SIZE = 64              # the _int4g64 expert target (g64: more scale overhead, better e2e per #38)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def _calib_corpus(tok: NemotronTokenizer) -> mx.array:
    """Agentic-domain calibration (project code + instruction docs), to CALIB_TOKENS, no BOS —
    mirrors `parity.nemotron_ultra_awq_slice_test._calib_ids` (the de-risk used the same corpus)."""
    files = (sorted(REPO.glob("src/quanta/**/*.py")) + sorted(REPO.glob("parity/*.py"))
             + [REPO / "INITIAL_PROMPT.md", REPO / "CLAUDE.md"])
    text = "\n\n".join(p.read_text() for p in files if p.exists())
    return mx.array(tok.encode(text, add_bos=False)[:CALIB_TOKENS])  # Nemotron uses no BOS


def run(smoke: bool = False) -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    tok = NemotronTokenizer(ULTRA)
    ids = _calib_corpus(tok)
    out = OUT + "_smoke" if smoke else OUT
    kw = dict(n_layers=2, expert_subset=range(8)) if smoke else {}  # slice: mamba L0 + first MoE L1, 8 experts
    print(f"{'SMOKE ' if smoke else ''}calibration tokens: {ids.shape[0]} | int4-AWQ g{GROUP_SIZE} experts, "
          f"int8 dense, bf16 SSM/norms/head -> {out}", flush=True)
    t0 = time.perf_counter()
    stats = bake_nemotron(ULTRA, out, ids, include_head=True, group_size=GROUP_SIZE,
                          expert_method="awq", scale_dtype=mx.bfloat16, **kw)
    for fn in _TOKENIZER_FILES:
        src = Path(ULTRA) / fn
        if src.exists():
            shutil.copy(src, Path(out) / fn)
    print(f"NEMOTRON-ULTRA BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv[1:])
