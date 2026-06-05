"""Nemotron-3-Ultra-550B RTN fallback bake → ~/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64.

The U3 fallback arm. Byte-identical quant mix to the AWQ bake
(`parity/run_bake_nemotron_ultra_int4awq_g64.py`) **except** the routed relu² experts use plain int4
**RTN** (s=1, data-free) instead of AWQ — int8 affine dense (mamba in/out-proj, attention q/k/v/o,
latent fc1/fc2, shared expert), bf16 SSM core + every norm + router + embeddings/head, unchanged.

Why: finding #38, codified in `bake._bake_expert` — AWQ misfires on the relu² down-proj (degenerate
per-channel scales) so it *regresses* e2e ppl, while plain int4 RTN is ~lossless e2e at the SAME
4-bit footprint. The U2 de-risk's "AWQ ties/helps RTN" was recon-only + L1-only (recon does NOT
predict e2e — settled finding); U3 then measured the AWQ bake at **+11.2% ppl vs bf16**. This bake is
the head-to-head RTN arm the U3 comparison ranks against AWQ + bf16.

RTN is **data-free for the experts** (`bake_nemotron` skips `capture_calibration` when
`expert_method="rtn"`) and for the int8 dense (plain affine), so this is FASTER than the AWQ bake;
the calib corpus below is built only to keep the script a true clone of the AWQ driver — it is
unused in RTN mode. Streamed one layer resident at a time (rule 8): the per-MoE expert stack
(~21.5 GiB bf16) is the peak; the 1023 GiB whole model is never loaded. Self-contained: the HF
tokenizer is copied in. **Run solo (OOM hazard — one model resident at a time).**

    uv run --with tokenizers python -m parity.run_bake_nemotron_ultra_int4rtn_g64           # full
    uv run --with tokenizers python -m parity.run_bake_nemotron_ultra_int4rtn_g64 --smoke   # tiny slice
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
OUT = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64"
REPO = Path("/Users/pmrj/Environment/agentic_ai/finally_quanta")
CALIB_TOKENS = 4096          # unused in RTN mode (data-free experts); kept to mirror the AWQ driver
GROUP_SIZE = 64              # the _int4g64 expert target — same footprint as the AWQ arm
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def _calib_corpus(tok: NemotronTokenizer) -> mx.array:
    """Mirror the AWQ driver's corpus verbatim (unused for RTN — experts are data-free int4)."""
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
    print(f"{'SMOKE ' if smoke else ''}calibration tokens: {ids.shape[0]} (unused, RTN) | int4-RTN "
          f"g{GROUP_SIZE} experts, int8 dense, bf16 SSM/norms/head -> {out}", flush=True)
    t0 = time.perf_counter()
    stats = bake_nemotron(ULTRA, out, ids, include_head=True, group_size=GROUP_SIZE,
                          expert_method="rtn", scale_dtype=mx.bfloat16, **kw)
    for fn in _TOKENIZER_FILES:
        src = Path(ULTRA) / fn
        if src.exists():
            shutil.copy(src, Path(out) / fn)
    print(f"NEMOTRON-ULTRA RTN BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv[1:])
