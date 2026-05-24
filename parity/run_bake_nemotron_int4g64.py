"""Nemotron-H bake → ~/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64 (background).

The corrected scheme after the #38 finding: routed experts are **plain affine int4 g64**
(``expert_method="rtn"`` → ``s=1``, no AWQ), int8 affine dense (mamba in/out-proj, attention
q/k/v/o, latent fc1/fc2, shared expert), bf16 SSM core + every norm + router + embeddings/head.

AWQ was actively harmful: its activation scaling collapses on the relu^2 down-projection
(degenerate per-channel scales) → +75% e2e ppl. Plain int4 is lossless e2e (+0.1% g128 / -2.5%
g64 vs the bf16 5.981 reference; measured in parity.nemotron_quantsim_ppl) at the same 4-bit
footprint, so decode stays bandwidth-cheap (~115 tok/s at g64) — the whole point. RTN needs no
calibration pass, so this bake is fast. bf16 scales keep experts at 4.5 bpp (vs fp32's 5.0).
Self-contained: the HF tokenizer is copied in.

    uv run --with tokenizers python -m parity.run_bake_nemotron_int4g64
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import mlx.core as mx

from quanta.nemotron.bake import bake_nemotron
from quanta.nemotron.tokenizer import NemotronTokenizer

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
OUT = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
GROUP_SIZE = 64
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def run() -> None:
    mx.set_cache_limit(32 * 1024**3)  # cap the MLX buffer cache so the resident set can't balloon
    tok = NemotronTokenizer(MODEL)
    ids = mx.array(tok.encode("calibration unused for rtn", add_bos=False))  # RTN ignores calib_ids
    print(f"int4 RTN (plain, no AWQ) g{GROUP_SIZE} experts, int8 dense, bf16 SSM/norms/head", flush=True)
    t0 = time.perf_counter()
    stats = bake_nemotron(MODEL, OUT, ids, include_head=True, group_size=GROUP_SIZE,
                          expert_method="rtn", scale_dtype=mx.bfloat16)
    for fn in _TOKENIZER_FILES:
        src = Path(MODEL) / fn
        if src.exists():
            shutil.copy(src, Path(OUT) / fn)
    print(f"NEMOTRON int4-g64 BAKE DONE in {(time.perf_counter() - t0) / 3600:.2f}h\n{stats}", flush=True)


if __name__ == "__main__":
    run()
