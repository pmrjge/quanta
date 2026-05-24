"""Capture EAGLE-3 training features from the resident int2g64 Kimi (runs alone — 389 GiB).

Teacher-forced pass over an agentic corpus; saves (low/mid/high fused hidden, input token, target's
argmax next token) to a sibling working dir. PoC scale by default; pass a token budget to scale up.

    uv run --with tiktoken python -m parity.eagle_capture [n_tokens]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

from quanta.eagle.capture import capture_features, save_features
from quanta.runtime import ResidentModel
from quanta.tokenizer import KimiTokenizer

ART = "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64"
OUT = "/Users/pmrj/models/kimi_eagle/features.safetensors"
REPO = Path("/Users/pmrj/Environment/quant/finally_quanta")
LAYERS = (10, 30, 50)  # low / mid / high of 61 decoder layers (EAGLE-3 fuses three)


def _corpus(tok: KimiTokenizer, n_tokens: int) -> list[int]:
    files = (sorted(REPO.glob("src/quanta/**/*.py")) + sorted(REPO.glob("parity/*.py"))
             + [REPO / "INITIAL_PROMPT.md", REPO / "CLAUDE.md"])
    text = "\n\n".join(p.read_text() for p in files if p.exists())
    return tok.encode(text, add_bos=True)[:n_tokens]


def run() -> None:
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    mx.set_wired_limit(int(490 * 1024**3))
    t0 = time.perf_counter()
    rm = ResidentModel(ART)
    tok = KimiTokenizer(ART, bos_id=rm.cfg.bos_token_id)
    ids = _corpus(tok, n_tokens)
    print(f"corpus: {len(ids)} tokens | capture layers {LAYERS}", flush=True)
    feat3, ins, tgts = capture_features(rm, ids, LAYERS, chunk=2048)
    print(f"feat3 {feat3.shape} {feat3.dtype} | in {tuple(ins.shape)} | tgt {tuple(tgts.shape)}", flush=True)
    save_features(OUT, feat3, ins, tgts, LAYERS)
    print(f"saved {OUT} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    run()
