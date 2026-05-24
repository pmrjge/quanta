"""Debug: log MiMo residual-stream RMS layer-by-layer to localize the e2e incoherence (#60).

All per-layer components match HF (~1e-6) and masks match transformers exactly, yet the full forward
is near-random (ppl ~95760). This streams the first N layers and prints RMS / max-abs / finiteness of
the residual stream after each, to find the layer where it diverges.

    uv run --with tokenizers python -m parity.mimo_debug_rms [N] [L]
"""

from __future__ import annotations

import sys

import mlx.core as mx
from tokenizers import Tokenizer

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.loader import MiMoSourceCheckpoint
from quanta.mimo.reference import _build_layer, full_causal_mask, sliding_window_mask

ART = "/Users/pmrj/models/MiMo-V2.5"
PROSE = ("The history of writing traces the development of expressing language by systems of "
         "markings. True writing encodes a linguistic utterance so that another reader can "
         "reconstruct the exact words written down, distinguishing it from proto-writing.")


def _stat(h: mx.array) -> str:
    f = h.astype(mx.float32)
    rms = float(mx.sqrt(mx.mean(f * f)).item())
    mab = float(mx.max(mx.abs(f)).item())
    fin = bool(mx.all(mx.isfinite(f)).item())
    return f"rms={rms:8.3f} max={mab:9.2f} finite={fin}"


def run() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cfg = MiMoV2Config.from_pretrained(ART)
    tok = Tokenizer.from_file(f"{ART}/tokenizer.json")
    ids = [cfg.bos_token_id] + tok.encode(PROSE).ids
    if len(sys.argv) > 2:
        ids = ids[: int(sys.argv[2])]
    ck = MiMoSourceCheckpoint(ART, cfg)
    embed = ck.read("model.embed_tokens.weight").astype(mx.bfloat16)
    h = embed[mx.array(ids)][None]
    mx.eval(h)
    ck.release()
    length = h.shape[1]
    fm = full_causal_mask(length, mx.bfloat16)
    sm = sliding_window_mask(length, cfg.sliding_window, mx.bfloat16)
    print(f"tokens={length}  embed: {_stat(h)}", flush=True)
    for i in range(n):
        layer = _build_layer(cfg, ck, i)
        h = layer(h, sm if cfg.is_swa(i) else fm, offset=0)
        mx.eval(h)
        ck.release()
        del layer
        kind = ("swa" if cfg.is_swa(i) else "full") + ("+moe" if cfg.is_moe(i) else "+dense")
        print(f"L{i:2d} {kind:9s}: {_stat(h)}", flush=True)


if __name__ == "__main__":
    run()
