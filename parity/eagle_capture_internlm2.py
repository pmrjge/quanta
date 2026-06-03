"""Capture EAGLE-3 features from the resident InternLM2.5-7B int8g64 bake (deferred — GPU job).

Mirrors :mod:`parity.eagle_capture_minimax` for InternLM2.5. Run AFTER the int8g64 bake exists and
**ALONE** (one model resident — the OOM-reboot rule). The same per-2048-block teacher-forced capture
as the MiniMax/Kimi versions; the forward callable wires InternLM2's signature
(:func:`quanta.internlm2.eagle.internlm2_capture_forward`, ``caches=None`` prefill, which records each
capture-layer's post-layer residual ``[T, H]`` and the target's own argmax label).

The corpus is tokenized HERE in InternLM2's own SentencePiece space — the shared
``corpus_mix.safetensors`` ids are in a *different* model's vocab, which would make both the captured
features and the target-argmax labels meaningless. One BOS per source document; capped at
``MAX_TOKENS`` so the capture cost / peak feature memory stays bounded.

    uv run --with sentencepiece python -m parity.eagle_capture_internlm2
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from quanta.eagle.capture import capture_features_to_shards_fn
from quanta.internlm2.eagle import DEFAULT_CAPTURE_LAYERS, internlm2_capture_forward
from quanta.internlm2.runtime import InternLM2ResidentModel
from quanta.internlm2.tokenizer import InternLM2Tokenizer

ART = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
RAW = [
    "/Users/pmrj/models/corpus/raw_coding.txt",
    "/Users/pmrj/models/corpus/raw_math.txt",
    "/Users/pmrj/models/corpus/raw_research.txt",
]
OUT = "/Users/pmrj/models/internlm2_eagle/features_int8g64"
LAYERS = DEFAULT_CAPTURE_LAYERS                     # low / mid / high of 32 (8, 16, 24)
MAX_TOKENS = 2_000_000                              # bound capture cost / feature memory (corpus ~1M)


def _tokenize_corpus(tok: InternLM2Tokenizer, paths: list[str], max_tokens: int) -> mx.array:
    """Concatenate the InternLM2-tokenized raw corpus (one BOS per document) into ``[N]`` int32 ids,
    capped at ``max_tokens``. Tokenizing in InternLM2's vocab is mandatory — the features are its
    activations and the labels are its own argmax, so a foreign tokenizer would poison both."""
    ids: list[int] = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        ids.extend(tok.encode(text, add_bos=True))
        if len(ids) >= max_tokens:
            break
    return mx.array(ids[:max_tokens], dtype=mx.int32)


def run() -> None:
    mx.set_wired_limit(int(64 * 1024**3))           # 7B int8 ~9 GiB resident; generous ceiling
    mx.set_cache_limit(4 * 1024**3)
    t0 = time.perf_counter()
    rm = InternLM2ResidentModel(ART, packed=True)
    tok = InternLM2Tokenizer.from_pretrained(ART)
    ids = _tokenize_corpus(tok, RAW, MAX_TOKENS)
    fwd = internlm2_capture_forward(rm)
    print(f"corpus: {ids.shape[0]} tokens | base {Path(ART).name} | capture layers {LAYERS}",
          flush=True)
    info = capture_features_to_shards_fn(fwd, ids, LAYERS, OUT, chunk=2048, shard_tokens=131072)
    print(f"captured {info['total_tokens']} tok -> {len(info['shards'])} shards in "
          f"{(time.perf_counter() - t0) / 60:.1f} min | {OUT}", flush=True)


if __name__ == "__main__":
    run()
