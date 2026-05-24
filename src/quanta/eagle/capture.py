"""EAGLE-3 feature capture from the quantized (resident) Kimi target.

Runs a teacher-forced forward over a token corpus in chunks, and per position records:
  * ``feat3`` — the concatenated low/mid/high decoder-layer hidden states ``[N, 3H]`` (the drafter's
    input feature), via ``ResidentModel(..., capture_layers=...)``;
  * ``in_tokens`` — the input token at that position (for the frozen embedding);
  * ``targets`` — the **target model's own argmax next token** (``argmax`` of its logits), so the
    drafter is trained to predict what the *quantized target* would produce (best alignment for the
    lossless verify step), not merely the corpus text.

Each chunk is an independent teacher-forced segment (context within the chunk). Memory-disciplined:
chunk-by-chunk, logits discarded after argmax. EAGLE is trained against the **quantized** model, so
capture uses the resident artifact, not the bf16 source.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def capture_features(
    model, token_ids, layers: tuple[int, ...], *, chunk: int = 2048,
) -> tuple[mx.array, mx.array, mx.array]:
    """``(feat3 [N,3H] bf16, in_tokens [N] int32, targets [N] int32)`` over ``token_ids``."""
    feats: list[mx.array] = []
    ins: list[mx.array] = []
    tgts: list[mx.array] = []
    n = len(token_ids)
    for c0 in range(0, n, chunk):
        seg = mx.array(token_ids[c0:c0 + chunk], dtype=mx.int32)
        if seg.shape[0] < 1:
            break
        logits, caps = model(seg, caches=None, sparse=None, absorbed=False, capture_layers=layers)
        f3 = mx.concatenate([caps[i] for i in layers], axis=-1)  # [T, 3H]
        tgt = mx.argmax(logits[0], axis=-1).astype(mx.int32)     # [T] target's predicted next token
        mx.eval(f3, tgt)
        feats.append(f3.astype(mx.bfloat16))
        ins.append(seg)
        tgts.append(tgt)
    return mx.concatenate(feats, 0), mx.concatenate(ins, 0), mx.concatenate(tgts, 0)


def save_features(path: str | Path, feat3: mx.array, in_tokens: mx.array, targets: mx.array,
                  layers: tuple[int, ...]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {"feat3": feat3, "in_tokens": in_tokens, "targets": targets,
                                    "layers": mx.array(layers, dtype=mx.int32)})


def load_features(path: str | Path) -> dict[str, mx.array]:
    return mx.load(str(path))
