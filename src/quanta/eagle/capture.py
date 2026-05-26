"""EAGLE-3 feature capture from a quantized resident target — Kimi entry points + model-agnostic
``_fn`` variants.

Runs a teacher-forced forward over a token corpus in chunks, and per position records:
  * ``feat3`` — the concatenated capture-layer hidden states ``[N, n_layers * H]`` (the drafter's
    input feature), via ``ResidentModel(..., capture_layers=...)``;
  * ``in_tokens`` — the input token at that position (for the frozen embedding);
  * ``targets`` — the **target model's own argmax next token** (``argmax`` of its logits), so the
    drafter is trained to predict what the *quantized target* would produce (best alignment for the
    lossless verify step), not merely the corpus text.

Each chunk is an independent teacher-forced segment (context within the chunk). Memory-disciplined:
chunk-by-chunk, logits discarded after argmax. EAGLE is trained against the **quantized** model, so
capture uses the resident artifact, not the bf16 source.

The original Kimi-flavored wrappers (``capture_features`` / ``capture_features_to_shards``) are
preserved for backward-compat; both route through forward-callable variants
(``capture_features_fn`` / ``capture_features_to_shards_fn``) so the same teacher-forced capture now
serves any model whose runtime exposes ``(ids, capture_layers=) -> (logits, caps)``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

ForwardFn = Callable[..., tuple[mx.array, dict[int, mx.array]]]


def _kimi_forward(model) -> ForwardFn:
    """Forward callable for the Kimi runtime — preserves the original sparse/absorbed kwargs."""
    def fwd(ids, capture_layers):
        return model(ids, caches=None, sparse=None, absorbed=False, capture_layers=capture_layers)
    return fwd


def capture_features(
    model, token_ids, layers: tuple[int, ...], *, chunk: int = 2048,
) -> tuple[mx.array, mx.array, mx.array]:
    """``(feat3 [N, n_layers*H] bf16, in_tokens [N] int32, targets [N] int32)`` over ``token_ids`` —
    Kimi-flavored wrapper. Other models use :func:`capture_features_fn`."""
    return capture_features_fn(_kimi_forward(model), token_ids, layers, chunk=chunk)


def capture_features_to_shards(
    model, token_ids, layers: tuple[int, ...], out_dir: str | Path, *,
    chunk: int = 2048, shard_tokens: int = 131072, prefix: str = "feat",
) -> dict:
    """OOM-safe capture for large corpora — Kimi-flavored wrapper. Other models use
    :func:`capture_features_to_shards_fn`."""
    return capture_features_to_shards_fn(_kimi_forward(model), token_ids, layers, out_dir,
                                        chunk=chunk, shard_tokens=shard_tokens, prefix=prefix)


def capture_features_fn(
    forward_fn: ForwardFn, token_ids, layers: tuple[int, ...], *, chunk: int = 2048,
) -> tuple[mx.array, mx.array, mx.array]:
    """``(feat3, in_tokens, targets)`` over ``token_ids`` — model-agnostic. ``forward_fn(ids,
    capture_layers) -> (logits, caps)`` does one teacher-forced chunk forward (no cache)."""
    feats: list[mx.array] = []
    ins: list[mx.array] = []
    tgts: list[mx.array] = []
    n = len(token_ids)
    for c0 in range(0, n, chunk):
        seg = mx.array(token_ids[c0:c0 + chunk], dtype=mx.int32)
        if seg.shape[0] < 1:
            break
        logits, caps = forward_fn(seg, layers)
        f3 = mx.concatenate([caps[i] for i in layers], axis=-1)  # [T, n_layers*H]
        tgt = mx.argmax(logits[0], axis=-1).astype(mx.int32)     # [T] target's predicted next token
        mx.eval(f3, tgt)
        feats.append(f3.astype(mx.bfloat16))
        ins.append(seg)
        tgts.append(tgt)
    return mx.concatenate(feats, 0), mx.concatenate(ins, 0), mx.concatenate(tgts, 0)


def capture_features_to_shards_fn(
    forward_fn: ForwardFn, token_ids, layers: tuple[int, ...], out_dir: str | Path, *,
    chunk: int = 2048, shard_tokens: int = 131072, prefix: str = "feat",
) -> dict:
    """OOM-safe capture for large corpora (1M+ tokens) — model-agnostic. Identical per-chunk
    teacher-forced capture as :func:`capture_features_fn`, but **flushes each ~``shard_tokens``
    group to disk and frees** instead of accumulating the whole corpus in RAM. Peak extra memory is
    one shard on top of the resident model. Returns ``{shards, total_tokens, layers}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats: list[mx.array] = []
    ins: list[mx.array] = []
    tgts: list[mx.array] = []
    paths: list[str] = []
    held = 0
    total = 0
    shard = 0

    def flush() -> None:
        nonlocal held, shard
        if held == 0:
            return
        p = out / f"{prefix}_{shard:04d}.safetensors"
        save_features(p, mx.concatenate(feats, 0), mx.concatenate(ins, 0),
                      mx.concatenate(tgts, 0), layers)
        paths.append(str(p))
        feats.clear()
        ins.clear()
        tgts.clear()
        held = 0
        shard += 1

    n = len(token_ids)
    for c0 in range(0, n, chunk):
        seg = mx.array(token_ids[c0:c0 + chunk], dtype=mx.int32)
        if seg.shape[0] < 1:
            break
        logits, caps = forward_fn(seg, layers)
        f3 = mx.concatenate([caps[i] for i in layers], axis=-1)
        tgt = mx.argmax(logits[0], axis=-1).astype(mx.int32)
        mx.eval(f3, tgt)
        feats.append(f3.astype(mx.bfloat16))
        ins.append(seg)
        tgts.append(tgt)
        held += int(seg.shape[0])
        total += int(seg.shape[0])
        if held >= shard_tokens:
            flush()
    flush()
    return {"shards": paths, "total_tokens": total, "layers": layers}


def save_features(path: str | Path, feat3: mx.array, in_tokens: mx.array, targets: mx.array,
                  layers: tuple[int, ...]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {"feat3": feat3, "in_tokens": in_tokens, "targets": targets,
                                    "layers": mx.array(layers, dtype=mx.int32)})


def load_features(path: str | Path) -> dict[str, mx.array]:
    return mx.load(str(path))


def load_feature_shards(out_dir: str | Path, prefix: str = "feat") -> dict[str, mx.array]:
    """Concatenate sharded features written by :func:`capture_features_to_shards` (or the ``_fn``
    variant) into one ``{feat3, in_tokens, targets, layers}`` dict (train-time only — the target
    model is **not** resident then, so holding the full feature set is safe)."""
    paths = sorted(Path(out_dir).glob(f"{prefix}_*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no shards '{prefix}_*.safetensors' under {out_dir}")
    parts = [mx.load(str(p)) for p in paths]
    return {
        "feat3": mx.concatenate([d["feat3"] for d in parts], 0),
        "in_tokens": mx.concatenate([d["in_tokens"] for d in parts], 0),
        "targets": mx.concatenate([d["targets"] for d in parts], 0),
        "layers": parts[0]["layers"],
    }
