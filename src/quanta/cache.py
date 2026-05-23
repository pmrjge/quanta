"""MLA KV cache — stores only the compressed latent, not materialized per-head K/V.

Per layer we keep the post-layernorm latent ``c_kv`` ``[B, S, kv_lora]`` and the
roped ``k_pe`` ``[B, 1, S, rope]`` (single MQA head). That's ~576 floats/token/layer
(vs ~16k for full K/V) — the property that makes MLA long-context and prefix caching
cheap. ``k_nope``/``value`` are reconstructed from ``c_kv`` on use (expanded path) or
attended directly (absorbed path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx


class MLACache:
    def __init__(self) -> None:
        self.c_kv: mx.array | None = None  # [B, S, kv_lora]
        self.k_pe: mx.array | None = None  # [B, 1, S, rope] (roped)

    @property
    def offset(self) -> int:
        """Number of positions already cached (the next token's absolute index)."""
        return 0 if self.c_kv is None else self.c_kv.shape[1]

    def update(self, c_kv_new: mx.array, k_pe_new: mx.array) -> tuple[mx.array, mx.array]:
        if self.c_kv is None:
            self.c_kv, self.k_pe = c_kv_new, k_pe_new
        else:
            self.c_kv = mx.concatenate([self.c_kv, c_kv_new], axis=1)
            self.k_pe = mx.concatenate([self.k_pe, k_pe_new], axis=2)
        return self.c_kv, self.k_pe


def prefix_key(token_ids: list[int]) -> str:
    """Stable short key for a prefix's token ids (use as the on-disk cache filename)."""
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:16]


def save_caches(path: str | Path, caches: list[MLACache]) -> None:
    """Persist per-layer MLA latents (opt-in; for cross-session prefix reuse)."""
    flat: dict[str, mx.array] = {}
    for i, c in enumerate(caches):
        if c.c_kv is None:
            raise ValueError(f"cache {i} is empty")
        flat[f"{i}.c_kv"] = c.c_kv
        flat[f"{i}.k_pe"] = c.k_pe
    mx.save_safetensors(str(path), flat)


def load_caches(path: str | Path) -> list[MLACache]:
    flat = mx.load(str(path))
    n = 1 + max(int(k.split(".")[0]) for k in flat)
    caches = []
    for i in range(n):
        c = MLACache()
        c.c_kv, c.k_pe = flat[f"{i}.c_kv"], flat[f"{i}.k_pe"]
        caches.append(c)
    return caches
