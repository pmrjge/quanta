"""MLA KV cache — stores only the compressed latent, not materialized per-head K/V.

Per layer we keep the post-layernorm latent ``c_kv`` ``[B, S, kv_lora]`` and the roped ``k_pe``
``[B, 1, S, rope]`` (single MQA head) — ~576 floats/token/layer (vs ~16k for full K/V), the
property that makes MLA long-context and prefix caching cheap.

Optional **int8** latent (``quantized=True``): ``c_kv`` is stored as affine int8, per-token,
group-128 (``k_pe`` stays bf16 — only 64 dims, RoPE-sensitive). ~1.9× on the resident/persisted
latent. Lossy, so it is **off by default** until the teacher-forced ppl gate proves the e2e
delta is negligible (rule-4). Per-token quantization makes incremental decode bit-identical to
bulk prefill. Reads dequantize to bf16 so the attention path is unchanged; a chunked
``quantized_matmul`` absorbed path (to also cut the decode *peak* at 1M context) is a perf
follow-up that does not change the int8 values or the ppl.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx

_BITS = 8


class MLACache:
    def __init__(self, quantized: bool = False, group_size: int = 128) -> None:
        self.quantized = quantized
        self.group_size = group_size
        self.c_kv: mx.array | None = None     # bf16 [B,S,kv_lora]  OR int8 codes [B,S,kv_lora/4]
        self.c_scales: mx.array | None = None  # [B,S,kv_lora/group] (quantized only)
        self.c_biases: mx.array | None = None
        self.k_pe: mx.array | None = None      # [B,1,S,rope] (roped, always bf16)

    @property
    def offset(self) -> int:
        """Number of positions already cached (the seq axis is preserved by per-token quant)."""
        return 0 if self.c_kv is None else self.c_kv.shape[1]

    def _dequant(self) -> mx.array:
        return mx.dequantize(self.c_kv, self.c_scales, self.c_biases,
                             group_size=self.group_size, bits=_BITS)

    def update(self, c_kv_new: mx.array, k_pe_new: mx.array) -> tuple[mx.array, mx.array]:
        """Append the new tokens; return the full ``(c_kv, k_pe)``. With ``quantized`` the stored
        latent is int8 (per-token) and the returned ``c_kv`` is its bf16 dequant."""
        self.k_pe = k_pe_new if self.k_pe is None else mx.concatenate([self.k_pe, k_pe_new], axis=2)
        if not self.quantized:
            self.c_kv = c_kv_new if self.c_kv is None else mx.concatenate([self.c_kv, c_kv_new], axis=1)
            return self.c_kv, self.k_pe

        b, m, lora = c_kv_new.shape  # quantize per token (independent rows) → decode==prefill exactly
        q, s, bi = mx.quantize(c_kv_new.reshape(b * m, lora), group_size=self.group_size, bits=_BITS)
        q, s, bi = q.reshape(b, m, -1), s.reshape(b, m, -1), bi.reshape(b, m, -1)
        if self.c_kv is None:
            self.c_kv, self.c_scales, self.c_biases = q, s, bi
        else:
            self.c_kv = mx.concatenate([self.c_kv, q], axis=1)
            self.c_scales = mx.concatenate([self.c_scales, s], axis=1)
            self.c_biases = mx.concatenate([self.c_biases, bi], axis=1)
        bs, sq, lo4 = self.c_kv.shape
        cur = mx.dequantize(self.c_kv.reshape(bs * sq, lo4), self.c_scales.reshape(bs * sq, -1),
                            self.c_biases.reshape(bs * sq, -1), group_size=self.group_size, bits=_BITS)
        return cur.reshape(bs, sq, -1).astype(k_pe_new.dtype), self.k_pe


def prefix_key(token_ids: list[int]) -> str:
    """Stable short key for a prefix's token ids (use as the on-disk cache filename)."""
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:16]


def save_caches(path: str | Path, caches: list[MLACache]) -> None:
    """Persist per-layer MLA latents (opt-in; for cross-session prefix reuse). int8 caches save
    the codes + scales + biases (~1.9× smaller on disk)."""
    flat: dict[str, mx.array] = {}
    for i, c in enumerate(caches):
        if c.c_kv is None:
            raise ValueError(f"cache {i} is empty")
        flat[f"{i}.c_kv"], flat[f"{i}.k_pe"] = c.c_kv, c.k_pe
        if c.quantized:
            flat[f"{i}.c_scales"], flat[f"{i}.c_biases"] = c.c_scales, c.c_biases
    mx.save_safetensors(str(path), flat)


def load_caches(path: str | Path) -> list[MLACache]:
    flat = mx.load(str(path))
    n = 1 + max(int(k.split(".")[0]) for k in flat)
    caches = []
    for i in range(n):
        quant = f"{i}.c_scales" in flat
        c = MLACache(quantized=quant)
        c.c_kv, c.k_pe = flat[f"{i}.c_kv"], flat[f"{i}.k_pe"]
        if quant:
            c.c_scales, c.c_biases = flat[f"{i}.c_scales"], flat[f"{i}.c_biases"]
        caches.append(c)
    return caches
