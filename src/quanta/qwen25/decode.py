"""Per-layer KV cache state for incremental Qwen2.5-14B-Instruct-1M decode.

The dense pure-attention architecture means decode state is just a list of per-layer
:class:`~quanta.qwen25.attention.KVCache` — no SSM state, no compressor pool, no indexer top-k.
:class:`Qwen25Cache` is the slim wrapper the runtime + generator + (future) spec-decoder pass
around; it stays in lock-step with :class:`~quanta.qwen25.attention.Qwen25Attention.__call__`
(every layer's KV cache grows by ``T`` per forward pass).

Mirrors the simpler subset of :mod:`quanta.qwen35.decode` (full-attention layers only) so the
serving stack can be wired up the same way.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen25.attention import KVCache
from quanta.qwen25.config import Qwen25Config


class Qwen25Cache:
    """Decode-side cache for a Qwen2.5 resident model: one ``KVCache`` per decoder layer.

    Constructed empty and grown by the model's forward pass. Supports:

    * ``replicate(B)`` — zero-copy fan-out to ``B`` parallel branches (for tree spec-decode /
      batched verify; matches the contract :mod:`quanta.spec` expects).
    * ``truncate(length)`` — drop the last ``offset - length`` cached positions on every layer
      (lossless spec-decode rollback after a rejected draft).
    * ``offset`` — current cached sequence length (consistent across layers).
    """

    def __init__(self, cfg: Qwen25Config, *, quantized: bool = False, group_size: int = 64) -> None:
        self.cfg = cfg
        self.quantized = quantized
        self.group_size = group_size
        self.layers: list[KVCache] = [
            KVCache(quantized=quantized, group_size=group_size)
            for _ in range(cfg.num_hidden_layers)
        ]

    @property
    def offset(self) -> int:
        """Current cached length. All layers grow in lock-step, so layer 0 is authoritative."""
        return self.layers[0].offset

    def as_list(self) -> list[KVCache]:
        """The underlying per-layer KV caches in layer order — pass straight to ``Qwen25Model``."""
        return self.layers

    def truncate(self, length: int) -> None:
        """Lossless rollback: drop everything past ``length`` on every layer (spec-decode reject).

        Re-slicing along axis 2 mirrors :class:`quanta.qwen35.decode.Qwen35Cache.truncate`; ``length``
        ≤ current offset.
        """
        if length < 0 or length > self.offset:
            raise ValueError(f"truncate length {length} not in [0, {self.offset}]")
        if length == self.offset:
            return
        for c in self.layers:
            if not c.quantized:
                if c.k is not None:
                    c.k = c.k[:, :, :length, :]
                    c.v = c.v[:, :, :length, :]
            else:
                if c.k_q is not None:
                    c.k_q = c.k_q[:, :, :length, :]
                    c.k_s = c.k_s[:, :, :length, :]
                    c.k_b = c.k_b[:, :, :length, :]
                    c.v_q = c.v_q[:, :, :length, :]
                    c.v_s = c.v_s[:, :, :length, :]
                    c.v_b = c.v_b[:, :, :length, :]

    def replicate(self, batch_size: int) -> "Qwen25Cache":
        """Zero-copy fan-out for tree spec-decode batched verify.

        Returns a new :class:`Qwen25Cache` whose per-layer KVs *broadcast* the current state across
        ``batch_size`` rows. MLX arrays are immutable so the replicated rows share storage with the
        source; a subsequent ``update`` on the replica creates new concatenated arrays per branch.
        """
        new = Qwen25Cache(self.cfg, quantized=self.quantized, group_size=self.group_size)
        for i, c in enumerate(self.layers):
            r = new.layers[i]
            if not c.quantized:
                if c.k is not None:
                    r.k = mx.broadcast_to(c.k, (batch_size,) + c.k.shape[1:])
                    r.v = mx.broadcast_to(c.v, (batch_size,) + c.v.shape[1:])
            else:
                if c.k_q is not None:
                    r.k_q = mx.broadcast_to(c.k_q, (batch_size,) + c.k_q.shape[1:])
                    r.k_s = mx.broadcast_to(c.k_s, (batch_size,) + c.k_s.shape[1:])
                    r.k_b = mx.broadcast_to(c.k_b, (batch_size,) + c.k_b.shape[1:])
                    r.v_q = mx.broadcast_to(c.v_q, (batch_size,) + c.v_q.shape[1:])
                    r.v_s = mx.broadcast_to(c.v_s, (batch_size,) + c.v_s.shape[1:])
                    r.v_b = mx.broadcast_to(c.v_b, (batch_size,) + c.v_b.shape[1:])
        return new
