"""Content-addressed snapshots of RECURRENT state at block boundaries — the piece that lets a hybrid
model (Mamba / Gated-DeltaNet) skip a shared prefix during prefill (#152 step 3).

Why this exists. The paged KV cache dedups a shared prompt prefix's **attention** KV. But Nemotron
(40 Mamba layers) and Qwen3.5 (45 GatedDeltaNet layers) interleave recurrent layers whose state at
position *n* depends on **all** tokens ``0..n-1`` — so a hybrid prefill cannot simply skip the prefix
for the attention layers while the recurrent layers reprocess it (one forward processes one position
range through every layer). The way out: the recurrent state at a block boundary is a **deterministic
function of the prefix tokens**, exactly like the attention KV. So it is keyed by the SAME chain hash
(:func:`quanta.paged.block_pool.compute_block_hash`) and restored on a prefix hit — then the forward
processes **only the suffix** through every layer (recurrent layers seeded from the restored boundary
state, attention layers re-referencing the resident prefix KV blocks). The two caches share the chain
hash, so they agree on exactly which whole blocks of prefix are reusable.

Cost. A recurrent snapshot is O(1) per layer (e.g. Mamba ``ssm_state [1,H,N,P]`` + ``conv_state
[1,K-1,Cdim]``), independent of sequence length — far smaller than a block of KV — so snapshotting at
block boundaries is cheap. This store holds **opaque payloads**: the runtime hands over whatever
per-recurrent-layer state it wants restored; this class never interprets it (model-agnostic, rule 5
keeps it mlx-lm-free — it is plain bookkeeping + an :class:`~collections.OrderedDict` LRU).

Boundary convention. Full block ``bi`` covers tokens ``[bi*block_size, (bi+1)*block_size)``; the
snapshot stored at ``hashes[bi]`` is the recurrent state **after** block ``bi`` (absolute position
``(bi+1)*block_size``), so restoring it seeds a forward that resumes at that position.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from quanta.paged.block_pool import BlockHash, compute_block_hash


@dataclass
class RecurrentCacheStats:
    """Metrics for the recurrent boundary cache (folded into the engine's paged stats)."""

    block_size: int = 0
    capacity: int = 0
    stored_boundaries: int = 0   # distinct boundaries resident now
    snapshot_stores: int = 0     # cumulative new boundary snapshots written
    snapshot_hits: int = 0       # cumulative lookups that found a snapshot
    snapshot_misses: int = 0     # cumulative lookups that found nothing
    evictions: int = 0           # cumulative LRU evictions


class RecurrentPrefixCache:
    """LRU map ``block-boundary chain-hash -> opaque recurrent-state payload``.

    The chain hash is computed exactly as the paged KV manager computes it (same
    :func:`compute_block_hash`, same ``model_name``, same ``block_size``), so a boundary the attention
    blocks matched is the boundary this cache is keyed on. Bounded to ``capacity`` distinct boundaries;
    the least-recently-used boundary is evicted when full.
    """

    def __init__(self, *, block_size: int, model_name: str = "", capacity: int = 4096) -> None:
        if block_size < 1:
            raise ValueError(f"block_size {block_size} < 1")
        if capacity < 1:
            raise ValueError(f"capacity {capacity} < 1")
        self.block_size = block_size
        self.model_name = model_name
        self.capacity = capacity
        self._store: OrderedDict[BlockHash, Any] = OrderedDict()  # hash -> payload (LRU)
        self._stats = RecurrentCacheStats(block_size=block_size, capacity=capacity)

    # --- hashing -------------------------------------------------------------
    def _chain_hashes(self, token_ids: list[int]) -> list[BlockHash]:
        """Per-full-block chain hashes for ``token_ids`` (parent-chained, position-aware), identical to
        the paged KV manager's. Length == ``len(token_ids) // block_size``."""
        n_full = len(token_ids) // self.block_size
        parent: BlockHash | None = None
        hashes: list[BlockHash] = []
        for bi in range(n_full):
            chunk = tuple(token_ids[bi * self.block_size:(bi + 1) * self.block_size])
            parent = compute_block_hash(parent, chunk, model_name=self.model_name)
            hashes.append(parent)
        return hashes

    # --- lookup --------------------------------------------------------------
    def lookup_at(self, token_ids: list[int], n_tokens: int) -> Any | None:
        """Snapshot at the EXACT boundary of ``n_tokens`` (must be a multiple of ``block_size``);
        ``None`` if absent. This is the engine entry point — it queries the boundary the paged KV
        manager just matched, so the two caches reuse the identical prefix length."""
        if n_tokens <= 0:
            return None
        if n_tokens % self.block_size:
            raise ValueError(f"n_tokens {n_tokens} not a multiple of block_size {self.block_size}")
        n_blocks = n_tokens // self.block_size
        hashes = self._chain_hashes(token_ids[:n_tokens])
        if n_blocks > len(hashes):
            return None
        h = hashes[n_blocks - 1]
        payload = self._store.get(h)
        if payload is None:
            self._stats.snapshot_misses += 1
            return None
        self._store.move_to_end(h)
        self._stats.snapshot_hits += 1
        return payload

    def match(self, token_ids: list[int]) -> tuple[int, Any | None]:
        """Longest stored full-block boundary that prefixes ``token_ids`` -> ``(n_tokens, payload)``
        (``(0, None)`` on a miss). Provided for callers that want the deepest reusable boundary; the
        engine uses :meth:`lookup_at` to stay locked to the attention match."""
        hashes = self._chain_hashes(token_ids)
        for bi in range(len(hashes) - 1, -1, -1):
            payload = self._store.get(hashes[bi])
            if payload is not None:
                self._store.move_to_end(hashes[bi])
                self._stats.snapshot_hits += 1
                return (bi + 1) * self.block_size, payload
        self._stats.snapshot_misses += 1
        return 0, None

    # --- store ---------------------------------------------------------------
    def store_at(self, token_ids: list[int], n_tokens: int, payload: Any) -> None:
        """Store ``payload`` (recurrent state after the first ``n_tokens``) at that boundary's chain
        hash. ``n_tokens`` must be a positive multiple of ``block_size``. The driver calls this once per
        full-block boundary it crossed in prefill / decode, so any later prefix that reaches the same
        boundary can resume from it. LRU-evicts past ``capacity``; idempotent on a resident boundary."""
        if payload is None:
            return
        if n_tokens <= 0 or n_tokens % self.block_size:
            raise ValueError(f"store_at n_tokens {n_tokens} must be a positive multiple of "
                             f"block_size {self.block_size}")
        n_blocks = n_tokens // self.block_size
        hashes = self._chain_hashes(token_ids[:n_tokens])
        if n_blocks > len(hashes):
            raise ValueError(f"store_at: token_ids has fewer than {n_tokens} tokens")
        h = hashes[n_blocks - 1]
        if h in self._store:
            self._store.move_to_end(h)
            return
        self._store[h] = payload
        self._stats.snapshot_stores += 1
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self._stats.evictions += 1

    def store(self, token_ids: list[int], boundary_payloads: list[Any]) -> None:
        """Record ``boundary_payloads[bi]`` (the recurrent state AFTER full block ``bi``) at that
        block's chain hash. ``boundary_payloads`` may be shorter than the number of full blocks (e.g.
        the runtime only checkpointed the final boundary) — only the boundaries present are stored.
        Re-storing a resident boundary just refreshes its LRU recency (idempotent on value)."""
        if not boundary_payloads:
            return
        hashes = self._chain_hashes(token_ids)
        for bi, payload in enumerate(boundary_payloads):
            if bi >= len(hashes) or payload is None:
                continue
            h = hashes[bi]
            if h in self._store:
                self._store.move_to_end(h)
                continue
            self._store[h] = payload
            self._stats.snapshot_stores += 1
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)  # evict LRU
                self._stats.evictions += 1

    # --- stats ---------------------------------------------------------------
    def get_stats(self) -> RecurrentCacheStats:
        self._stats.stored_boundaries = len(self._store)
        return self._stats
