"""Pure-Python block bookkeeping for the quanta paged KV cache.

No tensor data lives here — only allocation / ref-count / LRU-free-list / prefix-hash logic. The KV
*data* tensors live in :mod:`quanta.paged.paged_kv_cache`. This module is reimplemented from the
design of oMLX's ``cache/paged_cache.py`` (block pool + doubly-linked free queue + content-addressed
prefix blocks + reference counting), **mirrored, not imported** — quanta keeps mlx-lm and the oMLX
serving stack off its runtime path (engineering rule 5). Every loop here is coarse block-level
accounting at an IO/bookkeeping boundary (rule 3 permits bounded non-hot loops there); there is no
per-token / per-hidden Python loop.

One :class:`BlockAllocator` instance owns the blocks for **one** paged layer. The paged manager holds
``num_layers`` allocators; layers allocate independently (a block id is a row in *that* layer's data
pool). Prefix sharing is content-addressed: a full block's chain hash (depends on its token ids and
its parent block's hash, exactly like vLLM) maps to a resident block that later sequences re-reference
(ref-count++) instead of recomputing. Partial (not-yet-full) blocks are never hashed and never shared,
so they are always private — copy-on-write only ever has to clone a single partial tail block.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

BlockHash = bytes
_ROOT_SEED = b"quanta-paged-root"


def compute_block_hash(parent_hash: Optional[BlockHash], token_ids: tuple[int, ...],
                       *, model_name: str = "") -> BlockHash:
    """Content hash for one full block: sha256 over the parent block's hash + the model name + this
    block's token ids. Chaining on the parent makes the hash position-aware (block *k* only matches
    when blocks *0..k-1* matched too), so a cross-turn / cross-request prefix re-references the exact
    same chain of blocks. ``token_ids`` MUST be the full ``block_size`` ids of a complete block."""
    h = hashlib.sha256()
    h.update(parent_hash if parent_hash is not None else _ROOT_SEED)
    h.update(model_name.encode("utf-8"))
    h.update(b"|")
    # bounded loop over one block's ids (<= block_size) — accounting boundary, not a hot path.
    for t in token_ids:
        h.update(int(t).to_bytes(8, "little", signed=False))
    return h.digest()


@dataclass
class CacheBlock:
    """Metadata for one physical block (one row in a layer's data pool). Data is stored separately,
    indexed by ``block_id``; this is the bookkeeping record only (mirrors oMLX ``CacheBlock``)."""

    block_id: int
    ref_count: int = 0
    block_hash: Optional[BlockHash] = None
    token_count: int = 0
    prev_free: Optional["CacheBlock"] = None
    next_free: Optional["CacheBlock"] = None

    def is_full(self, block_size: int) -> bool:
        return self.token_count >= block_size

    def is_shared(self) -> bool:
        return self.ref_count > 1

    def reset_hash(self) -> None:
        self.block_hash = None


class FreeBlockQueue:
    """O(1) doubly-linked LRU free list (sentinel head/tail). ``popleft`` returns the *oldest* freed
    block (LRU eviction order); ``append`` pushes a newly-freed block to the back; ``remove`` unlinks
    an arbitrary block in O(1) (used to resurrect a still-hashed free block on a prefix hit)."""

    def __init__(self, blocks: list[CacheBlock]) -> None:
        self._head = CacheBlock(block_id=-1)  # sentinel
        self._tail = CacheBlock(block_id=-1)  # sentinel
        self._head.next_free = self._tail
        self._tail.prev_free = self._head
        self._len = 0
        for b in blocks:
            self.append(b)

    def __len__(self) -> int:
        return self._len

    def append(self, block: CacheBlock) -> None:
        last = self._tail.prev_free
        last.next_free = block
        block.prev_free = last
        block.next_free = self._tail
        self._tail.prev_free = block
        self._len += 1

    def popleft(self) -> CacheBlock:
        if self._len == 0:
            raise RuntimeError("paged KV: out of free blocks (raise max_blocks / KV budget)")
        block = self._head.next_free
        self.remove(block)
        return block

    def remove(self, block: CacheBlock) -> None:
        prev, nxt = block.prev_free, block.next_free
        if prev is None or nxt is None:
            raise RuntimeError("paged KV: removing a block not in the free queue")
        prev.next_free = nxt
        nxt.prev_free = prev
        block.prev_free = block.next_free = None
        self._len -= 1


class BlockAllocator:
    """Owns the blocks for one paged layer: a fixed pool of ``num_blocks`` ids, the LRU free queue,
    and the content-hash -> block map for prefix reuse. The data pool (same ``num_blocks`` rows) is
    held by the manager and indexed by ``block_id``; COW row-copies are performed there in response
    to :meth:`prepare_write` returning a (src, dst) signal."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks {num_blocks} < 1")
        self.num_blocks = num_blocks
        self._blocks = [CacheBlock(block_id=i) for i in range(num_blocks)]
        self._free = FreeBlockQueue(list(self._blocks))
        self._hash_to_block: dict[BlockHash, CacheBlock] = {}

    # --- block access --------------------------------------------------------
    def block(self, block_id: int) -> CacheBlock:
        return self._blocks[block_id]

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_cached(self) -> int:
        return len(self._hash_to_block)

    # --- allocation ----------------------------------------------------------
    def alloc(self) -> CacheBlock:
        """Take a fresh block for new (uncached) tokens. Pops the LRU free block; if that block was
        still serving as a cached prefix block, evict it from the hash map first (rule 6: a block can
        never be live under two identities)."""
        block = self._free.popleft()
        if block.block_hash is not None:
            self._hash_to_block.pop(block.block_hash, None)
            block.reset_hash()
        block.ref_count = 1
        block.token_count = 0
        return block

    def get_cached(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        """Look up a resident full block by content hash (prefix hit). Resurrects it from the free
        queue if it was idle, increments ref-count, returns it; ``None`` on miss."""
        block = self._hash_to_block.get(block_hash)
        if block is None:
            return None
        if block.ref_count == 0:
            self._free.remove(block)  # idle-but-cached -> becomes live again
        block.ref_count += 1
        return block

    def incref(self, block: CacheBlock) -> None:
        block.ref_count += 1

    def free(self, block: CacheBlock) -> None:
        """Drop one reference. At zero the block returns to the free queue but KEEPS its hash, so it
        stays a reusable cached-prefix block until it is actually repurposed by :meth:`alloc`."""
        if block.ref_count <= 0:
            raise RuntimeError(f"paged KV: double-free of block {block.block_id}")
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free.append(block)

    def register_full(self, block: CacheBlock, block_hash: BlockHash, token_count: int) -> None:
        """Content-address a block that just filled, so later prefixes can reuse it. Idempotent on a
        re-seen hash (keeps the first resident block; the duplicate is left to be freed by the caller)."""
        block.block_hash = block_hash
        block.token_count = token_count
        self._hash_to_block.setdefault(block_hash, block)
