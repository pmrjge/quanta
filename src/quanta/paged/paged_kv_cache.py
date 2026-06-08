"""Block-paged KV cache with copy-on-write prefix sharing (quanta-native, mlx-lm-free).

The serving win this enables: many concurrent / multi-turn agentic requests that share a common
prompt prefix (system prompt, conversation history) store that prefix's KV **once** and re-reference
it (ref-counted full blocks), instead of one private growing cache per request. It mirrors the design
of oMLX's ``cache/paged_cache.py`` (vLLM-style block pool) but is reimplemented in plain ``mlx.core``
so quanta keeps mlx-lm + the oMLX scheduler off its runtime path (rule 5).

Layout. One :class:`~quanta.paged.block_pool.BlockAllocator` per paged layer; each layer's KV data
lives in pre-allocated **pooled tensors** ``[max_blocks, block_size, n_kv, C]`` (one per int8 component
``k_q/k_s/k_b/v_q/v_s/v_b``, or ``k/v`` in bf16 mode), lazily sized from the first token's dtype/shape
so the round-trip matches the discrete cache exactly. A token's KV is written with an MLX slice-assign
into ``pool[block_id, intra]``; a sequence's stream is read back with a single vectorized ``mx.take``
over its block ids (no per-token Python loop — rule 3; the only loops are coarse, over layers / over
the few blocks a write spans).

Parity foundation. ``cache_quant.quantize_last_axis`` packs int8 along ``head_dim`` (the last axis);
blocks cut the **seq axis**. The axes are orthogonal, so a block always holds whole per-token quant
records — gather and copy-on-write never split a quant group, and the dequantized stream is
**bit-identical** to :class:`quanta.nemotron.attention.KVCache` fed the same tokens.

Protocol (why it is not a pure ``KVCache.update`` drop-in). Prefix hashing is over **token ids**, which
``update(k, v)`` never sees. So the driver (a batched session) brackets each step:

    n = mgr.match_prefix(seq, prompt_ids)        # reuse resident prefix blocks (ref++), returns n
    mgr.advance(seq, prompt_ids[n:])             # record ids + bump length for the uncached suffix
    for layer: k_full, v_full = view.update(k, v)  # write this layer's KV + return the full stream
    mgr.commit(seq)                              # content-hash any block that just filled

``view.update`` itself is signature-compatible with ``KVCache.update`` (write + return the full bf16
stream); ``advance`` / ``commit`` are the two extra session calls around the unchanged per-layer forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mlx.core as mx

from quanta.cache_quant import BITS, dequantize_last_axis, quantize_last_axis
from quanta.paged.block_pool import BlockAllocator, CacheBlock, compute_block_hash

_QUANT_COMPS = ("k_q", "k_s", "k_b", "v_q", "v_s", "v_b")
_BF16_COMPS = ("k", "v")
# single-stream (one latent KV per token, MQA — DSV4's compressed latent). Same block machinery,
# one logical stream instead of a k/v pair (#175): half the components, half the bytes of a k/v pair.
_SINGLE_QUANT_COMPS = ("kv_q", "kv_s", "kv_b")
_SINGLE_BF16_COMPS = ("kv",)


@dataclass
class PagedCacheStats:
    """Cache metrics, shaped to map onto oMLX ``BaseCacheStats`` for ``engine.get_cache_stats()``."""

    block_size: int = 0
    max_blocks_per_layer: int = 0
    num_layers: int = 0
    prefix_hit_blocks: int = 0       # full blocks served from a prefix match (cumulative)
    prefix_hit_tokens: int = 0       # tokens those blocks covered (the prefill saved)
    prefix_miss_blocks: int = 0      # full blocks that had to be freshly hashed (cumulative)
    cow_copies: int = 0              # copy-on-write block clones (cumulative)
    allocated_blocks: int = 0        # live (ref>0) blocks summed over layers, at snapshot time
    cached_blocks: int = 0           # resident content-addressed blocks summed over layers

    @property
    def hit_rate(self) -> float:
        seen = self.prefix_hit_blocks + self.prefix_miss_blocks
        return self.prefix_hit_blocks / seen if seen else 0.0

    @property
    def utilization(self) -> float:
        cap = self.max_blocks_per_layer * self.num_layers
        return self.allocated_blocks / cap if cap else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hit_rate"] = self.hit_rate
        d["utilization"] = self.utilization
        return d


@dataclass
class SeqHandle:
    """Per-sequence paged state: the token ids stored, and one block table (ordered blocks) + a
    written-position cursor per layer. ``length`` is the logical token count (shared across layers)."""

    seq_id: int
    token_ids: list[int] = field(default_factory=list)
    length: int = 0
    block_tables: list[list[CacheBlock]] = field(default_factory=list)
    n_written: list[int] = field(default_factory=list)


class PagedKVCacheManager:
    """Owns the paged KV for one model instance (all paged layers) across all in-flight sequences.

    ``(bits, group_size, quantized)`` are threaded from the model's own ``KVCache`` config — never
    hardcoded (rule 6: a wrong width silently mis-decodes). ``block_size`` is independent of
    ``group_size`` (the quant groups are on ``head_dim``, orthogonal to the seq axis blocks cut), so
    it is chosen purely for prefix-match granularity vs metadata overhead.
    """

    def __init__(self, *, num_layers: int, block_size: int = 32, max_blocks: int = 4096,
                 group_size: int = 128, bits: int = BITS, quantized: bool = True,
                 model_name: str = "", single_stream: bool = False) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers {num_layers} < 1")
        if block_size < 1:
            raise ValueError(f"block_size {block_size} < 1")
        self.num_layers = num_layers
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.group_size = group_size
        self.bits = bits
        self.quantized = quantized
        self.model_name = model_name
        # ``single_stream`` selects DSV4's one-latent-per-token codec (``write_one``/``gather_one``/
        # ``view_one``) over the k/v pair codec; the block pool / hashing / COW / LRU are identical.
        self.single_stream = single_stream
        if single_stream:
            self._comps = _SINGLE_QUANT_COMPS if quantized else _SINGLE_BF16_COMPS
        else:
            self._comps = _QUANT_COMPS if quantized else _BF16_COMPS
        self._allocs = [BlockAllocator(max_blocks) for _ in range(num_layers)]
        # per layer: {component_name: pooled tensor [max_blocks, block_size, *per_token_shape]} —
        # lazily allocated on first write so dtypes/shapes match mx.quantize exactly.
        self._pools: list[dict[str, mx.array] | None] = [None] * num_layers
        self._seqs: dict[int, SeqHandle] = {}
        self._next_id = 0
        self._stats = PagedCacheStats(block_size=block_size, max_blocks_per_layer=max_blocks,
                                      num_layers=num_layers)

    # --- sequence lifecycle --------------------------------------------------
    def new_sequence(self) -> SeqHandle:
        seq = SeqHandle(seq_id=self._next_id,
                        block_tables=[[] for _ in range(self.num_layers)],
                        n_written=[0] * self.num_layers)
        self._seqs[seq.seq_id] = seq
        self._next_id += 1
        return seq

    def free(self, seq: SeqHandle) -> None:
        """Drop the sequence's references. Full prefix blocks survive at ref 0 (still hashed) so a
        later request can re-hit them; they are only repurposed when the pool needs a fresh block."""
        for layer in range(self.num_layers):
            alloc = self._allocs[layer]
            for blk in seq.block_tables[layer]:
                alloc.free(blk)
            seq.block_tables[layer] = []
            seq.n_written[layer] = 0
        self._seqs.pop(seq.seq_id, None)

    def fork(self, seq: SeqHandle) -> SeqHandle:
        """Branch a sequence: a new handle sharing every block (ref++). A subsequent write to either
        branch's partial tail block triggers copy-on-write. (Used by the COW gate; serving spec-decode
        stays on discrete caches per the #152 scope guard.)"""
        nseq = SeqHandle(seq_id=self._next_id, token_ids=list(seq.token_ids), length=seq.length,
                         block_tables=[list(bt) for bt in seq.block_tables],
                         n_written=list(seq.n_written))
        self._seqs[nseq.seq_id] = nseq
        self._next_id += 1
        for layer in range(self.num_layers):
            alloc = self._allocs[layer]
            for blk in nseq.block_tables[layer]:
                alloc.incref(blk)
        return nseq

    def replicate(self, seq: SeqHandle, b: int) -> list[SeqHandle]:
        """Branch ``seq`` into ``b`` independent copy-on-write siblings — the sequence-level, paged
        analog of the discrete ``DSV4Cache.replicate(B)`` the batched tree-spec verify uses. Each
        branch shares every block (ref++) until it writes its partial tail, at which point COW isolates
        the divergence, so the original prefix stays read-only and the B paths cost one shared block
        set plus only their per-path tails.

        This is the CORRECT level for paged branching: one :meth:`fork` clones ALL layers of the
        sequence together (they share one block table per layer). The per-layer
        :meth:`PagedKVCacheView._copy` is the wrong hook — a view sees a single layer and cannot fork
        the shared sequence. Wiring this into the tree-spec verify loop (so serving spec-decode can run
        over paged caches instead of the discrete ones the #152 scope guard mandates today) is the
        deferred #158-160 follow-up; the primitive is gated in ``parity/paged_cache_test.py``."""
        if b < 1:
            raise ValueError(f"replicate(b) requires b >= 1 (got {b})")
        return [self.fork(seq) for _ in range(b)]

    # --- prefix matching -----------------------------------------------------
    def match_prefix(self, seq: SeqHandle, token_ids: list[int]) -> int:
        """Re-reference resident full blocks that match the leading full blocks of ``token_ids`` (in
        EVERY layer), seeding ``seq``'s block tables and returning the number of matched tokens. Only
        whole blocks are content-addressed, so the result is a multiple of ``block_size``."""
        if seq.length != 0:
            raise RuntimeError("match_prefix must run on a fresh sequence")
        n_full = len(token_ids) // self.block_size
        parent: bytes | None = None
        hashes: list[bytes] = []
        for bi in range(n_full):
            chunk = tuple(token_ids[bi * self.block_size:(bi + 1) * self.block_size])
            parent = compute_block_hash(parent, chunk, model_name=self.model_name)
            hashes.append(parent)
        matched = 0
        for bi, h in enumerate(hashes):
            blocks = [self._allocs[layer].get_cached(h) for layer in range(self.num_layers)]
            if any(b is None for b in blocks):
                # partial hit in some layer -> undo the increfs from this block and stop (only a
                # prefix shared by ALL layers is reusable).
                for layer, b in enumerate(blocks):
                    if b is not None:
                        self._allocs[layer].free(b)
                break
            for layer in range(self.num_layers):
                seq.block_tables[layer].append(blocks[layer])
            matched += 1
        n_tokens = matched * self.block_size
        seq.length = n_tokens
        for layer in range(self.num_layers):
            seq.n_written[layer] = n_tokens
        seq.token_ids = list(token_ids[:n_tokens])
        if matched:
            self._stats.prefix_hit_blocks += matched
            self._stats.prefix_hit_tokens += n_tokens
        return n_tokens

    # --- write path ----------------------------------------------------------
    def advance(self, seq: SeqHandle, new_token_ids: list[int]) -> None:
        """Record new token ids and grow the logical length (one call per step, before the layer
        writes). Block allocation happens lazily in :meth:`write` per layer."""
        seq.token_ids.extend(int(t) for t in new_token_ids)
        seq.length += len(new_token_ids)

    def _ensure_pools(self, layer: int, encoded: dict[str, mx.array], n_kv_etc: tuple[int, ...]) -> dict[str, mx.array]:
        pools = self._pools[layer]
        if pools is None:
            pools = {}
            for name, arr in encoded.items():
                # arr is token-major [T, *per_token_shape]; pool is [max_blocks, block_size, *shape].
                per_tok = arr.shape[1:]
                pools[name] = mx.zeros((self.max_blocks, self.block_size, *per_tok), dtype=arr.dtype)
            self._pools[layer] = pools
        return pools

    def _encode(self, k: mx.array, v: mx.array) -> dict[str, mx.array]:
        """Quantize (or pass through) k,v ``[1, n_kv, T, head_dim]`` -> token-major ``[T, ...]`` per
        component, ready to scatter into blocks."""
        if k.shape[0] != 1:
            raise ValueError(f"paged KV is per-sequence: expected batch 1, got {k.shape[0]}")
        if self.quantized:
            kq, ks, kb = quantize_last_axis(k, self.group_size, self.bits)
            vq, vs, vb = quantize_last_axis(v, self.group_size, self.bits)
            raw = {"k_q": kq, "k_s": ks, "k_b": kb, "v_q": vq, "v_s": vs, "v_b": vb}
        else:
            raw = {"k": k, "v": v}
        # [1, n_kv, T, C] -> [T, n_kv, C]
        return {name: arr[0].transpose(1, 0, 2) for name, arr in raw.items()}

    def _encode_one(self, kv: mx.array) -> dict[str, mx.array]:
        """Quantize (or pass through) a single latent stream ``[1, T, head_dim]`` -> token-major
        ``[T, head_dim]`` per component (the single-stream sibling of :meth:`_encode`)."""
        if kv.shape[0] != 1:
            raise ValueError(f"paged latent is per-sequence: expected batch 1, got {kv.shape[0]}")
        if self.quantized:
            q, s, b = quantize_last_axis(kv, self.group_size, self.bits)
            raw = {"kv_q": q, "kv_s": s, "kv_b": b}
        else:
            raw = {"kv": kv}
        return {name: arr[0] for name, arr in raw.items()}             # [1, T, C] -> [T, C]

    def _cow(self, layer: int, block_table: list[CacheBlock], bi: int) -> CacheBlock:
        alloc = self._allocs[layer]
        src = block_table[bi]
        dst = alloc.alloc()
        tc = src.token_count
        pools = self._pools[layer]
        if pools is not None and tc > 0:
            for name, pool in pools.items():
                pool[dst.block_id, :tc] = pool[src.block_id, :tc]
        dst.token_count = tc
        alloc.free(src)
        block_table[bi] = dst
        self._stats.cow_copies += 1
        return dst

    def write(self, seq: SeqHandle, layer: int, k: mx.array, v: mx.array) -> None:
        """Append this layer's KV for ``k.shape[2]`` tokens starting at the layer's write cursor
        ``n_written[layer]`` (which :meth:`advance` must have opened room for in ``length``). Writing a
        sub-range of the opened window is allowed (chunked prefill snapshots recurrent state at block
        boundaries → multiple sub-range writes per advance); successive writes advance the cursor.
        Allocates fresh blocks as the stream crosses block boundaries and clones a shared partial tail
        block (COW) before mutating it."""
        n_write = int(k.shape[2])
        if n_write == 0:
            return
        if v.shape[2] != n_write:
            raise ValueError(f"paged write layer {layer}: k has {n_write} tokens, v has {v.shape[2]}")
        self._write_encoded(seq, layer, self._encode(k, v), n_write)

    def write_one(self, seq: SeqHandle, layer: int, kv: mx.array) -> None:
        """Single-stream append (DSV4 latent): ``kv`` is ``[1, T, head_dim]`` (one latent per token,
        no k/v pair). Same write cursor / block-crossing / COW semantics as :meth:`write` — just the
        single-stream codec. Allowed only on a ``single_stream=True`` manager (rule 6, loud)."""
        if not self.single_stream:
            raise RuntimeError("write_one requires a single_stream=True manager; use write() for k/v")
        n_write = int(kv.shape[1])
        if n_write == 0:
            return
        self._write_encoded(seq, layer, self._encode_one(kv), n_write)

    def _write_encoded(self, seq: SeqHandle, layer: int, encoded: dict[str, mx.array],
                       n_write: int) -> None:
        """Scatter ``n_write`` pre-encoded token-major records (a component dict) into ``layer``'s
        blocks from the write cursor — the shared body of :meth:`write` (k/v) and :meth:`write_one`
        (single latent stream). Allocates / COW-clones blocks exactly as the public writers document."""
        if n_write == 0:
            return
        start = seq.n_written[layer]
        end = start + n_write
        if end > seq.length:
            raise ValueError(
                f"paged write layer {layer}: writing to {end} exceeds advanced length {seq.length} "
                f"(call advance() to open the positions first)")
        pools = self._ensure_pools(layer, encoded, ())
        block_table = seq.block_tables[layer]
        alloc = self._allocs[layer]
        pos = start
        while pos < end:
            bi = pos // self.block_size
            intra = pos % self.block_size
            take = min(self.block_size - intra, end - pos)
            if bi == len(block_table):
                blk = alloc.alloc()
                block_table.append(blk)
            elif bi < len(block_table):
                blk = block_table[bi]
                if blk.is_shared():                 # mutating a shared (forked) partial tail -> COW
                    blk = self._cow(layer, block_table, bi)
            else:
                raise RuntimeError(f"paged write: block gap at index {bi} (have {len(block_table)})")
            off = pos - start
            for name, arr in encoded.items():
                pools[name][blk.block_id, intra:intra + take] = arr[off:off + take]
            blk.token_count = max(blk.token_count, intra + take)
            pos += take
        seq.n_written[layer] = end

    def commit(self, seq: SeqHandle) -> None:
        """Content-hash every block that just filled (all layers), so later prefixes can reuse it.
        Run once per step after all layers wrote (hashing needs the step's token ids, now recorded)."""
        n_full = seq.length // self.block_size
        parent: bytes | None = None
        hashes: list[bytes] = []
        for bi in range(n_full):
            chunk = tuple(seq.token_ids[bi * self.block_size:(bi + 1) * self.block_size])
            parent = compute_block_hash(parent, chunk, model_name=self.model_name)
            hashes.append(parent)
        for layer in range(self.num_layers):
            alloc = self._allocs[layer]
            bt = seq.block_tables[layer]
            for bi in range(n_full):
                blk = bt[bi]
                if blk.block_hash is None and blk.is_full(self.block_size):
                    alloc.register_full(blk, hashes[bi], self.block_size)
                    self._stats.prefix_miss_blocks += 1

    # --- batched write path (#153) -------------------------------------------
    # Paged decode of ``B`` lock-step streams otherwise pays a per-stream Python loop: each stream
    # quantizes its one new token + slice-assigns it into its own tail block (:meth:`write`/
    # :meth:`write_one`), then a separate per-stream :meth:`gather` re-materializes every stream. #153
    # replaces that with ONE quantize + ONE fancy-index scatter across all ``B`` streams (the paged
    # sibling of the #18 ``_KVArena.append_batched``, but the physical target is each stream's block
    # table instead of a contiguous arena row). Affine int-bits over the last axis (head_dim) is
    # row-independent, so the batched quantize equals ``B`` separate quantizes row-for-row -> contents
    # are BIT-IDENTICAL to the per-stream path. Block resolution (alloc a fresh private tail block on a
    # boundary cross; COW-clone a shared partial tail first) stays bounded per-stream accounting OUTSIDE
    # the scatter (rule 3); in steady serving decode the tail is always private (COW only fires at
    # prefill), so the scatter never touches a shared block — and we assert it (rule 6). Generic over
    # the component dict, so the SAME primitive serves DSV4's single-stream latent AND the k/v keepers
    # (Nemotron / InternLM2.5). Gated model-free in ``parity/dsv4_paged_batched_test.py`` (M0);
    # dispatched behind ``PAGED_KV_BATCHED_DEFAULT`` at the call site once the steppers are wired (M3).
    def write_one_batched(self, seqs: list[SeqHandle], layer: int, kv: mx.array) -> None:
        """Batched single-stream append (DSV4 latent): ``kv`` is ``[B, 1, head_dim]`` for the
        ``B == len(seqs)`` lock-step streams, each appending ONE token at its own write cursor. ONE
        quantize + ONE scatter; equivalent to :meth:`write_one` on each stream. ``single_stream=True``
        only (rule 6)."""
        if not self.single_stream:
            raise RuntimeError("write_one_batched requires a single_stream=True manager; use write_batched")
        if int(kv.shape[0]) != len(seqs):
            raise ValueError(f"write_one_batched: {len(seqs)} seqs but kv batch {kv.shape[0]}")
        if int(kv.shape[1]) != 1:
            raise ValueError(f"write_one_batched expects one token per stream (T==1, got {kv.shape[1]})")
        if self.quantized:
            q, s, b = quantize_last_axis(kv, self.group_size, self.bits)            # [B, 1, *]
            encoded = {"kv_q": q[:, 0], "kv_s": s[:, 0], "kv_b": b[:, 0]}            # [B, *]
        else:
            encoded = {"kv": kv[:, 0]}                                              # [B, head_dim]
        self._write_encoded_batched(seqs, layer, encoded)

    def write_batched(self, seqs: list[SeqHandle], layer: int, k: mx.array, v: mx.array) -> None:
        """Batched k/v append (standard attention — Nemotron / InternLM2.5): ``k``/``v`` are
        ``[B, n_kv, 1, head_dim]`` for the ``B == len(seqs)`` lock-step streams. ONE quantize + ONE
        scatter; equivalent to :meth:`write` on each stream. ``single_stream=False`` only (rule 6)."""
        if self.single_stream:
            raise RuntimeError("write_batched is the k/v adapter; this manager is single_stream — use write_one_batched")
        if int(k.shape[0]) != len(seqs) or tuple(k.shape) != tuple(v.shape):
            raise ValueError(f"write_batched: {len(seqs)} seqs, k {tuple(k.shape)}, v {tuple(v.shape)}")
        if int(k.shape[2]) != 1:
            raise ValueError(f"write_batched expects one token per stream (T==1, got {k.shape[2]})")
        if self.quantized:
            kq, ks, kb = quantize_last_axis(k, self.group_size, self.bits)          # [B, n_kv, 1, *]
            vq, vs, vb = quantize_last_axis(v, self.group_size, self.bits)
            encoded = {"k_q": kq[:, :, 0], "k_s": ks[:, :, 0], "k_b": kb[:, :, 0],
                       "v_q": vq[:, :, 0], "v_s": vs[:, :, 0], "v_b": vb[:, :, 0]}   # [B, n_kv, *]
        else:
            encoded = {"k": k[:, :, 0], "v": v[:, :, 0]}                            # [B, n_kv, head_dim]
        self._write_encoded_batched(seqs, layer, encoded)

    def _write_encoded_batched(self, seqs: list[SeqHandle], layer: int,
                               encoded: dict[str, mx.array]) -> None:
        """Scatter ONE pre-encoded token per stream (component dict ``{name: [B, *per_tok]}``) into
        ``layer``'s blocks at each stream's write cursor — the batched sibling of :meth:`_write_encoded`.
        Resolves each stream's target block (alloc fresh on a boundary cross; COW-clone a shared partial
        tail first) as bounded per-stream accounting, asserts every resolved block is private (rule 6:
        never scatter into a shared block), then lands all ``B`` records with ONE fancy-index assign per
        component. The only tensor op is the scatter; there is no per-token / per-hidden Python loop."""
        b = len(seqs)
        if b == 0:
            return
        alloc = self._allocs[layer]
        pools = self._ensure_pools(layer, encoded, ())
        block_ids: list[int] = []
        intras: list[int] = []
        for seq in seqs:                          # bounded accounting (<= max_batch), no tensor compute
            pos = seq.n_written[layer]
            if pos >= seq.length:
                raise ValueError(
                    f"paged batched write layer {layer}: stream cursor {pos} >= advanced length "
                    f"{seq.length} (call advance() to open the position first)")
            bi = pos // self.block_size
            intra = pos % self.block_size
            block_table = seq.block_tables[layer]
            if bi == len(block_table):
                blk = alloc.alloc()
                block_table.append(blk)
            elif bi < len(block_table):
                blk = block_table[bi]
                if blk.is_shared():               # forked shared tail -> COW before mutating
                    blk = self._cow(layer, block_table, bi)
            else:
                raise RuntimeError(f"paged batched write: block gap at index {bi} "
                                   f"(have {len(block_table)})")
            if blk.is_shared():                   # post-COW the write block must be private (rule 6)
                raise RuntimeError(f"paged batched write: target block {blk.block_id} still shared")
            block_ids.append(blk.block_id)
            intras.append(intra)
            blk.token_count = max(blk.token_count, intra + 1)
        rows_arr = mx.array(block_ids, dtype=mx.int32)
        cols_arr = mx.array(intras, dtype=mx.int32)
        for name, arr in encoded.items():
            pool = pools[name]
            idx = (rows_arr, cols_arr) + (slice(None),) * (pool.ndim - 2)
            pool[idx] = arr
        for seq in seqs:                          # advance cursors only after the data has landed
            seq.n_written[layer] += 1

    # --- read path -----------------------------------------------------------
    def gather(self, seq: SeqHandle, layer: int) -> tuple[mx.array, mx.array]:
        """Materialize the ``[1, n_kv, n_written, head_dim]`` bf16 k,v stream for SDPA — one vectorized
        ``mx.take`` over the sequence's block ids per component, sliced to this layer's written extent
        (``n_written[layer]``, which equals ``length`` once every layer has caught up; it can trail
        mid-chunk during a chunked prefill, where the partially-written suffix must not be gathered)."""
        pools = self._pools[layer]
        n = seq.n_written[layer]
        if pools is None or n == 0:
            raise RuntimeError(f"paged gather layer {layer}: nothing written")
        ids = mx.array([blk.block_id for blk in seq.block_tables[layer]], dtype=mx.uint32)
        gathered: dict[str, mx.array] = {}
        for name, pool in pools.items():
            g = mx.take(pool, ids, axis=0)                      # [nb, block_size, n_kv, C]
            nb = g.shape[0]
            g = g.reshape(nb * self.block_size, *g.shape[2:])   # [nb*block_size, n_kv, C]
            g = g[:n].transpose(1, 0, 2)[None]                  # [1, n_kv, n_written, C]
            gathered[name] = g
        if self.quantized:
            k = dequantize_last_axis(gathered["k_q"], gathered["k_s"], gathered["k_b"],
                                     self.group_size, dtype=mx.bfloat16, bits=self.bits)
            v = dequantize_last_axis(gathered["v_q"], gathered["v_s"], gathered["v_b"],
                                     self.group_size, dtype=mx.bfloat16, bits=self.bits)
            return k, v
        return gathered["k"].astype(mx.bfloat16), gathered["v"].astype(mx.bfloat16)

    def gather_one(self, seq: SeqHandle, layer: int) -> mx.array:
        """Materialize the ``[1, n_written, head_dim]`` bf16 latent stream (single-stream sibling of
        :meth:`gather`) — one vectorized ``mx.take`` over the sequence's block ids, sliced to this
        layer's written extent. The reused prefix blocks + freshly-written suffix dequantize to a
        stream **bit-identical** to the discrete :class:`quanta.dsv4.decode._LayerCache.kv`."""
        pools = self._pools[layer]
        n = seq.n_written[layer]
        if pools is None or n == 0:
            raise RuntimeError(f"paged gather_one layer {layer}: nothing written")
        ids = mx.array([blk.block_id for blk in seq.block_tables[layer]], dtype=mx.uint32)
        gathered: dict[str, mx.array] = {}
        for name, pool in pools.items():
            g = mx.take(pool, ids, axis=0)                      # [nb, block_size, C]
            nb = g.shape[0]
            g = g.reshape(nb * self.block_size, *g.shape[2:])   # [nb*block_size, C]
            gathered[name] = g[:n][None]                        # [1, n_written, C]
        if self.quantized:
            return dequantize_last_axis(gathered["kv_q"], gathered["kv_s"], gathered["kv_b"],
                                        self.group_size, dtype=mx.bfloat16, bits=self.bits)
        return gathered["kv"].astype(mx.bfloat16)

    # --- batched read path (#153) --------------------------------------------
    def _gather_encoded_batched(self, seqs: list[SeqHandle], layer: int) -> tuple[dict[str, mx.array], int]:
        """Gather the ``B`` streams' written extents into padded ``[B, L_max, *per_tok]`` component
        arrays with ONE ``mx.take`` per component over a padded block-id matrix — the batched sibling of
        :meth:`gather`/:meth:`gather_one`'s per-stream ``mx.take`` + ``_pad_stack``. ``L_max =
        max(n_written[layer])``; each stream's block ids are tail-padded to the longest stream's block
        count with block 0, whose gathered rows land at positions ``>= n_written`` for that stream and so
        are never read as valid (the SDPA window/pad mask sends them to ``-inf`` — the #18 argument)."""
        pools = self._pools[layer]
        ns = [seq.n_written[layer] for seq in seqs]
        l_max = max(ns) if ns else 0
        if pools is None or l_max == 0:
            raise RuntimeError(f"paged batched gather layer {layer}: nothing written")        # rule 6
        bid_lists = [[blk.block_id for blk in seq.block_tables[layer]] for seq in seqs]
        max_nb = max(len(bl) for bl in bid_lists)
        bids = [bl + [0] * (max_nb - len(bl)) for bl in bid_lists]    # tail-pad (real blocks first)
        bids_arr = mx.array(bids, dtype=mx.uint32).reshape(-1)        # [B * max_nb]
        b = len(seqs)
        gathered: dict[str, mx.array] = {}
        for name, pool in pools.items():
            g = mx.take(pool, bids_arr, axis=0)                       # [B*max_nb, block_size, *per_tok]
            g = g.reshape(b, max_nb * self.block_size, *pool.shape[2:])  # [B, max_nb*block_size, ...]
            gathered[name] = g[:, :l_max]                             # [B, L_max, *per_tok]
        return gathered, l_max

    def gather_one_batched(self, seqs: list[SeqHandle], layer: int) -> mx.array:
        """Materialize the ``[B, L_max, head_dim]`` bf16 latent stream for all ``B`` streams (the
        single-stream sibling of :meth:`gather_one`) — ONE gather + ONE batched dequant, replacing the
        per-stream ``gather_one`` loop + ``_pad_stack``. ``single_stream=True`` only (rule 6)."""
        if not self.single_stream:
            raise RuntimeError("gather_one_batched needs a single_stream=True manager; use gather_batched")
        g, _ = self._gather_encoded_batched(seqs, layer)
        if self.quantized:
            return dequantize_last_axis(g["kv_q"], g["kv_s"], g["kv_b"], self.group_size,
                                        dtype=mx.bfloat16, bits=self.bits)
        return g["kv"].astype(mx.bfloat16)

    def gather_batched(self, seqs: list[SeqHandle], layer: int) -> tuple[mx.array, mx.array]:
        """Materialize the ``[B, n_kv, L_max, head_dim]`` bf16 k,v streams for all ``B`` streams (the
        k/v sibling of :meth:`gather`) — ONE gather + ONE batched dequant for each of k/v, replacing the
        per-stream ``gather`` loop + pad/stack. ``single_stream=False`` only (rule 6)."""
        if self.single_stream:
            raise RuntimeError("gather_batched is the k/v adapter; this manager is single_stream — use gather_one_batched")
        g, _ = self._gather_encoded_batched(seqs, layer)              # components [B, L_max, n_kv, *]
        if self.quantized:
            k = dequantize_last_axis(g["k_q"], g["k_s"], g["k_b"], self.group_size,
                                     dtype=mx.bfloat16, bits=self.bits)
            v = dequantize_last_axis(g["v_q"], g["v_s"], g["v_b"], self.group_size,
                                     dtype=mx.bfloat16, bits=self.bits)
        else:
            k, v = g["k"].astype(mx.bfloat16), g["v"].astype(mx.bfloat16)
        return k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)       # [B, n_kv, L_max, head_dim]

    def truncate(self, seq: SeqHandle, length: int) -> None:
        """Roll the whole sequence back to ``length`` tokens (drop trailing blocks). Block-granular;
        a tail that lands mid-block keeps that block and just lowers its ``token_count`` (the stale
        rows are overwritten by the next write)."""
        if length < 0 or length > seq.length:
            raise ValueError(f"truncate length {length} out of range [0, {seq.length}]")
        keep_blocks = (length + self.block_size - 1) // self.block_size
        for layer in range(self.num_layers):
            alloc = self._allocs[layer]
            bt = seq.block_tables[layer]
            for blk in bt[keep_blocks:]:
                alloc.free(blk)
            del bt[keep_blocks:]
            if keep_blocks and bt:
                bt[-1].token_count = min(bt[-1].token_count, length - (keep_blocks - 1) * self.block_size)
            seq.n_written[layer] = min(seq.n_written[layer], length)
        seq.length = length
        del seq.token_ids[length:]

    # --- stats ---------------------------------------------------------------
    def view(self, seq: SeqHandle, layer: int) -> "PagedKVCacheView":
        if self.single_stream:
            raise RuntimeError("view() is the k/v adapter; this manager is single_stream — use view_one()")
        return PagedKVCacheView(self, seq, layer)

    def view_one(self, seq: SeqHandle, layer: int) -> "PagedLatentCacheView":
        if not self.single_stream:
            raise RuntimeError("view_one() needs a single_stream=True manager (DSV4 latent); use view()")
        return PagedLatentCacheView(self, seq, layer)

    def get_stats(self) -> PagedCacheStats:
        live = cached = 0
        for alloc in self._allocs:
            cached += alloc.num_cached
            live += alloc.num_blocks - alloc.num_free
        self._stats.allocated_blocks = live
        self._stats.cached_blocks = cached
        return self._stats


class PagedKVCacheView:
    """Per-(sequence, layer) facade that is signature-compatible with
    :class:`quanta.nemotron.attention.KVCache` for the per-layer forward: ``offset`` and
    ``update(k, v) -> (k_full, v_full)``. Position advance + prefix hashing are driven by the manager
    (``advance`` / ``commit``) because they need token ids, which ``update`` never sees."""

    def __init__(self, manager: PagedKVCacheManager, seq: SeqHandle, layer: int) -> None:
        self._m = manager
        self._seq = seq
        self._layer = layer

    @property
    def offset(self) -> int:
        return self._seq.n_written[self._layer]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        self._m.write(self._seq, self._layer, k, v)
        return self._m.gather(self._seq, self._layer)

    def truncate(self, length: int) -> None:
        # per-layer truncate is driven at the sequence level by the manager (all layers move together)
        self._m.truncate(self._seq, length)

    def _copy(self) -> "PagedKVCacheView":
        # A per-layer view cannot replicate paged state: branching must fork the WHOLE sequence at once
        # (all layers share one block table per layer), so the correct primitive is the sequence-level
        # PagedKVCacheManager.replicate(seq, b) / .fork(seq) — not a per-view copy. Fail loud (rule 6)
        # rather than silently clone a single layer. Wiring tree-spec verify onto paged caches is #158-160.
        raise NotImplementedError(
            "paged per-layer replicate is the wrong abstraction (a view sees one layer); use "
            "PagedKVCacheManager.replicate(seq, b) for B-way COW branching (or .fork(seq) for one). "
            "Tree-spec verify (#158-160) still uses discrete caches per the #152 scope guard.")


class PagedLatentCacheView:
    """Per-(sequence, layer) facade over a SINGLE latent KV stream (DSV4's MQA latent) — the
    single-stream sibling of :class:`PagedKVCacheView`. Backs the latent surface of
    :class:`quanta.dsv4.decode._PagedLayerCache`: ``offset`` (== written latent length),
    ``append(kv)`` (write-only, the discarded-return hot path the DSV4 stepper uses), ``current()``
    (gather the full bf16 latent stream for the windowed SDPA + compressed-KV scoring), and
    ``truncate(length)``. Position advance + prefix hashing are driven by the manager
    (``advance`` / ``commit``) — those need token ids, which the per-layer write never sees."""

    def __init__(self, manager: PagedKVCacheManager, seq: SeqHandle, layer: int) -> None:
        self._m = manager
        self._seq = seq
        self._layer = layer

    @property
    def offset(self) -> int:
        return self._seq.n_written[self._layer]

    def append(self, kv: mx.array) -> None:
        """Write ``kv`` ``[1, T, head_dim]`` (T==1 at decode) at the write cursor. The DSV4 stepper
        reads the grown stream back via :meth:`current` (the discrete ``_LayerCache.append_kv`` is
        likewise void), so we don't gather here — one gather per step, not two."""
        self._m.write_one(self._seq, self._layer, kv)

    def current(self) -> mx.array:
        """The full ``[1, n_written, head_dim]`` bf16 latent stream (prefix blocks + suffix)."""
        return self._m.gather_one(self._seq, self._layer)

    def truncate(self, length: int) -> None:
        # per-layer truncate is driven at the sequence level by the manager (all layers move together)
        self._m.truncate(self._seq, length)
