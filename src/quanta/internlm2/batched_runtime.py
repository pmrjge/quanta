"""Batched-serving runtime for the RAM-resident InternLM2.5-7B-Chat-1M (#176) — the clean dense case.

InternLM2.5 is the simplest of the three #152 keepers: **32 dense GQA layers, no MoE, no SSM, no MTP,
no recurrent state at all.** That makes both halves of the batched/paged contract trivial relative to
Nemotron (hybrid Mamba+GQA) and DSV4 (compressed-latent + indexer):

* **Batched decode** is a bounded per-stream loop over the resident weights (composition — one weight
  set shared across all streams, so memory does NOT scale with ``max_batch``). There is no MoE to
  amortize across streams the way Nemotron/DSV4 do, so decode-step batching buys nothing here beyond
  the scheduler plumbing — the agentic win for this model is **prefix-KV sharing** (paging), not
  decode-step fan-in. Stacking the offset-independent projections/FFN across streams (looping only the
  per-stream SDPA) is a worthwhile future optimization (cf. #153) but is left out so the first cut is
  obviously bit-exact with single-stream.

* **Paging** is the textbook k/v-pair case: every layer's :class:`~quanta.internlm2.attention.KVCache`
  is paged through the shared :class:`~quanta.paged.PagedKVCacheManager` (``view`` / ``PagedKVCacheView``
  — NOT the single-stream latent codec DSV4 uses), and there is **nothing** to content-address at a
  block boundary (``has_recurrent_state = False``). So ``prefill_paged`` just runs the uncached suffix
  against the reused prefix blocks and returns **no** boundary payloads — the engine's paged admit path
  (``quanta.shim.omlx._BaseBatchedSession._admit_paged``) already short-circuits the recurrent branch
  when ``has_recurrent_state`` is False (``recurrent_in=None``).

Parity foundation (shared with Nemotron/DSV4 paging): int8 KV quant groups sit on ``head_dim`` (the
last axis) while blocks cut the seq axis — orthogonal, so a paged gather is bit-identical to the
discrete :class:`~quanta.internlm2.attention.KVCache` fed the same tokens. ``paged_kv_spec`` threads the
runtime's actual ``(group_size, bits, quantized)`` (never hardcoded — rule 6) so paged == discrete
exactly. Gated model-free in ``parity/internlm2_paged_test.py``; real-model teacher-forced ppl parity
(paged ON == OFF) is deferred to a one-model-at-a-time GPU session.

NTK note: InternLM2's RoPE base is dynamic above ``cfg.max_position_embeddings`` (262144) — it depends
on the running sequence length. Cross-request prefix reuse is therefore bit-exact only while the shared
prefix sits at ``seq_len <= max_position_embeddings`` (base == ``rope_theta``, constant), which is the
entire agentic regime; above 256K, dynamic-NTK makes a key's frozen add-time base length-dependent, an
inherent property of incremental NTK decode (not a paging artifact). Documented, not guarded — the
paged path defaults OFF (rule 4) and the win targets sub-256K prefixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from quanta.internlm2.config import InternLM2Config
from quanta.internlm2.decode import InternLM2Cache
from quanta.internlm2.runtime import InternLM2ResidentModel


class InternLM2BatchedResidentModel:
    """Batched-serving wrapper around :class:`InternLM2ResidentModel` — same resident weights, B
    concurrent decode streams sharing them, plus the #152 paged contract.

    Surface consumed by :class:`quanta.shim.omlx._BaseBatchedSession`:

    * unpaged: ``prefill(prompt_ids, cache)`` + ``step_batch(tokens, caches, offsets)`` + ``new_cache()``.
    * paged: ``paged_kv_spec`` / ``make_paged_state`` / ``prefill_paged`` / ``has_recurrent_state``.
    """

    # #152 paged contract: InternLM2.5 is PURE dense attention — every layer's k/v pair is paged, and
    # there is NO recurrent/derived state to snapshot at block boundaries (unlike Nemotron's Mamba or
    # DSV4's compressor). So has_recurrent_state is False and prefill_paged returns no boundary payloads;
    # get_recurrent_state is intentionally absent (the engine never calls it when this flag is False).
    has_recurrent_state = False

    def __init__(self, art_dir: str | Path, *, max_batch: int = 32,
                 n_layers: int | None = None) -> None:
        if max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {max_batch}")
        self.max_batch = int(max_batch)
        self._inner = InternLM2ResidentModel(art_dir, n_layers=n_layers)
        self._fused = True  # default to the batched-attention decode path (Approach-1); see step_batch
        # #153 paged KV loop-kill: when serving paged views, decode_batched swaps the per-stream KV
        # .update() loop for ONE write_batched + ONE gather_batched. Reads the SHARED flag (default OFF,
        # rule 4); InternLM2.5 graduates on its own real-model bench like Nemotron did (which carved out a
        # scoped flag once proven). Until then the proven per-stream paged loop stays the default.
        from quanta.paged import PAGED_KV_BATCHED_DEFAULT
        self._paged_kv_batched = bool(PAGED_KV_BATCHED_DEFAULT)

    @classmethod
    def from_inner(cls, inner: Any, *, max_batch: int = 32) -> "InternLM2BatchedResidentModel":
        """Build a batched model around an already-constructed resident-like ``inner`` WITHOUT touching
        the artifact loader. ``inner`` must duck :class:`InternLM2ResidentModel`'s surface
        (``__call__(token_ids, *, cache=, caches=, offset=, last_only=)`` + ``cfg`` / ``num_layers`` /
        ``new_cache`` / ``quantized_kv`` / ``kv_group_size`` / ``kv_bits``). Used by
        ``parity/internlm2_paged_test.py`` to stay model-free (no GPU, no 7B artifact)."""
        self = cls.__new__(cls)
        self.max_batch = int(max_batch)
        self._inner = inner
        self._fused = True
        from quanta.paged import PAGED_KV_BATCHED_DEFAULT   # #153 paged KV loop-kill (see __init__)
        self._paged_kv_batched = bool(PAGED_KV_BATCHED_DEFAULT)
        return self

    # --- shared-weight surface -------------------------------------------------
    @property
    def cfg(self) -> InternLM2Config:
        return self._inner.cfg

    @property
    def num_layers(self) -> int:
        return self._inner.num_layers

    def new_cache(self) -> InternLM2Cache:
        """A fresh per-slot discrete decode cache (unpaged path) — follows the runtime's KV flags."""
        return self._inner.new_cache()

    # --- unpaged surface (driven by _BaseBatchedSession non-paged path) --------
    def prefill(self, prompt_ids: mx.array, cache: InternLM2Cache) -> mx.array:
        """Single-stream prefill into ``cache`` (grows in place); returns ``[1, 1, vocab]`` last-position
        logits (``last_only`` — the only row the sampler needs; memory-safe at long prompts)."""
        return self._inner(prompt_ids, cache=cache, last_only=True)

    @staticmethod
    def _cache_offset(cache: Any) -> int:
        """A stream cache's absolute decode position — paged ``[view,…]`` list or a single InternLM2Cache."""
        return cache[0].offset if isinstance(cache, list) else cache.offset

    def step_batch(self, stream_token_ids: list[mx.array], stream_caches: list[Any],
                   offsets: list[int] | None = None) -> list[mx.array]:
        """One decode step across ``B = len(stream_token_ids)`` active streams.

        Default path (``self._fused``): a single **batched** forward via
        :meth:`InternLM2ResidentModel.decode_batched` — one fused SDPA + one batched matmul per layer
        across all ``B`` streams (Approach-1; the routed per-token work amortizes over ``B``), instead of
        ``B`` separate single-stream forwards. Engages only for plain single-token decode (every
        ``T_b == 1``); a multi-token tail replay (``T_b > 1``) falls back to the per-stream reference loop
        :meth:`_step_batch_looped`, which is also the parity baseline ``decode_batched`` is gated against
        (``parity/internlm2_batched_attention_test``).

        ``stream_token_ids[b]``: ``[T_b]`` mx.array (``T_b == 1`` for decode). ``stream_caches[b]``: the
        per-stream cache — an :class:`InternLM2Cache` (unpaged) OR the list of
        :class:`~quanta.paged.PagedKVCacheView` from :meth:`make_paged_state` (paged). ``offsets``: explicit
        abs position of each new token (paged step passes these); ``None`` ⇒ each cache's own ``offset``.
        Returns ``[1, T_b, vocab]`` per stream."""
        b = len(stream_token_ids)
        if b == 0:
            return []
        if b > self.max_batch:
            raise ValueError(f"B={b} exceeds max_batch={self.max_batch}")
        if len(stream_caches) != b:
            raise ValueError(f"stream_caches length {len(stream_caches)} != B={b}")
        if offsets is not None and len(offsets) != b:
            raise ValueError(f"offsets length {len(offsets)} != B={b}")
        if offsets is None:
            offsets = [self._cache_offset(c) for c in stream_caches]

        single = all(int(t.reshape(-1).shape[0]) == 1 for t in stream_token_ids)
        if self._fused and single and hasattr(self._inner, "decode_batched"):
            logits = self._inner.decode_batched(stream_token_ids, stream_caches, offsets,
                                                paged_batched=self._paged_kv_batched)  # [B,1,vocab]
            return [logits[s:s + 1] for s in range(b)]
        return self._step_batch_looped(stream_token_ids, stream_caches, offsets)

    def _step_batch_looped(self, stream_token_ids: list[mx.array], stream_caches: list[Any],
                           offsets: list[int]) -> list[mx.array]:
        """Per-stream decode reference (pre-Approach-1 path): ``B`` independent single-stream forwards.
        Retained as the parity baseline for the fused :meth:`step_batch` default and the multi-token
        tail-replay fallback."""
        outs: list[mx.array] = []
        for s in range(len(stream_token_ids)):
            outs.append(self._inner(stream_token_ids[s], caches=stream_caches[s], offset=offsets[s]))
        return outs

    # --- #152 paged contract ---------------------------------------------------
    @property
    def paged_kv_spec(self) -> dict:
        """Shape/codec the shared :class:`~quanta.paged.PagedKVCacheManager` must use to be bit-exact
        with the discrete :class:`~quanta.internlm2.attention.KVCache` — threaded from the runtime's own
        KV flags, never hardcoded (rule 6). All 32 layers are paged (dense); no ``single_stream`` (k/v
        pair), no recurrent cache."""
        return {"n_layers": self.num_layers,
                "group_size": self._inner.kv_group_size,
                "bits": self._inner.kv_bits,
                "quantized": self._inner.quantized_kv}

    def make_paged_state(self, manager: Any, seq: Any) -> list[Any]:
        """Per-stream paged state = one :class:`~quanta.paged.PagedKVCacheView` per layer over the shared
        manager (so the prefix blocks dedup). A plain list is exactly what the inner ``__call__`` accepts
        as ``caches=`` (it reads ``cache_list[0].offset`` for the RoPE position)."""
        return [manager.view(seq, i) for i in range(self.num_layers)]

    def prefill_paged(self, suffix_ids: Any, state: list[Any], *, offset: int,
                      recurrent_in: Any, block_size: int) -> tuple[mx.array, list]:
        """Prefill ONLY the uncached suffix into ``state`` (whose layers are paged views over the resident
        prefix blocks), with the suffix tokens placed at absolute positions ``[offset, offset + T)`` for
        RoPE. Dense pure-attention ⇒ NO recurrent state to restore (``recurrent_in`` is always ``None``
        here) and NO boundary snapshot to produce (returns ``[]``); the per-layer attention writes the
        suffix k/v through the views and reads back the full (prefix + suffix) stream for SDPA, which is
        bit-identical to a one-shot discrete prefill of the whole prompt (the paged gather == discrete
        KVCache foundation). ``block_size`` is unused (no boundary work)."""
        del recurrent_in, block_size  # no recurrent/derived state for a dense model
        ids = suffix_ids if isinstance(suffix_ids, mx.array) else mx.array(suffix_ids)
        ids = ids.reshape(-1)
        if int(ids.shape[0]) == 0:
            raise ValueError("prefill_paged: empty suffix (admit must leave >=1 token to recompute)")
        logits = self._inner(ids, caches=state, offset=int(offset), last_only=True)
        return logits, []
