"""Qwen3.5 decode cache — per-layer state for the 3:1 hybrid, with **lossless** rollback.

The cache holds, per decoder layer, the state its mixer type needs (chosen by
:meth:`Qwen35Config.is_linear_attention`):

* **full-attention** layers (gated GQA) → a :class:`quanta.qwen35.attention.KVCache` (grows the
  ``[B, n_kv, S, head_dim]`` k/v along the seq axis); rolling it back is a clean slice.
* **linear-attention** layers (Gated DeltaNet) → an O(1) **recurrent** state
  ``(conv_state, recurrent_state)``: ``conv_state`` ``[B, K-1, conv_dim]`` is the depthwise-conv
  window, ``recurrent_state`` ``[B, Hv, Dk, Dv]`` is the gated-delta-rule matrix. This state is a
  *summary* of every token consumed so far — it **cannot be sliced** like a KV cache.

The hard constraint (CLAUDE.md rule 6 / parity rule 4): a rejected speculative draft must be dropped
**losslessly** — after ``truncate(length)`` the cache must be bit-identical to one that only ever
consumed ``length`` tokens, for BOTH state types. KV slices trivially. For the recurrent state we make
rollback exact by **snapshotting** the ``(conv_state, recurrent_state)`` after every committed token:
``truncate(length)`` restores the snapshot taken at exactly ``length`` consumed tokens — the literal
post-``length`` state, so it is bit-exact, not a re-derivation. Snapshots are O(1) per layer and a
**bounded** number are retained (``snapshot_depth``); a rollback deeper than the retained snapshots
**raises** (it must NOT be left holding a diverged recurrent state). Native-MTP self-speculation
(``k == 1``) rolls back at most one trailing token, well within the default depth.

``__getitem__`` / ``__len__`` / ``.offset`` mirror :class:`quanta.dsv4.decode.DSV4Cache` and
:class:`quanta.nemotron.attention.KVCache` ergonomics so :func:`quanta.qwen35.generate.generate` and
:func:`quanta.qwen35.spec.spec_generate` drive it the same way the sibling stacks are driven.

The decode steppers themselves are NOT reimplemented here — the runtime calls the proven
:class:`quanta.qwen35.gated_deltanet.GatedDeltaNet` decode recurrence (#109) and
:class:`quanta.qwen35.attention.Qwen35Attention` KV decode (#110). This module is only the state
container + its lossless rollback. Gated model-free in ``parity/qwen35_decode_attn_test.py``.
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.attention import KVCache
from quanta.qwen35.config import Qwen35Config

# How many trailing committed-token snapshots each recurrent layer retains for rollback. Native-MTP
# (k == 1) rolls back ≤ 1 token; the small margin tolerates deeper verify windows without ever leaving
# the recurrent state diverged (a rollback past the oldest retained snapshot fails loud, rule 6).
DEFAULT_SNAPSHOT_DEPTH = 8


class _GDNLayerState:
    """Recurrent (Gated DeltaNet) decode state for one linear-attention layer + lossless rollback.

    Holds the *live* ``(conv_state, recurrent_state)`` the decode recurrence threads, the number of
    tokens consumed (``offset``), and a bounded history of ``(offset, conv_state, recurrent_state)``
    snapshots taken **after** each committed token. Because the recurrent state summarizes all prior
    tokens it cannot be sliced; rollback restores the literal snapshot at the target offset (bit-exact),
    and a target older than the retained snapshots raises rather than silently diverging (rule 6).
    """

    __slots__ = ("conv_state", "recurrent_state", "_off", "_snaps", "_depth")

    def __init__(self, snapshot_depth: int = DEFAULT_SNAPSHOT_DEPTH) -> None:
        self.conv_state: mx.array | None = None        # [B, K-1, conv_dim] depthwise-conv window
        self.recurrent_state: mx.array | None = None   # [B, Hv, Dk, Dv] gated-delta-rule matrix
        self._off: int = 0                             # tokens consumed by this layer
        # snapshots of the state AT each offset (offset -> (conv, recurrent)); offset 0 is the empty
        # state (None, None). A deque-like bounded list keyed by trailing window.
        self._snaps: list[tuple[int, mx.array | None, mx.array | None]] = [(0, None, None)]
        self._depth = max(1, int(snapshot_depth))

    @property
    def offset(self) -> int:
        return self._off

    def commit(self, conv_state: mx.array, recurrent_state: mx.array) -> None:
        """Record the post-step state after consuming one more token (advance offset, snapshot).

        The decode stepper returns the new ``(recurrent_state, conv_state)``; the runtime calls this to
        store them as live and snapshot them so a later ``truncate`` can restore this exact point. The
        snapshot list is trimmed to the last ``snapshot_depth`` committed offsets (plus the empty base)."""
        self.conv_state, self.recurrent_state = conv_state, recurrent_state
        self._off += 1
        self._snaps.append((self._off, conv_state, recurrent_state))
        # keep the empty base (offset 0) plus the last ``_depth`` committed snapshots
        if len(self._snaps) > self._depth + 1:
            self._snaps = [self._snaps[0]] + self._snaps[-self._depth:]

    def truncate(self, length: int) -> None:
        """Restore the exact state after consuming ``length`` tokens (drop the rolled-back tail).

        Lossless by construction: it reinstates the literal snapshot taken at offset ``length`` (not a
        re-derivation). A ``length`` older than the oldest retained snapshot (other than the empty base)
        raises — the recurrent state for that point was not kept, and leaving the current diverged state
        resident would silently break greedy-equivalence (rule 6)."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length >= self._off:
            return
        for off, conv, rec in self._snaps:
            if off == length:
                self.conv_state, self.recurrent_state = conv, rec
                self._off = length
                # drop any snapshots past the restore point so future commits extend cleanly
                self._snaps = [s for s in self._snaps if s[0] <= length]
                return
        oldest = min(off for off, _, _ in self._snaps if off > 0) if len(self._snaps) > 1 else 0
        raise ValueError(
            f"truncate({length}) rolls back past the retained recurrent snapshots (offset {self._off}, "
            f"oldest retained {oldest}); the Gated DeltaNet recurrent state for offset {length} was not "
            f"snapshotted. Native-MTP (k=1) rolls back ≤1 token; raise snapshot_depth for deeper "
            f"speculation rather than keeping a diverged recurrent state (rule 6).")

    def _copy(self) -> "_GDNLayerState":
        """Shallow per-layer copy for :meth:`Qwen35Cache.replicate` — shares the immutable MLX
        array refs (conv/recurrent state + every snapshot's tensors), copies the Python int +
        list bookkeeping so writes on the copy do not mutate this state. The snapshot list itself
        is Python-mutable; ``list(...)`` clones the spine, the tuples it holds reference the same
        immutable tensors (MLX arrays are immutable so divergent ``commit``/``truncate`` on the
        copy creates new arrays only — sibling caches are unaffected)."""
        new = _GDNLayerState(self._depth)
        new.conv_state = self.conv_state
        new.recurrent_state = self.recurrent_state
        new._off = self._off
        new._snaps = list(self._snaps)
        return new


class Qwen35Cache:
    """Per-layer decode cache for the Qwen3.5 hybrid: KV on full layers, recurrent state on linear.

    Built from the config schedule so each layer slot carries the right state type. ``offset`` reports
    the shared decode position (every layer advances in lock-step) and survives ``truncate``;
    ``truncate(length)`` rolls EVERY layer back losslessly (KV slice / recurrent snapshot restore).
    """

    def __init__(self, n_layers: int, cfg: Qwen35Config | None = None,
                 *, layer_is_linear=None, snapshot_depth: int = DEFAULT_SNAPSHOT_DEPTH,
                 quantized: bool = False, group_size: int = 64,
                 max_rollback: int = 1) -> None:
        """``max_rollback`` (≥ 1) enlarges the per-linear-layer recurrent-state snapshot depth so
        :meth:`truncate` can drop up to that many trailing tokens in a single call (k≥2 chained
        spec-decode in :mod:`quanta.qwen35.spec`). Default 1 = the k=1 spec ceiling. ``snapshot_depth``
        is taken as ``max(snapshot_depth, max_rollback)`` so callers that explicitly raise the depth
        still win, and a deeper-than-bounded rollback raises rather than silently diverging (rule 6).
        """
        if max_rollback < 1:
            raise ValueError(f"max_rollback must be >= 1 (got {max_rollback})")
        if cfg is not None and layer_is_linear is None:
            def layer_is_linear(i: int) -> bool:
                return cfg.is_linear_attention(i)
        if layer_is_linear is None:
            raise ValueError("Qwen35Cache needs cfg or layer_is_linear to type each layer")
        self.quantized = quantized
        self.group_size = group_size
        self.max_rollback = int(max_rollback)
        effective_depth = max(int(snapshot_depth), int(max_rollback))
        # ``quantized=True`` applies only to the full-attention KV caches (the linear-attention
        # layers carry an O(1) recurrent state — small, not a memory bottleneck, no benefit from int8).
        self.layers: list = [
            _GDNLayerState(effective_depth) if layer_is_linear(i)
            else KVCache(quantized=quantized, group_size=group_size)
            for i in range(n_layers)
        ]
        # Pinned per-request YaRN seq-len (None ⇒ derive from the live position). Holds the dynamic
        # YaRN factor fixed for the request's lifetime so the rotate-then-cache roped K stay
        # consistent; set via pin_yarn(), read via yarn_seq(). See yarn_seq for the >native rule.
        self.yarn_seq_hint: int | None = None

    def __getitem__(self, i: int):
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        """Tokens already consumed (positions cached); 0 before the first step. Every layer advances
        in lock-step, so the first populated layer reports the shared value (robust to a partial fill)."""
        for lc in self.layers:
            if isinstance(lc, KVCache):
                if lc.offset:
                    return lc.offset
            elif lc.offset:
                return lc.offset
        return 0

    def pin_yarn(self, max_seq_len: int) -> None:
        """Pin the dynamic-YaRN factor for this request to ``max_seq_len`` — the largest absolute
        position it will reach (e.g. ``prompt_len + max_new``). Every prefill/decode forward sharing
        this cache then uses the SAME factor, so the rotate-then-cache roped K stay consistent. Below
        the native window this is a no-op (factor 1.0); above it, pinning is what stops the dynamic
        factor drifting per step. Raises on a nonsensical value (rule 6)."""
        if max_seq_len < 1:
            raise ValueError(f"pin_yarn(max_seq_len) needs max_seq_len >= 1 (got {max_seq_len})")
        self.yarn_seq_hint = int(max_seq_len)

    def yarn_seq(self, live_pos: int, cfg: Qwen35Config) -> int:
        """Resolve the YaRN seq-len for a *cached* forward holding ``live_pos`` tokens (== offset+T).

        Static policy (``yarn_dynamic=False``): the factor ignores length, so the live position is
        safe. Dynamic policy: return the pinned value if :meth:`pin_yarn` was set, else the live
        position while it fits the native window. Refuses (rule 6) to derive a >native factor from the
        live position without a pin — that factor changes every step while the cached K were roped at
        the old factor, silently corrupting attention. Refuses a position past the pinned budget too."""
        if not cfg.yarn_dynamic:
            return live_pos
        pin = self.yarn_seq_hint
        if pin is None:
            if live_pos > cfg.yarn_original_max:
                raise RuntimeError(
                    f"qwen35: decode position {live_pos} exceeds the native window "
                    f"{cfg.yarn_original_max} with no pinned YaRN factor — call "
                    f"cache.pin_yarn(prompt_len + max_new) before decoding past native (rule 6: "
                    f"refusing to silently corrupt the rotate-then-cache KV with a drifting factor).")
            return live_pos
        if live_pos > pin:
            raise RuntimeError(
                f"qwen35: decode position {live_pos} exceeds the pinned YaRN budget {pin} "
                f"(request outgrew its pinned max length; re-pin or cap generation).")
        return pin

    def truncate(self, length: int) -> None:
        """Roll every layer back to exactly the state after consuming ``length`` tokens — losslessly
        for BOTH state types (KV slice; recurrent snapshot restore). Fails loud on a recurrent rollback
        deeper than the retained snapshots (rule 6) — the per-linear-layer snapshot ring is sized for
        ``max(snapshot_depth, max_rollback)`` (see :meth:`__init__`) so a within-bound rollback always
        succeeds and an out-of-bounds one raises from :meth:`_GDNLayerState.truncate` rather than
        silently keeping a diverged state."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        for lc in self.layers:
            if isinstance(lc, KVCache):
                _kv_truncate(lc, length)
            else:
                lc.truncate(length)

    def replicate(self, b: int) -> list["Qwen35Cache"]:
        """Return ``b`` parallel decode caches, each initially sharing this cache's prefix state.

        MLX arrays are immutable, so the replicas can share references to the prefix tensors at
        zero cost — subsequent ``update`` (KV) / ``commit`` (GDN) / ``truncate`` on any replica
        creates new arrays only, leaving the originals (and every other replica) untouched. The
        structural-sharing form of ``cache.replicate(B)`` for the batched tree-spec verify
        (docs/batched_tree_verify.md): each enumerated draft path advances its own replica through
        :meth:`Qwen35BatchedResidentModel.batch_step`, the original prefix stays read-only, and the
        B-wide verify amortizes the routed-MoE weight reads across all B paths in one batched MoE
        call. Handles both regime types uniformly: full-attention :class:`KVCache` and recurrent
        :class:`_GDNLayerState` both expose ``_copy`` with the same zero-cost semantics.

        The picked path's replica is discarded after the round — the commit-forward re-feeds the
        accepted prefix through the original (un-replicated) cache, exactly as today. Gated in
        ``parity/qwen35_batched_tree_verify_test.py``'s "cache invariance" assertion.
        """
        if b < 1:
            raise ValueError(f"replicate(B) requires B >= 1 (got {b})")
        return [self._copy() for _ in range(b)]

    def _copy(self) -> "Qwen35Cache":
        """Shallow per-layer copy: each layer's state is shared via its own ``_copy``
        (KV cache or GDN snapshot ring); writes diverge naturally under MLX immutability."""
        new = self.__new__(Qwen35Cache)
        new.quantized = self.quantized
        new.group_size = self.group_size
        new.max_rollback = self.max_rollback
        new.yarn_seq_hint = self.yarn_seq_hint     # replicas inherit the request's pinned factor
        new.layers = [lc._copy() for lc in self.layers]
        return new


def _kv_truncate(cache: KVCache, length: int) -> None:
    """Slice a GQA KV cache back to ``length`` cached positions (lossless — per-position storage).

    Handles both bf16 and int8 storage modes: for int8 the trio (codes, scales, biases) for K and V
    slices in lockstep along the seq axis so the rolled-back state is bit-identical to a fresh cache
    fed only those ``length`` positions."""
    if cache.offset == 0:
        return
    if length <= 0:
        cache.k = cache.v = None
        cache.k_q = cache.k_s = cache.k_b = None
        cache.v_q = cache.v_s = cache.v_b = None
        return
    if length < cache.offset:
        if cache.quantized:
            cache.k_q = cache.k_q[:, :, :length]
            cache.k_s = cache.k_s[:, :, :length]
            cache.k_b = cache.k_b[:, :, :length]
            cache.v_q = cache.v_q[:, :, :length]
            cache.v_s = cache.v_s[:, :, :length]
            cache.v_b = cache.v_b[:, :, :length]
        else:
            cache.k = cache.k[:, :, :length]
            cache.v = cache.v[:, :, :length]
