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


class Qwen35Cache:
    """Per-layer decode cache for the Qwen3.5 hybrid: KV on full layers, recurrent state on linear.

    Built from the config schedule so each layer slot carries the right state type. ``offset`` reports
    the shared decode position (every layer advances in lock-step) and survives ``truncate``;
    ``truncate(length)`` rolls EVERY layer back losslessly (KV slice / recurrent snapshot restore).
    """

    def __init__(self, n_layers: int, cfg: Qwen35Config | None = None,
                 *, layer_is_linear=None, snapshot_depth: int = DEFAULT_SNAPSHOT_DEPTH,
                 quantized: bool = False, group_size: int = 64) -> None:
        if cfg is not None and layer_is_linear is None:
            def layer_is_linear(i: int) -> bool:
                return cfg.is_linear_attention(i)
        if layer_is_linear is None:
            raise ValueError("Qwen35Cache needs cfg or layer_is_linear to type each layer")
        self.quantized = quantized
        self.group_size = group_size
        # ``quantized=True`` applies only to the full-attention KV caches (the linear-attention
        # layers carry an O(1) recurrent state — small, not a memory bottleneck, no benefit from int8).
        self.layers: list = [
            _GDNLayerState(snapshot_depth) if layer_is_linear(i)
            else KVCache(quantized=quantized, group_size=group_size)
            for i in range(n_layers)
        ]

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

    def truncate(self, length: int) -> None:
        """Roll every layer back to exactly the state after consuming ``length`` tokens — losslessly
        for BOTH state types (KV slice; recurrent snapshot restore). Fails loud on a recurrent rollback
        deeper than the retained snapshots (rule 6)."""
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        for lc in self.layers:
            if isinstance(lc, KVCache):
                _kv_truncate(lc, length)
            else:
                lc.truncate(length)


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
