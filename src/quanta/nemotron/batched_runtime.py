"""Batched-serving runtime for the RAM-resident Nemotron-H — Design A (per-stream state, stack MoE).

Wraps a :class:`quanta.nemotron.runtime.NemotronResidentModel` (load + weights are shared via
composition — one resident weight set across all streams) and adds a ``step_batch`` surface that
runs B concurrent decode streams in parallel. The bandwidth win is the **MoE**: per token we
gather only 22 routed slots out of 512, but for B streams ``mx.gather_qmm`` over a stacked
``[B, 1, dim]`` input amortizes the always-on shared-expert read + (when streams happen to
route the same expert) the routed expert reads — the **expert reads dominate decode
bandwidth**, the more streams the deeper the amortization. Top-22 over 512 means more
amortization opportunity than DSV4's top-6 over 256.

Per-stream caches keep this output-equivalent to single-stream:

* Per Mamba layer:  per-stream ``(ssm_state, conv_state)`` recurrence (small bandwidth — the
  state evolution is stream-local and can't be batched without dense materialization, which
  the SSM is sparse against by construction).
* Per GQA layer:    per-stream ``KVCache`` (each stream's offset differs; per-stream RoPE).
* Per MoE layer:    the mixer is **stateless** — stack the ``[B, 1, dim]`` post-norm hiddens
  across streams, run the existing batched MoE once (the moe ``__call__`` reshapes
  ``[B, T, dim] -> [B*T, dim]`` for routing/gather and back, so it is B-aware by construction),
  split back, add to per-stream residuals.

The result is bit-identical to running B independent single-stream decodes when the per-stream
states + inputs are identical (gated model-free in ``parity.nemotron_batched_test``): the only
data-mixing op is the stacked MoE call, and the MoE is per-row (route + dispatch are batch-of-
1-token operations under the hood — ``[B, 1, dim]`` is just ``[B, dim]`` after reshape, so each
row's routing decisions are independent of the others).

Loading is composed (one resident weight set), so memory does **not** scale with ``max_batch``.
Per-stream state scales: each stream holds its own ``ssm_state`` ``(1,H,N,P)`` per mamba layer +
``conv_state`` ``(1,K-1,Cdim)`` + growing ``KVCache`` per attention layer (only the 8 ``*`` layers
keep a KV cache; the 40 mamba layers + 40 MoE layers are stateless KV-wise).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from quanta.nemotron.attention import KVCache
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.model import NemotronBlock
from quanta.nemotron.runtime import NemotronResidentModel


def make_stream_state(cfg: NemotronHConfig) -> tuple[list, list, list]:
    """A fresh per-stream state for the prefill+decode lifecycle: empty KV caches on attention
    layers, ``None`` ssm-state on mamba layers, ``None`` conv-state on mamba layers.

    The conv/ssm states stay ``None`` so the first call (the prefill) takes the **chunked SSD**
    path (the only path that handles ``T > 1``), and that path fills both states with the
    correct post-prefill values for the subsequent step path. Calling decode steps on a state
    with ``conv_state=None`` would loudly fail (the step branch requires a populated rolling
    window), which is the intended contract — the caller is expected to run :meth:`prefill`
    before :meth:`step_batch` (rule-6: no silent defaults)."""
    kinds = cfg.layers_block_type
    caches = [KVCache() if k == "attention" else None for k in kinds]
    ssm = [None] * len(kinds)
    conv = [None] * len(kinds)
    return caches, ssm, conv


def replicate_state(state: tuple[list, list, list], b: int) -> list[tuple[list, list, list]]:
    """Return ``b`` parallel per-stream state triples, each initially sharing the prefix tensors.

    Nemotron's "cache" is a three-tuple ``(caches, ssm, conv)`` rather than a single object (the
    KV caches per attention layer + the Mamba ``(ssm, conv)`` recurrence state per mamba layer).
    Each replica gets:

    * a NEW per-layer ``caches`` list whose entries are :meth:`KVCache._copy` (shares the immutable
      KV codes/scales by reference — subsequent ``update`` on a replica creates new concatenated
      arrays and leaves the original cache + siblings untouched).
    * a NEW per-layer ``ssm`` / ``conv`` list (the Python list spine cloned via ``list(...)``).
      Each entry references the prefix's Mamba state tensors (MLX arrays are immutable; assigning
      ``ssm_s[i] = new_state`` after a mixer step only mutates THIS replica's list).

    Zero copy cost — the per-token weight reads dominate; this is just bookkeeping to give each
    replica its own mutation surface so the B-wide verify amortizes the routed-MoE weight reads.
    Drives :func:`quanta.nemotron.spec.batch_verify` for ``spec_generate_tree(batched=True)``.

    Sibling-replica isolation gated model-free in
    ``parity/nemotron_batched_tree_verify_test.py``'s "replicate fidelity" + "replica divergence".
    """
    if b < 1:
        raise ValueError(f"replicate_state(B) requires B >= 1 (got {b})")
    caches, ssm, conv = state
    out: list[tuple[list, list, list]] = []
    for _ in range(b):
        caches_r = [c._copy() if c is not None else None for c in caches]
        ssm_r = list(ssm) if ssm is not None else None
        conv_r = list(conv) if conv is not None else None
        out.append((caches_r, ssm_r, conv_r))
    return out


def make_step_state(cfg: NemotronHConfig) -> tuple[list, list, list]:
    """A pre-step state for the decode-only test path: ``conv_state`` zero-initialised on mamba
    layers so the O(1) step path engages from token 0 (used by :mod:`parity.nemotron_decode_test`'s
    pattern — feeding tokens one at a time without a chunked prefill). The batched runtime
    proper uses :func:`make_stream_state` (None conv) + a prefill call instead."""
    kinds = cfg.layers_block_type
    conv0 = mx.zeros((1, cfg.conv_kernel - 1, cfg.mamba_conv_dim))
    caches = [KVCache() if k == "attention" else None for k in kinds]
    ssm = [None] * len(kinds)
    conv = [conv0 if k == "mamba" else None for k in kinds]
    return caches, ssm, conv


def batched_decode_step(
    layers: list[NemotronBlock],
    embed_w: mx.array,
    norm_f: mx.array,
    lm_head_w: mx.array,
    norm_eps: float,
    stream_token_ids: list[mx.array],
    stream_caches: list[tuple[list, list, list]],
) -> list[mx.array]:
    """One batched decode step across ``B = len(stream_token_ids)`` active streams.

    Standalone (no shared state with any wrapper class) so the same code can be exercised by the
    real resident model AND a tiny model-free random-init NemotronModel in the parity test. The
    per-layer pattern matches :meth:`NemotronResidentModel.__call__` exactly — only difference is
    Mamba/Attention iterate streams while MoE concatenates and runs once.

    ``stream_token_ids[b]``: ``mx.array`` shape ``[T_b]`` (typically ``T_b == 1`` for decode).
    ``stream_caches[b]``: ``(caches, ssm, conv)`` — KV caches grow in place, mamba state lists
    are mutated in place. Returns ``[1, T_b, vocab]`` per stream."""
    b = len(stream_token_ids)
    if b == 0:
        return []
    if len(stream_caches) != b:
        raise ValueError(f"stream_caches length {len(stream_caches)} != B={b}")

    # all T_b must match — refuse mixed lengths loudly so the stacked MoE call is well-defined
    # (rule-6: silent padding here would change a stream's per-row count and routing).
    tbs = [int(ids.shape[0]) for ids in stream_token_ids]
    if any(t != tbs[0] for t in tbs):
        raise ValueError(f"per-stream T_b lengths must match (got {tbs}); pad upstream")

    # per-stream initial hidden: [1, T_b, hidden] each.
    hs: list[mx.array] = [embed_w[ids][None].astype(mx.bfloat16) for ids in stream_token_ids]

    for i, blk in enumerate(layers):
        if blk.kind in ("mamba", "attention"):
            # per-stream state-local mixer call (per CLAUDE.md: a bounded coarse loop over
            # streams is allowed — it's an IO/state-accounting boundary, not a compute-hot loop
            # over tokens/experts/hidden). The MoE — the bandwidth amortizer — stays batched.
            new_hs: list[mx.array] = []
            for s in range(b):
                caches_s, ssm_s, conv_s = stream_caches[s]
                y_s, ssm_s[i], conv_s[i] = blk(hs[s], cache=caches_s[i], ssm_state=ssm_s[i],
                                                conv_state=conv_s[i], use_fast=True)
                new_hs.append(y_s)
            hs = new_hs
        elif blk.kind == "moe":  # STACK across streams — the bandwidth amortizer.
            stacked = mx.concatenate(hs, axis=0)             # [B, T, hidden]
            # NemotronBlock.__call__ does ``x + mixer(norm(x))`` and threads None ssm/conv for
            # stateless layers — identical numerics to per-stream on each row of the stack.
            out_stacked, _, _ = blk(stacked, cache=None, ssm_state=None, conv_state=None,
                                     use_fast=True)
            # split back to per-stream [1, T, hidden] — slicing produces views over the same
            # underlying buffer, so per-stream eval resolves to the row this stream contributed.
            hs = [out_stacked[s : s + 1] for s in range(b)]
        else:
            raise ValueError(f"unknown block kind {blk.kind!r}")

    # final norm + lm_head, per-stream (the head is shared weights — per-stream is cheap at
    # T_b=1; stacking here would do the same op but per-stream matches the single-stream return).
    out_logits: list[mx.array] = []
    for s in range(b):
        h_s = mx.fast.rms_norm(hs[s], norm_f.astype(hs[s].dtype), norm_eps)
        out_logits.append(h_s @ lm_head_w.T)
    return out_logits


class NemotronBatchedResidentModel:
    """Batched-serving wrapper around :class:`NemotronResidentModel` — same resident weights, B
    concurrent decode streams sharing them.

    Surface:

    * :meth:`step_batch` — one decode step across ``B ≤ max_batch`` active streams. Mamba +
      attention layers run per-stream (with each stream's own state); MoE layers run **once**
      over a stacked ``[B, 1, dim]`` hidden, then split back. Returns per-stream
      ``[1, T_b, vocab]`` logits.
    * :meth:`prefill` — single-stream prefill (delegated to the inner resident model's chunked
      prefill path). Use this once per prompt to fill that stream's caches before stepping.
    * :meth:`make_stream_state` — factory for a fresh per-stream ``(caches, ssm, conv)`` triple.
    """

    def __init__(self, art_dir: str | Path, *, max_batch: int = 32, n_layers: int | None = None) -> None:
        if max_batch <= 0:
            raise ValueError(f"max_batch must be positive, got {max_batch}")
        self.max_batch = int(max_batch)
        self._inner = NemotronResidentModel(art_dir, n_layers=n_layers)

    # --- shared-weight surface (mirrors NemotronResidentModel) -----------------
    @property
    def cfg(self) -> NemotronHConfig:
        return self._inner.cfg

    @property
    def num_layers(self) -> int:
        return self._inner.num_layers

    @property
    def layers(self) -> list[NemotronBlock]:
        return self._inner.layers

    @property
    def embed_w(self) -> mx.array:
        return self._inner.embed_w

    @property
    def norm_f(self) -> mx.array:
        return self._inner.norm_f

    @property
    def lm_head_w(self) -> mx.array:
        return self._inner.lm_head_w

    # --- single-stream prefill (delegate) --------------------------------------
    def prefill(self, prompt_ids: mx.array, state: tuple[list, list, list]) -> mx.array:
        """Run a single-stream chunked prefill, filling the given per-stream state in place.

        ``prompt_ids``: 1-D ``mx.array`` of token ids; ``state``: a fresh ``(caches, ssm, conv)``
        triple from :meth:`make_stream_state`. Returns ``[1, T, vocab]`` logits (same as the
        underlying resident model's ``__call__``). KV caches grow in place; the mamba ``ssm`` /
        ``conv`` lists are mutated in place by the inner model (the inner ``__call__`` writes
        each ``ssm[i]``/``conv[i]`` and returns the same list objects)."""
        caches, ssm, conv = state
        # compiled=False: the compiled per-layer mixers were tuned for t==1 decode; prefill is
        # t>=1 with a different shape signature, so go through the eager path (the inner runtime
        # already does this branch — passing compiled=False here is explicit/defensive).
        logits, _ssm_out, _conv_out = self._inner(prompt_ids, caches=caches, ssm=ssm, conv=conv,
                                                   use_fast=True, compiled=False)
        return logits

    # --- per-stream state factory ----------------------------------------------
    def make_stream_state(self) -> tuple[list, list, list]:
        return make_stream_state(self.cfg)

    # --- batched decode step ---------------------------------------------------
    def step_batch(
        self,
        stream_token_ids: list[mx.array],
        stream_caches: list[tuple[list, list, list]],
        offsets: list[int] | None = None,
    ) -> list[mx.array]:
        """One decode step across ``B = len(stream_token_ids)`` active streams.

        ``stream_token_ids[b]``: ``mx.array`` shape ``[T_b]`` (typically ``T_b == 1`` for decode;
        spec-decode verify can pass a small T_b > 1 — same per-stream path either way).
        ``stream_caches[b]``: per-stream ``(caches, ssm, conv)`` triple (mutated in place — the KV
        cache grows, the mamba ``ssm``/``conv`` lists get the post-step state).
        ``offsets``: accepted for API symmetry but ignored — each stream's offset comes from its
        own ``KVCache`` (the attention module reads ``cache.offset`` directly).

        Returns ``[logits_b for b in range(B)]`` where each ``logits_b`` is ``[1, T_b, vocab]``."""
        del offsets  # offsets accepted for API symmetry; KVCache.offset is the source of truth
        b = len(stream_token_ids)
        if b > self.max_batch:
            raise ValueError(f"B={b} exceeds max_batch={self.max_batch}")
        return batched_decode_step(
            layers=self.layers,
            embed_w=self.embed_w,
            norm_f=self.norm_f,
            lm_head_w=self.lm_head_w,
            norm_eps=self.cfg.norm_eps,
            stream_token_ids=stream_token_ids,
            stream_caches=stream_caches,
        )

    # --- shared-offset batched step for tree-spec batched verify ----------------
    def batch_step(
        self,
        tokens,
        *,
        replicas: list[tuple[list, list, list]],
        offset: int,
        capture_layer: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """One batched decode step with a SHARED offset across all ``B = len(tokens)`` streams —
        the verify shape :func:`quanta.nemotron.spec.batch_verify` drives for batched tree-spec
        (docs/batched_tree_verify.md). Returns ``(logits [B,1,vocab], captured [B,1,hidden] or None)``.

        Differences from :meth:`step_batch`:

        * single position (``T == 1``) per call — the spec's verify loops :math:`depth+1` calls;
        * every replica shares ``offset`` (built from the same prefix via :func:`replicate_state`,
          so they advance lock-step); the attention layers' per-replica KV caches must all already
          sit at this offset (rule 6: validated, no silent drift);
        * optionally captures one layer's residual ``[B, 1, hidden]`` for the MTP feature.

        Reuses the same per-stream mamba/attention + batched-MoE pattern as
        :func:`batched_decode_step`; the only new pieces are the shared-offset contract, the
        capture, and the canonical ``[B, 1, vocab]`` stacked-logits return (vs ``step_batch``'s
        per-stream list).
        """
        b = len(tokens)
        if b < 1:
            raise ValueError("batch_step needs >= 1 stream")
        if b > self.max_batch:
            raise ValueError(f"batch_step: B={b} exceeds max_batch={self.max_batch}")
        if len(replicas) != b:
            raise ValueError(
                f"batch_step: len(replicas)={len(replicas)} != len(tokens)={b}"
            )
        # Every replica's attention-layer KV must sit at the shared offset (rule 6 / no silent drift).
        for s, (caches_s, _ssm_s, _conv_s) in enumerate(replicas):
            for li, c in enumerate(caches_s):
                if c is None:
                    continue
                if c.offset != offset:
                    raise ValueError(
                        f"batch_step: replicas[{s}].caches[{li}].offset={c.offset} != "
                        f"offset={offset} (all B replicas must sit at the shared verify offset)"
                    )
        if capture_layer is not None and not 0 <= capture_layer < self.num_layers:
            raise ValueError(
                f"batch_step: capture_layer={capture_layer} not in [0, {self.num_layers})"
            )

        # Per-stream embed (each row a single token).
        ids_b = mx.array([int(t) for t in tokens], dtype=mx.int32)              # [B]
        h_b = self.embed_w[ids_b][:, None].astype(mx.bfloat16)                  # [B, 1, hidden]
        hs: list[mx.array] = [h_b[s:s + 1] for s in range(b)]                   # B × [1, 1, hidden]

        captured: mx.array | None = None
        for i, blk in enumerate(self.layers):
            if blk.kind in ("mamba", "attention"):
                # Per-stream state-local mixer call (bounded B-loop, rule-3 acceptable boundary).
                # Each replica's caches/ssm/conv mutate independently — siblings unaffected via
                # MLX immutability + per-replica list spine ownership.
                new_hs: list[mx.array] = []
                for s in range(b):
                    caches_s, ssm_s, conv_s = replicas[s]
                    y_s, ssm_s[i], conv_s[i] = blk(
                        hs[s], cache=caches_s[i], ssm_state=ssm_s[i], conv_state=conv_s[i],
                        use_fast=True,
                    )
                    new_hs.append(y_s)
                hs = new_hs
            elif blk.kind == "moe":
                # Stack across replicas → ONE blk call → split. The gather_qmm routed-MoE reads
                # each touched expert's weights once for all B rows that route to it — the win.
                stacked = mx.concatenate(hs, axis=0)                            # [B, 1, hidden]
                out_stacked, _, _ = blk(stacked, cache=None, ssm_state=None, conv_state=None,
                                        use_fast=True)
                hs = [out_stacked[s:s + 1] for s in range(b)]
            else:
                raise ValueError(f"unknown block kind {blk.kind!r}")

            # Optional residual capture after layer i — stacked across replicas for the MTP feature.
            if capture_layer is not None and i == capture_layer:
                captured = mx.concatenate(hs, axis=0)                           # [B, 1, hidden]

        # Final norm + lm_head over the stacked [B, 1, hidden] (single matmul, all replicas).
        stacked_h = mx.concatenate(hs, axis=0)                                  # [B, 1, hidden]
        stacked_h = mx.fast.rms_norm(stacked_h, self.norm_f.astype(stacked_h.dtype),
                                     self.cfg.norm_eps)
        logits = stacked_h @ self.lm_head_w.T.astype(stacked_h.dtype)           # [B, 1, vocab]
        return logits, captured
