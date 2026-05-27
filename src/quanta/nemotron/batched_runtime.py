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
