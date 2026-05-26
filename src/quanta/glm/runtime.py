"""RAM-resident GLM-5.1 (``glm_moe_dsa``) runtime — load the baked artifact and run the resident forward.

Mirrors :class:`quanta.dsv4.runtime.DSV4ResidentModel` for GLM-5.1. The model is built **one decoder
layer at a time** (materialize that layer's params via :func:`quanta.glm.model.load_block`, ``mx.eval``
them, then drop the artifact's source shard handles before the next — rule 8), so peak load residency is
~one layer, not the whole checkpoint. The deployment target is the full quantized model pinned with
``mx.set_wired_limit``; ``n_layers`` builds a bounded prefix for validation.

Each layer is a :class:`quanta.glm.model.GLMDecoderLayer` (an ``nn.Module``) whose ``__call__`` is the
parity-gated prefill block and whose ``step`` is the parity-gated single-token decode stepper
(``incremental == prefill``; see ``parity/glm_forward_test.py`` / ``parity/glm_decode_attn_test.py``).
Because :class:`quanta.glm.artifact.GLMArtifact` duck-types
:class:`quanta.glm.loader.GLMSourceCheckpoint`'s per-kind accessor surface (``block_norms`` /
``attention`` / ``moe_router`` / ``shared_expert`` / ``expert_stacks`` / ``dense_mlp``),
:func:`load_block(art, cfg, i)` runs against the artifact unchanged — the resident layer is *the same
code* the bf16 reference forward uses.

``__call__`` has two regimes, matching the documented contract:

* **prefill** (``caches=None``): run each layer's reference ``__call__`` over the window
  ``positions = [offset, offset+T)``, so the resident prefill is output-equivalent to the reference
  forward by construction — and ``capture_layers`` returns the per-layer post-block residual ``[T,dim]``.
* **decode** (``caches`` given, ``T >= 1`` tokens): every token completes ALL layers — appending its
  MLA latent KV and advancing the DSA indexer key stream via :class:`quanta.glm.decode.GLMCache` —
  before the next begins, threading the absolute ``offset``. ``T == 1`` is plain decode
  (:func:`quanta.glm.generate.generate`); ``T > 1`` is how :func:`quanta.glm.spec.spec_generate` seeds
  the prompt and verifies ``[cur, draft]`` in one call. ``capture_layers`` returns the SAME residual the
  prefill path returns at those positions (the MTP/spec head feature), stacked as ``[T,dim]``.

The native MTP head (layer ``num_hidden_layers``) is loaded here (``with_mtp``, default on for the full
stack) and exposed via :meth:`mtp_head` so :mod:`quanta.glm.spec` drafts with the checkpoint's own head.
The int4/int8 packed-weight resident path (the bandwidth win) plugs in by swapping the gather path
inside :class:`quanta.glm.moe.SparseMoE`; this runtime's call convention does not change.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from quanta.glm.artifact import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    GLMArtifact,
)
from quanta.glm.decode import GLMCache
from quanta.glm.model import GLMDecoderLayer, _rms_w, load_block
from quanta.glm.mtp import MTPHead, build_mtp_layer


class GLMResidentModel:
    """RAM-resident GLM-5.1 decoder — prefill via the reference block, decode via the cache-threaded step.

    Built one layer at a time (materialize, then release the artifact's shard handles) for bounded load
    residency. ``n_layers`` builds a prefix for validation. ``__call__`` matches the contract
    :func:`quanta.glm.generate.generate` / :func:`quanta.glm.spec.spec_generate` (and the oMLX shim)
    expect."""

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None, use_fast: bool = True,
                 use_indexer: bool = True, with_mtp: bool | None = None,
                 dtype: mx.Dtype = mx.bfloat16) -> None:
        self.art = GLMArtifact(art_dir)
        self.cfg = self.art.cfg
        self.use_fast = use_fast
        self.use_indexer = use_indexer
        self._dtype = dtype
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[GLMDecoderLayer] = []
        for i in range(n):  # rule 8: build + eval one layer's params, then drop the source shard handles
            layer = load_block(self.art, self.cfg, i, dtype)
            mx.eval(layer.parameters())
            self.layers.append(layer)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n  # for GLMCache sizing by the oMLX engine / generate

        # embed / final norm / lm_head (bf16; head may be tied to the embedding)
        self.embed_w = self.art.read(EMBED_KEY).astype(dtype)
        self.norm_w = self.art.read(FINAL_NORM_KEY).astype(dtype)
        head_key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        self.lm_head_w = self.art.read(head_key).astype(dtype)
        mx.eval([self.embed_w, self.norm_w, self.lm_head_w])
        self.art.release()
        mx.clear_cache()

        # native MTP head (a full MoE decoder block at layer num_hidden_layers) — only for the full stack
        want_mtp = (n_layers is None) if with_mtp is None else with_mtp
        self._mtp: MTPHead | None = None
        if want_mtp and self.cfg.num_nextn_predict_layers > 0:
            self._load_mtp(dtype)

    def _load_mtp(self, dtype: mx.Dtype) -> None:
        """Materialize the native MTP block + its combine/readout params (rule 8: dropped after eval)."""
        p = self.art.mtp(0)
        layer = build_mtp_layer(p, self.cfg, dtype)
        combine = {
            "enorm": p["enorm"].astype(dtype),
            "hnorm": p["hnorm"].astype(dtype),
            "eh_proj": p["eh_proj"].astype(dtype),
            "shared_head_norm": p["shared_head_norm"].astype(dtype),
        }
        mx.eval([layer.parameters(), *combine.values()])
        self._mtp = MTPHead(combine, layer, self.cfg, use_fast=self.use_fast,
                            use_indexer=self.use_indexer)
        self.art.release()
        mx.clear_cache()

    def mtp_head(self) -> MTPHead:
        """The native MTP drafter (``mtp(prev_hidden, next_ids, embed, head) -> logits``); fails loud
        (rule 6) if it was not loaded (``with_mtp=False`` or a partial-stack build has no MTP block)."""
        if self._mtp is None:
            raise ValueError("MTP head not loaded (build with with_mtp=True over the full stack)")
        return self._mtp

    def make_caches(self) -> GLMCache:
        """A fresh per-layer decode cache (MLA KV + DSA indexer key). :mod:`quanta.glm.generate` /
        :mod:`quanta.glm.spec` prefer this factory so the resident model is self-contained. int8 MLA
        latent by default — matches the Kimi pattern (MLACache(quantized=True) since #47); pass
        ``GLMCache(n, quantized=False)`` directly for bf16."""
        return GLMCache(self.num_layers, quantized=True)

    def _head(self, h: mx.array) -> mx.array:
        """Final RMSNorm → lm_head: residual ``[B,T,dim] -> [B,T,vocab]``."""
        hh = _rms_w(h, self.norm_w, self.cfg.rms_norm_eps)
        return hh @ self.lm_head_w.T.astype(hh.dtype)

    def __call__(self, token_ids, *, caches=None, offset: int = 0, capture_layers=None):
        """Logits ``[1,T,vocab]`` (or ``(logits, {layer: hidden})`` when ``capture_layers`` is set).

        ``caches=None`` ⇒ prefill (reuse each layer's reference ``__call__`` over
        ``positions = [offset, offset+T)`` — parity-correct by construction). ``caches`` given ⇒ decode
        over ``T >= 1`` tokens, each stepped per layer through the cache-threaded
        :meth:`quanta.glm.model.GLMDecoderLayer.step` (output-equivalent to prefill at that absolute
        position). Every token completes ALL layers — appending its KV / advancing the indexer — before
        the next begins, so the run is causally identical to a batched cached forward; the cache is
        mutated in place and ``offset`` is the first token's absolute position. ``capture_layers``
        returns each captured layer's post-block residual stacked over the ``T`` positions as ``[T,dim]``
        (the same shape and values the prefill path returns at those positions)."""
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        ids = ids.reshape(1, -1)                                          # [1, T]
        t = ids.shape[1]
        h = self.embed_w[ids].astype(self._dtype)                        # [1,T,dim]
        want = set(capture_layers) if capture_layers else set()

        if caches is not None:
            caps_acc: dict[int, list[mx.array]] = {layer: [] for layer in want}
            cols: list[mx.array] = []
            for k in range(t):                                           # coarse bounded per-token loop
                h_col = h[:, k:k + 1]                                    # [1,1,dim] token k
                off_k = offset + k                                       # its absolute position
                for i, layer in enumerate(self.layers):
                    lc = caches[i]
                    h_col = layer.step(h_col, lc, off_k, use_fast=self.use_fast,
                                       use_indexer=self.use_indexer)
                    if i in want:
                        caps_acc[i].append(h_col[0, 0])                  # [dim] residual after layer i
                mx.eval(h_col)                                          # materialize this token's KV growth
                cols.append(h_col)
            h = cols[0] if t == 1 else mx.concatenate(cols, axis=1)     # [1,T,dim]
            logits = self._head(h)                                       # [1,T,vocab]
            if want:
                return logits, {layer: mx.stack(v, axis=0) for layer, v in caps_acc.items()}
            return logits

        positions = mx.arange(offset, offset + t)
        caps: dict[int, mx.array] = {}
        for i, layer in enumerate(self.layers):
            h = layer(h, positions, use_fast=self.use_fast, use_indexer=self.use_indexer)
            if i in want:
                caps[i] = h[0]                                           # [T, dim] residual after layer i
            mx.eval(h)
        logits = self._head(h)
        if want:
            return logits, caps
        return logits
