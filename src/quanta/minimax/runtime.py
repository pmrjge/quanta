"""RAM-resident MiniMax-M2.7 runtime — load the baked artifact and run the resident forward.

Mirrors :class:`quanta.dsv4.runtime.DSV4ResidentModel` (the proven serving template) and
:class:`quanta.nemotron.runtime.NemotronResidentModel` for MiniMax-M2.7. The model is built **one
decoder layer at a time** (materialize that layer's params, ``mx.eval`` them, then drop the
artifact's source shard handles before the next — rule 8), so peak load residency is ~one layer, not
the whole checkpoint. The deployment target is the full quantized model pinned with
``mx.set_wired_limit``; ``n_layers`` builds a bounded prefix for validation.

Each layer is a :class:`quanta.minimax.model.MiniMaxBlock` (``nn.RMSNorm`` + the GQA
:class:`quanta.minimax.attention.MiniMaxAttention` with per-layer QK-norm, + the routed expert
stacks dispatched by :func:`quanta.minimax.moe.minimax_moe`), filled by
:func:`quanta.minimax.model.load_block` straight from the artifact (which duck-types the source
checkpoint's ``attention`` / ``block_norms`` / ``moe`` surface). There is **no shared expert**.

``__call__`` has two regimes, matching the documented contract:

* **prefill** (``caches=None``): reuse the reference :meth:`MiniMaxBlock.__call__` per layer (the
  exact code ``parity/minimax_forward_test.py`` validates), so the resident prefill is
  output-equivalent to the reference by construction — and ``capture_layers`` returns the per-layer
  residual stream from this path.
* **decode** (``caches`` given, ``T >= 1`` tokens): each token is stepped through every layer's
  :meth:`MiniMaxBlock.__call__` with that layer's growing K/V cache (``caches[i]``). The attention
  sub-module's cache path is output-equivalent to the prefill attention at the same absolute position
  (it ropes/QK-norms the new q/k at ``cache.offset``, appends K/V, and SDPAs over the cached KV), and
  the MoE + residuals are the same code the prefill block uses — so one decode step equals
  ``MiniMaxBlock`` at that position. The :class:`quanta.minimax.decode.MiniMaxCache` is mutated in
  place; ``offset`` is the first token's absolute position.

The decode cache is seeded from the prompt by :func:`quanta.minimax.generate.generate` (it steps the
prompt through this same decode path), so prompt-stepping and generation share one parity-correct
path. The int8/int6 packed-weight resident path (the bandwidth win) plugs in by swapping the expert
stacks / attention projections inside each block's params; this runtime's call convention does not
change.

**No native MTP / spec-decode for MiniMax.** MiniMax-M2.7 ships **no** multi-token-prediction head
(62 decoder layers, max source layer index 61, zero ``nextn``/``mtp``/``eh_proj`` keys in the
checkpoint index — the config advertises ``use_mtp`` but the module weights are absent). There is
therefore no native MTP spec-decode path here (no ``mtp.py`` / ``spec.py``); a future EAGLE drafter
would be the only speculative path and is out of scope for this runtime.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from quanta.minimax.artifact import MiniMaxArtifact
from quanta.minimax.config import MiniMaxConfig
from quanta.minimax.decode import MiniMaxCache
from quanta.minimax.model import MiniMaxBlock, load_block


def _block_arrays(blk: MiniMaxBlock) -> list[mx.array]:
    """Every resident array of one layer. ``MiniMaxBlock`` assigns the MoE router + expert stacks as
    bare ``mx.array`` attributes, which ``nn.Module`` already registers as parameters (verified), so
    ``tree_flatten(blk.parameters())`` covers the norms, the attention projections / QK-norms, AND the
    router ``gate_weight`` / ``e_score_correction_bias`` / ``w1`` / ``w3`` / ``w2`` — one materialize
    pass over the whole layer."""
    return [v for _, v in tree_flatten(blk.parameters())]


class MiniMaxResidentModel:
    """RAM-resident MiniMax-M2.7 decoder — prefill via the reference block, decode via the cache path.

    Built one layer at a time (materialize, then release the artifact's shard handles) for bounded
    load residency. ``n_layers`` builds a prefix for validation. ``__call__`` matches the contract the
    oMLX shim's stepper and :func:`quanta.minimax.generate.generate` expect.
    """

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None) -> None:
        self.art = MiniMaxArtifact(art_dir)
        self.cfg: MiniMaxConfig = self.art.cfg
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[MiniMaxBlock] = []
        for i in range(n):  # rule-8: materialize one layer's params, eval, then drop source shards
            blk = MiniMaxBlock(self.cfg, i)
            load_block(blk, self.art, self.cfg, i)
            mx.eval(_block_arrays(blk))
            self.layers.append(blk)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n  # for MiniMaxCache sizing by the oMLX engine / generate

        # embed / final norm / lm_head (bf16; head may be tied to embed)
        self.embed_w = self.art.embed().astype(mx.bfloat16)
        self.norm_w = self.art.final_norm().astype(mx.bfloat16)
        self.lm_head_w = self.art.lm_head().astype(mx.bfloat16)
        mx.eval([self.embed_w, self.norm_w, self.lm_head_w])
        self.art.release()
        mx.clear_cache()

    def _head(self, h: mx.array) -> mx.array:
        """Final RMSNorm -> lm_head: residual ``[B,S,hidden] -> [B,S,vocab]`` (float32 norm + logits)."""
        hh = mx.fast.rms_norm(h.astype(mx.float32), self.norm_w.astype(mx.float32), self.cfg.norm_eps)
        return hh @ self.lm_head_w.astype(mx.float32).T

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        """Logits ``[1,T,vocab]`` (or ``(logits, {layer: hidden})`` when ``capture_layers`` is set).

        ``caches=None`` ⇒ prefill (reuse the reference ``MiniMaxBlock`` per layer — parity-correct).
        ``caches`` given ⇒ decode over ``T >= 1`` tokens, each stepped per layer through the block's
        cache path. Every token completes ALL layers — appending its K/V — before the next begins, so
        the run is causally identical to a batched cached forward; the cache is mutated in place and
        ``offset`` is the first token's absolute position. ``T == 1`` is the plain decode step
        (:func:`quanta.minimax.generate`); ``T > 1`` seeds the prompt / verifies a draft in one call.
        ``capture_layers`` returns each captured layer's post-layer residual over the ``T`` positions
        as ``[T, hidden]`` (the same shape the prefill path returns), identically in both regimes.
        """
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        ids = ids.reshape(1, -1)                                          # [1, T]
        h = self.embed_w[ids].astype(mx.bfloat16)                        # [1, T, hidden]

        if caches is not None:
            if len(caches) != self.num_layers:
                raise ValueError(f"caches has {len(caches)} layers, model has {self.num_layers}")
            cap_set = set(capture_layers) if capture_layers else set()
            caps_acc: dict[int, list[mx.array]] = {layer: [] for layer in cap_set}
            hts: list[mx.array] = []
            for t in range(h.shape[1]):                                   # bounded: over tokens
                ht = h[:, t:t + 1]                                        # [1,1,hidden] token t
                for i, blk in enumerate(self.layers):
                    ht = blk(ht, cache=caches[i], use_fast=True)
                    if i in cap_set:
                        caps_acc[i].append(ht[0, 0])                      # [hidden] residual after layer i
                mx.eval(ht)                                               # materialize this token's cache growth
                hts.append(ht)
            h = hts[0] if len(hts) == 1 else mx.concatenate(hts, axis=1)  # [1,T,hidden]
            logits = self._head(h)                                        # [1,T,vocab]
            if cap_set:
                return logits, {layer: mx.stack(v, axis=0) for layer, v in caps_acc.items()}
            return logits

        cap_set = set(capture_layers) if capture_layers else set()
        caps: dict[int, mx.array] = {}
        for i, blk in enumerate(self.layers):
            h = blk(h, use_fast=True)
            if i in cap_set:
                caps[i] = h[0]                                            # [T, hidden] residual after layer i
            mx.eval(h)
        logits = self._head(h)
        if cap_set:
            return logits, caps
        return logits


def make_cache(model: MiniMaxResidentModel) -> MiniMaxCache:
    """A fresh decode cache sized to ``model`` (one growing K/V cache per layer). int8 KV by default
    — matches the Kimi pattern (MLACache(quantized=True) since #47) so serving runs use the int8
    storage win at long context. For bf16 (e.g. tight parity vs the reference), construct
    ``MiniMaxCache(n, quantized=False)`` directly."""
    return MiniMaxCache(model.num_layers, quantized=True)
