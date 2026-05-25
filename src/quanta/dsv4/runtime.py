"""RAM-resident DeepSeek-V4-Flash runtime — load the baked artifact and run the resident forward.

Mirrors :class:`quanta.runtime.ResidentModel` (Kimi) and :class:`quanta.nemotron.runtime.NemotronResidentModel`
for DSV4-Flash. The model is built **one decoder layer at a time** (materialize that layer's params,
``mx.eval`` them, then drop the artifact's source shard handles before the next — rule-8), so peak
load residency is ~one layer, not the whole checkpoint. The deployment target is the full quantized
model pinned with ``mx.set_wired_limit``; ``n_layers`` builds a bounded prefix for validation.

Each layer's resident state is exactly the ``p`` dict that :func:`quanta.dsv4.model.load_block_params`
produces (attention + router + expert stacks + shared + norms + HC). Because :class:`DSV4Artifact`
(Unit 1) duck-types :class:`quanta.dsv4.loader.DeepSeekV4SourceCheckpoint`'s per-kind loader surface
(``block_norms`` / ``block_hc`` / ``attention`` / ``moe_router`` / ``shared_expert`` /
``expert_stacks``), ``load_block_params(art, ...)`` runs against the artifact unchanged.

``__call__`` has two regimes, matching the documented contract:

* **prefill** (``caches=None``): reuse the reference :func:`quanta.dsv4.model.dsv4_block` per layer (the
  exact code the forward parity gate ``parity/dsv4_forward_test.py`` validates), so the resident
  prefill is output-equivalent to the reference by construction — and ``capture_layers`` returns the
  per-layer HC residual stream from this path.
* **decode** (``caches`` given, a single token): one HC-wrapped block per layer — ``hc_pre`` →
  ``attn_norm`` → the sibling **attention** stepper (:func:`quanta.dsv4.decode.decode_step_dense` on
  ratio-0 layers, :func:`quanta.dsv4.decode.decode_step_compressed` otherwise, chosen by
  ``cfg.compress_ratio(layer_id)``) → ``hc_post`` → ``hc_pre`` → ``ffn_norm`` → ``dsv4_moe`` →
  ``hc_post``. The steppers are attention-only and output-equivalent to the prefill attention at the
  same absolute position; everything else (HC mixing, the MoE) is the same code the prefill block uses,
  so a single decode step equals ``dsv4_block`` at that position. The :class:`quanta.dsv4.decode.DSV4Cache`
  is mutated in place; ``offset`` is the new token's absolute position.

The decode cache is seeded from the prompt by :func:`quanta.dsv4.generate.generate` (it steps the
prompt through this same decode path — Unit 3 has no batched cache-fill), so the prompt-stepping and
generation share one parity-correct path. The int4/int8 packed-weight resident path (the bandwidth
win) plugs in by swapping the expert stacks / dense projections in the per-layer ``p`` dict; this
runtime's call convention does not change.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from quanta.dsv4.artifact import (
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    DSV4Artifact,
)
from quanta.dsv4.attention import _rms_w, rope_cos_sin
from quanta.dsv4.decode import decode_step_compressed, decode_step_dense
from quanta.dsv4.hyper import hc_expand, hc_head, hc_post, hc_pre
from quanta.dsv4.model import dsv4_block, load_block_params
from quanta.dsv4.moe import dsv4_moe


def _block_arrays(p: dict) -> list[mx.array]:
    """Every resident array of one layer's params dict (recurse into nested sub-dicts: attention holds
    ``compressor``/``indexer`` sub-dicts; experts/shared/router are flat dicts of stacks/tensors)."""
    out: list[mx.array] = []
    for v in p.values():
        if isinstance(v, dict):
            out.extend(_block_arrays(v))
        elif isinstance(v, mx.array):
            out.append(v)
    return out


class DSV4ResidentModel:
    """RAM-resident DeepSeek-V4-Flash decoder — prefill via the reference block, decode via the steppers.

    Built one layer at a time (materialize, then release the artifact's shard handles) for bounded
    load residency. ``n_layers`` builds a prefix for validation. ``__call__`` matches the contract the
    oMLX shim's DeepSeek stepper and :func:`quanta.dsv4.generate.generate` expect.
    """

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None) -> None:
        self.art = DSV4Artifact(art_dir)
        self.cfg = self.art.cfg
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[dict] = []
        for i in range(n):  # rule-8: materialize one layer's params, eval, then drop source shards
            p = load_block_params(self.art, self.cfg, i)
            mx.eval(_block_arrays(p))
            self.layers.append(p)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n  # for DSV4Cache sizing by the oMLX engine / generate

        # embed / final HC-head params / final norm / lm_head (bf16; head may be tied to embed)
        self.embed_w = self.art.read(EMBED_KEY)
        self.final_hc = {k: self.art.read(f"hc_head_{k}").astype(mx.float32)
                         for k in ("fn", "base", "scale")}
        self.norm_w = self.art.read(FINAL_NORM_KEY)
        head_key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        self.lm_head_w = self.art.read(head_key)
        mx.eval([self.embed_w, self.norm_w, self.lm_head_w, *self.final_hc.values()])
        self.art.release()
        mx.clear_cache()
        self._rope_cache: dict[int, tuple[mx.array, mx.array]] = {}  # per-layer grown RoPE tables

    def _head(self, h: mx.array) -> mx.array:
        """Final HC-head reduction -> RMSNorm -> lm_head: HC residual ``[B,S,hc,dim] -> [B,S,vocab]``."""
        hh = hc_head(h, self.final_hc["fn"], self.final_hc["scale"], self.final_hc["base"],
                     self.cfg.hc_mult, self.cfg.norm_eps, self.cfg.hc_eps)
        hh = _rms_w(hh, self.norm_w, self.cfg.norm_eps)
        return hh @ self.lm_head_w.T.astype(hh.dtype)

    def _decode_block(self, h: mx.array, p: dict, layer_id: int, cache, ids: mx.array,
                      cos: mx.array, sin: mx.array, offset: int) -> mx.array:
        """One single-token decoder block on the HC residual ``[1,1,hc,dim] -> [1,1,hc,dim]``.

        Mirrors :func:`quanta.dsv4.model.dsv4_block` exactly, but swaps the batched attention call for
        the sibling decode stepper (attention-only, cache-threaded) — so HC mixing and the MoE are the
        same code the prefill block uses and the step equals ``dsv4_block`` at this position.
        """
        cfg = self.cfg
        eps, hc, iters, heps = cfg.norm_eps, cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps
        step = decode_step_dense if cfg.compress_ratio(layer_id) == 0 else decode_step_compressed

        res = h
        x, post, comb = hc_pre(h, p["hc_attn_fn"], p["hc_attn_scale"], p["hc_attn_base"], hc, iters, eps, heps)
        x = _rms_w(x, p["attn_norm"], eps)
        x = step(x, p["attn"], cfg, layer_id, cache, cos, sin, offset)
        h = hc_post(x, res, post, comb)

        res = h
        x, post, comb = hc_pre(h, p["hc_ffn_fn"], p["hc_ffn_scale"], p["hc_ffn_base"], hc, iters, eps, heps)
        x = _rms_w(x, p["ffn_norm"], eps)
        x = dsv4_moe(x, p["router"], p["experts"], p["shared"], cfg, layer_id, ids)
        h = hc_post(x, res, post, comb)
        return h

    def _rope(self, layer_id: int, length: int) -> tuple[mx.array, mx.array]:
        """Full per-layer RoPE ``(cos, sin)`` for absolute positions ``[0, length)`` (the decode
        stepper slices ``[offset:offset+1]`` and the compressor reads earlier window-start rows). Cached
        and grown per layer: each row ``k`` is ``f(k)`` independent of the total length (YaRN ``freqs``
        are length-independent), so a longer table extends the shorter one — caching avoids rebuilding
        ``O(offset)`` rows every step. Per-layer because the regime sets theta/YaRN."""
        cached = self._rope_cache.get(layer_id)
        if cached is not None and cached[0].shape[0] >= length:
            return cached
        orig, theta = self.cfg.attn_rope(layer_id)
        cos, sin = rope_cos_sin(self.cfg.rope_head_dim, length, orig, theta,
                                self.cfg.rope_factor, self.cfg.beta_fast, self.cfg.beta_slow)
        self._rope_cache[layer_id] = (cos, sin)
        return cos, sin

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None):
        """Logits ``[1,T,vocab]`` (or ``(logits, {layer: hidden})`` when ``capture_layers`` is set).

        ``caches=None`` ⇒ prefill (reuse the reference ``dsv4_block`` per layer — parity-correct).
        ``caches`` given ⇒ decode over ``T >= 1`` tokens, each stepped per layer through the
        HC-wrapped attention steppers (regime chosen per layer by ``cfg.compress_ratio``). Every token
        completes ALL layers — appending its KV — before the next begins, so the run is causally
        identical to a batched cached forward; the cache is mutated in place and ``offset`` is the first
        token's absolute position. ``T == 1`` is the plain decode step (:func:`quanta.dsv4.generate`);
        ``T > 1`` is how :func:`quanta.dsv4.spec.spec_generate` seeds the prompt and verifies
        ``[cur, draft]`` in one call. ``capture_layers`` returns each captured layer's HC residual
        stacked over the ``T`` positions as ``[T, hc, dim]`` (same shape the prefill path returns).
        """
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        ids = ids.reshape(1, -1)                                          # [1, T]
        h = hc_expand(self.embed_w[ids].astype(mx.bfloat16), self.cfg.hc_mult)  # [1,T,hc,dim]

        if caches is not None:
            cap_set = set(capture_layers) if capture_layers else set()
            caps_acc: dict[int, list[mx.array]] = {layer: [] for layer in cap_set}
            hts: list[mx.array] = []
            for t in range(h.shape[1]):
                ht = h[:, t:t + 1]                               # [1,1,hc,dim] token t
                idt = ids[:, t:t + 1]                            # [1,1] its id (MoE hash routing)
                off_t = offset + t                               # its absolute position
                for i, p in enumerate(self.layers):
                    cos, sin = self._rope(i, off_t + 1)
                    ht = self._decode_block(ht, p, i, caches, idt, cos, sin, off_t)
                    if i in cap_set:
                        caps_acc[i].append(ht[0, 0])             # [hc, dim] residual after layer i
                mx.eval(ht)                                      # materialize this token's cache growth
                hts.append(ht)
            h = hts[0] if len(hts) == 1 else mx.concatenate(hts, axis=1)  # [1,T,hc,dim]
            logits = self._head(h)                              # [1,T,vocab]
            if cap_set:
                return logits, {layer: mx.stack(v, axis=0) for layer, v in caps_acc.items()}
            return logits

        cap_set = set(capture_layers) if capture_layers else set()
        caps: dict[int, mx.array] = {}
        for i, p in enumerate(self.layers):
            h = dsv4_block(h, p, self.cfg, i, ids)
            if i in cap_set:
                caps[i] = h[0]  # [T, hc, dim] HC residual after layer i
            mx.eval(h)
        logits = self._head(h)
        if cap_set:
            return logits, caps
        return logits
