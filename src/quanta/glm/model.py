"""GLM-5.1 (``glm_moe_dsa``) full decoder forward (MLX) — embed → 78 blocks → norm → logits.

Assembles the validated GLM components (:mod:`quanta.glm.attention` MLA, :mod:`quanta.glm.indexer` DSA
Lightning-Indexer, :mod:`quanta.glm.moe` noaux_tc MoE + shared) into the DeepSeek-V3-style decoder,
grounded in :mod:`quanta.glm.config` / :mod:`quanta.glm.loader`:

* ``h = embed[ids]`` — a plain residual stream ``[B,T,dim]`` (no Hyper-Connections; GLM is standard
  pre-norm DeepSeek-V3).
* each block (``GLMDecoderLayer``): ``h += attn(input_layernorm(h))`` then ``h += ffn(post_attention_
  layernorm(h))``, where ``ffn`` is the **dense SwiGLU MLP** for the first ``first_k_dense_replace=3``
  layers and the **sparse MoE** otherwise (``cfg.is_dense_layer``); the indexer (when active) feeds the
  attention a top-``index_topk`` additive mask.
* ``logits = rmsnorm(h, final_norm) @ lm_head.T``.

Two consumers, one set of layers:

* :func:`glm_logits` — the **streamed bf16 reference forward** (the parity / teacher-forced-ppl
  vehicle): it materializes **one layer's** params at a time via
  :class:`quanta.glm.loader.GLMSourceCheckpoint` (rule 8 — the per-layer expert stacks are the memory
  peak, freed before the next layer) and runs the naive (or fast) path. This is what
  ``parity/glm_forward_test.py`` diffs naive-vs-fast on a tiny config.
* :class:`GLMResidentModel` — an ``nn.Module`` exposing the single-token decode contract
  ``model(token_ids, *, caches, offset, capture_layers) -> logits`` (or ``(logits, {layer: hidden})``)
  that :mod:`quanta.glm.generate` / :mod:`quanta.glm.spec` consume; it threads a per-layer KV cache and
  is incremental-equal to the prefill path (proven model-free in ``parity/glm_forward_test.py``).

The real bf16 teacher-forced-ppl gate (the true arbiter) loads the full checkpoint and is **deferred**
to a GPU session — its invocation is sketched at the bottom of this module's docstring, not run here::

    # DEFERRED (needs the ~1.5 TB bf16 checkpoint + GPU; do NOT run in a model-free session):
    #   from quanta.glm.config import GLMConfig
    #   from quanta.glm.loader import GLMSourceCheckpoint
    #   from quanta.glm.model import glm_teacher_forced_ppl
    #   cfg = GLMConfig.from_pretrained("/Users/pmrj/models/GLM-5.1")
    #   ck = GLMSourceCheckpoint("/Users/pmrj/models/GLM-5.1", cfg)
    #   ppl = glm_teacher_forced_ppl(ck, ids, cfg)   # ids [1,S] with the correct BOS
    # Arbiter = teacher-forced ppl on real prose + top-1 next-token agreement vs the bf16 reference,
    # NOT greedy generation and NOT per-expert reconstruction error (see CLAUDE.md Settled Findings).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.glm.attention import MLAAttention
from quanta.glm.config import GLMConfig
from quanta.glm.indexer import LightningIndexer
from quanta.glm.moe import SparseMoE


def _rms_w(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """Weighted RMSNorm over the last dim (fp32 accumulation, like ``nn.RMSNorm`` / HF)."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (w.astype(mx.float32) * xf).astype(x.dtype)


class DenseMLP(nn.Module):
    """Dense SwiGLU MLP (DeepSeek-V3 ``DeepseekV3MLP``) — the first ``first_k_dense_replace`` layers."""

    def __init__(self, cfg: GLMConfig) -> None:
        super().__init__()
        inter = cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class GLMDecoderLayer(nn.Module):
    """One GLM decoder block: pre-norm residual around MLA (+ DSA indexer) and dense/MoE FFN."""

    def __init__(self, cfg: GLMConfig, layer_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = MLAAttention(cfg)
        self.indexer = LightningIndexer(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = DenseMLP(cfg) if cfg.is_dense_layer(layer_id) else SparseMoE(cfg)

    def __call__(self, h: mx.array, positions: mx.array, *, use_fast: bool = False,
                 use_indexer: bool = True) -> mx.array:
        """Prefill block. ``h``: ``[B,T,dim]``; ``positions``: absolute pos ``[T]``. Returns ``[B,T,dim]``."""
        x = self.input_layernorm(h)
        q_latent = self.self_attn.q_a_layernorm(self.self_attn.q_a_proj(x))
        mask = self.indexer.select_mask(x, q_latent, positions, use_fast=use_fast) if use_indexer else None
        h = h + self.self_attn(x, positions, use_fast=use_fast, index_mask=mask)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h

    def step(self, h_t: mx.array, layer_cache, offset: int, *, use_fast: bool = False,
             use_indexer: bool = True) -> mx.array:
        """One decode token through the block at absolute position ``offset`` (incremental == prefill).
        ``layer_cache`` holds the MLA KV latent cache (``.kv``) and the indexer key cache (``.idx``).

        The indexer appends its key and produces the additive ``[B,1,S]`` selection mask, then the MLA
        appends its KV and attends under that mask (both caches grow in lock-step, so ``S`` matches)."""
        x = self.input_layernorm(h_t)
        mask = None
        if use_indexer:
            q_latent = self.self_attn.q_a_layernorm(self.self_attn.q_a_proj(x))
            mask = self.indexer.step_mask(x, q_latent, layer_cache.idx, offset, use_fast=use_fast)
        a = self.self_attn.step(x, layer_cache.kv, offset, use_fast=use_fast, index_mask=mask)
        h_t = h_t + a
        h_t = h_t + self.mlp(self.post_attention_layernorm(h_t))
        return h_t


# --- streamed bf16 reference forward (parity / ppl vehicle, one layer resident) ----------------------
def load_block(ck, cfg: GLMConfig, layer_id: int, dtype: mx.Dtype = mx.bfloat16) -> GLMDecoderLayer:
    """Build one decoder block and load its params from the streamed checkpoint (rule 8: only this
    layer's weights resident; the caller drops it before the next)."""
    layer = GLMDecoderLayer(cfg, layer_id)
    a = ck.attention(layer_id)
    ix = a["indexer"]
    norms = ck.block_norms(layer_id)
    upd: dict[str, mx.array] = {
        "input_layernorm.weight": norms["input_layernorm"],
        "post_attention_layernorm.weight": norms["post_attention_layernorm"],
        "self_attn.q_a_proj.weight": a["q_a_proj"],
        "self_attn.q_a_layernorm.weight": a["q_a_layernorm"],
        "self_attn.q_b_proj.weight": a["q_b_proj"],
        "self_attn.kv_a_proj_with_mqa.weight": a["kv_a_proj_with_mqa"],
        "self_attn.kv_a_layernorm.weight": a["kv_a_layernorm"],
        "self_attn.kv_b_proj.weight": a["kv_b_proj"],
        "self_attn.o_proj.weight": a["o_proj"],
        "indexer.wq_b.weight": ix["wq_b"],
        "indexer.wk.weight": ix["wk"],
        "indexer.weights_proj.weight": ix["weights_proj"],
        "indexer.k_norm.weight": ix["k_norm_weight"],
        "indexer.k_norm.bias": ix["k_norm_bias"],
    }
    layer.load_weights([(k, v.astype(dtype)) for k, v in upd.items()], strict=False)
    if cfg.is_dense_layer(layer_id):
        dm = ck.dense_mlp(layer_id)
        layer.mlp.load_weights([(f"{p}.weight", dm[p].astype(dtype))
                                for p in ("gate_proj", "up_proj", "down_proj")], strict=False)
    else:
        r = ck.moe_router(layer_id)
        sh = ck.shared_expert(layer_id)
        es = ck.expert_stacks(layer_id)
        layer.mlp.gate.weight = r["weight"].astype(dtype)
        layer.mlp.gate.e_score_correction_bias = r["e_score_correction_bias"].astype(mx.float32)
        layer.mlp.load_weights([
            ("shared_gate.weight", sh["gate_proj"].astype(dtype)),
            ("shared_up.weight", sh["up_proj"].astype(dtype)),
            ("shared_down.weight", sh["down_proj"].astype(dtype)),
        ], strict=False)
        layer.mlp.set_experts(es["gate_proj"].astype(dtype), es["up_proj"].astype(dtype),
                              es["down_proj"].astype(dtype))
    return layer


def glm_logits(ck, ids: mx.array, cfg: GLMConfig, *, dtype: mx.Dtype = mx.bfloat16,
               use_fast: bool = False, use_indexer: bool = True) -> mx.array:
    """Teacher-forced logits ``[B,S,vocab]`` for token ids ``[B,S]`` (streamed, one layer resident)."""
    b, s = ids.shape
    h = ck.embed()[ids].astype(dtype)                                 # [B,S,dim]
    positions = mx.arange(s)
    for layer_id in range(cfg.num_hidden_layers):
        layer = load_block(ck, cfg, layer_id, dtype)
        h = layer(h, positions, use_fast=use_fast, use_indexer=use_indexer)
        mx.eval(h)                                                    # free this layer's expert stacks
    h = _rms_w(h, ck.final_norm().astype(dtype), cfg.rms_norm_eps)
    return h @ ck.lm_head().astype(h.dtype).T


def glm_teacher_forced_ppl(ck, ids: mx.array, cfg: GLMConfig, *, dtype: mx.Dtype = mx.bfloat16) -> float:
    """Mean teacher-forced perplexity of ``ids`` ``[1,S]`` (next-token CE over positions 0..S-2).

    DEFERRED heavy gate — needs the full bf16 checkpoint + GPU; see this module's docstring. Kept here
    so the arbiter has a single canonical entry point for the later GPU session."""
    logits = glm_logits(ck, ids, cfg, dtype=dtype).astype(mx.float32)[0]   # [S,vocab]
    tgt = ids[0, 1:]
    lse = mx.logsumexp(logits[:-1], axis=-1)
    tok = mx.take_along_axis(logits[:-1], tgt[:, None], axis=-1)[:, 0]
    return float(mx.exp(mx.mean(lse - tok)).item())


# --- resident decode model (the generate/spec contract) ---------------------------------------------
class _LayerKVCache:
    """Per-layer MLA latent KV cache: the post-layernorm latent ``c_kv`` ``[B,S,kv_lora]`` and the
    roped single-head ``k_pe`` ``[B,1,S,rope]`` (mirrors :class:`quanta.cache.MLACache`, bf16-only)."""

    __slots__ = ("c_kv", "k_pe")

    def __init__(self) -> None:
        self.c_kv: mx.array | None = None
        self.k_pe: mx.array | None = None

    def update(self, c_kv_new: mx.array, k_pe_new: mx.array) -> tuple[mx.array, mx.array]:
        self.c_kv = c_kv_new if self.c_kv is None else mx.concatenate([self.c_kv, c_kv_new], axis=1)
        self.k_pe = k_pe_new if self.k_pe is None else mx.concatenate([self.k_pe, k_pe_new], axis=2)
        return self.c_kv, self.k_pe


class _IndexKeyCache:
    """Per-layer DSA indexer key cache: the roped index keys ``[B,1,S,index_head_dim]`` (single MQA
    head). All cached keys are causally valid (one appended per consumed token)."""

    __slots__ = ("k",)

    def __init__(self) -> None:
        self.k: mx.array | None = None

    def update(self, k_new: mx.array) -> mx.array:
        self.k = k_new if self.k is None else mx.concatenate([self.k, k_new], axis=2)
        return self.k


class _LayerCache:
    """A decoder layer's decode state: the MLA KV cache (``.kv``) and the indexer key cache (``.idx``),
    which grow in lock-step (one position each per token)."""

    __slots__ = ("kv", "idx")

    def __init__(self) -> None:
        self.kv = _LayerKVCache()
        self.idx = _IndexKeyCache()


class GLMDecodeCache:
    """Decode cache for the GLM attention stack: one :class:`_LayerCache` per decoder layer.

    Matches the ``offset`` / ``truncate(length)`` ergonomics :mod:`quanta.glm.spec` /
    :mod:`quanta.glm.generate` expect (and :class:`quanta.cache.MLACache` / the DSV4 cache): every
    layer advances in lock-step, so ``offset`` reads any populated layer; ``truncate`` slices the
    per-position streams so the kept prefix is bit-identical to having only fed that many tokens
    (speculative-decode rollback)."""

    def __init__(self, n_layers: int) -> None:
        self.layers: list[_LayerCache] = [_LayerCache() for _ in range(n_layers)]

    def __getitem__(self, i: int) -> _LayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        for lc in self.layers:
            if lc.kv.c_kv is not None:
                return lc.kv.c_kv.shape[1]
        return 0

    def truncate(self, length: int) -> None:
        if length < 0:
            raise ValueError(f"truncate length {length} < 0")
        if length >= self.offset:
            return
        for lc in self.layers:
            if lc.kv.c_kv is None:
                continue
            lc.kv.c_kv = lc.kv.c_kv[:, :length]
            lc.kv.k_pe = lc.kv.k_pe[:, :, :length]
            if lc.idx.k is not None:
                lc.idx.k = lc.idx.k[:, :, :length]


class GLMResidentModel(nn.Module):
    """Resident GLM decoder exposing ``model(token_ids, *, caches, offset, capture_layers)``.

    The minimal contract :mod:`quanta.glm.generate` / :mod:`quanta.glm.spec` consume: a single-token
    (or short-window) forward that threads a per-layer KV + indexer cache (:class:`GLMDecodeCache`, via
    :meth:`make_caches`) and returns ``logits`` ``[1,T,vocab]`` — optionally with a ``{layer: hidden}``
    capture of the post-block residual (the MTP/EAGLE feature). Incremental-equal to the streamed
    prefill path (gated model-free)."""

    def __init__(self, cfg: GLMConfig, *, use_fast: bool = True, use_indexer: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_fast = use_fast
        self.use_indexer = use_indexer
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [GLMDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

    def make_caches(self) -> GLMDecodeCache:
        """A fresh per-layer decode cache (MLA KV + indexer key). :mod:`quanta.glm.spec` prefers this
        factory over the lazy ``glm.decode`` fallback, so the resident model is self-contained."""
        return GLMDecodeCache(self.num_layers)

    def __call__(self, token_ids: mx.array, *, caches=None, offset: int = 0, capture_layers=None):
        """Decode/prefill a window of ``token_ids`` ``[T]`` (or ``[1,T]``) starting at absolute
        position ``offset``; returns ``logits`` ``[1,T,vocab]`` (or ``(logits, caps)`` if
        ``capture_layers``). With a ``caches`` object each token is appended incrementally; without one
        a self-contained prefill over ``positions = [offset, offset+T)`` is run."""
        ids = mx.array(token_ids).reshape(1, -1)
        t = ids.shape[1]
        h = self.embed_tokens(ids)
        caps: dict[int, mx.array] = {}
        want = set(capture_layers or ())
        if caches is None:                                            # self-contained prefill window
            positions = mx.arange(offset, offset + t)
            for i, layer in enumerate(self.layers):
                h = layer(h, positions, use_fast=self.use_fast, use_indexer=self.use_indexer)
                if i in want:
                    caps[i] = h[0]
        else:                                                        # incremental: one position per step
            for i, layer in enumerate(self.layers):
                lc = caches[i]
                cols = []
                for k in range(t):                                   # bounded by the (tiny) window width
                    h_col = layer.step(h[:, k:k + 1], lc, offset + k,
                                       use_fast=self.use_fast, use_indexer=self.use_indexer)
                    cols.append(h_col)
                h = mx.concatenate(cols, axis=1) if t > 1 else cols[0]
                if i in want:
                    caps[i] = h[0]
        logits = self.lm_head(self.norm(h))
        return (logits, caps) if capture_layers else logits
