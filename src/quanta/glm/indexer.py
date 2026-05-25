"""DSA Lightning-Indexer for GLM-5.1 (``glm_moe_dsa``) — DeepSeek-Sparse-Attention token selector.

GLM-5.1 carries a DeepSeek-V3.2-style **Lightning Indexer** on every attention block. For each query
position it scores all (causal) key positions with a cheap multi-head dot product and lets the main MLA
attend only the **top-``index_topk``** of them (sparse attention for long context). Grounded in
:mod:`quanta.glm.config` / :mod:`quanta.glm.loader` (keys ``wq_b`` / ``wk`` / ``weights_proj`` /
``k_norm.{weight,bias}``):

* **index query**: ``wq_b(q_latent)`` reshaped to ``[B,T,index_n_heads,index_head_dim]`` — reuses the
  main MLA q-latent (``q_a_layernorm(q_a_proj(x))``); partial **interleaved** RoPE on the last
  ``qk_rope_head_dim`` dims (``indexer_rope_interleave``).
* **index key** (MQA, single head): ``k_norm(wk(x))`` → ``[B,T,index_head_dim]`` (``k_norm`` is a
  ``LayerNorm`` with bias); same partial interleaved RoPE.
* **per-head weights**: ``weights_proj(x)`` → ``[B,T,index_n_heads]`` (scaled by
  ``index_head_dim**-0.5 · index_n_heads**-0.5``, matching :mod:`quanta.dsv4.indexer` — a positive
  global factor that does not change the top-k argsort).
* **score**: ``score[i,j] = Σ_h weights[i,h] · relu(q_idx[i,h] · k_idx[j])`` for causal ``j`` (else
  ``-inf``); the kept set is the causal ∧ top-``index_topk`` positions.

:meth:`LightningIndexer.select_mask` returns an **additive** ``[B,T,Tkv]`` mask (0 keep / ``-inf``
drop) that the MLA ANDs into its causal mask. When ``index_topk >= Tkv`` (or any time every causal
token survives) this is exactly the causal mask, so the masked attention is **bit-identical to dense
causal MLA** — the parity-first ``keep-all == dense`` gate (#84). The Hadamard-rotate + fp4 fake-quant
of the reference indexer cancel for selection (orthonormal rotation, argsort-invariant scale) and are
omitted, as in :mod:`quanta.dsv4.indexer`. The decode stepper mirrors prefill at one position.

Gated model-free (tiny random weights) in ``parity/glm_forward_test.py`` / ``parity/glm_moe_test.py``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.glm.attention import build_inv_freq, rope_cos_sin, rope_fast, rope_naive
from quanta.glm.config import GLMConfig

_NEG = -mx.inf
# k_norm is HF DeepseekV3 indexer LayerNorm (eps 1e-6); matches the loader's k_norm.{weight,bias}.
_KNORM_EPS = 1e-6


class LightningIndexer(nn.Module):
    """DSA top-``index_topk`` selector. Consumes the MLA q-latent + the block input ``x``."""

    def __init__(self, cfg: GLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self.weight_scale = self.head_dim ** -0.5 * self.n_heads ** -0.5

        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=_KNORM_EPS)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False)

    def _q_index(self, q_latent: mx.array, positions: mx.array, use_fast: bool) -> mx.array:
        """Index query ``[B,H_idx,T,index_head_dim]`` from the MLA q-latent (partial interleaved RoPE)."""
        b, t, _ = q_latent.shape
        q = self.wq_b(q_latent).reshape(b, t, self.n_heads, self.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))                              # [B,H_idx,T,index_head_dim]
        return self._rope_partial(q, positions, use_fast)

    def _k_index(self, x: mx.array, positions: mx.array, use_fast: bool) -> mx.array:
        """Index key ``[B,1,T,index_head_dim]`` from the block input (LayerNorm + partial RoPE; MQA)."""
        b, t, _ = x.shape
        k = self.k_norm(self.wk(x)).reshape(b, t, 1, self.head_dim)
        k = mx.transpose(k, (0, 2, 1, 3))                             # [B,1,T,index_head_dim]
        return self._rope_partial(k, positions, use_fast)

    def _rope_partial(self, x: mx.array, positions: mx.array, use_fast: bool) -> mx.array:
        """RoPE the last ``qk_rope_head_dim`` dims of ``x`` (``[B,H,T,index_head_dim]``); pass the rest
        through. ``positions`` are the absolute positions of the ``T`` axis."""
        rd = self.rope
        head, tail = x[..., :-rd], x[..., -rd:]
        if use_fast:
            off = int(positions[0].item()) if x.shape[2] > 0 else 0
            tail = rope_fast(tail, build_inv_freq(self.cfg), offset=off)
        else:
            cos, sin = rope_cos_sin(self.cfg, positions)
            cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
            tail = rope_naive(tail, cos, sin)
        return mx.concatenate([head, tail], axis=-1)

    def _score(self, qi: mx.array, ki: mx.array, weights: mx.array) -> mx.array:
        """``score[b,i,j] = Σ_h weights[b,i,h] · relu(qi[b,h,i,:] · ki[b,0,j,:])`` → ``[B,Tq,Tkv]``."""
        per_head = mx.einsum("bhid,bxjd->bhij", qi.astype(mx.float32), ki.astype(mx.float32))  # x==1
        per_head = mx.maximum(per_head, 0.0)                          # relu per (head, i, j)
        wq = mx.transpose(weights.astype(mx.float32), (0, 2, 1))      # [B,H_idx,Tq]
        return mx.einsum("bhij,bhi->bij", per_head, wq)               # weighted sum over heads

    def _select_mask(self, score: mx.array, causal: mx.array, kv_len: int) -> mx.array:
        """Additive ``[B,Tq,Tkv]`` mask (0 keep / ``-inf`` drop) = causal ∧ top-``index_topk``.

        ``score`` already has ``-inf`` on non-causal entries; ``causal`` is the boolean causal mask.
        When ``index_topk >= kv_len`` every causal token is kept ⇒ the mask is exactly causal ⇒ the
        masked attention equals dense causal MLA."""
        k = min(self.topk, kv_len)
        if k >= kv_len:
            keep = causal
        else:
            thr = mx.sort(score, axis=-1)[..., kv_len - k][..., None]  # k-th largest per query
            keep = (score >= thr) & causal
        return mx.where(keep, mx.array(0.0, mx.float32), mx.array(_NEG, mx.float32))

    def select_mask(self, x: mx.array, q_latent: mx.array, positions: mx.array, *,
                    use_fast: bool = False) -> mx.array:
        """Prefill selection: additive ``[B,T,T]`` mask for the main MLA. ``x``: block input
        ``[B,T,dim]``; ``q_latent``: the MLA q-latent ``[B,T,q_lora]``."""
        b, t, _ = x.shape
        qi = self._q_index(q_latent, positions, use_fast)
        ki = self._k_index(x, positions, use_fast)
        weights = self.weights_proj(x) * self.weight_scale            # [B,T,H_idx]
        score = self._score(qi, ki, weights)                         # [B,T,T]
        i = mx.arange(t)[:, None]
        j = mx.arange(t)[None, :]
        causal = (j <= i)[None]                                       # [1,T,T]
        score = mx.where(causal, score, mx.array(_NEG, mx.float32))
        return self._select_mask(score, causal, t)

    def step_mask(self, x_t: mx.array, q_latent_t: mx.array, k_index_cache, offset: int, *,
                  use_fast: bool = False) -> mx.array:
        """Decode selection at absolute position ``offset``: additive ``[B,1,S]`` mask. ``k_index_cache``
        appends this token's index key and returns the full stream ``[B,1,S,index_head_dim]`` — so the
        single query scores every cached key, mirroring prefill at this position."""
        positions = mx.array([offset])
        qi = self._q_index(q_latent_t, positions, use_fast)          # [B,H_idx,1,d]
        ki_new = self._k_index(x_t, positions, use_fast)             # [B,1,1,d]
        ki = k_index_cache.update(ki_new)                            # [B,1,S,d] full stream
        kv_len = ki.shape[2]
        weights = self.weights_proj(x_t) * self.weight_scale         # [B,1,H_idx]
        score = self._score(qi, ki, weights)                         # [B,1,S] (all cached are causal)
        causal = mx.ones((1, 1, kv_len), dtype=mx.bool_)
        return self._select_mask(score, causal, kv_len)
