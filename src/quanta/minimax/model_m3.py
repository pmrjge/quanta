"""MiniMax-M3-VL text decoder block — MLX-native ``mlx.nn`` modules (M1 layer parity).

The runnable reference forward for the M3 text backbone (``MiniMaxM3SparseForCausalLM``): one
decoder layer built as composed ``mlx.nn`` modules (rule 1) over ``mx.fast`` primitives (rule 2),
**dense full attention** (the trained block-sparse indexer is inert at ``T <= sparse_topk_blocks *
sparse_block_size`` ⇒ sparse == dense at short context, so parity is established dense-first; the
indexer is the long-context serving lever — M3). It is **additive**: the sibling M2.7
``quanta.minimax.model`` / ``moe`` / ``attention`` modules (a DIFFERENT architecture) are untouched.

Architecture (empirically grounded — ``config_m3`` + the real checkpoint headers/values, see the M0
fit-test and the M1 layer gate). M3 = the MiniMax-M2 backbone (transformers ``minimax_m2``:
sigmoid-noaux MoE, per-(q/k) RMSNorm, partial rotate-half RoPE) **plus** five deltas, each pinned
against an authoritative sibling:

* **Per-head QK-norm** (``qk_norm_type="per_head"``): RMSNorm of width ``head_dim`` applied to q,k
  AFTER the reshape to ``[B,T,H,head_dim]`` and BEFORE RoPE (≈ ``quanta.qwen35``), NOT M2's
  full-width pre-reshape norm.
* **Gemma ``(1 + weight)`` RMSNorm** (``use_gemma_norm``) on EVERY norm — input/post-attention/
  final layer norms AND the per-head q/k norms AND the indexer norms. The fold is applied at LOAD
  time (:func:`one_plus`); the forward runs plain ``weight * normed`` (the ``quanta.qwen35``
  convention). [PINNED: a single ``use_gemma_norm`` flag ⇒ uniform ``(1+w)``; the decisive check is
  the M2 teacher-forced ppl arbiter — a wrong fold degrades ppl uniformly.]
* **Clamped SwiGLU-OpenAI** activation (``hidden_act="swigluoai"``, ``swiglu_alpha=1.702``,
  ``swiglu_limit=7.0``) for the dense FFN AND the experts AND the shared expert — :func:`swigluoai`,
  byte-pinned to ``transformers`` ``GptOssExperts._apply_gate`` (w1=gate→swish branch, w3=up→
  ``(up+1)`` branch, w2=down).
* **Router ``* routed_scaling_factor`` (2.0) + a shared expert** on top of the M2 sigmoid-noaux
  router — :func:`route_noaux`, identical math to the parity-gated ``quanta.nemotron`` / ``dsv4``
  router (sigmoid; bias added for SELECTION only; weights gathered from the pure sigmoid; renorm;
  scale). No DeepSeek group machinery (M3 has no ``n_group``/``topk_group``).
* The shared expert has **no scalar gate** (the checkpoint ships no ``shared_expert_gate`` — rule-6
  coverage is exact without one), unlike Qwen2-MoE / qwen35.

Two output-equivalent routed paths are gated (rule 4): the dense oracle (run every expert, mask) and
the sparse ``mx.gather_mm`` dispatch; the packed-int6 ``mx.gather_qmm`` resident sibling is added at
the bake/serving milestones (M2/M3) on the SAME codes.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from quanta.cache_quant import dequantize_last_axis, quantize_last_axis
from quanta.minimax.config_m3 import MiniMaxM3Config

# ----------------------------------------------------------------------------- #
# Norms / RoPE helpers (self-contained; mirror the parity-gated quanta.qwen35).
# ----------------------------------------------------------------------------- #


def one_plus(w: mx.array) -> mx.array:
    """Gemma ``(1 + weight)`` RMSNorm fold (``use_gemma_norm``). The loader applies this so the
    forward can run plain ``weight * normed`` (``mx.fast.rms_norm`` / :func:`rms_norm`). Applies to
    every M3 RMSNorm — input/post-attention/final + per-head q/k + indexer norms. Kept in the source
    dtype (bf16)."""
    return w + 1.0


def rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """Weighted RMSNorm over the last axis, computed in fp32 (per-head q/k norm path). ``w`` is the
    already-``(1+w)``-folded weight."""
    xf = x.astype(mx.float32)
    xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (w.astype(mx.float32) * xf).astype(x.dtype)


def inv_freq(rotary_dim: int, theta: float) -> mx.array:
    """Plain (no-YaRN) partial-RoPE inverse frequencies for the rotated dims, ``[rotary_dim//2]``.
    M3 declares no rope-scaling, so this is the textbook ``1/theta**(2i/rotary_dim)``."""
    idx = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    return 1.0 / (theta ** (idx / rotary_dim))


def rope_fast(x: mx.array, inv: mx.array, rd: int, offset: int) -> mx.array:
    """``mx.fast.rope`` rotate-half over the first ``rd`` dims (the rest pass through). mlx ``freqs``
    is the *period* (angle = pos/freqs) ⇒ pass ``1/inv``; ``traditional=False`` is the rotate-half
    (split-half: dim ``i`` pairs with ``i+rd/2``) variant MiniMax/Qwen use (HF ``rotate_half``)."""
    return mx.fast.rope(x, dims=rd, traditional=False, base=None, scale=1.0, offset=offset,
                        freqs=1.0 / inv)


def rope_explicit(x: mx.array, inv: mx.array, rd: int, offset: int) -> mx.array:
    """Explicit rotate-half RoPE on the first ``rd`` dims — the short-sequence reference for
    :func:`rope_fast`. ``x`` ``[B,H,T,D]``."""
    b, h, t, d = x.shape
    pos = (mx.arange(t, dtype=mx.float32) + offset)[:, None]
    ang = pos * inv[None, :]
    cos = mx.cos(ang)[None, None]
    sin = mx.sin(ang)[None, None]
    xr = x[..., :rd]
    x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    rot = mx.concatenate([o1, o2], axis=-1)
    return mx.concatenate([rot, x[..., rd:]], axis=-1)


def causal_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    """Lower-right causal additive mask (query j at abs pos ``kv_len-q_len+j``)."""
    off = kv_len - q_len
    j = mx.arange(q_len)[:, None]
    i = mx.arange(kv_len)[None, :]
    return mx.where(i <= j + off, mx.array(0.0, dtype), mx.array(float("-inf"), dtype))


# ----------------------------------------------------------------------------- #
# Clamped SwiGLU-OpenAI activation (gpt-oss).
# ----------------------------------------------------------------------------- #


def swigluoai(gate: mx.array, up: mx.array, alpha: float, limit: float) -> mx.array:
    """Clamped SwiGLU-OAI, byte-pinned to ``transformers`` ``GptOssExperts._apply_gate``:

        gate = clamp(gate, max=limit)        # upper-clamped only (min=None)
        up   = clamp(up, -limit, +limit)
        glu  = gate * sigmoid(alpha * gate)  # swish with learnable-style alpha
        out  = (up + 1) * glu

    ``gate`` = the w1 (swish) branch, ``up`` = the w3 (linear ``(up+1)`` multiplier) branch — both
    ``[..., inter]`` post-projection. Computed in fp32 for numerical stability across the clamp
    (the activation is precision-sensitive; the reference oracle matches in fp32)."""
    g = gate.astype(mx.float32)
    u = up.astype(mx.float32)
    g = mx.minimum(g, limit)                 # clamp(max=limit), no lower bound
    u = mx.clip(u, -limit, limit)
    glu = g * mx.sigmoid(alpha * g)
    return (u + 1.0) * glu


# ----------------------------------------------------------------------------- #
# Routing (sigmoid noaux_tc + bias; == quanta.nemotron / dsv4).
# ----------------------------------------------------------------------------- #


def route_noaux(xf: mx.array, gate_w: mx.array, bias: mx.array, cfg: MiniMaxM3Config,
                ) -> tuple[mx.array, mx.array]:
    """Top-k sigmoid-noaux routing. ``xf`` ``[N,hidden]`` → ``(idx [N,topk] int32, w [N,topk] f32)``.

    The MiniMax-M2 / DeepSeek-noaux scheme (no group machinery): score by ``sigmoid(logits)``, add
    the correction ``bias`` for SELECTION only, gather the top-k weights from the PURE sigmoid
    (without bias), renormalize (``norm_topk_prob``), then scale by ``routed_scaling_factor``. ``gate_w``
    ``[E,hidden]`` and ``bias`` ``[E]`` are fp32 in the checkpoint; the matmul runs fp32."""
    topk = cfg.num_experts_per_tok
    logits = xf.astype(mx.float32) @ gate_w.astype(mx.float32).T          # [N, E]
    scores = mx.sigmoid(logits)
    choice = scores + bias.astype(mx.float32)[None]                       # bias: selection only
    idx = mx.argpartition(-choice, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
    w = mx.take_along_axis(scores, idx, axis=-1)                          # weights: pure sigmoid
    if cfg.norm_topk_prob:
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


# ----------------------------------------------------------------------------- #
# Attention — GQA + per-head QK-norm + partial RoPE (dense full attention).
# ----------------------------------------------------------------------------- #


class KVCache:
    """Plain GQA KV cache: ``[B, n_kv, S, head_dim]`` k/v, grown along the seq axis.

    Two storage modes (mirrors :class:`quanta.internlm2.attention.KVCache`):

    * ``quantized=False`` (default): bf16 verbatim — the M1/M2 parity path / short-context decode.
    * ``quantized=True``: per-token, per-group affine int-``bits`` over ``head_dim`` (the last axis)
      via :mod:`quanta.cache_quant`. ``update`` dequantizes the full cache to bf16 for the SDPA return,
      so the attention path is unchanged. **int8 g64 is the M3-4 serving lever** (GQA 4 kv heads ⇒ the
      KV is already cheap; int8 halves it again). The quant groups sit on ``head_dim`` while the paged
      block-pool cuts the seq axis — orthogonal, so a paged gather is **bit-identical** to this discrete
      cache fed the same tokens (the :class:`quanta.paged.PagedKVCacheManager` foundation), which is what
      ``parity/minimax_m3_paged_test`` gates paged == discrete against."""

    def __init__(self, *, quantized: bool = False, group_size: int = 64, bits: int = 8) -> None:
        self.quantized = quantized
        self.group_size = group_size
        self.bits = bits
        # bf16 mode
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        # int<bits> mode (codes + per-group scales/biases, concatenated along the seq axis)
        self.k_q: mx.array | None = None
        self.k_s: mx.array | None = None
        self.k_b: mx.array | None = None
        self.v_q: mx.array | None = None
        self.v_s: mx.array | None = None
        self.v_b: mx.array | None = None

    @property
    def offset(self) -> int:
        if self.quantized:
            return 0 if self.k_q is None else self.k_q.shape[2]
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if not self.quantized:
            if self.k is None:
                self.k, self.v = k, v
            else:
                self.k = mx.concatenate([self.k, k], axis=2)
                self.v = mx.concatenate([self.v, v], axis=2)
            return self.k, self.v
        k_qn, k_sn, k_bn = quantize_last_axis(k, self.group_size, bits=self.bits)
        v_qn, v_sn, v_bn = quantize_last_axis(v, self.group_size, bits=self.bits)
        if self.k_q is None:
            self.k_q, self.k_s, self.k_b = k_qn, k_sn, k_bn
            self.v_q, self.v_s, self.v_b = v_qn, v_sn, v_bn
        else:
            self.k_q = mx.concatenate([self.k_q, k_qn], axis=2)
            self.k_s = mx.concatenate([self.k_s, k_sn], axis=2)
            self.k_b = mx.concatenate([self.k_b, k_bn], axis=2)
            self.v_q = mx.concatenate([self.v_q, v_qn], axis=2)
            self.v_s = mx.concatenate([self.v_s, v_sn], axis=2)
            self.v_b = mx.concatenate([self.v_b, v_bn], axis=2)
        k_full = dequantize_last_axis(self.k_q, self.k_s, self.k_b, self.group_size,
                                      dtype=k.dtype, bits=self.bits)
        v_full = dequantize_last_axis(self.v_q, self.v_s, self.v_b, self.group_size,
                                      dtype=v.dtype, bits=self.bits)
        return k_full, v_full


class MiniMaxM3Attention(nn.Module):
    """GQA full attention: per-head QK-norm (before RoPE) + partial rotate-half RoPE, no output gate.

    Dense at short context (the trained sparse indexer ``index_{q,k}_*`` is loaded by policy but
    consumed only by the M3 long-context sparse path — inert here because top-k blocks == all blocks
    at ``T <= sparse_topk_blocks*sparse_block_size``)."""

    def __init__(self, cfg: MiniMaxM3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads          # 64
        self.nkv = cfg.num_key_value_heads         # 4
        self.hd = cfg.head_dim                     # 128
        self.rep = cfg.n_rep                       # 16
        self.rd = cfg.rotary_dim                   # 64
        self.scale = cfg.attn_scale                # head_dim**-0.5
        self.eps = cfg.norm_eps
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)
        self.q_norm = mx.ones((self.hd,))          # (1+w)-folded by the loader
        self.k_norm = mx.ones((self.hd,))
        self._inv = inv_freq(self.rd, cfg.rope_theta)

    def _project(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.nh, self.hd)
        k = self.k_proj(x).reshape(b, t, self.nkv, self.hd)
        v = self.v_proj(x).reshape(b, t, self.nkv, self.hd)
        q = rms_norm(q, self.q_norm, self.eps)                 # per-head QK-norm BEFORE RoPE
        k = rms_norm(k, self.k_norm, self.eps)
        q = mx.transpose(q, (0, 2, 1, 3))                      # [B,H,T,D]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v

    def __call__(self, x, *, cache=None, use_fast=True):
        b, t, _ = x.shape
        offset = cache.offset if cache is not None else 0
        q, k, v = self._project(x)
        rope = rope_fast if use_fast else rope_explicit
        q = rope(q, self._inv, self.rd, offset)
        k = rope(k, self._inv, self.rd, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]
        kr = mx.repeat(k, self.rep, axis=1)                    # GQA: kv head -> its query group
        vr = mx.repeat(v, self.rep, axis=1)
        if use_fast:
            mask = "causal" if t > 1 else None
            out = mx.fast.scaled_dot_product_attention(q, kr, vr, scale=self.scale, mask=mask)
        else:
            scores = (q @ mx.swapaxes(kr, -1, -2)) * self.scale + causal_mask(t, kv_len, q.dtype)
            wts = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            out = wts @ vr
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, self.nh * self.hd)
        return self.o_proj(out)

    def _project_chunked(self, x, chunk: int):
        """:meth:`_project` applied in ``<=chunk`` row-slices (concat over the batch axis 0) so each
        packed q/k/v projection matmul stays in the batch-M bit-exact ``mx.quantized_matmul`` regime
        (#153 option B / M0: ``mx.quantized_matmul`` is a per-row gemv — batch-M bit-exact — only for
        ``M<=~10``, switching to a reordering tiled GEMM at ``M>=12``). :meth:`_project` is fully
        per-row (projection + per-head RMSNorm + reshape + transpose — no cross-row op), so this
        equals the full-batch :meth:`_project` bit-for-bit; chunking only bounds each matmul's M.
        (``b <= chunk`` → a single call, no split.)"""
        b = x.shape[0]
        if b <= chunk:
            return self._project(x)
        qs, ks, vs = [], [], []
        for lo in range(0, b, chunk):
            q_c, k_c, v_c = self._project(x[lo:lo + chunk])
            qs.append(q_c)
            ks.append(k_c)
            vs.append(v_c)
        return (mx.concatenate(qs, axis=0), mx.concatenate(ks, axis=0), mx.concatenate(vs, axis=0))

    def decode_step_batched(self, x, *, kv_for_layer, offsets, chunk: int,
                            paged_batched: bool = False) -> mx.array:
        """One batched ``B``-stream single-token decode through ``B`` *ragged* per-stream KV caches —
        the #153 GQA loop-kill (M3-3). M3 is all-GQA so this is the whole mixer (no GDN hybrid, no
        YaRN: one fixed ``inv_freq``, nothing length-dependent to thread).

        ``x`` ``[B,1,hidden]`` are the ``B`` streams' post-input-norm residuals (stacked).
        ``kv_for_layer[s]`` is stream ``s``'s :class:`KVCache` for THIS layer (mutated in place by the
        shared helper); ``offsets[s]`` is stream ``s``'s absolute decode position (== its cache offset
        before the step). Returns the o-projected attention output ``[B,1,hidden]`` — the same
        residual ``y`` the per-stream :meth:`__call__` (``use_fast=True``) returns for each stream.

        Batches the four big projections (q/k/v/o) across the ``B`` streams in ``<=chunk`` row-slices
        (#153 option B: each packed ``mx.quantized_matmul`` stays in the M=1-equivalent gemv regime,
        so the chunked projection equals the per-stream loop bit-for-bit — the weights read
        ``⌈B/chunk⌉×`` vs ``B×``, the bandwidth win), then:

        * **per-stream RoPE** — looped, NOT a batched reimpl: each stream rotates with its OWN absolute
          offset via the exact :func:`rope_fast` kernel the single-stream path runs, so it is
          bit-identical per row (a hand-rolled batched RoPE drifts at bf16 on real values and compounds
          across layers — the ``feedback_batched_rope_bf16`` memory). M3 has ONE ``inv_freq`` (native
          1M, no YaRN) so there is no per-stream frequency, only the per-stream offset. Bounded IO loop
          over the small batch (rule 3); RoPE is cheap vs the projections / SDPA / MoE.
        * **one fused KV-update + SDPA** via the shared
          :func:`quanta.modeling.batched_attention.batched_decode_attention_kv` (the same #153 primitive
          InternLM2.5 / Nemotron / qwen35 use; GQA repeat inside the helper). When ``paged_batched`` is
          True AND ``kv_for_layer`` holds :class:`quanta.paged.PagedKVCacheView` (M3-4 paged serving),
          the per-stream ``.update()`` loop becomes ONE ``write_batched`` scatter + ONE ``gather_batched``
          over the shared manager (the paged KV loop-kill); otherwise (discrete caches, or the flag off)
          the bounded per-stream ``.update()`` then ONE padded SDPA. Both end in the same fused SDPA, so
          the choice is bit-identical (gated in ``parity/minimax_m3_paged_test``).

        Row ``s`` equals :meth:`__call__` on stream ``s`` (at its own offset, ``use_fast=True``) against
        its own cache: the q/k/v/o projections are bit-exact once packed + chunked and the per-stream
        RoPE is bit-identical, so the ONLY divergence is the fused padded-SDPA reduction-order ULP — the
        greedy-token-equivalent class the project accepts for batched/tiled paths. Gated in
        ``parity/minimax_m3_loopkill_test.py``; re-gated @ 397B in ``parity/minimax_m3_loopkill_real.py``."""
        from quanta.modeling.batched_attention import batched_decode_attention_kv

        b, t, _ = x.shape
        if t != 1:
            raise ValueError(f"decode_step_batched is a single-token step; got T={t} (rule 6)")
        if not len(kv_for_layer) == len(offsets) == b:
            raise ValueError(
                f"decode_step_batched: B mismatch x={b} kv={len(kv_for_layer)} offsets={len(offsets)}")
        q, k, v = self._project_chunked(x, chunk)  # chunked: each packed proj M<=chunk (bit-exact, M0)
        # per-stream RoPE: loop the exact mx.fast.rope kernel — only the absolute offset differs per
        # stream (M3 has no YaRN). Bit-identical per row to __call__. Bounded IO loop (rule 3).
        q_rows, k_rows = [], []
        for s in range(b):
            off_s = int(offsets[s])
            q_rows.append(rope_fast(q[s:s + 1], self._inv, self.rd, off_s))
            k_rows.append(rope_fast(k[s:s + 1], self._inv, self.rd, off_s))
        q = mx.concatenate(q_rows, axis=0) if b > 1 else q_rows[0]      # [B,nh,1,hd]
        k = mx.concatenate(k_rows, axis=0) if b > 1 else k_rows[0]      # [B,n_kv,1,hd]
        # one fused KV-update + ONE padded SDPA across all B streams (GQA repeat inside the shared
        # helper). paged_batched=True over paged views ⇒ ONE write_batched + ONE gather_batched (the
        # M3-4 paged KV loop-kill); else the bounded per-stream .update() loop — bit-identical (rule 4).
        out = batched_decode_attention_kv(q, k, v, list(kv_for_layer), scale=self.scale,
                                          n_rep=self.rep, paged_batched=paged_batched)  # [B,nh,1,hd]
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, t, self.nh * self.hd)   # [B,1,nh*hd]
        # o-projection in the SAME <=chunk row-slices (bit-exact regime); the fused SDPA above is the
        # only op that spans all B (its softmax reorder is the lone greedy-token-equivalent ULP).
        if b <= chunk:
            return self.o_proj(out)
        return mx.concatenate([self.o_proj(out[lo:lo + chunk]) for lo in range(0, b, chunk)], axis=0)


# ----------------------------------------------------------------------------- #
# Dense FFN (layers 0-2) — clamped SwiGLU.
# ----------------------------------------------------------------------------- #


class MiniMaxM3DenseMLP(nn.Module):
    """Dense feed-forward (layers 0-2): clamped-SwiGLU ``down( swigluoai(gate(x), up(x)) )``, width
    ``dense_intermediate_size``."""

    def __init__(self, cfg: MiniMaxM3Config) -> None:
        super().__init__()
        self.cfg = cfg
        di = cfg.dense_intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, di, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, di, bias=False)
        self.down_proj = nn.Linear(di, cfg.hidden_size, bias=False)

    def __call__(self, x):
        h = swigluoai(self.gate_proj(x), self.up_proj(x), self.cfg.swiglu_alpha, self.cfg.swiglu_limit)
        return self.down_proj(h.astype(x.dtype))


# ----------------------------------------------------------------------------- #
# MoE (layers 3-59) — noaux router + clamped-SwiGLU experts + shared expert.
# ----------------------------------------------------------------------------- #


def _routed_sparse(xf: mx.array, idx: mx.array, gate_up: mx.array, down: mx.array,
                   inter: int, alpha: float, limit: float) -> mx.array:
    """Sparse routed clamped-SwiGLU via ``mx.gather_mm`` over pre-stacked experts (rule 7).

    ``xf`` ``[N,hidden]``; ``idx`` ``[N,topk]``; ``gate_up`` ``[E,2*inter,hidden]`` (w1 stacked over
    w3); ``down`` ``[E,hidden,inter]`` (w2). Returns per-(token,slot) outputs ``[N,topk,hidden]``."""
    n, hidden = xf.shape
    topk = idx.shape[1]
    mc = n * topk
    exp = idx.reshape(-1)
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)
    col = xf[:, :, None].astype(gate_up.dtype)                            # [N, hidden, 1]
    gu = mx.gather_mm(gate_up, col, lhs_indices=exp, rhs_indices=tok)[:, :, 0]   # [mc, 2*inter]
    h = swigluoai(gu[:, :inter], gu[:, inter:], alpha, limit)             # [mc, inter] (fp32)
    h = h[:, :, None].astype(down.dtype)
    d = mx.gather_mm(down, h, lhs_indices=exp, rhs_indices=mx.arange(mc, dtype=mx.int32))[:, :, 0]
    return d.reshape(n, topk, hidden)


def _routed_sparse_packed(xf: mx.array, idx: mx.array, gate_up: dict, down: dict,
                          inter: int, alpha: float, limit: float) -> mx.array:
    """Sparse routed clamped-SwiGLU via ``mx.gather_qmm`` over the **packed int6** expert stacks —
    the memory-lean serving sibling of :func:`_routed_sparse` (rule 7, the M3 resident path).

    ``gate_up`` / ``down`` are packed affine triplets ``{"packed","scale","bias","group_size","bits"}``
    (the ``[E,2*inter,hidden]`` / ``[E,hidden,inter]`` int6 codestream held verbatim — NEVER a bf16
    ``[E,*,*]`` array), from :meth:`quanta.minimax.artifact_m3.MiniMaxM3Artifact.moe_packed`, so the
    routed experts stay int6-resident (the ~300 GiB serving footprint). ``mx.gather_qmm`` dequantizes
    each routed expert's codes inline — exactly **two** calls (fused ``gate_up``, then ``down``).

    Output-equivalent to :func:`_routed_sparse` on the SAME codes (greedy-exact: ``gather_qmm`` fuses
    the dequant that :meth:`MiniMaxM3Artifact._dequant` does separately before ``gather_mm``; only the
    kernel differs, ~ULP). **Batch-invariant** — the same per-(token,slot) M=1 matvec structure as
    ``gather_mm``, so it does not reorder accumulation across batch-M (the served B>1 path is safe).

    ``gather_qmm`` arg order differs from ``gather_mm``: the **activation comes first** (``lhs_indices``
    gathers its rows), the packed weight stack second (``rhs_indices`` gathers ``E``); ``transpose=True``
    matches the ``[E,out,in]`` layout (``x[.,in] @ W[.,out,in].T``). The qmm output follows the baked
    ``scale`` dtype (bf16); the swish branch upcasts to fp32 inside :func:`swigluoai` (clamp precision),
    then the down activation is cast back to the down ``scale`` dtype — bit-matching the bf16 path's
    ``.astype(down.dtype)`` so the two paths stay greedy-exact."""
    n, hidden = xf.shape
    topk = idx.shape[1]
    mc = n * topk
    exp = idx.reshape(-1)                                                 # [mc] expert id per slot
    tok = mx.repeat(mx.arange(n, dtype=mx.int32), topk)                   # [mc] token per slot
    rows = mx.arange(mc, dtype=mx.int32)                                  # identity lhs gather
    x_in = xf[tok]                                                        # [mc, hidden] per-slot acts
    gu = mx.gather_qmm(x_in[:, None, :], gate_up["packed"], gate_up["scale"], gate_up["bias"],
                       lhs_indices=rows, rhs_indices=exp, transpose=True,
                       group_size=int(gate_up["group_size"]), bits=int(gate_up["bits"]))[:, 0, :]
    h = swigluoai(gu[:, :inter], gu[:, inter:], alpha, limit)             # [mc, inter] (fp32)
    h = h.astype(down["scale"].dtype)                                     # match the bf16 path's down input
    d = mx.gather_qmm(h[:, None, :], down["packed"], down["scale"], down["bias"],
                      lhs_indices=rows, rhs_indices=exp, transpose=True,
                      group_size=int(down["group_size"]), bits=int(down["bits"]))[:, 0, :]
    return d.reshape(n, topk, hidden)


def _routed_dense(xf: mx.array, idx: mx.array, w: mx.array, gate_up: mx.array, down: mx.array,
                  inter: int, alpha: float, limit: float) -> mx.array:
    """Dense oracle: run EVERY expert on every token, combine only the top-k. Parity reference for
    :func:`_routed_sparse` (small E only)."""
    n, hidden = xf.shape
    e = gate_up.shape[0]
    xd = xf.astype(gate_up.dtype)
    gu = mx.einsum("nh,eoh->neo", xd, gate_up)                            # [N, E, 2*inter]
    h = swigluoai(gu[..., :inter], gu[..., inter:], alpha, limit)         # [N, E, inter]
    d = mx.einsum("nei,ehi->neh", h.astype(down.dtype), down)            # [N, E, hidden]
    gates = mx.zeros((n, e), dtype=mx.float32)
    rows = mx.repeat(mx.arange(n, dtype=mx.int32), idx.shape[1])
    gates[rows, idx.reshape(-1)] = w.reshape(-1).astype(mx.float32)
    return mx.sum(d.astype(mx.float32) * gates[:, :, None], axis=1)       # [N, hidden]


class MiniMaxM3MoE(nn.Module):
    """Sparse MoE: 128 routed experts top-4 (noaux sigmoid) + 1 shared expert, clamped-SwiGLU.

    Routed experts are held pre-stacked ``[E, 2*inter, hidden]`` (gate_up = w1 over w3) and
    ``[E, hidden, inter]`` (down = w2), ``gather_mm``-ready. The shared expert is a plain
    clamped-SwiGLU MLP (NO scalar gate), added to the routed sum. Router ``gate``/``bias`` are
    fp32."""

    def __init__(self, cfg: MiniMaxM3Config) -> None:
        super().__init__()
        self.cfg = cfg
        h, e = cfg.hidden_size, cfg.num_local_experts
        inter, si = cfg.moe_intermediate_size, cfg.shared_intermediate_size
        self.gate = mx.zeros((e, h), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((e,), dtype=mx.float32)
        self.experts_gate_up = mx.zeros((e, 2 * inter, h))   # [E, 2*inter, h] (w1 over w3)
        self.experts_down = mx.zeros((e, h, inter))          # [E, h, inter]   (w2)
        self.shared_gate_proj = mx.zeros((si, h))
        self.shared_up_proj = mx.zeros((si, h))
        self.shared_down_proj = mx.zeros((h, si))
        self.token_chunk = 8192

    def set_experts(self, gate_up: mx.array, down: mx.array) -> None:
        self.experts_gate_up, self.experts_down = gate_up, down

    def set_experts_packed(self, gate_up: dict, down: dict) -> None:
        """Hold the routed experts as **packed int6 affine triplets** (NOT dequantized) for the
        resident ``mx.gather_qmm`` serving path. ``gate_up`` / ``down`` are
        ``{"packed","scale","bias","group_size","bits"}`` dicts (from
        :meth:`quanta.minimax.artifact_m3.MiniMaxM3Artifact.moe_packed`). :meth:`__call__`
        auto-detects the dict (→ :func:`_routed_sparse_packed`, ``gather_qmm``) vs a bf16 stack
        (→ :func:`_routed_sparse`, ``gather_mm``) — same dispatch, greedy-exact on the same codes."""
        self.experts_gate_up, self.experts_down = gate_up, down

    def _shared(self, xf: mx.array) -> mx.array:
        """Shared clamped-SwiGLU expert (no scalar gate). ``xf`` ``[N,hidden]`` -> ``[N,hidden]`` fp32."""
        xd = xf.astype(self.shared_gate_proj.dtype)
        h = swigluoai(xd @ self.shared_gate_proj.T, xd @ self.shared_up_proj.T,
                      self.cfg.swiglu_alpha, self.cfg.swiglu_limit)        # [N, si] fp32
        return h.astype(self.shared_down_proj.dtype) @ self.shared_down_proj.T

    def __call__(self, x, *, sparse: bool = True):
        b, s, hidden = x.shape
        n = b * s
        inter = self.cfg.moe_intermediate_size
        alpha, limit = self.cfg.swiglu_alpha, self.cfg.swiglu_limit
        xf = x.reshape(n, hidden)
        idx, w = route_noaux(xf, self.gate, self.e_score_correction_bias, self.cfg)
        # auto-detect: packed int6 triplets (a dict) → gather_qmm (resident); bf16 stacks → gather_mm
        # (the parity reference). Same dispatch / same matvec, greedy-exact on the same codes (rule 4).
        packed = isinstance(self.experts_gate_up, dict)
        if sparse:
            routed_fn = _routed_sparse_packed if packed else _routed_sparse
            chunk = self.token_chunk if self.token_chunk and self.token_chunk > 0 else n
            multi = n > chunk
            parts = []
            for c0 in range(0, n, chunk):  # bounded chunked-prefill loop; experts stay vectorized
                c1 = min(c0 + chunk, n)
                slots = routed_fn(xf[c0:c1], idx[c0:c1], self.experts_gate_up,
                                  self.experts_down, inter, alpha, limit)  # [nc, topk, hidden]
                rc = mx.sum(slots.astype(mx.float32) * w[c0:c1][:, :, None], axis=1)
                parts.append(rc)
                if multi:
                    mx.eval(rc)
            routed = parts[0] if not multi else mx.concatenate(parts, axis=0)
        elif packed:  # rule 6: the dense oracle is bf16-einsum only — never a packed dict
            raise ValueError("MiniMaxM3MoE(sparse=False) is the bf16 dense oracle; packed int6 "
                             "experts require sparse=True (gather_qmm). Refusing to dequant silently.")
        else:
            routed = _routed_dense(xf, idx, w, self.experts_gate_up, self.experts_down,
                                   inter, alpha, limit)
        y = routed.astype(mx.float32) + self._shared(xf).astype(mx.float32)
        return y.astype(x.dtype).reshape(b, s, hidden)


# ----------------------------------------------------------------------------- #
# Decoder block.
# ----------------------------------------------------------------------------- #


class MiniMaxM3Block(nn.Module):
    """One decoder layer: ``x + attn(in_norm(x))`` then ``x + ffn(post_norm(x))``.

    ``ffn`` is :class:`MiniMaxM3DenseMLP` on layers 0-2 (``cfg.is_dense_layer``) and
    :class:`MiniMaxM3MoE` on layers 3-59 (``cfg.is_moe_layer``). Norms are Gemma ``(1+w)`` RMSNorm
    (the loader folds ``+1`` into ``.weight``)."""

    def __init__(self, cfg: MiniMaxM3Config, layer_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.is_moe = cfg.is_moe_layer(layer_id)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.self_attn = MiniMaxM3Attention(cfg)
        self.mlp = MiniMaxM3MoE(cfg) if self.is_moe else MiniMaxM3DenseMLP(cfg)

    def __call__(self, x, *, cache=None, use_fast=True, sparse=True):
        y = self.self_attn(self.input_layernorm(x), cache=cache, use_fast=use_fast)
        x = x + y
        h = self.post_attention_layernorm(x)
        y = self.mlp(h, sparse=sparse) if self.is_moe else self.mlp(h)
        return x + y
