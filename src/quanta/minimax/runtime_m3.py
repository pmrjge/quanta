"""RAM-resident MiniMax-M3-VL text-decoder runtime — load the baked int6 artifact and serve.

Mirrors :class:`quanta.qwen35.runtime.Qwen35ResidentModel` (the sibling sparse-MoE serving runtime)
for the MiniMax-M3 60-layer text backbone. The model is built **one decoder layer at a time**
(materialize that layer's params from :class:`quanta.minimax.artifact_m3.MiniMaxM3Artifact`,
``mx.eval`` them, drop the artifact's shard handles before the next — rule 8), so peak load residency
is ~one layer, not the whole 329.6 GiB checkpoint. The deployment target holds the full quantized
model RAM-resident pinned with ``mx.set_wired_limit``; ``n_layers`` builds a bounded prefix for
validation.

Each layer is a :class:`quanta.minimax.model_m3.MiniMaxM3Block` — the exact module the M1/M2 parity
gates validate — populated from the artifact. The **routed experts are held packed int6**
(``packed_experts=True``, the default): ``art.moe_packed(i)`` returns the affine-triplet codestream
verbatim and :meth:`MiniMaxM3MoE.set_experts_packed` wires it to the ``mx.gather_qmm`` dispatch, so
the ~300 GiB of experts stay int6-resident (greedy-exact on the SAME codes the bf16 ``gather_mm``
reference dequantizes — gated in ``parity/minimax_m3_runtime_test.py``). The **int8 mixer** (GQA
q/k/v/o, the dense-FFN gate/up/down on layers 0–2, the shared expert) is dequantized to bf16 on read
(``art.attention`` / ``art.dense_mlp`` / the shared keys of ``moe_packed``) and run as plain bf16
``nn.Linear`` — the proven M1/M2 reference forward (a packed-int8 mixer that holds those projections
``nn.QuantizedLinear``-resident is a later memory milestone; the ~10 GiB it would save is far under
the 160 GiB headroom). Norms are Gemma ``(1+w)`` RMSNorm (the loader folds ``+1`` at load); the
router ``gate`` + ``e_score_correction_bias`` stay native **F32** (routing precision — a bf16
downcast could flip a top-k tie).

Because the resident block IS the reference block, ``__call__`` has two output-equivalent regimes:

* **prefill** (``caches=None``): run each ``MiniMaxM3Block`` over the whole window with no cache —
  identical to the M1/M2 streamed reference (``parity/minimax_m3_ppl.streamed_logits``).
* **decode / cached** (``caches`` given, ``T >= 1``): run each block threading its per-layer GQA
  :class:`quanta.minimax.model_m3.KVCache` — the cache grows along the seq axis and the attention
  reads its ``offset`` for partial RoPE. A ``T``-token cached forward over fresh caches is identical
  to the ``caches=None`` prefill (full causal attention either way); a continuation
  (``cache.offset > 0``) attends the new tokens against the grown KV with a bottom-right causal mask.
  M3 is **natively 1M** (no YaRN), so there is no length-dependent RoPE factor to pin.

The trained block-sparse indexer is inert at short context (top-16 blocks == all blocks at
``T <= sparse_topk_blocks*sparse_block_size``), so this dense-attention path is the served forward up
to ~2K tokens; the indexer is the long-context serving lever (a later milestone).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from quanta.minimax.artifact_m3 import MiniMaxM3Artifact
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.loader_m3 import DENSE_MLP_PROJS
from quanta.minimax.model_m3 import KVCache, MiniMaxM3Block, one_plus
from quanta.minimax.quant_policy_m3 import LM_PREFIX

_ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _load_quant_triplet(art: MiniMaxM3Artifact, base: str
                        ) -> tuple[mx.array, mx.array, mx.array, int, int]:
    """A packed affine weight's three siblings (``.weight_packed`` / ``.weight_scale`` /
    ``.weight_bias`` — verbatim, no dequant) plus its ``(bits, group_size)`` from the manifest.

    The decode width travels with the artifact (rule 6 — the baked manifest is the single source of
    truth, never a hardcoded width that could silently mis-decode a differently-baked artifact).
    Mirrors :func:`quanta.qwen35.runtime._load_quant_triplet`. Fail loud if ``base`` is not an
    ``affine_packed`` weight (a dense projection has no packed codes to hold)."""
    meta = art.manifest.get(base)
    if meta is None or meta.get("format") != "affine_packed":
        raise ValueError(f"{base}: not an affine_packed weight (format="
                         f"{None if meta is None else meta.get('format')!r}); cannot pack (rule 6)")
    return (art.raw(base),
            art.get(base + ".weight_scale"),
            art.get(base + ".weight_bias"),
            int(meta["bits"]), int(meta["group_size"]))


def _packed_linear(art: MiniMaxM3Artifact, base: str, ref: nn.Linear) -> nn.QuantizedLinear:
    """Build a bias-free :class:`mlx.nn.QuantizedLinear` from the artifact's packed int8 triplet at
    ``base``, sized to the freshly-built ``ref`` ``nn.Linear`` it replaces (its ``[out, in]`` shape).

    ``nn.QuantizedLinear.__call__`` dispatches to ``mx.quantized_matmul(transpose=True)`` (rule 1 /
    rule 2), so swapping it in for the ``nn.Linear`` leaves the mixer forward (``self.q_proj(x)`` /
    ``self.gate_proj(x)`` …) UNCHANGED while holding the int8 weight PACKED (the ~6 GiB the
    dequant-to-bf16 path doubled). The fused ``mx.quantized_matmul`` keeps the dequantized int8
    weight at full precision (the bf16 path rounds it first), so it is the MORE precise sibling —
    greedy-exact, and batch-M bit-exact for the M=1 per-stream decode (the substrate the batched
    loop-kill will later chunk over; mirrors ``quanta.qwen35.runtime._packed_linear``)."""
    out_dims, in_dims = int(ref.weight.shape[0]), int(ref.weight.shape[1])
    packed, scale, wbias, bits, gs = _load_quant_triplet(art, base)
    ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=gs, bits=bits)
    ql.weight, ql.scales, ql.biases = packed, scale, wbias
    return ql


def _load_block(art: MiniMaxM3Artifact, cfg: MiniMaxM3Config, i: int, *,
                packed: bool = False, packed_experts: bool = True) -> MiniMaxM3Block:
    """Build one runnable :class:`MiniMaxM3Block` for layer ``i`` from the artifact tensors.

    Norms folded Gemma ``(1+w)`` (input/post + per-head q/k). The **int8 mixer** — GQA q/k/v/o (all
    60 layers) and the dense-FFN gate/up/down (layers 0–2) — is held two ways (rule 4):
    ``packed=False`` (the M1/M2 reference / fallback) dequantizes it to bf16 ``nn.Linear``;
    ``packed=True`` holds each projection as a packed ``nn.QuantizedLinear`` (``mx.quantized_matmul``)
    — the ~6 GiB memory lever + the batch-M bit-exact substrate, greedy-exact on the SAME int8 codes.
    The **shared expert stays bf16** either way (it runs batched inside the one MoE call — exactly the
    ``quanta.qwen35`` convention; packing it is a trivial later memory tweak under the huge headroom).
    Routed experts: ``packed_experts=True`` (default) holds them as packed int6 triplets
    (``art.moe_packed`` → ``set_experts_packed`` → ``mx.gather_qmm``, the resident path);
    ``packed_experts=False`` dequantizes them to bf16 (``art.moe`` → ``gather_mm``, the parity
    reference). Router ``gate`` + ``e_score_correction_bias`` stay native F32 either way."""
    blk = MiniMaxM3Block(cfg, i)
    nm = art.block_norms(i)
    blk.input_layernorm.weight = one_plus(nm["input_layernorm"])
    blk.post_attention_layernorm.weight = one_plus(nm["post_attention_layernorm"])

    ap = f"{LM_PREFIX}layers.{i}.self_attn."
    if packed:
        m = blk.self_attn
        for proj in _ATTN_PROJS:                                   # int8 q/k/v/o → mx.quantized_matmul
            setattr(m, proj, _packed_linear(art, ap + proj, getattr(m, proj)))
        m.q_norm = one_plus(art.read(ap + "q_norm.weight"))        # per-head q/k norm (1+w), bf16
        m.k_norm = one_plus(art.read(ap + "k_norm.weight"))
    else:
        at = art.attention(i)
        blk.self_attn.q_proj.weight = at["q_proj.weight"]
        blk.self_attn.k_proj.weight = at["k_proj.weight"]
        blk.self_attn.v_proj.weight = at["v_proj.weight"]
        blk.self_attn.o_proj.weight = at["o_proj.weight"]
        blk.self_attn.q_norm = one_plus(at["q_norm.weight"])       # per-head q/k norm (1+w)
        blk.self_attn.k_norm = one_plus(at["k_norm.weight"])

    if cfg.is_moe_layer(i):
        moe = art.moe_packed(i) if packed_experts else art.moe(i)
        blk.mlp.gate = moe["gate"]                                 # F32 (routing precision)
        blk.mlp.e_score_correction_bias = moe["e_score_correction_bias"]   # F32
        if packed_experts:
            blk.mlp.set_experts_packed(moe["experts_gate_up"], moe["experts_down"])
        else:
            blk.mlp.set_experts(moe["experts_gate_up"], moe["experts_down"])
        blk.mlp.shared_gate_proj = moe["shared_gate_proj"]         # shared expert bf16 (no scalar gate)
        blk.mlp.shared_up_proj = moe["shared_up_proj"]
        blk.mlp.shared_down_proj = moe["shared_down_proj"]
    elif packed:
        mp = f"{LM_PREFIX}layers.{i}.mlp."                         # int8 dense FFN → mx.quantized_matmul
        for proj in DENSE_MLP_PROJS:
            setattr(blk.mlp, proj, _packed_linear(art, mp + proj, getattr(blk.mlp, proj)))
    else:
        dm = art.dense_mlp(i)
        blk.mlp.gate_proj.weight = dm["gate_proj"]
        blk.mlp.up_proj.weight = dm["up_proj"]
        blk.mlp.down_proj.weight = dm["down_proj"]
    return blk


def _block_arrays(blk: MiniMaxM3Block) -> list[mx.array]:
    """Every resident array of one block — nn params plus the MoE expert stacks / router / shared
    expert (plain attrs). Under ``packed_experts`` the routed stacks are packed triplet dicts
    (``{packed,scale,bias,...}``), so eval their component arrays — the int6 codes must be
    materialized and pinned (rule 8); the int metadata is not an array and is not eval'd."""
    arrs = [v for _, v in tree_flatten(blk.parameters())]
    if blk.is_moe:
        mlp = blk.mlp
        for stack in (mlp.experts_gate_up, mlp.experts_down):
            if isinstance(stack, dict):
                arrs += [stack["packed"], stack["scale"], stack["bias"]]
            elif isinstance(stack, mx.array):
                arrs.append(stack)
        for attr in ("gate", "e_score_correction_bias", "shared_gate_proj",
                     "shared_up_proj", "shared_down_proj"):
            v = getattr(mlp, attr)
            if isinstance(v, mx.array):
                arrs.append(v)
    return arrs


def _live_kv_arrays(caches) -> list[mx.array]:
    """Every realized KV array across the per-layer caches — eval'd at each chunk boundary so the lazy
    prefill graph does not pile up across chunks (rule 8). Collects a discrete :class:`KVCache`'s
    storage (the bf16 ``k``/``v`` OR the int8 ``k_q``/``k_s``/… codes). A
    :class:`quanta.paged.PagedKVCacheView` holds no arrays itself — its writes land in the manager's
    pool, realized transitively when the chunk's last hidden is eval'd — so it contributes none."""
    arrs: list[mx.array] = []
    for c in caches:
        for name in ("k", "v", "k_q", "k_s", "k_b", "v_q", "v_s", "v_b"):
            a = getattr(c, name, None)
            if isinstance(a, mx.array):
                arrs.append(a)
    return arrs


def chunked_prefill(layers, embed_w: mx.array, norm_w: mx.array, lm_head_w: mx.array,
                    cfg: MiniMaxM3Config, token_ids, caches, *, inputs_embeds: mx.array | None = None,
                    chunk_tokens: int = 4096, use_fast: bool = True, sparse: bool = True) -> mx.array:
    """Long-context single-stream prefill: consume ``token_ids`` (or pre-spliced ``inputs_embeds``,
    the multimodal path — exactly one of the two) into ``caches`` in blocks of ``chunk_tokens``;
    return the LAST position's logits ``[1,1,vocab]`` (same contract as a cached
    :meth:`MiniMaxM3ResidentModel.__call__` over the last token / the batched ``prefill``).

    The feasibility substrate for the 1M window. The single-shot prefill holds the whole
    ``[1,T,hidden]`` window resident; this driver runs each :class:`quanta.minimax.model_m3.MiniMaxM3Block`
    over ONE bounded chunk at a time — every layer extends its (int8/bf16) GQA
    :class:`quanta.minimax.model_m3.KVCache` and a chunk's queries attend the grown KV with a
    bottom-right causal mask (``mx.fast.scaled_dot_product_attention`` ``mask="causal"`` is bottom-right
    aligned — the exact path the M3-1 cached forward and the qwen35 shipped chunked prefill both use).
    The fused flash-attn kernel never materializes the ``[chunk, kv_len]`` score matrix, so the
    per-chunk transient is O(chunk) and the whole prompt costs ONE forward per chunk.

    M3 is **natively 1M** (no YaRN) and **all dense GQA** (no GDN recurrent state), so — unlike
    :func:`quanta.qwen35.runtime.chunked_prefill` — there is no per-request RoPE factor to pin and no
    recurrent continuation: each chunk just reads its absolute position from the cache offset (read
    before the chunk's KV write). The trained block-sparse indexer (a later milestone) is the COMPUTE
    lever on top of this — dense chunked prefill is correct (it attends every prior key) but O(T²) in
    compute; it is feasible at 1M in MEMORY (O(chunk) transient + the int8 KV), and the indexer makes it
    feasible in TIME.

    A non-empty cache continues from its offset (multi-turn / paged-suffix prefix extension; each cache
    may be a discrete :class:`quanta.minimax.model_m3.KVCache` OR a
    :class:`quanta.paged.PagedKVCacheView`). Bit-identical to the single-shot prefill on the bf16 mixer
    (the chunk boundaries only re-cut the same per-row causal attention + per-token KV quant — no
    cross-token op changes, and the paged gather is bit-identical to the discrete cache); greedy-token-
    equivalent on the packed serving mixer (the projection ``mx.quantized_matmul`` runs at batch-M=chunk
    vs M=T — the documented #153 batch-M ULP). Gated in ``parity/minimax_m3_prefill_chunked_test.py``,
    re-gated @ 397B in ``parity/minimax_m3_prefill_chunked_real.py``. The ``inputs_embeds`` path (the
    multimodal admit — the image-spliced stream from :func:`quanta.minimax.model_vision_m3.splice_image_embeddings`)
    chunks the embedding rows directly instead of the embed lookup; everything downstream is identical."""
    if (token_ids is None) == (inputs_embeds is None):
        raise ValueError("chunked_prefill: pass exactly one of token_ids / inputs_embeds (rule 6)")
    if inputs_embeds is not None:
        emb = inputs_embeds[0] if inputs_embeds.ndim == 3 else inputs_embeds   # [T, hidden]
        if emb.ndim != 2:
            raise ValueError(f"inputs_embeds must be [T,hidden] or [1,T,hidden], got "
                             f"{inputs_embeds.shape}")
        total = int(emb.shape[0])

        def chunk_hidden(lo: int, hi: int) -> mx.array:
            return emb[lo:hi][None].astype(mx.bfloat16)
    else:
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids, dtype=mx.int32)
        ids = ids.reshape(-1)
        total = int(ids.shape[0])

        def chunk_hidden(lo: int, hi: int) -> mx.array:
            return embed_w[ids[lo:hi]][None].astype(mx.bfloat16)              # [1,tlen,hidden]
    if total < 1:
        raise ValueError("chunked_prefill needs >= 1 token")
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be >= 1 (got {chunk_tokens})")
    if len(caches) != len(layers):
        raise ValueError(f"len(caches)={len(caches)} != num_layers={len(layers)} "
                         f"(one KV cache per layer; refusing to mis-thread state — rule 6)")
    h_last = None
    for lo in range(0, total, chunk_tokens):           # bounded chunk loop (rule 3 — IO/admit boundary)
        h = chunk_hidden(lo, min(lo + chunk_tokens, total))
        for blk, cache in zip(layers, caches, strict=True):
            h = blk(h, cache=cache, use_fast=use_fast, sparse=sparse)
        h_last = h[:, -1:]
        mx.eval([h_last, *_live_kv_arrays(caches)])    # bound the lazy graph per chunk (rule 8)
        mx.clear_cache()                               # chunk transients do not pool up
    hh = mx.fast.rms_norm(h_last, norm_w.astype(h_last.dtype), cfg.norm_eps)
    return hh @ lm_head_w.T.astype(hh.dtype)                       # [1,1,vocab]


class MiniMaxM3ResidentModel:
    """RAM-resident MiniMax-M3 text decoder — prefill via the reference block, decode via cached KV.

    Built one layer at a time (materialize, then release the artifact's shard handles) for bounded
    load residency (rule 8). ``n_layers`` builds a prefix for validation. ``packed_experts=True``
    (default) holds the routed experts packed int6 (``gather_qmm``) — the resident-memory + bandwidth
    lever, greedy-exact on the SAME codes as the bf16 ``gather_mm`` reference; ``packed_experts=False``
    dequantizes them to bf16 (the parity fallback).

    ``packed`` (default ``False`` — this single-stream model is the bf16-mixer parity reference)
    holds the int8 mixer (GQA q/k/v/o + dense-FFN) packed as ``nn.QuantizedLinear``
    (``mx.quantized_matmul``) instead of dequantized to bf16 — the ~6 GiB memory lever + the batch-M
    bit-exact substrate, greedy-exact on the SAME int8 codes. The serving entry point
    (:class:`quanta.minimax.batched_runtime_m3.MiniMaxM3BatchedResidentModel`) constructs the inner
    model with ``packed=True``; the shared expert stays bf16 either way."""

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None,
                 packed: bool = False, packed_experts: bool = True,
                 kv_quantized: bool = False, kv_group_size: int = 64, kv_bits: int = 8) -> None:
        self.art = MiniMaxM3Artifact(art_dir)
        self.cfg: MiniMaxM3Config = self.art.cfg
        self.packed = bool(packed)
        self.packed_experts = bool(packed_experts)
        # KV storage mode for make_caches (default bf16 — the M1/M2 parity reference; int8 g64 is the
        # M3-4 serving lever, opt-in). Exposed as quantized_kv/kv_group_size/kv_bits so a paged session
        # can build a bit-identical PagedKVCacheManager from these (never hardcoded — rule 6).
        self.quantized_kv = bool(kv_quantized)
        self.kv_group_size = int(kv_group_size)
        self.kv_bits = int(kv_bits)
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[MiniMaxM3Block] = []
        for i in range(n):  # rule 8: materialize one layer's params, eval, then drop source shards
            blk = _load_block(self.art, self.cfg, i, packed=packed, packed_experts=packed_experts)
            mx.eval(_block_arrays(blk))
            self.layers.append(blk)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n

        # embed / final norm (1+w) / lm_head (untied: tie_word_embeddings=False), bf16
        self.embed_w = self.art.embed()
        self.norm_w = one_plus(self.art.final_norm())
        self.lm_head_w = self.art.lm_head()
        mx.eval([self.embed_w, self.norm_w, self.lm_head_w])
        self.art.release()
        mx.clear_cache()

    @classmethod
    def from_blocks(cls, layers: list[MiniMaxM3Block], embed_w: mx.array, norm_w: mx.array,
                    lm_head_w: mx.array, cfg: MiniMaxM3Config, *, kv_quantized: bool = False,
                    kv_group_size: int = 64, kv_bits: int = 8) -> MiniMaxM3ResidentModel:
        """Construct from pre-built blocks + final-form weights (bypasses artifact load) so the
        model-free parity gate can drive the resident forward on a tiny synthetic model without a
        checkpoint. ``norm_w`` is the already-``(1+w)``-folded final norm; ``embed_w``/``lm_head_w``
        verbatim. ``packed_experts`` is detected from the first MoE block's expert stack; ``packed``
        (the int8 mixer) from whether ``self_attn.q_proj`` is an ``nn.QuantizedLinear``. KV flags
        default bf16 (the parity reference); a paged/serving caller passes ``kv_quantized=True``."""
        self = cls.__new__(cls)
        self.art = None
        self.cfg = cfg
        self.layers = list(layers)
        self.num_layers = len(layers)
        self.embed_w = embed_w
        self.norm_w = norm_w
        self.lm_head_w = lm_head_w
        self.quantized_kv = bool(kv_quantized)
        self.kv_group_size = int(kv_group_size)
        self.kv_bits = int(kv_bits)
        moe_blocks = [b for b in layers if b.is_moe]
        self.packed_experts = bool(moe_blocks) and isinstance(moe_blocks[0].mlp.experts_gate_up, dict)
        self.packed = bool(layers) and isinstance(layers[0].self_attn.q_proj, nn.QuantizedLinear)
        return self

    # --- cache factory ---------------------------------------------------------
    def make_caches(self) -> list[KVCache]:
        """A fresh per-layer GQA KV cache (one :class:`KVCache` per decoder layer) in the configured KV
        mode (``quantized_kv`` / ``kv_group_size`` / ``kv_bits`` — bf16 by default, int8 g64 the serving
        lever). M3 is uniform full-attention so every layer caches identically."""
        return [KVCache(quantized=self.quantized_kv, group_size=self.kv_group_size, bits=self.kv_bits)
                for _ in range(self.num_layers)]

    def _head(self, h: mx.array) -> mx.array:
        """Final Gemma ``(1+w)`` RMSNorm → lm_head: residual ``[1,T,hidden] → [1,T,vocab]``."""
        hh = mx.fast.rms_norm(h, self.norm_w.astype(h.dtype), self.cfg.norm_eps)
        return hh @ self.lm_head_w.T.astype(hh.dtype)

    def embed_tokens(self, token_ids) -> mx.array:
        """Token-id → embedding lookup ``[T, hidden]`` (bf16) — the input the multimodal splice
        (:func:`quanta.minimax.model_vision_m3.splice_image_embeddings`) edits before it is fed back
        as ``inputs_embeds``."""
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        return self.embed_w[ids.reshape(-1)].astype(mx.bfloat16)   # [T, hidden]

    def _initial_hidden(self, token_ids, inputs_embeds: mx.array | None) -> mx.array:
        """Resolve the layer-0 input ``[1,T,hidden]`` from EXACTLY one of token_ids / inputs_embeds.

        ``inputs_embeds`` (the multimodal path — the image-spliced stream) bypasses the embed lookup;
        a ``[T,hidden]`` or ``[1,T,hidden]`` array is accepted. The token-id path is byte-for-byte the
        original ``self.embed_w[ids][None]`` (so every existing text gate stays bit-exact)."""
        if (token_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of token_ids / inputs_embeds (rule 6)")
        if inputs_embeds is not None:
            e = inputs_embeds[None] if inputs_embeds.ndim == 2 else inputs_embeds
            if e.ndim != 3:
                raise ValueError(f"inputs_embeds must be [T,hidden] or [1,T,hidden], got "
                                 f"{inputs_embeds.shape}")
            return e.astype(mx.bfloat16)
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        return self.embed_w[ids.reshape(-1)][None].astype(mx.bfloat16)   # [1,T,hidden]

    def __call__(self, token_ids=None, *, inputs_embeds: mx.array | None = None,
                 caches: list[KVCache] | None = None, use_fast: bool = True,
                 sparse: bool = True) -> mx.array:
        """Logits ``[1,T,vocab]`` from ``token_ids`` OR pre-spliced ``inputs_embeds`` (exactly one —
        the multimodal prefill feeds the image-spliced embedding stream).

        ``caches=None`` ⇒ prefill (run each reference ``MiniMaxM3Block`` with no cache — the M1/M2
        parity-correct path). ``caches`` given ⇒ a cached forward over ``T >= 1`` tokens: each block
        threads its per-layer :class:`KVCache` (grown in place; the attention reads ``cache.offset``
        for partial RoPE and applies a bottom-right causal mask). A ``T``-token cached forward over
        fresh caches is output-equivalent to the ``caches=None`` prefill; a continuation attends the
        new tokens against the grown KV. Gated in ``parity/minimax_m3_runtime_test.py`` (text) and
        ``parity/minimax_m3_splice_test.py`` (the ``inputs_embeds`` equivalence)."""
        h = self._initial_hidden(token_ids, inputs_embeds)         # [1,T,hidden]

        if caches is None:
            for blk in self.layers:
                h = blk(h, cache=None, use_fast=use_fast, sparse=sparse)
                mx.eval(h)                                          # bound the per-layer graph
            return self._head(h)

        if len(caches) != self.num_layers:
            raise ValueError(f"len(caches)={len(caches)} != num_layers={self.num_layers} "
                             f"(one KV cache per layer; refusing to mis-thread state — rule 6)")
        for blk, cache in zip(self.layers, caches, strict=True):
            h = blk(h, cache=cache, use_fast=use_fast, sparse=sparse)
        mx.eval(h)
        return self._head(h)

    # --- long-context chunked prefill (the feasible admit path past chat lengths) -
    def prefill_chunked(self, token_ids=None, caches: list[KVCache] | None = None, *,
                        inputs_embeds: mx.array | None = None, chunk_tokens: int = 4096,
                        use_fast: bool = True, sparse: bool = True) -> mx.array:
        """Consume ``token_ids`` (or pre-spliced ``inputs_embeds`` — the multimodal admit) into
        ``caches`` in ``chunk_tokens`` blocks; return the last position's logits ``[1,1,vocab]`` (same
        contract as a cached :meth:`__call__` over the last token). The feasible path past chat lengths
        — the single-shot :meth:`__call__` holds the whole ``[1,T,hidden]`` window resident, this
        bounds the per-chunk transient to O(chunk). See :func:`chunked_prefill` (the shared driver;
        gated in ``parity/minimax_m3_prefill_chunked_test.py``)."""
        if caches is None or len(caches) != self.num_layers:
            raise ValueError(f"len(caches)={None if caches is None else len(caches)} != "
                             f"num_layers={self.num_layers} (one KV cache per layer; refusing to "
                             f"mis-thread state — rule 6)")
        return chunked_prefill(self.layers, self.embed_w, self.norm_w, self.lm_head_w, self.cfg,
                               token_ids, caches, inputs_embeds=inputs_embeds,
                               chunk_tokens=chunk_tokens, use_fast=use_fast, sparse=sparse)

    # --- multimodal prefill (text + image): embed → splice → forward ------------
    def multimodal_prefill(self, token_ids, vision_tokens: mx.array, *,
                           caches: list[KVCache] | None = None, chunk_tokens: int | None = None,
                           use_fast: bool = True, sparse: bool = True) -> mx.array:
        """One-call image+text prefill: embed ``token_ids``, splice the merged ViT ``vision_tokens``
        into the ``image_token_index`` placeholder rows
        (:func:`quanta.minimax.model_vision_m3.splice_image_embeddings`), then run the decoder over the
        resulting ``inputs_embeds``. ``caches=None`` ⇒ single-shot prefill (logits ``[1,T,vocab]``);
        ``caches`` given ⇒ a cached forward that seeds the KV (``chunk_tokens`` set ⇒ the O(chunk)
        chunked admit, returning ``[1,1,vocab]``). ``vision_tokens`` is the tower output for all images
        concatenated in prompt order (``M == #placeholders``); the splice fails loud on a mismatch
        (rule 6). Gated model-free in ``parity/minimax_m3_splice_test.py`` and re-gated @ 397B in
        ``parity/minimax_m3_multimodal_real.py``."""
        from quanta.minimax.model_vision_m3 import splice_image_embeddings  # noqa: PLC0415

        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids, dtype=mx.int32)
        ids = ids.reshape(-1)
        text_embeds = self.embed_w[ids].astype(mx.bfloat16)        # [T, hidden]
        spliced = splice_image_embeddings(text_embeds, ids, vision_tokens, self.cfg.image_token_index)
        if chunk_tokens is not None:
            if caches is None:
                raise ValueError("chunked multimodal_prefill needs caches (rule 6)")
            return self.prefill_chunked(caches=caches, inputs_embeds=spliced,
                                        chunk_tokens=chunk_tokens, use_fast=use_fast, sparse=sparse)
        return self(inputs_embeds=spliced, caches=caches, use_fast=use_fast, sparse=sparse)

    # --- minimal greedy generate (serving convenience; not the ppl arbiter) ----
    def generate(self, prompt_ids, *, max_new: int = 32, use_fast: bool = True,
                 sparse: bool = True) -> list[int]:
        """Greedy decode: prefill ``prompt_ids`` into a fresh KV cache, then step one token at a time
        (stop on the config eos set). Convenience for serving / a quick sanity check — NOT the
        quant arbiter (that is teacher-forced ppl, ``parity/minimax_m3_ppl.py``; reasoning models
        loop under greedy regardless of quant — CLAUDE.md methodology #4)."""
        ids = list(int(t) for t in prompt_ids)
        if not ids:
            raise ValueError("prompt_ids is empty (generate needs >= 1 token)")
        caches = self.make_caches()
        logits = self(mx.array(ids, dtype=mx.int32), caches=caches, use_fast=use_fast, sparse=sparse)
        nxt = int(mx.argmax(logits[0, -1]).item())
        out = [nxt]
        stop = set(self.cfg.eos_token_ids)
        for _ in range(max_new - 1):
            if nxt in stop:
                break
            logits = self(mx.array([nxt], dtype=mx.int32), caches=caches,
                          use_fast=use_fast, sparse=sparse)
            nxt = int(mx.argmax(logits[0, -1]).item())
            out.append(nxt)
        return out
