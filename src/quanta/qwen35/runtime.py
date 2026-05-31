"""RAM-resident Qwen3.5-397B-A17B runtime — load the baked artifact and run the resident forward.

Mirrors :class:`quanta.dsv4.runtime.DSV4ResidentModel` and
:class:`quanta.nemotron.runtime.NemotronResidentModel` (the hybrid analogue) for the Qwen3.5 3:1
hybrid text decoder. The model is built **one decoder layer at a time** (materialize that layer's
params from :class:`Qwen35Artifact`, ``mx.eval`` them, drop the artifact's shard handles before the
next — rule 8), so peak load residency is ~one layer, not the whole checkpoint. The deployment target
holds the full quantized model RAM-resident pinned with ``mx.set_wired_limit``; ``n_layers`` builds a
bounded prefix for validation.

Each layer is a :class:`quanta.qwen35.model.Qwen35Block` — the exact module the forward parity gate
(``parity/qwen35_forward_test.py``) validates — populated from the artifact's dequantized weights
(the int4/int8 packed-weight resident path swaps in by replacing the per-block linears /
expert stacks; the call convention does not change). Because the resident block IS the reference
block, ``__call__`` has two regimes that are output-equivalent by construction:

* **prefill** (``caches=None``): run each ``Qwen35Block`` over the whole window with fresh per-layer
  state (chunked Gated-DeltaNet on the 45 linear layers, KV-from-offset-0 GQA on the 15 full layers,
  MoE every layer) — identical to the reference forward; ``capture_layers`` returns the post-layer
  residual stream from this path.
* **decode** (``caches`` given, ``T >= 1`` tokens): thread, per layer, EITHER the Gated-DeltaNet
  recurrent state (linear layers) OR the GQA KV cache (full layers) — chosen by
  ``cfg.is_linear_attention(i)`` — plus the (stateless) MoE every layer, via the SAME ``Qwen35Block``
  forward. Every token completes ALL layers (advancing its state) before the next begins, so the run is
  causally identical to a batched cached forward; the :class:`quanta.qwen35.decode.Qwen35Cache` is
  mutated in place and ``offset`` is the first token's absolute position. ``T == 1`` is the plain decode
  step; ``T > 1`` is how :func:`quanta.qwen35.spec.spec_generate` verifies ``[cur, draft]`` in one call.

Dynamic-YaRN consistency: the full-attention RoPE factor depends on the sequence length
(:meth:`Qwen35Config.effective_yarn_factor`). ``__call__`` accepts a ``seq_hint`` (the total context
length) so a decode step uses the SAME factor the matching prefill used — defaulting to the cache-aware
length per token. The RoPE construction itself is reused from :mod:`quanta.qwen35.attention` (via the
block's ``Qwen35Attention`` mixer); the runtime adds no new RoPE math.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.attention import Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.decode import Qwen35Cache
from quanta.qwen35.gated_deltanet import GatedDeltaNet
from quanta.qwen35.loader import LM_PREFIX
from quanta.qwen35.model import Qwen35Block

# --- artifact -> module weight wiring ----------------------------------------
_LINEAR_PROJS = ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj")
_FULL_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _one_plus(w: mx.array) -> mx.array:
    """Qwen3.5 ``Qwen3_5MoeRMSNorm`` applies ``(1 + weight)`` (Gemma-style; see the HF reference), so
    the resident norm weight is ``source + 1`` and the plain ``weight·normed`` forward (nn.RMSNorm /
    mx.fast.rms_norm / the attention ``_rms``) matches the reference. Applies to input/post-attention
    layernorms, the per-head q/k norms, and the final norm — but NOT the GatedDeltaNet gated norm
    (``Qwen3_5MoeRMSNormGated`` uses plain ``weight``). Kept in the source dtype (bf16)."""
    return w + 1.0


def _load_quant_triplet(art: Qwen35Artifact, base: str
                        ) -> tuple[mx.array, mx.array, mx.array, int, int]:
    """A packed affine weight's three siblings (``.weight_packed`` / ``.weight_scale`` /
    ``.weight_bias`` — verbatim, no dequant) plus its ``(bits, group_size)`` from the manifest.

    The decode width travels with the artifact (rule-6 — the baked manifest is the single source of
    truth, never a hardcoded width that could silently mis-decode a differently-baked artifact).
    Mirrors :func:`quanta.internlm2.runtime._load_quant_triplet`. Fail loud if ``base`` is not an
    ``affine_packed`` weight (a dense projection has no packed codes to hold)."""
    meta = art.manifest.get(base)
    if meta is None or meta.get("format") != "affine_packed":
        raise ValueError(f"{base}: not an affine_packed weight (format="
                         f"{None if meta is None else meta.get('format')!r}); cannot pack (rule-6)")
    return (art.raw(base),
            art.get(base + ".weight_scale"),
            art.get(base + ".weight_bias"),
            int(meta["bits"]), int(meta["group_size"]))


def _packed_linear(art: Qwen35Artifact, base: str, ref: nn.Linear) -> nn.QuantizedLinear:
    """Build a bias-free :class:`mlx.nn.QuantizedLinear` from the artifact's packed triplet at
    ``base``, sized to the freshly-built ``ref`` ``nn.Linear`` it replaces (its ``[out, in]`` shape).

    ``nn.QuantizedLinear.__call__`` dispatches to ``mx.quantized_matmul(transpose=True)`` (rule 1 /
    rule 2), so swapping it in for the ``nn.Linear`` leaves the mixer forward (``self.in_proj_qkv(x)``
    / ``self.q_proj(x)`` …) UNCHANGED while holding the weight PACKED — and the matmul is batch-M
    bit-exact for the M=1 per-stream loop AND (chunked ≤8) for the batched loop-kill (#153 option B,
    M0 ``c503657``: ``mx.quantized_matmul`` is a per-row gemv bit-exact only for M≤~10)."""
    out_dims, in_dims = int(ref.weight.shape[0]), int(ref.weight.shape[1])
    packed, scale, wbias, bits, gs = _load_quant_triplet(art, base)
    ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=gs, bits=bits)
    ql.weight, ql.scales, ql.biases = packed, scale, wbias
    return ql


def _load_block(art: Qwen35Artifact, cfg: Qwen35Config, i: int, *, packed: bool = False) -> Qwen35Block:
    """Build one runnable :class:`Qwen35Block` for layer ``i`` from the artifact tensors.

    ``packed=False`` dequantizes the mixer projections to dense bf16 ``nn.Linear`` (the parity
    reference / fallback). ``packed=True`` (#153 option B) holds each mixer projection as a packed
    ``nn.QuantizedLinear`` (``mx.quantized_matmul``) instead — batch-M bit-exact, the prerequisite for
    the chunked batched loop-kill (a dense-bf16 GEMM reorders its accumulation across batch-M; see
    ``feedback_batched_rope_bf16`` + M0). MoE / norms / conv / ``A_log`` / ``dt_bias`` are identical
    either way. Both mixer halves convert: GDN ``in_proj_*``/``out_proj`` (M1) and GQA ``q/k/v/o``
    projections (M2)."""
    blk = Qwen35Block(cfg, i)
    norms = art.block_norms(i)
    blk.input_layernorm.weight = _one_plus(norms["input_layernorm"])          # (1+w) convention
    blk.post_attention_layernorm.weight = _one_plus(norms["post_attention_layernorm"])

    m = blk.mixer
    if cfg.is_linear_attention(i):
        assert isinstance(m, GatedDeltaNet)
        if packed:
            # #153 option B (M1): the five GDN projections stay PACKED (nn.QuantizedLinear →
            # mx.quantized_matmul), never dequantized — batch-M bit-exact. The dense control tensors
            # (conv / A_log / dt_bias / per-head norm) read bf16 exactly as the dequant path does
            # (read() casts dense→bf16; the mixer casts A_log/norm back to fp32 at use — same as
            # linear_attn()). conv1d is a per-row windowed sum, not a batch-M GEMM, so bf16 is fine.
            p = f"{LM_PREFIX}layers.{i}.linear_attn."
            for proj in _LINEAR_PROJS:
                setattr(m, proj, _packed_linear(art, p + proj, getattr(m, proj)))
            m.conv_weight = art.read(p + "conv1d.weight")
            if m.conv_weight.ndim == 3:  # [C,1,K] → [C,K]
                m.conv_weight = m.conv_weight.reshape(m.conv_weight.shape[0], m.conv_weight.shape[-1])
            if art.has(p + "conv1d.bias"):
                m.conv_bias = art.read(p + "conv1d.bias")
            m.A_log = art.read(p + "A_log")
            m.dt_bias = art.read(p + "dt_bias")
            m.norm = art.read(p + "norm.weight")
        else:
            la = art.linear_attn(i)
            for proj in _LINEAR_PROJS:
                getattr(m, proj).weight = la[f"{proj}.weight"]
            m.conv_weight = la["conv1d.weight"]
            # depthwise conv may ship as [C,1,K]; the mixer expects [C,K] (loader squeezes it)
            if m.conv_weight.ndim == 3:
                m.conv_weight = m.conv_weight.reshape(m.conv_weight.shape[0], m.conv_weight.shape[-1])
            m.conv_bias = la.get("conv1d.bias", m.conv_bias)
            m.A_log = la["A_log"]
            m.dt_bias = la["dt_bias"]
            m.norm = la["norm.weight"]
    else:
        assert isinstance(m, Qwen35Attention)
        if packed:
            # #153 option B (M2): the four GQA projections stay PACKED (nn.QuantizedLinear); the
            # per-head q/k RMSNorm weights read bf16 with the (1+w) convention (as the dequant path).
            p = f"{LM_PREFIX}layers.{i}.self_attn."
            for proj in _FULL_PROJS:
                setattr(m, proj, _packed_linear(art, p + proj, getattr(m, proj)))
            m.q_norm = _one_plus(art.read(p + "q_norm.weight"))
            m.k_norm = _one_plus(art.read(p + "k_norm.weight"))
        else:
            fa = art.full_attn(i)
            for proj in _FULL_PROJS:
                getattr(m, proj).weight = fa[f"{proj}.weight"]
            m.q_norm = _one_plus(fa["q_norm.weight"])                         # (1+w) convention
            m.k_norm = _one_plus(fa["k_norm.weight"])

    moe = art.moe(i)
    blk.mlp.gate = moe["gate"]
    blk.mlp.set_experts(moe["experts_gate_up"], moe["experts_down"])
    blk.mlp.shared_gate_proj = moe["shared_gate_proj"]
    blk.mlp.shared_up_proj = moe["shared_up_proj"]
    blk.mlp.shared_down_proj = moe["shared_down_proj"]
    blk.mlp.shared_expert_gate = moe["shared_expert_gate"]
    return blk


def _block_arrays(blk: Qwen35Block) -> list[mx.array]:
    """Every resident array of one block — nn params plus the MoE expert stacks (plain attrs)."""
    arrs = [v for _, v in tree_flatten(blk.parameters())]
    arrs += [blk.mlp.experts_gate_up, blk.mlp.experts_down, blk.mlp.gate,
             blk.mlp.shared_gate_proj, blk.mlp.shared_up_proj, blk.mlp.shared_down_proj,
             blk.mlp.shared_expert_gate]
    return arrs


class Qwen35ResidentModel:
    """RAM-resident Qwen3.5 hybrid decoder — prefill via the reference block, decode via cached state.

    Built one layer at a time (materialize, then release the artifact's shard handles) for bounded load
    residency. ``n_layers`` builds a prefix for validation. ``__call__`` matches the spec/decode/generate
    contract the sibling stacks use.
    """

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None,
                 packed: bool = False) -> None:
        self.art = Qwen35Artifact(art_dir)
        self.cfg = self.art.cfg
        self.packed = packed     # #153 option B: hold mixer projections packed (mx.quantized_matmul)
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[Qwen35Block] = []
        for i in range(n):  # rule-8: materialize one layer's params, eval, then drop source shards
            blk = _load_block(self.art, self.cfg, i, packed=packed)
            mx.eval(_block_arrays(blk))
            self.layers.append(blk)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n

        # embed / final norm / lm_head (bf16; head may be tied to the embedding)
        self.embed_w = self.art.embed()
        self.norm_w = _one_plus(self.art.final_norm())          # (1+w) Qwen3_5MoeRMSNorm convention
        self.lm_head_w = self.art.lm_head()
        mx.eval([self.embed_w, self.norm_w, self.lm_head_w])
        self.art.release()
        mx.clear_cache()

    # --- cache factory (consumed by generate / spec) -------------------------
    def make_caches(self) -> Qwen35Cache:
        """A fresh per-layer decode cache typed by the config schedule (KV / recurrent). int8 KV on
        full-attn layers by default — Kimi pattern (#47); linear-attn layers stay recurrent
        (O(1) state, no benefit from int8)."""
        return Qwen35Cache(self.num_layers, self.cfg, quantized=True)

    def embed(self) -> mx.array:
        return self.embed_w

    def lm_head(self) -> mx.array:
        return self.lm_head_w

    def _head(self, h: mx.array) -> mx.array:
        """Final RMSNorm -> lm_head: residual ``[1,T,hidden] -> [1,T,vocab]``."""
        hh = mx.fast.rms_norm(h, self.norm_w.astype(h.dtype), self.cfg.norm_eps)
        return hh @ self.lm_head_w.T.astype(hh.dtype)

    def __call__(self, token_ids, *, caches=None, offset=0, capture_layers=None,
                 use_fast: bool = True, seq_hint=None):
        """Logits ``[1,T,vocab]`` (or ``(logits, {layer: hidden})`` when ``capture_layers`` is set).

        ``caches=None`` ⇒ prefill (run each reference ``Qwen35Block`` with fresh per-layer state —
        parity-correct). ``caches`` given ⇒ decode over ``T >= 1`` tokens, each stepped per layer
        through the SAME block forward threading the layer's KV / recurrent state. Every token completes
        ALL layers — advancing its state — before the next begins, so the run is causally identical to a
        batched cached forward; the cache is mutated in place and ``offset`` is the first token's absolute
        position. ``capture_layers`` returns each captured layer's residual stacked over the ``T``
        positions as ``[T, hidden]`` (the same shape the prefill path returns), for the MTP feature.
        """
        ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
        ids = ids.reshape(-1)                                  # [T]
        h = self.embed_w[ids][None].astype(mx.bfloat16)        # [1,T,hidden]
        cap_set = set(capture_layers) if capture_layers else set()

        if caches is None:
            caps: dict[int, mx.array] = {}
            for i, blk in enumerate(self.layers):
                h, _, _ = blk(h, cache=None, state=None, conv_state=None, use_fast=use_fast,
                              seq_hint=seq_hint if seq_hint is not None else (offset + h.shape[1]))
                if i in cap_set:
                    caps[i] = h[0]                              # [T, hidden] residual after layer i
                mx.eval(h)
            logits = self._head(h)
            if cap_set:
                return logits, caps
            return logits

        caps_acc: dict[int, list[mx.array]] = {layer: [] for layer in cap_set}
        hts: list[mx.array] = []
        for t in range(h.shape[1]):
            ht = h[:, t:t + 1]                                 # [1,1,hidden] token t
            off_t = offset + t                                 # its absolute position
            hint = seq_hint if seq_hint is not None else caches.yarn_seq(off_t + 1, self.cfg)
            for i, blk in enumerate(self.layers):
                ht = self._decode_block(ht, blk, i, caches, off_t, use_fast, hint)
                if i in cap_set:
                    caps_acc[i].append(ht[0, 0])               # [hidden] residual after layer i
            mx.eval(ht)                                        # materialize this token's state growth
            hts.append(ht)
        h = hts[0] if len(hts) == 1 else mx.concatenate(hts, axis=1)  # [1,T,hidden]
        logits = self._head(h)
        if cap_set:
            return logits, {layer: mx.stack(v, axis=0) for layer, v in caps_acc.items()}
        return logits

    def _decode_block(self, ht: mx.array, blk: Qwen35Block, i: int, caches: Qwen35Cache,
                      offset: int, use_fast: bool, seq_hint: int) -> mx.array:
        """One single-token decoder block ``[1,1,hidden] -> [1,1,hidden]`` threading the layer's state.

        Reuses the reference :class:`Qwen35Block` forward (so the MoE + residual mixing are the exact
        prefill code); the only difference from prefill is that the mixer reads/writes the cached
        KV / recurrent state instead of starting fresh. Output-equivalent to ``Qwen35Block`` at this
        absolute position (gated == prefill in ``parity/qwen35_decode_attn_test.py``)."""
        lc = caches[i]
        if blk.is_linear:
            # Engage the O(1) decode recurrence from token 0: the GatedDeltaNet module treats
            # conv_state=None as PREFILL, so seed zero state + zero conv window when the layer is
            # empty (matching Qwen35Model.make_state) — a fresh decode from offset 0 == prefill there.
            m = blk.mixer
            state = lc.recurrent_state
            conv = lc.conv_state
            if conv is None:
                state = mx.zeros((1, m.hv, m.dk, m.dv), dtype=mx.float32)
                conv = mx.zeros((1, m.k - 1, m.conv_dim), dtype=ht.dtype)
            out, rec, conv = blk(ht, state=state, conv_state=conv)
            lc.commit(conv, rec)
            return out
        # full attention: the block mutates the KV cache in place; pin the YaRN factor via seq_hint
        out, _, _ = blk(ht, cache=lc, use_fast=use_fast, seq_hint=seq_hint)
        return out
