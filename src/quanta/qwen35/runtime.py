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
from mlx.utils import tree_flatten

from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.attention import Qwen35Attention
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.decode import Qwen35Cache
from quanta.qwen35.gated_deltanet import GatedDeltaNet
from quanta.qwen35.model import Qwen35Block

# --- artifact -> module weight wiring ----------------------------------------
_LINEAR_PROJS = ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj")
_FULL_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _load_block(art: Qwen35Artifact, cfg: Qwen35Config, i: int) -> Qwen35Block:
    """Build one runnable :class:`Qwen35Block` for layer ``i`` from the dequantized artifact tensors."""
    blk = Qwen35Block(cfg, i)
    norms = art.block_norms(i)
    blk.input_layernorm.weight = norms["input_layernorm"]
    blk.post_attention_layernorm.weight = norms["post_attention_layernorm"]

    m = blk.mixer
    if cfg.is_linear_attention(i):
        assert isinstance(m, GatedDeltaNet)
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
        fa = art.full_attn(i)
        for proj in _FULL_PROJS:
            getattr(m, proj).weight = fa[f"{proj}.weight"]
        m.q_norm = fa["q_norm.weight"]
        m.k_norm = fa["k_norm.weight"]

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

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None) -> None:
        self.art = Qwen35Artifact(art_dir)
        self.cfg = self.art.cfg
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[Qwen35Block] = []
        for i in range(n):  # rule-8: materialize one layer's params, eval, then drop source shards
            blk = _load_block(self.art, self.cfg, i)
            mx.eval(_block_arrays(blk))
            self.layers.append(blk)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n

        # embed / final norm / lm_head (bf16; head may be tied to the embedding)
        self.embed_w = self.art.embed()
        self.norm_w = self.art.final_norm()
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
            hint = seq_hint if seq_hint is not None else off_t + 1
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
