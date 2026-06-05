"""Resident quantized Nemotron-H runtime — load the baked artifact and run packed weights.

Mirrors :class:`quanta.runtime.ResidentModel` (Kimi) for the hybrid. The artifact stores dense
always-on linears (mamba in/out-proj, attention q/k/v/o, latent fc1/fc2, shared expert) as affine
int8, routed experts as packed int4, and SSM core / norms / router / embeddings / head as bf16.
Because the mixers call their projections as ``proj(x)``, dense linears just become
``nn.QuantizedLinear`` (same call, forward unchanged); routed experts run through
:class:`NemotronQuantizedMoE` (gather_qmm over packed ``[E,*]`` stacks). The whole model is held
RAM-resident (~68 GB int4) and decode reads packed 4-bit weights — the bandwidth win.

``NemotronResidentModel`` matches :class:`NemotronModel`'s call signature (returns
``(logits, ssm, conv)``), so ``quanta.nemotron.generate.generate`` and the ppl harness run on it
directly. Built one layer at a time (materialize, then drop the shard mmaps) for bounded load
residency. Output-equivalent to the dequantized reference (gated in parity.nemotron_resident_ppl).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.attention import KVCache
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.model import NemotronBlock
from quanta.nemotron.moe import NemotronQuantizedMoE
from quanta.nemotron.mtp import NemotronMTP

EMBED, NORMF, HEAD = "backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"


def _qlin(art: NemotronArtifact, key: str) -> nn.QuantizedLinear:
    """nn.QuantizedLinear from an affine-int8 ``key`` (weight_packed/scale/bias)."""
    m = art.manifest[key]
    packed = art.raw(key + ".weight_packed")
    out, in_packed = packed.shape
    in_ = in_packed * 32 // m["bits"]
    ql = nn.QuantizedLinear(in_, out, bias=False, group_size=m["group_size"], bits=m["bits"])
    ql.weight, ql.scales, ql.biases = packed, art.raw(key + ".weight_scale"), art.raw(key + ".weight_bias")
    return ql


def _packed_stack(art: NemotronArtifact, mpre: str, proj: str, n: int) -> dict[str, mx.array]:
    """Stack ``n`` experts' packed/scale/bias for one projection → ``[E,*]`` for gather_qmm."""
    p = [art.raw(f"{mpre}experts.{e}.{proj}.weight_packed") for e in range(n)]
    s = [art.raw(f"{mpre}experts.{e}.{proj}.weight_scale") for e in range(n)]
    b = [art.raw(f"{mpre}experts.{e}.{proj}.weight_bias") for e in range(n)]
    return {"packed": mx.stack(p), "scale": mx.stack(s), "bias": mx.stack(b)}


def build_resident_block(art: NemotronArtifact, cfg: NemotronHConfig, i: int) -> NemotronBlock:
    """Build one runnable quantized block (mamba / attention / moe) from the artifact."""
    kind = cfg.layer_kind(i)
    blk = NemotronBlock(cfg, kind)
    pre, mpre = f"backbone.layers.{i}.", f"backbone.layers.{i}.mixer."
    blk.norm.weight = art.raw(pre + "norm.weight")  # per-layer input norm (bf16)
    m = blk.mixer
    if kind == "mamba":
        m.in_proj, m.out_proj = _qlin(art, mpre + "in_proj"), _qlin(art, mpre + "out_proj")
        m.norm.weight = art.raw(mpre + "norm.weight")  # gated RMSNorm (bf16)
        m.conv_weight, m.conv_bias = art.raw(mpre + "conv1d.weight"), art.raw(mpre + "conv1d.bias")
        m.A_log, m.D, m.dt_bias = art.raw(mpre + "A_log"), art.raw(mpre + "D"), art.raw(mpre + "dt_bias")
    elif kind == "attention":
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(m, proj, _qlin(art, mpre + proj))
    else:  # moe → quantized latent MoE
        up0 = mpre + "experts.0.up_proj"
        q = NemotronQuantizedMoE(cfg, group_size=art.manifest[up0]["group_size"], bits=art.manifest[up0]["bits"])
        q.gate_weight = art.raw(mpre + "gate.weight")
        q.e_score_correction_bias = art.raw(mpre + "gate.e_score_correction_bias")
        q.fc1_latent_proj, q.fc2_latent_proj = _qlin(art, mpre + "fc1_latent_proj"), _qlin(art, mpre + "fc2_latent_proj")
        q.shared_up = _qlin(art, mpre + "shared_experts.up_proj")
        q.shared_down = _qlin(art, mpre + "shared_experts.down_proj")
        n = cfg.n_routed_experts
        q.set_experts(_packed_stack(art, mpre, "up_proj", n), _packed_stack(art, mpre, "down_proj", n))
        blk.mixer = q
    return blk


def build_resident_mtp(art: NemotronArtifact, cfg: NemotronHConfig, *,
                       draft_topk: int | None = None) -> NemotronMTP:
    """Build the runnable quantized native-MTP draft head from the baked **sidecar** artifact
    (``...-quanta_int4rtn_g64_mtp``) — the MTP-M2 loader, mirroring :func:`build_resident_block` for
    the head's two stacked sub-blocks + the fusion.

    Same per-tensor policy as the backbone (the sidecar was baked under it, ``quant_policy.classify``):
    the 512 routed experts of the moe sub-block run **packed int4** via ``mx.gather_qmm``
    (:class:`NemotronQuantizedMoE`); the always-on dense projections (``eh_proj``, attn q/k/v/o, latent
    fc1/fc2, shared expert) are **int8** ``nn.QuantizedLinear`` (``_qlin``); the fusion / sub-block /
    final norms + router gate/bias stay **bf16**. Returns a :class:`quanta.nemotron.mtp.NemotronMTP`
    holder — the drafter :func:`quanta.nemotron.spec.spec_generate_k` consumes as
    ``mtp(prev_hidden, token_emb, head) -> (logits, new_hidden)``. ``draft_topk`` (optional) routes the
    moe sub-block through fewer experts (a lighter drafter; the main verify path is unaffected, so it is
    a pure speed lever — losslessness holds). The MTP source weights are tiny (~6.6 GiB), loaded in one
    shot (no per-layer streaming needed)."""
    holder = NemotronMTP(cfg, draft_topk=draft_topk)
    mtp = holder.module
    l0, l1 = "mtp.layers.0.", "mtp.layers.1."

    # fusion: enorm/hnorm (bf16) + eh_proj (int8 QuantizedLinear, concat([enorm(e), hnorm(hid)]) -> hid)
    mtp.enorm.weight = art.raw(l0 + "enorm.weight")
    mtp.hnorm.weight = art.raw(l0 + "hnorm.weight")
    mtp.eh_proj = _qlin(art, l0 + "eh_proj")

    # attn sub-block (mtp.layers.0): pre-norm (bf16) + int8 q/k/v/o
    a = mtp.attn_block
    a.norm.weight = art.raw(l0 + "norm.weight")
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(a.mixer, proj, _qlin(art, l0 + "mixer." + proj))

    # moe sub-block (mtp.layers.1): pre-norm (bf16) + quantized latent MoE (packed int4 experts)
    mpre = l1 + "mixer."
    up0 = mpre + "experts.0.up_proj"
    q = NemotronQuantizedMoE(cfg, group_size=art.manifest[up0]["group_size"], bits=art.manifest[up0]["bits"])
    q.gate_weight = art.raw(mpre + "gate.weight")
    q.e_score_correction_bias = art.raw(mpre + "gate.e_score_correction_bias")
    q.fc1_latent_proj, q.fc2_latent_proj = _qlin(art, mpre + "fc1_latent_proj"), _qlin(art, mpre + "fc2_latent_proj")
    q.shared_up = _qlin(art, mpre + "shared_experts.up_proj")
    q.shared_down = _qlin(art, mpre + "shared_experts.down_proj")
    n = cfg.n_routed_experts
    q.set_experts(_packed_stack(art, mpre, "up_proj", n), _packed_stack(art, mpre, "down_proj", n))
    mtp.moe_block.norm.weight = art.raw(l1 + "norm.weight")
    mtp.moe_block.mixer = q

    mtp.final_layernorm.weight = art.raw(l1 + "final_layernorm.weight")

    arrs = [v for _, v in tree_flatten(mtp.parameters())]
    arrs += list(q._up.values()) + list(q._down.values())
    mx.eval(arrs)
    return holder


def _block_arrays(blk: NemotronBlock) -> list[mx.array]:
    """All resident arrays of a block — nn params plus the MoE packed stacks (dict attrs, not params)."""
    arrs = [v for _, v in tree_flatten(blk.parameters())]
    if isinstance(blk.mixer, NemotronQuantizedMoE):
        arrs += list(blk.mixer._up.values()) + list(blk.mixer._down.values())
    return arrs


class NemotronResidentModel:
    """RAM-resident quantized Nemotron-H — same call signature as :class:`NemotronModel`."""

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None) -> None:
        self.art = NemotronArtifact(art_dir)
        self.cfg = NemotronHConfig.from_pretrained(art_dir)
        self.kinds = self.cfg.layers_block_type
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers: list[NemotronBlock] = []
        for i in range(n):  # one layer resident at load, then drop the shard mmaps
            blk = build_resident_block(self.art, self.cfg, i)
            mx.eval(_block_arrays(blk))
            self.layers.append(blk)
            self.art.release()
            mx.clear_cache()
        self.embed_w = self.art.raw(EMBED)            # bf16 dense [vocab, hidden]
        self.norm_f = self.art.raw(NORMF)
        self.lm_head_w = self.art.raw(EMBED if self.cfg.tie_word_embeddings else HEAD)
        mx.eval(self.embed_w, self.norm_f, self.lm_head_w)
        self.art.release()
        self._cmix: dict[int, object] | None = None  # lazy per-block compiled decode mixers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    # --- spec-contract adapter (quanta.nemotron.spec) --------------------------
    def make_caches(self, *, max_rollback: int = 8) -> tuple[list, list, list]:
        """A fresh ``(caches, ssm, conv)`` triple for the spec contract
        (:func:`quanta.nemotron.spec._capture_state` calls this with no args).

        Per **attention** layer a :class:`~quanta.nemotron.attention.KVCache` (sized by
        ``max_rollback`` — spec-decode rolls a verify of ``k+1`` consumed tokens back to the pre-verify
        offset, so ``max_rollback >= k+1``; default **8** covers ``k <= 7``, matching the batched
        runtime). ``ssm``/``conv`` start ``[None] * n`` (the prefill fills the Mamba recurrence); they
        are **real lists** so the inner ``__call__`` mutates ``ssm[i]``/``conv[i]`` in place and the
        spec loop's snapshot/restore sees the updates."""
        kinds = self.cfg.layers_block_type
        gs = min(128, getattr(self.cfg, "head_dim", 128))  # cap KV group_size at head_dim (real=128)
        caches = [KVCache(max_rollback=max_rollback, group_size=gs) if k == "attention" else None
                  for k in kinds]
        n = len(kinds)
        return caches, [None] * n, [None] * n

    def truncate(self, caches, length: int) -> None:
        """Roll the per-attention-layer KV caches back to ``length`` consumed positions (drop rejected
        spec drafts) — the ``model.truncate`` hook :func:`quanta.nemotron.spec._rollback` prefers.

        Only the **sliceable** KV is handled here. The Mamba ``(ssm, conv)`` recurrence is a summary of
        every consumed token and CANNOT be sliced; the spec loop restores it from a pre-verify snapshot
        + re-runs the committed prefix (:func:`quanta.nemotron.spec.spec_generate` /
        :func:`~quanta.nemotron.spec.spec_generate_k`). ``KVCache.truncate`` fails loud past
        ``max_rollback`` (rule 6 — never silently keep a diverged state)."""
        if caches is None:
            return
        for c in caches:
            if c is not None:
                c.truncate(length)

    def _decode_mixers(self) -> dict[int, object]:
        """Compile each mamba/moe mixer's decode step once (static [1,1,H] shapes) — fuses the
        per-layer op sequence so decode launches far fewer kernels (the step is op-launch bound).
        Attention keeps a growing KV cache (variable shape) so it stays eager. Weights are
        captured per block; output-equivalent to the eager path (mx.compile only fuses)."""
        if self._cmix is None:
            cmix: dict[int, object] = {}
            for i, blk in enumerate(self.layers):
                if blk.kind == "mamba":
                    cmix[i] = mx.compile(lambda x, s, c, b=blk: b.mixer(x, state=s, conv_state=c))
                elif blk.kind == "moe":
                    cmix[i] = mx.compile(lambda x, b=blk: b.mixer(x))
            self._cmix = cmix
        return self._cmix

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None, offset=0,
                 capture_layers=None, use_fast=True, compiled=True, mamba_chunked_cont=False):
        """Logits ``[1, t, vocab]`` + updated mamba state lists. All-``None`` state ⇒ fresh prefill.
        At decode (``t == 1``) the mamba/moe mixers run through compiled fused graphs by default.

        ``offset`` is accepted for the spec contract (:func:`quanta.nemotron.spec._forward`) and
        intentionally **ignored** — each attention layer derives its absolute position from its own
        ``KVCache.offset`` and the Mamba layers are offset-free (the recurrence carries its own state),
        so the position is fully implicit in the threaded ``caches``/``ssm``/``conv``. Kept as a named
        parameter so the resident model is a drop-in for the spec loop without an adapter wrapper.

        ``capture_layers`` (iterable of ``int`` layer indices or ``None``) is the adapter promised
        by :func:`quanta.nemotron.spec._forward` for native-MTP speculation: when set, the
        per-layer post-residual hidden state ``h`` is captured *after* layer ``i`` for every
        ``i in capture_layers``, and the return shape switches to ``(logits, caps_dict)`` (mirrors
        :class:`quanta.dsv4.runtime.DSV4ResidentModel`'s capture contract). Default ``None``
        preserves the legacy ``(logits, ssm, conv)`` return for all existing callers (generate
        loop, oMLX engine, plain k=2 spec which only needs the next-token logits).
        """
        h = self.embed_w[token_ids][None].astype(mx.bfloat16)
        n = len(self.layers)
        caches = caches if caches is not None else [None] * n
        ssm = ssm if ssm is not None else [None] * n
        conv = conv if conv is not None else [None] * n
        cmix = self._decode_mixers() if (compiled and h.shape[1] == 1) else None
        capture_set = set(int(i) for i in capture_layers) if capture_layers else None
        caps: dict[int, mx.array] = {} if capture_set is not None else None  # type: ignore[assignment]
        for i, blk in enumerate(self.layers):
            if cmix is not None and blk.kind == "mamba":
                y, ssm[i], conv[i] = cmix[i](blk.norm(h), ssm[i], conv[i])
                h = h + y
            elif cmix is not None and blk.kind == "moe":
                h = h + cmix[i](blk.norm(h))
            else:
                h, ssm[i], conv[i] = blk(h, cache=caches[i], ssm_state=ssm[i], conv_state=conv[i],
                                         use_fast=use_fast, chunked_cont=mamba_chunked_cont)
            if capture_set is not None and i in capture_set:
                # Strip the leading batch-1 dim — match the ``[T, hidden]`` capture shape
                # convention used by :class:`quanta.dsv4.runtime.DSV4ResidentModel` and
                # consumed by :func:`quanta.nemotron.spec.spec_generate_tree` as
                # ``caps[last][-1][None, None] -> [1, 1, hidden]`` for the MTP feature.
                caps[i] = h[0]
        h = mx.fast.rms_norm(h, self.norm_f.astype(h.dtype), self.cfg.norm_eps)
        logits = h @ self.lm_head_w.T
        if caps is not None:
            return logits, caps
        return logits, ssm, conv
