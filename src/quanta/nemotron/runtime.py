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
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.model import NemotronBlock
from quanta.nemotron.moe import NemotronQuantizedMoE

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

    def __call__(self, token_ids, *, caches=None, ssm=None, conv=None,
                 capture_layers=None, use_fast=True, compiled=True, mamba_chunked_cont=False):
        """Logits ``[1, t, vocab]`` + updated mamba state lists. All-``None`` state ⇒ fresh prefill.
        At decode (``t == 1``) the mamba/moe mixers run through compiled fused graphs by default.

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
