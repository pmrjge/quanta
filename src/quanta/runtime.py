"""Resident quantized runtime — load a baked artifact and build runnable decoder layers.

The artifact stores attention / dense-MLP / lm_head weights as affine int8 (mx.quantize
layout), routed experts as packed int3/int4, and shared expert + norms + router as bf16.
Because ``MLAAttention``/``DenseMLP`` invoke their projections as ``proj(x)``, we just swap
``nn.Linear`` → ``nn.QuantizedLinear`` (same call) and the existing forward runs unchanged;
routed experts use :class:`QuantizedSparseMoE` (gather_qmm). Norms/shared/router load bf16.

``ResidentModel`` mirrors :class:`KimiModel.__call__`, so :func:`quanta.generate.generate`
and the ppl harness run on it directly. Built for the full ~427 GB artifact held RAM-resident
(``mx.set_wired_limit``); ``build_resident_layer`` is the per-layer unit, validatable on a
single baked layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from quanta.config import KimiTextConfig
from quanta.loader import TEXT_PREFIX
from quanta.modeling.decoder import DenseDecoderLayer, MoEDecoderLayer
from quanta.modeling.quantized import QuantizedSparseMoE
from quanta.modeling.xattention import DEFAULT_SPARSE, XAttnConfig

LM_HEAD_KEY = "language_model.lm_head"
EMBED_KEY = f"{TEXT_PREFIX}embed_tokens"
FINAL_NORM_KEY = f"{TEXT_PREFIX}norm.weight"


class ResidentArtifact:
    """Reader over a baked artifact dir (config.json + manifest.json + index + safetensors)."""

    def __init__(self, art_dir: str | Path) -> None:
        self.dir = Path(art_dir)
        self.cfg = KimiTextConfig.from_pretrained(art_dir)  # self-contained config
        self.weight_map: dict[str, str] = json.loads(
            (self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.manifest: dict[str, dict] = json.loads((self.dir / "manifest.json").read_text())["tensors"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    def get(self, name: str) -> mx.array:
        fn = self.weight_map[name]
        if fn not in self._shards:
            self._shards[fn] = mx.load(str(self.dir / fn))
        return self._shards[fn][name]

    def release(self) -> None:
        self._shards.clear()


def _quant_linear(art: ResidentArtifact, key: str) -> nn.QuantizedLinear:
    """Build an nn.QuantizedLinear from an int8/affine ``key`` (weight_packed/scale/bias)."""
    m = art.manifest[key]
    packed = art.get(f"{key}.weight_packed")
    out, in_packed = packed.shape
    in_ = in_packed * 32 // m["bits"]
    ql = nn.QuantizedLinear(in_, out, bias=False, group_size=m["group_size"], bits=m["bits"])
    ql.weight = packed
    ql.scales = art.get(f"{key}.weight_scale")
    ql.biases = art.get(f"{key}.weight_bias")
    return ql


def _load_quant_attention(layer, art: ResidentArtifact, pre: str) -> None:
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        setattr(layer.self_attn, proj, _quant_linear(art, f"{pre}self_attn.{proj}"))
    layer.self_attn.q_a_layernorm.weight = art.get(f"{pre}self_attn.q_a_layernorm.weight")
    layer.self_attn.kv_a_layernorm.weight = art.get(f"{pre}self_attn.kv_a_layernorm.weight")
    layer.input_layernorm.weight = art.get(f"{pre}input_layernorm.weight")
    layer.post_attention_layernorm.weight = art.get(f"{pre}post_attention_layernorm.weight")


def build_resident_layer(art: ResidentArtifact, layer_idx: int):
    """Build a runnable quantized decoder layer (dense L0 or MoE) from the artifact."""
    cfg = art.cfg
    pre = f"{TEXT_PREFIX}layers.{layer_idx}."
    if cfg.is_dense_layer(layer_idx):
        layer = DenseDecoderLayer(cfg)
        _load_quant_attention(layer, art, pre)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, proj, _quant_linear(art, f"{pre}mlp.{proj}"))
        return layer

    layer = MoEDecoderLayer(cfg)
    _load_quant_attention(layer, art, pre)
    egs = art.manifest[f"{pre}mlp.experts.0.gate_proj"]["group_size"]  # bake-wide uniform; gather_qmm takes one
    qmoe = QuantizedSparseMoE(cfg, group_size=egs)
    qmoe.gate.weight = art.get(f"{pre}mlp.gate.weight")
    qmoe.gate.e_score_correction_bias = art.get(f"{pre}mlp.gate.e_score_correction_bias")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        setattr(qmoe.shared_experts, proj, _quant_linear_or_dense(art, qmoe.shared_experts, proj, pre))
    qmoe.set_experts(*_load_expert_stacks(art, cfg, pre))
    layer.mlp = qmoe
    return layer


def _quant_linear_or_dense(art, module, proj, pre):
    """Shared expert is bf16-dense in the artifact → keep the nn.Linear, just load its weight."""
    lin = getattr(module, proj)
    lin.weight = art.get(f"{pre}mlp.shared_experts.{proj}.weight")
    return lin


def _load_expert_stacks(art: ResidentArtifact, cfg: KimiTextConfig, pre: str):
    """Group routed experts by ``(projection, quant width)`` into packed stacks for gather_qmm.

    The bake's DP allocates bits per ``(expert, projection)`` — an expert may be ``gate=int3``
    yet ``down=int4`` — so each projection is stacked per width independently. Returns
    ``(stacks, pbits, pslot)``: ``stacks[proj][bits]`` the packed stack, and per projection each
    expert's width (``pbits[proj][e]``) and its slot within that width's stack (``pslot[proj][e]``).
    """
    projs = ("gate", "up", "down")
    by: dict[str, dict[int, dict[str, list]]] = {p: {} for p in projs}
    pbits = {p: [0] * cfg.n_routed_experts for p in projs}
    pslot = {p: [0] * cfg.n_routed_experts for p in projs}
    for e in range(cfg.n_routed_experts):
        for p in projs:
            k = f"{pre}mlp.experts.{e}.{p}_proj"
            bits = art.manifest[k]["bits"]
            d = by[p].setdefault(bits, {"packed": [], "scale": [], "bias": []})
            pbits[p][e] = bits
            pslot[p][e] = len(d["packed"])
            d["packed"].append(art.get(f"{k}.weight_packed"))
            d["scale"].append(art.get(f"{k}.weight_scale"))
            d["bias"].append(art.get(f"{k}.weight_bias"))
    stacks = {p: {bits: {kk: mx.stack(vv) for kk, vv in d.items()} for bits, d in by[p].items()}
              for p in projs}
    pbits = {p: mx.array(pbits[p], mx.int32) for p in projs}
    pslot = {p: mx.array(pslot[p], mx.int32) for p in projs}
    return stacks, pbits, pslot


def _quant_embedding(art: ResidentArtifact, key: str) -> nn.QuantizedEmbedding:
    m = art.manifest[key]
    packed = art.get(f"{key}.weight_packed")
    num, in_packed = packed.shape
    dims = in_packed * 32 // m["bits"]
    emb = nn.QuantizedEmbedding(num, dims, group_size=m["group_size"], bits=m["bits"])
    emb.weight = packed
    emb.scales = art.get(f"{key}.weight_scale")
    emb.biases = art.get(f"{key}.weight_bias")
    return emb


def _layer_arrays(layer) -> list[mx.array]:
    """Every resident array of a built layer — nn params **and** the MoE expert stacks (which are
    plain dict attributes, not nn parameters), so we can materialize a layer before dropping the
    source shard mmaps."""
    arrs = [v for _, v in tree_flatten(layer.parameters())]
    mlp = getattr(layer, "mlp", None)
    if isinstance(mlp, QuantizedSparseMoE):
        arrs += [a for bw in mlp._stacks.values() for s in bw.values() for a in s.values()]
        arrs += list(mlp._pbits.values()) + list(mlp._pslot.values())
        arrs += [a for bw in mlp._rmap.values() for a in bw.values()]
    return arrs


class ResidentModel:
    """RAM-resident quantized Kimi-K2.6 — same call signature as :class:`KimiModel`.

    Built one layer at a time (materialize its weights, then drop the source shard mmaps) so peak
    load residency is ~one layer, not the whole shard set — the deployment target is the full
    quantized model pinned with ``mx.set_wired_limit``. ``generate`` / the ppl harness run on it
    directly. ``n_layers`` builds a prefix for bounded validation.
    """

    def __init__(self, art_dir: str | Path, *, n_layers: int | None = None) -> None:
        self.art = ResidentArtifact(art_dir)
        self.cfg = self.art.cfg
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        self.layers = []
        for i in range(n):  # memory discipline: materialize each layer, then release its shard mmaps
            layer = build_resident_layer(self.art, i)
            mx.eval(_layer_arrays(layer))
            self.layers.append(layer)
            self.art.release()
            mx.clear_cache()
        self.num_layers = n  # for KV-cache sizing by the omlx engine / generate
        self.embed = _quant_embedding(self.art, EMBED_KEY)
        self.norm_w = self.art.get(FINAL_NORM_KEY)
        self.lm_head = _quant_linear(self.art, LM_HEAD_KEY)
        mx.eval([v for _, v in tree_flatten(self.embed.parameters())], self.norm_w,
                [v for _, v in tree_flatten(self.lm_head.parameters())])
        self.art.release()

    def __call__(
        self, token_ids: mx.array, *, n_layers: int | None = None, use_fast: bool = True,
        caches: list | None = None, offset: int = 0,
        sparse: XAttnConfig | None = DEFAULT_SPARSE, absorbed: bool = False,
    ) -> mx.array:
        n = len(self.layers) if n_layers is None else n_layers
        h = self.embed(token_ids)[None].astype(mx.bfloat16)
        pos = mx.arange(offset, offset + h.shape[1])
        for i in range(n):
            layer = self.layers[i]
            if sparse is not None and isinstance(layer.mlp, QuantizedSparseMoE):
                layer.self_attn.sparse = sparse
            layer.self_attn.absorbed = absorbed
            cache = caches[i] if caches is not None else None
            h = layer(h, pos, use_fast=use_fast, cache=cache)
            if cache is not None:
                mx.eval(h, cache.c_kv, cache.k_pe)
            else:
                mx.eval(h)
        h = mx.fast.rms_norm(h, self.norm_w, self.cfg.rms_norm_eps)
        return self.lm_head(h)
