"""Full Kimi-K2.6 text model — streamed, one decoder layer resident at a time.

Assembles embed → 61 decoder layers (L0 dense, L1..L60 MoE) → final RMSNorm →
lm_head. Each layer's weights are loaded, run, and released before the next, so
peak residency is ~one layer (the memory discipline). Experts are dequantized to
bf16 here (forward-path correctness); the resident quantized runtime comes after
the bake.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from quanta.cache import MLACache
from quanta.config import KimiTextConfig
from quanta.loader import TEXT_PREFIX, SourceCheckpoint
from quanta.modeling.decoder import DenseDecoderLayer, MoEDecoderLayer

LM_HEAD_KEY = "language_model.lm_head.weight"
FINAL_NORM_KEY = TEXT_PREFIX + "norm.weight"


def load_module_weights(module, weights: dict[str, mx.array]) -> None:
    """Assign weights by exact param-name match; raise on any key with no param."""
    names = {k for k, _ in tree_flatten(module.parameters())}
    for k in weights:
        if k not in names:
            raise KeyError(f"weight key {k!r} has no matching param in {type(module).__name__}")
    module.load_weights([(k, v) for k, v in weights.items()], strict=False)


def load_layer_raw(
    ckpt: SourceCheckpoint, cfg: KimiTextConfig, layer_idx: int, dtype: mx.Dtype
) -> dict:
    """Stream one layer's source tensors (dense or MoE) cast to ``dtype``."""
    if cfg.is_dense_layer(layer_idx):
        w = {k: v.astype(dtype) for k, v in ckpt.load_dense_layer(layer_idx).items()}
        return {"kind": "dense", "weights": w}
    ne = {k: v.astype(dtype) for k, v in ckpt.load_moe_nonexpert(layer_idx).items()}
    experts = ckpt.load_expert_stacks(
        layer_idx, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size, dtype=dtype
    )
    return {"kind": "moe", "weights": ne, "experts": experts}


def build_runtime_layer(cfg: KimiTextConfig, raw: dict):
    """Instantiate the runtime decoder module for a streamed layer."""
    if raw["kind"] == "dense":
        layer = DenseDecoderLayer(cfg)
        load_module_weights(layer, raw["weights"])
        return layer
    layer = MoEDecoderLayer(cfg)
    load_module_weights(layer, raw["weights"])
    layer.mlp.set_experts(raw["experts"]["gate"], raw["experts"]["up"], raw["experts"]["down"])
    return layer


class KimiModel:
    def __init__(self, cfg: KimiTextConfig, ckpt: SourceCheckpoint, dtype: mx.Dtype = mx.bfloat16):
        self.cfg = cfg
        self.ckpt = ckpt
        self.dtype = dtype

    def __call__(
        self,
        token_ids: mx.array,
        *,
        n_layers: int | None = None,
        use_fast: bool = False,
        caches: list | None = None,
        offset: int = 0,
    ) -> mx.array:
        """Forward (logits) for ``token_ids``. With ``caches`` (one MLACache per layer)
        and ``offset`` (positions already cached) this is the incremental path used for
        chunked prefill / prefix reuse."""
        cfg = self.cfg
        n = cfg.num_hidden_layers if n_layers is None else n_layers
        h = self.ckpt.embed_tokens(token_ids)[None].astype(self.dtype)
        pos = mx.arange(offset, offset + h.shape[1])
        for i in range(n):
            raw = load_layer_raw(self.ckpt, cfg, i, self.dtype)
            layer = build_runtime_layer(cfg, raw)
            cache = caches[i] if caches is not None else None
            h = layer(h, pos, use_fast=use_fast, cache=cache)
            if cache is not None:
                mx.eval(h, cache.c_kv, cache.k_pe)  # materialize before layer weights are freed
            else:
                mx.eval(h)
            del layer, raw
            self.ckpt.release()
        norm_w = self.ckpt.read(FINAL_NORM_KEY).astype(self.dtype)
        h = mx.fast.rms_norm(h, norm_w, cfg.rms_norm_eps)
        lm_head = self.ckpt.read(LM_HEAD_KEY).astype(self.dtype)
        return h @ lm_head.T

    def build_prefix_cache(
        self, prefix_ids: mx.array, *, n_layers: int | None = None, use_fast: bool = False
    ) -> list[MLACache]:
        """Prefill a prefix and return its per-layer MLA caches (logits discarded).

        Reuse across requests by passing the returned caches to ``__call__`` with
        ``offset=caches[0].offset``; or persist them with ``cache.save_caches``."""
        n = self.cfg.num_hidden_layers if n_layers is None else n_layers
        caches = [MLACache() for _ in range(n)]
        self(prefix_ids, n_layers=n, caches=caches, use_fast=use_fast)
        mx.eval([c.c_kv for c in caches], [c.k_pe for c in caches])
        return caches

    def continue_from_cache(
        self, suffix_ids: mx.array, caches: list[MLACache], *, use_fast: bool = False
    ) -> mx.array:
        """Logits for ``suffix_ids`` appended after a cached prefix (skips its prefill)."""
        return self(
            suffix_ids, n_layers=len(caches), caches=caches, offset=caches[0].offset, use_fast=use_fast
        )
