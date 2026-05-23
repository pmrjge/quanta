"""Streamed, layer-by-layer source-checkpoint loader (pure MLX, offline).

Reads tensors from the sharded ``safetensors`` source via ``mx.load`` (lazy /
memory-mapped). Two access patterns, both avoiding full-model / full-tensor
residency:

* :meth:`SourceCheckpoint.read` — materialize one named tensor (a layer's weight
  must be resident to run; one layer at a time honors the memory discipline).
* :meth:`SourceCheckpoint.gather_rows` — streamed sliced read (e.g. embedding
  rows) done on the CPU stream so only the touched rows are paged in, never the
  full ``[vocab, hidden]`` tensor.

``mx.load`` returns a lazy dict instantly even for a 9.8 GiB shard; data is read
only when an array is evaluated, so opening a shard to pull a few tensors is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.compressed_int4 import dequantize_packed_int4

TEXT_PREFIX = "language_model.model."

# Attention + norm suffixes are shared by dense (L0) and MoE (L1+) layers.
ATTENTION_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
)
DENSE_MLP_SUFFIXES: tuple[str, ...] = (
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
DENSE_LAYER_SUFFIXES: tuple[str, ...] = ATTENTION_SUFFIXES + DENSE_MLP_SUFFIXES

# MoE non-expert suffixes: router (bf16 weight + fp32 correction bias) + shared expert (bf16).
MOE_ROUTER_SUFFIXES: tuple[str, ...] = (
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
)
SHARED_EXPERT_SUFFIXES: tuple[str, ...] = (
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)
_EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")


class SourceCheckpoint:
    """Lazy, streamed reader over the sharded source checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    def _shard(self, key: str) -> dict[str, mx.array]:
        if key not in self.weight_map:
            raise KeyError(f"tensor not in source index: {key}")
        fn = self.weight_map[key]
        shard = self._shards.get(fn)
        if shard is None:
            shard = mx.load(str(self.dir / fn))  # lazy / mmap
            self._shards[fn] = shard
        return shard

    def read(self, key: str) -> mx.array:
        """Materialize one full tensor (resident). Use for weights you must run."""
        arr = self._shard(key)[key]
        mx.eval(arr)
        return arr

    def gather_rows(self, key: str, ids: mx.array) -> mx.array:
        """Streamed sliced read of selected rows (CPU stream → only touched pages)."""
        arr = self._shard(key)[key]
        with mx.stream(mx.cpu):
            rows = arr[ids]
            mx.eval(rows)
        return rows

    def release(self) -> None:
        """Drop cached shard handles so materialized tensors can be freed."""
        self._shards.clear()

    # --- convenience -----------------------------------------------------------
    def layer_key(self, layer_idx: int, suffix: str) -> str:
        return f"{TEXT_PREFIX}layers.{layer_idx}.{suffix}"

    def load_dense_layer(self, layer_idx: int) -> dict[str, mx.array]:
        """Load one dense decoder layer's tensors (keyed by suffix), all resident."""
        out = {suf: self.read(self.layer_key(layer_idx, suf)) for suf in DENSE_LAYER_SUFFIXES}
        mx.eval(list(out.values()))
        return out

    def embed_tokens(self, ids: mx.array) -> mx.array:
        """Streamed embedding lookup for the given token ids → ``[len(ids), hidden]``."""
        return self.gather_rows(f"{TEXT_PREFIX}embed_tokens.weight", ids)

    def load_moe_nonexpert(self, layer_idx: int) -> dict[str, mx.array]:
        """Attention + norms + router + shared expert for an MoE layer (no routed experts)."""
        suffixes = ATTENTION_SUFFIXES + MOE_ROUTER_SUFFIXES + SHARED_EXPERT_SUFFIXES
        out = {suf: self.read(self.layer_key(layer_idx, suf)) for suf in suffixes}
        mx.eval(list(out.values()))
        return out

    def load_expert_stacks(
        self,
        layer_idx: int,
        n_experts: int,
        moe_intermediate: int,
        hidden: int,
        *,
        group_size: int = 32,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> dict[str, mx.array]:
        """Dequantize all routed experts into stacked weights ``[E, out, in]`` (bf16).

        Streamed per expert (dequant → place → drop the packed shard handle) so only
        the growing bf16 stacks stay resident, not the whole shard set.
        """
        gate = mx.zeros((n_experts, moe_intermediate, hidden), dtype)
        up = mx.zeros((n_experts, moe_intermediate, hidden), dtype)
        down = mx.zeros((n_experts, hidden, moe_intermediate), dtype)
        dims = {
            "gate_proj": (gate, moe_intermediate, hidden),
            "up_proj": (up, moe_intermediate, hidden),
            "down_proj": (down, hidden, moe_intermediate),
        }
        for e in range(n_experts):
            base = self.layer_key(layer_idx, f"mlp.experts.{e}.")
            for proj in _EXPERT_PROJS:
                stack, out_f, in_f = dims[proj]
                packed = self.read(base + proj + ".weight_packed")
                scale = self.read(base + proj + ".weight_scale")
                stack[e] = dequantize_packed_int4(packed, scale, out_f, in_f, group_size, dtype)
            if e % 16 == 15:
                mx.eval(gate, up, down)
                self.release()  # drop materialized packed tensors held by shard caches
        mx.eval(gate, up, down)
        self.release()
        return {"gate": gate, "up": up, "down": down}
