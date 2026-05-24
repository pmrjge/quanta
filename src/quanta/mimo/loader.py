"""Streamed, layer-by-layer loader for the MiMo-V2.5 block-fp8 source checkpoint (pure MLX).

Reads tensors from the sharded ``safetensors`` (``model_pp0_ep*_shard*.safetensors``) via
``mx.load`` (lazy / mmap), one text layer resident at a time (rule-8). MiMo's matmul weights are
DeepSeek-style **block-fp8** (``F8_E4M3``, which ``mx.load`` returns as ``uint8``) paired with a
sibling ``*.weight_scale_inv``; :meth:`read_dequant` detects that pairing and dequantizes via
:mod:`quanta.mimo.fp8`. Norms, router (``gate.weight`` / ``e_score_correction_bias``), attention
sinks, ``o_proj`` and embeddings ship bf16/f32 and are read through unchanged.

Two source traps are handled here (see :class:`quanta.mimo.config.MiMoV2Config`):

* **Fused qkv.** ``self_attn.qkv_proj`` is one fused ``[Q|K|V]`` tensor; :meth:`attention_tensors`
  dequantizes then splits at the exact per-layer-type offsets (``cfg.qkv_sizes``) and asserts the
  split sums to the stored ``out`` — never a uniform chunk (vLLM #42803).
* **Block-fp8 padding.** The full-attn qkv scale grid has trailing padding rows; the dequant slice
  drops them (gated in ``parity/mimo_fp8_dequant_test.py``).

``shape``/``has`` are cheap wiring checks; ``read``/``read_dequant`` materialize one tensor;
``release`` drops shard handles so a finished layer can be freed.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.fp8 import dequant_block_fp8


class MiMoSourceCheckpoint:
    """Lazy, streamed reader over the sharded MiMo-V2.5 block-fp8 source checkpoint."""

    def __init__(self, model_dir: str | Path, cfg: MiMoV2Config | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg or MiMoV2Config.from_pretrained(model_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._shards: dict[str, dict[str, mx.array]] = {}

    # --- raw access ------------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self.weight_map

    def _lazy(self, key: str) -> mx.array:
        if key not in self.weight_map:
            raise KeyError(f"tensor not in source index: {key}")
        fn = self.weight_map[key]
        shard = self._shards.get(fn)
        if shard is None:
            shard = mx.load(str(self.dir / fn))  # lazy / mmap
            self._shards[fn] = shard
        return shard[key]

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._lazy(key).shape)

    def read(self, key: str) -> mx.array:
        """Materialize a tensor verbatim (bf16/f32 passthrough)."""
        arr = self._lazy(key)
        mx.eval(arr)
        return arr

    def read_dequant(self, key: str, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        """Materialize a weight, dequantizing if it is block-fp8 (has a ``*.weight_scale_inv``)."""
        scale_key = key + "_scale_inv"
        if self.has(scale_key):
            out = dequant_block_fp8(self._lazy(key), self._lazy(scale_key), dtype=dtype)
            mx.eval(out)
            return out
        return self.read(key)

    def release(self) -> None:
        self._shards.clear()

    # --- key helpers -----------------------------------------------------------
    def _p(self, i: int) -> str:
        return f"model.layers.{i}."

    def expert_key(self, i: int, e: int, proj: str) -> str:
        return f"{self._p(i)}mlp.experts.{e}.{proj}.weight"

    # --- per-layer text loaders (materialized; one layer resident) -------------
    def norm_tensors(self, i: int) -> dict[str, mx.array]:
        p = self._p(i)
        return {
            "input_layernorm": self.read(p + "input_layernorm.weight"),
            "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight"),
        }

    def attention_tensors(self, i: int) -> dict[str, mx.array]:
        """Dequant fused qkv, split into Q/K/V at exact per-layer-type offsets; read o_proj + sink."""
        p = self._p(i)
        swa = self.cfg.is_swa(i)
        q_size, k_size, v_size = self.cfg.qkv_sizes(swa)
        qkv = self.read_dequant(p + "self_attn.qkv_proj.weight")  # [q+k+v, hidden] bf16
        if qkv.shape[0] != q_size + k_size + v_size:
            raise ValueError(f"L{i} fused qkv out={qkv.shape[0]} != q+k+v="
                             f"{q_size + k_size + v_size} (swa={swa}); refusing to split (vLLM #42803)")
        out = {
            "q_proj": qkv[:q_size],
            "k_proj": qkv[q_size:q_size + k_size],
            "v_proj": qkv[q_size + k_size:],
            "o_proj": self.read_dequant(p + "self_attn.o_proj.weight"),
        }
        sink = p + "self_attn.attention_sink_bias"
        if self.cfg.has_attn_sink(swa):
            if not self.has(sink):
                raise ValueError(f"L{i} expects attention_sink_bias (swa={swa}) but it is missing")
            out["attention_sink_bias"] = self.read(sink)
        mx.eval(list(out.values()))
        return out

    def dense_mlp_tensors(self, i: int) -> dict[str, mx.array]:
        p = self._p(i) + "mlp."
        out = {proj: self.read_dequant(p + f"{proj}.weight") for proj in ("gate_proj", "up_proj", "down_proj")}
        mx.eval(list(out.values()))
        return out

    def moe_router_tensors(self, i: int) -> dict[str, mx.array]:
        p = self._p(i) + "mlp.gate."
        out = {"weight": self.read(p + "weight")}                      # [n_experts, hidden] f32
        if self.has(p + "e_score_correction_bias"):
            out["e_score_correction_bias"] = self.read(p + "e_score_correction_bias")
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, i: int, n_experts: int | None = None) -> dict[str, mx.array]:
        """Dequant routed experts into ``[E, out, in]`` bf16 stacks (gate/up/down).

        Streamed per expert (dequant -> place -> drop shard handles every 16) so only the bf16
        stacks stay resident, never the whole shard set. ``n_experts`` defaults to the full count.
        """
        ne = n_experts if n_experts is not None else self.cfg.n_routed_experts
        first = {proj: self.read_dequant(self.expert_key(i, 0, proj))
                 for proj in ("gate_proj", "up_proj", "down_proj")}
        stacks = {proj: mx.zeros((ne, *first[proj].shape), first[proj].dtype) for proj in first}
        for proj in first:
            stacks[proj][0] = first[proj]
        for e in range(1, ne):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                stacks[proj][e] = self.read_dequant(self.expert_key(i, e, proj))
            if e % 16 == 15:
                mx.eval(list(stacks.values()))
                self.release()
        mx.eval(list(stacks.values()))
        self.release()
        return stacks
