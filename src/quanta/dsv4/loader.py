"""Streamed, layer-by-layer loader for the DeepSeek-V4-Flash source checkpoint (pure MLX + numpy).

The 46 ``model-*.safetensors`` shards mix **F8_E4M3** (non-expert weights), **I8** (packed fp4
experts), **F8_E8M0** (all scales), **BF16** and **F32**. ``mx.load`` refuses ``F8_E8M0`` and aborts
the whole shard, so this reader **mmaps** each shard and decodes raw byte ranges via
:mod:`quanta.dsv4.fp` (one text layer resident at a time — rule-8). :meth:`read_dequant` dispatches
on the stored weight dtype: ``F8_E4M3`` -> block-fp8 dequant, ``I8`` -> packed-fp4 dequant; tensors
without a sibling ``.scale`` (norms, router gate, attention sinks, embeddings, HC params, hash
tables) pass through in their native dtype.

Tensor names follow the DeepSeek inference layout: ``embed/head/norm``, ``hc_head_*``,
``layers.N.{attn_norm,ffn_norm,hc_attn_*,hc_ffn_*}``, ``layers.N.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,
wo_a,wo_b,attn_sink}`` (+ ``compressor.*`` when the layer compresses, + ``indexer.*`` on ratio-4
layers), ``layers.N.ffn.{gate.weight, gate.bias|gate.tid2eid, experts.E.{w1,w2,w3}, shared_experts.*}``,
and ``mtp.0.*``.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import mlx.core as mx

from quanta.dsv4 import fp
from quanta.dsv4.config import DeepSeekV4Config


class DeepSeekV4SourceCheckpoint:
    """Lazy, streamed, mmap-backed reader over the DSV4-Flash fp8/fp4 source checkpoint."""

    def __init__(self, model_dir: str | Path, cfg: DeepSeekV4Config | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg or DeepSeekV4Config.from_pretrained(model_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._shards: dict[str, tuple] = {}   # fn -> (file, mmap, header, base_offset)

    # --- raw mmap access -------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self.weight_map

    def _shard(self, fn: str) -> tuple:
        s = self._shards.get(fn)
        if s is None:
            f = open(self.dir / fn, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = struct.unpack("<Q", mm[:8])[0]
            hdr = json.loads(mm[8:8 + n])
            hdr.pop("__metadata__", None)
            s = (f, mm, hdr, 8 + n)
            self._shards[fn] = s
        return s

    def _meta(self, key: str) -> dict:
        if key not in self.weight_map:
            raise KeyError(f"tensor not in source index: {key}")
        _, _, hdr, _ = self._shard(self.weight_map[key])
        return hdr[key]

    def _raw(self, key: str) -> mx.array:
        """Decode a tensor to its natural MLX dtype (zero-copy from mmap into numpy, then MLX)."""
        _, mm, hdr, base = self._shard(self.weight_map[key])
        m = hdr[key]
        b, e = m["data_offsets"]
        buf = memoryview(mm)[base + b:base + e]
        return fp.decode_buffer(m["dtype"], m["shape"], buf)

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._meta(key)["shape"])

    def dtype_str(self, key: str) -> str:
        return self._meta(key)["dtype"]

    def read(self, key: str) -> mx.array:
        """Materialize a tensor verbatim in its native dtype (bf16/f32/i64 passthrough)."""
        a = self._raw(key)
        mx.eval(a)
        return a

    def read_dequant(self, weight_key: str, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        """Materialize a weight, dequantizing fp8/fp4 if a sibling ``.scale`` exists."""
        if not weight_key.endswith(".weight"):
            return self.read(weight_key)
        scale_key = weight_key[:-len(".weight")] + ".scale"
        if not self.has(scale_key):
            return self.read(weight_key)
        wdt = self._meta(weight_key)["dtype"]
        w, s = self._raw(weight_key), self._raw(scale_key)
        if wdt == "F8_E4M3":
            out = fp.dequant_block_fp8(w, s, dtype=dtype)
        elif wdt == "I8":
            out = fp.dequant_group_fp4(w, s, dtype=dtype)
        else:
            raise ValueError(f"{weight_key}: has .scale but unexpected weight dtype {wdt!r}")
        mx.eval(out)
        return out

    def release(self) -> None:
        for f, mm, _, _ in self._shards.values():
            mm.close()
            f.close()
        self._shards.clear()

    # --- top-level tensors -----------------------------------------------------
    def embed(self) -> mx.array:
        return self.read("embed.weight")

    def head(self) -> mx.array:
        return self.read("head.weight")

    def final_norm(self) -> mx.array:
        return self.read("norm.weight")

    def final_hc(self) -> dict[str, mx.array]:
        return {k: self.read(f"hc_head_{k}") for k in ("fn", "base", "scale")}

    # --- per-block helpers -----------------------------------------------------
    def _bp(self, i: int) -> str:
        return f"layers.{i}."

    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i)
        return {"attn_norm": self.read(p + "attn_norm.weight"),
                "ffn_norm": self.read(p + "ffn_norm.weight")}

    def block_hc(self, i: int) -> dict[str, mx.array]:
        """HC mixing params for the attn + ffn sub-blocks (f32: fn [mix_hc, hc*dim], base, scale[3])."""
        p = self._bp(i)
        out = {}
        for which in ("attn", "ffn"):
            for k in ("fn", "base", "scale"):
                out[f"hc_{which}_{k}"] = self.read(p + f"hc_{which}_{k}")
        mx.eval(list(out.values()))
        return out

    def _compressor(self, prefix: str) -> dict[str, mx.array]:
        """ape (f32), norm (bf16), wkv/wgate (bf16) for a Compressor at ``prefix``."""
        out = {"ape": self.read(prefix + "ape"),
               "norm": self.read(prefix + "norm.weight"),
               "wkv": self.read_dequant(prefix + "wkv.weight"),
               "wgate": self.read_dequant(prefix + "wgate.weight")}
        mx.eval(list(out.values()))
        return out

    def indexer(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i) + "attn.indexer."
        out = {"wq_b": self.read_dequant(p + "wq_b.weight"),          # fp8 -> bf16
               "weights_proj": self.read(p + "weights_proj.weight"),  # bf16
               "compressor": self._compressor(p + "compressor.")}
        mx.eval([out["wq_b"], out["weights_proj"]])
        return out

    def attention(self, i: int) -> dict[str, mx.array]:
        """Low-rank q/kv + grouped-O attention tensors (+ compressor/indexer when present)."""
        p = self._bp(i) + "attn."
        out: dict = {
            "wq_a": self.read_dequant(p + "wq_a.weight"),
            "q_norm": self.read(p + "q_norm.weight"),
            "wq_b": self.read_dequant(p + "wq_b.weight"),
            "wkv": self.read_dequant(p + "wkv.weight"),
            "kv_norm": self.read(p + "kv_norm.weight"),
            "wo_a": self.read_dequant(p + "wo_a.weight"),             # fp8 in ckpt -> bf16
            "wo_b": self.read_dequant(p + "wo_b.weight"),
            "attn_sink": self.read(p + "attn_sink"),                  # f32 [n_heads]
        }
        if self.cfg.has_compressor(i):
            out["compressor"] = self._compressor(p + "compressor.")
        if self.cfg.has_indexer(i):
            out["indexer"] = self.indexer(i)
        return out

    # --- MoE -------------------------------------------------------------------
    def moe_router(self, i: int) -> dict[str, mx.array]:
        """Router: gate.weight (bf16) + hash table (i64) for hash layers, else score bias (f32)."""
        p = self._bp(i) + "ffn.gate."
        out = {"weight": self.read(p + "weight")}
        if self.cfg.is_hash(i):
            if not self.has(p + "tid2eid"):
                raise ValueError(f"L{i} is a hash layer but tid2eid is missing")
            out["tid2eid"] = self.read(p + "tid2eid")                 # [vocab, topk] i64
        else:
            if self.has(p + "bias"):
                out["bias"] = self.read(p + "bias")                   # [n_experts] f32
        return out

    def shared_expert(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i) + "ffn.shared_experts."
        out = {proj: self.read_dequant(p + f"{proj}.weight") for proj in ("w1", "w2", "w3")}
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, i: int, n_experts: int | None = None) -> dict[str, mx.array]:
        """Dequant routed fp4 experts into ``[E, out, in]`` bf16 stacks (w1 gate, w3 up, w2 down).

        Streamed per expert (dequant -> place -> drop mmaps every 16) so only the bf16 stacks stay
        resident. ``w1/w3``: ``[E, moe_inter, hidden]``; ``w2``: ``[E, hidden, moe_inter]``.
        """
        ne = n_experts if n_experts is not None else self.cfg.n_routed_experts

        def ek(e: int, proj: str) -> str:
            return f"{self._bp(i)}ffn.experts.{e}.{proj}.weight"

        first = {proj: self.read_dequant(ek(0, proj)) for proj in ("w1", "w2", "w3")}
        stacks = {proj: mx.zeros((ne, *first[proj].shape), first[proj].dtype) for proj in first}
        for proj in first:
            stacks[proj][0] = first[proj]
        for e in range(1, ne):
            for proj in ("w1", "w2", "w3"):
                stacks[proj][e] = self.read_dequant(ek(e, proj))
            if e % 16 == 15:
                mx.eval(list(stacks.values()))
                self.release()
        mx.eval(list(stacks.values()))
        self.release()
        return stacks

    # --- native MTP block ------------------------------------------------------
    def mtp(self, j: int = 0) -> dict[str, mx.array]:
        """MTP block tensors: projections/norms + the inherited Block (attn, ffn, norms, HC)."""
        p = f"mtp.{j}."
        out = {
            "e_proj": self.read_dequant(p + "e_proj.weight"),
            "h_proj": self.read_dequant(p + "h_proj.weight"),
            "enorm": self.read(p + "enorm.weight"),
            "hnorm": self.read(p + "hnorm.weight"),
            "norm": self.read(p + "norm.weight"),
        }
        for k in ("fn", "base", "scale"):
            out[f"hc_head_{k}"] = self.read(p + f"hc_head_{k}")
        mx.eval(list(out.values()))
        return out
