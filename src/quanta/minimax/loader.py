"""Streamed, layer-by-layer fp8->bf16 source loader for the MiniMax-M2.7 checkpoint (pure MLX + numpy).

The 130 ``model-*.safetensors`` shards are **block-fp8**: ``q/k/v/o`` and the routed experts
``w1/w2/w3`` are ``F8_E4M3`` ``[out,in]`` each paired with a sibling ``.weight_scale_inv`` ``F32``
``[ceil(out/128),ceil(in/128)]`` block-scale grid (one fp32 scale per ``[128,128]`` weight block);
the router ``gate``, ``e_score_correction_bias``, ``lm_head``, ``embed_tokens`` and all norms stay
unquantized (``config.quant_modules_to_skip`` / ``modules_to_not_convert``) so they carry **no**
``.weight_scale_inv``. Dequant is ``fp8.astype(bf16) * block_expand(scale_inv)``.

Two differences from :mod:`quanta.dsv4.loader` drive this being a separate reader:

* MiniMax block scales are plain **fp32**, not the OCP **e8m0** power-of-two of DSV4 — they are read
  verbatim and multiplied in, so only the **weight** needs LUT decode (reused from
  :func:`quanta.dsv4.fp.e4m3_to_float`, the e4m3fn-correct table: NaN at ``0x7f``/``0xff``, max 448 —
  matching this checkpoint's ``fmt="float8_e4m3fn"``). ``mx.load`` cannot read ``F8_E4M3`` shards, so
  (as in DSV4) each shard is **mmapped** and the requested byte range is decoded in MLX.
* There is **no shared expert** and **no MTP** in this checkpoint (verified against the index): the
  accessor surface is ``embed`` / ``lm_head`` / ``final_norm`` / ``block_norms`` / ``attention`` /
  ``moe`` only. Mixtral expert naming: ``w1``=gate, ``w3``=up, ``w2``=down.

Per the project's memory-safety rules this keeps **one text layer resident at a time** (rule 8) and
**fails loud** on any missing key (rule 6). The actual heavy weight load is **deferred to a GPU
session** — this module never loads real tensors at import/test time; the model-free gate
(``parity/minimax_loader_test.py``) validates the dequant math on tiny tensors and the key/shape
schema from safetensors headers only.

Tensor key templates (empirically confirmed against ``~/models/MiniMax-M2.7`` —
``model.safetensors.index.json`` + per-shard headers):

* top-level: ``model.embed_tokens.weight`` (bf16), ``model.norm.weight`` (bf16), ``lm_head.weight``
  (bf16).
* ``model.layers.{i}.``: ``input_layernorm.weight`` / ``post_attention_layernorm.weight`` /
  ``self_attn.q_norm.weight`` / ``self_attn.k_norm.weight`` (bf16);
  ``self_attn.{q,k,v,o}_proj.weight`` (fp8 + ``.weight_scale_inv``);
  ``block_sparse_moe.gate.weight`` (f32) + ``block_sparse_moe.e_score_correction_bias`` (f32, no
  scale); ``block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`` (fp8 + ``.weight_scale_inv``).
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import mlx.core as mx

from quanta.dsv4 import fp
from quanta.minimax.config import MiniMaxConfig

EMBED_KEY = "model.embed_tokens.weight"
FINAL_NORM_KEY = "model.norm.weight"
LM_HEAD_KEY = "lm_head.weight"

_SCALE_SUFFIX = ".weight_scale_inv"


def dequant_block_fp8_f32(w_u8: mx.array, scale_f32: mx.array,
                          block: tuple[int, int] = (128, 128),
                          dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    """Block-fp8 dequant with an **fp32** block-scale grid (MiniMax flavour).

    ``w_u8``: e4m3fn bytes ``[out,in]`` (uint8). ``scale_f32``: fp32 grid
    ``[ceil(out/block_r), ceil(in/block_c)]`` — each ``[block_r, block_c]`` weight tile shares one
    scale. Returns ``w * block_expand(scale)`` in ``dtype``. The last row/col block may be partial;
    the expanded grid is clipped to ``[out,in]`` so partial tiles dequant correctly.
    """
    if w_u8.ndim != 2:
        raise ValueError(f"dequant_block_fp8_f32 expects a 2-D weight, got shape {tuple(w_u8.shape)}")
    if scale_f32.ndim != 2:
        raise ValueError(f"dequant_block_fp8_f32 expects a 2-D scale grid, got shape "
                         f"{tuple(scale_f32.shape)}")
    br, bc = int(block[0]), int(block[1])
    out, inn = int(w_u8.shape[0]), int(w_u8.shape[1])
    nbo, nbi = (out + br - 1) // br, (inn + bc - 1) // bc
    if int(scale_f32.shape[0]) < nbo or int(scale_f32.shape[1]) < nbi:
        raise ValueError(f"fp8 scale grid {tuple(scale_f32.shape)} too small for weight "
                         f"{(out, inn)} at block {(br, bc)} (need >= {(nbo, nbi)})")
    wf = fp.e4m3_to_float(w_u8)                       # [out, in] f32 (e4m3fn LUT)
    sf = scale_f32.astype(mx.float32)
    sf = mx.repeat(mx.repeat(sf, br, axis=0), bc, axis=1)[:out, :inn]
    return (wf * sf).astype(dtype)


class MiniMaxSourceCheckpoint:
    """Lazy, streamed, mmap-backed reader over the MiniMax-M2.7 block-fp8 source checkpoint.

    Accessors return **bf16** per-kind params for one layer at a time (dequanting fp8 keys via the
    fp32-scale block-fp8 primitive). The heavy real load is deferred to a GPU session; the model-free
    gate never instantiates this against real tensors.
    """

    def __init__(self, model_dir: str | Path, cfg: MiniMaxConfig | None = None) -> None:
        self.dir = Path(model_dir)
        self.cfg = cfg if cfg is not None else MiniMaxConfig.from_pretrained(self.dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._shards: dict[str, tuple] = {}   # fn -> (file, mmap, header, base_offset)

    @property
    def num_layers(self) -> int:
        return self.cfg.num_hidden_layers

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
        if key not in self.weight_map:
            raise KeyError(f"tensor not in source index: {key}")
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
        """Materialize a tensor verbatim in its native dtype (bf16/f32 passthrough; fails loud)."""
        a = self._raw(key)
        mx.eval(a)
        return a

    def read_dequant(self, weight_key: str, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        """Materialize a weight, dequantizing block-fp8 if a sibling ``.weight_scale_inv`` exists.

        A ``.weight`` with no sibling scale is an **unquantized** (bf16/f32) tensor — returned
        verbatim. A scaled weight whose stored dtype is not ``F8_E4M3`` is a schema violation and
        fails loud (rule 6); we never guess the wrong decode.
        """
        if not weight_key.endswith(".weight"):
            return self.read(weight_key)
        scale_key = weight_key[:-len(".weight")] + _SCALE_SUFFIX
        if not self.has(scale_key):
            return self.read(weight_key)
        wdt = self._meta(weight_key)["dtype"]
        if wdt != "F8_E4M3":
            raise ValueError(f"{weight_key}: has {_SCALE_SUFFIX} but unexpected weight dtype {wdt!r} "
                             f"(expected F8_E4M3)")
        w, s = self._raw(weight_key), self._raw(scale_key)
        out = dequant_block_fp8_f32(w, s, block=self.cfg.weight_block_size, dtype=dtype)
        mx.eval(out)
        return out

    def release(self) -> None:
        for f, mm, _, _ in self._shards.values():
            mm.close()
            f.close()
        self._shards.clear()

    # --- top-level tensors -----------------------------------------------------
    def embed(self) -> mx.array:
        return self.read(EMBED_KEY)

    def lm_head(self) -> mx.array:
        """Output projection (``tie_word_embeddings=False`` here, so a distinct ``lm_head.weight``)."""
        key = EMBED_KEY if self.cfg.tie_word_embeddings else LM_HEAD_KEY
        return self.read(key)

    def final_norm(self) -> mx.array:
        return self.read(FINAL_NORM_KEY)

    # --- per-layer helpers -----------------------------------------------------
    def _bp(self, i: int) -> str:
        return f"model.layers.{i}."

    def block_norms(self, i: int) -> dict[str, mx.array]:
        p = self._bp(i)
        return {
            "input_layernorm": self.read(p + "input_layernorm.weight"),
            "post_attention_layernorm": self.read(p + "post_attention_layernorm.weight"),
        }

    def attention(self, i: int) -> dict[str, mx.array]:
        """GQA q/k/v/o projections (fp8 -> bf16) + per-layer QK RMSNorm weights (bf16) for layer ``i``."""
        p = self._bp(i) + "self_attn."
        out = {
            "q_proj": self.read_dequant(p + "q_proj.weight"),
            "k_proj": self.read_dequant(p + "k_proj.weight"),
            "v_proj": self.read_dequant(p + "v_proj.weight"),
            "o_proj": self.read_dequant(p + "o_proj.weight"),
            "q_norm": self.read(p + "q_norm.weight"),
            "k_norm": self.read(p + "k_norm.weight"),
        }
        mx.eval(list(out.values()))
        return out

    # --- MoE -------------------------------------------------------------------
    def moe_router(self, i: int) -> dict[str, mx.array]:
        """Router: ``gate.weight`` (f32) + ``e_score_correction_bias`` (f32) — both unquantized."""
        p = self._bp(i) + "block_sparse_moe."
        out = {
            "weight": self.read(p + "gate.weight"),
            "e_score_correction_bias": self.read(p + "e_score_correction_bias"),
        }
        mx.eval(list(out.values()))
        return out

    def expert_stacks(self, i: int, n_experts: int | None = None) -> dict[str, mx.array]:
        """Dequant the routed fp8 experts of layer ``i`` into ``[E, out, in]`` bf16 stacks.

        ``w1`` (gate) / ``w3`` (up): ``[E, moe_inter, hidden]``; ``w2`` (down): ``[E, hidden,
        moe_inter]``. Streamed per expert (dequant -> place -> drop mmaps every 16) so only the bf16
        stacks stay resident (rule 8) — this single stack is the layer's largest working set.
        """
        ne = n_experts if n_experts is not None else self.cfg.num_local_experts

        def ek(e: int, proj: str) -> str:
            return f"{self._bp(i)}block_sparse_moe.experts.{e}.{proj}.weight"

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

    def moe(self, i: int) -> dict[str, mx.array]:
        """The full MoE block of layer ``i``: router (gate + bias) + the routed expert stacks.

        No shared expert in this checkpoint (``shared_intermediate_size == 0``); refuse to invent one.
        """
        if self.cfg.has_shared_expert:
            raise ValueError(f"L{i}: config reports a shared expert (shared_intermediate_size="
                             f"{self.cfg.shared_intermediate_size}) but this checkpoint has none")
        return {
            "router": self.moe_router(i),
            "experts": self.expert_stacks(i),
        }
