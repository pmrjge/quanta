"""MiMo-V2.5 hyperparameters (XiaomiMiMo/MiMo-V2.5, multimodal MoE, 1M context).

Parsed from the source ``config.json`` into a frozen dataclass. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import. MiMo-V2.5 (``model_type="mimo_v2"``) is a DeepSeek-V3-
style sparse-MoE text backbone (L0 dense, L1-47 MoE; 256 experts, top-8, sigmoid ``noaux_tc``,
**no shared expert**) with a **hybrid full/SWA attention** stack, **fused qkv**, **partial RoPE**,
per-head **attention-sink bias** on SWA layers, native **MTP** heads, plus a Qwen2.5-VL **vision**
tower and a **speech/audio** encoder. Source weights are DeepSeek-style **block-fp8** (e4m3,
[128,128]); see :mod:`quanta.mimo.fp8`.

Two source traps are encoded here so downstream code can't diverge on them (verify-empirically rule):

* **Fused-qkv split (vLLM #42803 class).** ``qkv_proj`` is one fused ``[Q|K|V]`` tensor with
  *asymmetric, per-layer-type* sizes (full-attn uses ``num_key_value_heads=4``; SWA uses ``8``).
  Splitting at the wrong offsets (or uniform chunking) fills K/V slots with Q values → token-loop
  garbage. :meth:`qkv_sizes` returns the exact ``(q,k,v)`` for each layer type.
* **Partial RoPE.** Only ``int(head_dim*partial_rotary_factor)`` leading dims of each head are
  rotated (192*0.334 = 64); the rest pass through. :meth:`rope_dim`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MiMoV2Config:
    """Hyperparameters of the MiMo-V2.5 text decoder (+ preserved multimodal sub-configs)."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int          # dense (L0) FFN width

    # per-layer routing/typing (length == num_hidden_layers)
    hybrid_layer_pattern: tuple[int, ...]   # 0 = full attention, 1 = sliding-window (SWA)
    moe_layer_freq: tuple[int, ...]         # 0 = dense MLP, 1 = MoE

    # attention — full-attention layers
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    rope_theta: float
    # attention — sliding-window (SWA) layers
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    swa_v_head_dim: int
    swa_rope_theta: float
    sliding_window: int
    # shared attention knobs
    partial_rotary_factor: float
    attention_value_scale: float    # V is scaled by this before SDPA (0.707)
    attention_bias: bool
    attention_projection_layout: str  # "fused_qkv" for V2.5
    add_full_attention_sink_bias: bool
    add_swa_attention_sink_bias: bool

    # MoE (DeepSeek-V3-style sigmoid noaux_tc; no shared expert in V2.5)
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int            # 0 for V2.5
    moe_intermediate_size: int
    scoring_func: str
    topk_method: str
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float     # null in config -> 1.0

    # norm / positions / tokens
    norm_eps: float
    max_position_embeddings: int     # 1_048_576 (1M) — operating default
    rope_scaling: dict
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    eos_token_ids: tuple[int, ...]   # generation stop-set (from generation_config.json)
    tie_word_embeddings: bool

    # multimodal token ids
    image_token_id: int
    video_token_id: int
    audio_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int

    # native multi-token prediction (DeepSeek-MTP style): model.mtp.layers.{0..mtp_num_layers-1}
    mtp_num_layers: int

    # source quantization (DeepSeek block-fp8) + preserved encoder sub-configs
    quantization_config: dict
    vision_config: dict = field(default_factory=dict)
    audio_config: dict = field(default_factory=dict)
    processor_config: dict = field(default_factory=dict)

    # --- per-layer typing -----------------------------------------------------
    def is_swa(self, i: int) -> bool:
        return bool(self.hybrid_layer_pattern[i])

    def is_moe(self, i: int) -> bool:
        return bool(self.moe_layer_freq[i])

    # --- per-layer-type attention geometry (the fused-qkv / partial-RoPE guards) ----------
    def attn_heads(self, swa: bool) -> int:
        return self.swa_num_attention_heads if swa else self.num_attention_heads

    def attn_kv_heads(self, swa: bool) -> int:
        return self.swa_num_key_value_heads if swa else self.num_key_value_heads

    def attn_head_dim(self, swa: bool) -> int:
        return self.swa_head_dim if swa else self.head_dim

    def attn_v_head_dim(self, swa: bool) -> int:
        return self.swa_v_head_dim if swa else self.v_head_dim

    def attn_rope_theta(self, swa: bool) -> float:
        return self.swa_rope_theta if swa else self.rope_theta

    def sliding_window_for(self, swa: bool) -> int | None:
        return self.sliding_window if swa else None

    def has_attn_sink(self, swa: bool) -> bool:
        return self.add_swa_attention_sink_bias if swa else self.add_full_attention_sink_bias

    def qkv_sizes(self, swa: bool) -> tuple[int, int, int]:
        """Exact fused-qkv split ``(q_size, k_size, v_size)`` for this layer type.

        full-attn -> (12288, 768, 512) [kv=4]; SWA -> (12288, 1536, 1024) [kv=8]. NEVER split a
        fused qkv tensor any other way (vLLM #42803: uniform chunking corrupts K/V)."""
        nh, nkv = self.attn_heads(swa), self.attn_kv_heads(swa)
        hd, vhd = self.attn_head_dim(swa), self.attn_v_head_dim(swa)
        return nh * hd, nkv * hd, nkv * vhd

    def o_in_features(self, swa: bool) -> int:
        """o_proj in_features = num_attention_heads * v_head_dim (heads expanded post-attention)."""
        return self.attn_heads(swa) * self.attn_v_head_dim(swa)

    def rope_dim(self, swa: bool) -> int:
        """Rotated dims per head = int(head_dim * partial_rotary_factor); must be even."""
        d = int(self.attn_head_dim(swa) * self.partial_rotary_factor)
        if d % 2 != 0:
            raise ValueError(f"MiMoV2 rope_dim must be even, got {d} "
                             f"(head_dim={self.attn_head_dim(swa)}, factor={self.partial_rotary_factor})")
        return d

    def attn_scale(self, swa: bool) -> float:
        return self.attn_head_dim(swa) ** -0.5

    @property
    def n_moe_layers(self) -> int:
        return sum(self.moe_layer_freq)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> MiMoV2Config:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        gen = {}
        gp = d / "generation_config.json"
        if gp.exists():
            gen = json.loads(gp.read_text())

        n_layers = int(cfg["num_hidden_layers"])
        hybrid = tuple(int(x) for x in cfg["hybrid_layer_pattern"])
        moe_freq = tuple(int(x) for x in cfg["moe_layer_freq"])
        if len(hybrid) != n_layers or len(moe_freq) != n_layers:
            raise ValueError("hybrid_layer_pattern / moe_layer_freq length != num_hidden_layers")

        eos = cfg.get("eos_token_id", 151645)
        eos_single = int(eos if isinstance(eos, int) else eos[0])
        gen_eos = gen.get("eos_token_id", eos)
        eos_ids = tuple(int(x) for x in (gen_eos if isinstance(gen_eos, list) else [gen_eos]))

        rsf = cfg.get("routed_scaling_factor", None)
        n_shared = cfg.get("n_shared_experts", None)

        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=n_layers,
            intermediate_size=int(cfg["intermediate_size"]),
            hybrid_layer_pattern=hybrid,
            moe_layer_freq=moe_freq,
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg["head_dim"]),
            v_head_dim=int(cfg.get("v_head_dim", cfg["head_dim"])),
            rope_theta=float(cfg["rope_theta"]),
            swa_num_attention_heads=int(cfg.get("swa_num_attention_heads", cfg["num_attention_heads"])),
            swa_num_key_value_heads=int(cfg.get("swa_num_key_value_heads", cfg["num_key_value_heads"])),
            swa_head_dim=int(cfg.get("swa_head_dim", cfg["head_dim"])),
            swa_v_head_dim=int(cfg.get("swa_v_head_dim", cfg.get("v_head_dim", cfg["head_dim"]))),
            swa_rope_theta=float(cfg.get("swa_rope_theta", cfg["rope_theta"])),
            sliding_window=int(cfg.get("sliding_window", cfg.get("sliding_window_size", 128))),
            partial_rotary_factor=float(cfg.get("partial_rotary_factor", 1.0)),
            attention_value_scale=float(cfg.get("attention_value_scale", 1.0)),
            attention_bias=bool(cfg.get("attention_bias", False)),
            attention_projection_layout=str(cfg.get("attention_projection_layout", "split")),
            add_full_attention_sink_bias=bool(cfg.get("add_full_attention_sink_bias", False)),
            add_swa_attention_sink_bias=bool(cfg.get("add_swa_attention_sink_bias", False)),
            n_routed_experts=int(cfg["n_routed_experts"]),
            num_experts_per_tok=int(cfg["num_experts_per_tok"]),
            n_shared_experts=int(n_shared) if n_shared else 0,
            moe_intermediate_size=int(cfg["moe_intermediate_size"]),
            scoring_func=str(cfg.get("scoring_func", "sigmoid")),
            topk_method=str(cfg.get("topk_method", "noaux_tc")),
            n_group=int(cfg.get("n_group", 1)),
            topk_group=int(cfg.get("topk_group", 1)),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            routed_scaling_factor=float(rsf) if rsf is not None else 1.0,
            norm_eps=float(cfg.get("layernorm_epsilon", cfg.get("rms_norm_eps", 1e-5))),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 1048576)),
            rope_scaling=dict(cfg.get("rope_scaling") or {}),
            bos_token_id=int(gen.get("bos_token_id", cfg.get("bos_token_id", 151643))),
            eos_token_id=eos_single,
            pad_token_id=int(cfg.get("pad_token_id", 151643)),
            eos_token_ids=eos_ids,
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            image_token_id=int(cfg.get("image_token_id", 151655)),
            video_token_id=int(cfg.get("video_token_id", 151656)),
            audio_token_id=int((cfg.get("processor_config") or {}).get("audio_token_id", 151669)),
            vision_start_token_id=int(cfg.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(cfg.get("vision_end_token_id", 151653)),
            mtp_num_layers=int(cfg.get("num_nextn_predict_layers", cfg.get("num_mtp_layers", 0))),
            quantization_config=dict(cfg.get("quantization_config") or {}),
            vision_config=dict(cfg.get("vision_config") or {}),
            audio_config=dict(cfg.get("audio_config") or {}),
            processor_config=dict(cfg.get("processor_config") or {}),
        )
