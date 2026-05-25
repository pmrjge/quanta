"""Qwen3.5-397B-A17B hyperparameters (``model_type="qwen3_5_moe"``,
``Qwen3_5MoeForConditionalGeneration``).

Parsed from the source ``config.json`` (its nested ``text_config``) into a frozen dataclass. Pure
Python + ``json`` — no ``torch``/``transformers``/``mlx`` import. Grounded in the on-disk checkpoint
at ``~/models/Qwen3.5-397B-A17B`` (``config.json`` + ``generation_config.json`` +
``tokenizer_config.json`` + the safetensors weight-key index/shapes), NOT assumed from a sibling
model — per the project's "verify each model's formats empirically" rule.

Architecture (empirically confirmed from ``config.json`` + ``model.safetensors.index.json`` shapes):

* **60 decoder layers, 3:1 HYBRID attention** (``full_attention_interval=4``): 45
  ``linear_attention`` layers + 15 ``full_attention`` layers (full at 3,7,…,59). ``layer_types``
  (len 60) is the authoritative per-layer schedule; we refuse to load if its length != n_layers.
* **Linear attention = Gated DeltaNet** (Qwen3-Next style, O(1) recurrent state — the 1M enabler):
  ``in_proj_qkv`` 4096->12288 (q 2048 + k 2048 = 16 key-heads × 128; v 8192 = 64 value-heads × 128),
  ``in_proj_a``/``in_proj_b`` 4096->64 (per-value-head decay + write-strength gates),
  ``in_proj_z`` 4096->8192 (output gate), ``conv1d`` depthwise over all 12288 channels (kernel 4),
  ``A_log``/``dt_bias`` [64] fp32 SSM decay, ``norm`` [128] fp32 gated per-head RMSNorm,
  ``out_proj`` 8192->4096. ``mamba_ssm_dtype="float32"`` (state kept in fp32).
* **Full attention = gated GQA** (Qwen3-Next style): ``q_proj`` 4096->16384 = 32 heads × 128... no —
  32 heads × 256 head_dim × **2** (query + fused per-head output gate, ``attn_output_gate``);
  ``k_proj``/``v_proj`` 4096->512 = 2 KV-heads × 256 (GQA 16:1); ``o_proj`` 8192->4096; per-head
  ``q_norm``/``k_norm`` [256] RMSNorm. **Partial mRoPE**: only ``partial_rotary_factor=0.25`` of each
  256-dim head (= 64 dims) is rotated, interleaved multimodal sections [11,11,10] (sum 32 = 64/2),
  ``rope_theta=1e7``.
* **MoE on all 60 layers**: 512 routed experts (pre-stacked ``experts.gate_up_proj`` [512,2048,4096]
  + ``experts.down_proj`` [512,4096,1024] — gather_qmm-ready), **top-10**, per-expert FFN width 1024;
  **softmax** routing (only ``router_aux_loss_coef`` is present — NOT a DeepSeek noaux_tc sigmoid+bias
  scheme) with ``norm_topk_prob`` re-normalization. **1 shared expert** (width 1024) gated by a
  sigmoid scalar ``shared_expert_gate`` [1,4096].
* **Native MTP**: ``mtp_num_hidden_layers=1`` — ``mtp.fc`` fuses norm(embed)+norm(hidden) [8192->4096]
  into one full-attn+MoE block; weights ARE present (``mtp.*``), so lossless self-spec decode
  (``qwen3_next_mtp``) is in scope. ``mtp_use_dedicated_embeddings=false`` (reuses the main embed).
* **Source**: bf16 (no source quantization) — full quant headroom for the bake (int4 experts).
* **Multimodal**: a ViT under ``model.visual.*`` (deferred). Text decoder under
  ``model.language_model.*``; ``embed_tokens``/``norm`` are namespaced there, ``lm_head`` at top level.
* **Tokens**: **no BOS** (``add_bos_token=false``, ``bos_token=null``); chat ``<|im_start|>``/
  ``<|im_end|>`` with ``<think>`` reasoning on by default; generation stop set {248046 ``<|im_end|>``,
  248044 ``<|endoftext|>``}; pad 248044; untied ``lm_head``.

**Long-context policy (quanta, NOT in source config):** the artifact is baked for 1,010,000 tokens
via *dynamic* YaRN — ``yarn_factor=4`` over ``yarn_original_max=262144`` applied only when a sequence
exceeds the native window (``effective_yarn_factor``). Fields default to that policy; the long-context
ppl/needle gate is the arbiter that the scaling is correct.

Fields not present in ``config.json`` (MoE top-k normalization / scoring details) are read with
documented defaults and **must be confirmed against the reference forward / torch oracle** in the
MoE-parity stage — they are not structural invariants, so ``from_pretrained`` does not refuse on them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Qwen35Config:
    """Hyperparameters of the Qwen3.5-397B-A17B text decoder (+ 1 native MTP module)."""

    # core
    vocab_size: int                  # 248320
    hidden_size: int                 # 4096
    num_hidden_layers: int           # 60

    # hybrid attention schedule (authoritative per-layer typing)
    layer_types: tuple[str, ...]     # len == n_layers; "linear_attention" | "full_attention"
    full_attention_interval: int     # 4 (every 4th layer is full)

    # full attention (gated GQA + partial mRoPE + per-head QK-norm)
    num_attention_heads: int         # 32 query heads
    num_key_value_heads: int         # 2 KV heads (GQA 16:1)
    head_dim: int                    # 256
    attn_output_gate: bool           # q_proj emits query AND a fused per-head output gate (2x width)
    partial_rotary_factor: float     # 0.25 — fraction of each head rotated by RoPE
    rope_theta: float                # 1e7
    mrope_section: tuple[int, ...]   # (11, 11, 10) — interleaved multimodal RoPE sections
    mrope_interleaved: bool          # True
    use_qk_norm: bool                # per-head RMSNorm q_norm/k_norm before RoPE

    # linear attention (Gated DeltaNet)
    linear_num_key_heads: int        # 16
    linear_num_value_heads: int      # 64
    linear_key_head_dim: int         # 128
    linear_value_head_dim: int       # 128
    linear_conv_kernel_dim: int      # 4 (causal depthwise conv over the qkv stream)
    mamba_ssm_dtype: str             # "float32" — recurrent state precision

    # MoE (512 experts top-10 softmax, + 1 shared expert)
    num_experts: int                 # 512
    num_experts_per_tok: int         # top-10
    moe_intermediate_size: int       # per-expert FFN width (1024)
    shared_expert_intermediate_size: int  # 1024 (>0 => has a shared expert)
    scoring_func: str                # "softmax" — confirm at oracle stage
    norm_topk_prob: bool             # re-normalize the top-k routing weights — confirm at oracle
    router_aux_loss_coef: float      # 0.001

    # native multi-token prediction (weights present)
    num_mtp_modules: int             # mtp_num_hidden_layers (1)
    mtp_use_dedicated_embeddings: bool  # False (reuses the main token embedding)

    # norm / activation / positions / tokens
    hidden_act: str                  # "silu"
    norm_eps: float                  # rms_norm_eps (1e-6)
    max_position_embeddings: int     # native window (262144)
    eos_token_id: int                # primary generation eos (248046 = <|im_end|>)
    eos_token_ids: tuple[int, ...]   # full stop set (248046, 248044)
    pad_token_id: int                # 248044 (<|endoftext|>)
    tie_word_embeddings: bool        # False

    # --- defaulted: tokens + long-context policy + deferred-stage payloads -----
    bos_token_id: int | None = None  # a bos id exists (248044) but is NOT prepended
    add_bos_token: bool = False      # tokenizer does not add BOS

    # quanta long-context policy (dynamic YaRN to 1M) — NOT from the source config
    max_context: int = 1_010_000     # baked target context (extensible 1M)
    yarn_factor: float = 4.0         # official YaRN scaling factor over the native window
    yarn_original_max: int = 262_144  # baseline below which NO scaling is applied
    yarn_dynamic: bool = True        # length-adaptive: scale only when seq_len > yarn_original_max

    # source quantization (bf16 source => empty) + deferred vision tower config
    quantization_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"layer_types length {len(self.layer_types)} != num_hidden_layers "
                             f"{self.num_hidden_layers} (cannot type layers; refusing to guess)")
        bad = set(self.layer_types) - {"linear_attention", "full_attention"}
        if bad:
            raise ValueError(f"unknown layer_types {sorted(bad)} (expected linear_attention/"
                             f"full_attention)")
        # interleaved mRoPE sections must tile the rotary half-dim
        if self.mrope_section and sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)} "
                             f"!= rotary_dim//2 ({self.rotary_dim // 2})")

    # --- derived geometry: full attention --------------------------------------
    @property
    def q_dim(self) -> int:
        """Query dim = ``num_attention_heads * head_dim`` (8192) — the query half of q_proj."""
        return self.num_attention_heads * self.head_dim

    @property
    def q_proj_out(self) -> int:
        """``q_proj`` output width: ``2 * q_dim`` (16384) when ``attn_output_gate`` else ``q_dim``."""
        return self.q_dim * (2 if self.attn_output_gate else 1)

    @property
    def kv_dim(self) -> int:
        """Key/value projection dim = ``num_key_value_heads * head_dim`` (512)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """GQA repeat factor = ``num_attention_heads // num_key_value_heads`` (16)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def rotary_dim(self) -> int:
        """Rotated dims per head = ``round(partial_rotary_factor * head_dim)`` (64)."""
        return int(round(self.partial_rotary_factor * self.head_dim))

    # --- derived geometry: linear (Gated DeltaNet) attention -------------------
    @property
    def linear_qkv_dim(self) -> int:
        """``in_proj_qkv`` width = q + k (key heads) + v (value heads) = 2048+2048+8192 (12288).
        Also the depthwise ``conv1d`` channel count."""
        return (2 * self.linear_num_key_heads * self.linear_key_head_dim
                + self.linear_num_value_heads * self.linear_value_head_dim)

    @property
    def linear_k_dim(self) -> int:
        """Query/key projection width each = ``linear_num_key_heads * linear_key_head_dim`` (2048)."""
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_v_dim(self) -> int:
        """Value / output-gate width = ``linear_num_value_heads * linear_value_head_dim`` (8192)."""
        return self.linear_num_value_heads * self.linear_value_head_dim

    # --- derived: MoE ----------------------------------------------------------
    @property
    def moe_gate_up_out(self) -> int:
        """Stacked ``experts.gate_up_proj`` output width = ``2 * moe_intermediate_size`` (2048)."""
        return 2 * self.moe_intermediate_size

    @property
    def has_shared_expert(self) -> bool:
        return self.shared_expert_intermediate_size > 0

    # --- per-layer typing ------------------------------------------------------
    def _layer_type(self, layer_id: int) -> str:
        if not 0 <= layer_id < len(self.layer_types):
            raise IndexError(f"layer_id {layer_id} out of range for layer_types "
                             f"(len {len(self.layer_types)})")
        return self.layer_types[layer_id]

    def is_full_attention(self, layer_id: int) -> bool:
        return self._layer_type(layer_id) == "full_attention"

    def is_linear_attention(self, layer_id: int) -> bool:
        return self._layer_type(layer_id) == "linear_attention"

    # --- long-context (dynamic YaRN) policy ------------------------------------
    def effective_yarn_factor(self, seq_len: int) -> float:
        """RoPE scaling factor to use for a sequence of ``seq_len`` tokens under the baked policy.

        Static policy (``yarn_dynamic=False``): always ``yarn_factor`` (matches vLLM/sglang, but the
        upstream README warns it degrades short context). Dynamic policy (default): ``1.0`` while the
        sequence fits the native window, otherwise scale up only as far as needed
        (``seq_len / yarn_original_max``), capped at ``yarn_factor`` — so short prompts pay no tax and
        long jobs reach ``yarn_original_max * yarn_factor`` (≈1.05M ≥ the 1.01M target)."""
        if not self.yarn_dynamic:
            return self.yarn_factor
        if seq_len <= self.yarn_original_max:
            return 1.0
        return min(self.yarn_factor, seq_len / self.yarn_original_max)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen35Config:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        tc = cfg.get("text_config", cfg)  # text decoder hyperparams are nested for the MM wrapper
        gen: dict = {}
        gp = d / "generation_config.json"
        if gp.exists():
            gen = json.loads(gp.read_text())
        tok: dict = {}
        tp = d / "tokenizer_config.json"
        if tp.exists():
            tok = json.loads(tp.read_text())

        n_layers = int(tc["num_hidden_layers"])
        interval = int(tc.get("full_attention_interval", 4))
        # layer_types is authoritative; synthesize from the interval ONLY if the key is absent.
        if "layer_types" in tc:
            layer_types = tuple(str(x) for x in tc["layer_types"])
        else:
            layer_types = tuple("full_attention" if (i + 1) % interval == 0 else "linear_attention"
                                for i in range(n_layers))

        rope = tc.get("rope_parameters", tc.get("rope_scaling", {})) or {}
        head_dim = int(tc["head_dim"])
        partial = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0)))
        mrope_section = tuple(int(x) for x in rope.get("mrope_section", []))

        eos = gen.get("eos_token_id", tc.get("eos_token_id"))
        eos_ids = tuple(int(x) for x in (eos if isinstance(eos, list) else [eos] if eos is not None
                                         else []))
        native_max = int(tc.get("max_position_embeddings", 262_144))

        return cls(
            vocab_size=int(tc["vocab_size"]),
            hidden_size=int(tc["hidden_size"]),
            num_hidden_layers=n_layers,
            layer_types=layer_types,
            full_attention_interval=interval,
            num_attention_heads=int(tc["num_attention_heads"]),
            num_key_value_heads=int(tc["num_key_value_heads"]),
            head_dim=head_dim,
            attn_output_gate=bool(tc.get("attn_output_gate", False)),
            partial_rotary_factor=partial,
            rope_theta=float(rope.get("rope_theta", tc.get("rope_theta", 1e7))),
            mrope_section=mrope_section,
            mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
            use_qk_norm=bool(tc.get("use_qk_norm", True)),
            linear_num_key_heads=int(tc["linear_num_key_heads"]),
            linear_num_value_heads=int(tc["linear_num_value_heads"]),
            linear_key_head_dim=int(tc["linear_key_head_dim"]),
            linear_value_head_dim=int(tc["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(tc["linear_conv_kernel_dim"]),
            mamba_ssm_dtype=str(tc.get("mamba_ssm_dtype", "float32")),
            num_experts=int(tc["num_experts"]),
            num_experts_per_tok=int(tc["num_experts_per_tok"]),
            moe_intermediate_size=int(tc["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(tc.get("shared_expert_intermediate_size", 0)),
            scoring_func=str(tc.get("scoring_func", "softmax")),
            norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
            router_aux_loss_coef=float(tc.get("router_aux_loss_coef", 0.0)),
            num_mtp_modules=int(tc.get("mtp_num_hidden_layers", 0)),
            mtp_use_dedicated_embeddings=bool(tc.get("mtp_use_dedicated_embeddings", False)),
            hidden_act=str(tc.get("hidden_act", "silu")),
            norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            max_position_embeddings=native_max,
            eos_token_id=int(eos_ids[0]) if eos_ids else 248046,
            eos_token_ids=eos_ids or (248046, 248044),
            pad_token_id=int(gen.get("pad_token_id", tc.get("pad_token_id", 248044))),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings",
                                                                          False))),
            bos_token_id=(int(gen["bos_token_id"]) if gen.get("bos_token_id") is not None else None),
            add_bos_token=bool(tok.get("add_bos_token", False)),
            yarn_original_max=native_max,  # scale relative to the native window
            quantization_config=dict(cfg.get("quantization_config") or {}),
            vision_config=dict(cfg.get("vision_config") or {}),
        )
