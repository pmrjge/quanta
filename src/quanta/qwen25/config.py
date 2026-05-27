"""Qwen2.5-14B-Instruct-1M hyperparameters (``model_type="qwen2"``, ``Qwen2ForCausalLM``).

Parsed from the source ``config.json`` into a frozen dataclass. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import (rule 5). Grounded in the on-disk checkpoint at
``~/models/Qwen2.5-14B-Instruct-1M`` (``config.json`` + ``generation_config.json`` +
``tokenizer_config.json`` + the safetensors weight-key index), NOT assumed from a sibling model.

Architecture (empirically confirmed from the source files):

* **48 dense decoder layers** — no MoE, no hybrid SSM, no native MTP. Pure-attention GQA stack.
* **GQA attention with QKV biases** (Qwen2 quirk; Qwen3 drops them, Qwen3-Next gates the output):
  40 query heads × 8 KV heads (5:1), ``head_dim=128``. ``q_proj`` ``[5120 -> 5120]`` + bias [5120];
  ``k_proj``/``v_proj`` ``[5120 -> 1024]`` + biases [1024]; ``o_proj`` ``[5120 -> 5120]`` (no bias).
  **No** per-head QK-norm, **no** output gate, **full** RoPE on every head dim (no partial rotation).
* **SwiGLU FFN**: ``gate_proj``/``up_proj`` ``[5120 -> 13824]``, ``down_proj`` ``[13824 -> 5120]``.
* **RoPE, ``rope_theta=1e7``, ``rope_scaling=null``** — Qwen2.5-1M extends to 1M via *dual chunk
  attention* (DCA), NOT YaRN. The source ``dual_chunk_attention_config`` (``chunk_size=262144``,
  ``local_size=8192``, ``original_max_position_embeddings=262144``) is authoritative — above 256K
  the runtime serves via DCA; below it falls back to plain RoPE. ``max_position_embeddings`` is
  the baked target (1010000 by default).
* **Untied lm_head** (``tie_word_embeddings=false``).
* **Tokens.** Tokenizer has ``bos_token=null`` / ``add_bos_token=false`` (a BOS id 151643 exists
  as ``<|endoftext|>`` but is never prepended); the model's generation eos is **151645**
  (``<|im_end|>``); ``generation_config.json`` stop set is ``(151645, 151643)``; pad is 151643.

The vision-related ``<|vision_start|>`` / ``<|image_pad|>`` special tokens are reserved in the
tokenizer but the **text-only** checkpoint we serve has no vision tower — they are decoded
verbatim like any other added token and never enter the compute path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen25Config:
    """Hyperparameters of the Qwen2.5-14B-Instruct-1M decoder (dense GQA, no MoE/MTP/SSM)."""

    # core
    vocab_size: int                  # 152064
    hidden_size: int                 # 5120
    num_hidden_layers: int           # 48
    intermediate_size: int           # 13824

    # GQA attention
    num_attention_heads: int         # 40 query heads
    num_key_value_heads: int         # 8 KV heads (GQA 5:1)
    head_dim: int                    # 128 (= hidden_size / num_attention_heads)
    attention_bias: bool             # True for Qwen2: q/k/v have bias (o has none)

    # RoPE + long-context (dual chunk attention — NOT YaRN)
    rope_theta: float                # 1e7 (the long base — DCA assumes this)
    dca_chunk_size: int              # 262144 (native window per chunk)
    dca_local_size: int              # 8192 (inter-chunk local window / "successor" overlap)
    dca_original_max: int            # 262144 (below this: plain RoPE; above: DCA)
    max_position_embeddings: int     # 1010000 (the baked target context)

    # norm / activation / tying
    hidden_act: str                  # "silu"
    norm_eps: float                  # rms_norm_eps (1e-5)
    tie_word_embeddings: bool        # False — separate lm_head

    # tokens
    eos_token_id: int                # 151645 — <|im_end|>, the generation eos the model emits
    eos_token_ids: tuple[int, ...]   # (151645, 151643) — full generation stop set
    pad_token_id: int                # 151643 — <|endoftext|>
    bos_token_id: int | None         # 151643 (exists but NOT prepended; add_bos_token=False)
    add_bos_token: bool              # False — tokenizer never auto-prepends

    # --- derived geometry ------------------------------------------------------
    @property
    def q_dim(self) -> int:
        """Query projection out width = ``num_attention_heads * head_dim`` (5120)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """K/V projection out width = ``num_key_value_heads * head_dim`` (1024)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """GQA repeat factor = ``num_attention_heads // num_key_value_heads`` (5)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def use_dca(self) -> bool:
        """Whether DCA is configured (chunk_size > 0). True for Qwen2.5-1M."""
        return self.dca_chunk_size > 0

    def needs_dca(self, seq_len: int) -> bool:
        """Whether DCA should kick in for a sequence of ``seq_len`` tokens.

        Below the native window (``dca_original_max=262144``) plain RoPE is exact; above it the
        DCA chunked attention path takes over. The decision is made *per request* by the runtime.
        """
        return self.use_dca and seq_len > self.dca_original_max

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen25Config:
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text())
        gen: dict = {}
        gp = d / "generation_config.json"
        if gp.exists():
            gen = json.loads(gp.read_text())
        tok: dict = {}
        tp = d / "tokenizer_config.json"
        if tp.exists():
            tok = json.loads(tp.read_text())

        hidden_size = int(cfg["hidden_size"])
        n_heads = int(cfg["num_attention_heads"])
        head_dim = int(cfg.get("head_dim", hidden_size // n_heads))

        dca = cfg.get("dual_chunk_attention_config") or {}
        chunk = int(dca.get("chunk_size", 0))
        local = int(dca.get("local_size", 0))
        orig_max = int(dca.get("original_max_position_embeddings",
                                cfg.get("max_position_embeddings", 0)))

        eos = gen.get("eos_token_id", cfg.get("eos_token_id"))
        eos_ids = tuple(int(x) for x in (eos if isinstance(eos, list) else [eos] if eos is not None
                                         else []))

        return cls(
            vocab_size=int(cfg["vocab_size"]),
            hidden_size=hidden_size,
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            intermediate_size=int(cfg["intermediate_size"]),
            num_attention_heads=n_heads,
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=head_dim,
            # Qwen2 attention biases on q/k/v: the field is named "attention_bias" but Qwen2's
            # config.json doesn't ship it (the bias is implicit in the checkpoint shapes). Default
            # True for Qwen2; falsifiable by the loader's bias key check.
            attention_bias=bool(cfg.get("attention_bias", True)),
            rope_theta=float(cfg.get("rope_theta", 1e7)),
            dca_chunk_size=chunk,
            dca_local_size=local,
            dca_original_max=orig_max,
            max_position_embeddings=int(cfg.get("max_position_embeddings", 1_010_000)),
            hidden_act=str(cfg.get("hidden_act", "silu")),
            norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            eos_token_id=int(eos_ids[0]) if eos_ids else 151645,
            eos_token_ids=eos_ids or (151645, 151643),
            pad_token_id=int(gen.get("pad_token_id", cfg.get("pad_token_id", 151643))),
            bos_token_id=(int(cfg["bos_token_id"]) if cfg.get("bos_token_id") is not None
                          else None),
            add_bos_token=bool(tok.get("add_bos_token", False)),
        )
