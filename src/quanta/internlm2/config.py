"""InternLM2.5-7B-Chat-1M hyperparameters (``model_type="internlm2"``, ``InternLM2ForCausalLM``).

Parsed from the source ``config.json`` into a frozen dataclass. Pure Python + ``json`` — no
``torch``/``transformers``/``mlx`` import (rule 5). Grounded in the on-disk checkpoint at
``~/models/internlm2_5-7b-chat-1m`` (``config.json`` + ``generation_config.json`` +
``tokenizer_config.json`` + the safetensors weight-key index), NOT assumed from a sibling model.

Architecture (empirically confirmed from the source files):

* **32 dense decoder layers** — no MoE, no hybrid SSM, no native MTP. Pure-attention GQA stack.
* **GQA attention, NO biases** (``config.bias=False``): 32 query heads × 8 KV heads (4:1),
  ``head_dim=128``. The source ships **fused ``wqkv``** of shape
  ``[(num_heads + 2·num_kv_heads) · head_dim, hidden_size]`` = ``[6144, 4096]``, laid out
  per-kv-head: ``(num_kv_heads, num_kv_groups + 2, head_dim)`` = ``(8, 6, 128)`` row groups,
  with slots ``[0..3]`` = q-heads of that kv-head, slot ``-2`` = k-head, slot ``-1`` = v-head.
  Loader deinterleaves once at load time → standard ``wq``/``wk``/``wv``/``wo``.
* **SwiGLU FFN, InternLM2 naming**: ``w1``/``w3``/``w2`` (gate / up / down). ``w1``/``w3``
  ``[4096 → 14336]``, ``w2`` ``[14336 → 4096]``.
* **RoPE, ``rope_theta=5e7``, dynamic-NTK ``factor=2.5``**: Llama-style ``rotate_half`` over
  the full ``head_dim`` (not the Qwen2 interleaved-pair form). Above
  ``max_position_embeddings=262144`` the base is rescaled per the NTK formula
  ``base · ((factor · seq_len / max_pos) − (factor − 1)) ^ (dim / (dim − 2))``, applied on
  every forward pass that crosses the trained window.
* **Untied lm_head** (``tie_word_embeddings=False``). The output projection lives at the
  bare top-level key ``output.weight`` (not ``lm_head.weight``, not ``model.lm_head``).
* **Tokens.** ``bos_token_id=1`` (``<s>``), ``add_bos_token=True``; the model's nominal eos is
  ``eos_token_id=2`` (``</s>``), but ``generation_config.json`` ships the full stop set
  ``(2, 92542)`` where ``92542`` is the chat eos ``<|im_end|>`` the assistant actually emits
  to end a turn. ``pad_token_id=2``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InternLM2Config:
    """Hyperparameters of the InternLM2.5-7B-Chat-1M decoder (dense GQA, no MoE/MTP/SSM)."""

    # core
    vocab_size: int                  # 92544
    hidden_size: int                 # 4096
    num_hidden_layers: int           # 32
    intermediate_size: int           # 14336

    # GQA attention (NO biases — bias=False everywhere for InternLM2)
    num_attention_heads: int         # 32 query heads
    num_key_value_heads: int         # 8 KV heads (GQA 4:1)
    head_dim: int                    # 128 (= hidden_size / num_attention_heads)
    attention_bias: bool             # False for InternLM2 (config.bias=False)

    # RoPE + long-context (dynamic-NTK scaling — NOT YaRN, NOT DCA)
    rope_theta: float                # 5e7 (50 000 000)
    rope_scaling_type: str           # "dynamic"   — dynamic-NTK extrapolation
    rope_scaling_factor: float       # 2.5         — NTK factor
    max_position_embeddings: int     # 262144 (the trained ceiling — NTK extends past it to 1M)

    # norm / activation / tying
    hidden_act: str                  # "silu"
    norm_eps: float                  # rms_norm_eps (1e-5)
    tie_word_embeddings: bool        # False — separate `output.weight`

    # tokens
    eos_token_id: int                # 2          — </s>, nominal
    eos_token_ids: tuple[int, ...]   # (2, 92542) — full generation stop set (<|im_end|> = chat eos)
    pad_token_id: int                # 2
    bos_token_id: int                # 1          — <s>
    add_bos_token: bool              # True — tokenizer auto-prepends <s>

    # --- derived geometry ------------------------------------------------------
    @property
    def q_dim(self) -> int:
        """Query projection out width = ``num_attention_heads · head_dim`` (4096)."""
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        """K/V projection out width = ``num_key_value_heads · head_dim`` (1024)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """GQA repeat factor = ``num_attention_heads // num_key_value_heads`` (4)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def num_key_value_groups(self) -> int:
        """Same as :attr:`n_rep`; named per InternLM2's own ``num_key_value_groups`` for the
        ``wqkv`` deinterleave (``gs = num_key_value_groups + 2`` row-slots per kv-head)."""
        return self.n_rep

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def fused_qkv_out(self) -> int:
        """Out-feature count of the source ``wqkv`` weight before splitting.

        ``(num_heads + 2·num_kv_heads) · head_dim`` = ``(32 + 16) · 128`` = ``6144`` for InternLM2.5-7B.
        """
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    def ntk_base(self, seq_len: int) -> float:
        """Effective RoPE base for a forward pass of length ``seq_len``.

        Below the trained ceiling (``max_position_embeddings``), the source base ``rope_theta`` is
        used verbatim. Above it, the InternLM2 dynamic-NTK formula rescales:

            ``base · ((factor · seq_len / max_pos) − (factor − 1)) ^ (dim / (dim − 2))``

        with ``factor = rope_scaling_factor`` and ``dim = head_dim``. The runtime calls this
        once per forward pass with the *current* total sequence length (cache.offset + T) and
        passes the result to ``mx.fast.rope(..., base=…)``.
        """
        if self.rope_scaling_type != "dynamic" or seq_len <= self.max_position_embeddings:
            return float(self.rope_theta)
        factor = float(self.rope_scaling_factor)
        scale = (factor * seq_len / self.max_position_embeddings) - (factor - 1.0)
        return float(self.rope_theta) * scale ** (self.head_dim / (self.head_dim - 2))

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> InternLM2Config:
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

        rope_scaling = cfg.get("rope_scaling") or {}
        rs_type = str(rope_scaling.get("type", "none"))
        rs_factor = float(rope_scaling.get("factor", 1.0))

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
            # InternLM2 config exposes a single ``bias`` field for every projection — and ships
            # ``False`` for InternLM2.5. Fall back to ``False`` if absent (matches the architecture).
            attention_bias=bool(cfg.get("bias", False)),
            rope_theta=float(cfg.get("rope_theta", 1e4)),
            rope_scaling_type=rs_type,
            rope_scaling_factor=rs_factor,
            max_position_embeddings=int(cfg.get("max_position_embeddings", 262144)),
            hidden_act=str(cfg.get("hidden_act", "silu")),
            norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            eos_token_id=int(eos_ids[0]) if eos_ids else 2,
            eos_token_ids=eos_ids or (2, 92542),
            pad_token_id=int(gen.get("pad_token_id", cfg.get("pad_token_id", 2))),
            bos_token_id=int(cfg.get("bos_token_id", 1)),
            add_bos_token=bool(tok.get("add_bos_token", True)),
        )
