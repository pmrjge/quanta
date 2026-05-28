"""InternLM2.5-7B-Chat-1M support (``model_type="internlm2"``, ``InternLM2ForCausalLM``).

A **dense** 32-layer transformer (no MoE, no hybrid SSM, no MTP) with GQA attention
(32 / 8 heads, head_dim 128, **no biases** — InternLM2 sets ``bias=False`` for every
projection), SwiGLU FFN (hidden 4096 → inter 14336), full RoPE on the entire head_dim
(``rope_theta=5e7``) with **dynamic-NTK scaling** (``factor=2.5``) — *not* YaRN, *not*
DCA — to extend the trained 256K window out to 1M context.

The InternLM2 checkpoint stores attention with a **fused** ``wqkv`` weight whose rows
are laid out *per-kv-head*: each block of ``(num_key_value_groups + 2) · head_dim`` rows
covers one kv-head's q-heads (the first ``num_key_value_groups`` slots), its k-head
(slot −2), and its v-head (slot −1). The :mod:`~quanta.internlm2.loader` deinterleaves
this once at load time and presents three standard ``wq/wk/wv`` projections, so the
bake / artifact / runtime see a uniform GQA layout — no fused-qkv handling on the hot
path.

Tokenizer: SentencePiece (BPE) with InternLM2's six added tokens
(``<|im_start|>``/``<|im_end|>``/``<|action_start|>``/``<|action_end|>``/
``<|interpreter|>``/``<|plugin|>``) plus the standard ``<s>``/``</s>``/``<unk>``.
Chat template is the inline ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` form from
``tokenizer_config.json``; BOS (``<s>``) is auto-prepended (``add_bos_token=True``).
The generation stop set is ``{2 (</s>), 92542 (<|im_end|>)}`` per
``generation_config.json``.

Quant policy (mirrors :mod:`quanta.qwen25.bake`): int8 g64 affine for the attention
matmul weights (``wq``/``wk``/``wv``/``wo``), int4 g64 affine for the SwiGLU FFN
(``w1``/``w3``/``w2``), bf16 dense for everything else (norms, tied-or-untied embed,
``output``/``lm_head``).
"""
