"""Qwen2.5-14B-Instruct-1M support (``model_type="qwen2"``, ``Qwen2ForCausalLM``).

A **dense** 48-layer transformer (no MoE, no hybrid SSM, no MTP) with GQA attention (40 / 8 heads,
head_dim 128, **QKV biases** — Qwen2 quirk, dropped in Qwen3), SwiGLU FFN (hidden 5120 → inter
13824), full RoPE (``rope_theta=1e7``) and **dual chunk attention** (chunk 262144, local 8192) for
the 1M-context window — *not* YaRN. The simplest of quanta's target architectures: int8 attention +
int4 FFN g64 affine quant, bf16 norms/biases/embed/lm_head.
"""
