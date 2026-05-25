"""MiniMax-M2.7 (``minimax_m2``) support for quanta — GQA (full softmax) + partial RoPE + per-layer
QK-norm + sigmoid noaux_tc MoE (256 experts top-8, no shared expert) + 3 native MTP modules, from a
block-fp8 source checkpoint.

Submodules are imported directly (``quanta.minimax.config``, ``quanta.minimax.loader``,
``quanta.minimax.tokenizer``, …) to keep package import side-effect-free and avoid pulling MLX in
before it is needed.
"""
