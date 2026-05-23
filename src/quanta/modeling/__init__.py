"""MLX-native runtime modules for the Kimi-K2.6 text decoder.

Built as ``mlx.nn`` modules (CLAUDE.md rule 1). The naive, obviously-correct path
is the default; ``mx.fast`` fused kernels (rope, SDPA) are opt-in via ``use_fast``
and are gated against the naive path by the parity harness before being trusted.
"""
