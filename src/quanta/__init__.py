"""quanta — parity-first MLX quantization + sparse-MoE inference runtime.

See ``CLAUDE.md`` for the permanent engineering rules (build layers as ``mlx.nn``
modules, prefer ``mlx.fast`` primitives, no Python loops on compute paths) and the
parity-first methodology that gates every component against a numeric reference
before it is optimized or its quantization is judged.
"""

__version__ = "0.1.0"
