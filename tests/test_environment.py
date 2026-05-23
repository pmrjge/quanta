"""Environment smoke checks.

Not behavioural tests — these assert the toolchain this project commits to is
importable and that the ``mlx.fast`` primitives we build on exist, so a broken or
downgraded environment fails fast and visibly rather than at runtime.
"""

import mlx.core as mx
import mlx.nn as nn


def test_mlx_core_works() -> None:
    assert int(mx.array([1, 2, 3]).sum().item()) == 6


def test_mlx_nn_module_available() -> None:
    # Layers are built as mlx.nn modules (the simplification rule).
    assert hasattr(nn, "Module")


def test_mlx_fast_primitives_available() -> None:
    # The optimization rule: build on mlx.fast primitives, not hand-rolled loops.
    for primitive in ("rms_norm", "scaled_dot_product_attention", "rope"):
        assert hasattr(mx.fast, primitive), f"missing mlx.fast.{primitive}"
