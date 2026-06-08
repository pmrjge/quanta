"""Aggregate guard: every MODEL-FREE parity gate must pass.

This is the runner the ``parity/`` gates never had — the gap that let ``dsv4_tree_spec_test`` and
``qwen35_omlx_engine_test`` silently rot (stub signatures drifting from the real interfaces they
stand in for) across many commits. Each model-free ``parity/*_test.py`` is discovered (filesystem
scan, see :mod:`parity._modelfree`) and run in an isolated subprocess; a nonzero exit fails the
corresponding case with the gate's own banner, so ``pytest tests/`` now catches this class of rot.

Marked ``slow``: it spawns ~one subprocess per gate (~minutes wall-clock). It RUNS by default so the
rot is caught automatically; for the fast inner loop skip it with::

    pytest tests/ -m "not slow"

Real-weight (SOLO, 9-306 GiB) gates are excluded by construction — this never loads a resident
model. The standalone, streaming/parallel equivalent is ``parity.run_modelfree_sweep``.
"""

from __future__ import annotations

import pytest

from parity._modelfree import discover_model_free_gates, run_gate

_GATES = discover_model_free_gates()


def test_model_free_gates_discovered() -> None:
    """Guard discovery itself: an empty/tiny set (moved dir, broken heuristic) would make the sweep
    vacuously pass. Pin a sane floor well below the current count (~98)."""
    assert len(_GATES) >= 50, (
        f"only {len(_GATES)} model-free gates discovered — discovery likely broke "
        f"(expected ~98). Gates: {_GATES[:5]}..."
    )


@pytest.mark.slow
@pytest.mark.parametrize("module", _GATES, ids=lambda m: m.removeprefix("parity."))
def test_parity_gate(module: str) -> None:
    """Run one model-free gate in isolation; assert it exits 0."""
    result = run_gate(module)
    assert result.ok, (
        f"{module} exited {result.returncode} in {result.seconds:.1f}s | {result.summary}"
    )
