"""Aggregate guard: every MODEL-FREE parity gate must pass — plus the hardening guards that keep
the runner's safety from drifting.

This is the runner the ``parity/`` gates never had — the gap that let ``dsv4_tree_spec_test`` and
``qwen35_omlx_engine_test`` silently rot (stub signatures drifting from the real interfaces they
stand in for) across many commits, invisible because nothing exercised the gates in aggregate.

Two tiers:

* **Fast (default lane).** Pure-function guards over the classifier and the partition — they run in
  ``pytest tests/ -m "not slow"`` in milliseconds: the count manifest (catches a real-weight gate
  that evades detection — it would overshoot the model-free bucket), the ``*_real_test.py`` naming
  guard, the misnamed-gate scanner (a model-free gate hidden behind a non-``_test.py`` name), and
  unit tests that NEGATIVE-test the skip / vacuous-pass / real-weight logic without a resident env.
* **Slow (``slow`` marker, runs by default).** One isolated subprocess per model-free gate; a
  nonzero exit (or a vacuous-pass / swallowed-exception signal) fails the case with the gate's own
  banner. Skip the inner loop with ``pytest tests/ -m "not slow"``.

Real-weight (SOLO, 9-306 GiB) gates are excluded by construction — this never loads a resident
model. The standalone, streaming/parallel equivalent is ``parity.run_modelfree_sweep``.
"""

from __future__ import annotations

import pytest

from parity._modelfree import (
    EXPECTED_MODEL_FREE,
    EXPECTED_REAL_WEIGHT,
    EXPECTED_TOTAL,
    GateResult,
    _missing_optional_dep,
    _suspect_reason,
    classify_all,
    discover_model_free_gates,
    is_real_weight,
    run_gate,
    scan_misnamed_gates,
)

_GATES = discover_model_free_gates()


# --- Fast guards: the runner's safety can't silently drift ------------------------------------ #


def test_partition_manifest() -> None:
    """Every ``*_test.py`` gate is consciously classified. A drift here means a gate was added or
    removed: confirm its bucket and bump the ``EXPECTED_*`` constants. The dangerous case this
    catches: a real-weight gate that evades detection lands in the model-free bucket, so
    ``model_free`` overshoots its pin — the silent fail-open becomes a loud, must-look failure."""
    model_free, real_weight = classify_all()
    total = len(model_free) + len(real_weight)
    assert total == EXPECTED_TOTAL, (
        f"{total} `*_test.py` gates (expected {EXPECTED_TOTAL}) — a gate was added/removed; "
        f"confirm its bucket and update EXPECTED_TOTAL in parity/_modelfree.py."
    )
    assert len(real_weight) == EXPECTED_REAL_WEIGHT, (
        f"{len(real_weight)} real-weight gates (expected {EXPECTED_REAL_WEIGHT}); update "
        f"EXPECTED_REAL_WEIGHT if you intentionally added/removed a SOLO gate."
    )
    assert len(model_free) == EXPECTED_MODEL_FREE, (
        f"{len(model_free)} model-free gates (expected {EXPECTED_MODEL_FREE}). If you did NOT add a "
        f"model-free gate, a real-weight gate may have EVADED detection and landed here — check the "
        f"newest gate and mark it `# parity-gate: real-weight` or rename it `*_real_test.py`. "
        f"Otherwise bump EXPECTED_MODEL_FREE."
    )


def test_no_swept_real_test_named() -> None:
    """The smoking-gun guard: no SWEPT (model-free) gate may be named ``*_real_test.py`` — that name
    is the canonical real-weight signal, so such a gate would be a fail-open instance."""
    offenders = [m for m in _GATES if m.endswith("_real_test")]
    assert not offenders, (
        f"gates named like real-weight but classified model-free (swept): {offenders} — they would "
        f"load resident weights in the sweep. Add a real-weight marker/sentinel or fix the name."
    )


def test_no_misnamed_gates() -> None:
    """No model-free assertion gate may hide behind a non-``*_test.py`` name (it would never be
    swept). New offenders must be renamed ``*_test.py`` or added to ``_KNOWN_NON_GATES``."""
    flagged = scan_misnamed_gates()
    assert not flagged, (
        f"non-`*_test.py` files that look like uncovered model-free gates: {flagged} — rename them "
        f"`*_test.py` to sweep them, or document them in _KNOWN_NON_GATES (parity/_modelfree.py)."
    )


# --- Fast unit tests: NEGATIVE-test the new classifier / skip / vacuous-pass logic ------------ #


def test_is_real_weight_signals() -> None:
    assert is_real_weight("x = '/Users/pmrj/models/Foo'")              # literal path marker
    assert is_real_weight("mx.set_wired_limit(490)")                   # wired-limit marker
    assert is_real_weight("", name="foo_real_test.py")                 # name convention
    assert is_real_weight("p = expanduser('~/models/Bar')")           # ~/models fail-open shape
    assert is_real_weight("# parity-gate: real-weight\nx=1")          # explicit opt-out
    assert not is_real_weight("x = mx.zeros((4, 4))")                  # plain model-free
    assert not is_real_weight("# parity-gate: model-free\nset_wired_limit  # in a comment")
    # Real-weight precedence beats the model-free override.
    assert is_real_weight("# parity-gate: real-weight\n# parity-gate: model-free")


def test_missing_optional_dep() -> None:
    assert _missing_optional_dep("ModuleNotFoundError: No module named 'safetensors'") == "safetensors"
    assert _missing_optional_dep("No module named 'transformers.models'") == "transformers"
    assert _missing_optional_dep("No module named 'numpy'") is None      # base dep ⇒ a real failure
    assert _missing_optional_dep("all good") is None


def test_suspect_reason() -> None:
    assert _suspect_reason(0, "ok\nTraceback (most recent call last):\n...")   # swallowed exception
    assert _suspect_reason(0, "ran\nPARITY-CHECKS: 0\n")                       # no assertions ran
    assert not _suspect_reason(0, "PASS\nPARITY-CHECKS: 7\n")                  # clean pass
    assert not _suspect_reason(0, "all good")                                  # clean, no markers
    assert not _suspect_reason(1, "Traceback (most recent call last):")       # rc!=0 already a fail


def test_gate_result_semantics() -> None:
    clean = GateResult("parity.x", 0, 0.1, "PASS")
    assert clean.ok and not clean.failed
    skipped = GateResult("parity.x", 1, 0.1, "", skipped=True, skip_reason="missing dep")
    assert not skipped.ok and not skipped.failed              # skip is neither
    suspect = GateResult("parity.x", 0, 0.1, "", suspect_reason="swallowed")
    assert not suspect.ok and suspect.failed                  # rc0 but vacuous ⇒ fail
    crashed = GateResult("parity.x", 1, 0.1, "boom")
    assert not crashed.ok and crashed.failed


# --- Slow lane: actually run every model-free gate -------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("module", _GATES, ids=lambda m: m.removeprefix("parity."))
def test_parity_gate(module: str) -> None:
    """Run one model-free gate in isolation; assert it passed (exit 0, no vacuous-pass signal).
    A missing ``reference``-extra dep is a SKIP, not a failure (clean base-deps-only env)."""
    result = run_gate(module)
    if result.skipped:
        pytest.skip(f"{module}: {result.skip_reason}")
    assert result.ok, (
        f"{module} exited {result.returncode} in {result.seconds:.1f}s "
        f"| {result.suspect_reason or result.summary}"
    )
