"""Aggregate guard: every MODEL-FREE parity gate must pass — plus the hardening guards that keep
the runner's safety from drifting.

This is the runner the ``parity/`` gates never had — the gap that let ``dsv4_tree_spec_test`` and
``qwen35_omlx_engine_test`` silently rot (stub signatures drifting from the real interfaces they
stand in for) across many commits, invisible because nothing exercised the gates in aggregate.

Two tiers:

* **Fast (default lane).** Pure-function guards over the classifier and the partition — they run in
  ``pytest tests/ -m "not slow"`` in milliseconds: the identity-pinned partition manifest (catches a
  real-weight gate that evades detection — it shows up as a new name in the model-free bucket, and
  an offsetting add+remove that counts alone would miss), the ``*_real_test.py`` naming guard, the
  misnamed-gate scanner (a model-free gate hidden behind a non-``_test.py`` name, incl. pytest
  style), the allowlist-staleness guard, and unit tests that NEGATIVE-test the skip / vacuous-pass /
  real-weight logic without a resident env.
* **Slow (``slow`` marker, runs by default).** One isolated subprocess per model-free gate; a
  nonzero exit (or a vacuous-pass / swallowed-exception signal) fails the case. Skip the inner loop
  with ``pytest tests/ -m "not slow"``.

Real-weight (SOLO, 9-306 GiB) gates are excluded by construction — this never loads a resident
model. The standalone, streaming/parallel equivalent is ``parity.run_modelfree_sweep``.
"""

from __future__ import annotations

import os

import pytest

from parity._modelfree import (
    GateResult,
    _looks_like_gate,
    _missing_optional_dep,
    _rss_gib,
    _suspect_reason,
    discover_model_free_gates,
    is_real_weight,
    load_manifest,
    manifest_diff,
    optional_deps,
    run_gate,
    scan_misnamed_gates,
    stale_allowlist_entries,
)

_GATES = discover_model_free_gates()
_OPTIONAL = optional_deps()


# --- Fast guards: the runner's safety can't silently drift ------------------------------------ #


def test_partition_manifest() -> None:
    """The live partition must equal the pinned name set in parity/gate_manifest.json. Identity, not
    counts: this catches a gate added/removed/reclassified AND an offsetting add+remove that keeps
    the count constant. The dangerous case: a real-weight gate that evades detection appears in
    'added' (and joins model_free) — a loud, named failure instead of a 306 GiB load in CI."""
    diff = manifest_diff()
    assert not any(diff.values()), (
        f"gate partition drifted from the pinned manifest: {diff}. Regenerate with "
        f"`uv run python -m parity.run_modelfree_sweep --update-manifest` and REVIEW the diff — a "
        f"gate in 'added' that joins model_free will be SWEPT, so it must not load real weights."
    )


def test_manifest_well_formed() -> None:
    pin = load_manifest()
    assert pin["model_free"] and pin["real_weight"], "manifest buckets must be non-empty"
    assert not (set(pin["model_free"]) & set(pin["real_weight"])), "a gate is in both buckets"


def test_no_swept_real_test_named() -> None:
    """Smoking-gun guard: no SWEPT (model-free) gate may be named ``*_real_test.py`` — that name is
    the canonical real-weight signal, so such a gate would be a fail-open instance."""
    offenders = [m for m in _GATES if m.endswith("_real_test")]
    assert not offenders, (
        f"gates named like real-weight but classified model-free (swept): {offenders} — they would "
        f"load resident weights in the sweep. Add a real-weight marker/sentinel or fix the name."
    )


def test_no_misnamed_gates() -> None:
    """No model-free assertion gate may hide behind a non-``*_test.py`` name (incl. pytest-style):
    it would never be swept. New offenders must be renamed ``*_test.py`` or documented."""
    flagged = scan_misnamed_gates()
    assert not flagged, (
        f"non-`*_test.py` files that look like uncovered model-free gates: {flagged} — rename them "
        f"`*_test.py` to sweep them, or document them in _KNOWN_NON_GATES (parity/_modelfree.py)."
    )


def test_allowlist_not_stale() -> None:
    """Every _KNOWN_NON_GATES entry must still exist — a removed/renamed file should not leave a
    silent suppression behind."""
    stale = stale_allowlist_entries()
    assert not stale, f"_KNOWN_NON_GATES entries whose file is gone (remove them): {stale}"


# --- Fast unit tests: NEGATIVE-test the new classifier / skip / vacuous-pass logic ------------ #


def test_is_real_weight_signals() -> None:
    assert is_real_weight("x = '/Users/pmrj/models/Foo'")              # literal path marker
    assert is_real_weight("mx.set_wired_limit(490)")                   # wired-limit marker
    assert is_real_weight("", name="foo_real_test.py")                 # name convention
    assert is_real_weight("p = Path.home() / 'models' / art")         # home-relative LOAD idiom
    assert is_real_weight("p = expanduser('~/models/Bar')")           # expanduser load idiom
    assert is_real_weight("d = os.environ['X']; load(d, 'models')")   # env-relative load idiom
    assert is_real_weight("# parity-gate: real-weight\nx=1")          # explicit opt-out
    assert not is_real_weight("x = mx.zeros((4, 4))")                  # plain model-free
    # The qwen35_forward_test shape: a bare `~/models` literal in COMMENTED-OUT code is NOT a load.
    assert not is_real_weight("# cfg = Cfg.from_pretrained('~/models/Foo')  # deferred\nx = 1")


def test_optional_deps_from_pyproject() -> None:
    """The skip-eligible set is read from pyproject's extras (never drifts), unioned with baseline."""
    assert {"safetensors", "transformers", "sentencepiece"} <= _OPTIONAL   # baseline reference extra
    assert "omlx" in _OPTIONAL                                             # the omlx extra, derived
    assert {"pillow", "PIL"} <= _OPTIONAL   # pillow dist + its PIL import-name alias (skip-eligible)


def test_missing_optional_dep() -> None:
    assert _missing_optional_dep("No module named 'safetensors'", _OPTIONAL) == "safetensors"
    assert _missing_optional_dep("No module named 'omlx.cli'", _OPTIONAL) == "omlx"
    assert _missing_optional_dep("No module named 'numpy'", _OPTIONAL) is None   # base dep ⇒ a fail
    assert _missing_optional_dep("all good", _OPTIONAL) is None


def test_suspect_reason() -> None:
    assert _suspect_reason(0, "ok\nTraceback (most recent call last):\n...")   # swallowed exception
    assert _suspect_reason(0, "ran\nPARITY-CHECKS: 0\n")                       # no assertions ran
    assert not _suspect_reason(0, "PASS\nPARITY-CHECKS: 7\n")                  # proof-of-work passes
    # PARITY-CHECKS>0 is the escape hatch: a gate that legitimately renders a Traceback is trusted.
    assert not _suspect_reason(0, "PARITY-CHECKS: 3\nTraceback (most recent call last):\n...")
    assert not _suspect_reason(0, "all good")                                  # clean, no markers
    assert not _suspect_reason(1, "Traceback (most recent call last):")       # rc!=0 already a fail


def test_looks_like_gate() -> None:
    assert _looks_like_gate("if __name__=='__main__':\n    assert 1 == 1")     # classic
    assert _looks_like_gate("def test_foo():\n    assert x")                   # pytest style, no main
    assert _looks_like_gate("def test_y():\n    np.testing.assert_allclose(a, b)")
    assert not _looks_like_gate("x = 1\nprint('hi')")                          # no assertions
    assert not _looks_like_gate("def helper():\n    return 1")                 # not a test/runnable


def test_gate_result_semantics() -> None:
    clean = GateResult("parity.x", 0, 0.1, "PASS")
    assert clean.ok and not clean.failed
    skipped = GateResult("parity.x", 1, 0.1, "", skipped=True, skip_reason="missing dep")
    assert not skipped.ok and not skipped.failed              # skip is neither
    suspect = GateResult("parity.x", 0, 0.1, "", suspect_reason="swallowed")
    assert not suspect.ok and suspect.failed                  # rc0 but vacuous ⇒ fail
    crashed = GateResult("parity.x", 1, 0.1, "boom")
    assert not crashed.ok and crashed.failed


def test_rss_gib() -> None:
    """The watchdog's RSS probe: positive for a live pid (this process), 0.0 for a bogus/dead one —
    a transient `ps` miss must read as 0, never as a spurious over-ceiling that false-kills a gate."""
    assert _rss_gib(os.getpid()) > 0.0
    assert _rss_gib(2_000_000_000) == 0.0   # a pid that cannot exist ⇒ ps prints nothing


# --- Slow lane: actually run every model-free gate -------------------------------------------- #


@pytest.mark.slow
def test_run_gate_memory_watchdog() -> None:
    """The RSS ceiling kills + fails LOUD a swept gate that crosses it — the runtime backstop for a
    real-weight gate that evaded static detection (it would otherwise fault in hundreds of GiB and
    OOM the box). Forced cheaply: any mlx-importing gate blows past a 50 MiB ceiling within the
    first poll, so no multi-GiB allocation is needed. rc 137 = the SIGKILL we sent."""
    tripped = run_gate(_GATES[0], rss_ceiling_gib=0.05)
    assert tripped.returncode == 137 and tripped.failed, tripped
    assert "ceiling" in tripped.suspect_reason


@pytest.mark.slow
@pytest.mark.parametrize("module", _GATES, ids=lambda m: m.removeprefix("parity."))
def test_parity_gate(module: str) -> None:
    """Run one model-free gate in isolation; assert it passed (exit 0, no vacuous-pass signal).
    A missing optional-extra dep is a SKIP, not a failure (clean base-deps-only env)."""
    result = run_gate(module, optional=_OPTIONAL)
    if result.skipped:
        pytest.skip(f"{module}: {result.skip_reason}")
    assert result.ok, (
        f"{module} exited {result.returncode} in {result.seconds:.1f}s "
        f"| {result.suspect_reason or result.summary}"
    )
