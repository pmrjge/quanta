"""Discovery + isolated-subprocess runner for the MODEL-FREE parity gates.

Shared by the pytest collector (``tests/test_parity_modelfree.py``) and the standalone sweep
(``parity.run_modelfree_sweep``) so the two can never disagree on WHICH gates are model-free or HOW
they are run.

A parity gate is *model-free* when it allocates only a few KB-MB of stub tensors and is safe to run
anywhere, anytime. The *real-weight* gates (which load 9-306 GiB resident and must run SOLO, one
model at a time) are excluded by :func:`is_real_weight` — a multi-signal detector that fails toward
EXCLUSION: the cost of a false *inclusion* is a 300 GiB load in CI, while a false *exclusion* merely
means a fast gate is run SOLO by hand.

This is the runner the ``parity/`` gates never had — the gap that let ``dsv4_tree_spec_test`` and
``qwen35_omlx_engine_test`` silently rot (stub signatures drifting from the real interfaces they
stand in for), invisible because nothing exercised the gates in aggregate. The hardening here closes
the residual exposure that the first cut left open:

* **Fail-open exclusion.** The two original markers (``/Users/pmrj/models`` / ``set_wired_limit``)
  alone fail OPEN — a real-weight gate that loads via ``~/models``/``expanduser`` or a *symlinked*
  artifact dir, relying on MLX's default wired limit, slips through (the two
  ``*_omlx_v1_messages_smoke.py`` server smokes are exactly this shape — saved today only by NOT
  being named ``*_test.py``). The detector now also keys on the ``*_real_test.py`` name convention,
  an explicit sentinel, and an ``expanduser``+``models`` heuristic; and :func:`classify_all` is
  pinned by a count manifest (see ``EXPECTED_*``) so an undetected real gate — which would land in
  the model-free bucket — overshoots the pin and trips ``test_partition_manifest`` LOUDLY.
* **Vacuous pass.** ``run_gate`` no longer trusts the exit code alone: a returncode-0 run that
  printed a Traceback (a swallowed exception) or an explicit ``PARITY-CHECKS: 0`` is reported as a
  failure (:func:`_suspect_reason`).
* **reference-extra coupling.** ~11 gates import ``safetensors`` (the offline-only ``reference``
  extra). On a base-deps-only env they're SKIPPED, not failed (:func:`_missing_optional_dep`).
* **Framing.** A green sweep proves interface + logic on synthetic stubs — NOT real-model numeric
  parity, which lives entirely in the excluded SOLO gates (:func:`summary_banner`).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARITY_DIR.parent
DEFAULT_TIMEOUT = 300

# --- Real-weight classification --------------------------------------------------------------- #
# Literal source markers that a gate loads a named resident artifact / pins the wired limit.
_REAL_WEIGHT_MARKERS = ("/Users/pmrj/models", "set_wired_limit")
# Explicit one-line opt-outs (drop either as a comment in a gate to force its bucket). Real-weight
# wins outright; model-free only overrides the *fuzzy* heuristics (a marker appearing in a comment).
REAL_WEIGHT_SENTINEL = "parity-gate: real-weight"
MODEL_FREE_SENTINEL = "parity-gate: model-free"
# Naming convention for real-weight gates. Only 5/50 use it today, but going forward it is the
# canonical signal — a gate so named is excluded even if it forgets the markers (closes fail-open).
_REAL_WEIGHT_NAME_SUFFIX = "_real_test.py"


def is_real_weight(src: str, name: str = "") -> bool:
    """True when a gate loads real resident weights (SOLO-only ⇒ excluded from the sweep).

    Multi-signal and fail-toward-exclusion. ``src`` is the gate source; ``name`` its filename.
    Precedence: an explicit ``REAL_WEIGHT_SENTINEL`` or the ``*_real_test.py`` name forces real;
    a ``MODEL_FREE_SENTINEL`` then overrides the fuzzy marker/expanduser heuristics (the escape
    hatch for a marker that only appears in a comment); otherwise markers / ``~/models`` loads win.
    """
    if REAL_WEIGHT_SENTINEL in src:
        return True
    if name.endswith(_REAL_WEIGHT_NAME_SUFFIX):
        return True
    if MODEL_FREE_SENTINEL in src:
        return False
    if any(marker in src for marker in _REAL_WEIGHT_MARKERS):
        return True
    # `~/models`-style loads that dodge the literal-path marker — the documented fail-open shape.
    if "expanduser" in src and "models" in src:
        return True
    return False


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def classify_all() -> tuple[list[str], list[str]]:
    """Partition every ``parity/*_test.py`` gate into ``(model_free, real_weight)`` module-name
    lists. Filesystem + source-text only (no imports), so it is safe at pytest collection time."""
    model_free: list[str] = []
    real_weight: list[str] = []
    for path in sorted(PARITY_DIR.glob("*_test.py")):
        mod = f"parity.{path.stem}"
        (real_weight if is_real_weight(_read(path), path.name) else model_free).append(mod)
    return model_free, real_weight


def discover_model_free_gates() -> list[str]:
    """Sorted ``parity.<stem>`` module names of every model-free gate. New gates are picked up
    automatically — the whole point: a gate added tomorrow is swept with no edit here."""
    return classify_all()[0]


def real_weight_gates() -> list[str]:
    """Sorted module names of the real-weight (SOLO, EXCLUDED) gates — surfaced for framing."""
    return classify_all()[1]


# --- Partition manifest (the fail-open backstop) ---------------------------------------------- #
# EVERY `*_test.py` gate must be consciously classified. Adding/removing a gate changes one of these
# counts and trips ``test_partition_manifest``, which refuses to pass until a human confirms the new
# gate's bucket and bumps the number. Crucially: a real-weight gate that EVADES detection lands in
# the model-free bucket, so ``model_free`` overshoots its pin — the silent fail-open becomes a loud,
# must-look failure. Bump these (and only these) when you intentionally add or remove a gate.
EXPECTED_TOTAL = 148
EXPECTED_MODEL_FREE = 98
EXPECTED_REAL_WEIGHT = 50

# --- Misnamed-gate scanner (coverage convention) ---------------------------------------------- #
# Non-`*_test.py` files that look like model-free assertion gates but are intentionally NOT part of
# the swept suite. Each needs a justification; ``scan_misnamed_gates`` fails on any NEW such file,
# forcing it to be renamed `*_test.py` (and swept) or added here on purpose.
_KNOWN_NON_GATES = {
    "hierarchical_routing_ablation.py":
        "research ablation (synthetic routing recall study), not a correctness gate",
    "nemotron_omlx_v1_messages_smoke.py":
        "real-weight e2e oMLX server smoke on a resident artifact — SOLO-only, not swept",
    "qwen25_omlx_v1_messages_smoke.py":
        "real-weight e2e oMLX server smoke on a resident artifact — SOLO-only, not swept",
}
_RUNNER_FILES = ("_modelfree.py", "run_modelfree_sweep.py", "__init__.py")
_ASSERT_RE = re.compile(r"^\s*assert\s|raise\s+SystemExit|raise\s+AssertionError", re.MULTILINE)


def scan_misnamed_gates() -> list[str]:
    """Basenames of non-``*_test.py`` parity files that smell like model-free assertion gates
    (``__main__`` + assertions + not real-weight) and are NOT in the documented allowlist.

    A non-empty result is a silent-coverage-gap: a gate that would never be swept because of its
    name. The fix is to rename it ``*_test.py`` (so discovery picks it up) or, if it is genuinely
    not a gate, add it to ``_KNOWN_NON_GATES`` with a reason.
    """
    flagged: list[str] = []
    for path in sorted(PARITY_DIR.glob("*.py")):
        name = path.name
        if name.endswith("_test.py") or name in _RUNNER_FILES or name in _KNOWN_NON_GATES:
            continue
        src = _read(path)
        if "__main__" in src and _ASSERT_RE.search(src) and not is_real_weight(src, name):
            flagged.append(name)
    return flagged


# --- Gate execution --------------------------------------------------------------------------- #
# The offline-only `reference` extra. A base-deps-only env (clean CI) skips gates that import these
# rather than reporting a false failure.
_OPTIONAL_DEPS = ("safetensors", "transformers", "sentencepiece")
_MISSING_DEP_RE = re.compile(r"No module named '([\w.]+)'")
_TRACEBACK_SIG = "Traceback (most recent call last):"
_CHECKS_RE = re.compile(r"PARITY-CHECKS:\s*(\d+)")


def _missing_optional_dep(output: str) -> str | None:
    """If a failed run is explained solely by an absent ``reference``-extra module, return its
    name (so the gate is SKIPPED, not failed). On a dev env with the extra installed, never fires."""
    for m in _MISSING_DEP_RE.finditer(output):
        top = m.group(1).split(".")[0]
        if top in _OPTIONAL_DEPS:
            return top
    return None


def _suspect_reason(returncode: int, output: str) -> str:
    """Non-empty when a returncode-0 run shows evidence it did NOT pass cleanly — the vacuous-pass
    guard. Catches a swallowed exception (caught, Traceback printed, exited 0) and an explicit
    zero-check marker. Tight enough to be false-positive-free on the current suite (no gate prints a
    Traceback on success). Clean assertion-erosion inside a still-green gate stays out of reach
    without per-gate check counts — the opt-in ``PARITY-CHECKS: <n>`` contract a gate can print."""
    if returncode != 0:
        return ""  # already a failure; the returncode carries it
    if _TRACEBACK_SIG in output:
        return "exited 0 but printed a Traceback (swallowed exception)"
    m = _CHECKS_RE.search(output)
    if m and int(m.group(1)) == 0:
        return "exited 0 but reported PARITY-CHECKS: 0 (no assertions ran)"
    return ""


def _summary(out: str) -> str:
    """The last 'interesting' line of a gate's output (its PASS/FAIL/error banner), truncated."""
    keys = ("tests passed", "pass", "fail", "error", "traceback", "exception",
            "assert", "systemexit", "timeout", "parity-checks")
    last = ""
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in keys):
            last = line.strip()
    return last[:120]


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate run. ``skipped`` (missing optional dep) is neither pass nor fail."""

    module: str
    returncode: int
    seconds: float
    summary: str
    skipped: bool = False
    skip_reason: str = ""
    suspect_reason: str = ""

    @property
    def ok(self) -> bool:
        """Passed = ran, exited 0, and showed no swallow/vacuous-pass evidence."""
        return not self.skipped and self.returncode == 0 and not self.suspect_reason

    @property
    def failed(self) -> bool:
        """Genuinely failed (a skip is not a failure)."""
        return not self.skipped and not self.ok


def run_gate(module: str, *, timeout: int = DEFAULT_TIMEOUT) -> GateResult:
    """Run one gate as ``python -m <module>`` in an ISOLATED subprocess (fresh mlx state, no
    cross-gate contamination — the way the gates are designed to run). A timeout is reported as
    returncode 124, never raised, so a single hang can't abort a whole sweep. A returncode-0 run is
    additionally screened for vacuous-pass evidence, and a missing ``reference``-extra dep yields a
    SKIP rather than a false failure.
    """
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GateResult(module, 124, time.time() - t0, f"TIMEOUT >{timeout}s")
    out = proc.stdout + proc.stderr
    secs = time.time() - t0
    if proc.returncode != 0:
        dep = _missing_optional_dep(out)
        if dep is not None:
            return GateResult(module, proc.returncode, secs, _summary(out),
                              skipped=True,
                              skip_reason=f"missing optional dep '{dep}' (reference extra)")
    return GateResult(module, proc.returncode, secs, _summary(out),
                      suspect_reason=_suspect_reason(proc.returncode, out))


def summary_banner() -> str:
    """One-line framing: what a green sweep does and does NOT prove (the false-confidence guard)."""
    mf, rw = classify_all()
    return (f"{len(mf)} model-free gates swept here (interface + logic on synthetic stubs); "
            f"{len(rw)} real-weight SOLO gates NOT run here — real-model numeric parity "
            f"(teacher-forced ppl / layer parity) lives there and stays a manual SOLO step.")
