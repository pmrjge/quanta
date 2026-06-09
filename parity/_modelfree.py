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
the residual exposure surfaced in two rounds of risk audit:

* **Fail-open exclusion.** The detector keys on the ``*_real_test.py`` name, the literal markers
  (``/Users/pmrj/models`` / ``set_wired_limit``), a ``~/models`` *loading idiom* in code
  (``Path.home(`` / ``expanduser`` / ``os.environ`` / ``getenv`` together with ``models`` — NOT the
  bare ``~/models`` literal, which appears in *commented-out* code in model-free gates), and an
  explicit ``# parity-gate: real-weight`` sentinel — and it fails toward exclusion.
* **Identity-pinned manifest.** :data:`MANIFEST_PATH` pins the exact *name set* of each bucket (not
  just counts — counts miss an offsetting add+remove). :func:`manifest_diff` reports any gate added,
  removed, or moved between buckets; the pytest guard refuses to pass until a human regenerates the
  manifest (``--update-manifest``) and reviews the diff. A real-weight gate that EVADES detection
  shows up as a NEW name in ``model_free`` — the silent fail-open becomes a loud, named failure.
* **Vacuous pass.** :func:`run_gate` does not trust the exit code alone: an rc-0 run that printed a
  Traceback (a swallowed exception) is a failure, UNLESS it also printed a positive
  ``PARITY-CHECKS: <n>`` (the opt-in contract — a gate that proves it ran ``n>0`` assertions is
  trusted even when it legitimately renders a Traceback); ``PARITY-CHECKS: 0`` always fails.
* **reference-extra coupling.** Gates importing an *optional* dep (the ``reference``/``omlx`` extras,
  derived from ``pyproject.toml`` so the set never drifts) are SKIPPED, not failed, on a base-deps
  env.
* **Framing.** A green sweep proves interface + logic on synthetic stubs — NOT real-model numeric
  parity, which lives entirely in the excluded SOLO gates (:func:`summary_banner`).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARITY_DIR.parent
DEFAULT_TIMEOUT = 300

# --- Real-weight classification --------------------------------------------------------------- #
# Literal source markers that a gate loads a named resident artifact / pins the wired limit.
_REAL_WEIGHT_MARKERS = ("/Users/pmrj/models", "set_wired_limit")
# Explicit one-line opt-out (drop as a comment in a gate to force it out of the sweep). There is NO
# model-free force-INCLUDE sentinel by design: a lever that overrides the markers to force a gate
# into the sweep is a fail-open footgun (it could mask a real `set_wired_limit`). Excludes only.
REAL_WEIGHT_SENTINEL = "parity-gate: real-weight"
# Naming convention for real-weight gates — going forward the canonical signal: a gate so named is
# excluded even if it forgets the markers.
_REAL_WEIGHT_NAME_SUFFIX = "_real_test.py"
# `~/models`-style LOADING idioms. Keyed on the load call (Path.home()/expanduser/env), NOT the bare
# `~/models` literal — that literal appears in commented-out deferred code in genuinely model-free
# gates (e.g. qwen35_forward_test), so matching it would wrongly EXCLUDE them.
_HOME_LOAD_IDIOMS = ("Path.home(", "expanduser", "os.environ", "os.getenv", "getenv")


def is_real_weight(src: str, name: str = "") -> bool:
    """True when a gate loads real resident weights (SOLO-only ⇒ excluded from the sweep).

    Multi-signal and fail-toward-exclusion. ``src`` is the gate source; ``name`` its filename.
    """
    if REAL_WEIGHT_SENTINEL in src:
        return True
    if name.endswith(_REAL_WEIGHT_NAME_SUFFIX):
        return True
    if any(marker in src for marker in _REAL_WEIGHT_MARKERS):
        return True
    if "models" in src and any(idiom in src for idiom in _HOME_LOAD_IDIOMS):
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


# --- Identity-pinned partition manifest (the fail-open backstop) ------------------------------ #
# Pins the exact NAME SET of each bucket. Counts alone miss an offsetting add+remove (two changes
# that keep the count constant); a name set catches add / remove / bucket-move by identity. A
# real-weight gate that evades detection appears as a NEW name in `model_free` — loud, not silent.
MANIFEST_PATH = PARITY_DIR / "gate_manifest.json"
_MANIFEST_COMMENT = (
    "Pinned classification of every parity/*_test.py gate (stems). Regenerate with "
    "`uv run python -m parity.run_modelfree_sweep --update-manifest` after adding/removing/"
    "reclassifying a gate, and REVIEW the diff — a gate appearing in model_free will be SWEPT, so "
    "it must not load real weights. This is the fail-open backstop; do not hand-edit casually."
)


def current_partition() -> dict[str, list[str]]:
    """The live partition as sorted bare stems (no ``parity.`` prefix), bucketed by classification."""
    mf, rw = classify_all()
    strip = lambda mods: sorted(m.removeprefix("parity.") for m in mods)  # noqa: E731
    return {"model_free": strip(mf), "real_weight": strip(rw)}


def load_manifest() -> dict[str, list[str]]:
    """Read the pinned manifest. Fails loud (rule 6) if absent or malformed — a missing backstop
    must never silently pass."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"gate manifest missing: {MANIFEST_PATH}. Regenerate with "
            f"`uv run python -m parity.run_modelfree_sweep --update-manifest`."
        )
    data = json.loads(MANIFEST_PATH.read_text())
    if not isinstance(data.get("model_free"), list) or not isinstance(data.get("real_weight"), list):
        raise ValueError(f"malformed gate manifest {MANIFEST_PATH}: need list 'model_free' & "
                         f"'real_weight'.")
    return {"model_free": list(data["model_free"]), "real_weight": list(data["real_weight"])}


def manifest_diff() -> dict[str, list[str]]:
    """``{added, removed, moved}`` between the live partition and the pinned manifest. All-empty
    means the partition is exactly as pinned. ``moved`` = a gate that changed bucket."""
    cur, pin = current_partition(), load_manifest()
    cur_mf, cur_rw = set(cur["model_free"]), set(cur["real_weight"])
    pin_mf, pin_rw = set(pin["model_free"]), set(pin["real_weight"])
    cur_all, pin_all = cur_mf | cur_rw, pin_mf | pin_rw
    return {
        "added": sorted(cur_all - pin_all),
        "removed": sorted(pin_all - cur_all),
        "moved": sorted((cur_mf & pin_rw) | (cur_rw & pin_mf)),
    }


def write_manifest() -> dict[str, list[str]]:
    """Regenerate the pinned manifest from the live partition; returns it. The one supported way to
    update the backstop — forces the diff into the commit for review."""
    part = current_partition()
    payload = {"_comment": _MANIFEST_COMMENT, **part}
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return part


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
# Broad enough to catch pytest-style gates too: bare `assert`, `raise SystemExit/AssertionError`,
# unittest/`np.testing` assertions, and `def test_` functions (a gate with no `__main__`).
_ASSERT_RE = re.compile(
    r"^\s*assert\s|raise\s+SystemExit|raise\s+AssertionError|self\.assert|assertEqual|"
    r"np\.testing\.assert|npt\.assert|^\s*def\s+test_",
    re.MULTILINE,
)


def _looks_like_gate(src: str) -> bool:
    """A runnable assertion gate: a ``__main__`` entry or pytest-style ``def test_``, plus an
    assertion of some flavor."""
    runnable = "__main__" in src or re.search(r"^\s*def\s+test_", src, re.MULTILINE)
    return bool(runnable and _ASSERT_RE.search(src))


def scan_misnamed_gates() -> list[str]:
    """Basenames of non-``*_test.py`` parity files that smell like model-free assertion gates and
    are NOT in the documented allowlist. A non-empty result is a silent-coverage-gap: a gate that
    would never be swept because of its name. Fix by renaming it ``*_test.py`` or documenting it in
    ``_KNOWN_NON_GATES``."""
    flagged: list[str] = []
    for path in sorted(PARITY_DIR.glob("*.py")):
        name = path.name
        if name.endswith("_test.py") or name in _RUNNER_FILES or name in _KNOWN_NON_GATES:
            continue
        src = _read(path)
        if _looks_like_gate(src) and not is_real_weight(src, name):
            flagged.append(name)
    return flagged


def stale_allowlist_entries() -> list[str]:
    """Allowlisted ``_KNOWN_NON_GATES`` names whose file no longer exists (dead entries). Keeps the
    allowlist honest — a removed/renamed file should not leave a silent suppression behind."""
    return sorted(n for n in _KNOWN_NON_GATES if not (PARITY_DIR / n).exists())


# --- Gate execution --------------------------------------------------------------------------- #
# Baseline optional (extra) deps; the live set is read from pyproject so it never drifts (#8).
_OPTIONAL_DEP_BASELINE = ("safetensors", "transformers", "sentencepiece")
_REQ_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+")
_MISSING_DEP_RE = re.compile(r"No module named '([\w.]+)'")
_TRACEBACK_SIG = "Traceback (most recent call last):"
_CHECKS_RE = re.compile(r"PARITY-CHECKS:\s*(\d+)")


def optional_deps() -> frozenset[str]:
    """Import-name candidates for every dep declared under any ``[project.optional-dependencies]``
    extra in pyproject (so a new extra dep is skip-eligible automatically), unioned with the
    baseline so the set is never *narrower* than the known offline deps. Parse failure falls back to
    the baseline (never silently empty)."""
    names = set(_OPTIONAL_DEP_BASELINE)
    try:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        for reqs in data.get("project", {}).get("optional-dependencies", {}).values():
            for req in reqs:
                m = _REQ_NAME_RE.match(req.strip())
                if m:
                    dist = m.group(0)
                    names.add(dist)
                    names.add(dist.replace("-", "_"))  # crude dist→import (exact for our extras)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        pass  # baseline still applies — documented fallback, not a silent wrong result
    return frozenset(names)


def _missing_optional_dep(output: str, optional: frozenset[str] | None = None) -> str | None:
    """If a failed run is explained solely by an absent *optional* (extra) module, return its name
    (so the gate is SKIPPED, not failed). On a dev env with the extras installed, never fires."""
    opt = optional if optional is not None else optional_deps()
    for m in _MISSING_DEP_RE.finditer(output):
        top = m.group(1).split(".")[0]
        if top in opt:
            return top
    return None


def _suspect_reason(returncode: int, output: str) -> str:
    """Non-empty when a returncode-0 run shows evidence it did NOT pass cleanly — the vacuous-pass
    guard. A positive ``PARITY-CHECKS: <n>`` (n>0) is the opt-in proof-of-work contract: it is
    trusted outright (and is the escape hatch for a gate that legitimately renders a Traceback).
    ``PARITY-CHECKS: 0`` always fails. Absent the contract, a printed Traceback under rc-0 (a
    swallowed exception) fails. Clean assertion-erosion inside a still-green gate stays out of reach
    without the contract — which is exactly why the contract exists."""
    if returncode != 0:
        return ""  # already a failure; the returncode carries it
    m = _CHECKS_RE.search(output)
    if m:
        return "" if int(m.group(1)) > 0 else \
            "exited 0 but reported PARITY-CHECKS: 0 (no assertions ran)"
    if _TRACEBACK_SIG in output:
        return "exited 0 but printed a Traceback (swallowed exception)"
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


def run_gate(module: str, *, timeout: int = DEFAULT_TIMEOUT,
             optional: frozenset[str] | None = None) -> GateResult:
    """Run one gate as ``python -m <module>`` in an ISOLATED subprocess (fresh mlx state, no
    cross-gate contamination — the way the gates are designed to run). A timeout is reported as
    returncode 124, never raised, so a single hang can't abort a whole sweep. A returncode-0 run is
    additionally screened for vacuous-pass evidence, and a missing optional-extra dep yields a SKIP
    rather than a false failure. ``optional`` is the extra-dep set (computed once by the sweep)."""
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
        dep = _missing_optional_dep(out, optional)
        if dep is not None:
            return GateResult(module, proc.returncode, secs, _summary(out),
                              skipped=True,
                              skip_reason=f"missing optional dep '{dep}' (an extra)")
    return GateResult(module, proc.returncode, secs, _summary(out),
                      suspect_reason=_suspect_reason(proc.returncode, out))


def summary_banner() -> str:
    """One-line framing: what a green sweep does and does NOT prove (the false-confidence guard)."""
    mf, rw = classify_all()
    return (f"{len(mf)} model-free gates swept here (interface + logic on synthetic stubs); "
            f"{len(rw)} real-weight SOLO gates NOT run here — real-model numeric parity "
            f"(teacher-forced ppl / layer parity) lives there and stays a manual SOLO step.")
