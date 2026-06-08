"""Discovery + isolated-subprocess runner for the MODEL-FREE parity gates.

Shared by the pytest collector (``tests/test_parity_modelfree.py``) and the standalone sweep
(``parity.run_modelfree_sweep``) so the two can never disagree on WHICH gates are model-free or HOW
they are run.

A parity gate is *model-free* when its source references NEITHER a ``/Users/pmrj/models`` artifact
path NOR :func:`mx.set_wired_limit` — i.e. it allocates only a few KB-MB of stub tensors and is safe
to run anywhere, anytime (rule 8: no large allocations). The *real-weight* gates (which load 9-306
GiB resident and must run SOLO, one model at a time) are excluded by exactly this heuristic; erring
toward exclusion is deliberate — CI must never spin up a 306 GiB load.

This is the runner the ``parity/`` gates never had — the gap that let ``dsv4_tree_spec_test`` and
``qwen35_omlx_engine_test`` silently rot (stub signatures drifting from the real interfaces they
stand in for) across many commits, invisible because nothing exercised the gates in aggregate.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PARITY_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARITY_DIR.parent
DEFAULT_TIMEOUT = 300

# A gate that loads real resident weights names a models artifact path or pins the wired limit.
_REAL_WEIGHT_MARKERS = ("/Users/pmrj/models", "set_wired_limit")


def is_real_weight(src: str) -> bool:
    """True when a gate's SOURCE shows it loads real resident weights (SOLO-only ⇒ excluded)."""
    return any(marker in src for marker in _REAL_WEIGHT_MARKERS)


def discover_model_free_gates() -> list[str]:
    """Sorted ``parity.<stem>`` module names of every model-free ``parity/*_test.py`` gate.

    Filesystem-only (no imports) so it is safe at pytest collection time. New gates are picked up
    automatically — the whole point: a gate added tomorrow is swept with no edit here.
    """
    mods: list[str] = []
    for path in sorted(PARITY_DIR.glob("*_test.py")):
        try:
            src = path.read_text()
        except OSError:
            continue
        if is_real_weight(src):
            continue
        mods.append(f"parity.{path.stem}")
    return mods


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate run."""

    module: str
    returncode: int
    seconds: float
    summary: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _summary(out: str) -> str:
    """The last 'interesting' line of a gate's output (its PASS/FAIL/error banner), truncated."""
    keys = ("tests passed", "pass", "fail", "error", "traceback", "exception",
            "assert", "systemexit", "timeout")
    last = ""
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in keys):
            last = line.strip()
    return last[:120]


def run_gate(module: str, *, timeout: int = DEFAULT_TIMEOUT) -> GateResult:
    """Run one gate as ``python -m <module>`` in an ISOLATED subprocess (fresh mlx state, no
    cross-gate contamination — the way the gates are designed to run). A timeout is reported as
    returncode 124, never raised, so a single hang can't abort a whole sweep.
    """
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
        return GateResult(module, proc.returncode, time.time() - t0,
                          _summary(proc.stdout + proc.stderr))
    except subprocess.TimeoutExpired:
        return GateResult(module, 124, time.time() - t0, f"TIMEOUT >{timeout}s")
