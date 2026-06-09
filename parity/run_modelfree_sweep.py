"""Standalone sweep: run every MODEL-FREE parity gate and report a pass/fail/skip matrix.

The committed, reusable form of the throwaway hygiene driver that first caught two silently-rotted
gates (``dsv4_tree_spec_test``, ``qwen35_omlx_engine_test`` — stub signatures drifting from the real
interfaces, invisible because nothing exercised the ``parity/`` gates in aggregate). Run it any time
as a fast pre-commit / periodic health check of the model-free gate suite::

    uv run python -m parity.run_modelfree_sweep                 # sequential, streaming
    uv run python -m parity.run_modelfree_sweep --jobs 6        # concurrent (model-free ⇒ safe)
    uv run python -m parity.run_modelfree_sweep --strict-skips  # CI-grade: fail if any gate SKIPs
    uv run python -m parity.run_modelfree_sweep --update-manifest   # regenerate the pinned manifest

Exits nonzero iff any gate FAILED (a SKIP — a gate needing an offline extra on a base-deps-only env
— is not a failure unless ``--strict-skips``), so it doubles as a CI step. Real-weight (SOLO, 9-306
GiB) gates are excluded by construction (see :mod:`parity._modelfree`); this never loads a resident
model. A green run proves interface + logic on synthetic stubs, NOT real-model numeric parity —
that lives in the excluded SOLO gates. For the pytest-integrated form see
``tests/test_parity_modelfree.py``; both share ``parity._modelfree`` so they can't disagree on the
gate set or how a gate is run.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from parity._modelfree import (
    DEFAULT_TIMEOUT,
    GateResult,
    discover_model_free_gates,
    manifest_diff,
    optional_deps,
    run_gate,
    summary_banner,
    write_manifest,
)


def _update_manifest() -> int:
    """Regenerate the pinned partition manifest and print the diff it resolves."""
    before = manifest_diff()
    part = write_manifest()
    changed = {k: v for k, v in before.items() if v}
    print(f"[manifest] wrote {len(part['model_free'])} model_free / "
          f"{len(part['real_weight'])} real_weight gates", flush=True)
    if changed:
        print(f"[manifest] resolved diff: {changed}", flush=True)
        if changed.get("added"):
            print("[manifest] REVIEW: newly-swept (model_free) gates must NOT load real weights — "
                  "a real-weight gate evading detection would appear here.", flush=True)
    else:
        print("[manifest] no change (already in sync)", flush=True)
    return 0


def _print_row(i: int, n: int, r: GateResult) -> None:
    if r.skipped:
        tag = "SKIP"
    elif r.ok:
        tag = "PASS"
    else:
        tag = f"FAIL(rc{r.returncode})"
    detail = r.skip_reason or r.suspect_reason or r.summary
    print(f"[{i:>3}/{n}] {tag:>10}  {r.module:<46} {r.seconds:5.1f}s | {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all model-free parity gates.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent gates (default 1; model-free gates are safe to parallelize)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"per-gate timeout in seconds (default {DEFAULT_TIMEOUT})")
    ap.add_argument("--strict-skips", action="store_true",
                    help="treat a SKIP (missing offline extra) as a failure — CI installs the "
                         "extras (`uv sync --extra reference`), so a skip there means missing "
                         "coverage, not an expected base-deps state")
    ap.add_argument("--update-manifest", action="store_true",
                    help="regenerate parity/gate_manifest.json from the live partition and exit")
    args = ap.parse_args()

    if args.update_manifest:
        return _update_manifest()

    gates = discover_model_free_gates()
    optional = optional_deps()  # computed once (reads pyproject), shared by every gate run
    print(f"[sweep] {summary_banner()}", flush=True)
    print(f"[sweep] running {len(gates)} model-free gates "
          f"(jobs={args.jobs}, timeout={args.timeout}s, strict_skips={args.strict_skips})\n",
          flush=True)

    def _run(mod: str) -> GateResult:
        return run_gate(mod, timeout=args.timeout, optional=optional)

    results: list[GateResult] = []
    if args.jobs <= 1:
        for i, mod in enumerate(gates, 1):
            r = _run(mod)
            results.append(r)
            _print_row(i, len(gates), r)
    else:
        # ThreadPoolExecutor.map yields in submission order; each gate is its own OS process, so the
        # threads only block on subprocess IO — true parallelism despite the GIL. Safe to fan out:
        # model-free gates allocate KB-MB, so N concurrent copies stay far under any ceiling.
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(_run, gates), 1):
                results.append(r)
                _print_row(i, len(gates), r)

    skips = [r for r in results if r.skipped]
    fails = [r for r in results if r.failed]
    passes = [r for r in results if r.ok]
    print(f"\n[sweep] {len(passes)} pass / {len(fails)} fail / {len(skips)} skip "
          f"of {len(results)}", flush=True)
    if skips:
        verb = "FAIL (--strict-skips)" if args.strict_skips else "SKIPPED"
        print(f"\n{verb} — missing offline extra; run `uv sync --extra reference` for full "
              f"coverage:", flush=True)
        for r in skips:
            print(f"  SKIP  {r.module}  | {r.skip_reason}", flush=True)
    if fails:
        print("\nFAILURES:", flush=True)
        for r in fails:
            print(f"  FAIL(rc{r.returncode})  {r.module}  | {r.suspect_reason or r.summary}",
                  flush=True)
    return 1 if (fails or (args.strict_skips and skips)) else 0


if __name__ == "__main__":
    sys.exit(main())
