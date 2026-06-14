"""Gate: the per-model batched-decode operating point (the throughput knee) is wired as the default
serving capacity — MODEL-FREE (fake artifact dirs, ~0 GB; no runtime load, no GPU).

The serving engine drives ``batched_stream_generate`` with ``capacity`` concurrent decode slots. When
the caller passes no explicit ``batch_size`` we default to the model's per-model operating point
(:data:`quanta.shim.omlx.BEST_BATCH`) — the measured throughput knee for a worker (DSV4 ⇒ 48 #19,
Nemotron ⇒ 32 #20, InternLM2.5 ⇒ 32 #21) or the latency-first low-B point for the orchestrator
(Qwen3.5 ⇒ 4 #26, NOT a measured knee) — falling back to the generic :data:`DEFAULT_BATCH_CAPACITY`
for any model with no declared point (GLM-5.1, MiniMax, Qwen2.5, Kimi/DSV3). This gates that wiring
without loading a model:

  1. **resolver** — ``_best_batch_for`` returns 48 for ``deepseek_v4*``, 32 for ``nemotron*`` and
     ``internlm2*``, 4 for ``qwen3_5*`` (incl. realistic suffixes), and ``None`` for every model with
     no declared point / unknown / empty model_type. Rule 6: a worker knee is never invented — only
     ones we benched appear; the Qwen3.5 point is an explicit latency choice, not a throughput claim.
  2. **operating values** — the hardcoded constants are exactly the declared points (48 / 32 / 32 / 4),
     guarding a typo or an accidental edit from silently changing the production batch size.
  3. **engine default capacity** — a fake-artifact engine resolves ``_default_capacity(n)`` to the
     operating point (clamped to ``n`` prompts on hand) for declared models, and to the generic
     fallback for the rest — the exact value ``batched_stream_generate`` uses when ``batch_size`` is absent.
  4. **no best-B without a batched runtime** — every prefix in ``BEST_BATCH`` is one that
     ``_make_batched_session`` actually dispatches to a batched session (so we never advertise a knee
     for a model that cannot batch).
  5. **hard batch cap** — Nemotron's knee is a CEILING, not just a default: ``_resolve_capacity`` clamps
     an explicit ``batch_size > 32`` DOWN to 32 (decode regresses past it — warned, rule 6), while
     DSV4 / Qwen3.5 HONOR an explicit over-knee B (``_hard_batch_cap`` is None — Qwen3.5's low-B is a
     latency pin, never a cap). Nemotron's default + under-knee batch are unaffected.

An explicit ``batch_size`` otherwise bypasses the default (benches/tests still pin B) — EXCEPT it is
clamped down to a model's hard ceiling where one exists (Nemotron's 32, item 5). Resolution now lives in
``_resolve_capacity``; the explicit-B path is exercised here and by every ``*_batched_*`` test.

    uv run python -m parity.omlx_best_batch_test
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

from quanta.shim.omlx import (
    BEST_BATCH,
    DEFAULT_BATCH_CAPACITY,
    SERVING_BATCH_CAP,
    QuantaOmlxEngine,
    _best_batch_for,
)


def _fake_artifact(model_type: str, *, nested: bool) -> str:
    """A synthetic quanta artifact dir holding just enough for ``detect_quanta_artifact`` to read the
    ``model_type`` (a few bytes; no weights). ``nested`` puts it under ``text_config`` (Qwen) vs
    top-level (DSV4/Nemotron/InternLM2)."""
    d = Path(tempfile.mkdtemp(prefix="omlxbestb_"))
    cfg = {"text_config": {"model_type": model_type}} if nested else {"model_type": model_type}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "manifest.json").write_text(json.dumps({"format": "quanta", "tensors": {}}))
    return str(d)


def _run_resolver() -> bool:
    """`_best_batch_for`: measured models resolve to their knee (any suffix); everything else None."""
    measured = {
        "deepseek_v4": 48, "deepseek_v4_pro": 48, "deepseek_v4_text": 48,
        "nemotron": 32, "nemotron_h": 32,
        "internlm2": 32, "internlm2_5": 32,                    # worker knee (#21); canonical + suffix
        "qwen3_5": 4, "qwen3_5_moe": 4, "qwen3_5_moe_text": 4,  # orchestrator point (#26); canonical
        #                                                        model_type is the underscore form
        "minimax_m3": 32, "minimax_m3_vl": 32,                 # M3-7b provisional point (= serving cap)
    }
    unmeasured = ("glm_moe_dsa", "minimax_m2", "qwen2", "deepseek", "kimi_k2", "", None)
    hit = all(_best_batch_for(mt) == b for mt, b in measured.items())
    miss = all(_best_batch_for(mt) is None for mt in unmeasured)
    ok = hit and miss
    print(f"  [{'OK' if ok else 'FAIL'}] resolver: declared→point={hit}  undeclared/unknown→None={miss}")
    return ok


def _run_knee_values() -> bool:
    """The hardcoded constants are exactly the declared operating points (typo/accidental-edit guard)."""
    table = dict(BEST_BATCH)
    ok = (table.get("deepseek_v4") == 48 and table.get("nemotron") == 32
          and table.get("internlm2") == 32 and table.get("qwen3_5") == 4
          and table.get("minimax_m3") == 32 and len(BEST_BATCH) == 5
          and SERVING_BATCH_CAP == 32)
    print(f"  [{'OK' if ok else 'FAIL'}] operating values: deepseek_v4=48, nemotron=32, internlm2=32, "
          f"qwen3_5=4, minimax_m3=32, fallback={DEFAULT_BATCH_CAPACITY}, serving_cap={SERVING_BATCH_CAP}, "
          f"n_declared={len(BEST_BATCH)}")
    return ok


def _run_engine_default_capacity() -> bool:
    """`_default_capacity(n)` on a fake-artifact engine == the value the serving default resolves to:
    the declared operating point (worker knee or orchestrator low-B), clamped to the uniform serving
    cap (DSV4 knee 48 → 32) and to prompts on hand, for declared models; the generic fallback for an
    undeclared one (GLM here)."""
    eng_dsv4 = QuantaOmlxEngine(_fake_artifact("deepseek_v4", nested=False))
    eng_nemo = QuantaOmlxEngine(_fake_artifact("nemotron_h", nested=False))
    eng_intern = QuantaOmlxEngine(_fake_artifact("internlm2", nested=False))
    eng_qwen = QuantaOmlxEngine(_fake_artifact("qwen3_5_moe_text", nested=True))
    eng_glm = QuantaOmlxEngine(_fake_artifact("glm_moe_dsa", nested=False))
    eng_m3 = QuantaOmlxEngine(_fake_artifact("minimax_m3_vl", nested=False))
    checks = {
        "dsv4 @100→32 (serving cap; knee 48)": eng_dsv4._default_capacity(100) == 32,
        "dsv4 @4→4 (prompt clamp)": eng_dsv4._default_capacity(4) == 4,
        "nemotron @100→32": eng_nemo._default_capacity(100) == 32,
        "nemotron @16→16 (clamp)": eng_nemo._default_capacity(16) == 16,
        "internlm2 @100→32": eng_intern._default_capacity(100) == 32,
        "qwen3_5 @100→4 (orchestrator)": eng_qwen._default_capacity(100) == 4,
        "qwen3_5 @1→1 (single-stream clamp)": eng_qwen._default_capacity(1) == 1,
        "glm @100→8 (fallback)": eng_glm._default_capacity(100) == DEFAULT_BATCH_CAPACITY,
        "minimax_m3 @100→32 (serving cap)": eng_m3._default_capacity(100) == 32,
    }
    ok = all(checks.values())
    bad = [k for k, v in checks.items() if not v]
    print(f"  [{'OK' if ok else 'FAIL'}] engine _default_capacity: " +
          ("all 9 correct" if ok else f"WRONG: {bad}"))
    return ok


def _run_hard_batch_cap() -> bool:
    """Every throughput WORKER's batch is HARD-capped at the uniform :data:`SERVING_BATCH_CAP` (32):
    ``_resolve_capacity`` clamps an explicit over-cap ``batch_size`` DOWN to 32 for Nemotron, InternLM2.5
    AND DSV4 — DSV4's benched knee is 48 but serving is held at 32 (decoupled from BEST_BATCH). The
    Qwen3.5 ORCHESTRATOR is EXEMPT: its B=4 latency pin is a soft default, an over-knee batch is HONORED
    (32), only prompt-clamped; ``_hard_batch_cap`` is None for it. Under-cap batches pass through."""
    eng_nemo = QuantaOmlxEngine(_fake_artifact("nemotron_h", nested=False))
    eng_intern = QuantaOmlxEngine(_fake_artifact("internlm2", nested=False))
    eng_dsv4 = QuantaOmlxEngine(_fake_artifact("deepseek_v4", nested=False))
    eng_qwen = QuantaOmlxEngine(_fake_artifact("qwen3_5_moe_text", nested=True))
    eng_m3 = QuantaOmlxEngine(_fake_artifact("minimax_m3_vl", nested=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             # the over-cap clamp warns by design (rule 6)
        checks = {
            "nemotron _hard_batch_cap==32": eng_nemo._hard_batch_cap() == 32,
            "nemotron explicit 48 → 32 (HARD cap)": eng_nemo._resolve_capacity(100, 48) == 32,
            "nemotron explicit 16 → 16 (under cap, honored)": eng_nemo._resolve_capacity(100, 16) == 16,
            "nemotron default(None) → 32": eng_nemo._resolve_capacity(100, None) == 32,
            "nemotron cap clamps to prompts (48→32→4)": eng_nemo._resolve_capacity(4, 48) == 4,
            "internlm2 _hard_batch_cap==32": eng_intern._hard_batch_cap() == 32,
            "internlm2 explicit 48 → 32 (HARD cap)": eng_intern._resolve_capacity(100, 48) == 32,
            "internlm2 default(None) → 32": eng_intern._resolve_capacity(100, None) == 32,
            "dsv4 _hard_batch_cap==32 (knee 48, serving-capped)": eng_dsv4._hard_batch_cap() == 32,
            "dsv4 explicit 64 → 32 (HARD cap, was knee 48)": eng_dsv4._resolve_capacity(100, 64) == 32,
            "dsv4 default(None) → 32 (capped from knee 48)": eng_dsv4._resolve_capacity(100, None) == 32,
            "qwen3_5 _hard_batch_cap is None (orchestrator pin)": eng_qwen._hard_batch_cap() is None,
            "qwen3_5 explicit 32 → 32 (NOT capped)": eng_qwen._resolve_capacity(100, 32) == 32,
            "minimax_m3 _hard_batch_cap==32 (serving cap)": eng_m3._hard_batch_cap() == 32,
            "minimax_m3 explicit 64 → 32 (HARD cap)": eng_m3._resolve_capacity(100, 64) == 32,
            "minimax_m3 default(None) → 32": eng_m3._resolve_capacity(100, None) == 32,
        }
    ok = all(checks.values())
    bad = [k for k, v in checks.items() if not v]
    print(f"  [{'OK' if ok else 'FAIL'}] hard batch cap: " +
          ("nemotron/internlm2/dsv4 clamp >32→32; qwen3_5 orchestrator honors explicit B"
           if ok else f"WRONG: {bad}"))
    return ok


def _run_no_orphan_knee() -> bool:
    """Every BEST_BATCH prefix is a model ``_make_batched_session`` dispatches to a batched session —
    we never declare a knee for a model that cannot batch. (DSV4/Nemotron/InternLM2.5/Qwen3.5/MiniMax-M3
    are the five batched classes; the measured/provisional points must be a subset.)"""
    batchable = ("deepseek_v4", "nemotron", "qwen3_5", "qwen3.5", "internlm2", "minimax_m3")
    ok = all(any(prefix.startswith(b) or b.startswith(prefix) for b in batchable)
             for prefix, _ in BEST_BATCH)
    print(f"  [{'OK' if ok else 'FAIL'}] no orphan knee: every BEST_BATCH prefix has a batched runtime")
    return ok


def run() -> None:
    ok = True
    print("\n=== oMLX per-model best-B operating point (throughput knee → default capacity) ===")
    ok &= _run_resolver()
    ok &= _run_knee_values()
    ok &= _run_engine_default_capacity()
    ok &= _run_hard_batch_cap()
    ok &= _run_no_orphan_knee()
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    run()
