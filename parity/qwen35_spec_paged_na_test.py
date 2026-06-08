"""Model-free gate: tree-spec-over-paged is N/A for Qwen3.5 — it has no paged cache (#158-160 M2).

#158-160 makes an EXISTING paged cache satisfy the batched tree-spec ``replicate(B)`` contract so a
*paged* request can run tree-spec verify. That presupposes the model HAS a paged cache. Qwen3.5 does
NOT: its production serving is **unpaged** (the omlx shim builds the Qwen3.5 batched session with
``paged_kv=False``; the #153 loop-kill is the UNPAGED "option-B" path), and the batched runtime exposes
no paged contract (no ``make_paged_state`` / ``paged_kv_spec``). So there is no paged cache to make
replicate-able — the #158-160 cache-half is **inapplicable by construction** for Qwen3.5, and tree-spec
already runs over its DISCRETE :class:`~quanta.qwen35.decode.Qwen35Cache` (gated end-to-end in
``parity/qwen35_batched_tree_verify_test.py``).

This gate PINS that architectural fact so it can't drift silently:

  A. **paged-contract discriminant** — :class:`~quanta.dsv4.batched_runtime.DSV4BatchedResidentModel`
     and :class:`~quanta.nemotron.batched_runtime.NemotronBatchedResidentModel` both expose the paged
     contract (``make_paged_state`` + ``paged_kv_spec``); :class:`~quanta.qwen35.batched_runtime.
     Qwen35BatchedResidentModel` exposes NEITHER. (If someone later adds Qwen3.5 paging this FAILS,
     prompting them to wire tree-spec-over-paged for it or update this N/A rationale — rule 6, no
     silent drift.)
  B. **discrete tree-spec is available** — Qwen3.5 HAS the algorithm (``spec_generate_tree``) and its
     discrete ``Qwen35Cache`` satisfies the ``replicate(B)`` cache contract tree-spec needs (structural
     check: replicate returns B distinct caches at the same offset; the behavioral bit-identity is the
     job of ``qwen35_batched_tree_verify_test``). So Qwen3.5 tree-spec works — just over the only cache
     it has (discrete), not a paged one.

    uv run python -m parity.qwen35_spec_paged_na_test
"""

from __future__ import annotations

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.qwen35.batched_runtime import Qwen35BatchedResidentModel
from quanta.qwen35.decode import Qwen35Cache
from quanta.qwen35.spec import spec_generate_tree

PAGED_CONTRACT = ("make_paged_state", "paged_kv_spec")


def test_paged_contract_discriminant() -> None:
    """DSV4 + Nemotron expose the paged contract; Qwen3.5 exposes NEITHER method."""
    for cls in (DSV4BatchedResidentModel, NemotronBatchedResidentModel):
        for attr in PAGED_CONTRACT:
            assert hasattr(cls, attr), f"{cls.__name__} must expose paged-contract method {attr!r}"
    for attr in PAGED_CONTRACT:
        assert not hasattr(Qwen35BatchedResidentModel, attr), (
            f"Qwen35BatchedResidentModel unexpectedly exposes {attr!r} — Qwen3.5 gained a paged cache; "
            f"tree-spec-over-paged is no longer N/A, wire it (#158-160) or update this rationale.")
    print("[OK] paged-contract discriminant: DSV4 + Nemotron paged; Qwen3.5 has no make_paged_state/"
          "paged_kv_spec (no paged cache to make replicate-able)")


def test_discrete_tree_spec_available() -> None:
    """Qwen3.5 has the tree-spec algorithm + its discrete Qwen35Cache satisfies replicate(B)."""
    assert callable(spec_generate_tree), "qwen35.spec.spec_generate_tree must exist (the algorithm)"
    cache = Qwen35Cache(2, layer_is_linear=lambda i: False, max_rollback=3)   # 2 full-attn layers
    assert hasattr(cache, "replicate"), "Qwen35Cache must expose replicate(B) (tree-spec cache contract)"
    reps = cache.replicate(3)
    assert len(reps) == 3 and len({id(r) for r in reps}) == 3, "replicate(B) must return B distinct caches"
    assert all(r.offset == cache.offset == 0 and len(r) == len(cache) for r in reps), (
        "each replica starts at the source offset with the same layer count")
    print("[OK] discrete tree-spec available: spec_generate_tree + Qwen35Cache.replicate(B) "
          "(behavioral bit-identity gated by qwen35_batched_tree_verify_test)")


def main() -> int:
    tests = [test_paged_contract_discriminant, test_discrete_tree_spec_available]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        return 1
    print("PASS — tree-spec-over-paged is N/A for Qwen3.5 (no paged cache; serving is unpaged); "
          "tree-spec runs over its discrete cache (#158-160 M2: documented + pinned N/A)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
