"""Model-free gate: InternLM2.5 EAGLE-3 spec-decode wired into the oMLX shim's spec dispatch.

The shim's ``QuantaOmlxEngine._dispatch_spec_k`` routes a ``spec_k>1`` single-stream request to the
right per-model speculative path — native MTP for DSV4/Nemotron, and (wired here) the **EAGLE-3 trained
drafter** for InternLM2.5, the only serving keeper whose spec is a drafter rather than native MTP.
EAGLE is **lossless**: the emitted stream is bit-identical to plain greedy regardless of drafter quality
(the target's own verify arbitrates; the drafter only changes *speed*).

This gate proves the SHIM PLUMBING on a tiny random-init model — NO checkpoint, NO GPU, NO artifact on
disk — reusing the same toy model + untrained drafter the EAGLE adapter gate
(:mod:`parity.internlm2_eagle_spec_test`) uses:

  A. **shim spec == greedy** — the engine's ``_dispatch_spec_k`` (model_type ``internlm2``, ``spec_k>1``)
     reproduces plain greedy token-for-token, for several ``k`` (the EAGLE guarantee, now through the
     shim entry point rather than the adapter directly).
  B. **injected-state precedence** — an ``eagle=`` ctor injection is returned by ``_ensure_eagle``
     verbatim (tests / advanced callers bypass the artifact-sidecar load).
  C. **missing sidecar fails loud** — no injected state + no embedded ``eagle/`` sidecar ⇒ a clear
     ``OmlxShimError`` naming ``embed_eagle`` (rule 6 — never silently downgrade a spec request).

Real-weight losslessness + the 1.42× speedup at the int4-PTQ serving operating point are the separate
solo bench :mod:`parity.internlm2_eagle_spec_bench`.

    uv run python -m parity.internlm2_omlx_eagle_test
"""

from __future__ import annotations

from quanta.shim.omlx import OmlxShimError, QuantaOmlxEngine
from parity.internlm2_eagle_spec_test import (
    _MAX_NEW,
    _PROMPT,
    _TINY_LAYERS,
    _build_bf16,
    _build_packed,
    _greedy,
    _resident,
    _tiny_cfg,
    _tiny_drafter,
)


class _IL2Engine(QuantaOmlxEngine):
    """``QuantaOmlxEngine`` pinned to model_type ``internlm2`` so the EAGLE spec branch fires without a
    real artifact on disk (mirrors the nemotron engine test's model_type-override subclass)."""

    @property
    def model_type(self) -> str:
        return "internlm2"


def _build():
    """A tiny packed InternLM2 resident (production-shaped) + an untrained drafter + frozen embed/head."""
    cfg = _tiny_cfg()
    rm = _resident(_build_packed(_build_bf16(cfg), cfg), cfg)
    drafter = _tiny_drafter(cfg)
    embed, head = rm.embed_head()
    return rm, drafter, embed, head


def _engine(rm, drafter, embed, head, *, eagle: bool = True) -> _IL2Engine:
    state = (drafter, embed, head, _TINY_LAYERS, None) if eagle else None
    return _IL2Engine("internlm2-fake-root", runtime=rm, tokenizer=None, eagle=state)


def test_shim_spec_eq_greedy() -> None:
    rm, drafter, embed, head = _build()
    eng = _engine(rm, drafter, embed, head)
    greedy = _greedy(rm, _PROMPT, _MAX_NEW)
    for k in (2, 4):
        spec = eng._dispatch_spec_k(prompt_ids=list(_PROMPT), spec_k=k, max_new=_MAX_NEW, eos_id=None)
        assert spec == greedy, f"shim EAGLE spec(k={k}) != greedy:\n  spec  ={spec}\n  greedy={greedy}"
    print(f"A shim spec==greedy: {len(greedy)} toks, k in {{2,4}} (lossless via _dispatch_spec_k)  ok")


def test_injected_eagle_precedence() -> None:
    rm, drafter, embed, head = _build()
    eng = _engine(rm, drafter, embed, head)
    st = eng._ensure_eagle()
    assert st[0] is drafter and st[1] is embed and st[2] is head and st[3] == _TINY_LAYERS \
        and st[4] is None, "injected eagle state must be returned verbatim"
    print("B injected eagle state used verbatim (no load_eagle disk access)  ok")


def test_no_sidecar_fails_loud() -> None:
    rm, drafter, embed, head = _build()
    eng = _engine(rm, drafter, embed, head, eagle=False)   # no injected state; fake root has no eagle/
    raised = False
    try:
        eng._ensure_eagle()
    except OmlxShimError as e:
        raised = "embed_eagle" in str(e)
    assert raised, "expected OmlxShimError naming embed_eagle when no drafter sidecar is present"
    print("C missing drafter sidecar fails loud (names embed_eagle)  ok")


def run() -> None:
    test_shim_spec_eq_greedy()
    test_injected_eagle_precedence()
    test_no_sidecar_fails_loud()
    print("PASS — InternLM2.5 EAGLE-3 spec-decode wired into the oMLX shim (spec==greedy via dispatch, "
          "injected-state precedence, missing-sidecar fail-loud)")


if __name__ == "__main__":
    run()
