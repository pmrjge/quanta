"""Generation coherence + oMLX shim smoke for the int8 InternLM2.5-7B-Chat-1M artifact.

(1) Direct: the packed resident runtime + ``generate`` over a chat-templated prompt (greedy) must
    produce coherent English — the actual motivation for the InternLM2.5 pivot (the Qwen2.5-1M
    long-context runtime produced degenerate output; this must not).
(2) Shim: ``QuantaOmlxEngine(artifact)`` must route ``model_type='internlm2'`` through the wiring
    added to ``_default_runtime_loader`` / ``_make_stepper`` and emit non-empty coherent text — the
    end-to-end serving path (loader → runtime → single-token stepper → tokenizer detok).

Heavy (loads the int8 artifact); run in a GPU/memory session:

    uv run --with numpy python -m parity.internlm2_generate_smoke

NOTE — the full >262144-token dynamic-NTK long-context stress is a separate heavy benchmark (a 262K
prefill is O(T^2) and minutes-long), deliberately not run here. Its correctness is already pinned by
the exact (0.0-error) NTK-formula + chunked-prefill-offset checks in internlm2_remaining_risk_test
plus the parity-correct forward (internlm2_bf16_ppl / internlm2_packed_ppl). To run it manually:
feed a >262144-token prompt to generate() and confirm coherent continuation (NTK base auto-engages).
"""

from __future__ import annotations

import asyncio

from quanta.internlm2.generate import generate
from quanta.internlm2.runtime import InternLM2ResidentModel
from quanta.internlm2.tokenizer import InternLM2Tokenizer

ARTIFACT = "/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64"
PROMPT = "What is the capital of France? Answer in one short sentence."


def _coherent(text: str) -> bool:
    """Cheap degeneracy check: non-empty, has whitespace-separated words, not a single repeated token."""
    words = text.split()
    return len(words) >= 3 and len(set(words)) >= 3


def _direct(rt: InternLM2ResidentModel, tok: InternLM2Tokenizer) -> bool:
    prompt_ids = tok.encode_chat([{"role": "user", "content": PROMPT}], add_generation_prompt=True)
    out = generate(rt, prompt_ids, max_new_tokens=48, temperature=0.0)   # greedy
    text = tok.decode(out, skip_special_tokens=True)
    ok = _coherent(text)
    print(f"[direct]  {len(out)} toks  coherent={ok}\n  {text!r}\n")
    return ok


def _shim() -> bool:
    from quanta.shim.omlx import QuantaOmlxEngine

    engine = QuantaOmlxEngine(ARTIFACT)
    assert engine.model_type == "internlm2", f"shim routed model_type={engine.model_type!r}"
    out = asyncio.run(engine.generate(PROMPT, max_tokens=48, temperature=0.0, add_bos=True))
    text = getattr(out, "text", str(out))
    ok = _coherent(text)
    print(f"[shim]    model_type={engine.model_type}  coherent={ok}\n  {text!r}\n")
    return ok


def run() -> None:
    tok = InternLM2Tokenizer.from_pretrained(ARTIFACT)
    rt = InternLM2ResidentModel(ARTIFACT, packed=True)
    d = _direct(rt, tok)
    del rt
    import mlx.core as mx
    mx.clear_cache()
    s = _shim()
    print("PASS" if (d and s) else "FAIL")
    assert d and s


if __name__ == "__main__":
    run()
