"""A/B the GatedDeltaNet recurrence form on real Qwen3.6 weights — one model load, no reload.

The forward is broken (0/32 prefill teacher-forced top-1). The GatedDeltaNet (30/40 layers) flags its
core recurrence as an unconfirmed assumption: ``DELTA_RULE=True`` (u = v − Sᵀk delta correction) vs
``False`` (plain gated-linear S ← g·S + β·k⊗v). Both keep scan==chunk==decode, so no model-free gate
distinguishes them — only real weights + a ground-truth signal (teacher-forced next-token) can. This
loads the bf16-dequant reference once and measures prefill teacher-forced top-1 under each form by
monkeypatching the module global between forwards (the recurrence reads it at call time).

A jump from ~0 to high pins the recurrence form. If BOTH stay ~0 the bug is elsewhere in the GDN
(conv / a-discretization / gated-norm / qkv split) or another component — narrowing the search.

~65 GB resident — run SOLO.

    uv run python -u -m parity.qwen36_gdn_probe
"""

from __future__ import annotations

import mlx.core as mx

import quanta.qwen35.gated_deltanet as gdn
from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
TEXT = ("The quick brown fox jumps over the lazy dog. The capital of France is Paris, and the "
        "capital of Japan is Tokyo. Water is made of hydrogen and oxygen.")


def _agree(model, ids) -> float:
    lg = model(mx.array(ids))
    mx.eval(lg)
    pred = mx.argmax(lg[0, :-1], axis=-1)
    return float(mx.mean((pred == mx.array(ids[1:])).astype(mx.float32)).item())


def run() -> None:
    mx.set_wired_limit(int(120 * 1024 ** 3))
    model = Qwen35ResidentModel(ART)
    tok = Qwen35Tokenizer.from_pretrained(ART)
    ids = tok.encode(TEXT, add_bos=False)
    print(f"loaded; text={len(ids)} tokens, resident≈{mx.get_active_memory()/1024**3:.1f} GiB", flush=True)

    for flag in (True, False):
        gdn.DELTA_RULE = flag                                 # recurrence reads this at call time
        a = _agree(model, ids)
        print(f"  DELTA_RULE={flag!s:5}  prefill teacher-forced top-1 = {a:.3f}  "
              f"({int(a*(len(ids)-1))}/{len(ids)-1})", flush=True)
    gdn.DELTA_RULE = True                                     # restore default


if __name__ == "__main__":
    run()
