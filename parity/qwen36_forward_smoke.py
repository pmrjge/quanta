"""First real-weights forward smoke for the baked Qwen3.6-35B-A3B artifact (bf16-dequant reference).

Loads :class:`quanta.qwen35.runtime.Qwen35ResidentModel` (which dequantizes the baked int4/int8 weights
back to bf16 and runs the proven naive ``Qwen35Block`` forward — the parity REFERENCE, not yet the packed
gather_qmm path) + the self-contained tokenizer, then greedy-decodes a couple of trivial prompts. This
is the FIRST exercise of the full runtime forward (GatedDeltaNet + gated-GQA + partial-mRoPE + sparse
MoE routing) on real Qwen3.6 weights — a coherence smell test, NOT the teacher-forced-ppl arbiter.
Reasoning models can loop under greedy regardless of quant (project memory), so this only asserts the
output is non-empty + finite + decodes to text; coherence is eyeballed.

~70 GB resident (bf16 dequant) — run SOLO on the M3 Ultra.

    uv run python -u -m parity.qwen36_forward_smoke
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.generate import generate
from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
PROMPTS = (
    "What is the capital of France? Answer in one word.",
    "Count from 1 to 5.",
)


def run() -> None:
    mx.set_wired_limit(int(120 * 1024 ** 3))
    print(f"loading bf16-dequant reference runtime from {ART} ...", flush=True)
    model = Qwen35ResidentModel(ART)
    tok = Qwen35Tokenizer.from_pretrained(ART)
    cfg = model.cfg
    eos = tuple(cfg.eos_token_ids)
    print(f"loaded: {model.num_layers} layers, experts={cfg.num_experts} top-{cfg.num_experts_per_tok}, "
          f"resident≈{mx.get_active_memory() / 1024 ** 3:.1f} GiB, eos={eos}", flush=True)

    ok = True
    for q in PROMPTS:
        ids = tok.encode_chat([{"role": "user", "content": q}], add_generation_prompt=True)
        out = generate(model, ids, max_new_tokens=64, temperature=0.0, eos_id=eos)
        text = tok.decode(out, skip_special_tokens=True)
        good = len(out) > 0 and isinstance(text, str) and len(text.strip()) > 0
        ok = ok and good
        print(f"\n  [{'OK' if good else 'XX'}] prompt={q!r}", flush=True)
        print(f"      prompt_ids={len(ids)}  gen_ids={len(out)}", flush=True)
        print(f"      OUT: {text!r}", flush=True)

    print(f"\n{'SMOKE PASS' if ok else 'SMOKE FAIL'} (coherence eyeballed; ppl is the real arbiter)",
          flush=True)
    assert ok


if __name__ == "__main__":
    run()
