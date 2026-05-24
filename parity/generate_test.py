"""Bounded end-to-end check of the decode loop (8 layers, few tokens — memory-safe).

Greedy decode via ``generate`` (KV-cached, absorbed-MLA steps) must agree with a one-shot
teacher-forced recompute of the same prompt+continuation: at each generated position the
teacher-forced argmax should be the token the loop produced. (Composes the absorbed-MLA
and KV-cache equivalences; depth is irrelevant to the mechanics, so 8 layers suffices.)
Disagreements are only acceptable as exact fp ties — reported via the logit gap.

    uv run --with tiktoken python -m parity.generate_test
"""

from __future__ import annotations

import mlx.core as mx

from quanta.config import KimiTextConfig
from quanta.generate import generate
from quanta.loader import SourceCheckpoint
from quanta.model import KimiModel
from quanta.tokenizer import KimiTokenizer

MODEL = "/Users/pmrj/models/Kimi-K2.6"
N_LAYERS = 8
K = 4


def run() -> None:
    cfg = KimiTextConfig.from_pretrained(MODEL)
    model = KimiModel(cfg, SourceCheckpoint(MODEL), mx.bfloat16)
    tok = KimiTokenizer(MODEL, bos_id=cfg.bos_token_id)

    prompt = tok.encode("The capital of France is", add_bos=True)
    p = len(prompt)
    print(f"\n=== decode loop check (layers={N_LAYERS}, prompt={p} tok, gen={K}) ===")

    for absorbed in (False, True):  # expanded isolates loop mechanics; absorbed is the decode path
        gen = generate(model, prompt, max_new_tokens=K, n_layers=N_LAYERS, temperature=0.0,
                       sparse=None, decode_absorbed=absorbed, quantized_kv=False)  # bf16 KV: this gates exactness
        lf = model(mx.array(prompt + gen), n_layers=N_LAYERS, sparse=None)[0].astype(mx.float32)
        agree, max_gap = 0, 0.0
        for i, tid in enumerate(gen):
            row = lf[p - 1 + i]  # logits that predict the i-th generated token
            agree += int(int(mx.argmax(row).item()) == tid)
            max_gap = max(max_gap, (mx.max(row) - row[tid]).item())  # 0 ⇒ exact match to one-shot
        path = "absorbed" if absorbed else "expanded"
        print(f"  decode={path:8s} ids={gen}  agree {agree}/{K}  max logit gap {max_gap:.4f}")
    print("expanded ~0 gap ⇒ loop+cache exact; absorbed gap is bf16 absorbed-vs-expanded drift")


if __name__ == "__main__":
    run()
