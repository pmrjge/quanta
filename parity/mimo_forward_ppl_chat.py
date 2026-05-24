"""Confirm broken-vs-wrong-eval: teacher-forced ppl on a CHAT-TEMPLATED sequence.

Raw-prose ppl was ~90k. A working LM can't give ~90k on plain English (that's near-uniform), but to
be rigorous we also try the model's own chat template (some heavily instruct/reasoning-tuned models
go OOD on un-templated text). If chat ppl is also huge -> the checkpoint/model is broken (matches my
forward provably reproducing HF's reference). If chat ppl is sane -> the model works and raw prose
was simply the wrong eval.

    uv run --with tokenizers --with jinja2 python -m parity.mimo_forward_ppl_chat
"""

from __future__ import annotations

import time

from transformers import AutoTokenizer

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.loader import MiMoSourceCheckpoint
from quanta.mimo.reference import teacher_forced_ppl

ART = "/Users/pmrj/models/MiMo-V2.5"
MSGS = [
    {"role": "user", "content": "Write a short paragraph about the history of writing."},
    {"role": "assistant", "content":
        "The history of writing traces the development of expressing language by systems of "
        "markings. True writing encodes a linguistic utterance so that another reader can "
        "reconstruct the exact words written down, distinguishing it from proto-writing. One of the "
        "earliest forms is cuneiform, which was used for thousands of years in the ancient Near East "
        "before alphabetic scripts gradually replaced it across much of the world."},
]


def run() -> None:
    cfg = MiMoV2Config.from_pretrained(ART)
    tk = AutoTokenizer.from_pretrained(ART, trust_remote_code=True)
    text = tk.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=False)
    ids = tk(text, add_special_tokens=True).input_ids
    print(f"chat tokens: {len(ids)} | head {ids[:8]}", flush=True)

    ck = MiMoSourceCheckpoint(ART, cfg)
    t0 = time.perf_counter()
    ppl = teacher_forced_ppl(cfg, ck, ids)
    print(f"chat teacher-forced ppl: {ppl:.3f}   ({time.perf_counter() - t0:.1f}s)", flush=True)
    print("model WORKS (raw-prose was the wrong eval)" if ppl < 30 else "STILL garbage -> checkpoint/model broken")


if __name__ == "__main__":
    run()
