"""#60 e2e gate: MiMo-V2.5 streamed bf16 forward + teacher-forced perplexity on real prose.

Per-layer math is gated vs HF (``mimo_layer_parity`` / ``mimo_moe_parity``); this confirms the full
48-layer streamed integration is COHERENT via teacher-forced perplexity — the methodology's
end-to-end arbiter. The sequence is longer than the SWA window (128) so sliding-window attention is
actually exercised. Text-only (vision/audio are #64/#65). The streamed forward dequantizes every
layer once, so expect a few minutes.

    uv run --with tokenizers python -m parity.mimo_forward_ppl
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
from tokenizers import Tokenizer

from quanta.mimo.config import MiMoV2Config
from quanta.mimo.loader import MiMoSourceCheckpoint
from quanta.mimo.reference import teacher_forced_ppl

ART = "/Users/pmrj/models/MiMo-V2.5"

PROSE = (
    "The history of writing traces the development of expressing language by systems of markings. "
    "In the history of how writing systems have evolved in human civilizations, more complete "
    "writing systems were preceded by proto-writing, systems of ideographic or early mnemonic "
    "symbols. True writing, in which the content of a linguistic utterance is encoded so that "
    "another reader can reconstruct, with a fair degree of accuracy, the exact utterance written "
    "down, is a later development. It is distinguished from proto-writing, which typically avoids "
    "encoding grammatical words and affixes, making it difficult or impossible to reconstruct the "
    "exact meaning intended by the writer unless a great deal of context is already known in "
    "advance. One of the earliest forms of written expression is cuneiform, which emerged in the "
    "ancient Near East and was used for thousands of years before being gradually replaced by "
    "alphabetic scripts. The invention of writing transformed human societies, enabling the "
    "recording of laws, the administration of complex states, the preservation of literature, and "
    "the accumulation of knowledge across generations."
)


def run() -> None:
    dtype = mx.float32 if (len(sys.argv) > 1 and sys.argv[1] in ("f32", "float32")) else mx.bfloat16
    cfg = MiMoV2Config.from_pretrained(ART)
    tok = Tokenizer.from_file(f"{ART}/tokenizer.json")
    ids = [cfg.bos_token_id] + tok.encode(PROSE).ids
    print(f"tokens: {len(ids)} (bos + {len(ids) - 1}) | swa window {cfg.sliding_window} | dtype {dtype}", flush=True)

    ck = MiMoSourceCheckpoint(ART, cfg)
    t0 = time.perf_counter()
    ppl = teacher_forced_ppl(cfg, ck, ids, dtype=dtype)
    dt = time.perf_counter() - t0
    print(f"teacher-forced ppl: {ppl:.3f}   ({dt:.1f}s streamed, {cfg.num_hidden_layers} layers)", flush=True)
    print("PASS" if ppl < 12.0 else "FAIL (incoherent — pipeline bug)")


if __name__ == "__main__":
    run()
