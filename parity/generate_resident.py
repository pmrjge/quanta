"""Bigger-prompt e2e test of the resident quantized Kimi: long teacher-forced ppl (bf16-KV vs
int8-KV) + a 512-token sampled generation with the int8 KV cache.

The int8-vs-bf16 ppl delta is the gate for defaulting int8 KV (rule-4). Generation is sampled
(temp 0.7), not greedy — reasoning models loop under greedy regardless of quant. Loads the full
artifact resident (~485 GiB int3 / ~388 GiB int2) so run it when nothing else holds memory.

    uv run --with tiktoken python -m parity.generate_resident [int3|int2]
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.cache import MLACache
from quanta.generate import generate
from quanta.runtime import ResidentModel
from quanta.tokenizer import KimiTokenizer

ARTS = {
    "int3": "/Users/pmrj/models/Kimi-K2.6-quanta_int3",
    "int2": "/Users/pmrj/models/Kimi-K2.6-quanta_int2g4",
    "int2g64": "/Users/pmrj/models/Kimi-K2.6-quanta_int2g64",
}

PROMPT = (
    "The history of computing is a story of abstraction layered upon abstraction. The earliest "
    "machines were mechanical, their logic etched into gears and cams; a change of function meant "
    "a change of hardware. The decisive break came with the stored-program idea: instructions and "
    "data would share the same memory, so a single physical machine could become any machine simply "
    "by loading different software. From that idea flowed everything else — operating systems that "
    "multiplex scarce processors among many tasks, compilers that translate human-readable languages "
    "into machine code, and networks that let distant computers cooperate as if they were one. Each "
    "layer hides the messy details of the layer beneath it, letting programmers reason in terms that "
    "match the problem rather than the silicon. The cost of this convenience is distance from the "
    "metal: a modern web request may pass through dozens of layers, each adding latency and the "
    "possibility of failure, and few engineers understand the whole stack end to end. Yet the "
    "abstractions hold, most of the time, because each layer honors a contract with the layers above "
    "and below. When those contracts are violated — a library changes behavior, a protocol is "
    "misimplemented, a cache returns stale data — the failures are often subtle and hard to trace, "
    "precisely because the abstraction that usually helps us now hides the cause. Good engineering, "
    "then, is partly the art of knowing which abstraction is leaking and when to look beneath it."
)
GEN_SEED = "Explain, in a few clear paragraphs, how a modern CPU executes a single line of code."


def ppl(model: ResidentModel, ids: list[int], quantized_kv: bool) -> tuple[float, float]:
    arr = mx.array(ids)
    caches = [MLACache(quantized=quantized_kv) for _ in range(model.num_layers)]
    logits = model(arr, caches=caches, sparse=None)[0]
    lg = logits[:-1].astype(mx.float32)
    tgt = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
    return float(mx.exp(ce.mean()).item()), float((mx.argmax(lg, axis=-1) == tgt).astype(mx.float32).mean().item())


def run() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "int3"
    art = ARTS[which]
    mx.set_wired_limit(int(490 * 1024**3))
    t0 = time.perf_counter()
    rm = ResidentModel(art)
    tok = KimiTokenizer(art, bos_id=rm.cfg.bos_token_id)
    ids = tok.encode(PROMPT, add_bos=True)

    pb, ab = ppl(rm, ids, quantized_kv=False)
    pq, aq = ppl(rm, ids, quantized_kv=True)
    print(f"\n=== resident {which} — bigger prompt ({len(ids)} tok), teacher-forced ===")
    print(f"  bf16 KV : ppl {pb:.3f}  top1 {ab:.3f}")
    print(f"  int8 KV : ppl {pq:.3f}  top1 {aq:.3f}   (Δppl {100 * (pq - pb) / pb:+.2f}%)")

    seed = tok.encode(GEN_SEED, add_bos=True)
    out = generate(rm, seed, max_new_tokens=512, temperature=0.7, top_p=0.9,
                   eos_id=tok.eos_id, sparse=None, quantized_kv=True)
    print(f"\n=== {which} — 512-token generation (int8 KV, temp 0.7), {time.perf_counter() - t0:.0f}s total ===")
    print(tok.decode(out))


if __name__ == "__main__":
    run()
