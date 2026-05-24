"""Nemotron-H bf16 teacher-forced perplexity — the reference for the int4 quant gate (#38).

Streamed forward: one decoder layer's source weights resident at a time (build -> load -> run ->
release), so peak residency is ~one layer (~6 GiB) not the whole 120B (~240 GiB) — the rule-8
memory discipline. Single teacher-forced pass (prefill only; no decode state). The layers run in
bf16; the final norm + lm_head + softmax are fp32 for a stable ppl.

    uv run --with tokenizers python -m parity.nemotron_ppl
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import NemotronSourceCheckpoint
from quanta.nemotron.model import NemotronBlock, load_block
from quanta.nemotron.tokenizer import NemotronTokenizer

MODEL = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
EMBED, NORMF, HEAD = "backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"
PROSE = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria convert "
    "light energy into chemical energy stored in sugars. Inside the chloroplasts, chlorophyll "
    "absorbs sunlight, which drives the splitting of water molecules into oxygen, protons, and "
    "electrons. The oxygen is released into the atmosphere as a byproduct, while the energy "
    "captured is used to fix carbon dioxide from the air into glucose. This remarkable reaction "
    "sustains nearly all life on Earth, forming the base of the food chain and regulating the "
    "balance of oxygen and carbon dioxide in the atmosphere."
)


def streamed_logits(ck: NemotronSourceCheckpoint, cfg: NemotronHConfig, ids: mx.array,
                    n_layers: int | None = None) -> mx.array:
    """Logits ``[t, vocab]`` via a one-layer-resident streamed forward (prefill)."""
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    h = ck.read(EMBED)[ids][None].astype(mx.bfloat16)
    mx.eval(h)
    ck.release()
    for i in range(n):
        block = NemotronBlock(cfg, cfg.layer_kind(i))
        load_block(block, ck, cfg, i)
        h, _, _ = block(h)
        mx.eval(h)
        del block
        ck.release()
        mx.clear_cache()
    norm_w = ck.read(NORMF)
    h = mx.fast.rms_norm(h.astype(mx.float32), norm_w.astype(mx.float32), cfg.norm_eps)
    head = ck.read(EMBED if cfg.tie_word_embeddings else HEAD).astype(mx.float32)
    return (h @ head.T)[0]


def run() -> None:
    cfg = NemotronHConfig.from_pretrained(MODEL)
    ck = NemotronSourceCheckpoint(MODEL)
    tok = NemotronTokenizer(MODEL)
    ids = tok.encode(PROSE, add_bos=False)[:192]
    arr = mx.array(ids)
    t0 = time.perf_counter()
    logits = streamed_logits(ck, cfg, arr)
    lg = logits[:-1].astype(mx.float32)
    targets = arr[1:]
    ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
    ppl = mx.exp(ce.mean()).item()
    acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
    print(f"\n=== Nemotron-H bf16 reference (streamed, tokens={len(ids)}) ===")
    print(f"perplexity           : {ppl:.3f}")
    print(f"top-1 next-token acc : {acc:.3f}")
    print(f"elapsed              : {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run()
