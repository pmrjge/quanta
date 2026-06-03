---
name: streaming-layerwise-loading
description: "Load source tensors streaming (sliced), run/quant one layer at a time — never hold full tensors or whole model resident"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Load source weights **streaming**, not as full tensors:

- **Bake / quant:** read only the slice being processed (per-expert,
  per-projection, per-column-block), dequantize → quantize → write → release.
  Never materialize a full tensor stack or a whole layer's dequantized experts
  at once.
- **Running the full source model** (teacher-forced ppl, layer-by-layer parity):
  hold **one decoder layer resident at a time** — load layer N, run, release,
  load N+1. Never load the whole model.
- Even inputs like `embed_tokens`: gather only the rows needed (sliced read),
  not the full `[vocab, hidden]` tensor.

**Why:** user emphasized this repeatedly on 2026-05-23 ("load tensors streaming,
not full tensors" for the quant; "run model layer-by-layer" for the full source).
Mirrors CLAUDE.md rule 3 (only coarse layer loops allowed at load/bake) and rule
8 (one text layer resident). The source is ~1 TB int4 and dequant expands it;
nothing else fits the 490 GiB ceiling during bake.

**How to apply:** the loader API must support streamed single-tensor / sliced
reads and a one-layer-at-a-time iterator; design every bake/parity loop to
release the prior layer/expert before loading the next. See [[mlx-only-implementation]].
