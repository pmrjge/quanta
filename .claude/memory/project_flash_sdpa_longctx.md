---
name: prior-quantification-flash-sdpa
description: Prior project ran 256K ctx via flash tiled SDPA; quanta attention must use mx.fast.sdpa for long context
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

The previous effort (`quantification`) implemented MLA attention as **flash tiled
SDPA** specifically to fit **256K context** on the M3 Ultra (512 GB). quanta's
runtime attention must likewise use a flash/tiled path
(`mx.fast.scaled_dot_product_attention`, `mask="causal"`) for the production
long-context path. A naive materialized `[B, H, T, T]` scores matrix is acceptable
ONLY for the small-T parity gate — it blows up at 256K (64 heads).

**Why:** user stated this on 2026-05-23. `max_position_embeddings = 262144`; a
dense scores tensor at that length is intractable, flash SDPA keeps it bounded.

Prior tiling (user recollection, approximate): the flash attention tiled into
**128×128 and 64×64 blocks** (query-block × key-block). `mx.fast.sdpa` tiles
internally, so prefer it (CLAUDE.md rule 2); these block sizes are the reference
point if a hand-rolled tiled kernel is ever needed (e.g. if SDPA can't serve the
MLA-compressed cache).

**How to apply:** keep naive explicit attention as the parity reference; make
`mx.fast.sdpa` a first-class, parity-validated runtime path. Handle the MLA
head-dim mismatch (qk=192 vs v=128) inside SDPA (zero-pad V to 192, slice output
back to 128). The true 256K path also needs the MLA-compressed KV cache /
matrix-absorb (a separate, parity-gated milestone — a known suspect in CLAUDE.md);
the small-T gate uses full MHA. Validate flash-vs-naive equivalence before trusting
it. See [[mlx-only-implementation]], [[streaming-layerwise-loading]].
