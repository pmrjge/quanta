---
name: prefill-optimization-landscape
description: "How to cut prefill latency for quanta on M3 Ultra; key fact — prefill is FLOP-bound, so quant doesn't help it"
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Reducing PREFILL latency for quanta (Kimi-K2.6, single M3 Ultra). From 2026-05-23
research (4 parallel web-research streams).

**Key fact:** on M3 Ultra prefill is COMPUTE/FLOP-bound (decode is bandwidth-bound).
So int3/int4 expert quantization gives ~ZERO prefill speedup — `gather_qmm`
dequantizes to bf16 then runs a FLOP-bound GEMM. Quant is a memory-fitting +
*decode* tool, not a prefill lever. M-series has no FP8 tensor-core matmul. (MLX
Discussion #3209.)

Levers by priority:
- **Output-equivalent (parity-safe):** (a) prefix caching + disk-persisted MLA
  latents — biggest single-user win, amplified by MLA's tiny ~512-d/token latent;
  (b) sorted-by-expert grouped GEMM via `gather_qmm(sorted_indices=True)` (~2× MoE
  prefill, only on the post-bake quantized path; mlx PR #2078); (c) expanded MLA
  for prefill / absorbed MLA for decode.
- **Lossy (needs a QUALITY gate — ppl/retrieval, not numeric parity):** training-
  free sparse prefill attention (MInference ~10×@1M, XAttention ~13.5×@256K) attacks
  O(T²); blocker = integrating with MLA's compressed KV. MoBA/NSA are NOT in K2.6
  (they require training).
- **Hygiene:** chunked prefill (bounds memory, not single-req latency); `mx.compile`
  the norm/RoPE glue; `set_wired_limit`; one `mx.eval`/token + `async_eval`.

**How to apply:** at 256K, attention O(T²) dominates → sparse attention is the
asymptotic prize but must be ppl-gated. At moderate ctx, MoE GEMM dominates → sorted
`gather_qmm`. Caching makes *repeated* prefill ~free. **All four implemented +
parity/ppl-green, behind flags defaulting to the naive path:** #1 prefix caching
(`build_prefix_cache`/`continue_from_cache`), #2 sorted dispatch
(`SparseMoE.sort_dispatch`), #3 absorbed MLA (`MLAAttention.absorbed`), #4 XAttention
block-sparse prefill (`MLAAttention.sparse` / `XAttnConfig`) — antidiagonal scoring +
nucleus select, with a mask path (quality) and a **chunked block-gather** path
(bounded-memory long ctx, auto-sized to `max_alloc_gb`, fails loud); ~free at τ≥0.8 on
the long-doc ppl gate. Remaining for true 256K speed: gather still recomputes per-chunk
scoring (cheap) and the win needs real sparse patterns. See [[prior-quantification-flash-sdpa]].
