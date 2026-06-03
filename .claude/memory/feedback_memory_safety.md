---
name: feedback-memory-safety
description: Never run unbounded / O(T²)-scale allocations; guard large allocs to fail loud BEFORE allocating
metadata:
  node_type: memory
  type: feedback
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Never write or run code that can allocate unbounded or O(T²)-scale memory — not even
a throwaway benchmark or op-test. Guard every large allocation so it **raises before
allocating** (fail loud), and bound it with an explicit cap.

**Why:** A synthetic XAttention benchmark swept context length up to 32768 × 64 heads
and, on non-sparse synthetic data, the block-gather path didn't prune (max_kept ≈ all
blocks), so it tried to materialize a per-head block mask of hundreds of TB. It
**OOM'd the host and forced a reboot** (lost work, user trust). The model harnesses
already stream one layer at a time (good — CLAUDE.md rule #8), but ad-hoc
benchmarks/op-tests bypassed that discipline. CLAUDE.md rule #6 (fail loud) + #8
(memory discipline) cover the principle; this is the concrete scar.

**How to apply:**
- Attention / long-context code must cap allocations: `XAttnConfig.budget` bounds
  kept blocks/query; `XAttnConfig.max_alloc_gb` makes both the gather and mask paths
  raise `MemoryError` before allocating beyond it. Keep these guards.
- Estimate bytes before running anything at scale; if a tensor is `O(T²·H)`, don't.
- Test at small, bounded sizes (the parity harness uses T=500, H=4 — CPU-safe). Do
  **not** sweep large T synthetically to "show speed"; on a 512 GB host even one bad
  op = reboot.
- This is a single-user machine the user is actively on — a runaway alloc is not just
  a failed run, it takes down their whole session.
