# PLAN.md — active task handover (#18: kill the per-stream KV-update IO loop)

> Durable, repo-tracked handover for the in-flight task. Read `CLAUDE.md` first
> (permanent rules + model facts), then this. The ephemeral plan-mode file lives at
> `~/.claude/plans/moonlit-doodling-gray.md` (kept in sync with this doc); this
> PLAN.md is the authoritative durable copy.

---

## Governing cadence (standing user instruction — DO NOT VIOLATE)

> "continue, don't spawn subagents or agents, just focus on a single thread of
> line of thinking, and commit each submilestone and wait for me to compact
> context because it is exponentially growing as more work branches"

This means, for every milestone:
1. **Single linear thread.** NO subagents, NO `Agent`/`Task` tools, NO `Workflow`.
   One stream of reasoning, implement directly.
2. **Implement → parity green → commit → STOP.** After each milestone's gate is
   green, commit it (named files only), then **STOP and wait for the user to
   compact context** before starting the next milestone. Do not roll milestones
   together.
3. Committing per-milestone IS authorized by the instruction above. Otherwise the
   normal rule holds: do **not** commit unless asked.

### Commit rules (from CLAUDE.md + standing constraints)
- Add files **by name** (never `git add -A`, never blind add). Never push unless
  asked. Never skip hooks.
- Commit trailer (project CLAUDE.md is authoritative — use **4.7**, not 4.8):
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Repo commits land directly on `main` (every prior milestone did, incl. M0/M1).

### Other standing constraints
- **RUN ONLY ONE MODEL AT A TIME** (OOM-reboot hazard; a prior bench OOM'd and
  rebooted the host). All heavy/real-model loads are solo GPU sessions.
- No `mlx-lm` on the runtime path (torch/transformers offline-only, `reference`
  extra). Keep `~/models/Kimi-K2.6`; never delete.
- Rule 3 (no hot-path Python loops; bounded IO/accounting loops over `B ≤
  max_batch` are OK), rule 4 (optimizations flag-guarded, default = proven path
  until parity-green), rule 6 (fail loud), rule 8 (layer-by-layer memory).

---

## What #18 is

In DSV4 batched decode, `B` streams decode in lock-step but each owns a **ragged,
independently-grown** per-stream `_LayerCache`. Every step runs a Python loop over
streams that (a) quantizes the new token and grows the cache with `mx.concatenate`,
and for compressed layers (b) rolls the raw-hidden ring and conditionally pools.
Then `_pad_stack` re-reads every stream, dequantizes, and re-pads to `[B, L_max, …]`.

**#18 kills that per-stream loop.** Chosen approach (user-selected, "Full kill:
batched buffer"): replace the `B` ragged per-stream caches with a **persistent,
`max_batch`-sized batched KV arena** + a slot→row free-list, so the hot-path write
is ONE scatter (`arena[rows, cols, :] = codes`) and the read is ONE gather + one
batched dequant — no per-stream Python loop. Output-equivalent and **flag-guarded**
(`kv_arena`), defaulting to the proven per-stream path until parity is green (rule 4).

### Accepted equivalence bar (unchanged across all milestones)
- **B=1 bit-exact** (`|Δ| == 0`; no padding, no batched-kernel reorder).
- **B≥2 greedy-exact** (`max|Δ| < 5e-4`, argmax-stable; the pad+mask SDPA reorders
  the softmax reduction → bf16 ULPs).
- **AND** per-stream `kv_length()` / `n_comp()` match the per-stream loop.

### Why bit-exact contents are guaranteed
The arena reuses `quanta.cache_quant.quantize_last_axis` / `dequantize_last_axis`
**verbatim** (no kernel reimpl — avoids the bf16-drift trap). Affine int-bits over
the **last axis is row-independent**, so a batched `[B,1,D]` quantize equals `B`
separate `[1,1,D]` quantizes row-for-row. Stale/zero arena padding beyond a row's
length is sent to `-inf` by the existing SDPA window/pad mask → numerically inert.

---

## Status

| Milestone | State | Commit | Gate |
|---|---|---|---|
| **M0** — arena store + free-list + flag | ✅ DONE | `41a4d0f` | `parity/dsv4_kv_arena_test.py` |
| **M1** — Stage A: dense stepper on arena | ✅ DONE | `6f33cc1` | `parity/dsv4_batched_attention_test.py` (dense + arena) |
| **M2** — Stage B1: batched ring buffer | ⏭ NEXT | — | extend `parity/dsv4_kv_arena_test.py` (ring) |
| **M3** — Stage B2: compressed stepper on arena | ☐ | — | `parity/dsv4_batched_attention_test.py` (compressed + arena) |
| **M4** — flip default + session/prefill + regression | ☐ | — | full suite |
| **M5** — real-model B-sweep bench | ☐ (deferred, solo GPU) | — | `parity/dsv4_batched_bench.py` |

Flag `kv_arena` stays **default OFF** through M0–M3 (each proves its sub-parity
with the arena toggled ON *inside the test*, by calling the stepper with an arena).
**M4 flips the default ON.**

---

## Design recap — the batched KV arena

Owned by `DSV4BatchedResidentModel` (M4), one set per decoder layer, sized to
`max_batch` (`R`):

- **Latent KV store** (all layers): int8 codes `[R, L_cap, D/pack]` + scales/biases
  `[R, L_cap, D/group]` (same `cache_quant` codec) + `lengths[R]`. `quantized=False`
  ⇒ bf16 `[R, L_cap, D]` (the tiny-head_dim test path; real DSV4 head_dim=128 ⇒ int8).
- **Compressed extras** (ratio>0 layers, M3): pooled-KV arena `ckv` `[R, C_cap, …]`
  + `n_comp[R]`; indexer `ikv` arena (DSA layers); a **batched** ring `[R, cap, …]`
  (fixed `cap`, already non-ragged).
- **Free-list:** `alloc()` / `free(row)` (raise on exhaustion + double-free, rule 6).

**Two write surfaces, bit-identical contents:**
1. `append_row(row, kv)` — per-row slice-assign (prefill / multi-token tail, via
   `_ArenaLayerView`). Bounded one-row write, not a hot loop.
2. `append_batched(rows, kv)` — the hot decode path: one quantize of `[B,1,D]` +
   one scatter `arena[rows, cols, :] = codes`. The loop-kill.

**Hot read:** `read_batched(rows)` = `mx.take(arena, rows, 0)[:, :L_max]` + one
batched dequant → replaces `_pad_stack`.

**MLX (0.31.2, verified):** `arena[rows, cols, :] = vals` is a bit-exact 2D
fancy-index scatter; `mx.take(arena, rows, axis=0)` gathers; `mx.scatter` does NOT
exist (use fancy-index assign). Do **not** reimplement quant/SDPA kernels.

**Scope boundary:** arena is the **batched-serving** decode path
(`_DSV4BatchedSession`) only. Tree-spec `replicate`/`_copy` structural sharing
(EAGLE-3, forgone on DSV4) stays on per-stream `DSV4Cache` (flag OFF). Downside: a
dense arena pads to the longest live stream (`L_cap`), so skewed-length batches
waste KV memory; batched-paged is the noted future option if that bites.

---

## Files (with current line anchors)

- **`src/quanta/dsv4/decode.py`** — primary.
  - M0 (done): `_grow_seq` (716), `_KVArena` (725) with `append_row` (790),
    `append_batched` (807), `read_batched` (840), `truncate_row`/`reset_row`;
    `_KVArenaSet` (868); `_ArenaLayerView` (909).
  - M1 (done): `decode_step_dense_batched` (942) now takes keyword-only
    `arena`/`rows`; arena path = `append_batched` + `read_batched`, else the
    per-stream `lcs` loop + `_pad_stack` (reference). Fail-loud on arena/rows mismatch.
  - M2: add a **batched ring** helper next to `_push_ring` (581) — a `[R,cap,…]`
    roll. M3: `decode_step_compressed_batched` (1010) rewrite + masked-batched
    `_maybe_pool`/`_pool_one_window` (518/552) + `append_ckv`/`ikv` arenas.
- **`src/quanta/dsv4/batched_runtime.py`** — M4.
  - `_kv_arena` flag set in `_init_from_inner` (165, INERT today). `make_cache`
    (168) → return arena-backed handle; `prefill` (173) seeds via `_ArenaLayerView`;
    `_decode_batched_single` (426) → dispatch to the arena steppers when
    `self._kv_arena`; runtime owns the `_KVArenaSet` + stream→row map.
- **`src/quanta/dsv4/attention.py`** — reuse `sdpa_window_sink_batched` verbatim
  (no change expected; confirm gather+dequant shapes match its inputs).
- **`src/quanta/shim/omlx.py`** — M4. `_DSV4BatchedSession` (840) + base
  `admit` (711) / `release` (752) / `_new_cache` (851) alloc/free the arena row.
  Arena is the **non-paged** batched path; the paged path (`has_recurrent_state`,
  `_admit_paged` 762) is separate and unchanged.
- **`parity/dsv4_kv_arena_test.py`** *(model-free)* — M0 round-trip (done) + M2 ring.
- **`parity/dsv4_batched_attention_test.py`** — M1 dense arena (done) + M3 compressed.
- **`parity/dsv4_batched_bench.py`** — M5 arena-vs-loop real-model sweep (deferred).

---

## Per-milestone detail

### M0 — arena store + free-list + flag ✅ (`41a4d0f`)
Added `_KVArena` / `_KVArenaSet` / `_ArenaLayerView` + the `kv_arena=False`
constructor flag (mirrors `_fused`). Existing path untouched. Gate
`parity/dsv4_kv_arena_test.py`: scatter-write/gather-read round-trip bit-identical
to per-stream `_LayerCache` (int8 g128 head_dim=128 + bf16 head_dim=16, ragged
non-contiguous rows `[4,1,3,0]`), `_ArenaLayerView` surface match, free-list +
doubling-growth invariants.

### M1 — Stage A: dense stepper on the arena ✅ (`6f33cc1`)
`decode_step_dense_batched` gained the arena path behind keyword-only
`arena`/`rows`: hot path = `arena.append_batched(rows, kv)` (one scatter) +
`arena.read_batched(rows)` (one gather + dequant), killing the `append_kv` loop +
`_pad_stack` for ratio-0 layers. Per-stream `lcs` path is the **default** reference
(`arena=None`); runtime/session wiring deferred to M4. Gate (model-free, in
`dsv4_batched_attention_test.py`): `_arena_from_layer_caches` code-copies a seeded
set into a `_KVArena`, runs the arena stepper, asserts arena == per-stream loop
(B=1 bit-exact, ragged B≥2 ≤1.79e-07 < 5e-4, `length()` matches) AND arena ==
`_LayerCache` batched path **bit-exactly** (0.0). M0 gate + `dsv4_batched_test`
runtime parity + ruff + compileall all still green.

> Note: `_cfg()` in `parity/dsv4_batched_test.py` uses `HEAD_DIM = 8` (< 32 quant
> floor), so the latent cache auto-resolves to **bf16** there — M1's stepper-level
> parity exercises the bf16 arena path; the int8 codec round-trip is M0's gate.
> This division is intentional (stepper wiring is dtype-agnostic past the codec).

### M2 — Stage B1: batched ring buffer ⏭ NEXT
Replace per-stream `_push_ring` (decode.py:581) with a batched `[R, cap, …]` roll
(isolated; no quant ⇒ bit-exact; **not yet wired into a stepper**). `_push_ring`
keeps the last `cap = (2 if overlap else 1)*ratio + (max_rollback-1)` raw-hidden
positions via concat-then-trim; the batched form holds `[R, cap, dim]` and rolls
all rows at once. **Parity:** add a model-free ring check to
`parity/dsv4_kv_arena_test.py` — batched roll == per-stream `_push_ring` roll,
bit-exact, across ragged per-row push counts (some rows past `cap`, some not).
Then commit (`#18 M2`), STOP for compaction.

### M3 — Stage B2: compressed stepper on the arena ☐
Wire `decode_step_compressed_batched` (decode.py:1010): latent write/read (reuse
M1) + batched ring (M2) + **masked compute-all pool** with masked scatter-append
into `ckv`/`ikv` arenas (bump `n_comp`) — compute the pooled window for ALL streams,
then masked-scatter only rows where `(offset+1) % ratio == 0`. The `for s in
range(b)` body is fully gone. **Parity:** extend
`parity/dsv4_batched_attention_test.py` compressed case (layers 1 ratio-4-indexer,
2 ratio-3, ragged lens) arena-path vs per-stream — same bar; `n_comp()` /
`kv_length()` match.

### M4 — flip default + session/prefill integration + regression ☐
`make_cache()` reserves a row via the free-list and returns an
`_ArenaLayerView`-backed handle (`.row`, `__getitem__(i)`); `prefill` seeds that row
unchanged; `admit`/`release` (omlx.py) call alloc/free; `_decode_batched_single`
dispatches to the arena steppers; flip `kv_arena` default **ON** (per-stream loop
retained as the flagged reference). **Parity:** full
`dsv4_batched_attention_test.py` green by default; tree-verify + paged gates + the
broad suite. Before M4's commit run the full regression (below).

### M5 — real-model DSV4 B-sweep bench ☐ (deferred, solo GPU)
`parity/dsv4_batched_bench.py` (arena vs per-stream loop) on the baked DSV4 bake
**alone** (verify memory clear first; one model at a time). Not a blocker for M0–M4.

---

## Gate commands

Per-milestone (run before each commit):
```bash
uv run python -m parity.dsv4_kv_arena_test                       # M0, M2
uv run --with numpy python -m parity.dsv4_batched_attention_test # M1, M3, M4
uv run --with numpy python -m parity.dsv4_batched_test           # regression: default per-stream path
```
Before M4's commit (full regression), also:
```bash
uv run --with pytest pytest tests/ -q
uv run --with ruff ruff check src tests
uv run python -m compileall -q src tests
uv lock --check
git diff --check
```
M5 (solo GPU, deferred): `uv run python -u -m parity.dsv4_batched_bench`.

---

## Immediate next action (for the resuming agent)

Start **M2 — batched ring buffer**, exactly as scoped above:
1. Add a batched `[R, cap, …]` roll helper in `decode.py` (alongside `_push_ring`,
   line 581) — isolated, no stepper wiring yet.
2. Add a model-free ring check to `parity/dsv4_kv_arena_test.py`: batched roll ==
   per-stream `_push_ring`, bit-exact, ragged push counts.
3. Run `uv run python -m parity.dsv4_kv_arena_test` (+ ruff + compileall on touched
   files). Green ⇒ commit `#18 M2` (named files, project trailer, no push).
4. **STOP and wait for the user to compact** before M3.

Do not start M2 until the user says go (they compact between milestones).
