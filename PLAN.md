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
| **M2** — Stage B1: batched ring buffer | ✅ DONE | `05d1171` | extend `parity/dsv4_kv_arena_test.py` (ring) |
| **M3** — Stage B2: compressed stepper on arena | ✅ DONE | `bf7af6b` | `parity/dsv4_batched_attention_test.py` (compressed + arena) |
| **M4** — flip default + session/prefill + regression | ✅ DONE | `e08888d` | full suite (parity trio + omlx/paged/tree-verify + pytest/ruff/compileall) |
| **M5** — real-model B-sweep bench | ☐ (deferred, solo GPU) | — | `parity/dsv4_batched_bench.py` |

Flag `kv_arena` was **default OFF** through M0–M3 (each proved its sub-parity with
the arena toggled ON *inside the test*, by calling the stepper with an arena).
**M4 flipped the default ON** (`e08888d`): the serving path now leases an arena row
per stream via `make_cache`; a discrete `DSV4Cache` still takes the proven
per-stream loop (the batched steppers dispatch on the cache TYPE, not the flag).
Only M5 (the real-model bench, solo GPU) remains.

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

- **`src/quanta/dsv4/decode.py`** — primary (anchors current as of `e08888d`).
  - M0: `_grow_seq` (774), `_KVArena` (783) with `append_row` (848), `append_batched` (865),
    `read_batched` (898), `truncate_row`/`reset_row`; `_KVArenaSet` (947); `_ArenaLayerView` (1005).
  - M1: `decode_step_dense_batched` (1181) takes keyword-only `arena`/`rows`; arena path =
    `append_batched` + `read_batched`, else the per-stream `lcs` loop + `_pad_stack` (reference).
  - M2: `_ring_cap` (613, shared) + `_push_ring_batched` (632) — fixed-width `[R,cap,dim]` roll.
  - M3: `decode_step_compressed_batched` (1300) arena path; `_compressed_update_arena` (1251) =
    latent scatter + ring roll + masked compute-all pool; `_pool_one_window_b` (552); `_CompArena`
    (1040: ckv/ikv `_KVArena`s + `[R,cap,dim]` ring; `roll_ring`/`append_pooled`).
  - **M4 (done):** `_KVArena.seed_row` (925, verbatim code-copy migration); `_KVArenaSet` owns
    `self.comp` (one `_CompArena` per ratio>0 layer via `comp_specs`; `free` resets comp rows);
    `_CompArena.reset_row`/`seed_row` (1105)/`_seed_ring` (migrate prefilled ckv/ikv/ring, ring
    right-aligned); `_ArenaCacheHandle` (1136: leased row as a DSV4Cache-shaped object —
    `__getitem__`→`_ArenaLayerView`, `.offset`, `.row`, `seed_comp`).
- **`src/quanta/dsv4/batched_runtime.py`** — **M4 (done).** `_init_from_inner` (≈170) builds the
  `_KVArenaSet` (latent + comp) when `kv_arena` (default ON); `make_cache` (186) leases a row →
  `_ArenaCacheHandle`; `free_cache` (198) returns the row; `prefill` (206) calls `seed_comp` after
  the inner forward; `step_batch` (354) fails loud if a handle hits the non-fused/T>1 path;
  `_decode_batched_single` (477) dispatches handles → arena steppers (else per-stream `lcs`), keyed
  on the cache TYPE. Defaults flipped ON in `__init__`/`from_inner`/`_init_from_inner`.
- **`src/quanta/dsv4/batched_generate.py`** — **M4 (done).** `free_cache` on every stream retirement
  + a try/finally so an early/error exit never leaks a leased arena row (no-op for non-arena caches).
- **`src/quanta/dsv4/attention.py`** — `sdpa_window_sink_batched` reused verbatim (no change).
- **`src/quanta/shim/omlx.py`** — **M4 (done).** `_DSV4BatchedSession.release` (856) returns the
  arena row to the free-list (no-op for a discrete cache / the paged path, which is separate:
  `_admit_paged` 762, base `release` 752, `_new_cache` 853).
- **`parity/dsv4_kv_arena_test.py`** *(model-free)* — M0 round-trip + M2 ring (done).
- **`parity/dsv4_batched_attention_test.py`** — M1 dense arena + M3 compressed arena (done).
- **`parity/dsv4_batched_test.py`** — Design-A regression **+ M4 end-to-end arena serving check** (done).
- **`parity/dsv4_batched_bench.py`** — M5 arena-vs-loop real-model sweep (deferred; see M5 note).

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

### M2 — Stage B1: batched ring buffer ✅ (`05d1171`)
Added `_push_ring_batched` (decode.py:632) + a shared `_ring_cap` (613) used by both
`_push_ring` and the batched form. The batched ring is a **fixed-width** `[R, cap, dim]`
(`cap = (2 if overlap else 1)*ratio + (max_rollback-1)`), zero-padded at the FRONT with
the newest vector at `[:, -1]`; a row pushed `n` times holds its valid tail in the last
`min(n,cap)` columns — bit-identical to that row's per-stream `_push_ring`. Isolated (no
stepper wiring; wired in M3). Gate (model-free, `dsv4_kv_arena_test.py`): batched roll ==
per-stream `_push_ring` bit-exact across ragged push counts (some rows past `cap`).

### M3 — Stage B2: compressed stepper on the arena ✅ (`bf7af6b`)
`decode_step_compressed_batched` (decode.py:1175) gained the arena path behind keyword-only
`arena`/`rows`/`comp`, killing the per-stream `for s in range(b)` cache-update + pool loop
for ratio>0 layers. `_compressed_update_arena` (1126) = ONE latent scatter
(`_KVArena.append_batched`, reused M1) + ONE batched ring roll (`_CompArena.roll_ring`:
gather active rows → `_push_ring_batched` → scatter back) + ONE **compute-all pool**
(`_pool_one_window_b` (552) over all `B` rows) **masked-scattered** into the ckv/ikv arenas
(`_CompArena.append_pooled`) — only rows where `(offset+1) % ratio == 0` append + bump
`n_comp`. `_pool_one_window_b` carries the two things global single-stream but ragged in a
batch: per-row prev-window validity (`overlap and offset//ratio>=1`; else the window-0
`-inf` pad) and per-row window-start RoPE row (gathered at `(offset//ratio)*ratio`). New
`_CompArena` (1000: ckv/ikv `_KVArena`s + `[R,cap,dim]` ring). `_decode_indexer_select_batched`
(1096) now takes a padded `ikv` array so both stores feed the shared SDPA tail. Per-stream
`lcs` path stays the proven default (`arena=None`); runtime/session wiring deferred to M4.
Gate (`dsv4_batched_attention_test.py` compressed, ratio-4-indexer + ratio-3, ragged offsets
incl. window-closing cases — B=1 close, ragged 3/4, all-close; ratio-4 hits both prev-valid
`c>=1` and the `c==0` window-0 pad in one batched pool): B=1 bit-exact, B≥2 ≤1.83e-07 (<5e-4),
arena == `_LayerCache` batched path **bit-exact (0.0)**, `kv_length()`/`n_comp()` match.

### M4 — flip default + session/prefill integration + regression ✅ (`e08888d`)
Wired the arena through the whole batched-serving decode path and flipped `kv_arena`
default **ON** (`__init__`/`from_inner`/`_init_from_inner`). The per-stream loop is the
retained flagged reference; **dispatch keys off the cache TYPE, not the flag**, so a discrete
`DSV4Cache` always takes the per-stream path (existing DSV4Cache callers/tests unchanged).
- `_KVArenaSet` now owns one `_CompArena` per compressed layer (`comp_specs`, default None =
  latent-only for the M0/M2 gate); `free(row)` resets the comp rows alongside latent.
- **Prefill→decode handoff (Approach B):** prefill is the *unchanged* single-stream decode
  loop driven through an `_ArenaCacheHandle` — latent KV lands in the arena row (each layer's
  `_ArenaLayerView.append_kv`), the derived ckv/ikv/ring land per-object on the views; then
  `_ArenaCacheHandle.seed_comp` migrates them into the `_CompArena` set by copying the STORED
  codes **verbatim** (`_KVArena.seed_row` / `_CompArena.seed_row`/`_seed_ring`) — bit-exact, no
  re-quantize (the bf16-drift trap). The ring is right-aligned into the fixed-width `[R,cap,dim]`.
- `make_cache` leases a row → handle; `prefill` calls `seed_comp`; `_decode_batched_single`
  dispatches handles to the arena steppers (ONE scatter + ONE gather/layer); `free_cache`
  returns the row; `step_batch` fails loud if a handle hits the non-fused/T>1 path (rule 6).
- **Row lifecycle / no leaks:** `_DSV4BatchedSession.release` (omlx.py) frees the row;
  `batched_generate` frees on every retirement **and** in a try/finally so an error path
  (e.g. an empty prompt mid-admit) never leaks a leased row. `dsv4_paged_ppl._score_off`
  repointed to a discrete `DSV4Cache` (its documented intent) so the flip doesn't lease/exhaust.
- **Gate added** to `dsv4_batched_test.py`: end-to-end arena serving — `make_cache`→`prefill`
  (seeds latent arena + migrates comp)→`step_batch` on the arena == single-stream across 4 decode
  steps that cross window boundaries, `n_comp` per-stream match, and lease/free/realloc.
**Wiring caveat (from M3, now handled):** the compressed stepper's prev-window validity keys off
`offset//ratio>=1`, NOT `ring.shape[1]` — inside `_compressed_update_arena`; M4 only feeds the arena.

### M5 — real-model DSV4 B-sweep bench ☐ (deferred, solo GPU)
`parity/dsv4_batched_bench.py` (arena vs per-stream loop) on the baked DSV4 bake
**alone** (verify memory clear first; one model at a time). Not a blocker for M0–M4.
**Update before running (M4 fallout):** M4 flipped `kv_arena` default ON, so `make_cache()`
returns an `_ArenaCacheHandle`. The bench's looped path (`model._fused=False`) must use a discrete
`DSV4Cache` (or build the looped runtime with `kv_arena=False`) — a handle on the non-fused/T>1
path now fails loud (rule 6). Target comparison: **arena** (`kv_arena=True`, `make_cache`→handle,
fused) vs **per-stream loop** (`DSV4Cache`, `_fused=False`). Expect the arena to cut the per-stream
KV-update IO; the win grows with B and with compressed-layer share.

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

**M0–M4 are complete and committed** (M4 = `e08888d`). The batched KV arena is the **default**
serving decode path and is parity-green end-to-end (per-stream loop retained as the flagged
reference; dispatch keys off the cache type). The ONLY remaining item is **M5 — the real-model
B-sweep bench** (`parity/dsv4_batched_bench.py`), which is **deferred** (solo GPU, one model at a
time) and **not a correctness blocker** — #18's correctness is fully gated model-free.

To run M5 (only when the GPU is free and no other model is resident): first update the bench per the
M5 note above (looped path on a discrete `DSV4Cache`), confirm memory is clear, then
`uv run python -u -m parity.dsv4_batched_bench`. Otherwise **#18 is effectively done**.

Per the standing cadence, STOP here and wait for the user to compact before any M5 work.
