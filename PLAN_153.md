# PLAN_153.md — active task handover (#153: batched-paged KV — bring the #18 loop-kill to the paged path)

> Durable, repo-tracked handover for the NEXT task. Read `CLAUDE.md` first (permanent
> rules + model facts), then `PLAN.md` (#18, the batched KV arena — DONE M0–M5; the
> machinery #153 reuses), then this. **Status: DSV4 core M0–M1 DONE (storage primitives + dense
> paged stepper `_PagedKVArena`, bit-exact model-free); M2 (compressed stepper) next. Multi-model
> loop-kill (user's order nemotron→internlm2→qwen3.6): Nemotron DONE + default GRADUATED ON
> (real-model bench +18%@B48); InternLM2.5 DONE (wire via shared `batched_decode_attention_kv`,
> default OFF — its own bench/graduate is the next ask); Qwen3.6 LAST (unpaged+hybrid, big).**

---

## M0 — DONE (what actually shipped, vs the original design below)

The batched block-table scatter/gather landed as **generic methods on
`PagedKVCacheManager`** — `write_one_batched`/`write_batched` (ONE quantize + ONE
fancy-index scatter across all B streams' tail blocks) + `gather_one_batched`/
`gather_batched` (ONE `mx.take` over a padded block-id matrix + ONE batched dequant) —
**not** a separate `_PagedKVArena` storage class as the design below first sketched. Why
the change: the block pool already abstracts k/v-pair vs single-stream via its component
dict (`for name, arr in encoded.items()`), and already owns alloc / COW / codec, so the
batched siblings of `_write_encoded`/`gather`/`gather_one` are **generic over the
component dict for free** — one primitive serves **both** DSV4's single-stream latent AND
the k/v keepers, with no duplicated alloc/COW logic. A `_PagedKVArena` still appears in
**M1**, but only as a *thin per-layer adapter* `(manager, seqs, layer)` presenting the
steppers' `append_batched(rows, kv)`/`read_batched(rows)` interface by delegating to these
manager methods (the `rows` lease-indices collapse to "this batch's seqs").

- **Files:** `src/quanta/paged/paged_kv_cache.py` (`write_*_batched`/`gather_*_batched` +
  `_write_encoded_batched`/`_gather_encoded_batched`), `src/quanta/paged/__init__.py`
  (flag `PAGED_KV_BATCHED_DEFAULT=False`, rule 4), `parity/dsv4_paged_batched_test.py`
  (model-free gate).
- **Gate (green):** single-stream latent (int8 g128 hd=128) batched == `_LayerCache`
  bit-exact; k/v pair (n_kv=2) batched == per-stream paged loop bit-exact; COW (forked
  shared partial tail cloned by the batched writer, parent intact). Block-boundary
  crossings + non-contiguous interleaved block ids exercised. Regressions green
  (`dsv4_paged_latent_test` `|Δ|=0`, `dsv4_batched_test`); ruff/compileall/lock/diff clean.
- **COW-free decode:** the writer COW-clones a shared partial tail *before* the scatter
  (bounded per-stream accounting, outside the tensor op) and asserts the write block is
  private (rule 6). In steady serving decode the tail is always private (COW only fires at
  prefill), so the scatter never touches a shared block.

### Multi-model scope (user directive: apply the loop-kill to **nemotron → internlm2 → qwen3.6, IN
THAT ORDER**; one milestone per commit, STOP to compact between)
The M0 primitive lives on the shared `PagedKVCacheManager`, so it serves **every paged keeper at the
storage layer in one shot**. Per-model the loop to kill is the per-stream KV `.update()` inside each
runtime's FUSED batched attention. **Shared k/v entries in `quanta.modeling.batched_attention`:**
`_sdpa_padded` (factored SDPA tail) + `batched_decode_attention_padded` (consumes a pre-padded
`[B,n_kv,L_max,D]` from a paged `gather_batched`, the k/v sibling of DSV4's `_PagedKVArena`) +
**`batched_decode_attention_kv`** (the single-sourced #153 KV-step: given projected+RoPE'd q/k/v + the
per-stream layer caches, either the per-stream `.update()` loop OR — `paged_batched` + paged views — ONE
`write_batched` + ONE `gather_batched` + the padded SDPA; InternLM2.5's two `decode_batched` call it,
Nemotron's `_fused_attn_layer` inlines the equivalent core). (Started BEFORE DSV4 M2/M3 — user's order.)
- **Nemotron** (k/v, paged) — **✅ DONE `833c8a4`, default GRADUATED ON (this commit).**
  `_fused_attn_layer` already fuses attn; killed its per-stream `KVCache.update()` loop → ONE
  `write_batched` + ONE `gather_batched` + `batched_decode_attention_padded` when caches are
  `PagedKVCacheView`s + `paged_batched` on, threaded through `batched_decode_step_fused`/`_native`.
  Gate `nemotron_batched_attention_test.py` §D BIT-exact (model-free). **Graduated to ON** via a
  **Nemotron-scoped** flag `NEMOTRON_PAGED_KV_BATCHED_DEFAULT=True` (the shared `PAGED_KV_BATCHED_DEFAULT`
  stays OFF so DSV4/InternLM2.5 are untouched + DSV4 M3 not preempted) after the real-model bench
  `parity/nemotron_paged_batched_bench.py` (int4-g64 120B-A12B, prod paged + form-2 session, distinct
  prompts) proved greedy-exact + a real win:

  | B | loop tok/s | loopkill tok/s | loopkill/loop | greedy |
  |---|---|---|---|---|
  | 1 | 27.8 | 27.5 | 0.99× | bit-exact |
  | 32 | 126.3 | 145.8 | **1.15×** | greedy-exact |
  | 48 | 122.3 | 144.5 | **1.18×** | greedy-exact |

  Better than the prior "marginal" expectation: the per-stream `loop` REGRESSES B=32→48 (126→122 —
  Python-loop overhead doesn't scale) while `loopkill` holds (146→144), so the win GROWS with B. The
  bench doubles as the real-model correctness gate for the quantized k/v `write_batched`/`gather_batched`
  at `head_dim=128` (the §D gate used bf16 to stay head_dim-agnostic). `run()` §D + the bench pin the
  default-ON.
- **InternLM2.5** (k/v, paged — "small model") — **✅ DONE (wire, this commit).** Pure dense GQA, 32
  layers, no recurrent state. Killed the per-stream KV `.update()` loop in BOTH `decode_batched` paths —
  bf16 `InternLM2Model` (`internlm2/model.py`) AND packed `_PackedModel` (`internlm2/runtime.py`) — by
  routing their identical KV-update+SDPA tail through the new shared `batched_decode_attention_kv`.
  `paged_batched` threaded: wrapper `InternLM2BatchedResidentModel._paged_kv_batched` ← **shared**
  `PAGED_KV_BATCHED_DEFAULT` (OFF, rule 4) → `step_batch` → `InternLM2ResidentModel.decode_batched`
  (delegator) → inner. Gate `internlm2_batched_attention_test.py` §C BIT-exact (`max|Δ|=0`, full
  `decode_batched` paged loop-kill == per-stream paged loop, B=1 + ragged B=3 boundary-crossing, bf16
  head_dim-agnostic) + pins default-OFF. **Default stays OFF** (unlike Nemotron) — graduates on its OWN
  real-model bench (the `internlm2_5-7b-chat-1m-quanta_int8g64` bake, BEST_BATCH=32, KV-bound ⇒ expected
  a BIGGER win than Nemotron since EVERY layer is attention, not just 8). That bench is the next ask.
- **Qwen3.5/3.6** (`qwen35`) — **LAST, big.** UNPAGED (`shim/omlx._make_batched_session` forces it
  off paged) + hybrid (GDN recurrent + GQA) + NO fused attn yet (loops the whole mixer per stream) +
  serves at B=4. NOT a paged-primitive wire: needs its OWN unpaged GQA arena (#18-style) + batched GDN
  state (like Nemotron's `BatchedMambaState`) + fused attn. Multi-milestone, small win.
- **DSV4** (single-stream latent, paged) — the core #153 path; M1 done, M2–M3 remain (deferred behind
  the multi-model order the user chose).

---

## Governing cadence (standing user instruction — DO NOT VIOLATE)

Same as #18: **single linear thread, NO subagents/agents/workflows.** Implement →
parity green → commit each milestone (named files, trailer `Co-Authored-By: Claude
Opus 4.7 (1M context) <noreply@anthropic.com>`, no push, no `-A`) → **STOP and wait
for the user to compact** before the next milestone. Commits land on `main`.
**Model-free** — all M0–M3 gates run on tiny configs, no GPU (only the M4 bench loads
a model, and it is deferred exactly like #18 M5).

---

## What #153 is — and why it matters

**Production serves DSV4 through the PAGED path, not the #18 arena.** The engine
defaults `PAGED_KV_DEFAULT=True` (`src/quanta/paged/__init__.py:27`,
`shim/omlx.py:988`); the DSV4 batched session builds a `PagedKVCacheManager`, so
`admit`/`step_batch` dispatch to `_admit_paged`/`_step_paged` (`omlx.py:720,733`) and
`make_cache()`→arena is never called (the arena is the **unpaged** batched path only).

**But the paged decode still pays the exact per-stream KV-update loop #18 killed.**
`_step_paged` (`omlx.py:793`) calls `self._rt.step_batch(tokens, paged_states, offsets)`
(`omlx.py:810`). The paged `DSV4Cache` (from `make_paged_state`→`paged_cache`,
`decode.py:439`) has no `.row`, so `_decode_batched_single` (`batched_runtime.py:494`)
takes `arena_path=False` → the per-stream `lcs` branch (`batched_runtime.py:513`):
for each of B streams, `decode_step_*_batched(..., lcs, ...)` runs `_PagedLayerCache.
append_kv` (→ `PagedLatentCacheView.append` → per-stream block write) + reads `kv`
(→ `gather_one`, a per-stream `mx.take` block-gather), then `_pad_stack` re-pads to
`[B,L_max,D]`. **That B-stream Python loop + per-stream gather + pad is the #153 target.**

So the M5 win (`arena/bat` +37% @ B=32, the prod operating point) is **not realized in
production today** — paged leaves it on the table. #153 closes that: ONE batched
block-table scatter write + ONE batched block-table gather read, replacing the loop.

---

## Design — the batched-paged latent store (reuses #18 wholesale)

**Key insight: only the LATENT store is paged.** The derived ckv/ikv/ring are
per-stream in the paged path (restored from boundary snapshots), exactly as the arena's
`_CompArena` batches them. The batched steppers `decode_step_{dense,compressed}_batched`
already take a keyword `arena=` (any object exposing `append_batched(rows,codes)` +
`read_batched(rows)`) and `comp=` (a `_CompArena`). **So if a block-paged store presents
`_KVArena`'s batched interface, the steppers + `_CompArena` drop in unchanged** — the
genuinely new code is the block-table index math.

### New: `_PagedKVArena` (block-table batched scatter/gather over the existing block pool)
Backs `_KVArena`'s interface, but physical location comes from per-stream block tables
instead of a contiguous `[R,L_cap]` arena:

- **Batched write** (ONE scatter): for B streams each appending one token at position
  `pos_s`, physical target is `(blk_id_s, intra_s)` where `blk_id_s =
  block_table[s][pos_s // bs].block_id`, `intra_s = pos_s % bs`. Gather `blk_ids[B]`,
  `intras[B]`; then per component `pool[name][blk_ids, intras, :] = codes[name]` — the
  same 2D fancy-index scatter #18 M0 validated bit-exact (MLX 0.31.2). Replaces the
  B-stream `_write_encoded` loop.
- **Batched read** (ONE gather): build a padded block-id matrix `bids[B, max_nb]`
  (front/zero-padded), `mx.take(pool, bids.reshape(-1), 0)` → `[B*max_nb, bs, C]` →
  reshape `[B, max_nb*bs, C]` → slice `[:, :L_max]`. ONE gather + ONE batched dequant;
  stale padding past each stream's `n_s` is sent to `-inf` by the existing SDPA pad/window
  mask (inert, the #18 argument). Replaces per-stream `gather_one` + `_pad_stack`.
- **Codec verbatim**: reuse `quantize_last_axis`/`dequantize_last_axis` and the existing
  block-pool component layout (`kv_q/kv_s/kv_b` or `kv`, pools `[num_blocks, bs, C]`) — NO
  kernel reimpl (bf16-drift trap).

### COW-free decode (the one subtlety to assert, rule 6)
During DECODE the write-block is always the sequence's PRIVATE growing tail (a fresh
block is `alloc()`'d private when the tail fills; the shared/frozen prefix blocks are
read-only). COW only fires on the first suffix write into a shared partial boundary block
— which happens in `prefill_paged`, not decode. So the batched decode scatter is COW-free.
M0 ASSERTS each target block is non-shared (fail loud) rather than silently corrupt a
shared block; any COW stays per-stream bookkeeping done OUTSIDE the hot scatter (it never
runs in the steady decode loop).

### Dispatch (no new handle type)
`_decode_batched_single` already routes per-stream paged caches through the `lcs` loop.
Add a branch: when the caches are paged (`caches[0].layers[0]` is a `_PagedLayerCache`),
extract the B views' block tables and route to the paged-arena stepper path —
`decode_step_*_batched(arena=paged_set.latent[i], rows=<seqs/views>, comp=paged_set.comp[i])`
with the SAME stepper call. `rows` generalizes from int lease-indices to the per-stream
block-table source (the paged store reads each row's block table per step). The discrete
and #18-arena paths are untouched.

### Equivalence bar (identical to #18)
- **B=1 bit-exact** (`|Δ|==0`); **B≥2 greedy-exact** (`max|Δ|<5e-4`, argmax-stable);
  **AND** `kv_length()`/`n_comp()` match the per-stream paged `lcs` loop.
- The latent read is gather+dequant only (no SDPA reorder at the store level), so the
  `_PagedKVArena` round-trip should be **bit-exact** even B≥2 (the 5e-4 is for the SDPA tail).

---

## Milestones (mirroring #18; M0–M3 model-free, M4 deferred GPU)

- **M0 ✅ DONE — batched scatter/gather on `PagedKVCacheManager` + flag.** Generic over the
  component dict ⇒ serves single-stream latent AND k/v in one primitive (see the M0-DONE
  section above for the design deviation from `_PagedKVArena`). Flag
  `PAGED_KV_BATCHED_DEFAULT` (default OFF, rule 4). Gate
  `parity/dsv4_paged_batched_test.py` green: batched write/read == per-stream
  (`write_one`/`gather_one` and `write`/`gather`) bit-exact across ragged + boundary-crossing
  block tables + a COW case. Regressions + ruff/compileall/lock/diff clean.
- **M1 ✅ DONE — dense stepper on the paged-arena.** `_PagedKVArena`
  (`quanta.dsv4.decode`): a thin per-layer adapter `(manager, seqs, layer)` presenting
  `_KVArena`'s `append_batched`/`read_batched` over the manager's `write_one_batched`/
  `gather_one_batched` (codec-agnostic — forwards, so int8 latent AND bf16 both ride it).
  `decode_step_dense_batched`'s arena path runs UNCHANGED on it (`rows` == `range(B)`). New
  dispatch branch in `_decode_batched_single` routes a paged DSV4Cache's DENSE layers through it,
  **gated on `self._paged_kv_batched`** (← `PAGED_KV_BATCHED_DEFAULT`, OFF, rule 4): paged caches
  already reach this method and take the per-stream `lcs` branch, so the FLAG (not cache type)
  engages the new path — M3 flips it. Compressed paged layers keep the per-stream loop until M2
  (both write the SAME paged latent store; M0 proved batched-scatter == per-stream-write bit-exact,
  so a mixed forward stays exact). Gate (`parity/dsv4_paged_batched_test.py`): dense paged-batched
  == per-stream paged `lcs` loop, **BIT-exact (`max|Δ|=0`)** across ragged B=4 (boundary-crossing,
  3 steps) + B=1 — both paths run the SAME batched SDPA, differing only in how the latent window is
  materialized (masked padding inert), so even B≥2 is exact. Regressions green
  (`dsv4_batched_attention_test`, `dsv4_paged_latent_test` incl. real `_DSV4BatchedSession`
  admit/reuse `|Δ|=0` with the flag off, `dsv4_batched_test`) + ruff/compileall/lock/diff/pytest.
- **M2 — compressed stepper + batched derived.** Reuse `_CompArena` for ckv/ikv/ring, seeded
  from each stream's per-stream paged-cache derived state (the snapshot/restore lifecycle) and
  snapshotted at boundaries. The hard milestone (derived batching × paged boundary snapshots).
  Gate: compressed paged-batched == per-stream paged loop.
- **M3 — wire `_step_paged` + flag default ON + regression.** `_step_paged` leases/uses the
  batched-paged handle, threads `mgr.advance`/`commit` + recurrent snapshots, frees on release;
  dispatch keys off cache type. Flip `paged_kv_batched` default ON after parity. Gate: extend
  `parity/dsv4_paged_latent_test.py` for the batched-paged path + full regression
  (pytest/ruff/compileall/lock/diff).
- **M4 — real-model B-sweep bench (DEFERRED, solo GPU).** Like #18 M5: paged-batched vs
  per-stream paged loop on the real DSV4-Flash bake. Not a correctness blocker (M0–M3 gated
  model-free). Expect the same +Nx @ B that M5 showed for the unpaged arena.

---

## Risks / open questions (resolve as encountered)
1. **Block-table assembly cost.** Building `bids[B,max_nb]` per step is a bounded per-stream
   `block_id` gather (B×max_nb) — accounting, not hot per-token IO (rule 3 OK), but if it
   shows up, cache the block-id arrays on the SeqHandle and extend incrementally per step.
2. **Derived-state ↔ paged snapshot lifecycle (M2).** The arena seeded `_CompArena` from
   per-stream prefill (`seed_comp`); paged seeds derived from `restore_derived` boundary
   snapshots + suffix pooling. M2 must batch that without breaking prefix-reuse bit-exactness.
3. **max_nb / padding growth.** Skewed-length batches pad the gather to the longest stream's
   block count; same dense-padding tradeoff as the arena (noted, acceptable; paged still wins
   on prefix-sharing memory).
4. **`rows` generalization.** Decide whether `_PagedKVArena` takes SeqHandles, views, or a
   prebuilt block-id matrix; keep `_KVArena`'s interface stable so the steppers stay shared.

## File anchors (as of #18 done, `94ae260`)
- `src/quanta/paged/paged_kv_cache.py` — `_write_encoded` (289, per-stream write loop to
  batch), `gather`/`gather_one` (346/371, per-stream read to batch), `truncate` (392),
  `PagedLatentCacheView` (462), `SeqHandle` (block_tables/n_written/length).
- `src/quanta/dsv4/decode.py` — `_PagedLayerCache` (402), `paged_cache` (439); reuse
  `_KVArena` (783), `_CompArena` (1040), `decode_step_{dense,compressed}_batched` (1181/1300).
- `src/quanta/dsv4/batched_runtime.py` — `_decode_batched_single` (477, add paged branch),
  `make_paged_state`/`prefill_paged` (242/251), `paged_kv_spec` (232).
- `src/quanta/shim/omlx.py` — `_step_paged` (793), `_admit_paged` (762), `release` (752/857).
- New: `parity/dsv4_paged_batched_test.py` (model-free M0–M2 gate).

## Gate commands
```bash
uv run --with numpy python -m parity.dsv4_paged_batched_test  # M0–M2 (model-free; M1+ needs numpy fixtures)
uv run --with numpy python -m parity.dsv4_paged_latent_test  # M3 regression (paged latent)
uv run --with numpy python -m parity.dsv4_batched_test       # #18 regression (unchanged)
# Before M3 commit: pytest tests/ -q · ruff check src tests · compileall · uv lock --check · git diff --check
```

## Alternative (smaller, if prefix-sharing is not needed)
If production does not need paged prefix-sharing, the far cheaper path to "arena in prod"
is to default the engine to the **unpaged** batched path (the #18 arena is already its
default). #153 is the right answer only if you want BOTH prefix-sharing AND the KV-loop-kill.
