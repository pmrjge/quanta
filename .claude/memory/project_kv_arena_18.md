# #18 — batched KV arena (kill the per-stream KV-update IO loop)

**Durable handover: repo `PLAN.md`.** Ephemeral plan: `~/.claude/plans/moonlit-doodling-gray.md`.

## What / why
DSV4 batched decode grows `B` ragged per-stream `_LayerCache` streams with a Python
loop (per-stream `append_kv` quantize+`concatenate`) + a `_pad_stack` readback every
step. #18 replaces them with a persistent `max_batch`-sized **batched KV arena** +
slot→row free-list: hot path = ONE scatter write (`arena[rows,cols,:]=codes`) + ONE
gather read (`mx.take` + one batched dequant). Output-equivalent, flag-guarded
(`kv_arena`, default OFF until M4).

## Cadence (STANDING user instruction — do not violate)
Single linear thread, **NO subagents/agents/workflows**. Implement → parity green →
commit each milestone (named files, trailer `Co-Authored-By: Claude Opus 4.7 (1M
context) <noreply@anthropic.com>`, no push, no `-A`) → **STOP and wait for the user
to compact** before the next milestone. Commits land on `main`.

## Status
- **M0 ✅ `41a4d0f`** — `_KVArena`/`_KVArenaSet`/`_ArenaLayerView` + `kv_arena` flag; gate `parity/dsv4_kv_arena_test.py`.
- **M1 ✅ `6f33cc1`** — `decode_step_dense_batched` arena path via keyword-only `arena`/`rows`; gate extends `parity/dsv4_batched_attention_test.py` dense case.
- **M2 ✅ `05d1171`** — `_push_ring_batched` + shared `_ring_cap` (fixed-width `[R,cap,dim]`, zero-padded front, newest at `[:,-1]`); model-free ring check in `dsv4_kv_arena_test.py`.
- **M3 ✅ `bf7af6b`** — `decode_step_compressed_batched` arena path (keyword-only `arena`/`rows`/`comp`): `_compressed_update_arena` = ONE latent scatter + ONE ring roll (`_CompArena.roll_ring`: gather/roll/scatter) + ONE compute-all pool (`_pool_one_window_b`, per-row prev-valid + per-row window-start RoPE) masked-scattered into ckv/ikv `_KVArena`s. New: `_CompArena` (ckv/ikv `_KVArena` + `[R,cap,dim]` ring), `_pool_one_window_b`; `_decode_indexer_select_batched` now takes a padded `ikv` array (both stores feed shared SDPA tail). The `for s in range(b)` body is gone on the arena path.
- **M4 ✅ `e08888d`** — flipped `kv_arena` **default ON**. `_KVArenaSet` owns one `_CompArena` per ratio>0 layer (`comp_specs`, default None=latent-only; `free` resets comp rows). `_ArenaCacheHandle` = a leased row presented as a DSV4Cache-shaped object (`__getitem__`→`_ArenaLayerView`, `.offset`, `.row`, `seed_comp`). **Approach B handoff:** `make_cache`→handle; `prefill` runs the UNCHANGED single-stream forward (latent→arena via the views; derived ckv/ikv/ring→per-object) then `seed_comp` migrates the derived state into `_CompArena` by copying STORED codes VERBATIM (`_KVArena.seed_row`/`_CompArena.seed_row`/`_seed_ring`, ring right-aligned) — bit-exact, NO re-quantize (the bf16-drift trap). `_decode_batched_single` dispatches handles→arena steppers (ONE scatter + ONE gather/layer); `free_cache`+session `_DSV4BatchedSession.release`+`batched_generate` (free-on-retire + try/finally) return the row. Gate: `dsv4_batched_test` end-to-end arena serving (make_cache→prefill→step_batch == single-stream, n_comp match, lease/free/realloc) + full suite green.
- **M5 ✅ `f4935b5`** — real-model B-sweep bench (`parity/dsv4_batched_bench.py`), rewritten to a **3-path** sweep (looped `_fused=False`+discrete `DSV4Cache` / batched `_fused=True`+discrete / arena `_fused=True`+`make_cache()` handle — dispatch keys off cache TYPE, one resident model serves all three) + a real-model **greedy-exact** correctness gate (`looped==batched==arena`). Ran on the DeepSeek-V4-Flash int4-g64 bake (~180 GiB, solo GPU) thru **B=32** (prod operating point; B=48/64 skipped — prefill is token-by-token ⇒ seeding O(B), curve already monotone). EVERY row `tok=ok` (arena greedy-exact vs per-stream loop on the REAL model — first live int8-latent arena exercise, real head_dim=128). `arena/bat` (the #18 KV-loop-kill, isolated — same fused attn, only the KV store differs): 0.96→1.02→1.07→1.13→1.24→**1.37× @ B=32** = +37% decode tok/s (108.5 vs 79.0). `bat/loop` (attn-batching) 9.49× @ B=32. Mem 184/191 GiB < 220 wired. **#18 DONE (M0–M5).** Optional later: `...dsv4_batched_bench 48,64` (solo GPU) for the asymptote.

## Key invariants / gotchas
- Equivalence bar: **B=1 bit-exact**, **B≥2 greedy-exact** (`max|Δ|<5e-4`, argmax-stable) AND `kv_length()`/`n_comp()` match.
- Bit-identical contents: reuse `cache_quant.quantize_last_axis`/`dequantize_last_axis` verbatim (NO kernel reimpl — bf16-drift trap). Affine int-bits over LAST axis is row-independent ⇒ batched `[B,1,D]` quantize == B separate quantizes row-for-row.
- Stale/zero arena padding past a row's length is sent to `-inf` by the SDPA window/pad mask ⇒ inert. `read_batched` slices `[:, :L_max]`, `L_max=max(lengths[rows])`.
- MLX 0.31.2: `arena[rows,cols,:]=vals` bit-exact 2D fancy-index scatter; `mx.take(...,axis=0)` gathers; `mx.scatter` does NOT exist.
- M1/M3 empirical: arena vs `_LayerCache` batched is **bit-exact (0.0)** — identical codes, padding, pool, SDPA; arena vs per-stream loop ≤1.83e-07.
- **M3 prev-window validity (the subtlety):** per-stream `_maybe_pool` decides the overlap `prev` window via `lc.ring.shape[1] >= 2*ratio`. The fixed-width batched ring is ALWAYS `cap` wide (zero-padded front), so that test does NOT translate. Equivalent per-row condition: `prev_valid = overlap and (offset//ratio >= 1)` (i.e. closing window `c>=1`); `c==0` ⇒ window-0 pad (`kv=0`, `score=-inf`) via `mx.where`, matching `_pool_one_window`'s `prev is None`. Per-row window-start RoPE row gathered at `(offset//ratio)*ratio`. Pooled values are bit-identical because matmul/softmax/RMSNorm/RoPE are per-row independent.
- M3: pool is **compute-all then masked-scatter** — pool ALL B rows, append only `closing=[(off+1)%ratio==0]` rows (gather `ck_all[closing]`) into ckv/ikv + bump n_comp; discarded non-closer lanes can't NaN (after the roll every active row has ≥1 valid pos ⇒ no all-`-inf` softmax row). ckv/ikv grow in lockstep ⇒ share n_comp.
- M3: ring persistence = `[R,cap,dim]` arena; hot path gathers active rows (`mx.take`), rolls (`_push_ring_batched`), scatters back (`ring[rows_arr]=active`, 1-D fancy-index — verified surgical, idle rows untouched).
- Test note: `parity/dsv4_batched_test._cfg()` has `HEAD_DIM=8` (<32 quant floor) ⇒ latent auto-resolves to bf16; M1 stepper parity exercises the bf16 arena path (int8 codec round-trip is M0's gate).
- Scope: arena is the NON-paged batched-serving decode path only; tree-spec `replicate`/`_copy` + the paged path stay on per-stream `DSV4Cache`.
- **M4 dispatch = cache TYPE, not the flag:** `_decode_batched_single`/`step_batch` use the arena iff the cache is an `_ArenaCacheHandle` (has `.row`); a discrete `DSV4Cache` always takes the per-stream loop even on a `kv_arena=True` runtime. The flag's only job is what `make_cache` returns. So EVERY `make_cache` caller (`batched_generate`, the omlx session, the M5 bench) MUST `free_cache` the handle on retire/release, else the arena caps total served requests at `max_batch` (exhaustion is loud, rule 6). `free_cache` is a no-op for a discrete cache, so the non-arena path is unaffected.

## Gates
`uv run python -m parity.dsv4_kv_arena_test` (M0/M2) · `uv run --with numpy python -m parity.dsv4_batched_attention_test` (M1/M3/M4) · `uv run --with numpy python -m parity.dsv4_batched_test` (regression). Before M4 commit: pytest + ruff + compileall + `uv lock --check` + `git diff --check`.
