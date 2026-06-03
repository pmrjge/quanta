---
name: project_paged_batched_153
metadata:
  node_type: memory
  type: project
  originSessionId: 54134555-11aa-4dc5-b309-cf0ebb30e2bd
---

**#153 — batched-paged KV: bring the [[project_kv_arena_18]] loop-kill to the PAGED path.**
Durable handover: repo `PLAN_153.md`. Cadence (standing): single linear thread, NO
subagents; implement → gate green → commit each milestone (named files, trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`, no `-A`, no push)
→ STOP for the user to compact. Model-free M0–M3 (no GPU); M4 bench deferred like #18 M5.

**Why:** production serves the paged keepers through `PagedKVCacheManager` (not the #18
unpaged arena), and paged batched decode STILL pays the per-stream KV-update loop #18 killed
(`_step_paged` → `step_batch` → `_decode_batched_single` `arena_path=False` → per-stream
`lcs` loop). So the M5 win (`arena/bat` +37% @ B=32) is NOT realized in prod. #153 closes it.

**M0 ✅ `62609ba`** — batched block-table scatter/gather landed as **generic methods on
`PagedKVCacheManager`**: `write_one_batched`/`write_batched` (ONE quantize + ONE fancy-index
scatter across all B streams' tail blocks; COW-clones a shared partial tail BEFORE the
scatter, asserts the write block private, rule 6) + `gather_one_batched`/`gather_batched`
(ONE `mx.take` over a padded block-id matrix + ONE batched dequant; tail-pad real-blocks-
first, stale positions ≥ n_written masked by SDPA). **Design deviation from PLAN_153:** NOT a
separate `_PagedKVArena` storage class — the block pool already abstracts k/v-pair vs
single-stream via its component dict, so the batched siblings of `_write_encoded`/`gather`
are **generic over components for free** ⇒ ONE primitive serves DSV4 single-stream latent AND
the k/v keepers, no duplicated alloc/COW. `_PagedKVArena` survives in **M1** only as a thin
per-layer adapter `(manager, seqs, layer)` presenting the steppers' `append_batched`/
`read_batched`. Flag `PAGED_KV_BATCHED_DEFAULT=False` (rule 4; M3 flips). Gate
`parity/dsv4_paged_batched_test.py` (model-free): single-stream == `_LayerCache` bit-exact,
k/v == per-stream paged loop bit-exact, COW case, ragged+boundary-crossing block tables.

**How to apply (multi-model — the standing ask "same optimization for nemotron, qwen, small
model"):** the M0 primitive is on the SHARED manager ⇒ serves every paged keeper at the
storage layer in one shot. **Nemotron** (k/v, paged): applies but MARGINAL — its lever is the
Mamba recurrent state (`BatchedMambaState`, already batched), not the few attn layers' KV
loop. **DSV4** (single-stream latent, paged): the active path (M1–M3). **InternLM2.5** (k/v,
paged — assumed "small model"): applies fully. **Qwen3.5/3.6**: **UNPAGED** (`_make_batched_
session` forces it off paged) ⇒ M0 does NOT touch it; "same optimization for Qwen" is a
SEPARATE task (first check whether its discrete batched decode even loops per-stream vs an
already-rectangular KV cache; if so it needs a #18-arena-style fix in ITS runtime).

**M1 ✅ `c442c31`** — dense stepper on the paged arena. `_PagedKVArena` (`quanta.dsv4.decode`): a
thin per-layer adapter `(manager, seqs, layer)` presenting `_KVArena`'s `append_batched`/
`read_batched` over `write_one_batched`/`gather_one_batched` (codec-agnostic — int8 AND bf16 ride
it). New dispatch branch in `_decode_batched_single` routes a paged DSV4Cache's DENSE layers through
it, **gated on `self._paged_kv_batched`** (← `PAGED_KV_BATCHED_DEFAULT`, OFF, rule 4 — paged caches
already reach this method + take the per-stream `lcs` branch, so the FLAG not cache type engages the
new path; M3 flips). Compressed paged layers keep the per-stream loop until M2. Gate green BIT-exact
(`max|Δ|=0`, ragged B=4 boundary-crossing + B=1 — same batched SDPA both paths); regressions
(`dsv4_batched_attention_test`, `dsv4_paged_latent_test` incl. real session admit/reuse `|Δ|=0`,
`dsv4_batched_test`) + ruff/compileall/lock/diff/pytest clean.

**Multi-model follow-on (user directive: apply the loop-kill to nemotron → internlm2 → qwen3.6, IN
THAT ORDER; one milestone per commit, STOP to compact between).** Started BEFORE DSV4 M2/M3 (user
chose the multi-model order). Shared primitive: `quanta.modeling.batched_attention` got
`_sdpa_padded` (factored tail) + **`batched_decode_attention_padded`** (consumes a pre-padded
`[B,n_kv,L_max,D]` from a paged `gather_batched`) — the k/v sibling of DSV4's `_PagedKVArena`.
- **Nemotron ✅ `833c8a4` (wire) + `7f49bd9` (bench + default GRADUATED ON)** — `_fused_attn_layer`
  already fuses attn; killed its per-stream `KVCache.update()` loop: when caches are `PagedKVCacheView`s
  + `paged_batched` on → ONE `write_batched` + ONE `gather_batched` + `batched_decode_attention_padded`,
  threaded through `batched_decode_step_fused`/`_native`. Gate `nemotron_batched_attention_test.py` §D
  BIT-exact (`max|Δ|=0`, bf16 KV head_dim-agnostic). **Real-model bench `parity/nemotron_paged_batched_bench.py`**
  (int4-g64 120B-A12B, prod paged+form-2 session, distinct prompts, B∈{1,32,48}): greedy-exact loop==loopkill
  at every B (B=1 bit-exact; first quantized k/v write/gather_batched at head_dim=128) AND **loopkill/loop
  = 1.18× @ B=48, 1.15× @ B=32** (per-stream `loop` REGRESSES 126→122 tok/s B=32→48 — Python-loop doesn't
  scale — while loopkill holds 146→144, so the win GROWS with B). NOT marginal. ⇒ graduated to ON via a
  **Nemotron-scoped** `NEMOTRON_PAGED_KV_BATCHED_DEFAULT=True` (NOT the shared `PAGED_KV_BATCHED_DEFAULT`,
  which stays OFF so DSV4/InternLM2.5 are untouched + DSV4 M3 not preempted; rule 4 = one revert flag).
  Bench is solo-GPU (one model at a time). §D `run()` pins the default-ON.
- **InternLM2.5 ✅ `fad71bb` (wire) + `c1db9f6` (bench + default GRADUATED ON).** Dense GQA, 32 layers, no
  recurrent state. Killed the per-stream KV `.update()` loop in BOTH `decode_batched` paths (bf16
  `InternLM2Model` model.py + packed `_PackedModel` runtime.py — identical tail) by routing through the NEW
  single-sourced shared helper **`batched_decode_attention_kv`** (`modeling/batched_attention.py`):
  per-stream `.update()` loop OR (`paged_batched`+paged views) ONE `write_batched`+`gather_batched`+padded
  SDPA. `paged_batched` threaded `_paged_kv_batched` → step_batch → delegator → inner. Gate
  `internlm2_batched_attention_test.py` §C BIT-exact (`max|Δ|=0`, FULL decode_batched paged loop-kill ==
  per-stream paged loop, B=1+ragged B=3 boundary-crossing, bf16) + pins the default. **Real-model bench
  `parity/internlm2_paged_batched_bench.py`** (int8-g64 7B-Chat-1M, prod paged `_InternLM2BatchedSession`,
  distinct prompts, B∈{1,32,48}, drives raw token-id lists — no tokenizer): greedy-exact loop==loopkill at
  every B (B=1 bit-exact 46.3/45.9; first QUANTIZED int8-g64 k/v write/gather_batched at head_dim=128 — §C
  used bf16) AND **loopkill/loop = 3.20× @ B=32 (104→332 tok/s), 3.16× @ B=48** (per-stream `loop` REGRESSES
  104→102 B=32→48 while loopkill holds flat ~332→322 ⇒ win doesn't fade with B). FAR bigger than Nemotron's
  (+15% @ B=32) because InternLM2.5 is DENSE — ALL 32 layers are attention (Nemotron trims only 8 `*`).
  Active 9.5 GiB / peak 10.3 GiB @ B=48 (int8 weight floor; fits trivially). ⇒ graduated to ON via an
  **InternLM2.5-scoped** `INTERNLM2_PAGED_KV_BATCHED_DEFAULT=True` (NOT the shared `PAGED_KV_BATCHED_DEFAULT`,
  which now governs **DSV4 only** + stays OFF so DSV4 M3 not preempted; rule 4 = one revert flag). §C
  `run()` pins the default-ON. Nemotron's `_fused_attn_layer` still inlines the equivalent (didn't refactor
  — minimal diff; could dedupe later). **Next: Qwen3.6 (LAST, big — unpaged + hybrid).**
- **Qwen3.6 (`qwen35`) — LAST, big. UNPAGED + hybrid (45 GDN recurrent + 15 GQA, 3:1), serves at B=4.**
  `batched_decode_step` looped the WHOLE mixer per stream (`for s in range(b)`); kill GQA (M1) then GDN
  (M2, the bigger lever — 3× the layers). **M1 ✅ `ee305dc`** — batched GQA loop-kill for the SERVING
  decode step. New `Qwen35Attention.decode_step_batched` (attention.py): batched q/k/v/o projections
  (each weight read ONCE for all B — the bandwidth win), a **per-stream RoPE kernel loop** (offset AND
  dynamic-YaRN `inv_freq` differ per stream once any crosses the 262K native window — bf16-drift trap,
  loop the exact `mx.fast.rope`, never a batched reimpl), then the SHARED #153 primitive
  **`batched_decode_attention_kv`** (REUSED from InternLM2.5/Nemotron — unpaged discrete `KVCache`
  exposes `.update()→(k_all,v_all)`, so it takes the bounded per-stream `.update()` + ONE padded SDPA
  branch; paged_batched=False). ⇒ NOT net-new for GQA — only the M2 batched GDN state is genuinely
  net-new. Flag **`QWEN35_BATCHED_LOOPKILL_DEFAULT=False`** (Qwen-scoped, UNPAGED — not the shared paged
  flag; rule 4, M3 flips): `batched_decode_step(...,loopkill=)`; `step_batch` passes `self._loopkill`;
  **`prefill`/`__call__` pin loopkill=False** (single-stream generate/spec contract stays on the proven
  bit-exact path); **`batch_step` (tree-spec verify) untouched** (needs per-replica rollback). Gate
  `parity/qwen35_batched_loopkill_test.py` (model-free, reuses qwen35_batched_test's tiny builder):
  greedy-token agreement loop-kill ON == per-stream loop OFF — **B=1 BIT-exact (|Δ|=0, all-zero pad mask
  == mask=None here)**, ragged B=3 greedy-exact 1e-6, + a restructure-regression guard (OFF == single-
  stream). Regressions: qwen35_batched_test (default OFF) + tree_verify green; ruff/compileall/lock/diff/
  pytest clean. **M2 ✅ `17c14dd`** — batched GDN (linear-attn) loop-kill, the bigger lever (45 of 60
  layers). KEY REALIZATION: `GatedDeltaNet.__call__` decode is ALREADY fully batched over the leading B
  axis with NO cross-row op (conv window, gated-delta recurrence, both gated RMSNorms, all projections
  act per-row) ⇒ M2 is state PLUMBING, not new compute (NOT a net-new `BatchedGDNState` recurrence as
  the plan guessed). New module-level `_gdn_step_batched(mixer, lcs, h_norm)` (batched_runtime.py, sibling
  of `_gdn_step_through_cache`; NOT a mixer method — GDN batching is Qwen-only state plumbing, unlike
  GQA's cross-model shared SDPA helper): gather B streams' `(conv,recurrent)` into `[B,...]` (seed zeros
  per-row where `conv_state is None` — the fresh-stream path, bit-identical to the per-stream seed), ONE
  `m(h_norm, state, conv_state)` recurrence (in_proj_qkv/out_proj/conv read ONCE for all B — bandwidth
  win), scatter+`commit(conv_out[s:s+1], rec_out[s:s+1])` per `_GDNLayerState` (offset + snapshot ring
  preserved). Dispatch restructured to nest GDN+GQA under one `if loopkill:` (shared input-norm +
  residual; per-stream `else` unchanged). PARITY: GDN is **B=1 BIT-exact** (b==1 path is a strict
  `[1,1,hidden]` passthrough — no concat) but **B>1 fp-tolerance ~1e-7** (the fp32 projection-GEMM
  batch-M accumulation reorder: `[B,1,h]@W` row-s ≠ `[1,1,h]@W`, measured 3.8e-6 standalone, confirmed
  benign not a bug; bf16 matmul was 0 — GDN runs fp32 so it shows; NO SDPA softmax to reorder ⇒ tighter
  than GQA), greedy-stable. ⚠️ My initial "bit-exact for all B" claim was WRONG — the §GDN gate caught it
  (rule 6), corrected to B=1-exact/B>1-fp-tol in all docstrings. MEMORY: committed state slices are MLX
  VIEWS (confirmed empirically: holding `big[3:4]` retains the full `[B,...]` parent, 134MB for a 4MB
  row); SAFE because the snapshot ring is depth-bounded AND continuous-batching streams commit in lockstep
  ⇒ ≤ `snapshot_depth` recent batched buffers ever retained (== per-stream total), buffers age out within
  depth steps, no stale-row pinning (active streams never pause). Gate §GDN: `_gdn_step_batched` vs
  `_gdn_step_through_cache` over B seeded with DISTINCT states incl. a fresh zero-seed stream — output
  residual AND committed `(conv,recurrent)` + offset; B=1 |Δ|=0, B=3 out 1.2e-7 / state 2.4e-7 (<1e-4);
  path-exercised now requires ≥1 GDN layer. Regressions green. **M3/M4 (bench-then-graduate, user chose
  "bench first" over flip-first this session — matching the cohort): the real-model bench KILLED the
  graduation — the Qwen mixer loop-kill is NOT greedy-exact on the real 40-layer (30 GDN+10 GQA)
  int4g64 bake.** Extended `parity/qwen35_batched_bench.py` to the cohort pattern (loop vs loopkill,
  greedy-exact + tok/s, flips only `model._loopkill`, drives `step_batch`). Result: B=1 greedy-exact
  1.02× (bit-exact anchor holds); **B=4 DIVERGES** (stream-1 step-0 flip). Probes proved it's a REAL bug
  not a near-tie: **|Δlogit| up to 1.30 >> LOGIT_TOL 5e-3**, per-layer post-mixer |Δ| compounds
  GEOMETRICALLY from layer 1 (~1e-4 → 1e-2 over 8 layers, data-dependent — stream 3 drifts first). **ROOT
  CAUSE = the bf16-drift trap:** Qwen `Qwen35ResidentModel` DEQUANTIZES the GDN/GQA projections
  (`in_proj_qkv/a/b/z`, `out_proj`, `q/k/v/o_proj`) to dense bf16 `nn.Linear` (`runtime.py:76`
  `getattr(m,proj).weight=la[...]`, `la`=dequantized); the loop-kill BATCHES them (`[B,1,h]@W`) and a
  **dense bf16 matmul reorders across batch-M** — micro-test `[8,1,4096]@[4096,12288]`: dense bf16
  max|Δ|=**1.0**, dense fp32 4.4e-4, but **quantized int8/int4 g64 = 0.0 BIT-EXACT** (the MoE is bit-exact
  for this reason — gather_qmm). The fp32 `gdn_step` recurrence is pure elementwise + `mx.sum(axis=2)` =
  bit-exact across B (EXONERATED); RoPE is correctly looped; padded-SDPA reorder is a smaller secondary
  source. **COHORT DIFFERENCE (why InternLM2.5 got 3.2× greedy-exact, Qwen can't):** InternLM2.5's prod
  path is the PACKED `runtime.py` (`_qmm`=`mx.quantized_matmul`, projections kept QUANTIZED ⇒ batch-M
  bit-exact); Qwen dequantizes ⇒ drifts. Throughput was SMALL anyway (~**9% @ B=4**, the prod operating
  point; B=1 1.02×) — exactly PLAN_153's "small win" prediction. **DECISION: do NOT graduate;
  `QWEN35_BATCHED_LOOPKILL_DEFAULT` STAYS OFF** (the per-stream path `_gqa/_gdn_step_through_cache` is
  bit-exact — the proven default). M1/M2 loop-kill code stays behind the flag (rule 4, parity-gated at
  tiny dims only). **USER DECISION (this session): option B + operating point re-pinned B=32.** Build a packed/quantized-
  projection Qwen runtime (mirror InternLM2 `_PackedModel`/`_qmm`, rule 1 via `nn.QuantizedLinear`) so the
  mixer projections (`in_proj_qkv/a/b/z`+`out_proj`, `q/k/v/o_proj`) run `mx.quantized_matmul` = batch-M
  bit-exact ⇒ loop==loopkill greedy-exact; then re-bench at B=32 + graduate. MoE (gather_mm matvecs) +
  `gdn_step` (fp32 elementwise) are already batch-invariant — DON'T touch. Two coupled graduations (rule
  4): `packed` default flips after packed-vs-bf16 forward parity; `QWEN35_BATCHED_LOOPKILL_DEFAULT` flips
  after the B=32 bench; enforce loopkill⇒packed. **HANDOFF WRITTEN for a fresh agent: `PLAN_153.md`
  "Qwen3.6 — option B" section (finding + root cause + design + milestones M0–M4 + file anchors + gates +
  kick-off prompt).** Bench `parity/qwen35_batched_bench.py` COMMITTED this session (cohort pattern,
  fails-loud, operating point B=32, the now-disproven argmax-stable over-claim corrected); throwaway probes
  deleted. Reusable MLX fact (CORRECTED by option-B M0 `c503657`): **`mx.quantized_matmul`/`mx.gather_mm`
  are batch-M bit-exact ONLY for M≤~10 (a per-row gemv kernel); at B≥12 a tiled GEMM REORDERS the
  K-reduction (bf16 catastrophic ~2.25/proj, fp32 benign ~7e-4). dense bf16 reorders at ANY B>1 (~1.0),
  dense fp32 ~4.4e-4.** The bench-first gate did its job — it stopped a
  silent bf16-drift regression from shipping. (Rejected: (A) abandon — per-stream is bit-exact but forgoes
  the B=32 win; a PARTIAL loop-kill that loops projections + batches only SDPA/state yields <9%.)
  **option-B M0 ✅ `c503657`** (model-free batch-M proof, `parity/qwen35_batched_loopkill_test.py` §M0 +
  the `_chunked_qmm` primitive): the PLAN premise was TOO STRONG (see the corrected MLX fact above). A
  FULL-BATCH packed loop-kill is NOT bit-exact at B=32 — full-batch quantized reorders [B32=2.25]; the
  prior "bit-exact" micro-test used exactly B=8 and extrapolated. **USER DECISION = chunked sub-batch of
  ≤8**: chunk the loop-kill projections into ≤8-row slices (4 chunks @B=32), each an M≤8
  `quantized_matmul` (bit-exact regime), == per-stream M=1 BIT-FOR-BIT at ANY B. M0 gate (root-cause
  shape [B,1,4096]@[4096,12288]): chunked-8 [B1/4/8/32=0] BIT-EXACT (the fix), full-batch quantized
  [B32=2.25] (why chunk), dense-bf16 [B4=3e-2 B8=1e-1 B32=1.0] (the original bug). `_chunked_qmm` lives in
  the test; M1/M2 build it into the packed steppers. **Revised milestones: M1 packed+chunked GDN, M2
  packed+chunked GQA, M3 wire + graduate `packed` default, M4 solo-GPU re-bench @B32 + graduate
  `QWEN35_BATCHED_LOOPKILL_DEFAULT` (enforce loopkill⇒packed).** Gate self-protects the chunk: if MLX
  shifts the gemv→GEMM threshold <8, both chunk_exact + full_threshold fail loudly.
  **option-B M1–M4 ✅ COMPLETE (`cf299c3`/`9350482`/`0bacba8`/`6231a1d`). #153 Qwen3.6 DONE.** M1
  packed+chunked GDN: `runtime._load_block(packed)` builds `in_proj_*`/`out_proj` as `nn.QuantizedLinear`
  (`_load_quant_triplet`/`_packed_linear` mirror InternLM2); `_gdn_step_batched` applies the recurrence in
  ≤`QWEN35_LOOPKILL_CHUNK`=8-row chunks (every GDN op per-row ⇒ chunked==full==per-stream bit-for-bit;
  chunking only bounds the projection matmul's M). M2 packed+chunked GQA: `_load_block` packs `q/k/v/o`;
  `Qwen35Attention.decode_step_batched(...,chunk)` chunks a new `_project_chunked` + `o_proj` ≤8 (per-stream
  RoPE + ONE fused padded SDPA across ALL B unchanged — projections bit-exact, only SDPA softmax reorders).
  M3 wired `packed` through `Qwen35ResidentModel`/`Qwen35BatchedResidentModel`/shim `from_inner`;
  `loopkill ⇒ packed` enforced (`_check_loopkill_requires_packed`, at construction AND every step_batch);
  `qwen35_forward_test` packed==bf16 GREEDY-EXACT (|Δ|=1.3e-6 — SAME codes, only the matmul kernel differs,
  so teacher-forced ppl can't distinguish ⇒ greedy-exact is the model-free graduation gate); **graduated
  `packed=True` default**. M4 real-model bench (`parity/qwen35_batched_bench.py`, Qwen3.6-35B-A3B int4g64,
  loaded PACKED — the packed artifact LOAD, untested model-free, WORKS on the real bake): **greedy-exact
  loop==loopkill at EVERY B AND a win — 1.63× @ B=32** (0.98/1.20/1.45/1.67× @ B=1/4/8/16; the win GROWS
  with B; the dequant path DIVERGED at B≥2 |Δlogit|≈1.3 — option B fixed it); peak 79.3 GiB @B=32 (the
  bf16-dequant MoE experts dominate resident — PRE-EXISTING, NOT packed by option B which scopes only the
  mixer projections — ≪490 GiB). **Graduated `QWEN35_BATCHED_LOOPKILL_DEFAULT=True`**; `from_inner` gained a
  `loopkill` override (`None`⇒the graduated global, so the serving `from_inner` at shim/omlx inherits the
  loop-kill paired with the inner's `packed`; a bf16 per-stream test passes `loopkill=False` to construct
  without packing — else `loopkill ⇒ packed` raises at construction). The deliberate **B=4 latency-first
  ORCHESTRATOR pin** (`shim BEST_BATCH` `qwen3_5`, #26) is LEFT UNTOUCHED — the loop-kill helps at any B≥2
  (1.20×@B4); B=32 is the bench/graduation point, NOT the serving pin (Qwen serves single-stream-ish, sweeps
  to `SERVING_BATCH_CAP=32` if a session pins higher). **Reusable MLX fact:** chunked-≤8 `mx.quantized_matmul`
  == per-stream M=1 BIT-FOR-BIT at any B — the batch-M loop-kill fix for a QUANTIZED (not dequant-bf16)
  projection; the cohort (InternLM2/Nemotron) avoided the bug by keeping projections packed via `_qmm`,
  Qwen dequantized ⇒ drifted ⇒ option B re-packs them.

Also committed (item-1, #18 follow-on): **`14e01b9`** B=48 within-noise gate (`dsv4_b48_noise.py`) +
honest bench verdict + PLAN.md M5 note.

**DSV4 #153 core M2 ✅ `35dcd78`** — compressed paged-batched stepper. **DESIGN DEVIATION from the
PLAN's "reuse `_CompArena`, batch the derived" guess** (smallest-safe-change + rule-6 honest scope):
the #153 lever is the per-stream LATENT write/gather loop, so M2 batches ONLY the latent (rides M1's
`_PagedKVArena` — ONE block-table scatter + ONE gather) and keeps the derived ckv/ikv/ring PER-STREAM
on each `_PagedLayerCache`. That (a) is the smallest safe change, (b) leaves the paged boundary-snapshot
lifecycle UNCHANGED ⇒ **M3's `_step_paged` needs NO batched-derived-snapshot machinery — the "hard part"
the PLAN flagged is SIDESTEPPED, not solved** (derived stays exactly where the snapshot/restore lives),
(c) stays fully BIT-exact: latent via M0, derived via the SAME per-stream `_maybe_pool` (no batched-pool
reorder). Why safe: `_maybe_pool`/`_push_ring` read only the raw-hidden RING + ckv/ikv, never `lc.kv`
(the latent), so splitting latent-batched from derived-per-stream is clean. New **paged-hybrid path** in
`decode_step_compressed_batched` (`arena`+`rows`+`lcs` given, `comp=None`): `arena.append_batched` +
per-stream `_maybe_pool` + `arena.read_batched`; compressed dispatch now **3-way** (full-arena #18 /
paged-hybrid #153 M2 / per-stream ref; full-arena branch keyed on `comp is not None`).
`_decode_batched_single` `elif paged_path:` now covers DENSE (M1) AND COMPRESSED (M2), still gated on
`self._paged_kv_batched` (OFF, rule 4; M3 flips). Gate
`parity/dsv4_paged_batched_test._run_compressed_stepper`: compressed paged-batched == per-stream paged
loop **BIT-exact (max|Δ|=0)** across ragged B=4 + B=1, ratio-4 +indexer AND ratio-3 no-indexer, matching
`n_comp` + latent lengths. Regressions green (dsv4_batched_attention_test full-arena compressed intact,
dsv4_batched_test #18 arena serving, dsv4_paged_latent_test |Δ|=0); pytest/ruff/compileall/lock/diff
clean. GATE FIXTURE NOTE: seeds each stream's raw-hidden ring identically for ref & bat (a fresh ring
underflows `_pool_one_window`; pool realism is gated in dsv4_batched_attention_test §B). `_mgr` got an
`n_layers` arg (default 1) so the compressed gate stores latent at the regime's layer index.
(`PLAN_153.md` status line was bumped to "✅ #153 COMPLETE across ALL keepers" in-repo at `861ccc1` after
M4 — the earlier in-repo staleness is resolved.)

**DSV4 #153 core M3 ✅ `d19a254`** — graduate batched-paged KV default ON. **NO `_step_paged`/omlx.py
change was needed** (M2's prediction held): the M1/M2 dispatch in `_decode_batched_single` already routes
a paged DSV4Cache through `_PagedKVArena` gated on `self._paged_kv_batched`, and `_step_paged` already
drives it via `step_batch` — so M3 = flip `PAGED_KV_BATCHED_DEFAULT` False→True (DSV4-scoped; read at
construction) + add the SESSION-level integration gate + fix the two now-stale "OFF by default / M3 flips
it" comments in batched_runtime.py + rewrite the flag's defining comment in paged/__init__.py. Because M2
kept the derived ckv/ikv/ring per-stream, the boundary-snapshot lifecycle is unchanged ⇒ NO batched-
snapshot machinery (the PLAN's "hard part" stayed sidestepped). **Gate `parity/dsv4_paged_latent_test.py`
§C** (new `_run_engine_batched`): drives the REAL `_DSV4BatchedSession` decode loop (admit → `_step_paged`
→ `step_batch` → `_decode_batched_single` paged path) over ragged B=3 crossing block boundaries DURING
decode, TWO bars — (1) **ON==OFF |Δ|=0.00e+00 BIT-exact** (flipping the flag is output-inert vs the
per-stream paged loop: both paged, same batched SDPA; latent IO differs only by the M0-equal
scatter/gather); (2) **ON≈discrete |Δ|=2e-3 argmax_mismatch=0 GREEDY-exact** vs the single-stream ground
truth. `decode_snapshots=5` proves the boundary path fired mid-decode. KEY GOTCHA: §A/§B hit 0.00e+00
because they compare single-stream-vs-single-stream (only storage differs); §C is the FIRST to compare
batched-vs-single-stream, which is bf16 GREEDY-exact (<1 ULP reorder, compounds over depth/steps —
[[feedback_batched_rope_bf16]]), NOT bit-exact. My first run wrongly gated bar 2 at 5e-4 and "FAILED" at
2e-3 — the fix was to use the correct arbiter (argmax stability for B≥2, + a generous gross-error guard),
NOT to loosen a tight logit bar. Regressions green: dsv4_paged_batched_test (M0–M2 stepper bit-exact),
dsv4_batched_attention_test (#18 arena |Δ|bat=0), dsv4_batched_test (#18 serving); pytest/ruff/compileall/
lock/diff clean.

**DSV4 #153 core M4 ✅ `cb2476b`** — real-model paged-batched bench (the deferred solo-GPU bench, paged
sibling of [[project_kv_arena_18]] M5). NEW `parity/dsv4_paged_batched_bench.py` drives the REAL
`_DSV4BatchedSession` paged latent decode over B∈{1,32,48} with DISTINCT prompts (defeats prefix-dedup ⇒
each stream its own full latent KV, the real B-stream load), flipping ONLY `_paged_kv_batched` between loop
(per-stream `lcs` latent: append_kv + gather_one + _pad_stack) and loopkill (#153: ONE `write_one_batched`
scatter + ONE `gather_one_batched` gather, then the SAME batched SDPA). Result on DSV4-Flash int4-g64
(43 layers, ~180 GiB, packed_experts):

| B | loop tok/s | loopkill tok/s | ratio | tok |
|---|---|---|---|---|
| 1 | 6.5 | 6.5 | 1.01× | bit-exact |
| 32 | 69.2 | 77.9 | **1.13×** | bit-exact |
| 48 | 80.2 | 90.6 | **1.13×** | bit-exact |

**loop == loopkill BIT-exact at EVERY B** (`tok=ok`) — NOT merely greedy-exact: both paths run the one
batched SDPA, only the latent store materialization differs and that is |Δ|=0 (M0). **+13% decode tok/s
@ B=32 (prod operating point) & B=48**, holding (loop does NOT regress past B=32 here, unlike
Nemotron/InternLM2.5, but loopkill keeps its +13% lead). SMALLER than #18 M5's unpaged arena/bat +37%
because DSV4 batches ONLY the latent (derived ckv/ikv/ring stay per-stream, M2 design) and MoE dominates
decode FLOPs — the +13% is the latent-loop slice in isolation. Peak ~192 GiB (≪490 ceiling). Flag was
already ON since M3 (model-free); M4 confirms on the real 43-layer forward + quantifies the win (rule 4).
Comment-only update to paged/__init__.py records M4 done at the flag's authoritative comment; no code/flag
change. Solo-GPU (the run loaded ONLY DSV4-Flash). Gate:
`uv run python -m parity.dsv4_paged_batched_bench` (loop==loopkill bit-exact + win@B32, fails loud).

**DSV4 #153 core M0–M4 COMPLETE.** No DSV4 #153 work remains. The whole #153 (multi-model KV loop-kill) is
now done across ALL keepers — DSV4 (this), Nemotron, InternLM2.5, Qwen3.6 — each graduated ON via its
scoped flag with a real-model bench. `PLAN_153.md` status bumped to "✅ #153 COMPLETE across ALL keepers"
in-repo at `861ccc1` (doc-only: top Status block + DSV4 per-model line + DSV4 M2/M3/M4 milestone entries).
