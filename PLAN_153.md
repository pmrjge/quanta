# PLAN_153.md — active task handover (#153: batched-paged KV — bring the #18 loop-kill to the paged path)

> Durable, repo-tracked handover for the NEXT task. Read `CLAUDE.md` first (permanent
> rules + model facts), then `PLAN.md` (#18, the batched KV arena — DONE M0–M5; the
> machinery #153 reuses), then this. **Status: ✅ #153 COMPLETE across ALL keepers (DSV4, Nemotron,
> InternLM2.5, Qwen3.6).** DSV4 core M0–M4 DONE: M0 `62609ba` (storage primitives), M1 `c442c31` (dense
> paged stepper `_PagedKVArena`), M2 `35dcd78` (compressed/latent-only stepper, derived per-stream), M3
> `d19a254` (graduated `PAGED_KV_BATCHED_DEFAULT` ON, bit-exact model-free), M4 `cb2476b` (real-model
> paged-batched bench — loop==loopkill BIT-exact at B∈{1,32,48} AND +13% decode tok/s @ B=32/48). Multi-model
> loop-kill (user's order nemotron→internlm2→qwen3.6): Nemotron DONE + default GRADUATED ON
> (real-model bench +18%@B48); InternLM2.5 DONE + default GRADUATED ON (real-model bench **3.20×@B32**,
> scoped flag); **Qwen3.6: option B COMPLETE — M0–M4 ✅** (`c503657`/`cf299c3`/`9350482`/`0bacba8`/M4).
> The packed+chunked runtime fixed the dequant-bf16 batch-M reorder: the real-model B-sweep bench
> (`parity/qwen35_batched_bench.py` on the Qwen3.6-35B-A3B int4-g64 bake) is **greedy-exact loop==loopkill
> at every B AND a win — 1.63× @ B=32** (1.20/1.45/1.67× @ B=4/8/16; the dequant path diverged at B≥2,
> |Δlogit|≈1.3). `QWEN35_BATCHED_LOOPKILL_DEFAULT` + packed both GRADUATED ON; `loopkill ⇒ packed`
> enforced. See "Qwen3.6 — option B". **Follow-up task (NOT started):** the option-B bench peaks at
> **79.3 GiB for a 21 GiB int4 artifact** because the routed experts are still **dequantized to bf16** —
> keep them packed + `gather_qmm` (~79 → ~30 GiB). Full handover: **`PLAN_qwen35_experts.md`**.

---

## Qwen3.6 — option B: packed-projection runtime (CURRENT ASK, B=32)

**Status: COMPLETE — M0–M4 ✅.** The prior session's M1 (`ee305dc`) + M2 (`17c14dd`) built the hybrid
loop-kill behind `QWEN35_BATCHED_LOOPKILL_DEFAULT=False`; the real-model bench **caught a real bug** (the
loop-kill was NOT greedy-exact at B>1 — dequantized dense-bf16 projections reorder across batch-M). Option
B (this work) built the packed+chunked runtime that fixes it and graduated both defaults:

- **M0 ✅ `c503657`** — chunked-≤8 `mx.quantized_matmul` is batch-M bit-exact (the mechanism).
- **M1 ✅ `cf299c3`** — packed+chunked GDN mixer (`_load_block(packed)` + chunked `_gdn_step_batched`).
- **M2 ✅ `9350482`** — packed+chunked GQA mixer (`_project_chunked` + chunked `o_proj`).
- **M3 ✅ `0bacba8`** — wired `packed` through; `loopkill ⇒ packed` enforced; `qwen35_forward_test`
  packed==bf16 greedy-exact (|Δ|=1.3e-6); **graduated `packed=True` default**.
- **M4 ✅** — real-model B-sweep (`parity/qwen35_batched_bench.py`, Qwen3.6-35B-A3B int4-g64): **greedy-exact
  loop==loopkill at every B AND a win — 1.63× @ B=32** (1.20/1.45/1.67× @ B=4/8/16; the win grows with B);
  **graduated `QWEN35_BATCHED_LOOPKILL_DEFAULT=True`** (`from_inner` gained a `loopkill` override so bf16
  per-stream tests still construct). The deliberate B=4 latency-first ORCHESTRATOR pin (`BEST_BATCH`
  `qwen3_5`, #26) is LEFT untouched — the loop-kill helps at any B≥2; B=32 is the bench/graduation point,
  not the serving pin.

Original framing (kept for context): the user chose **option B** (build the packed runtime so batched
projections are bit-exact) and **re-pinned the bench operating point to B=32**.

### The finding (bench on `Qwen3.6-35B-A3B-quanta_int4g64`, 40 layers = 30 GDN + 10 GQA)

| B | loop tok/s | loopkill tok/s | ratio | greedy |
|---|---|---|---|---|
| 1 | 25.1 | 25.6 | 1.02× | **bit-exact** ✅ |
| 4 | 57.5 | 62.8 | 1.09× | **DIVERGES** ❌ (stream 1 step 0: loop=2901 vs loopkill=2222) |

- |Δlogit| probe (B=4, step 0): worst **1.30** ≫ LOGIT_TOL 5e-3 ⇒ **real bug, not a near-tie**.
- per-layer post-mixer |Δ|: 0 at layer 0, then **geometric compounding from layer 1** (1.2e-4 → 7.8e-3 by
  layer 8), data-dependent across BOTH mixer types — there is no single buggy layer.

### Root cause — the bf16-drift trap (`feedback_batched_rope_bf16`)

`runtime.py:_load_block` **dequantizes** the mixer projections to dense-bf16 `nn.Linear` (lines 76/89).
The loop-kill batches them (`[B,1,h] @ W.T`), and **a dense-bf16 GEMM reorders its accumulation across
batch-M** — different M selects a different kernel tiling, and bf16 accumulation is non-associative.
Micro-test, decisive:

| matmul | batched-vs-per-stream max\|Δ\| |
|---|---|
| dense **bf16** | **1.0** |
| dense fp32 | 4.4e-4 |
| **`mx.quantized_matmul`** int8/int4 g64, **M≤8** (B=8 or a ≤8 chunk) | **0.0** (bit-exact) |
| `mx.quantized_matmul` int8/int4 g64, **full-batch B≥12** | **~2.25** (gemv→GEMM K-reorder — M0) |
| `mx.gather_mm` per-row matvec (MoE) | **0.0** (bit-exact) |

**Why the MoE is exempt:** `qwen35_moe`/`_routed_sparse` dispatch via `mx.gather_mm` as per-(token,slot)
**matvecs** (each gathered row is `[out,h]@[h,1]`, M=1) → no batch-M GEMM tiling → batch-invariant. The
`gdn_step` recurrence (fp32 elementwise + `mx.sum` over fixed axes) and RoPE (correctly per-stream-looped)
are exempt too. **Only the dense-bf16 projection GEMM drifts.** **Why the cohort didn't hit this:**
InternLM2.5's prod path is the PACKED `_PackedModel` whose `_qmm = mx.quantized_matmul(transpose=True)`
keeps projections quantized → batch-M bit-exact → greedy-exact (3.20×@B32). Qwen dequantizes → drifts.

### The fix — packed-projection runtime (`nn.QuantizedLinear`, rule 1)

`mx.quantized_matmul` is batch-M **bit-exact only for M≤~10** (M0 finding `c503657` — a per-row gemv kernel;
at B≥12 it switches to a tiled GEMM that reorders the K-reduction, ~2.25/proj in bf16). So making the
projections quantized closes the drift **only if each matmul stays in that regime** — i.e. the loop-kill
projections are **chunked into ≤8-row slices** (the user's decision; the `_chunked_qmm` helper proven in
the M0 gate). Chunked, the packed loop-kill equals the per-stream M=1 loop **bit-for-bit at any B**, with no
change to the mixer forward math. The artifact already exposes
everything: `art.raw(key)` → `.weight_packed`; `art.get(base+".weight_scale"/".weight_bias")` → siblings;
`manifest[base]["bits"/"group_size"]` → codec (rule 6 — read it from the manifest, fail loud on
non-uniform/missing, never a hardcoded width).

**Approach:** add `packed: bool` to `Qwen35ResidentModel` (thread through `Qwen35BatchedResidentModel`).
When `packed=True`, `_load_block` builds each mixer projection as a bias-free `nn.QuantizedLinear`
populated from the packed triplets — NOT `mx.dequantize`. `nn.QuantizedLinear.__call__` dispatches to
`mx.quantized_matmul`, so `self.q_proj(x)` / `self.in_proj_qkv(x)` etc. are UNCHANGED, and BOTH the
per-stream loop AND the batched loop-kill become batch-M bit-exact. `packed=False` keeps the dequantized
`nn.Linear` path as the parity reference. MoE / norms / conv / `A_log` / `dt_bias` are untouched.

Projections to convert (all `bias=False`, all `affine_packed` in the artifact):
- GDN: `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`.
- GQA: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

(Fallback if `nn.QuantizedLinear`'s ctor/layout fights the artifact triplets: mirror InternLM2.5 exactly —
store the triplets and call a local `_qmm = mx.quantized_matmul(x, packed, scale, bias, transpose=True,
group_size, bits)`. Identical numerics; rule 1 prefers `nn.QuantizedLinear`.)

**Two coupled graduations (rule 4):** (1) `packed` False→True after the packed-vs-bf16 forward parity gate
(greedy-exact + teacher-forced ppl) is green; (2) `QWEN35_BATCHED_LOOPKILL_DEFAULT` False→True after the
B=32 re-bench is greedy-exact + a win. **Loop-kill REQUIRES packed** (only bit-exact when projections are
quantized) — assert/enforce `loopkill ⇒ packed`.

### Milestones (model-free M0–M3; M4 = deferred solo-GPU bench)

- **M0 ✅ DONE (`c503657`) — model-free batch-M parity proof.** `parity/qwen35_batched_loopkill_test.py`
  §M0: built `nn.QuantizedLinear` from `mx.quantize` codes + a dense-bf16 `nn.Linear` from `mx.dequantize`
  of the SAME codes; compared ONE `[B,1,h]` batched matmul vs B per-stream `[1,1,h]` (M=1) at the bench's
  root-cause shape. **Finding:** `mx.quantized_matmul` is bit-exact only for M≤~10; full-batch B≥12
  reorders (~2.25/proj, bf16) — the PLAN's "batch-M bit-exact" premise was too strong. **Fix locked
  (chunking):** chunked-≤8 quantized is BIT-EXACT vs per-stream at B∈{1,4,8,32} (the `_chunked_qmm`
  primitive), while full-batch quantized [B32=2.25] + dense-bf16 [B4=3e-2,B32=1.0] both reorder. Gate
  green; self-protects the chunk size (if MLX drops the threshold <8, chunk_exact + full_threshold fail).
- **M1 — packed GDN mixer** (30/40 layers, the bigger lever). Loader builds GDN projections as
  `nn.QuantizedLinear` under `packed`; the batched stepper applies each via `_chunked_qmm` (≤8-row chunks,
  the M0 mechanism) so B>1 stays in the bit-exact regime. Gate: packed GDN batched decode == packed GDN
  per-stream loop **bit-exact** at B=1 AND B>1 (GDN has no SDPA/softmax reorder — fully bit-exact once
  projections are quantized + chunked); packed vs dequant single-stream greedy-exact.
- **M2 — packed GQA mixer** (10/40 layers). Same for q/k/v/o (also chunked via `_chunked_qmm`). Gate:
  packed GQA batched == packed per-stream loop **greedy-exact** at B>1 (the q/k/v/o projections are
  bit-exact once chunked; the padded-SDPA softmax reorder stays argmax-stable ULP), bit-exact at B=1;
  packed vs dequant single-stream greedy-exact.
- **M3 — wire packed into the runtimes + parity-gate the packed forward.** Thread `packed` through
  `Qwen35ResidentModel` → `Qwen35BatchedResidentModel`. Gate: `qwen35_forward_test` packed-vs-bf16 forward
  greedy-exact (+ teacher-forced ppl on real prose per methodology, if run); full model-free regression
  (`qwen35_batched_test`, `qwen35_batched_loopkill_test`). Graduate `packed=True` default.
- **M4 — re-bench at B=32 + graduate loop-kill (DEFERRED, solo GPU).**
  `uv run python -m parity.qwen35_batched_bench 32` on the real int4-g64 bake with `packed=True`: loop vs
  loopkill MUST be greedy-exact (now that projections are quantized + chunked ≤8) AND a win at B=32. If green → flip
  `QWEN35_BATCHED_LOOPKILL_DEFAULT=True`, set the serving operating point to B=32 (orchestrator pin in
  `shim/omlx`), update parity default-ON pins. RUN SOLO (OOM-reboot hazard; one model at a time).

### File anchors
- `src/quanta/qwen35/runtime.py` — `_load_block` (64; lines 76/89 = the dequant assignments to swap),
  `_LINEAR_PROJS`/`_FULL_PROJS` (51-52), `Qwen35ResidentModel.__init__` (120; add `packed`).
- `src/quanta/qwen35/artifact.py` — `raw` (124), `get` (63), `manifest` (59); `linear_attn`/`full_attn`
  (151/161 are `read`=dequant — add packed-triplet accessors or read raw+siblings in the loader).
- `src/quanta/qwen35/attention.py` — `Qwen35Attention.__init__` (190; the `nn.Linear` projs), `_project`
  (208), `decode_step_batched` (256; UNCHANGED — calls `self._project`).
- `src/quanta/qwen35/gated_deltanet.py` — `GatedDeltaNet.__init__` (215; the `nn.Linear` projs), `__call__`
  (245; UNCHANGED).
- `src/quanta/qwen35/batched_runtime.py` — `Qwen35BatchedResidentModel.__init__` (284; thread `packed`),
  `_gdn_step_batched` (101)/`decode_step_batched` callers (UNCHANGED), flag (66).
- `src/quanta/internlm2/runtime.py` — `_PackedModel` (212), `_qmm` (149), `_load_quant_triplet` (92): the
  EXACT pattern to mirror.

### Gates
```bash
uv run --with numpy python -m parity.qwen35_batched_loopkill_test   # M0–M2 (model-free)
uv run --with numpy python -m parity.qwen35_batched_test            # regression
uv run --with numpy python -m parity.qwen35_forward_test            # M3 packed-vs-bf16 forward
uv run python -m parity.qwen35_batched_bench 32                     # M4 (solo GPU; loop==loopkill + win@B32)
# before each commit: pytest tests/ -q · ruff check src tests · compileall · uv lock --check · git diff --check
```

### Kick-off prompt for the next agent
```
quanta #153 — Qwen3.6 option B: build the packed-projection runtime so batched-decode mixer projections
are bit-exact, then graduate the hybrid loop-kill at B=32.

Read first, in order: CLAUDE.md (rules + model facts); MEMORY.md + memory/project_paged_batched_153.md
(cross-session memory); PLAN_153.md → the "Qwen3.6 — option B" section (full design, milestones M0–M4,
file anchors, gates); then src/quanta/internlm2/runtime.py (_PackedModel / _qmm — the pattern to mirror).

Context: Qwen #153 M1+M2 built the hybrid loop-kill behind QWEN35_BATCHED_LOOPKILL_DEFAULT (default off).
The real-model bench (parity/qwen35_batched_bench.py) caught that it is NOT greedy-exact at B>1: the
dequantized dense-bf16 mixer projections reorder their accumulation across batch-M (|Δlogit|≈1.3, compounds
over depth). Fix = keep projections quantized via nn.QuantizedLinear, BUT M0 (c503657) found
mx.quantized_matmul is batch-M bit-exact ONLY for M≤~10 (full-batch B≥12 reorders too) — so the loop-kill
projections must be CHUNKED into ≤8-row slices (each an M≤8 quantized_matmul; the _chunked_qmm primitive in
the M0 gate). The MoE (gather_mm matvecs) and gdn_step (fp32 elementwise) are already batch-invariant — do
NOT touch them. Loop-kill requires packed; enforce loopkill ⇒ packed.

M0 DONE (c503657 — model-free batch-M proof + the chunking decision). Start with M1 (packed+chunked GDN):
the loader builds GDN projections as nn.QuantizedLinear under `packed`, and the batched stepper applies
each via _chunked_qmm (≤8-row chunks). Then M2 (packed+chunked GQA), M3 (wire + parity-gate the packed
forward + graduate the packed default), M4 (solo-GPU re-bench at B=32 + graduate the loop-kill). Operating
point is B=32.

Cadence (STANDING — do not violate): single linear thread, NO subagents/workflows. Implement → gate green
→ commit each milestone (named files only, never `git add -A`; trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`; no push; no hook skip; commits land
on main) → STOP for the user to compact before the next milestone. RUN ONLY ONE MODEL AT A TIME (M4 only;
OOM-reboot hazard). Never delete ~/models/Kimi-K2.6. No mlx-lm/torch on the runtime hot path.

Start with M0.
```

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
  stays OFF so DSV4 is untouched + DSV4 M3 not preempted; InternLM2.5 has since graduated to its own scoped
  flag too) after the real-model bench
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
- **InternLM2.5** (k/v, paged — "small model") — **✅ DONE `fad71bb` (wire) + `c1db9f6` (bench + default
  GRADUATED ON).** Pure dense GQA, 32 layers, no recurrent state. Killed the per-stream KV `.update()`
  loop in BOTH `decode_batched` paths — bf16 `InternLM2Model` (`internlm2/model.py`) AND packed
  `_PackedModel` (`internlm2/runtime.py`) — by routing their identical KV-update+SDPA tail through the new
  shared `batched_decode_attention_kv`. `paged_batched` threaded: wrapper
  `InternLM2BatchedResidentModel._paged_kv_batched` ← **InternLM2.5-scoped** `INTERNLM2_PAGED_KV_BATCHED_DEFAULT`
  → `step_batch` → `InternLM2ResidentModel.decode_batched` (delegator) → inner. Gate
  `internlm2_batched_attention_test.py` §C BIT-exact (`max|Δ|=0`, full `decode_batched` paged loop-kill ==
  per-stream paged loop, B=1 + ragged B=3 boundary-crossing, bf16 head_dim-agnostic) + pins default-ON.
  **Graduated to ON** via the scoped flag (NOT the shared `PAGED_KV_BATCHED_DEFAULT`, now DSV4-only, so DSV4
  M3 not preempted) after `parity/internlm2_paged_batched_bench.py` (int8-g64 7B bake, prod paged
  `_InternLM2BatchedSession`, distinct prompts, raw token-id lists — no tokenizer) proved greedy-exact + a
  big win:

  | B | loop tok/s | loopkill tok/s | loopkill/loop | greedy |
  |---|---|---|---|---|
  | 1 | 46.3 | 45.9 | 0.99× | bit-exact |
  | 32 | 103.6 | 331.8 | **3.20×** | greedy-exact |
  | 48 | 101.9 | 322.0 | **3.16×** | greedy-exact |

  FAR bigger than Nemotron's (+15%@B32) because InternLM2.5 is DENSE — ALL 32 layers are attention, so
  EVERY layer's KV loop is killed (Nemotron trims only its 8 `*`). Same regression signature as Nemotron:
  per-stream `loop` REGRESSES B=32→48 (104→102) while `loopkill` holds flat (~332→322), so the win does
  not fade with B. The bench doubles as the real-model gate for the QUANTIZED int8-g64 k/v
  `write_batched`/`gather_batched` at `head_dim=128` (§C used bf16). Active 9.5 / peak 10.3 GiB @ B=48.
- **Qwen3.5/3.6** (`qwen35`) — **M1+M2 loop-kill BUILT (flag off); bench BLOCKED graduation (bf16-drift);
  option B IN PROGRESS — M0 ✅ `c503657`.** UNPAGED (`shim/omlx` forces `paged_kv=False`) + hybrid (GDN + GQA), so NOT a
  paged-primitive wire. M1 (`ee305dc`, batched GQA) + M2 (`17c14dd`, batched GDN) landed the loop-kill
  behind `QWEN35_BATCHED_LOOPKILL_DEFAULT=False`; the real-model bench found it is NOT greedy-exact at B>1
  (dequantized dense-bf16 projections reorder across batch-M, |Δlogit|≈1.3). Fix = packed-projection
  runtime (`nn.QuantizedLinear`) with the loop-kill projections **chunked ≤8** (M0 `c503657` found
  full-batch quantized also reorders at B≥12); operating point re-pinned **B=32**. Full design +
  milestones (M0 ✅) in the **"Qwen3.6 — option B"** section above.
- **DSV4** (single-stream latent, paged) — the core #153 path; ✅ **M0–M4 DONE** (`62609ba`/`c442c31`/
  `35dcd78`/`d19a254`/`cb2476b`); `PAGED_KV_BATCHED_DEFAULT` GRADUATED ON at M3; real-model bench at M4
  (loop==loopkill bit-exact, +13% @ B=32/48).

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

## Milestones (mirroring #18; M0–M3 model-free, M4 deferred GPU) — ✅ ALL DONE

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
- **M2 ✅ DONE (`35dcd78`) — compressed stepper, latent-only batched.** Batches ONLY the latent (rides
  `_PagedKVArena`); the derived ckv/ikv/ring stay PER-STREAM, so the paged boundary-snapshot lifecycle is
  UNCHANGED — the PLAN's "batch the derived/`_CompArena`" guess was SIDESTEPPED (not the lever; smallest-safe
  change). New paged-hybrid path in `decode_step_compressed_batched` (arena+rows+lcs, comp=None), 3-way
  dispatch. Gate `parity/dsv4_paged_batched_test.py` §`_run_compressed_stepper`: compressed paged-batched ==
  per-stream paged loop **BIT-exact (`max|Δ|=0`)** (ratio-4 +idx & ratio-3, B=4 + B=1).
- **M3 ✅ DONE (`d19a254`) — graduated `PAGED_KV_BATCHED_DEFAULT` ON.** No `_step_paged`/omlx change needed —
  M1/M2 flag-dispatch already wired `_step_paged`→`step_batch`→`_decode_batched_single` paged path. Gate
  `parity/dsv4_paged_latent_test.py` §C: `_run_engine_batched` drives the REAL `_DSV4BatchedSession` decode
  over ragged B=3 crossing block boundaries — ON==OFF bit-exact (`|Δ|=0`) AND greedy-exact vs discrete
  (argmax_mismatch=0, decode_snapshots=5). Full regression green.
- **M4 ✅ DONE (`cb2476b`) — real-model B-sweep bench (solo GPU).** `parity/dsv4_paged_batched_bench.py`
  drives the REAL `_DSV4BatchedSession` paged decode on the DSV4-Flash int4-g64 bake, distinct prompts,
  flips `rt._paged_kv_batched`: loop == loopkill **BIT-exact at B∈{1,32,48}** (both run the one batched SDPA;
  latent store differs only by M0-equal scatter/gather) AND **+13% decode tok/s @ B=32 & B=48** (peak
  ~192 GiB). Smaller than #18 M5's unpaged arena/bat +37% because DSV4 batches only the latent (derived
  per-stream) and MoE dominates decode FLOPs. Confirmation + win quantification (flag already ON since M3).

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
