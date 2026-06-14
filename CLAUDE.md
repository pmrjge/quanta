# CLAUDE.md — quanta

`quanta` is a **parity-first**, MLX-native quantization + sparse-MoE inference
runtime. It is a clean restart of the prior `quantification` effort, keeping the
hard-won findings (below) but rebuilding the runtime so that **every component is
gated against a numeric reference before it is optimized or its quantization is
judged.** The served fleet (below) is complete and parity-gated; the next model is **MiniMax-M3**, when it ships.

The previous build reached a runtime that produced incoherent output (teacher-
forced perplexity ~165 with BOS, ~5–9× fuzzy on trivial tasks) and the failure
was wrongly attributed to int3 expert quantization. It is **not** the experts
(see Settled Findings). It is a localized bug in the forward pass that was never
caught because the runtime was built and refactored **without a parity gate**.
That is the mistake this project exists to not repeat.

---

## Active task — MiniMax-M3-VL build (full VL, int4-g64). Fleet otherwise complete.

**NOW IN FLIGHT: MiniMax-M3-VL serving runtime + int4 bake.** Handover **`PLAN_minimax_m3.md`**.
The landed `~/models/MiniMax-M3` is **MiniMax-M3-VL** (`minimax_m3_vl`, 809.5 GiB bf16 / 59 shards)
— a **different architecture** from the in-tree `quanta.minimax` module (which targets the old
**M2.7**: all-MoE, full softmax, 256 experts, no shared expert, fp8). So M3 is a **real build**, not
validate-at-scale: 60L (3 dense + 57 MoE), GQA 64q/4kv + partial RoPE + per-head QK-norm + **Gemma
`(1+w)` norms** (≈ qwen35), 128 experts top-4 **+1 shared**, **sigmoid noaux_tc** routing (≈
nemotron/dsv4), **clamped SwiGLU-OAI** activation (NEW), a **native TRAINED block-sparse attention
indexer** on layers 3–59 (NEW; `index_{q,k}_proj/norm`, top-16 blocks of 128 — sparse==dense at ≤2K
ctx so build dense-first), a **CLIP-ViT vision tower** (full-VL per user), 1M native context, and
**MTP declared 7 but ZERO `mtp.*` weights** (→ refined to 0; native-MTP N/A, the Nex pattern). The
M3 code is **additive** (`*_m3.py`); the M2.7 files are left intact (retire later). Decisions (user):
**full VL now** + **int4-g64** (the only served width going forward — int6 was the original margin
pick, retired once int4 measured ~lossless e2e: see the **int4-g64 switch** entry below). **M0 ✅ (this
commit)** — groundwork,
model-free / header-only (no 809 GB load): `config_m3.MiniMaxM3Config` (nested text+vision parse, eos
`(200020,)`, MTP refine 7→0, per-layer dense/MoE + sparse-attn typing) + `quant_policy_m3` (key→scheme
+ resident projection). Coverage proven exact vs the real index (rule 6): **23,416 tensors = 1108
dense / 420 int8 / 21,888 expert_int**; **int6-g64 = 329.6 GiB resident** (160.8 GiB headroom under
490.4; int4 ref 233.4). **M1a ✅ (this commit)** — module + model-free layer parity
(`model_m3.py` + `parity/minimax_m3_layer_test.py`, 12 checks). No HF/sglang M3 forward exists
(checkpoint ships only `configuration_*.py`), so risky formulas are pinned to transformers SIBLINGS,
isolated: **clamped SwiGLU-OAI == `GptOssExperts._apply_gate`** (w1=gate/swish, w3=up, w2=down;
α=1.702/limit=7.0), **sigmoid-noaux router == `MiniMaxM2TopKRouter` + `routed_scaling_factor` 2.0**
(no DeepSeek groups), **partial RoPE == `minimax_m2.apply_rotary_pos_emb`** (no YaRN); **Gemma `(1+w)`
applies to ALL non-gated norms** (empirically: q/k/index norms 0-centered) and the **shared expert has
NO scalar gate** (no `shared_expert_gate` key). Full block + MoE-sparse==dense + fast==naive vs a
numpy-fp64 ref. Gates: `minimax_m3_fit_test` (SOLO, 13) + `minimax_m3_config_test` (24) +
`minimax_m3_layer_test` (model-free, 12). Manifest 102 model_free / 53 real_weight. **M1b ✅ (this
commit)** — real-weight at-scale layer parity: new `loader_m3.MiniMaxM3SourceCheckpoint` (lazy
single-shard reader, text decoder; `moe()` pre-stacks the **per-expert** `experts.{e}.{w1,w2,w3}`
into `[E,2*inter,h]`/`[E,h,inter]` at load time; text-only — refuses vision keys, the ViT is a
separate VL track) + SOLO gate `parity/minimax_m3_layer_parity.py` (non-`_test.py`, excluded from the
sweep; loads only L0+L3, streamed+released, rule 8) diffing the `model_m3` block in fp32 vs a
self-contained **numpy-fp64** oracle on identical real dequantized weights. Machine-precision:
dense-L0 Δ 7.8e-7, MoE-L3 Δ 8.6e-7, fast==naive ~1e-7, sparse `gather_mm`==dense 4e-7, router
(real F32 gate/bias) set-match + wΔ 6e-8, indexer shapes ✓ (8 checks). Validates loading +
real-shape/dtype wiring (hidden 6144, GQA 64q/4kv, 128 experts top-4 + shared) + the
per-expert→stacked pack at 397B-class scale. **M2a ✅ (this commit)** — the int6-g64 bake +
artifact-reader path: `bake_m3.bake_minimax_m3` (streamed one text layer resident, rule 8) writes a
self-contained int6/int8/bf16 bundle — routed experts **int6-g64** (pre-stacked 3-D, `gather_qmm`-ready,
rule 3), GQA q/k/v/o + dense-FFN + shared expert int8, norms / router `gate`+`bias` (**f32**) /
trained indexer (bf16) / embed / head dense, **full VL** = the 523 vision tensors copied dense bf16; M3
is **natively 1M** (no YaRN) so it asserts + declares the window; `generation_config.json` (eos
200020) + tokenizer + VL `preprocessor_config.json` copied; self-contained audit fails loud.
`artifact_m3.MiniMaxM3Artifact` **duck-types `loader_m3`** (one forward serves bf16 source + int6
artifact) and returns the router **gate/bias at native F32 via `get()`** (NOT bf16-downcast — would flip
top-k ties; only gate+bias are F32, rest bf16, confirmed on disk). Gates: model-free
`parity/minimax_m3_bake_test.py` (12 checks, in the sweep — synthetic checkpoint through the real bake
then back through both readers: every dequant == source-RTN bit-exact, F32 router preserved, manifest
schemes, raw/refusals, native-1M, self-contained) + `parity/run_bake_minimax_m3_int6g64.py` (SOLO;
`--smoke` ran on real weights in 2.7s → 6.1 GiB artifact, readback of int8/int6 stacks/F32 gate/bf16
indexer/packed-int6 triplet all correct). Manifest 103 model_free / 53 real_weight. **M2b ✅ (this
commit)** — full bake + teacher-forced ppl arbiter. `run_bake_minimax_m3_int6g64` ran SOLO →
`~/models/MiniMax-M3-quanta_int6g64` in **3.9 min** (RTN, no GPTQ): **329.6 GiB** on disk (== the M0
projection, < 490.4 ceiling), self-contained (0 symlinks / sidecars present / no leaks / 30 shards),
**full VL** (523 vision tensors dense bf16), native 1M; counts int8 420 / expert_int 114 (57 MoE × 2) /
dense 1108. New SOLO arbiter `parity/minimax_m3_ppl.py` (non-`_test.py` ⇒ excluded from the sweep; two
streamed `MiniMaxM3Block` forwards one-layer-resident, rule 8; tokenizer built directly from
`MiniMaxM3Config` which duck-types bos/eos, BPE reads only `tokenizer.json`, `add_bos_token` absent ⇒
raw): **637 tok, all 60 layers — bf16 ppl 4.96 / acc 0.591** ⇒ the pinned Gemma `(1+w)` fold is
**CONFIRMED e2e** (the decisive check: no HF/sglang M3 forward exists to diff against; a wrong fold
degrades ppl uniformly to the hundreds — 4.96 is a healthy 397B value), **int6-g64 ppl 5.00 / Δppl
+0.82% / acc 0.591 / top-1 agree 0.943** ⇒ ~lossless, **SHIP int6-g64** (the user's int6-margin choice
over int4 validated e2e — *superseded: int6 retired for int4-g64, see the int4-g64 switch entry at the
end of this section*). **M3 serving is now decomposed into sub-milestones (Nex-style: M3-1,
M3-2, …).** **M3-1 ✅ (this commit)** — the resident single-stream serving runtime. `model_m3` gains
the **packed-int6 `gather_qmm`** routed path (`_routed_sparse_packed` + `MiniMaxM3MoE.set_experts_packed`
+ auto-detect: a triplet dict ⇒ `gather_qmm`, a bf16 stack ⇒ `gather_mm`; `sparse=False` on packed
refuses, rule 6) — output-equivalent to the bf16 reference on the SAME codes, and the resident path
actually dequantizes int6 at *higher* precision than the bf16-rounded reference. New
`runtime_m3.MiniMaxM3ResidentModel` loads the int6 artifact **one text layer resident at a time**
(rule 8), routed experts held **packed int6** (`artifact_m3.moe_packed` → `set_experts_packed`, the
~300 GiB resident lever) over the **int8 mixer dequantized to bf16** (q/k/v/o + dense-FFN + shared
expert; the proven M1/M2 forward — a packed-int8 mixer that saves ~10 GiB is a later memory milestone,
far under the 160 GiB headroom); router gate/bias native **F32**; prefill (`caches=None`) == the M1/M2
streamed reference, decode threads per-layer GQA `KVCache` (`make_caches`), plus a minimal greedy
`generate`. Gates: model-free `parity/minimax_m3_runtime_test.py` (9 checks, in the sweep — tiny
synthetic, packed==bf16 top-1-exact + logit-rel, cached==prefill **bit-exact**, incremental-decode==
full-prefill **bit-exact**, rule-4 dense==sparse, rule-6 packed-refuses-`sparse=False`, `generate`
smoke) + SOLO `parity/minimax_m3_runtime_real.py` (non-`_test.py`, excluded; the **397B resident
re-gate** — all 60 layers RAM-resident in **33s load**, packed `gather_qmm` vs the streamed `gather_mm`
reference on the real int6 codes: **ppl 5.870 / Δppl +0.171% / top-1 agree 0.969**, ships the M2b int6
quality). Manifest 104 model_free / 53 real_weight. **M3-2 ✅ (this commit)** — batched serving
(Design A) + the packed-int8 mixer. The int8 mixer (GQA q/k/v/o on all 60 layers + the dense-FFN
gate/up/down on layers 0–2) is now held **packed `nn.QuantizedLinear`** (`mx.quantized_matmul`) via
`runtime_m3._packed_linear` + a new `MiniMaxM3ResidentModel(packed=…)` flag (default `False` — the
bf16-mixer M1/M2 parity reference; the serving runtime sets `True`) — the ~6 GiB memory lever + the
batch-M bit-exact substrate, greedy-exact on the SAME int8 codes; the **shared expert stays bf16**
(the qwen35 convention — it runs batched inside the one MoE call; packing it is a trivial later tweak
under the huge headroom). New `batched_runtime_m3.MiniMaxM3BatchedResidentModel` (default
`packed=True` + `packed_experts=True`) wraps the resident model for B>1 serving — **Design A**:
per-stream GQA `KVCache` lists, a bounded per-stream attention step (M=1 ⇒ bit-exact vs single-stream),
then ONE batched FFN over the stacked `[B,1,hidden]` (the routed-expert `gather_qmm` reads each
touched expert tile ONCE for all B rows that route to it — the bandwidth win; the existing MoE is
B-aware via its `[B,S,h]→[N,h]` reshape); `step_batch` / `prefill` / `make_batch_caches`, ragged
per-stream offsets, rule-6 desync/over-batch refusals. Gates: model-free
`parity/minimax_m3_batched_test.py` (19 checks, in the sweep — packed-mixer==bf16-mixer greedy-exact,
batched==single-stream **bit-exact** on the synthetic incl. ragged offsets + B=1, rule-6) + SOLO
`parity/minimax_m3_batched_real.py` (non-`_test.py`, excluded; the **397B re-gate** off ONE 325 GiB
resident load: packed mixer+experts vs the streamed bf16 reference **ppl 5.879 / Δppl +0.316% / agree
0.953**; batched B=8 ragged == single-stream greedy-token-equivalent — at scale the lone cross-stream
op, the F32 router GEMM at M=B, flips a routing near-tie on 1/8 streams, the documented batched
boundary; decode **2.32× aggregate @ B=8** — the batching lever, climbing with B). Manifest 105
model_free / 53 real_weight. **M3-3 ✅ (this commit)** — the GQA loop-kill: ONE batched attention
across streams (the bigger B>1 lever now the MoE read is amortized). `model_m3.MiniMaxM3Attention`
gains `decode_step_batched` (batched **chunked** q/k/v/o projections + a per-stream RoPE *kernel* loop
— only the absolute offset differs, M3 has no YaRN — + the shared fused padded SDPA
`quanta.modeling.batched_attention.batched_decode_attention_kv`, the #153 primitive InternLM2/Nemotron/
qwen35 use) + `_project_chunked` (≤`chunk` row-slices keep each packed `mx.quantized_matmul` in the
M=1-equivalent gemv regime ⇒ **bit-exact projections**; #153 option B). `batched_runtime_m3` gets a
`loopkill` flag (**graduated ON** — `MINIMAX_M3_BATCHED_LOOPKILL_DEFAULT`, chunk
`MINIMAX_M3_LOOPKILL_CHUNK=8`): the `if loopkill` branch in `batched_decode_step` runs the batched
attention; the M3-2 per-stream loop stays the rule-4 reference (pinned `loopkill=False` in the M3-2
gate). `_check_loopkill_requires_packed` enforces **loop-kill ⇒ packed** at construction AND every
`step_batch` (a dense-bf16 projection reorders across batch-M — rule 4/6). Output-equivalent: only the
fused padded-SDPA softmax reorders ⇒ greedy-token-equivalent. Gates: model-free
`parity/minimax_m3_loopkill_test.py` (24 checks, in the sweep — **§M0** chunked-8 int8
`quantized_matmul` bit-exact vs the M=1 loop at B∈{1,4,8,32} / full-batch reorders @ B≥12, loop-kill ==
per-stream loop **and** == single-stream on the synthetic incl. ragged + B=1, loop-kill⇒packed refused
at construction AND at step) + SOLO `parity/minimax_m3_loopkill_real.py` (non-`_test.py`, excluded; the
**397B re-gate** off ONE resident load — loop-kill == the per-stream loop **BIT-EXACT** (top-1 1.0000 /
rel 0, 64/64 over 8 decode steps × B=8: same batched MoE, bit-exact chunked projections, bit-identical
RoPE, SDPA reorder ~0 at these lengths) and == single-stream (top-1 1.000, no near-tie flip); ships the
M2b int6 quality (ppl 5.879 / Δppl +0.316% / agree 0.953); decode **1.19× over the per-stream loop @
B=8 / 2.83× aggregate B=1→B=8** — the mixer-read bandwidth win on top of M3-2's batched MoE). Manifest
106 model_free / 53 real_weight. **M3-4 ✅ (this commit)** — paged-KV + prefix caching (int8 KV). M3 is
the **clean dense-GQA paged case** (all 60 layers attention, NO recurrent state — like InternLM2.5), so
it exposes the **#152 paged contract** the shared `quanta.shim.omlx._BaseBatchedSession` drives:
`model_m3.KVCache` gains int8 modes (`quantized`/`group_size`/`bits`, mirroring `quanta.internlm2` +
`cache_quant`; **default bf16** = the M1/M2 reference, **int8-g64** the serving lever — quant groups on
`head_dim` are orthogonal to the seq-axis blocks ⇒ a paged gather is **bit-identical** to the discrete
cache); `MiniMaxM3Attention.decode_step_batched` gains a `paged_batched` flag → the shared
`batched_decode_attention_kv` does ONE `write_batched` + ONE `gather_batched` over paged views (the paged
KV loop-kill) instead of the per-stream `.update()` loop — bit-identical (both end in the same fused
SDPA). `batched_runtime_m3.MiniMaxM3BatchedResidentModel` exposes `has_recurrent_state=False` +
`paged_kv_spec` + `make_paged_state` + `prefill_paged` (dense ⇒ no boundary payloads, `recurrent_in` must
be None) + a `_paged_kv_batched` flag (`MINIMAX_M3_PAGED_KV_BATCHED_DEFAULT`, **graduated ON**);
`step_batch` auto-detects paged views; KV mode is **int8-g64 on `__init__`** (the serving default), bf16
on `from_inner` (model-free gates). Gates: model-free `parity/minimax_m3_paged_test.py` (19 checks, in
the sweep — paged prefix-reuse + suffix == discrete continue-from-prefix **BIT-EXACT** for BOTH int8-g32
+ bf16 KV, prefix blocks dedup, paged loop-kill == per-stream paged loop **bit-exact**, dense emits no
boundary payloads, rule-6 `recurrent_in`/offset refusals) + SOLO `parity/minimax_m3_paged_real.py`
(non-`_test.py`, excluded; the **397B re-gate** off ONE int8-KV resident load: **paged == discrete
BIT-EXACT** (|Δ| 0), the paged KV loop-kill == the per-stream paged loop **BIT-EXACT** (|Δ| 0 @ B=8
ragged), **int8 KV near-lossless** (bf16 ppl 5.879 → int8-KV ppl 5.927, **Δppl +0.823%** / top-1 agree
0.949), paged decode == single-stream (top-1 1.000), reuse-after-free **bit-exact** (|Δ| 0)). **Finding:**
paged prefix reuse is **bit-exact when the committing prefill SHAPE matches** (the orthogonal-axes
foundation); a re-admit whose prior commit used a *different* shape is **greedy-token-equivalent**, not
bit-exact — a packed projection at batch-M=A tiles its K-reduction differently than at M=B (the #153
finding, now surfacing in prefill: the same tokens prefilled in different-length batches give
quant-ULP-different KV codes, compounding over 60 layers). Manifest 107 model_free / 53 real_weight.
**M3-5 ✅ (this commit)** — long-context chunked prefill (the long-admit lever). The single-shot prefill
holds the whole `[1,T,hidden]` window resident and runs one O(T²) attention; chunked prefill consumes the
prompt in seq blocks, each chunk extending every layer's GQA KV with a bottom-right causal mask
(`mx.fast.scaled_dot_product_attention` `mask="causal"` is bottom-right aligned — the M3-1 cached-forward /
qwen35 shipped-chunked path), so the per-chunk transient is **O(chunk)** (the fused flash-attn kernel never
materializes the `[chunk, kv_len]` scores) and a 1M-token prompt admits in O(chunk) memory + the int8 KV.
M3 is all dense GQA (no GDN, no YaRN) so — unlike `quanta.qwen35.runtime.chunked_prefill` — there is no
per-request RoPE factor to pin and no recurrent continuation; each chunk just reads its absolute position
from the cache offset. New `runtime_m3.chunked_prefill` (shared driver, one bounded `MiniMaxM3Block`
forward per chunk, per-chunk `mx.eval`+`mx.clear_cache`, rule 8) + `MiniMaxM3ResidentModel.prefill_chunked`;
`batched_runtime_m3` gains `MINIMAX_M3_PREFILL_CHUNK_TOKENS`=4096 + `MINIMAX_M3_CHUNKED_PREFILL_FROM`
(=chunk+1) and routes `prefill` (and thus the paged `prefill_paged` admit) through `prefill_chunked` above
the threshold — short chat prompts keep the bit-exact single-shot path (M3-1/2/3/4 gates untouched). Works
over discrete `KVCache` lists OR `PagedKVCacheView` lists (the manager allows sub-range writes from the
open cursor — chunked-over-paged). Bit-exact to single-shot on the bf16 mixer (chunk boundaries only re-cut
the same per-row causal attention + per-token KV quant); greedy-token-equivalent on the packed serving
mixer (the projections run at batch-M=chunk vs M=T — the #153 batch-M ULP). Gates: model-free
`parity/minimax_m3_prefill_chunked_test.py` (41 checks, in the sweep — bf16 chunked == single-shot
**BIT-EXACT** across chunk sizes incl. ragged + per-token, int8-KV bit-exact for chunk≥2 [ct=1 hits the
decode `mask=None` kernel ⇒ greedy-equiv, the documented int8 decode-vs-prefill boundary], continue-from-
non-empty-cache, chunked-over-paged == discrete == single-shot for both KV modes, rule-6, threshold routing)
+ SOLO `parity/minimax_m3_prefill_chunked_real.py` (the **397B re-gate** off ONE resident load: chunked ==
single-shot **greedy-token-equivalent** [last-tok top-1 ==, rel 3.36e-2 — the packed batch-M ULP],
chunked-over-paged == discrete-chunked **BIT-EXACT** [|Δ| 0 — the M3-4 orthogonal-axes foundation holds
under chunked writes], chunked-seeded 6-step decode == single-shot-seeded **1.000** — the served STATE is
correct). Manifest 108 model_free / 53 real_weight.
**int4-g64 switch ✅ (this commit)** — the served width is now **int4-g64** (int6 retired; user: "only
4bit from now on"). The bake pipeline was already bits-agnostic (`bake_minimax_m3(expert_bits=…)`, the
artifact reader + `gather_qmm` read the width from the manifest), so the switch is: `run_bake_minimax_m3_int4g64`
(default `--bits 4`, `--bits 6` reproduces the retired arm) → **233.4 GiB** on disk (== the M0 int4
projection, 96 GiB under int6's 329.6; full VL 523 vision tensors; native 1M; 3.3 min RTN), the arbiter
`minimax_m3_ppl` (`--artifact` override) + the 5 serving `_real` gates + the model-free `minimax_m3_bake_test`
(now sweeps **int4 AND int6**, 24 checks) + the M0 `minimax_m3_fit_test` (SHIP→int4) all repointed.
**int4 WEIGHTS are lossless** (arbiter, streamed dequant-on-read vs bf16: **Δppl −0.24%** @637 tok / +0.70%
@256, agree 0.915 — the bf16 ppl 4.96 re-confirms the (1+w) fold). **Served via packed `gather_qmm`** the
fused low-bit kernel costs **+2.86% ppl vs the bf16 source** @256 (vs int6 serving ~+1%): the fused kernel
accumulates at int4's larger group scales differently than dequant-then-bf16-matmul (`gather_qmm` vs the
`gather_mm` reference diverge +2.14% @ runtime / +1.69% @ batched on the SAME codes — vs int6's
0.17%/0.32%). **Healthy, not lossless** — the real int4 serving cost. Serving re-gate @ 233 GiB (all SOLO):
**paged ✅** (paged==discrete BIT-EXACT, int8-KV Δppl −1.99%, decode top-1 1.000) + **chunked ✅**
(greedy-equiv, seeded decode 1.000) pass unchanged; **runtime/batched** `DPPL_CEILING` raised **1.0→4.0**
(the int4 kernel gap is intrinsic, not a regression — anchored by the lossless arbiter; the fleet ships int4
`gather_qmm` everywhere); **loop-kill AUTO-OFF at int4** — at int6 it was BIT-EXACT vs the per-stream loop,
at int4 the coarse MoE amplifies the batched-SDPA reorder to **0.875 token-agree / 0.187 worst-rel** @ B=8
(scattered near-tie flips, NOT a systematic bug), so per **rule 4** (optimizations default to the naive path
until parity is proven) the user's call is to fall back to the proven per-stream loop. Implemented as a
**per-expert-width default**: `batched_runtime_m3._resolve_loopkill_default` graduates loop-kill ON only at
int6+/bf16 (proven), OFF at int4 — read from the packed triplet (`_served_expert_bits`, rule 6). Because the
M3-4 paged-batched attention is gated behind loop-kill (`paged = _paged_kv_batched AND _loopkill`) and is the
SAME batched-cross-stream-SDPA mechanism, it cascades off too ⇒ **int4 serves per-stream attention**
(keeping M3-2's batched MoE — the big read-amortization win — + paged KV + chunked prefill; losing M3-3's
~1.19× attention batching). The `loopkill_real`/`paged_real` gates now **force `loopkill=True`** to validate
the path is the bounded-reorder regime (loosened int4 ceilings 0.30/0.80) **and assert the int4 auto-default
is OFF**; `minimax_m3_loopkill_test` gains check (6) — the per-width resolver (int4→OFF, int6+/bf16→ON) +
the auto-wired default (27 checks). Manifest unchanged 108 / 53 (no gate added/removed — the bake runner
rename is a non-`_test.py` real-weight gate, untracked). **int6 artifact freed 2026-06-14** (user:
"delete int6 artifact and keep going with int4"; 330 GiB reclaimed, disk → 24%; `--bits 6` still
reproduces the retired arm). **M3-6 is decomposed into sub-milestones (Nex/M3-style); the user picked
the vision track first (full-VL requirement, lowest reference-risk, parity-gateable).**
**M3-6a / vision V1 ✅ (this commit)** — the CLIP-ViT vision tower forward. NEW additive
`model_vision_m3.py` (text/M2.7 untouched): the **Conv3d-as-linear patch embed** (the on-disk
`[1280,3,2,14,14]` conv reshaped `[1280,1176]`, `1176=3·2·14·14` in the processor's `[channel,temporal,h,w]`
flatten order ⇒ a plain `[1176→1280]` linear — Qwen2-VL style, NOT CLIP Conv2d), `pre_layrnorm`, 32
**pre-norm CLIP encoder layers** (biased q/k/v/out, full bidirectional attention, exact-erf GELU MLP,
LayerNorm; no learned pos-embed / CLS / post-norm — none ship), **3-D vision RoPE**, then the head whose
order is **forced by the on-disk input dims**: `multi_modal_projector` (`linear_1` input 1280 ⇒ runs
FIRST, per patch `1280→6144→6144`) then `patch_merge_mlp` (`linear_1` input `24576=4·6144` ⇒ groups each
consecutive-4 = one 2×2 spatial block, `24576→6144→6144`), so one image of `grid=(t,h,w)` patches →
`t·h·w` ViT tokens → `(t·h·w)/4` LLM tokens (== the processor's `num_tokens = grid.prod()//merge²`
placeholder count at `image_token_index` 200025). The **3-D RoPE** (`rope_mode="3d"`, θ=1e4) is the one
piece with **NO shipped reference** (transformers CLIP uses learned pos-embeds; even Qwen3-VL's *vision*
tower is 2-D h/w) — built on the **Qwen2.5-VL M-RoPE convention** (one shared `inv_freq` ladder over the
head_dim, freq pairs *sectioned* across t/h/w by `rope_section`), which **degenerates exactly to the 2-D
(h,w) vision rope for an image** (`grid_t=1` ⇒ t-position 0 ⇒ identity on the t-section). [PINNED-pending-
e2e: the exact `rope_section` split of the 40 freq pairs (default `(8,16,16)`, h==w symmetric) is the lone
knob no artifact fixes; **vision V2** settles it cheaply against a real image — the ViT is 1.6 GiB,
loadable standalone.] Gate: model-free `parity/minimax_m3_vision_test.py` (20 checks, in the sweep —
**CLIP encoder layer (RoPE off) == the REAL `transformers` `CLIPEncoderLayer`** on identical weights;
patch-embed / 3-D rope / projector / patch-merge / (t,h,w) position-ids (merge-block order) == a
**numpy-fp64 oracle**; rule-4 fast==naive attention; rule-6 `rope_section`-sum + indivisible-merge
refusals; the 2-D-degenerate property; per-image attention isolation). Manifest **109 model_free / 53
real_weight**.
**M3-6b / vision V2 ✅ (this commit)** — the native image processor + the real-weight standalone ViT
forward. NEW additive `image_m3.py` (numpy-only, no torch/torchvision/PIL — rule 5): reproduces the
shipped `image_processor.py` (`smart_resize` geometry verbatim → bicubic resize → rescale +
CLIP-normalize → temporal-dup → patchify+`permute(0,1,4,7,5,8,3,2,6,9)`) → `pixel_values [N,1176]` +
`grid_thw`; one image grid `(t,h,w)` → `t·h·w` patches → `(t·h·w)//merge²` LLM tokens (== the shipped
`num_tokens = grid.prod()//merge²` placeholder count at `image_token_index` 200025). The **resize
interpolation is best-effort** (no `torchvision` in-env to bit-pin it) but a **factor-aligned in-bounds
image** (H,W multiples of 28, `min_pixels 3136 ≤ h·w ≤ max_pixels 451584`) makes resize the **identity**
⇒ the whole processor is exactly pinnable — the path the gates and the V3 e2e use. New
`artifact_m3.vision_state()` (the 523 dense vision tensors → bf16, the ViT loads as a unit — a
justified 1.6 GiB rule-8 exception, a bidirectional encoder can't stream layer-by-layer) +
`model_vision_m3.load_vision_model(art_dir, rope_section=None)` (Conv3d `[1280,3,2,14,14]→[1280,1176]`
reshape; per-layer suffixes map 1:1; `multi_modal_projector`/`patch_merge_mlp` → `projector`/`patch_merge`;
**two-way coverage assertions** — every model param assigned AND every source key consumed, rule 6).
Gates: model-free `parity/minimax_m3_image_test.py` (23 checks, in the sweep — `smart_resize` geometry ==
verbatim oracle over a size grid + factor-aligned identity; rescale/normalize/patchify == a slow
nested-loop numpy-fp64 oracle; bicubic identity-on-same-size / constant-preservation / rows-sum-to-1;
`num_tokens` rule; rule-6 extreme-ratio + non-RGB refusals) + SOLO `parity/minimax_m3_vision_real.py`
(non-`_test.py`, excluded, ~1.6 GiB — loads the 523 dense ViT tensors from the int4 artifact, runs a
real 56×56 image → **4 tokens, std 5.66 / max|x| 392, finite & sane**; **rule-4 fast==naive layer-0 op**
rel 5.9e-3; the **2-D-degenerate property on real q** — the t-section [8 pairs] inert for an image;
**per-image isolation BIT-EXACT** [rel 0]). No numeric ViT reference exists (the checkpoint ships only
`configuration_*.py`), so V2 validates **mechanics + structural invariants**; the decisive arbiter for
the [PINNED-pending-e2e] `rope_section (8,16,16)` is the V3 multimodal e2e. Manifest **110 model_free / 53
real_weight** (+`minimax_m3_image_test`; the `_real` gate is untracked, the established convention).
**Next = vision V3** — the multimodal prefill splice (replace the `image_token_index` 200025
placeholders in the text embedding stream with the merged ViT tokens — the shipped
`processing_minimax.py` rule) + the **heavy 233 GiB e2e that settles `rope_section`** (image + prompt →
teacher-forced ppl / greedy caption, sweeping candidate sections — the only arbiter, since the section
is judged downstream by the LLM); **then** the oMLX shim (the `_MiniMaxM3BatchedSession` engine route +
chat template + `<mm:think>` reasoning + MiniMax nested-XML tool parser + the multimodal image input
path) and the **trained block-sparse attention** long-context compute lever (deferred — no M3 forward
exists; only sparse==dense-at-short-ctx is bit-gateable, the exact selection formula leans on a heavy
397B long-context ppl arbiter; it is a speed optimization on an already-correct dense path).

The rest of the served fleet is **complete, shipped, and parity-gated**. Per-model resident sizes are
in the Serving throughput table below; the detailed milestone handovers live in the `PLAN_*.md` files
and git history. (This section used to carry the full transient Nex/Nemotron milestone log — collapsed
2026-06-13 once the fleet shipped and the roadmap narrowed to MiniMax-M3.)

| model | architecture | artifact | status |
|---|---|---|---|
| InternLM2.5-7B | dense GQA, 1M ctx | int8g64 — 9 GiB | served; EAGLE spec (1.42×@k2) + MInference sparse-prefill M0–M10 |
| Nemotron-Super-120B | `nemotron_h` hybrid | int4g64 — 68 GiB | served + native-MTP spec sidecar |
| Nemotron-Ultra-550B | `nemotron_h` hybrid | int4rtn_g64 — 306 GiB | served + MTP sidecar; U4 decode/spec optimizations complete |
| DSV4-Flash | sparse MoE + compressed-KV | int4g64 — 180 GiB | served; tree-spec-over-paged complete (the keeper where B=1 spec helps) |
| Qwen3.6-35B-A3B | `quanta.qwen35` | int4g64 — 19 GiB | served |
| Nex-N2-Pro = Qwen3.5-397B-A17B | `qwen3_5_moe` | int4g64 — 214 GiB | **SHIPPED** (N0→N3-3b); config declares the 1M window |

Cross-fleet optimization tracks are all landed: #18 batched-KV arena, #152 paged-KV (default ON),
#153 paged loop-kill, #158-160 tree-spec-over-paged (M0–M3), qwen35 routed-expert packing. Handovers:
`PLAN.md` (#18), `PLAN_153.md`, `PLAN_minference.md`, `PLAN_nemotron_ultra.md`, `PLAN_nex_n2_pro.md`,
`PLAN_qwen35_experts.md`.

MiniMax-M3-VL is now **in flight** (see the Active task above; handover `PLAN_minimax_m3.md`). It is
the only forward model — weights landed 2026-06-13.

**Dropped — do NOT re-propose as next work** (settled 2026-06-13):
- **Kimi-K2.6 / GLM-5.1 / DeepSeek-V4-Pro** — user not interested. The DeepSeek-V3-family runtime risk
  (the founding int3-floor / forward-bug question) was retired by the shipped DSV4-Flash keeper. The
  Kimi-specific reference sections below (Model facts, Quantization policy, GPTQ) are retained only as
  general engineering reference, not as a target.
- **Nex-N2-Pro N3 tail** — shelved; Nex ships at int4-g64. (Was: N3-4 1M needle gate, paged-KV + prefix
  caching, MInference on the 15 full-attn layers, fused/batched GDN decode-step, multi-stream >B=32.)
- **Mellum2** — dropped earlier (context length too short).

**Cadence (standing user instruction):** single thread, NO subagents, commit each milestone, then STOP
for the user to compact.

---

## Permanent engineering rules (do not violate)

These are non-negotiable and apply to every line of runtime/bake code:

1. **Build layers as `mlx.nn` modules.** Subclass `mlx.nn.Module`; compose with
   `nn.Linear`/`nn.RMSNorm`/`nn.QuantizedLinear`/etc. where they fit. Do not
   hand-roll parameter plumbing that `mlx.nn` already gives you. Simplicity of
   the layer definition is a feature.
2. **Prefer `mlx.fast` primitives, maximally.** Use `mx.fast.rms_norm`,
   `mx.fast.scaled_dot_product_attention`, `mx.fast.rope`, and any other
   `mx.fast.*` fused kernel instead of an equivalent hand-written sequence of
   ops. If a needed primitive is missing, wrap the closest `mx.fast` op and note
   why; do not silently reimplement it slowly.
3. **No Python loops on compute/hot paths.** Vectorize. Use batched ops,
   `mx.gather_qmm`/`mx.quantized_matmul`, `mx.compile` for stable shapes,
   broadcasting, `vmap`, and segment/gather primitives. The ONLY loops allowed
   are coarse, bounded, non-hot ones: iterating layers at load/bake time (one
   text layer resident at a time), IO/accounting boundaries, and the bounded
   `group_size` inner loop inside the GPTQ block solver. A loop over tokens,
   over experts per token, or over hidden dims is a bug.
4. **Parity-first.** No component is "done" until it matches a reference forward
   numerically (see Methodology). Optimizations (matrix-absorb, fused kernels,
   sorted dispatch, speculative decode) must be **output-equivalent** to the
   naive path and are kept behind a flag that defaults to the naive path until
   parity is proven.
5. **No `mlx-lm` as a runtime dependency.** `transformers`/`torch` are allowed
   **offline only** (parity references, tokenizers, source-checkpoint loading)
   under the `reference` extra — never on the inference hot path.
6. **No silent failures.** Code must work correctly or fail loudly. Never drop a
   baked tensor, never dequantize at the wrong bits by falling back to a default,
   never emit wrong output silently. Refuse to load a layer that bakes a tensor
   with no runtime consumer.
7. **Keep MoE routing sparse.** Never materialize a dense `tokens × experts ×
   hidden` intermediate. Route top-k, gather, dispatch.
8. **Layer-by-layer memory discipline.** Bake/calibration/parity must not hold
   more than one text layer's source weights resident at a time unless a measured
   exception is justified in the commit.

---

## Hardware / deployment target

- One **M3 Ultra**, 512 GB unified memory. Usable working-set ceiling
  **≈ 490.4 GiB** (`mx.metal.device_info()` recommended max working set). The
  whole quantized model is held **RAM-resident** (no offload/streaming); all
  current targets must fit under that ceiling.
- MLX is the runtime. `mx.set_wired_limit` pins the resident weight set.

---

## Serving throughput — measured fleet baseline (M3 Ultra, 2026-06-06)

Steady-state **decode** tok/s through each tuned keeper's production batched/paged path with every
graduated optimization on (#153 loop-kill, fused-SSD-step, option-B packed experts), run **solo** at
the cohort operating point **B=32**; greedy/bit-exact correctness verified per run.

| model | resident | agg tok/s @ B=32 | per-stream | B=1 | note |
|---|---|---|---|---|---|
| InternLM2.5-7B int8g64 | 9 GiB | **327.4** | 10.2 | 45.5 | plateaus (318 @ B=48) |
| Nemotron-Super-120B int4g64 | 68 GiB | **205.9** | 6.4 | 27.7 | flat ~206 @ B=32–48 |
| Qwen3.6-35B-A3B int4g64 | 19 GiB | **175.6** | 5.5 | 28.6 | still climbing past B=32 |
| DSV4-Flash int4g64 | 180 GiB | **77.8** | 2.4 | 6.2 | 90.4 @ B=48; unpaged #18 arena ~108.5 @ B=32 |
| Nemotron-Ultra-550B int4rtn_g64 | 306 GiB | **65.5** | 2.05 | 10.5 | peak @ B=32 (63.6 @ B=48); ~78 streams to ceiling |

Throughput tracks size inversely (the 7B serves ~5× the 550B's tok/s); **batching is the lever**
(DSV4 12.5× / Ultra 6.2× aggregate B=1→B=32). These are *throughput* numbers — the B=1 spec-decode
*latency* levers (InternLM2.5 EAGLE 1.42×@k2; Ultra MTP best 0.92×, <1×) are a separate axis. Repro
(each SOLO): `parity/{internlm2,nemotron,dsv4}_paged_batched_bench`, `parity/qwen35_batched_bench`,
`parity/nemotron_ultra_decode_scale` (Ultra bounded via the module's `BATCHES`).

---

## Methodology: parity-first (the core discipline)

Before optimizing or quantizing anything, establish a **reference** and diff
against it. Order of operations for any new model or layer:

1. **Reference forward.** Build a dead-simple, obviously-correct forward in
   plain `mlx.core` from the *dequantized* source weights (or a HF/transformers
   reference, offline). No fused kernels, no absorb, no rotations.
2. **Numeric parity, layer by layer.** Run identical token ids through both the
   reference and the runtime; capture the residual stream after each decoder
   layer and diff. The **first** layer/op that diverges beyond fp tolerance is
   the bug. Bisect within a layer across: RMSNorm placement, MLA attention
   (q/k/v projections, RoPE freqs, softmax scale incl. YaRN `mscale`, the
   matrix-absorb path), top-k routing, expert dispatch, shared expert.
3. **Only then** turn on an optimization or tighten quantization, re-running
   parity each time. A green parity gate is the definition of "done".
4. **End-to-end arbiter = teacher-forced perplexity** on real prose (with the
   correct BOS), plus top-1 next-token agreement vs the bf16 reference — not
   greedy generation (reasoning models loop under greedy regardless of quant;
   test behavior before blaming quant) and not per-expert reconstruction error
   (it does not predict e2e quality — see Settled Findings).

---

## Model facts — Kimi-K2.6 (dropped target; retained as DeepSeek-family reference)

- DeepSeek-V3-style architecture. 61 decoder layers: **L0 dense**, **L1–L60 MoE**.
- MoE: **384 routed experts + 1 shared**, top-8, `noaux_tc` sigmoid routing with
  `e_score_correction_bias`. hidden=7168, moe_intermediate=2048.
- Attention: **MLA** (multi-head latent attention) with compressed KV latent;
  `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`,
  `kv_lora_rank`/`q_lora_rank` low-rank projections.
- RoPE: **YaRN**, `factor=64`, `rope_theta=50000`, `original_max=4096`,
  `beta_fast=32`, `beta_slow=1`, `mscale=1.0`, `mscale_all_dim=1.0`. The YaRN
  attention scale is `softmax_scale = (128+64)^-0.5 · mscale²` where
  `mscale = 0.1·ln(64)+1 ≈ 1.4159` (so `mscale² ≈ 2.005`). **factor is 64, not
  96** — a wrong factor uniformly degrades every token.
- Tokens: `bos=163584`. **Two distinct eos**: the tokenizer's nominal `[EOS]=163585`
  vs the model's *generation* eos `<|im_end|>=163586` (config.json / generation_config.json
  `eos_token_id`); plus end-of-turn `[EOT]=163593`. Generation/serving must stop on the set
  `{163585, 163586, 163593}` (`<|im_end|>` is the one the model actually emits to end a turn).
- Source checkpoint ships **int4** routed experts. Param split: routed gate+up
  ≈ 676.5B, routed down ≈ 338.2B (gate/up dominate ~2:1).

Kimi-K2.6 is a **dropped target** (2026-06-13) — these facts are kept as DeepSeek-V3-family
architecture reference only. `~/models/Kimi-K2.6` is no longer required by any active work;
keep or remove it at your discretion. Baked artifacts and their `<artifact>_offload` siblings
live under `~/models`, outside this repo.

---

## Quantization policy

- **Routed experts (gate/up/down):** affine integer, group-128, per-projection
  bits chosen by the byte budget. int8-everything is ~lossless (~0.78% recon) but
  ≈975 GiB — does not fit. The split that fits ≤490 GiB is roughly **gate/up
  int3 g128 + down int4 g128** (≈438 GiB). Affine carries the zero-point bias
  that `mx.gather_qmm` needs. Whether int3 routed is *sufficient for coherence*
  is an OPEN question to be answered **only through a parity-correct runtime**
  (the int3-floor question).
- **Shared expert (gate/up/down):** **bf16, never quantized.** It runs on every
  token and is a single expert per layer, so full precision on the always-on path
  is ~free. Computed as `routed(x) + shared(x)`.
- **Attention + other matmul weights:** int8 (affine) or mxfp8.
- **norms, biases, router control tensors, positional/control tensors,
  tokenizer/data metadata:** bf16/fp32.
- Effective bits (affine) = `bits + 32/group_size`: int3 g128 = 3.25, int4 g128 =
  4.25, int8 g128 = 8.25 bpp.

---

## GPTQ — and how the matrix inverse is overcome

GPTQ minimizes the layer-wise quadratic `‖WX − ŴX‖²` over the quantized weights
`Ŵ`. Because that loss is *exactly* quadratic in `W`, the curvature is the **exact
Hessian** `H = XᵀX` (`X` = calibration activations, `[n_rows, in]`). There is
nothing to Taylor-approximate in forming `H` — GPTQ *is* the second-order
(Gauss-Newton) method. The cost is the inverse `H⁻¹` (an `[in, in]` solve;
`in = 7168` for gate/up, `2048` for down), recomputed per expert × 384 experts ×
61 layers. We overcome it on five fronts:

1. **Cholesky-of-the-inverse, not a per-weight inverse.** The Optimal-Brain-
   Surgeon update for quantizing column `j` and compensating the remaining columns
   needs only the rows of the upper-triangular factor `R` with `Rᵀ R = H⁻¹`.
   Compute `R` **once** and read every update coefficient off it (`R[j,j]` and
   `R[j, j+1:]`). No rank-1 re-inversion per weight. This is the classic GPTQ
   reformulation: one `O(in³/3)` factorization replaces `O(in³)` of repeated
   inverse downdates with bad locality.

2. **Damping for positive-definiteness.** `H ← H + λ·mean(diag H)·I` (λ≈0.01) so
   the Cholesky never fails on a rank-deficient `H` (which happens whenever an
   expert saw too few calibration rows).

3. **MLX CPU Cholesky (~32× over numpy).** Use `mx.linalg.cholesky` /
   `mx.linalg.cholesky_inv` on the **CPU stream** (MLX 0.31 has no GPU Cholesky —
   it errors "pass a cpu stream"). Measured: MLX CPU Cholesky 0.077 s vs numpy
   `inv`+`chol` 2.5 s. The "inverse" is thus a fast triangular factorization.

4. **Low-rank + diagonal Woodbury for under-covered experts (the Kimi win).**
   Under sparse top-8 routing over 384 experts with an ~8192-token calibration
   set, most experts see `n ≪ in` rows. Inverting `[in, in]` is wasteful when the
   data has rank ≤ `n`. Use the identity (exact, not an approximation):

   ```
   (λI + XᵀX)⁻¹  =  (1/λ)I − (1/λ²) Xᵀ (I + (1/λ) X Xᵀ)⁻¹ X
   ```

   which replaces the `[in, in]` inverse with the much smaller `[n, n]` Gram
   inverse `(I + (1/λ) X Xᵀ)⁻¹`. Trigger it when `n < woodbury_ratio · in`
   (≈0.5). The `λI` damping keeps both forms PD.

5. **Block + batched trailing update; shared-Hessian tail.** Quantize columns in
   `group_size` (128) blocks. Within a block, a *bounded* sequential loop over its
   ≤128 columns applies the `R`-coefficient compensation (the only sequential
   work). Between blocks, **one batched GPU matmul** propagates accumulated quant
   error to all trailing columns across every expert in the chunk at once
   (`[E,in,in] @ [E,out,in]`), so ~all FLOPs stay in dense GEMMs. Experts with
   `n < min_calib_rows` (128) reuse a pooled per-layer "shared-H" factor instead
   of a degenerate per-expert one, so cold experts are still well-conditioned.

> Status note: GPTQ produced ~4× lower per-expert reconstruction error than DWQ
> but **identical end-to-end perplexity** — proof that the int3 *coding method* is
> not the e2e lever. GPTQ stays in the toolbox; it is only worth re-running once
> the runtime is parity-correct and the int3-floor question is actually
> measurable. **Do not chase expert-quant quality before the runtime is correct.**

---

## Settled findings — DO NOT re-explore (see memory + INITIAL_PROMPT.md)

- int4 source ⇒ DWQ ≈ AWQ ≈ ~no help (scale-only methods have no headroom once
  the int4 grid already discarded the info). GPTQ error-feedback is the only
  expert-coding lever that moves recon — but not e2e.
- 3–5% *compounded* expert error is infeasible by bit allocation under 490 GiB
  (int4-all ≈ 12% recon AND ≈517 GiB > ceiling; only int8 is <1% but ≈975 GiB).
- Per-expert / compounded reconstruction error does **not** predict e2e
  perplexity. The only arbiter is teacher-forced ppl through a correct runtime.
- The e2e degeneration is **uniform** (flat per-position, flat across depth,
  wrecks even literal repetition/counting) and **expert-coding-independent**
  (GPTQ ≈ DWQ) ⇒ a localized bug in the shared forward path, NOT the experts.
- Already eliminated as the cause: RoPE `factor` (correct, 64) and the YaRN
  `mscale²` attention scale (correctly applied). Remaining suspects: int8
  attention quant, MLA matrix-absorb decode, RoPE freq construction, R2/R3
  rotations, top-k routing, KV/latent cache across positions.
- Reasoning models loop under greedy decoding regardless of quant — diagnose with
  perplexity/parity, not generation.

---

## MLX gotchas (0.31.x, this machine)

- `mx.fast.hadamard_transform` is orthonormal for `n = m·2^k`, `m ∈ {1,12,20,28}`,
  `k ≥ 1` (7168 = 28·256 ✓). **18432 = 9·2048 has NO valid factorization and
  silently returns a wrong result** — guard it; the dense FFN R4 uses 9 blocks of
  2048. Bare 12/20/28 fail to JIT.
- No GPU Cholesky (`mx.linalg.cholesky` needs a CPU stream). `mx.linalg.expm` is
  **absent** (a learned-rotation/SpinQuant path must use Cayley/QR).
- nvfp4 = group-16, mxfp8 = group-32. Affine packing is a contiguous LSB-first
  bitstream (validated == `mx.quantize` for bits 3/4/8).
- MLX slice-assignment works; `mx.async_eval` overlaps decode; one `mx.eval` per
  token (not per layer) lets MLX overlap the whole layer graph.

---

## Verification commands

Run targeted first, then broad, before committing:

```bash
uv run --with pytest pytest tests/ -m "not slow" -q   # fast inner loop (env + discovery, ~instant)
uv run --with pytest pytest tests/ -q                 # full: ALSO runs the model-free parity sweep
uv run --with ruff ruff check src tests
uv run python -m compileall -q src tests
uv lock --check
git diff --check
```

`pytest tests/` now subprocess-runs **every model-free `parity/*_test.py` gate** (the `slow`-marked
sweep in `tests/test_parity_modelfree.py`, ~4 min for ~98 gates) — this is what catches stub-vs-real
interface rot (the kind that silently broke `dsv4_tree_spec_test` + `qwen35_omlx_engine_test`). Use
`-m "not slow"` for the fast inner loop (it still runs the instant **fail-open guards**). Standalone
streaming/parallel equivalent: `uv run python -m parity.run_modelfree_sweep [--jobs N]
[--strict-skips]`. Discovery is filesystem-only and auto-includes new gates; **real-weight (SOLO,
9-306 GiB) gates are excluded** by a multi-signal, fail-toward-exclusion detector (`*_real_test.py`
name; a `/Users/pmrj/models` or `set_wired_limit` marker; a `~/models` *load idiom* — `Path.home(`/
`expanduser`/`os.environ`/`getenv` **with** `models`, NOT the bare `~/models` literal, which appears
in commented-out code in model-free gates; or an explicit `# parity-gate: real-weight` sentinel) so
the sweep never loads a resident model.

**Enforcement (this is automated now, not by-hand):** a committed **pre-commit hook**
(`.githooks/pre-commit`, activate once with `git config core.hooksPath .githooks`) runs the fast
fail-open guards on every commit; the **CI workflow** (`.github/workflows/parity-gates.yml`,
Apple-silicon `macos-14`, `uv sync --extra reference`) runs the full suite + `--strict-skips` on
push/PR. Hardening (all gated in `tests/test_parity_modelfree.py`): the fail-open backstop is an
**identity-pinned manifest** (`parity/gate_manifest.json` — the exact NAME SET of each bucket, so an
offsetting add+remove that a count would miss is caught, and a real-weight gate that *evades*
detection shows up as a new name in `model_free` and fails LOUD); `run_gate` rejects a **vacuous
pass** (rc-0 with a printed Traceback, unless the gate prints the opt-in `PARITY-CHECKS: <n>` n>0
proof-of-work — also the escape hatch for a gate that legitimately renders a Traceback;
`PARITY-CHECKS: 0` always fails); a **misnamed-gate scanner** flags model-free gates (incl.
pytest-style) hidden behind a non-`_test.py` name; an **allowlist-staleness** guard; and a gate
needing an **optional extra** (the skip-eligible set is read from `pyproject.toml`, never hardcoded;
≈11 import `safetensors`) is **skipped, not failed**, on a base-deps-only env. Two **runtime
backstops** close the fail-open residual (a real-weight gate that evades the *static* `is_real_weight`
detector — e.g. an env-var artifact path with no `models` substring): `run_gate` polls each swept
gate's RSS and **kills + fails-loud any gate crossing a 4 GiB ceiling** (model-free gates peak
~0.4 GiB; the smallest real load is 9 GiB, so the load is killed before it faults in hundreds of GiB
and OOM-reboots the box), and the standalone sweep **refuses to run on manifest drift** (an
undetected gate shows up as an `added` name and stops the sweep before it loads;
`--update-manifest`/`--allow-drift` to resolve/override). A green sweep proves
**interface + logic on stubs, not real-model numeric parity** (that is the excluded SOLO gates).
**When you add/remove/reclassify a `parity/*_test.py` gate, regenerate the manifest:** `uv run
python -m parity.run_modelfree_sweep --update-manifest` and review the diff.

---

## Memory

Permanent cross-session memory for this project lives in the auto-memory dir and
is loaded via `MEMORY.md`. The permanent engineering rules above are mirrored
there as a feedback memory so they are never dropped. Settled findings and the
user profile are seeded so a fresh session starts informed, not from zero.

## Git / collaboration rules

- Do **not** commit unless explicitly asked. Add files by name (never blind
  `git add -A`). Never push unless asked. Never skip hooks.
- Commit trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Baked artifacts are immutable bundles; manifest references are relative,
  in-artifact only (no absolute/source/symlink/cache paths). Runtime offload
  state lives in the sibling `<artifact>_offload`, never inside the artifact, and
  `manifest.json` is never mutated at runtime.
