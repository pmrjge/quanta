---
name: project-model-targets
description: "SERVED FLEET (2026-06-11) = InternLM2.5-7B int8g64 (9 GiB) + Qwen3.6-35B int4g64 (19) + Nemotron-Super-120B int4g64 (68) + DSV4-Flash int4g64 (180) + Nemotron-Ultra-550B int4rtn (306) + Nex-N2-Pro/Qwen3.5-397B int4g64 (214, ACTIVE). Grew from the 2026-05-28 three-keeper set (Nemotron+DSV4+InternLM2.5). One model resident at a time."
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

**CURRENT STATE (2026-06-11) — the served fleet is now FIVE+ models, all int4/int8 resident, ONE AT A
TIME** (decode baseline table in CLAUDE.md): InternLM2.5-7B int8g64 **9 GiB**, Qwen3.6-35B-A3B int4g64
**19**, Nemotron-Super-120B int4g64 **68**, DSV4-Flash int4g64 **180**, Nemotron-Ultra-550B int4rtn_g64
**306**, **Nex-N2-Pro = Qwen3.5-397B-A17B int4g64 214 (SHIPPED, the ACTIVE task — N3 serving)**. NEW since
the three-keeper decision below: **Nemotron-Ultra-550B** ([[project-nemotron-ultra]], SHIPPED int4-RTN —
the group-wise-RMSNorm bug + AWQ→RTN e2e findings) and **Nex-N2-Pro** ([[project-nex-n2-pro]]). The
2026-05-28 "Qwen3.5 deprioritized/DROPPED" framing is **superseded** — Nex-N2-Pro IS a Qwen3.5-397B and
`quanta.qwen35` is its production runtime (the Qwen3.6-35B keeper proved the forward; Nex is
validate-at-scale + bake). The three-keeper text below is the historical decision; the **per-model
architecture + paging notes remain valid**. EAGLE (InternLM2.5) + MInference M0–M10 done; the whole
#152/#153 paged + #158-160 tree-spec-over-paged tracks are CLOSED across keepers.

---
**FOCUS DECISION (2026-05-28): the agentic serving loop targets only THREE models** — to keep the
loop simple/fast for agentic workflows:
1. **Nemotron** — Mamba-2 + GQA hybrid (linear-attn scales for long agentic context), native MTP.
2. **DeepSeek-V4-Flash (DSV4)** — sparse MoE + compressed-KV + Lightning-Indexer, native MTP.
3. **InternLM2.5-7B-Chat-1M** — small dense GQA, 1M ctx, cheap always-on model.
   *Replaces Qwen2.5-7B-1M (#163), whose long-context runtime is broken* — InternLM2.5 (#165) is the
   working small-dense-1M serve target. Immediate goal: get InternLM2.5 **on par** (bf16 ref-forward
   parity → bake → e2e ppl gate → oMLX shim wiring) with the other two.

**Kimi-K2.6** stays only as the int4 reference teacher (CLAUDE.md says never delete it) — it is **not**
a serving keeper: EAGLE-3 spec-decode on it was a benchmarked **no-op** ([[project-eagle3-speculative]]).
**NO further improvement work on Kimi (user directive 2026-05-28: "forget Kimi for improving more, only
focus on deepseek nemotron and small model").** Kimi is frozen as the reference teacher — never delete,
never optimize. All dev effort goes to the three keepers (DSV4 + Nemotron + InternLM2.5).
**GLM / MiniMax / Qwen3.5 / Qwen2.5** are fully built (forward + ppl + bake + runtime + serve, see the
task tracker) but **out of the agentic focus** — kept in-tree, not actively developed. None were
deleted; this is a priority decision, not a removal directive.

**#152 PAGED/BATCHED SCOPE — Qwen3.5 DROPPED (2026-05-28, user directive "forget Qwen35, only
nemotron, deepseek and the small model").** The #152 prefix-sharing paged-KV + unified BatchedEngine
work targets ONLY the three keepers: **Nemotron + DSV4 + InternLM2.5**. (The original #152 plan listed
Qwen3.5 as the third batched model — that was a mismatch with the keeper set; corrected here.) The
just-added Qwen3.5 batched-session + `parity/qwen35_unification_test.py` (#173) are now out of scope —
qwen35 stays in-tree but is not extended. Per-model paging:
- **Nemotron** (hybrid Mamba+GQA): page the 8 GQA `KVCache` layers; the 40 Mamba layers can't skip a
  shared prefix (recurrent state at pos n depends on all 0..n-1). VERIFIED in code: prefill is one
  full-prompt forward through all 88 interleaved layers. Suffix-compute is enabled by content-
  addressing the **recurrent boundary state** (`quanta.paged.recurrent_cache.RecurrentPrefixCache`,
  keyed by the SAME chain-hash as the attention blocks) — restore it + run only the suffix == full,
  proven model-free in `parity/paged_recurrent_suffix_test.py`.
- **InternLM2.5** (small dense GQA, 32 layers, NO recurrent/MoE): the CLEAN paging case — attention-KV
  reuse + suffix prefill, no recurrent snapshot. Gap: it has **no `batched_runtime` yet** (only
  single-stream serve from #165-171) — #152 must add one (simple: dense, no MoE amortization).
- **DSV4** (#175, biggest risk). DESIGN RESOLVED (research): compressor pools RAW HIDDEN (not latent KV)
  — so the prefix's compressed/indexer streams CANNOT be recomputed from the shared latent KV. Approach
  mirrors #174's two-cache split: (1) latent KV -> PagedKVCacheManager (shared prefix blocks); (2) the
  derived per-layer boundary state = (compressed-KV, indexer-KV, raw-hidden ring) -> content-addressed
  via RecurrentPrefixCache (opaque payload, same chain-hash) — NOT block-shared (dodges the
  block_size<->compress_ratio 4/128 alignment coupling). Suffix prefill appends latent KV + pools only
  its own windows seeded by the restored ring; the boundary-straddling window (ring reaching back
  coff*ratio raw-hidden across the boundary) is THE risk (guard at dsv4/decode.py:~358). HARD because
  DSV4Cache is bespoke (3 regimes: ratio 0 dense-window / 128 compressed / 4 compressed+indexer). Reuses
  the existing paged infra unchanged; only the DSV4 runtime wiring + a paged-latent _LayerCache variant
  are new. Gate: parity/dsv4_paged_latent_test.py across ratio-4 AND ratio-128 boundaries + real-model.

**#152 COMPLETE — CLOSED (2026-05-28). PAGED_KV_DEFAULT flipped to True.** All 5 build steps committed
(83fa58d paged core #172, 8ff8e81 engine unify #173, 8a69797 nemotron paging #174, f2e6e40 DSV4 latent
paging #175, 0394618 InternLM2.5 paging #176 + #177 nemotron-test fix). ALL THREE keepers' real paged
gates GREEN: Nemotron (`nemotron_paged_real_test` 10/10), InternLM2.5 (`internlm2_paged_real_test` 10/10),
**DSV4 (`dsv4_paged_real_test` top-1 10/10 AND per-step logits |Δ|=0.000e+00 BIT-EXACT** — DSV4's
token-by-token decode keeps the discrete/paged SDPA shapes identical; prefix reuse 32 tok/2 blocks +
derived boundary snapshot restored snapshot_hits=1). So PAGED_KV_DEFAULT is now **True** — the engine
defaults to the prefix-sharing paged path for the 3 keepers. Qwen3.5 is forced UNPAGED in
`_make_batched_session` (out of paged scope, no paged contract — rule 6: never paged_kv=True with no spec).
SETTLED REGRESSION FIX: the #177 group_size cap must use `getattr(cfg, "head_dim", 128)` (not
`cfg.head_dim`) — fake-cfg test runtimes (e.g. `nemotron_omlx_engine_test`'s SimpleNamespace) have no
head_dim; real cfg=128 -> min(128,128)=128 unchanged. RESOLVED (967bf22, 2026-05-28):
`parity/nemotron_tree_spec_test.py` now 6/6 — the model-free stub exposes `batch_step` (shared-offset
batched verify surface) so subtests (1)-(4) run the default `batched=True` tree-verify; subtest (5) is
pinned to `batched=False` for the per-path cache-rollback assertion. (Was 5/6: stub lacked batch_step.)

**#179 DEFERRED GPU PAGED-PPL GATES — DONE, ALL THREE GREEN (2026-05-28).** The one-model-at-a-time
follow-up to #152: teacher-forced ppl PAGED ON == OFF on each real keeper (the e2e arbiter) + an
engine-level `session.get_cache_stats()` smoke folded into the 3 `*_paged_real_test.py` gates. New
harnesses `parity/{internlm2,nemotron,dsv4}_paged_ppl.py` (from-scratch offset=0, recurrent_in=None;
score via the inner ResidentModel through the CANONICAL paged state — `paged_kv_spec` + `make_paged_state`,
all positions). ALL **BIT-EXACT** (|Δlogit|=0.00e+00, Δppl=0, top-1 identical): InternLM2.5 repeat 1.27 /
prose 11.39; Nemotron repeat 1.80 / prose 5.81 (≈ dequant ref 5.80); DSV4 repeat 1.34 / prose 5.54. The
stats fold asserts `get_cache_stats()` == the manager's `prefix_hit_{tokens,blocks}`; Nemotron/DSV4 also
check the `recurrent` sub-dict (`rec.get_stats()` snapshot_hits/stores). SETTLED (do not re-discover): the
from-scratch ppl gate is bit-exact because both paths are one-shot over the SAME q_len, varying ONLY the KV
storage (discrete cache vs paged view); for Nemotron hold `mamba_chunked_cont` identical (False) on both
sides so only the GQA-KV path differs. (The reuse path's residual |Δ|~2–3.25 from #174/#176 is continue-
from-prefix bf16 noise, top-1 unaffected; DSV4's reuse is |Δ|=0.)

**DSV4 RESIDENT FOOTPRINT = ~180 GiB, NOT ~389 GiB (CORRECTED 2026-05-28).** Measured the default load
`DSV4BatchedResidentModel(ART)`: `mx.get_peak_memory()`=179.9 GiB, active 178.8. ALL 43 expert layers load
**int4-packed** (uint32 codes via `expert_stacks_packed` → `mx.gather_qmm`, ~3.75 GiB/layer) because
`DSV4ResidentModel` defaults `packed_experts=True` (#141). The **~389 GiB** number is the bf16
`expert_stacks()`→`gather_mm` path (`packed_experts=False`, a short-prefix-debug/parity-ref fallback, NOT
the default). Stale `~389 GB` docstrings corrected in `parity/dsv4_paged_{ppl,real_test,latent_test}.py` +
`dsv4_decode_attn_test.py`. NOTE `parity/eagle_capture*.py`'s 389 GiB is **Kimi int2g64** (a different,
genuinely ~389-GiB model) — correct, left intact.

**INTERNLM2.5 BATCHED-DECODE ATTENTION = DEFAULT + REAL-MODEL VALIDATED (2026-05-28, 0d531c3).** The
fused `step_batch`->`decode_batched` path (Approach-1: pad B ragged streams' KV to L_max + additive mask
+ ONE `mx.fast.scaled_dot_product_attention` + batched `_qmm`) is now the DEFAULT batched decode for
InternLM2.5 (`_fused=True`; falls back to the retained `_step_batch_looped` only for multi-token steps).
Real int8-g64 bench `parity/internlm2_batched_bench.py` (1024-tok prompt, 64 gen/stream): B=1 BIT-EXACT
vs loop (0.00e0), B=4 ragged GREEDY-EXACT. Throughput agg tok/s fused/looped: B=1 43/48 (0.91x), B=4
85/59 (1.45x), B=8 94/50 (1.88x), B=16 114/52 (2.19x), B=32 136/50 (2.70x) — fused still CLIMBING at
B=32 (not plateaued), memory FLAT ~8.8 GiB across all B (resident weights amortized across streams). The
bf16 RoPE bug found+fixed in the process -> [[feedback-batched-rope-bf16]]. This CLOSED the /loop "verify
the remaining risk": the never-validated packed `decode_batched` path is now real-model green. B=1 0.91x
(vs single-stream) is a known minor wrapper cost, NOT the serving regime; a `B==1` fast-bypass is an
unstarted micro-opt. NOTE the bench/runtime use `InternLM2BatchedResidentModel` (the #176 paged batched
runtime); the model-free parity gate is `parity/internlm2_batched_attention_test.py`.

**#176 INTERNLM2.5 PAGING — DONE + MODEL-FREE BIT-EXACT + REAL-MODEL VALIDATED (2026-05-28).** The clean
dense case: dense GQA, NO recurrent/MoE/MTP state. New `src/quanta/internlm2/batched_runtime.py`
(`InternLM2BatchedResidentModel`, `has_recurrent_state=False`, `from_inner` for model-free gates): pages
every layer's k/v-pair `KVCache` via the existing `PagedKVCacheView` (NOT single-stream); `prefill_paged`
returns NO boundary payloads (nothing to content-address — the engine's `_admit_paged` already short-
circuits the recurrent branch when has_recurrent_state is False). `_InternLM2BatchedSession` + `internlm2`
dispatch arm added to omlx.py (the base `_BaseBatchedSession` does all admit/step/release). Gate
`parity/internlm2_paged_test.py`: **|Δ|=0.00e+00 bit-exact** (int8 g32 + bf16, core + real engine session
req1/req2). Real gate `parity/internlm2_paged_real_test.py` (int8g64 artifact, ~6 GB): **top-1 10/10**.
SETTLED FINDING (do not re-discover): the model-free gate MUST run the bf16 model AND use a discrete
CONTINUE-FROM-PREFIX reference (not one-shot) — the paged manager gathers bf16, so a float32 default-init
model OR a one-shot (different q_len) reference shows ~1e-3 bf16 noise that is NOT a paging bug.

**#177 NEMOTRON TEST FIX (2026-05-28, committed-pending with #176).** `parity/nemotron_batched_test.py`
crashed (pre-existing, independent of #152): tiny cfg head_dim=8 + int8-KV-default group_size=128 ->
mx.quantize needs head_dim % group_size == 0. Fix (user directive "no bigger KV cache"): head_dim=32 (min
valid — mx.quantize's smallest group_size is 32) + cap each Nemotron KV cache's group_size at head_dim
(`min(128, cfg.head_dim)`) in `make_step_state`/`make_stream_state` (batched_runtime) + `attn_caches`
(generate.py). Real Nemotron head_dim=128 -> min(128,128)=128, byte-identical to before (zero prod change).

**#175 DSV4 LATENT PAGING — DONE + MODEL-FREE VERIFIED BIT-EXACT (2026-05-28).** Built under the
constraint "finish DeepSeek without ever using the GPU or touching memory" → model-free gate only, real-
model ppl-parity DEFERRED to a one-model-at-a-time GPU session. Design as planned: (1) the **single
latent KV stream** (DSV4 is MQA — one latent shared as both K and V, `[B,S,head_dim]`) is paged via a NEW
**single-stream codec** on `PagedKVCacheManager` (`single_stream=True` → `write_one`/`gather_one`/
`view_one` + `PagedLatentCacheView`; half the components/bytes of a k/v pair; the k/v path is byte-for-
byte unchanged — `write` refactored to share `_write_encoded`, all k/v gates still pass). (2) the derived
**compressed-KV + indexer-KV + raw-hidden ring** (the compressor pools RAW hidden, NOT the latent, so
they can't be recomputed from the shared latent) are content-addressed at block boundaries via the
existing `RecurrentPrefixCache` as an opaque `_DerivedSnapshot` list (`snapshot_derived`/`restore_derived`
in dsv4/decode.py; `_PagedLayerCache` overrides only latent kv/append_kv/kv_length/truncate_kv, inherits
ckv/ikv/ring). New on `DSV4BatchedResidentModel`: `has_recurrent_state=True`, `paged_kv_spec`,
`make_paged_state`, `prefill_paged` (deepest-block split + snapshot, like Nemotron), `get_recurrent_state`;
`_latent_quant(head_dim)` mirrors `_LayerCache._resolve_quant` so paged==discrete quant exactly.
SETTLED FINDING (do not re-discover): **the snapshot approach is UNCONDITIONALLY correct re
block_size↔ratio alignment** — the ring snapshot already carries the `coff*ratio` raw-hidden tail, so a
boundary-STRADDLING first-suffix window (ratio-4 overlap `prev`/part-`cur` in the prefix; ratio-128
non-overlap when P not a ratio multiple) pools correctly. Gate `parity/dsv4_paged_latent_test.py`:
**|Δsuffix|=0.00e+00 bit-exact** across dense / ratio-4(+indexer) / ratio-128 regimes with P a block
multiple but NOT a ratio multiple (straddle=True), PLUS a real `_DSV4BatchedSession` 2-request reuse
(req1 from-scratch == discrete one-shot, req2 reuses prefix blocks+snapshot, hit_tok=8 snap_hits=1, both
|Δ|=0.00e+00). Only the DSV4 runtime wiring + the single-stream codec are new; the paged infra is reused.

Constraint reaffirmed (2026-05-28): real-model parity gates may load **one model at a time, never two
concurrently** (the OOM-reboot hazard, [[feedback-memory-safety]]).

**#174 NEMOTRON PAGING — DONE + REAL-MODEL VALIDATED (2026-05-28).** Paged attention KV + recurrent
boundary snapshots work; the real int4-g64 artifact gate (`parity/nemotron_paged_real_test.py`) is
top-1 **10/10** (paged suffix-compute == discrete full prefill, coherent prompt, varied tokens).
SETTLED FINDING (do not re-discover): a Nemotron Mamba **suffix prefill on a restored recurrent state
must use the CHUNKED-SSD continuation, NOT the per-token-step continuation** — the per-token branch
matches `batch_step` (spec-verify) but DIVERGES from the fresh chunked prefill in bf16 (the real gate
first showed 4/10 top-1 with per-token; fixed to 10/10 with chunked). Implementation: a `chunked_cont`
flag threaded `mamba_mixer → NemotronBlock → NemotronResidentModel.__call__ (mamba_chunked_cont)`,
used only by `NemotronBatchedResidentModel.prefill_paged`; it prepends the restored conv window for the
conv1d left-context then `ssd_chunked(state_in=state)`. Default False everywhere (decode/spec unchanged).
Residual logits max_abs ~3.25 (chunk-boundary bf16) doesn't flip argmax. Engine flag `paged_kv` (default
`PAGED_KV_DEFAULT=False`). **Pre-existing unrelated breakage found:** `parity/nemotron_batched_test.py`
crashes (`head_dim=8` tiny config + int8-KV-default → quantize g128 fails) — confirmed independent of
#152 (fails on committed baseline); flagged as a separate test-only fix.

----
Historical (2026-05-24) — full multi-model context, retained for the per-model architecture notes:

quanta is **multi-model**, not Kimi-only (earlier memory implied otherwise). Each target gets the
full parity-first treatment (reference forward → layer parity → bf16 teacher-forced ppl gate → bake
→ resident runtime) and its own tokenizer + oMLX engine. In flight as of 2026-05-24:

- **Kimi-K2.6** (primary / reference target) — DeepSeek-V3-style MLA, 384 routed +1 shared experts
  top-8; tiktoken tokenizer ([[project-tokenizer-eos]]); baked **int2-g64** + int4-g64 artifacts;
  EAGLE-3 spec-decode ([[project-eagle3-speculative]]).
- **Nemotron** (Nemotron-H / -3-Super) — **Mamba-2 + GQA hybrid** (SSD chunked prefill, conv1d
  decode state), latent relu² MoE noaux_tc top-22; HF fast tokenizer, **two eos** (`</s>`=2 vs chat
  `<|im_end|>`=11); int4-g64 experts. oMLX reasoning/tool formats are native — no patch
  ([[project-omlx-serving-contract]]). Full forward + int4 ppl gate done; the oMLX engine (Mamba/GQA
  generation loop, ≠ the MLA shim) is task #39.
- **DeepSeek-V4-Flash (DSV4)** — 43 layers + MTP, **Hyper-Connections** (Sinkhorn), 3 attention
  regimes (low-rank q/kv, Compressor KV pooling, Lightning-Indexer DSA sparse windowed), fp4/fp8/e8m0
  quant; **GPT-2 byte-level BPE** tokenizer (129280 vocab, pure-Python, needs the `regex` dep). Full
  forward ppl **5.091 / 57.4% top-1**; bake/runtime/MTP pending (#76–78).

CLAUDE.md still frames targets as "Kimi (then GLM-5.1, DeepSeek-V4-Pro)" — the actual in-flight set
is the three above. Per-model task ranges: #29–41 Nemotron, #67–79 DSV4 (the live task tracker is
the source of truth for status — don't duplicate it here).

**MiMo** (fp8 multimodal, ex-#55–66) was **removed from the project on 2026-05-25**: the
``src/quanta/mimo`` package + ``parity/mimo_*`` gates were deleted and its tasks dropped. The only
trace is git history (commit 25397f4); do not re-add it without a new directive.
