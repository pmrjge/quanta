# CLAUDE.md — quanta

`quanta` is a **parity-first**, MLX-native quantization + sparse-MoE inference
runtime. It is a clean restart of the prior `quantification` effort, keeping the
hard-won findings (below) but rebuilding the runtime so that **every component is
gated against a numeric reference before it is optimized or its quantization is
judged.** The immediate target is **Kimi-K2.6** (then GLM-5.1, DeepSeek-V4-Pro).

The previous build reached a runtime that produced incoherent output (teacher-
forced perplexity ~165 with BOS, ~5–9× fuzzy on trivial tasks) and the failure
was wrongly attributed to int3 expert quantization. It is **not** the experts
(see Settled Findings). It is a localized bug in the forward pass that was never
caught because the runtime was built and refactored **without a parity gate**.
That is the mistake this project exists to not repeat.

---

## Active task (transient — full handover in PLAN_nemotron_ultra.md)

**In flight: Nemotron-3-Ultra-550B serving runtime.** (The second agentic-stack model is **deferred to
MiniMax-M3 when it ships** — **Mellum2 was dropped, its context length is too short**; the `minimax`
module is already substantially ported in-tree.) Handover **`PLAN_nemotron_ultra.md`**. Quantize
Nemotron-Ultra (hybrid Mamba2+attn+MoE, `nemotron_h` — already supported; the 120B-Super sibling is
already baked int4) as **int4-RTN experts + int8 dense + bf16 core** (user pivoted experts → AWQ
mid-session and the U2 slice de-risk cleared AWQ on *recon*, but **U3's e2e ppl arbiter retired AWQ →
int4-RTN ships** — recon ≠ e2e, see below); **one model resident at a time**; drive Ultra to completion. **U0 ✅** — config adapter (`_hybrid_pattern`
normalises the newer explicit-`layers_block_type` schema, which omits `num_hidden_layers`) +
fit-check: Ultra parses, the derived split reproduces the explicit list bit-for-bit, the quant policy
covers all 51,023 tensors (rule #6), and the mix is resident at **305.9 GiB ≤ 490.4** (184.5 GiB
headroom — the original U0 projection said 289.7 but used g128; **reconciled**, see below) —
`parity/nemotron_ultra_fit_test.py`. **U1 ✅** — per-layer numeric parity vs an
**independent transformers `NemotronH*` reference** at full Ultra scale, layer-streamed (rule 8: one
real layer resident, the moe's ~21.5 GiB expert stacks the peak), `parity/nemotron_ultra_layer_parity.py`:
**mamba** our `MambaMixer` vs `NemotronHMamba2Mixer` (fp32, Δ 3.1e-04), **attn** ours vs transformers'
`apply_rotary_pos_emb`+`eager_attention_forward` (Δ 4.5e-06), **moe** router top-22 set+weights vs
`route_tokens_to_experts` (set-exact, w Δ 1.2e-07 — our `noaux_tc` sigmoid routing is provably exact)
+ experts/latent/shared vs inline-dense (Δ 7e-04) + chunk-invariant. **The gate caught a real
forward-path bug** (the kind CLAUDE.md's thesis warns about): the Mamba-2 **gated RMSNorm is
group-wise** (variance over `d_inner//n_groups`, NOT full `d_inner` — `Zamba2RMSNormGated`); ours was
a full-width `nn.RMSNorm` — *self-consistent* (prefill==decode) so the old self-consistency-only test
never caught it, but **42% off** the reference. Fixed via a new group-wise `MambaRMSNormGated`
(`mamba_mixer.py`, forward-only — corrects the **already-baked Super-120B** too; bf16 `norm.weight`
unchanged, no re-bake; **Super ppl re-measured under the fix** (`parity/nemotron_{ppl,int4_ppl,resident_ppl}`,
same unchanged 109-tok PROSE yardstick): bf16 **5.981→3.379**, served int4g64 **resident 3.305** (≈lossless,
−0.7% vs the 3.327 dequant ref) — the pre-fix baseline was measuring the degraded buggy-norm forward, which
the residual skip kept coherent-ish at 5.981; the fix recovered ~2.6 ppl points). **U2 de-risk ✅** — slice
diagnostic `parity/nemotron_ultra_awq_slice_test.py` (Ultra L1, the first MoE; streams layers 0–1, NO
21.5 GiB expert stack materialized): per warm expert, held-out activation-weighted recon error, AWQ vs
RTN. Finding #38's relu² down-proj AWQ collapse **does NOT reproduce at Ultra** — AWQ *helps* up-proj
(held-out ratio 0.806) and *ties* down-proj (0.984, 23/24 experts AWQ≤RTN); the relu² sparsity
precondition is present (99.74% near-zero channels) but AWQ's grid rejects the degenerate scales (range
≈1, not ≈1e6, so the folded `1/s` never blows up). Caveat: L1-only + activation-weighted recon ("far
more e2e-predictive than raw recon" per `bake/calibrate.py`, but NOT e2e ppl — U3 is the arbiter).
**U2 ✅** = full int4-AWQ g64 + int8 bake (0.48h solo; recipe since removed) →
`…-quanta_int4awq_g64`: 108 layers / 48 moe / 512 experts, **warm 24,235/24,576 (98.6%)** real AWQ scales,
341 cold→plain int4 RTN; **artifact audited self-contained + fully covered** (no symlinks, zero external
refs in index/manifest/config, weight_map relative, 42/42 shards, tokenizer in-artifact, manifest
`format=quanta` 49,983 tensors, all 108 layers + 512 up/512 down experts/moe + embed/head/norm_f);
**resident 336 GiB** (≤490.4, 154 GiB headroom — AWQ exceeds RTN's 306 because it stores fp32 affine
expert scales vs RTN's bf16). **Fit-test RECONCILED** (no longer the stale 289.7): the U0 projection
hardcoded **g128** but both Nemotron bakes ship **g64** (`…_int4{rtn,awq}_g64`) — `quant_policy`
constants → g64 + the per-expert bf16 `awq_scale` vector (`ones` under RTN, 0.33 GiB) now modelled →
**305.9 GiB**, and the fit-test gained a rule-#6 cross-check that the projection matches the on-disk
backbone (Δ **0.01%** vs 306.0 GiB / 39 shards; Super likewise 68.1==68.1). **U3 ✅ → SHIP int4-RTN**
(`parity/nemotron_ultra_ppl.py`,
3 sequential rule-8 streamed forwards over a held-out 1024-tok prose corpus, ≈10× the 109-tok pilot):
**bf16 ppl 3.835/acc 0.651**, **int4-AWQ 4.766/0.604/Δ +24.3%/agree 0.811**, **int4-RTN 3.845/0.644/Δ
+0.3%/agree 0.964** — RTN ~lossless, **AWQ regresses hard** (the relu² down-proj tax got *worse* with
more tokens, +11.2%→+24.3%; the U2 recon de-risk could not see it — recon ≠ e2e, settled finding). So
**finding #38 reproduced e2e** and the RTN fallback ships: `expert_method="rtn"`, data-free bake
(`parity/run_bake_nemotron_ultra_int4rtn_g64.py`, 0.10h, warm_experts 0; inventory-identical to AWQ),
same 4-bit footprint, **306 GiB resident (30 GiB < AWQ's 336** — RTN stores bf16 vs AWQ's fp32 expert
scales). AWQ artifact retired. **U4 in progress** (user picked the **packed int4 + `gather_qmm`**
resident-decode stream first; that path is already coded — built+gated for Super-120B in
`moe.py`/`runtime.py`): **M1 ✅** — `NemotronQuantizedMoE` gather_qmm over packed int4-g64 stacks ==
dequant bf16 reference at Ultra L1 (512 experts, latent 2048 / inter 5120), **rel err 0.028% « 2%**
(`parity/nemotron_ultra_qmoe_test.py`, quantized side built by the *real* runtime constructor
`build_resident_block(art, cfg, 1).mixer`; RTN ⇒ s=1, gather_qmm decodes the same grid). **M2 ✅** =
full-resident e2e ppl — loaded `NemotronResidentModel` over the **306 GiB RTN artifact** RAM-resident
(solo, 400 GiB wired, load 1.9 min, freed clean) and teacher-forced the **same** U3 1024-tok corpus
(metric imported verbatim): **ppl 3.839 / acc 0.646** == U3 streamed-dequant RTN ref **3.845 / 0.644**,
**Δ −0.1% « 2% — PASS** (`parity/nemotron_ultra_resident_ppl.py`; the −0.006 is resident bf16-head vs
streamed fp32-head, within noise). **Packed-int4 + gather_qmm stream COMPLETE e2e** — M1 gated the MoE
at one layer, M2 ran the whole 108-layer resident model so it also covers the **dense mamba/attn int8
`QuantizedLinear` wiring** end-to-end. **U4 next stream = native MTP spec-decode** (user-picked;
`mtp.py`/`spec.py` + the model-free `nemotron_mtp_spec_test` were built for Super but the head was never
baked/loaded — task #40): **MTP-M0 ✅** — native MTP draft-head **bf16 numeric parity** at full Ultra
scale (`parity/nemotron_ultra_mtp_parity.py`): build `NemotronMTPModule` (fuse `eh_proj(concat([enorm(
embed), hnorm(prev_hidden)]))` → attn sub-block `mtp.layers.0` → 512-expert relu² latent-moe
`mtp.layers.1` → final_layernorm → shared head), fill from the source's **1040 `mtp.*` tensors** (rule-6
coverage 1040/1040), diff vs an **independent inline reference** (raw-mx fusion/pre-norms/residuals/
readout + U1-gated standalone mixers) — **logits Δ 0.0 / new_hidden Δ 0.0 (bit-identical)**, rule-8
streamed (the 512-expert ~21.5 GiB bf16 stack the peak, solo). Gates the head's *structural assembly*;
the *functional* accept-rate is the separate **MTP-M2** gate (losslessness holds for ANY head quality —
the main model verifies every draft). **MTP-M1 ✅** — bake the head into a self-contained **sidecar**
(`…-quanta_int4rtn_g64_mtp`; the immutable backbone artifact is never touched, M2's loader pairs the
two): same policy as the backbone (int4-RTN experts + int8 dense + bf16 core; `quant_policy` already
classifies `mtp.*`), new `bake_nemotron_mtp` (bake.py) streams one expert resident (rule 8, 0.08 min,
data-free RTN warm 0) → **1040/1040** tensors in a single 6.56 GiB shard, audited self-contained (zero
path leaks, relative refs only, manifest **9 int8 / 7 bf16 / 1024 int4**). Gated solo
(`parity/nemotron_ultra_mtp_bake_parity.py`, two 21.5 GiB heads loaded **sequentially** — peak one head):
(1) coverage+format exact vs `classify` (1040/1040, dense/affine_packed/awq_packed), (2) **bit-exact
faithfulness** — an *independent* in-script RTN `quantize_affine` of the source reproduces the baked
packed/scale/bias **bit-for-bit** (eh_proj + experts 0/256/511; awq_scale==ones, s=1), (3) **recon
forward** baked-dequant head vs bf16 head through the *identical* M0-gated `NemotronMTPModule` (bf16
router ⇒ routing identical on both sides ⇒ the delta is pure quant): **logits Δ 7.0% / new_hidden Δ 7.8%
< 10%, top-1 agree 0.875** — the inherent int4-g64 expert recon (the bit-exact gate is the tight
correctness proof; recon is bounded-not-tight), and a *drafter* so it only moves accept-rate, never
correctness (main model verifies every draft). **MTP-M2 ✅** = native MTP spec-decode wired into the
resident loop + real lossless gate. (1) **Loader** `build_resident_mtp` (`runtime.py`) fills
`NemotronMTP` from the baked sidecar `mtp.*` — packed-int4 experts via `gather_qmm` + int8 dense
`QuantizedLinear` + bf16 core — mirroring `build_resident_block`. (2) **Resident spec adapter** on
`NemotronResidentModel`: `make_caches` (the `(caches, ssm, conv)` triple, `max_rollback=8`), `truncate`
(KV only — the Mamba `(ssm,conv)` summary can't be sliced, the spec loop handles it), `offset`
(accepted-and-ignored; the KV cache tracks position). (3) **The gate caught a real k=1 hybrid bug** (the
CLAUDE.md thesis again): `spec_generate` (k=1) never rolled the un-sliceable Mamba recurrence back on a
*rejected* draft (only `spec_generate_k` k≥2 did) — so k=1 corrupted the `(ssm,conv)` summary on the
hybrid. Fixed: on reject, snapshot/restore `(ssm,conv)` + re-run `[cur]` (gated on `ssm is not None`; the
stub/pure-attn path is byte-identical), gated **bit-exact model-free** (`nemotron_mtp_spec_test` gate 7 —
a Mamba-carrying stub whose argmax depends on a running recurrent state, so a non-rolled-back rejected
draft would diverge: spec==greedy with the rollback branch firing). **Real gate**
(`parity/nemotron_ultra_mtp_resident_spec.py`, solo, 306 GiB backbone + 6.56 GiB sidecar resident,
eager): `spec_generate_k` k∈{1,2,3} vs greedy on 64-tok prose → 48 tok. **k=2/k=3 EXACT (bit-identical
48/48)**; **k=1 bit-identical 24/48 then a *confirmed* bf16 ULP near-tie** (spec's token is greedy's
**rank-2** runner-up, **margin 0.125 ≈ 1 ULP** on greedy's own per-token step path) after which the two
*valid* greedy trajectories chaos-diverge. mean_accept **1.52/2, 1.81/3, 1.81/4** (the real trained head
drafts well). **Settled finding (new):** on a bf16 **Mamba hybrid** the spec VERIFY forward (T>1) and a
T=1 decode differ by ~1 bf16 ULP (`path_ulp`=0.1875 — attention `mask=None`-vs-`causal` + the recurrence;
the Mamba-mixer chunked-vs-step note is the same class), so **"spec == T=1 greedy" is the wrong
real-weight criterion** (CLAUDE.md: *test behavior with parity, not greedy generation* — a single ULP
near-tie flip cascades chaotically). The honest gate: **logic is bit-exact (gate 7); on real weights spec
is bit-identical up to the first divergence, and that first divergence (the only valid-prefix position)
must be a verified near-tie** — a large-margin/low-rank first divergence would FAIL as a logic bug.
**MTP-M3 ✅** (perf — `parity/nemotron_mtp_k_bench.py` re-pointed to `build_resident_mtp` + the Ultra
backbone, wall-clock spec-vs-**compiled**-greedy, solo ~313 GiB): single-stream B=1 lossless spec tops
out at **0.79× greedy** (`draft_topk=8 k=1`, 8.9 vs 11.2 tok/s, mean_accept 1.60/2; full sweep
`draft_topk∈{2,4,8,22}×k∈{1,2,3}` = **0.44–0.79×**, k=1 best at every topk) — in CLAUDE.md's pre-stated
0.5–0.8× band but **the assumed cause is wrong**: the economics probes show the 512-expert draft is *not*
the dominator (`t_draft ≈ 5 ms` flat across draft_topk « `t_main` 88.9 ms ⇒ `draft_topk` is near-inert as
a *speed* lever, it only moves accept quality). The tax is the **compiled-decode asymmetry** — greedy
runs the compiled T=1 fused mamba/moe graph (88.9 ms/tok) but spec's T=k+1 verify falls to **eager**
(`t_verify` 1.54/1.94/2.33× t_main at T=2/3/4) — plus the hybrid partial-reject 2nd main forward
(un-sliceable Mamba `(ssm,conv)` re-run, ≈0.4×t_main/round); together they outweigh the 1.60-tok/round
amortization (a closed-form `(t_verify+t_draft+reject·t_main)/mean_accept` predicts the measured sweep to
~1%). Reproduces M2 exactly (full-topk k=1 first-diverges at 24/48, the bf16 ULP near-tie; the bench
reports `match` as INFO, never asserts — M2 owns the losslessness proof). **>1× at B=1 needs a compiled
T>1 verify graph; serving throughput needs the already-built batched (B>1) tree-verify** — the
MTP-M3-perf follow-ups. **MTP-M3-perf (B) ✅** — bf16-drafter quality-ceiling counterfactual
(`parity/nemotron_mtp_bf16_drafter_bench.py`, solo ~330 GiB: int4 backbone **unchanged** + the
**un-quantized bf16 source `mtp.*` head** via M0's loader — *not* a dequantized int4 head; identical M3
economics+sweep): the perfect-quality drafter lands at **0.79× greedy** (8.8 tok/s) — *tied* with int4,
**below** the 0.88–1.26× prediction band — with Δaccept(bf16−int4) ≈ **0** (+0.00 at draft_topk≥4,
bit-identical accept 1.50/1.60/1.81; only +0.10 at the degenerate topk=2). The int4 quant tax on
accept-rate is **negligible** (the int4-RTN head already drafts as well as bf16 here — M1's 12.5% top-1
logit disagreement sits on low-confidence positions that don't dominate accepted-token mass; t_draft(bf16)
5.5–6.6 ms is even *higher* than int4's ~5, still « t_main). With M3's *lighter*-drafter direction (worse
via accept) this **brackets the drafter as near-inert at B=1** — the compiled T>1 verify graph (part A) is
the sole B=1 lever left to test. **MTP-M3-perf (A) ✅** — compiled the T>1 spec-VERIFY graph (new
`NemotronResidentModel.compile_verify`, default off → eager/byte-unchanged, rule 4; the guard fires only on a
Mamba *continuation* — some `conv` populated — never on fresh/chunked-suffix prefill, so prefill is
byte-identical), gated **output-equivalent** (`parity/nemotron_ultra_compiled_verify_parity.py`, solo:
compiled T>1 verify == eager on {logits, hidden, ssm, conv, follow-on T=1} for k∈{1,2,3}, **worst Δ 0.00e+00
— bit-identical**; `mx.compile` is pure fuse on the branch-3 per-token-step graph). Bench
(`parity/nemotron_mtp_compiled_verify_bench.py`, solo ~313 GiB, eager-then-compiled in ONE process): the
compiled verify is only **1.08–1.10× faster than eager** (T=2/3/4 134→124 / 169→156 / 203→184 ms; still
1.42–2.11× t_main=87.4 vs eager's 1.54–2.32×) — the eager T>1 verify was already a single, mostly
launch-amortized forward (NOT the per-token T==1 decode loop greedy runs), so `mx.compile` kernel-fusion has
little to remove. Best B=1 spec **0.79× → 0.84× greedy** (9.0→9.6 tok/s, `draft_topk=8 k=1`, accept 1.60) — a
real lift, **still <1×**. Reproduces M2 exactly (`acc==` every config; full-topk k=1 first-diverges 24/48 =
the bf16 ULP near-tie, else 48/48; `match` INFO, M2 owns losslessness). **So plain `mx.compile` is NOT the
>1× B=1 lever** — crossing 1× needs a **fused multi-token verify kernel** (one kernel for the whole T-step
mamba+moe, deeper than auto-fusion) or the already-built **batched (B>1) tree-verify** (throughput, not
single-stream latency). **MTP-M4 ✅** — first batched **tree-verify** with the *real* baked MTP head (the
prior Super gate used a random-init head → every `W^D` path sat at the ~1/W accept floor, no fan-out to
amortize): `parity/nemotron_ultra_tree_spec.py` (solo ~313 GiB), `spec_generate_tree(batched=True)` over the
int4-RTN backbone + int4 sidecar, no runtime change (`batched=True` already existed, model-free-gated).
**PARITY (rule 4) PASS** — batched==sequential **BIT-IDENTICAL** (W=2 D=2, 32 tok; the bf16 batched-moe-reorder
≥0.99 tolerance wasn't even needed), so the `B=W^D` one-`gather_qmm`-over-all-paths verify is output-equivalent
to the naive per-path verify; losslessness reproduces M2 exactly (both tree paths 30/32 vs greedy, first-div
pos 24 = the same bf16 ULP near-tie). **The trained head fans out as designed** — tree accept **1.96/3 (W2D2)
→ 2.35/3 (W4D2)** vs the k-chain's 1.81 (the random head could never show this). **BUT the `W^D`
weight-amortization thesis FAILS at B=1**: `bat/seq` **0.89–0.99×** (batched is *not* faster than sequential —
marginally slower) and the whole tree is **0.07–0.19× greedy** — the *worst* B=1 path measured, far below the
M3 k-chain (0.79×) and (A) compiled-verify (0.84×). Why: `batch_step` amortizes only the MoE (one `gather_qmm`
over `[B,1,hidden]`) but **48/108 layers are MoE — the other 60 (Mamba-2 + attn) run per-stream in a bounded
B-loop**, so B paths cost ~B× the *dominant* SSM/attn work and cancel the MoE win (the same M3 lesson: on this
hybrid the experts are NOT the single dominator). **So tree-verify is a serving-throughput (multi-stream B>1)
lever, not a single-stream B=1 latency lever** — and even for throughput the per-stream Mamba loop caps the
amortization on a hybrid. The decisive B=1 latency lever stays a **fused multi-token verify kernel**; the B>1
win wants genuine **multi-stream decode** (independent requests, not path-replication) + Mamba-state batching.
**U4/paged-KV ✅** — the deferred rule-4 **real-Ultra** gate for the #152 paged contract. The whole contract
(`make_paged_state`/`prefill_paged`/`step_batch(paged_batched=…)` + the #153 batched-KV loop-kill) was already
built on `NemotronBatchedResidentModel`, model-free-green (`paged_engine_equiv_test.py`) AND real-green on the
**Super-120B** sibling (#174 `nemotron_paged_real_test.py`); only the real-Ultra-artifact gate was "deferred,
one model at a time" (`batched_runtime.py:613`). `parity/nemotron_ultra_paged_real_test.py` (solo ~306 GiB, NO
MTP sidecar — paged-KV is backbone-only) closes it: drives the real serving path (`_BaseBatchedSession` paged
mode) — seq A stores a 32-tok/2-block prefix's int8 KV across all **12** attention layers + the Mamba recurrent
boundary snapshot, seq B reuses them and prefills only the 5-tok suffix, then 10 greedy steps. **paged ==
discrete top-1 10/10 (bit-exact)**; the paged manager covers exactly **12** attn layers (the count
`paged_kv_spec` derives from the artifact — NOT the Super's 8, the one thing that could have been stale at Ultra
scale); prefix reused 32 tok/2 blocks; recurrent boundary restored (snapshot 1/1); engine `get_cache_stats()`
agrees. logit max-abs 0.875 is INFO-only — the chunked-Mamba-SSD prefill resumed from the boundary + the
int8-KV reblock perturb logits at the **bf16-ULP class M2/M3 documented**, but top-1 (the rule-4 arbiter for
paged==discrete) never flips over 10 steps. So the paged contract is real-green at Ultra scale; the #153
loop-kill's *throughput* win (B>1 decode tok/s, +18% on Super) folds into the next stream — where MTP-M4 already
flagged that Ultra's per-stream Mamba loop (48/108 layers) caps the attention-KV-only amortization on a hybrid.
**U4/decode-economics ✅** — combined measure-first run for the two remaining U4 streams
(`parity/nemotron_ultra_decode_economics.py`, solo ~306 GiB, exit 0). The scoping discovery reframes
MTP-M4: its "60 Mamba/attn layers run per-stream" was the **T>1 verify** path (`batched_decode_step`);
the **T=1 multi-stream decode** path (`batched_decode_step_fused`/`_native`) already batches Mamba (ONE
`[B,…]` mixer call — `ssd_step_fused` is `grid=(p,h,bn)`, the mixer runs every op over the B axis), attn
(fused SDPA), MoE (stacked). **Stream A (multi-stream B>1 decode) — MTP-M4 pessimism OVERTURNED for
decode:** real-Ultra native (form-2 persistent `BatchedMambaState`, the prod serving path) aggregate
decode **scales 10.3→28.4→40.0→47.5 tok/s @ B=1/8/16/32 (4.61× @ B=32)**, loop-kill **1.77× @ B=8** over
the per-stream loop, parity-confirmed on real weights (native==fused==loop **bit-exact**, Δ=0); peak 367
GiB @ B=32 (room under 490 → B can go higher). So the hybrid DOES amortize across B in decode — Stream A
is a **characterization win, no kernel needed**; the sublinear per-stream drop (10.3→1.48) is the
batched-SSD-step-compute ceiling (the only residual A-lever). **Stream B (fused multi-token verify
kernel) — GO:** the B=1 T>1-verify component breakdown (T∈{1..4}) shows the T-growth is **59% Mamba
per-token step loop** (`mamba_mixer.py:148`, +77.8ms over T=1→4 — launch-bound, the part `mx.compile`
can't fuse across the sequential T-loop → MTP-M3 A's 1.08–1.10× ceiling) + **40% MoE** (`gather_qmm`,
+52.3ms — more distinct experts hit as T grows, weight-bandwidth, NOT fusable) + ~0% attn. So a fused
multi-token SSD scan kernel (extend the one-token `_ssd_step_kernel` in `mamba_ssd.py:97` to loop T
internally, carrying state) targets the **majority** grower; the MoE 40% caps the achievable speedup but
the lever is real. **U4/Stream-A (decode batch-scaling) ✅** — the measure-first half of Stream-A's
residual lever (`parity/nemotron_ultra_decode_scale.py`, solo ~306 GiB, exit 0): push the native form-2
serving decode sweep past B=32 toward the 490 GiB ceiling, guarded by an adaptive per-stream-memory
projection (never launch a B that could OOM — reboot hazard). **Aggregate decode throughput PLATEAUS at
~48 tok/s from B=32 on** — B=32 48.03 / B=48 47.78 / B=64 47.93 / B=80 48.32 tok/s (all ~48 ± run-noise;
per-stream 1.50→1.00→0.75→0.60, agg 4.75–4.78× B=1), so **B≈32 is the throughput knee** (367 GiB, 123 GiB
headroom) and B>32 buys **zero** aggregate — only per-user latency + memory (flat ~1.92 GiB/stream). The
guard skipped B=96 (projected 494.7 > the 465 safe ceiling); measured to B=80 @ 459.5 GiB ⇒ **extrapolated
max B ~83 @ 465 safe / ~94 @ 490 hard** (so B>32 is an admission/concurrency policy choice, not a
correctness limit). Parity self-check green (B=1 fused==loop |Δ|=0, B=4 native==fused |Δ|=0; the B=1/8/16/32
overlap rows reproduce 0de52a9). **Confirms the economics batched-SSD-step ceiling**: the per-stream Mamba
recurrence — NOT memory, NOT MoE bandwidth (both had headroom) — caps the B-amortization, so the ONLY lever
that lifts aggregate past ~48 tok/s is the batched-SSD-step tune, the **same `mamba_ssd.py` SSD-step surface
as Stream B's fused multi-token verify kernel** (one kernel effort moves both). **U4/Stream-B (fused
multi-token SSD-scan verify) ✅** — built `ssd_scan_fused` (`mamba_ssd.py`: extends the one-token
`_ssd_step_kernel` to loop T internally, carrying the N=128 state through the `new_state` buffer —
register-carry would spill — so the whole T-token verify recurrence is **one Metal launch per layer**; at
T=1 it equals `ssd_step_fused`), wired into `MambaMixer`'s T>1 continuation behind `FUSED_SSD_SCAN`
(default off, rule 4): when on, a **bit-identical** batched conv (the per-token `causal_conv1d_step`
rolling window materialised over T + reduced by the SAME `mx.sum` over K, same final `conv_state`) feeds
one `ssd_scan_fused`; off/T=1 the per-token loop is byte-unchanged. **Gated output-equivalent**: model-free
(`parity/nemotron_ssd_scan_kernel_test.py`) — scan == per-token `ssd_step` loop rel **2.2e-7** (barely
compounding over T), T=1 bit-exact to `ssd_step_fused`, batched conv **bit-identical**; real-weight
per-block (`parity/nemotron_ultra_fused_scan_parity.py`, rule-8 streamed — all **48** mamba blocks, given
identical inputs) — fused block output == eager **≤1 bf16 ULP** (83/144 bit-identical, worst rel 2.4e-3,
the abs deltas clean powers of two), conv **bit-identical** everywhere, ssm (fp32) **≤7.6e-6**. **The
parity gate caught the bf16 cascade** (the CLAUDE.md thesis): a first full-MODEL verify gate FAILED its
intermediate-state assertion (hiddenΔ 106, convΔ 2.00) **despite perfect top-1 agreement** — per-layer
tracing showed the bf16-cast mamba output is bit-identical for most layers but the ~2.2e-7 fp32 scan
reorder occasionally straddles a bf16 boundary and flips a SINGLE ULP (first ~2⁻⁶ at the 2nd mamba layer),
cascading through clean powers of two (2⁻⁶→2⁻⁴→…→10²) across 108 layers — the **exact M2/M3 settled
finding** (a single bf16 ULP near-tie cascades chaotically; "spec == T=1 greedy" is the wrong criterion).
The cascade afflicts ANY ULP-level reorder (bf16 chaos, not a fusion bug), so the honest decisive criterion
is **per-block equivalence + top-1** (what spec consumes), NOT intermediate-state magnitude; the gate was
redesigned to per-block. **Bench** (`parity/nemotron_ultra_fused_scan_bench.py`, solo ~313 GiB; E=eager /
F=fused / FC=fused+compiled, ordered so the only compiled-verify traces are flag-True): t_main 88.8 ms;
**t_verify F 1.07–1.12× / FC 1.10–1.15× vs eager** (the fused scan removes the per-token *launch* overhead,
not compute); **best B=1 spec 0.80× (eager ≈ M3 0.79) → 0.90× (F) → 0.92× (FC, draft_topk=8 k=1, accept
1.66/2) — the best B=1 single-stream spec-decode measured (+15% over the eager ceiling, beats the 0.84×
compiled ceiling), but STILL <1×**: the fused verify is 1.43–2.01× t_main, the residual T-growth is the
**unfused MoE `gather_qmm`** (40% per the economics — weight-bandwidth, NOT launch-bound, a scan kernel
can't touch it). `acc==` shows `!!` at bf16 near-ties (F/FC are ≤1 ULP, not bit-identical — near-ties flip
across modes; `match` mostly 48/48); losslessness owned by M2 (the int4 main model verifies every draft).
**So crossing 1× at B=1 now needs the MoE verify cost reduced (bandwidth, not launch — a harder lever); the
throughput lever stays the already-characterized B>1 tree-verify (MTP-M4).** **U4/MoE gather_qmm
batch-scaling ✅** (`parity/nemotron_ultra_moe_qmm_bench.py`, one real MoE layer, rule 8) — MEASURES the
Stream-B "reduce the MoE verify bandwidth" question instead of guessing: the routed `gather_qmm` is
**already fused** (sorted dispatch, the DSV4 `_swiglu_stack_packed` / qwen35 pattern) and amortizes hard at
batch — **per-token MoE cost 1195 µs @ B=1 → 209 µs @ B=32 (5.58× cheaper/token) → 182 µs @ B=128 (6.6×)**;
**sorted dispatch is 0.93× @ B=1** (pure overhead, no overlap) **but 1.81× @ B=32** (each touched expert's
int4 weights read once for all tokens routed to it). So the MoE "bandwidth lever" is **not a missing
fusion** — it's a **B=1-vs-B=32 regime**: a single stream can't amortize (why Stream B's B=1 spec stayed
<1×), but B=32 serving gets it for free. This **reconciles Stream A**: at B=32 the MoE is cheap (amortized
5.6×) so the per-stream **Mamba** recurrence is the decode ceiling (NOT the MoE), consistent with the ~48
tok/s knee. **U4/decode-step breakdown ✅** — the measure-first localization of that ~48 tok/s ceiling
BEFORE building any SSD kernel (`parity/nemotron_ultra_decode_step_breakdown.py`, solo ~306 GiB): a real
native (form-2) **T=1 decode** step decomposed by layer-kind + a mamba sub-breakdown + an **e2e
fused-step A/B**. Stream A *inferred* the ceiling is "the per-stream Mamba recurrence"; the breakdown keeps
the direction but **corrects the mechanism**. At B=32 the step is **MoE 47% + mamba 40% + attn 12%** (real
total 642 ms ⇒ 49.8 tok/s, reproducing the knee), and **every** kind amortizes per-token (moe 0.26× /
mamba 0.21× — the dense GEMMs read their weight once for all B tokens). The lone non-amortizer is the SSD
recurrence (per-stream state), and the sub-breakdown finds the real cost: the **composed `ssd_step`
explodes to 64% of the mamba block at B=32** (4.6 ms/block vs the projections' ~2 ms) because the **eager**
batched path materializes several `[B,H,N,P]` fp32 temporaries (~268 MiB each) it can't fuse — the
already-built **`ssd_step_fused` kernel does the identical work 3.86× faster** (in-kernel state carry, no
temporaries). So the serving lever is **NOT a new kernel** — it is **graduating `FUSED_SSD_STEP`** (shelved
as a "no-win", but that was B=1-*compiled*-only, where `mx.compile` fuses the composed ops). The **e2e A/B
confirms** it on the real native serving decode, **greedy-exact** (argmax_match; the |Δlogit| 2.12 is the
bf16-ULP reorder class): composed→fused **1.04× / 1.15× / 1.26× / 1.36× @ B=1/8/16/32** — **+36% aggregate
decode throughput at B=32 (49.4 → 67.0 tok/s)**, output-equivalent (so the lever is now *measured AND
parity-proven*, not assumed). **U4/fused-step graduation ✅** — wired the confirmed lever into the prod
serving path. The batched decode steppers (`batched_decode_step_fused`/`_native`) now use the fused
one-launch SSD step via a new module flag **`BATCHED_FUSED_SSD_STEP` (mamba_mixer, default ON)**, threaded
explicitly as a `fused_step` kwarg `NemotronBlock → MambaMixer` (rule 6: no leaked global state). The
per-stream-loop reference + the tree-spec `batch_step` stay **composed** (the naive baseline), and the
**compiled single-stream** path passes no `fused_step` ⇒ **unchanged** (fused is a ~3% loss there —
`mx.compile` already fuses the composed ops; the global `FUSED_SSD_STEP` force-on stays OFF). So
`step_batch_native` (the omlx serving entry, `shim/omlx.py`) is **+36% @ B=32 (49.4 → 67.0 tok/s)** for
free, greedy-exact. **Re-gated** model-free (`nemotron_batched_attention_test.py`): the existing
fused-vs-loop / native-vs-fused bit-exact guards pin `BATCHED_FUSED_SSD_STEP=False` (apples-to-apples,
isolating the *attention* fusion); a new **B2** proves the graduated step output-equivalent (fused ==
composed `|Δlogit|` 4.8e-7, greedy-exact); a default-ON pin fails loud on revert. The real-model
`_decode_compare` helper pins composed too (its B=1 bit-exact stays valid; the graduated path's
real-weight greedy-exactness is the breakdown bench's `_greedy_match_fused`). model-free gates green
(batched-attention re-gate, native-serving, loop-equiv, tree-verify, mtp-spec). **Remaining U4 work:** the
residual ceiling is now the **MoE+mamba co-dominant weight bandwidth** (B>32 = admission policy / a
quant-bits lever, not a kernel); the B=1 >1× spec lever stays fundamentally capped (a single stream can't
amortize). Stream A's serving recommendation is settled: **B≈32 throughput-optimal, now ~67 tok/s**. The
InternLM2.5 MInference track below is **COMPLETE (M0–M10 ✅)**.

**COMPLETE: InternLM2.5 sparse-prefill (MInference family) — M0–M10 ✅.** Handover
**`PLAN_minference.md`**. Reuse the validated block-sparse substrate (`quanta.modeling.xattention`,
`gather_sparse_attention`/`sparse_prefill_mask`, `threshold=1.0`==dense); M0 wired a `self.sparse`
hook into `InternLM2Attention` (default None = dense byte-unchanged). M1 measured XAttention's lossy
lever on the int8-g64 bake (`parity/internlm2_ppl_sparse.py`, solo GPU): prefill @ threshold 0.9 costs
**+0.31% ppl** (knee t=0.80 +2.39%) — "free"; gather speed-path == mask quality-path. **M2 added
MInference's A-shape selector** (sink block 0 + `local`-block window) onto the SAME execution via a
`selector` discriminant on `XAttnConfig` (`"xattn"` default byte-unchanged; `"ashape"` new) +
`select_keep` dispatch (`xattn` path byte-for-byte preserved) + model-free gate
`parity/internlm2_ashape_test.py`: A-shape keep-all **== dense EXACTLY**, gather==mask, measured cost
**L=4 (512-tok) +0.58% / L=2 (256-tok) +3.76%** (cheaper-but-lossier than XAttention, per MInference).
**M3 added MInference's vertical-slash selector** (online last-query-block probe → ONE global pattern:
top-`vert` vertical key-blocks ∪ top-`slash` slash block-offset bands, MInference §3) onto the SAME
execution via a `"vslash"` `select_keep` branch + precomputed global `index` threaded into every gather
chunk (so gather==mask); model-free `parity/internlm2_vslash_test.py` (causal/anchor/twin) + real-model
gate: vslash keep-all **== dense EXACTLY**, gather==mask @v3s3, measured cost **v3s3 +3.01% / v2s2
+7.29%** (lossiest of the three at this 7-block doc — vertical-slash is a long-context, per-head-assigned
pattern; integration green is the point, not winning at 7 blocks). **M4 made the selector per-head**: a
`head_selectors` tuple on `XAttnConfig` (None = uniform, byte-unchanged) routing each query head to its
own kind via `_select_keep_per_head` (bounded loop over the ≤3 KINDS, not heads → one `take_along_axis`;
each head's keep == the uniform keep for its kind — pure routing); offline policy `assign_head_selectors`
(cheapest candidate within `tol`, else accurate fallback); a parity-preserving `InternLM2Attention.
_attn_heads` extraction so the ppl harness measures per-head error vs dense. Model-free
`parity/internlm2_perhead_test.py` (policy + routing-exactness + mixed-keep-all==dense + gather==mask +
validation) + real-model gate: **perhead mixed keep-all == dense EXACTLY**, gather==mask (8.88e-3),
measured **+0.40% ppl** with the offline router assigning **86% xattn / 14% A-shape / 0% vslash** (Σ
32×32 heads) — buys back A-shape-L2's +3.76% → +0.40% (≈ best uniform xattn +0.31%) while running 14% of
heads on the cheaper static kernel; vslash 0% at 7 blocks (long-context pattern, per M3). **M5 made the
selector per-head *params*** (not just kind): a frozen `HeadSpec(kind, threshold, local, vert, slash)` +
`head_specs` tuple on `XAttnConfig` (None = M4/uniform, byte-unchanged; precedence over `head_selectors`,
both-set rejected) routing each head to its own (kind, params) via `_select_keep_per_head_specs` (bounded
loop over DISTINCT specs, not heads → one `take_along_axis`; vslash params shared via the threaded global
index, fail-loud pin; ashape/xattn params freely per-head); offline policy `assign_head_specs` = the dual
of M4's (most-accurate candidate within a kernel-aware FLOP `budget`, else cheapest); a parity-preserving
`_attn_qkv` extraction shared by `_attn_heads` + the new offline `_attn_keep_counts` (per-candidate cost =
mean kept blocks). Model-free `parity/internlm2_perhead_params_test.py` (budget policy + routing-exactness
incl. same-kind-different-params + mixed-keep-all==dense + gather==mask + validation) + real-model gate:
**perhd-p mixed keep-all == dense EXACTLY**, gather==mask (3.29e-3), measured **+0.15% ppl** — **beats M4's
per-head-kind +0.40% AND best uniform xattn +0.31%** — with the FLOP-budgeted search (budget=4 blocks)
assigning **75% ashape:L4 / 23% xattn:t0.9 / 1% vslash / 1% xattn:t0.95** (Σ 32×32 heads); per-head params
let 75% of heads run the cheap static kernel while each still gets its most-accurate-affordable approx, so
the aggregate beats any uniform — the MInference thesis. **M6 made per-head *vslash params* vary** (lifted
M5's vslash-pin): `vertical_slash_index` now returns **param-independent** masses `(key_mass, slash_mass)`
and the top-`vert`/`slash` cut moved into `select_keep`, so two heads read the ONE global probe yet cut
DIFFERENT vert/slash from the shared masses (`__post_init__` pin removed; M3/M4/M5 vslash *selections*
byte-identical — same masses + same top-k, relocated). Model-free `parity/internlm2_vslash_perhead_test.py`
(two vslash heads at different vert/slash each == its uniform spec & keep different blocks; config-vert/slash
irrelevance; mixed keep-all==dense; gather==mask) + real-model gate (ppl harness search grid gains a 2nd
vslash param v2s2+v3s3): **perhd-p keep-all == dense EXACTLY**, gather==mask (7.45e-4), measured **+0.04% ppl
— beats M5's +0.15%** with the FLOP-budgeted search assigning **73% ashape:L4 / 22% xattn:t0.9 / 4%
vslash:v3s3 / 1% xattn:t0.95** (4% of heads now run the WIDER vslash, vs M5's 1% — per-head vslash params pay
off even at 7 blocks; M1–M5 reproduced bit-identically). **M7 ✅** — **key-chunked the long-context
vertical-slash probe** so it scales to 100K+ where the old single-shot probe fail-loud `raise`d (the
full `[B,H,lp,S]` attention exceeds `max_alloc_gb`): when over budget, the probe softmax is taken in
**key chunks** via an online-softmax (flash) two-pass (`_vertical_slash_index_chunked`) — pass 1
accumulates the per-probe-row running max + normalizer over chunks (peak one `[B,H,lp,Sc]` chunk), pass
2 recomputes each chunk's final probs and accumulates the M6 param-independent masses (vertical
per-key-block; slash via a bounded overlapping offset-window). Peak memory O(one key chunk), not O(S);
**rule-4 safe** — the short-doc path (`gb ≤ max_alloc_gb`) is **byte-for-byte unchanged** (M1–M6 gates
bit-identical, all 0.00e+00), only the long-context branch is new and output-equivalent to single-shot
up to fp reassociation. Model-free `parity/internlm2_vslash_chunked_test.py` (synthetic q/k/v, forced
to chunk via a tiny `max_alloc_gb`): chunked == single-shot masses **key rel ≤2.1e-7 / slash ≤1.9e-7**
across {1,2,3} blk/chunk × {block-aligned T=896, ragged T=823}; param-independence Δ **0.0**; chunked
keep-all == causal **0 cells**; chunked gather == mask **1.4e-7**. **M8 ✅** — **timed the `gather=True`
speed path** the M1–M6 harness asserted but never measured (`parity/internlm2_prefill_bench.py`, solo,
ONE resident decoder layer, dense causal flash SDPA vs gather selectors across T {1K…64K}): two hard
gates — keep-all gather == dense **rel 4.7e-3** (bf16 floor) + M7's chunked probe on real weights (T=64K,
13 chunks, masses == single-shot **rel 1.2e-7**) — then the headline **O(T²)→O(T) crossover**: **ashape L8
0.7×@1K → 1.0×@8K → 2.3×@32K → 4.3×@64K** (kept 100%→3%), **vslash v8s8 → 3.4×@64K** (kept 4%), **xattn
t0.9 only 1.2×@64K** (kept ~63% — the antidiagonal nucleus is the LEAST sparse, hence slowest, exactly why
MInference assigns the cheap static ashape/vslash per head + reserves xattn for the heads needing its
quality). Block-sparse gather prefill is **up to 4.3× per attention layer at 64K**, crossover 8–16K. **M9 ✅** —
the per-head-GROUPED gather **fold** (answering *"combine the approaches to a fold on speed?"*): the
M4–M6 per-head assignment folds quality but NOT speed, because the gather sizes its work by ONE global
`max_kept` = the densest head's, so a mix of cheap ashape (~3% kept) + dense xattn (~63%) makes every
head pay the dense budget (naive per-head gather bottlenecked ≈ uniform xattn, 1.2×@64K). The fold
(`XAttnConfig.grouped_gather`, **since graduated to default-on** for per-head configs/rule-4) partitions
heads by distinct spec and gathers each group at its OWN `max_kept` (bounded loop, rule 3) —
**output-equivalent** (model-free `internlm2_grouped_gather_test`: grouped == naive **bit-exact rel
0.00e+00**, == mask 1.3e-7), measured
**3.2×@64K vs naive 1.2× = 2.64× faster than naive** (bench mix 28 cheap ashape + 4 dense xattn). So
combining the patterns DOES fold the speed — but only with per-group gathering. **M10 ✅** — the
long-context per-head ppl gate (`parity/internlm2_ppl_sparse_long.py`, solo, full-model teacher-forcing
at 16384 tok / 128 blocks on the int8-g64 bake): the three deferred long-context claims verified e2e —
**keep-all per-head-specs == dense (Δ 1e-5)**, **M7 chunked probe == single-shot BIT-IDENTICAL in
full-model ppl (Δ 0.00, 3 key chunks)**, **M9 grouped-fold gather == mask (Δ 5e-4)**. Quality frontier:
block-sparse prefill is **NOT free at long context** on this code-heavy corpus — the per-head assignment
costs **+31.8% ppl** (94% ashape:L8 / 6% vslash:v6s6) while the adaptive xattn nucleus is near-lossless
**+2.8% but keeps ~65%** (barely sparse ⇒ priced out of the FLOP budget); a real, steep speed/quality
tradeoff (vs the 7-block doc's +0.04%), vslash's long-range share rising only 4%→6%. **The MInference
sparse-prefill track is COMPLETE (M0–M10).** **Graduation ✅ (post-M10):** `grouped_gather` flipped to
**default-on** for per-head gather configs (the fold is the default; `False` = the naive single-`max_kept`
path) — rule-4-authorized since the equivalence is bit-exact (`internlm2_grouped_gather_test` check 6:
default-no-flag == naive **rel 0.00e+00**); uniform configs + the production uniform `DEFAULT_SPARSE` are a
no-op (serving unchanged until per-head sparse prefill is wired in, then free).

Prior InternLM2.5 **EAGLE spec-decode** track is **COMPLETE** (M0–M3, `ec0f6f3`; **1.42× lossless @
k=2** via drafter quantization — memory `project_internlm2_eagle.md`), and is now **wired into the oMLX
serving shim**: `QuantaOmlxEngine._dispatch_spec_k` routes `spec_k>1` on an InternLM2.5 artifact through
the EAGLE-3 drafter (`quanta.internlm2.eagle.spec_generate`) — the only keeper whose spec is a trained
drafter, not native MTP — with the drafter auto-loaded from an **embedded `eagle/` sidecar**
(`_ensure_eagle` → `quanta.eagle.artifact.load_eagle`; `parity/internlm2_embed_eagle.py` does the one-shot
embed) and PTQ'd to the int4-g64 serving operating point. Gated model-free
(`parity/internlm2_omlx_eagle_test.py`: shim `spec==greedy`, injected-state precedence, missing-sidecar
fail-loud) + **real-weight end-to-end** (`parity/internlm2_omlx_eagle_real_test.py`, solo: `_dispatch_spec_k`
== greedy **bit-exact** on the real int8-g64 bake + embedded drafter — lossless). The earlier batched-decode /
paged-KV / expert-footprint sweep across the serving keepers (DSV4, Nemotron, InternLM2.5, Qwen3.6)
is fully landed:

- **#18** — kill the per-stream KV-update IO loop in DSV4 batched decode via a persistent
  `max_batch` **batched KV arena** (ONE scatter + ONE gather; flag `kv_arena`, default ON):
  **COMPLETE M0–M5** (`41a4d0f`/`6f33cc1`/`05d1171`/`bf7af6b`/`e08888d`/`f4935b5`; M5 real-model
  bench arena **greedy-exact** vs the per-stream loop AND **+37% decode tok/s @ B=32**).
  Handover **`PLAN.md`**.
- **#152** — block-paged KV with copy-on-write prefix sharing: **CLOSED**; `PAGED_KV_DEFAULT`
  ON; all keepers real-paged-green.
- **#153** — bring the #18 loop-kill to the PROD **paged** path (ONE block-table scatter + ONE
  gather): **COMPLETE across all keepers + Qwen3.6** — DSV4 M0–M4
  (`62609ba`/`c442c31`/`35dcd78`/`d19a254`/`cb2476b`, +13% @ B=32/48), Nemotron (+18% @ B=48),
  InternLM2.5 (**3.20× @ B=32**), Qwen3.6 option-B (1.63× @ B=32) — each graduated ON behind its
  own scoped flag. Handover **`PLAN_153.md`**.
- **qwen35 routed-expert packing** — keep int4 experts packed + `mx.gather_qmm` instead of
  dequant-to-bf16: **COMPLETE** (`a6b3b49`/`d17882e`/`f720fda`, marked complete `b62596e`;
  resident **63→20 GiB**, greedy-exact, ppl unchanged). Handover **`PLAN_qwen35_experts.md`**.

Optional, non-blocking: extend the #18 bench to B=48/64 on a free solo GPU (largely subsumed —
#153 M4 already benched DSV4 at B=48). Cadence (standing user instruction): single thread, NO
subagents, commit each milestone, then STOP for the user to compact.

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

## Model facts — Kimi-K2.6

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

Keep `~/models/Kimi-K2.6` (the int4 source / reference teacher) — **never delete
it**. Baked artifacts and their `<artifact>_offload` siblings live under
`~/models`, outside this repo.

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
uv run --with pytest pytest tests/ -q
uv run --with ruff ruff check src tests
uv run python -m compileall -q src tests
uv lock --check
git diff --check
```

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
