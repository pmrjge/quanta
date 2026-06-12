# PLAN — Nex-N2-Pro serving runtime + int4/int6 bake (quanta)

**Model.** `nex-agi/Nex-N2-Pro` — a 397B-param MoE (A17B active) agentic-reasoning model, **post-trained
on Qwen3.5-397B-A17B** (`model_type=qwen3_5_moe`, `Qwen3_5MoeForConditionalGeneration`). Apache-2.0.
Local checkpoint `~/models/Nex-N2-Pro` — **739 GiB / 122 shards, bf16**. Recommended sampling temp 0.7
/ top-p 0.95 / top-k 40; served upstream with `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`.

**The reuse win.** This is the *exact* architecture the in-tree **`quanta.qwen35`** module already
targets (its `config.py` header literally reads "Qwen3.5-397B-A17B"). The module is generic, validated
+ baked at 35B scale (the **Qwen3.6-35B-A3B** fleet keeper); **Nex is the 397B sibling — same code,
bigger checkpoint, re-gate at scale** (the Nemotron Super→Ultra pattern). So this is *validate + bake
an existing runtime on a new post-trained checkpoint*, not build-from-scratch.

**Architecture** (confirmed from `config.json` + the on-disk index): 60 layers, **3:1 hybrid**
(`full_attention_interval=4`) = **45 Gated-DeltaNet linear-attention + 15 gated-GQA full-attention**;
hidden 4096; full attn 32 heads / **2 KV** (GQA), head_dim 256, **partial mRoPE 0.25** (rotary 64),
`attn_output_gate`, `rope_theta 1e7`, mrope_section [11,11,10]; linear attn 16 key / 64 value heads,
conv kernel 4, fp32 SSM state; **MoE on all 60 layers**: 512 experts **top-10** + **1 shared** (width
1024); vocab 248320, untied lm_head; **256K native** (`max_position_embeddings 262144`). A ViT
(`model.visual.*`, 333 tensors) — **text-only serving ignores it** (the loader is language-model-only).

## Goal / requirements (user)

1. **Runtime first, then quantization.** Parity-gated at every step (project methodology).
2. **Package int6-RTN or int4-RTN**, chosen by measured quality-vs-speed/VRAM (the e2e-ppl arbiter).
3. **Extendable to 1M context automatically** (dynamic YaRN — already coded in `qwen35`). **The
   generated artifact's `config.json` MUST declare the 1M context** (first-class, not a separate
   runtime flag). ✅ done in N0.
4. **All the optimizations** we apply to every keeper: packed-int4 experts + `gather_qmm`,
   batched/paged decode (#153 loop-kill), **sparse-prefill (MInference) on the 15 full-attn layers**,
   **prefix caching** (paged COW), fused decode-step kernels, multi-stream decode.

## Phased plan

- **N0 — groundwork (model-free / header-only, no 739 GB load). ✅ COMPLETE.**
- **N1 — layer parity @ 397B (SOLO). ✅ COMPLETE.** The `qwen35` runtime vs an independent
  `transformers` `Qwen3_5Moe` reference (transformers 5.9.0 ships `qwen3_5_moe`), **layer-streamed**
  (one real layer resident, rule 8), `parity/nex_n2_pro_layer_parity.py`: **deltanet** our
  `GatedDeltaNet` prefill vs `Qwen3_5MoeGatedDeltaNet` (pure-torch `torch_chunk_gated_delta_rule`
  fallback — no FLA) **Δ 1.95e-06** + prefill==decode 1.44e-06; **attn** our `Qwen35Attention` vs
  `Qwen3_5MoeAttention` (eager + partial-mRoPE rope + doubled-`q_proj` sigmoid output gate + per-head
  `(1+w)` q/k norm) **Δ 2.10e-06** + fast==naive 7.5e-08 + prefill==decode 4.8e-07; **moe** router
  top-10 **set-exact** (softmax + `norm_topk_prob` renorm — confirmed against the oracle, NOT
  DeepSeek sigmoid/noaux_tc) w Δ 4.9e-07 + experts/sigmoid-shared vs inline-dense 1.55e-03 + chunk Δ
  **0.0**; **block** our full `Qwen35Block` vs `Qwen3_5MoeDecoderLayer` (the end-to-end gate that
  exercises the `Qwen3_5MoeRMSNorm` **`(1+w)`** input/post norms + residual wiring + mixer dispatch)
  **linear L0 Δ 1.50e-06 / full L3 Δ 1.90e-06**. All fp32 cross-impl at machine precision — the
  whole forward path is correct at 397B (no forward bug surfaced; the qwen35 code was already correct
  from the 35B keeper, so N1 is the at-scale re-gate — the Super→Ultra pattern). The `(1+w)` fold
  lives in `runtime.py:_one_plus` (layer/q/k/final norms, NOT the gated-DeltaNet norm).
- **N2 — bake + bits decision. ✅ COMPLETE → SHIP int4-g64.** RTN is data-free/cheap; both arms baked,
  the e2e-ppl arbiter decided.
  - **int4-g64** → `~/models/Nex-N2-Pro-quanta_int4g64` (`parity/run_bake_nex_n2_pro_int4g64.py`,
    2.7 min): **214.1 GiB / 25 shards**, 60 layers / 512 experts, counts {int8 465, expert_int4 120,
    dense 453} (== N0 quant-policy projection exactly), MTP excluded (`include_mtp=False`).
  - **int6-g64** → `~/models/Nex-N2-Pro-quanta_int6g64` (`parity/run_bake_nex_n2_pro_int6g64.py`,
    3.2 min, `expert_bits=6`): **304.1 GiB / 31 shards**, SAME counts {int8 465, expert_int4 120
    (now int6), dense 453} (== the N0 int6 projection 304.1 GiB exactly). The bake gained an
    `expert_bits` knob (`bake.py`, default 4; threaded through `_bake_moe_block`/`_bake_mtp`/
    `_write_expert_stack`) — the int4 path is byte-identical, int6 is the same recipe at a wider grid
    (MLX affine {2,3,4,6,8}; `Qwen35Artifact`/`gather_qmm` decode at the manifest width, never a
    hardcoded 4).
  - **Both artifacts are self-contained (the user's rule, now ENFORCED as code).** `bake_qwen35` ends
    with `_audit_self_contained` (rule 6, fail-loud): no symlinks, required sidecars present
    (config/manifest/index/synthesized `generation_config.json` eos `[248046,248044]`/tokenizer), no
    path leak in any json metadata, relative weight_map, all shards present. Both **config declare the
    1M window** (`max_position_embeddings 1,010,000` + standard HF YaRN, dynamic-YaRN baseline 262144).
    Family-consistent names `_int4g64`/`_int6g64` (the Qwen3.6-35B keeper convention; `rtn` was
    Nemotron-only to disambiguate its AWQ artifact — qwen35 has no AWQ path).
  - **ppl arbiter** (`parity/nex_n2_pro_ppl.py`, SOLO; 3 sequential streamed forwards over the SAME
    645-tok held-out prose via the proven `_load_block(packed=False)` reference path — bf16 source /
    int4 dequant / int6 dequant, one block resident at a time, rule 8): **bf16 ppl 5.0386 / acc
    0.5590** (low-single-digit on real prose — the forward is e2e-coherent at 397B, the project
    thesis), **int4 5.0729 / acc 0.5559 / Δ +0.68% / agree 0.9472**, **int6 5.0237 / acc 0.5590 / Δ
    −0.30% / agree 0.9550**. **int4-RTN is ~lossless** (+0.68% ppl, −0.3% acc) — the Nemotron-Ultra
    finding (int4-RTN +0.3% on a bf16 source) reproduces; int6 (−0.30%, within noise) recovers <1pp
    for +90 GiB. teacher-forced ppl is THE arbiter (methodology #4); top-1 agreement ~0.95 is the
    *secondary* signal (noisy on prose — bf16-ULP near-tie flips at low-confidence positions, a settled
    finding), a >0.90 sanity floor not a tight gate. **Decision: SHIP int4-g64** (214 GiB, ~lossless,
    90 GiB lighter than int6).
- **N3 — serving + optimizations.**
  - **N3-1 ✅ — resident + batched serving re-gate @ 397B** (`parity/nex_n2_pro_batched_real.py`,
    SOLO; ONE 214.7 GiB load shared across 3 gates — the Super→Ultra re-gate of the already-built,
    35B-graduated `Qwen35BatchedResidentModel`):
    1. **resident e2e ppl == dequant-ref.** The served kernels — packed-int4 routed experts
       (`mx.gather_qmm`) + int8 mixer projections (`mx.quantized_matmul`) — teacher-forced on the SAME
       645-tok prose give ppl **5.0715 / acc 0.5621**, vs the streamed dequant int4 reference
       (`Qwen35Artifact` + `streamed_logits(packed=False)`, computed in-process first then freed)
       **5.0729 / 0.5559** — Δ **−0.03%** (packed is marginally *better*: it fuses the dequant at full
       precision, the bf16-dequant reference pre-rounds each weight). The resident serving forward is
       numerically faithful at 397B (and reproduces the N2 arbiter's 5.0729 exactly — deterministic).
    2. **batched #153 loop-kill greedy-exact.** `Qwen35BatchedResidentModel.step_batch` is
       loop==loopkill **bit/greedy-exact at every B∈{1,4,8,16,32}**, incl. the **chunked regime** —
       B=16/32 exceed the M0 `chunk=8`, so `_gdn_step_batched` / `Qwen35Attention.decode_step_batched`
       split the batched mixer into ≤8-row blocks that keep every `mx.quantized_matmul` in the
       batch-M bit-exact gemv regime (the option-B requirement, re-proven at the true 397B dims).
    3. **Design-A equivalence.** B=1 batched (prefill + `step_batch`) == single-stream
       `Qwen35ResidentModel` greedy-exact (24-tok autoregressive trace).
    Throughput (the serving fleet-baseline row): **B=1 14.0 → B=32 55.7 agg tok/s = 3.98× batching**;
    the hybrid loop-kill **1.15 / 1.41 / 1.50 / 1.48×** @ B=4/8/16/32 (best 1.50× @ B=16). Resident
    **215 → 261 GiB** @ B=1→32 (per-stream ~1.5 GiB; **229 GiB headroom** under the 490.4 ceiling ⇒ B
    can go far higher — B>32 is an admission-policy choice, not a memory limit). This also
    **re-confirms packed-int4 `gather_qmm` experts at scale** (graduated ON, greedy-exact). Qwen3.5
    serving is UNPAGED, so `step_batch` IS the prod decode hot path the gate times directly.
  - **N3-2 ✅ — `qwen3_coder` tool parser + `qwen3` reasoning parser** (the agentic serving surface;
    `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, per the upstream serving recipe in the
    header). The chat template (artifact `chat_template.jinja`) renders tool calls as the **nested-XML
    "pythonic" form** — `<tool_call>\n<function=NAME>\n<parameter=KEY>\nvalue\n</parameter>\n…\n
    </function>\n</tool_call>` (values may span multiple lines) — and pre-opens a **bare `<think>`** so
    the model's output is `{reasoning}\n</think>\n\n{answer}`. The decisive finding: **this tool markup
    is byte-identical to Nemotron-3's**, and the reasoning is the same pre-opened `<think>` — both
    already handled by oMLX's stock `_parse_xml_tool_calls` + `extract_thinking`, the path the quanta
    patch DELEGATES (gated for this exact form by `parity/nemotron_omlx_contract_test`). So Nex's
    tool+reasoning serving already *functions* via that proven delegation. Per rule 6 (don't silently
    depend on oMLX's regex) we still ship the strict quanta-owned parsers:
    `quanta.shim.tool_parsers.Qwen3CoderToolParser` (new — nested-XML, typed-value recovery via JSON,
    multi-line values, multiple calls, `<tool_response>` formatter) + the existing `Qwen3ReasoningParser`
    (its bare-opener case IS the pre-opened `<think>`). **`Qwen3CoderToolParser` is the ONE quanta parser
    deliberately kept OUT of the global `parse_quanta_tool_calls` dispatcher** — registering it would
    silently re-route Nemotron's delegated markup (the two formats are indistinguishable by text), so
    serving keeps delegating the shared XML form to oMLX and the class is the per-model option. Gated
    model-free: new `parity/qwen3_coder_tool_parser_test.py` (**24 checks** — extract/typed/multiline/
    multi-call + strictness vs Hermes/GLM/MiniMax/prose + the **dispatcher-exclusion that preserves the
    Nemotron delegation contract** + the pre-opened/explicit/truncated/none reasoning splits + the
    reasoning⊕tool compose), plus a `Qwen3CoderToolParser` conformance+exclusion block added to
    `parity/qwen35_omlx_engine_test.py`. Additive only (no existing parser or the dispatcher touched);
    full model-free sweep **100/100**, `tool_parsers_test` (Nemotron delegation) still green. Manifest
    +1 model_free (100/51).
  - **N3-3 ✅ — long-context chunked-prefill substrate** (the 1M-window feasibility lever; before
    this, NO feasible long prefill existed: the serving `prefill` is one-token-at-a-time (O(T) full
    forwards, measured **20.4 tok/s** ⇒ 32K ≈ 27 min, 1M ≈ 14 h) and the single-shot prefill path
    holds the whole `[1,T,hidden]` window with no decode cache; the Gated-DeltaNet within-chunk scan
    is a sequential per-token loop (O(T) tiny kernel launches per layer), and the mixer could not
    continue a prefill across chunks (the conv window restarted from zero-padding). Four pieces, all
    additive (default paths byte-unchanged):
    1. **`gdn_chunked_wy`** (`gated_deltanet.py`) — chunk-parallel WY/UT gated-delta-rule prefill,
       a 1:1 MLX port of the HF/fla `torch_chunk_gated_delta_rule` (the N1-gated reference): the
       within-chunk delta rule folds into batched matmuls over ALL chunks at once via the UT
       transform `T=(I−strictly_lower(diag(β)KKᵀ⊙Γ))⁻¹` (forward substitution over the ≤64 chunk
       rows, run ONCE for the whole call), then a bounded cross-chunk state-carry loop (~6 matmuls
       per 64-token chunk). Takes the **log** decay `dt·a` (never rounds through `exp→0→log`;
       extreme-decay stress gated, g underflowing to exactly 0). == `gdn_recurrence` at fp32 rel
       ~1e-6–1e-5.
    2. **Prefill continuation** — `causal_conv1d(state=...)` (the prior K-1 pre-activation rows
       replace the zero left-pad; **bit-exact** to the full-sequence conv split anywhere) +
       `GatedDeltaNet.__call__` now treats `conv_state` given with `T>1` as a *prefill
       continuation* (previously an invalid input that silently took token 0 only); also fixes the
       latent `t<K-1` fresh-prefill conv-window shape edge. `wy` threaded explicitly
       `Qwen35Block(gdn_wy=...)` → mixer (no leaked global state, rule 6; default False ⇒ every
       existing call byte-identical).
    3. **`chunked_prefill` driver** (`runtime.py`, + `Qwen35ResidentModel.prefill_chunked` /
       `Qwen35BatchedResidentModel.prefill_chunked`) — consumes a prompt into a `Qwen35Cache` one
       bounded chunk at a time (default 4096): GDN layers carry `(conv, recurrent)` via
       `_GDNLayerState.commit_block(n)` (new — offset advances by the block), full-attn layers
       extend their int8 KV (`mx.fast.scaled_dot_product_attention` `mask="causal"` verified
       bottom-right-aligned, Δ 0.0). Dynamic YaRN resolved ONCE per request via
       `caches.yarn_seq(start+T)` — past native it **requires `pin_yarn`** (rule 6). Ragged chunks
       pad internally with provable no-op steps; continuation onto a non-empty cache = multi-turn
       prefix extension.
    4. **Gates.** Model-free `parity/qwen35_prefill_chunked_test.py` (**28 checks**: WY==recurrence
       ==sequential-chunked incl. ragged/state-carry/extreme-decay; conv continuation BIT-exact;
       mixer two-chunk == single prefill (seq bit-exact at aligned cuts, ULP at a 1-token tail);
       driver chunked == single-shot == per-token (`chunk_tokens=1` IS the per-token serving
       semantics) for seq+WY × aligned+ragged, greedy continuation token-exact; two-call
       continuation; int8-KV chunked == per-token; YaRN unpinned-past-native raises + pinned ==
       single-shot; validation fail-loud). Real `parity/nex_n2_pro_prefill_chunked_real.py` (SOLO,
       214.7 GiB): chunked **greedy-exact vs the per-token serving prefill** (WY arm AND sequential
       arm, 24-tok traces; |Δlogit| 1.6–2.0 = the documented batch-M `quantized_matmul` ULP class,
       greedy-stable) and a **needle at 50% depth of a 32K haystack retrieved verbatim** (`739214`
       + clean `<|endoftext|>` stop). Throughput: **WY chunked prefill 193 tok/s @ 1K / 157.8 tok/s
       @ 32K** (peak 242.4 GiB) vs per-token 20.4 tok/s — **9.5× @ 1K**, and the WY arm is ~2× the
       sequential chunked arm (104 tok/s). 1M prefill extrapolates to **~2 h** (was ~14 h) —
       N3-4 is now feasible. Sweep manifest 100/52 (the new model-free gate added;
       `nemotron_bake_test` reclassified real-weight via the explicit sentinel — it streams the
       bf16 SOURCE checkpoint through an import the static detector can't see, and the sources are
       now deleted from `~/models`, artifacts only).
  - **Remaining N3.** **N3-4 = the 1M long-doc / needle gate** (the YaRN arbiter — needle past the
    262144 native window under the pinned dynamic-YaRN factor, `prefill_chunked` + `pin_yarn`;
    ~2 h at the measured 158 tok/s, attention-quadratic tail will slow late chunks — consider
    landing MInference sparse-prefill on the 15 full-attn layers first or measuring at 300–400K);
    **paged-KV + prefix caching** (only the 15 full-attn layers hold KV — the 45 linear layers are
    O(1) recurrent state, so 1M KV is ~4× cheaper than a dense model; int8 KV @ 1M ≈ 16 GiB);
    **MInference sparse-prefill** on the full-attn layers (InternLM2.5 M0–M10 substrate transfers);
    **fused/batched Gated-DeltaNet decode step** (the Nemotron `BATCHED_FUSED_SSD_STEP` win, +36% @
    B=32, applied to `gdn_step`); push multi-stream batched decode past B=32. **Native-MTP
    spec-decode is N/A for Nex** (no MTP weights) — an EAGLE-style external drafter is the only B=1
    latency path if wanted later.

## N0 — what landed (this commit)

Three model-specific fixes (all in `quanta.qwen35`, additive — the 35B keeper path is unchanged) +
two gates. The N0 fit-test **caught two real divergences** between Nex and the 35B-verified contract:

1. **EOS stop-set (rule 6).** Nex ships **no `generation_config.json`**, and `config.json` lists only
   `eos_token_id: 248044` (`<|endoftext|>`, a doc separator — never ends a chat turn). `from_pretrained`
   now derives the real ChatML stop set **{248046 `<|im_end|>`, 248044 `<|endoftext|>`}** from the
   tokenizer's `added_tokens` when no `generation_config` is present (canonical fallback if even that
   is unresolvable; the 35B path with a `generation_config` is byte-unchanged). The **bake synthesizes**
   a correct `generation_config.json` into the artifact when the source lacks one (was: hard-refuse).
2. **1M in the artifact config (the explicit requirement).** `_bake_long_context` now writes **standard
   HF YaRN** (`rope_type=yarn` / `factor=4` / `original_max_position_embeddings=262144`) on
   `rope_parameters` + a mirrored `rope_scaling`, **raises `max_position_embeddings` to 1,010,000**, and
   keeps the `quanta_long_context` block. `from_pretrained` reads `yarn_original_max` from
   `rope.original_max_position_embeddings` — **DECOUPLED** from `max_position_embeddings` — so the served
   window declares 1M while the **dynamic-YaRN baseline stays 262144** (`eff@8k=1.0`, `eff@1M=3.85`).
   The artifact is a first-class 1M model for *any* loader, mRoPE preserved alongside YaRN.
3. **MTP absent.** Nex **declares `mtp_num_hidden_layers=1` but ships ZERO `mtp.*` weights** (post-train
   dropped the head). `from_pretrained` refines `num_mtp_modules → 0` by index presence (rule 6: trust
   the weights). ⇒ **native-MTP spec-decode is unavailable for Nex**; the bake runs `include_mtp=False`.

**New:** `quanta/qwen35/quant_policy.py` — key→scheme map (`dense`/`int8`/`expert_int4`) built from the
**bake's actual suffix partition + the loader's enumeration** (single source of truth, can't drift) +
an analytic resident projection. **Coverage (rule 6): 1038 text tensors → 453 dense / 465 int8 / 120
expert_int4** (+333 vision excluded); the expected keymap EXACTLY equals Nex's on-disk index (loader
contract holds at 397B).

**Fit (real header shapes, 490.4 GiB ceiling):** **int4-g64 = 214.1 GiB** (experts 202.5 + int8 7.6 +
dense 4.0; 276 GiB headroom) / **int6-g64 = 304.1 GiB** (186 GiB headroom) — **both fit**. Source text
bf16 738.3 GiB (739.1 on-disk, header-accounted within 1%).

**Gates.** `parity/nex_n2_pro_fit_test.py` (real-path, SOLO/excluded from the sweep; reads index +
safetensors headers only, no tensor materialized; PARITY-CHECKS 11). `parity/qwen35_config_eos_yarn_test.py`
(model-free, in the sweep; 21 checks over synthetic temp-dir checkpoints — eos derivation, the 1M bake
round-trip, MTP refine). Manifest regenerated (99 model_free / 51 real_weight).

**Cadence:** single thread, NO subagents, commit each milestone, then STOP for the user to compact.
