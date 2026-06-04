# PLAN_minference.md — active task handover (InternLM2.5 sparse-prefill, MInference family)

> Durable, repo-tracked handover for the in-flight task. **Read `CLAUDE.md` first**
> (permanent rules + model facts), then this. This is the authoritative durable copy.
> Companion tracks: `PLAN.md` (#18 KV arena, DONE), `PLAN_153.md` (paged batched, DONE),
> `PLAN_qwen35_experts.md` (DONE). The prior InternLM2.5 EAGLE spec-decode track is **DONE**
> (`ec0f6f3`; see memory `project_internlm2_eagle.md`).

---

## Governing cadence (standing user instruction — DO NOT VIOLATE)

For every milestone:
1. **Single linear thread.** NO subagents, NO `Agent`/`Task`/`Workflow` tools. Implement directly.
2. **Implement → parity/quality gate green → commit → STOP.** After a milestone's gate is green,
   commit it (named files only), then **STOP and wait for the user to compact** before the next.
   Do not roll milestones together.
3. Committing per-milestone IS authorized by this cadence. Otherwise the normal rule holds: do
   **not** commit unless asked. **Never push unless asked. Never skip hooks. Never `git add -A`.**
4. Commit trailer (project CLAUDE.md is authoritative — **4.7**, not 4.8):
   `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
5. **One model resident at a time** (OOM-reboot hazard on the M3 Ultra). Real-model gates run solo.

---

## Why this track

InternLM2.5-7B-Chat-1M is the only serving keeper still paying **full O(T²) dense prefill** (DSV4 is
native-sparse; Nemotron-Mamba / Qwen3.6-GDN are linear). Training-free **sparse prefill** is the
asymptotic lever (MInference ~10×@1M, XAttention ~13.5×@256K). It is **lossy ⇒ quality-gated
(teacher-forced ppl / top-1), NOT numeric-parity-gated** (CLAUDE.md rules 4 & 6). This is the 2nd
InternLM2.5 speed lever; the 1st (EAGLE spec-decode) is DONE.

## KEY DECISION — reuse the substrate, do NOT build from scratch (user chose this)

MInference and the already-built **XAttention** are the *same family*; they differ only in the
**selection** method (MInference = per-head A-shape / vertical-slash / block-sparse patterns;
XAttention = antidiagonal block scoring + nucleus). The hard part — the bounded-memory, fail-loud,
**chunked block-gather execution** — already exists, validated and ppl-gated, in
**`src/quanta/modeling/xattention.py`**:
- `gather_sparse_attention(q, k, v, scale, cfg)` — the speed path (chunked, `max_alloc_gb`-bounded).
- `sparse_prefill_mask(q, k, scale, cfg)` — the additive-mask quality path.
- `XAttnConfig(block, stride, threshold, min_seq, gather, budget, max_alloc_gb)`. At `threshold=1.0`
  it keeps every causal block ⇒ **bit-equivalent to dense** (the parity anchor).
- Model-free gate: `parity/xattention_parity.py` (green). The MLA keepers wire it via a one-line
  `self.sparse: XAttnConfig | None` hook in `src/quanta/modeling/attention.py`.

So: reuse the execution substrate; layer MInference's *selectors* on top later. Smallest diff,
parity-anchored, no rebuild.

---

## Milestones (one per commit, then STOP to compact)

### M0 ✅ `871258f` — wire the substrate into InternLM2 (DONE)
- `src/quanta/internlm2/attention.py`: added `self.sparse: XAttnConfig | None = None` to
  `InternLM2Attention` (**default None ⇒ dense, byte-for-byte unchanged**). From-scratch prefill
  (`t == kv_len and t >= sparse.min_seq`) routes through `gather_sparse_attention` (gather path) or
  `sparse_prefill_mask` (mask path) over the **GQA-repeated `kr`/`vr`** — per-head scoring sees
  exactly what dense SDPA does, so `threshold=1.0` reproduces dense. Decode (`t==1`) and
  cache-continuation always stay dense. Mirrors `quanta.modeling.attention`'s hook.
- `parity/internlm2_xattn_test.py` (NEW, model-free, fp32, T=500, the real `InternLM2Attention`):
  keep-all mask path **== dense EXACTLY (0.00)**, gather path 5e-8, threshold=0.5 drops blocks
  (5e-2), `min_seq>T` gates off (0.00), perturb-final-token leaves block 0 unchanged (0.00 — no
  future leakage). Run: `uv run python -m parity.internlm2_xattn_test`.
- Gates: dense-path `internlm2_batched_attention_test` still greedy-exact `|Δlogit|=0.00`;
  `xattention_parity`, `pytest tests/`, ruff, compileall, `uv lock --check`, `git diff --check` clean.

### M1 ✅ — real-model long-doc ppl sweep (quality cost of the reused substrate) — DONE
Goal: measure how much ppl XAttention sparse prefill trades on InternLM2 across a threshold sweep —
the lossy lever's quality cost on the real bake. Solo GPU, one model resident.
- **GOTCHA:** `parity/ppl_long.py` is **Kimi-bound** (`KimiTextConfig` / `SourceCheckpoint` /
  `KimiTokenizer` / `layer.self_attn.sparse`). InternLM2 layers use `layer.attention.sparse`. Do
  **NOT** generalize the Kimi harness in place — write a sibling **`parity/internlm2_ppl_sparse.py`**.
- Shape it after `ppl_long.py`: load the resident **int8-g64 7B bake** via
  `InternLM2ResidentModel` (or stream layers like `ppl_long`), set `layer.attention.sparse = sp` per
  layer for each variant (`dense=None`, then `XAttnConfig(block=128, stride=16, threshold∈{0.95,0.9,0.8},
  min_seq=0)` mask path for quality, plus a `gather=True` twin), teacher-force real prose ≥ a few
  blocks (reuse `ppl_long.LONG_TEXT` or the repo PROSE fixture; tokenize in **InternLM2's own**
  SentencePiece via `Qwen35Tokenizer`-analogue / the InternLM2 tokenizer), report ppl / top-1 / Δppl%.
- NOTE: with fast SDPA + an additive block mask, MLX still computes the full QKᵀ ⇒ the **mask path
  measures quality only**; the **gather path is the actual prefill speedup**. Bench wall-clock on the
  gather path separately if a speed number is wanted (subtract nothing — prefill is the whole point).
- Gate = a sensible Δppl bar (e.g. ≤ ~1–2% at threshold 0.9 is "free"); commit `internlm2_ppl_sparse.py`
  + record the numbers. This is the quality baseline every later selector is judged against.

**RESULTS (32 layers, 823 tok / 7 blocks):** dense ppl **12.338** (top-1 44.2%) → mask-path Δppl
t=0.95 **+0.24%**, t=0.90 **+0.31%**, t=0.80 **+2.39%** — threshold 0.9 is **"free"** (≤2% bar),
knee ~t=0.80. **M1 invariant green:** the `gather=True` speed-path (12.370) == its mask quality-path
(12.376) to Δppl = 5.9e-3 (< 1%) — same blocks selected, only execution differs. Delivered
`parity/internlm2_ppl_sparse.py`: streams the int8-g64 bake ONE `_DecoderLayer` resident at a time
(rule-8) via `InternLM2Artifact` (dequant→bf16 — the packed `mx.quantized_matmul` runtime has no
`.sparse` hook; int8 weight quant is orthogonal to & ~lossless against the sparse approximation it
measures), pushing every variant's hidden state through each layer; original prose, `min_seq=0` forces
sparsity on, `budget=64` default never binds < 8192 tok. Run:
`uv run --with sentencepiece python -m parity.internlm2_ppl_sparse`.

### M2 ✅ — A-shape selector (MInference) — DONE
Added MInference's **A-shape** selection (attention sink block 0 + a `local`-block causal window — the
StreamingLLM pattern) as an *alternative selector* feeding the SAME validated block-gather /
additive-mask execution. Smallest-diff design: a `selector` discriminant on `XAttnConfig` (`"xattn"`
default ⇒ **byte-for-byte the pre-selector path**; `"ashape"` new) + a `local` window field,
dispatched by a new `select_keep(q,k,scale,cfg,q_offset) -> (keep, rank)` that **both**
`sparse_prefill_mask` and `gather_sparse_attention` now call. A-shape selects positionally (no
scoring): `ashape_keep` = `{0} ∪ {i-local+1..i}`; `rank` = block recency so a binding budget keeps
the nearest-to-diagonal local blocks. The MLA keepers' hook (`quanta.modeling.attention`) and the
InternLM2 hook get the selector for free — unchanged signatures, default selector preserves behavior.
- `src/quanta/modeling/xattention.py`: `XAttnConfig.{selector,local}` (+ validation), `ashape_keep`,
  `select_keep`; rerouted both execution paths. `xattn` path verified byte-for-byte unchanged.
- `parity/internlm2_ashape_test.py` (NEW, model-free, fp32, T=500, real `InternLM2Attention`):
  `ashape_keep` closed-form (sink+window, q_offset-shifted) exact; keep-all (local≥n_blocks) mask
  **== dense EXACTLY (0.00)** + gather 5e-8; A-shape **gather==mask** at L=1 (4e-8); tight window
  drops blocks (8e-2); budget cap engages; `min_seq>T` gates off (0.00); perturb-last-token leaves
  block 0 unchanged (0.00 — no future leak). Run: `uv run python -m parity.internlm2_ashape_test`.
- `parity/internlm2_ppl_sparse.py` (EXTENDED): A-shape variants (keep-all anchor, L=4, L=2, L=4
  gather twin) + gates alongside the M1 xattn gates (re-validated unchanged in the same solo run).
- Gates: M0 `internlm2_xattn_test` + `xattention_parity` still green (xattn unchanged);
  `internlm2_ashape_test` green; pytest/ruff/compileall/`uv lock --check`/`git diff --check` clean.

**RESULTS (32 layers, 823 tok / 7 blocks; same bake & doc as M1):** dense ppl **12.3379** (top-1
44.2%). **A-shape keep-all == dense EXACTLY (Δppl 0.00)** — real-model parity anchor. **A-shape
gather==mask @ L=4: Δppl 2.2e-3 (<1%)** — M2 invariant. Measured static-window cost: **L=4 (512-tok
window) +0.58%**, **L=2 (256-tok window) +3.76%** — A-shape is cheaper-but-lossier than XAttention
(t=0.9 +0.31%), exactly as MInference characterizes it (assigned per-head, not used at every head).
M1 xattn reproduced bit-identically in the same run (t=0.90 +0.31%, gather==mask 5.9e-3), confirming
the `select_keep` refactor did not regress the validated path.

### M3 ✅ — vertical-slash selector (MInference §3) — DONE
Added MInference's **vertical-slash** selector onto the SAME validated block-gather / additive-mask
execution. Unlike xattn/ashape (which select *locally* per query block), vertical-slash builds ONE
**global** pattern from an online probe of the LAST query block's attention to all keys, then applies
it to every query block. The probe (`vertical_slash_index`) runs the last real query block's actual
attention (`q_last @ kᵀ` softmax — a plain matmul, since the attention *weights* are needed, not the
output; guarded by `max_alloc_gb`), then decomposes it two ways:
- **vertical** — per-key-*column* mass summed over the probe queries, pooled to key blocks → top-`vert`
  key blocks kept as vertical stripes (columns every query attends); sink block 0 excluded from the
  top-k (force-kept anyway).
- **slash** — per-token-*offset* mass (query-pos − key-pos), summed via one gather+sum over the probe,
  pooled to block-offsets (`δ // block`) → top-`slash` block-offsets kept as diagonal bands; offset 0
  (the diagonal) excluded (force-kept).

The keep is then positional given (vert, slash): `keep[i,j] = causal & (j∈vert | (i−j)∈slash | j==0 |
j==i)`. Because the pattern is global, the caller computes the index **once** over the whole sequence
and threads it into every chunk of the gather path — so gather and mask select identically (the twin).
Smallest-diff design: a `"vslash"` branch on `select_keep(…, index=None)` + a `vert`/`slash` field on
`XAttnConfig`; `sparse_prefill_mask`/`gather_sparse_attention` precompute the index for vslash only.
xattn/ashape paths byte-for-byte unchanged.
- `src/quanta/modeling/xattention.py`: `XAttnConfig.{vert,slash}` (+ validation), `vertical_slash_index`,
  `select_keep` `"vslash"` branch + `index` param; both execution paths precompute & thread the index.
- `parity/internlm2_vslash_test.py` (NEW, model-free, fp32, T=500, real `InternLM2Attention`):
  `vertical_slash_index`+`select_keep` strictly causal (no future block) / diagonal+sink always kept /
  keep-all == full causal mask; keep-all (vert,slash ≥ n_blocks) mask **== dense EXACTLY (0.00)** +
  gather 5e-8; **gather==mask** at v=s=1 (5e-8); tight pattern drops blocks (3.7e-2); budget cap
  engages; `min_seq>T` gates off (0.00); perturb-last-token leaves block 0 unchanged (0.00 — block 0's
  selectable set is force-pinned to {0} by causality, so it is probe-independent). Run:
  `uv run python -m parity.internlm2_vslash_test`.
- `parity/internlm2_ppl_sparse.py` (EXTENDED): vslash variants (keep-all anchor, v3s3, v2s2, v3s3
  gather twin) + M3 gates alongside the M1/M2 gates (re-validated unchanged in the same solo run).
- Gates: M0 `internlm2_xattn_test` + M2 `internlm2_ashape_test` + `xattention_parity` still green
  (xattn/ashape unchanged); `internlm2_vslash_test` green; pytest/ruff/compileall/`uv lock
  --check`/`git diff --check` clean.

**RESULTS (32 layers, 823 tok / 7 blocks; same bake & doc as M1/M2):** dense ppl **12.3379** (top-1
44.2%). **vslash keep-all == dense EXACTLY (Δppl 0.00)** — real-model parity anchor. **vslash
gather==mask @ v3s3: 0.217% rel (Δppl 2.76e-2 < 1%)** — M3 invariant (kernel fp drift only; same global
index, budget never binds at 7 blocks). Measured cost: **v3s3 +3.01%**, **v2s2 +7.29%** (monotone in
kept blocks). At this **short** 7-block doc vertical-slash is the lossiest of the three (cf. xattn
t=0.9 +0.31%, ashape L=4 +0.58%) — expected: it is a *long-context, per-head-assigned* pattern (its
global vertical tokens + slash bands pay off at 100K+, not on a doc with little long-range structure);
the gate's job is correct integration (anchor + twin green) + an honest cost measurement, not to win at
7 blocks. M1/M2 reproduced bit-identically in the same run (xattn t=0.9 +0.31% / gather==mask 5.9e-3;
ashape L=4 +0.58% / gather==mask 2.2e-3), confirming the `index` refactor did not regress them.

### M4 ✅ — per-head offline pattern assignment — DONE
Made the selector **per head**: an offline search routes each query head to the cheapest selector kind
that still recalls its attention, and `select_keep` dispatches per head over the validated M1/M2/M3
selectors. The per-head path is a pure **routing layer** — it adds no new selection math: head `h`'s
kept-block mask is byte-identical to the uniform mask for `head_selectors[h]`. Smallest-diff design:
- a `head_selectors: tuple[str,…] | None` field on `XAttnConfig` (None ⇒ uniform `selector`, every path
  byte-for-byte unchanged; when set, a length-`num_query_heads` tuple of kinds — heads sharing a kind
  share that kind's params). `_select_keep_per_head` computes each *distinct* kind's keep/rank for ALL
  heads (a bounded loop over the ≤3 KINDS present — never a per-head/per-token hot loop, rule 3), stacks
  `[n_kind,B,H,Tq,Tk]`, and selects per head with one `take_along_axis`. vslash's global `index` is
  threaded to its sub-selection so per-head gather == mask. `_uses_vslash` fires the index precompute
  when ANY head uses vslash. xattn/ashape/vslash **uniform** paths byte-for-byte unchanged.
- `assign_head_selectors(errors[C,H], cand_kinds, tol)` — the offline policy: route each head to the
  cheapest candidate (rows ordered cheap→accurate by kernel cost) whose per-head error ≤ `tol`, else the
  most-accurate fallback. Pure/positional ⇒ unit-testable.
- `InternLM2Attention._attn_heads` — a **parity-preserving extraction** of `__call__`'s body up to (not
  including) the `wo` projection (`__call__` == `wo(transpose(_attn_heads(x)).reshape(…))`), so the ppl
  harness can read each head's attention output under a given selector and compare it to dense without
  duplicating projection / RoPE / dynamic-NTK / GQA-repeat / sparse-dispatch.

Deliverables:
- `src/quanta/modeling/xattention.py`: `XAttnConfig.head_selectors` (+ validation), `_uses_vslash`,
  `_select_keep_per_head`, `select_keep` per-head dispatch, `assign_head_selectors`; both execution
  paths' index precompute via `_uses_vslash`.
- `src/quanta/internlm2/attention.py`: `_attn_heads` extraction; slimmed `__call__` (pure, no behavior
  change — verified bit-identical by the M0/M2/M3 model-free gates that drive the public `__call__`).
- `parity/internlm2_perhead_test.py` (NEW, model-free, fp32, T=500, real `InternLM2Attention`):
  `assign_head_selectors` policy (cheapest-within-tol / fallback / boundary) + routing exactness
  (mixed `keep[:,h]` == uniform keep for head `h`'s kind) + uniform-as-per-head == uniform (0.00) +
  MIXED keep-all == dense, mask **(0.00)** & gather (5e-8) + gather==mask (4e-8) + sparsity-active +
  budget-capped + min_seq-gated + causal-safe + validation (unknown kind / length mismatch rejected).
  Run: `uv run python -m parity.internlm2_perhead_test`.
- `parity/internlm2_ppl_sparse.py` (EXTENDED): per-layer offline assignment derived on the DENSE
  stream's input (per-head L2 error of each candidate vs dense → cheapest within `tol=0.02`; candidates
  ashape L=2 / vslash v2s2 / xattn t=0.9, cheap→accurate by kernel cost, fallback xattn) + `perhead` /
  `perhead gat` (derived) + `perhead anchor` (fixed mixed keep-all) variants + M4 gates + a realized
  selector-mix readout, alongside the M1/M2/M3 gates (re-validated unchanged in the same solo run).
- Gates: M0 `internlm2_xattn_test` + M2 `internlm2_ashape_test` + M3 `internlm2_vslash_test` +
  `xattention_parity` still green (uniform paths unchanged); `internlm2_perhead_test` green;
  pytest/ruff/compileall/`uv lock --check`/`git diff --check` clean.

**RESULTS (32 layers, 823 tok / 7 blocks; same bake & doc as M1/M2/M3):** dense ppl **12.3379** (top-1
44.2%). **perhead mixed keep-all == dense EXACTLY (Δppl 0.00)** — the per-head parity anchor (routing is
exact regardless of which kinds mix). **perhead gather==mask: Δ=8.88e-3 (0.072% rel < 1%)** — the M4
twin. Measured: **perhead Δppl +0.40%** with the offline router assigning (Σ 32 layers × 32 heads =
1024) **86% of heads → xattn, 14% → the cheap static A-shape kernel, 0% → vslash**. The router buys back
nearly all of A-shape-L2's loss (**+3.76% → +0.40%**, approaching the best uniform xattn t=0.9 +0.31%)
while still running 14% of heads on the cheaper positional kernel — the MInference promise: cheap where
it suffices, accurate where needed, at bounded quality. vslash earning **0%** at 7 blocks is consistent
with M3 (vertical-slash is a long-context pattern; it never beats the cheaper ashape *or* the accurate
xattn fallback at this short doc). M1/M2/M3 reproduced bit-identically in the same run (xattn t=0.9
+0.31% / twin 5.94e-3; ashape keep-all 0.00, L=4 +0.58% / twin 2.16e-3; vslash keep-all 0.00, v3s3
+3.01% / twin 2.76e-2), confirming the `head_selectors` field + `_attn_heads` extraction regressed
nothing.

### M5 ✅ — per-head *params* (kernel-aware FLOP-budgeted search) — DONE
Generalized M4 from per-head *kind* (shared per-kind params) to per-head *params*: each query head carries
its OWN selector params (ashape `local`, xattn `threshold`), and the offline assignment becomes
MInference's **kernel-aware FLOP-budgeted search** — per head, the most accurate candidate (kind, params)
whose cost fits a FLOP budget. Like M4 it is a pure **routing layer** over the validated M1/M2/M3
selectors: head `h`'s kept-block mask is byte-identical to the uniform keep for `head_specs[h]`'s (kind,
params). Smallest-diff design:
- a frozen **`HeadSpec(kind, threshold, local, vert, slash)`** + a `head_specs: tuple[HeadSpec,…] | None`
  field on `XAttnConfig` (None ⇒ fall back to M4's `head_selectors` / uniform — every path byte-for-byte
  unchanged; takes precedence when set, both-set rejected). `_select_keep_per_head_specs` computes each
  *distinct* spec's keep/rank for ALL heads (a bounded loop over the DISTINCT specs present — the
  search-grid size, never per-head/per-token, rule 3), stacks `[n_spec,B,H,Tq,Tk]`, routes per head with
  one `take_along_axis`. vslash params are **shared, not per-head**: the global probe index is threaded
  once, so a vslash spec's vert/slash must equal the config's (fail-loud `__post_init__` guard);
  ashape/xattn select locally so their params are freely per-head. Per-head vslash *param* variation is
  deferred to M6 (which reworks the probe).
- `assign_head_specs(errors[C,H], costs[C], candidates, budget)` — the **dual** of M4's policy: per head
  the most accurate candidate whose kernel-aware cost ≤ budget (else the cheapest). Pure/positional ⇒
  unit-testable.
- `InternLM2Attention._attn_qkv` — a parity-preserving extraction of `_attn_heads`' front half (project +
  RoPE + cache + GQA-repeat), shared with the new offline `_attn_keep_counts` (per-head mean kept blocks
  for a candidate = its measured FLOP cost). `__call__`/`_attn_heads` behavior unchanged.

Deliverables:
- `src/quanta/modeling/xattention.py`: `HeadSpec`, `XAttnConfig.head_specs` (+ validation: non-empty,
  mutually-exclusive-with-`head_selectors`, vslash-pin), `_select_keep_per_head_specs`, `select_keep`
  per-spec dispatch (precedence over M4), `_uses_vslash` head_specs case, `assign_head_specs`.
- `src/quanta/internlm2/attention.py`: `_attn_qkv` extraction + offline `_attn_keep_counts`; imports.
- `parity/internlm2_perhead_params_test.py` (NEW, model-free, fp32, T=500, real `InternLM2Attention`):
  `assign_head_specs` budget policy (budget-excludes-accurate / full-budget / under-budget / tie) +
  routing exactness incl. **same-kind heads at different params** + uniform-as-per-head-specs == uniform
  (0.00) + MIXED keep-all == dense, mask **(0.00)** & gather (5e-8) + gather==mask (4e-8) +
  sparsity-active + budget-capped + min_seq-gated + causal-safe + validation (vslash-pin / both-set /
  non-HeadSpec / length mismatch rejected). Run: `uv run python -m parity.internlm2_perhead_params_test`.
- `parity/internlm2_ppl_sparse.py` (EXTENDED): per-layer per-candidate error (reusing the M4 dense
  calibration) + kernel-aware cost via `_attn_keep_counts`, sorted cheap→accurate, `assign_head_specs`
  at `ph2_budget=4.0` blocks; `perhd-p` / `perhd-p gat` (derived) + `perhd-p anchor` (mixed keep-all)
  variants + M5 gates + a realized (kind,params) mix readout, alongside the M1/M2/M3/M4 gates.
- Gates: M0 xattn + M2 ashape + M3 vslash + M4 perhead + `xattention_parity` still green (all prior paths
  unchanged); `internlm2_perhead_params_test` green; pytest/ruff/compileall/`uv lock --check`/`git diff
  --check` clean.

**RESULTS (32 layers, 823 tok / 7 blocks; same bake & doc as M1–M4):** dense ppl **12.3379** (top-1
44.2%). **perhd-p mixed keep-all == dense EXACTLY (Δppl 0.00)** — the per-head-params parity anchor.
**perhd-p gather==mask: Δ=3.29e-3 (<1%)** — the M5 twin. Measured: **perhd-p Δppl +0.15%** — **better than
M4's per-head-kind (+0.40%) AND the best uniform (xattn t=0.9 +0.31%)** — with the FLOP-budgeted search
(budget=4 blocks) assigning (Σ 32 layers × 32 heads = 1024) **ashape:L4 771 (75%), xattn:t0.9 239 (23%),
vslash:v2s2 6 (1%), xattn:t0.95 8 (1%)**. Per-head params let 75% of heads run the cheap static
ashape-L4 kernel (uniform +0.58% alone) while each head still gets its most-accurate-affordable
approximation, so the aggregate beats forcing any single pattern everywhere — the MInference thesis,
realized. M1/M2/M3/M4 reproduced bit-identically in the same run (xattn t=0.9 +0.31% / twin 5.94e-3;
ashape L=4 +0.58% / twin 2.16e-3; vslash v3s3 +3.01% / twin 2.76e-2; M4 perhead +0.40% / anchor 0.00 /
twin 8.88e-3 / mix 86/14/0), confirming the `head_specs` field + `_attn_qkv` extraction regressed nothing.

### M6 ✅ — per-head *vslash params* (param-independent probe) — DONE
Removed M5's vslash-pin: each head can now carry its OWN vertical-slash ``vert``/``slash`` (not just its
own ashape/xattn params), so the per-head search reaches a strictly larger menu. The lever is making the
online probe **param-independent**: :func:`vertical_slash_index` now returns the raw masses
``(key_mass [B,H,Tk], slash_mass [B,H,Tq])`` instead of a baked top-``vert``/``slash`` keep, and the top-k
cut moves into :func:`select_keep` (read from ``cfg.vert``/``cfg.slash``). Two heads read the ONE global
probe (threaded once over the whole sequence — that is what keeps gather == mask) yet cut **different**
vert/slash from the shared masses. Pure routing, exactly like M4/M5: head ``h``'s keep is byte-identical to
the uniform vslash keep for its own vert/slash. Smallest-diff design:
- ``vertical_slash_index`` → ``(key_mass, slash_mass)`` (param-independent; the mass computation is
  byte-for-byte the M3 path — only the in-probe top-k is removed). ``select_keep``'s ``"vslash"`` branch now
  does the top-``cfg.vert``/``cfg.slash`` cut from the threaded masses (so each per-head spec applies its
  own params); ``_select_keep_per_head_specs`` already passes ``sp.vert``/``sp.slash`` per spec ⇒ per-head
  vslash params fall out. The ``XAttnConfig.__post_init__`` vslash-pin is **removed** (a vslash ``HeadSpec``
  no longer has to match the config's vert/slash). xattn/ashape paths byte-for-byte unchanged; the
  uniform/M3/M4/M5 vslash *selections* are byte-identical (same masses + same top-k, just relocated).
- `parity/internlm2_vslash_perhead_test.py` (NEW, model-free, fp32, T=500, real `InternLM2Attention`):
  routing exactness with **two vslash heads at DIFFERENT vert/slash** (each == its uniform spec; the two
  keep different blocks ⇒ the params bite) + config-vert/slash-irrelevance (masses param-independent) +
  uniform-vslash-as-per-head == uniform + MIXED keep-all (2 vslash params + ashape + xattn) == dense (mask
  & gather) + gather==mask + sparsity-active + min_seq-gated + causal-safe.
  Run: `uv run python -m parity.internlm2_vslash_perhead_test`.
- `parity/internlm2_vslash_test.py` (UPDATED): `_check_index` now asserts the masses are
  **param-independent** (identical across cfg.vert/slash). `parity/internlm2_perhead_params_test.py`
  (UPDATED): the M5 vslash-pin rejection becomes "per-head vslash params now allowed".
- `parity/internlm2_ppl_sparse.py` (EXTENDED): the M5 per-head-params search grid gains a 2nd vslash
  param-point (v2s2 **and** v3s3); the derived ``head_specs`` drops its config vert/slash (param-
  independent); the realized mix readout records per-head vslash params.
- Gates: M0 xattn + M2 ashape + M3 vslash + M4 perhead + M5 perhead-params + `xattention_parity` still
  green (all prior selections byte-identical); `internlm2_vslash_perhead_test` green; pytest/ruff/
  compileall/`uv lock --check`/`git diff --check` clean.

**RESULTS (32 layers, 823 tok / 7 blocks; same bake & doc as M1–M5):** dense ppl **12.3379** (top-1
44.2%). **perhd-p mixed keep-all == dense EXACTLY (Δppl 0.00)** — the per-head-vslash-params parity anchor.
**perhd-p gather==mask: Δ=7.45e-4 (<1%)** — the M6 twin. Measured: **perhd-p Δppl +0.04%** — **better than
M5's +0.15%** (which itself beat the best uniform xattn t=0.9 +0.31%) — because the now-reachable 2nd vslash
param lets the FLOP-budgeted search (budget=4 blocks) assign (Σ 32×32 = 1024) **ashape:L4 747 (73%),
xattn:t0.9 230 (22%), vslash:v3s3 39 (4%), xattn:t0.95 8 (1%)**: 4% of heads now run the WIDER vslash:v3s3
(vs M5's 1% at v2s2), and routing those heads to their better-fitting vslash pattern buys the aggregate
down further. Per-head vslash *params* pay off even at this short 7-block doc — the long-context payoff
(where vertical-slash is designed to shine) awaits M7's chunked probe + wall-clock bench. M1–M5's other
streams reproduced bit-identically in the same run (xattn t=0.9 +0.31% / twin 5.94e-3; ashape L=4 +0.58% /
twin 2.16e-3; vslash v3s3 +3.01% / v2s2 +7.29% / keep-all 0.00 / twin 2.76e-2; M4 perhead +0.40% / anchor
0.00 / twin 8.88e-3 / mix 86/14/0), confirming the param-independent-masses refactor regressed nothing.

### M7 — key-chunk the long-context probe + gather-path wall-clock bench (remaining)
- **Long-context vertical-slash probe**: key-chunk the probe (currently one ``max_alloc_gb``-guarded
  ``[B,H,lp,S]`` matmul + an ``[B,H,lp,t]`` slash gather) so it scales past the gate's short doc to 100K+
  where vertical-slash earns real assignments (at 7 blocks it already earns 4% via M6's per-head params,
  but its design payoff is long-range). Build on M6's param-independent masses: accumulate ``key_mass`` /
  ``slash_mass`` over key chunks (online softmax; keep the single-shot ``mx.softmax`` path as the default so
  the short-doc gate stays byte-identical, chunk only when the probe would exceed ``max_alloc_gb``) and
  model-free-gate chunked masses ≈ single-shot + chunked keep-all == dense + chunked gather == mask.
- **Gather-path prefill bench**: a real wall-clock measurement of the ``gather=True`` speed path — the
  actual FLOP/memory win (the mask path measures quality only; with fast SDPA + an additive block mask MLX
  still computes the full QKᵀ). Pair it with the long-context probe so vertical-slash's long-range payoff
  is measured where it lands. Its own milestone + ppl gate vs M6.

---

## Artifact dependencies (paths — needed by the real-model gates)

These live **outside the repo** under `~/models` and do not survive a disk format — see restore note.
- **Bake (M1 loads this):** `/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64` (~8.3 GB).
- Source (for re-bake / tokenizer): `/Users/pmrj/models/internlm2_5-7b-chat-1m` (~14 GB).
- EAGLE sidecars (prior track, not needed for MInference but precious):
  `/Users/pmrj/models/internlm2_eagle/` — `drafter_int8g64_refined2.safetensors` is the 0.460 best.
- Reference teacher (NEVER delete): `/Users/pmrj/models/Kimi-K2.6` (554 GB).

## Fresh-machine restore checklist (after formatting the Mac)

1. **Clone + verify:** `git clone git@gitlab.com:pmrj/final_quanta.git` → confirm `git log` shows
   `871258f` (M0) on `main`. Reinstall `uv`; `uv sync`.
2. **Restore the agent memory** (the project brain) — now **committed in-repo at `.claude/memory/`**,
   so NO separate `~/.claude` backup is needed. After cloning, run `bash scripts/restore_claude_memory.sh`
   to copy it into `~/.claude/projects/<slug>/memory/` (or `--symlink` to keep future memory edits living
   in the repo). `MEMORY.md` + the `project_*.md` / `feedback_*.md` files seed a fresh session.
3. **Restore `~/models`** (or the subset you backed up) — at minimum the int8-g64 InternLM2 bake for
   M1. Bakes are regenerable from sources + repo bake scripts (hours each) if not backed up; sources
   are re-downloadable from HF.
4. **Re-run M0 gate to confirm the environment:** `uv run python -m parity.internlm2_xattn_test`
   (model-free — needs no `~/models` artifacts; if it passes, the code+env are good). Then proceed to M1.
