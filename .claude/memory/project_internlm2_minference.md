---
name: project-internlm2-minference
description: "InternLM2.5 sparse-prefill track (MInference family, lossy → ppl-gated). 2nd InternLM2.5 speed lever after EAGLE. KEY DECISION — reuse the bounded-memory chunked block-gather *execution* substrate (quanta.modeling.xattention), layer MInference's selectors on top. COMPLETE M0–M10: per-head (kind+params) selectors (XAttention/A-shape/vertical-slash) + grouped-gather fold (default-on) → up to 4.3×@64K per attn layer; ppl +0.04% @ a 7-block doc but +31.8% @ 16K code corpus (steep speed/quality at long ctx). The substrate now also transfers to Nex-N2-Pro's 15 full-attn layers."
metadata:
  node_type: memory
  type: project
  originSessionId: 78ef7db6-f4ec-4c53-857e-3bd77cc65962
---

**STATUS: COMPLETE (M0–M10) — see the foot of this file. The roadmap below ("M1 next", "M2+") is the
historical M0 plan, retained for the substrate-reuse decision; what actually shipped is at the bottom.**

**Why:** InternLM2.5 is the only keeper still paying full **O(T²) dense prefill** (DSV4 native-sparse,
Nemotron-Mamba/qwen35-GDN linear). The June-2026 prefill research ([[prefill-optimization-landscape]])
ranked training-free **sparse prefill** (MInference ~10×@1M, XAttention ~13.5×@256K) as the asymptotic
prize for it — lossy, so **ppl/retrieval-gated, NOT numeric-parity-gated** (CLAUDE.md rule 4/6). This is
the 2nd InternLM2.5 lever after EAGLE spec-decode ([[project-internlm2-eagle]], DONE).

**KEY DECISION (do not re-litigate — user chose "reuse substrate first"):** MInference and XAttention are
the SAME family; they differ only in the *selection* method (MInference = per-head A-shape / vertical-slash
/ block-sparse patterns; XAttention = antidiagonal block scoring + nucleus). The hard part — the
bounded-memory, fail-loud, **chunked block-gather execution** (`gather_sparse_attention`) + the additive-mask
path (`sparse_prefill_mask`) — already exists, model-agnostic and validated, in
**`quanta.modeling.xattention`** (`parity/xattention_parity.py` green; `threshold=1.0` == dense). The MLA
keepers wire it via a one-line `self.sparse: XAttnConfig | None` hook in `quanta.modeling.attention`.
InternLM2 had **no sparse hook at all**. So: reuse the execution substrate, add MInference's selectors as a
later milestone — smallest diff, parity-anchored, no rebuild.

**Roadmap (one milestone/commit, then STOP to compact; one model resident at a time):**
- **M0 ✅ `871258f`** — wire the substrate into `quanta.internlm2.attention.InternLM2Attention`: add
  `self.sparse: XAttnConfig | None = None` (default None ⇒ dense, **byte-for-byte unchanged**) and route
  from-scratch prefill (`t == kv_len and t >= sparse.min_seq`) through `gather_sparse_attention` (gather
  path) / `sparse_prefill_mask` (mask path) over the **GQA-repeated `kr`/`vr`** — per-head scoring sees
  exactly what dense SDPA does, so `threshold=1.0` reproduces dense. Decode (t==1) + cache-continuation
  always dense. Gate `parity/internlm2_xattn_test.py` (model-free, fp32, T=500, real `InternLM2Attention`):
  keep-all mask path == dense **EXACTLY (0.00)**, gather path 5e-8, threshold=0.5 drops blocks (5e-2),
  min_seq>T gates off (0.00), perturb-final-token leaves block 0 unchanged (0.00, no future leakage).
  Dense-path gate `internlm2_batched_attention_test` still greedy-exact |Δlogit|=0.00; pytest/ruff/
  compile/lock/diff clean.
- **M1 (next)** — real-model **long-doc teacher-forced ppl sweep** on the resident **int8-g64 7B bake**
  (`/Users/pmrj/models/internlm2_5-7b-chat-1m-quanta_int8g64`), solo GPU: dense vs XAttn thresholds, report
  ppl / top-1 / Δppl% — the quality cost of the reused substrate on InternLM2. **GOTCHA:** `parity/ppl_long.py`
  is **Kimi-bound** (KimiTextConfig/SourceCheckpoint/KimiTokenizer/`layer.self_attn.sparse`); InternLM2 layers
  use `layer.attention.sparse`. Write a sibling `parity/internlm2_ppl_sparse.py` (drive the resident bake via
  `InternLM2ResidentModel`, set `.sparse` per layer, real prose ≥ a few blocks) rather than generalize the
  Kimi harness in place. NOTE: with fast SDPA + an additive block mask MLX still computes full QKᵀ ⇒ the mask
  path measures *quality* only; the gather path is the actual speed win.
- **M2+** — add MInference's **vertical-slash / A-shape** per-head selection (offline pattern + online
  last-query-block index estimation) as an alternative *selector* feeding the SAME `gather_sparse_attention`
  execution, ppl-gated against dense + XAttn. This is where the genuine new work + the quality questions live.

---
**COMPLETE (M0–M10) — what actually shipped (handover `PLAN_minference.md`):**
- **M1** measured XAttention on the int8-g64 bake: prefill @ threshold 0.9 = **+0.31% ppl** ("free"); gather speed-path == mask quality-path.
- **M2 A-shape** (sink + local window), **M3 vertical-slash** (online probe → top-vert/top-slash bands), both onto the SAME `gather_sparse_attention` execution via a `selector` discriminant (`"xattn"` default byte-unchanged).
- **M4 per-head KIND**, **M5 per-head (kind, PARAMS)** via a frozen `HeadSpec` + offline FLOP-budgeted `assign_head_specs` — **+0.15% ppl** beats any uniform (75% heads on the cheap static kernel, each its most-accurate-affordable approx — the MInference thesis). **M6** per-head vslash params → **+0.04%**.
- **M7** key-chunked the long-context vslash probe (online-softmax two-pass, O(one key chunk); short-doc path byte-unchanged). **M8** timed the gather speed path: **O(T²)→O(T) crossover 8–16K, up to 4.3×@64K** per attn layer (ashape/vslash sparsest; xattn nucleus least sparse → slowest, hence assigned per-head).
- **M9 grouped-gather fold** (`grouped_gather`, since **default-on** for per-head configs, rule-4 bit-exact): partition heads by spec, gather each group at its OWN `max_kept` → **2.64× faster than naive** per-head gather (which paid the densest head's budget for all). **M10** long-context per-head ppl @ 16384 tok: keep-all==dense, M7 chunked==single-shot bit-identical, M9 fold==mask — and the real frontier: **+31.8% ppl** on a code-heavy corpus (94% ashape / 6% vslash) vs the adaptive xattn nucleus near-lossless **+2.8% but keeps ~65%** (priced out of the budget). A steep, real speed/quality tradeoff at long ctx (vs the 7-block doc's +0.04%).

See also [[prefill-optimization-landscape]], [[project-internlm2-eagle]] (1st lever, DONE), [[project-model-targets]], [[project-nemotron-ultra]] (rode the same session), [[project-nex-n2-pro]] (the substrate's next consumer — its 15 full-attn layers).
