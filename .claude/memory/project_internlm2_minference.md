---
name: project-internlm2-minference
description: InternLM2.5 sparse-prefill track (MInference family, lossy → ppl-gated). The 2nd InternLM2.5 speed lever after EAGLE spec-decode. KEY DECISION — do NOT build from scratch: the bounded-memory chunked block-gather *execution* substrate already exists, validated + ppl-gated, in quanta.modeling.xattention; reuse it and layer MInference's selectors on top. M0 871258f: wired the substrate into InternLM2Attention behind a self.sparse hook (default None = dense, byte-unchanged) + model-free integration parity gate. M1 next = real-model long-doc ppl sweep on the int8 bake (solo GPU). InternLM2 is the only keeper still paying full O(T²) dense prefill.
metadata:
  node_type: memory
  type: project
  originSessionId: 78ef7db6-f4ec-4c53-857e-3bd77cc65962
---

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

See also [[prefill-optimization-landscape]], [[project-internlm2-eagle]] (1st lever, DONE), [[project-model-targets]].
