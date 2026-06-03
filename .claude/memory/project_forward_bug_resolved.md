---
name: project-forward-bug-resolved
description: "The catastrophic-perplexity forward bug is FOUND and FIXED — it was the source int4 dequant sign convention, not the experts/int3"
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

The catastrophic teacher-forced perplexity (2.44M ppl, 0.000 top-1 here; ~165 in
the prior `quantification` project) is **resolved**. bf16 runtime now coherent:
**teacher-forced ppl 3.31, top-1 0.745** on ordinary English (`parity/ppl.py`).

Root cause: the source checkpoint's int4 routed experts are stored **offset-binary
(excess-8)** — code `c∈0..15` means `(c-8)·scale` — but `dequantize_packed_int4`
**sign-extended** (two's complement), injecting a ~-0.0067 DC bias into every
expert weight. Multiplied by the non-zero-mean RMSNorm input (Σx ≈ -155), that DC
bias added ~+1 to every expert output, detonating the residual at **L2** (per-pos
norm 3.5 → 7900), compounding ~1.6×/layer to a position-independent fixed point →
every position emits the same token (`' foss'`) → worse-than-uniform ppl. Fixed in
`src/quanta/compressed_int4.py` (one line: `(code - 8)` instead of sign-extend).

**Why:** The bug passed every numeric parity gate (L0/L1/full-model/cache) because
the self-authored plain-mlx reference dequantized the *same wrong way* — both wrong
identically, so runtime==reference held at 0.0 drift. Only the independent e2e
arbiter (perplexity) caught it. This vindicates the parity-first rule that the
arbiter must be independent of the thing it judges. It also confirms CLAUDE.md's
settled finding that the failure was NOT the experts/int3 coding — it was a
localized shared-forward bug. See [[feedback-kimi-not-deepseek]].

**How to apply:** The runtime is now parity-correct AND coherent in bf16, so the
**int3-floor question** (does int3 routed quant preserve coherence) is finally
*measurable* through this runtime — that was the whole point of building the ppl
gate. CLAUDE.md still frames this forward bug as open/unfound ("never caught"); that
narrative is now outdated — don't re-attribute the prior failure to quantization.
