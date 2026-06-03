---
name: feedback-kimi-not-deepseek
description: "Treat Kimi-K2.6 as its own architecture — verify formats/conventions empirically, don't assume DeepSeek-V3 parity"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Treat **Kimi-K2.6 as its own architecture**. The parity reference was transcribed
from `modeling_deepseek.py`, but Kimi deviates from DeepSeek-V3 in places, and an
assumed-DeepSeek convention can be silently wrong. Verify formats/conventions
**empirically against the actual checkpoint**, not against a DeepSeek transcription.

**Why:** The catastrophic forward bug ([[project-forward-bug-resolved]]) was exactly
a Kimi-specific deviation — the source int4 experts are **offset-binary (excess-8)**,
while the generic/DeepSeek assumption (and the code) was two's-complement
sign-extend. The user explicitly steered: "consider Kimi's architecture, not
DeepSeek-V3." A self-authored reference sharing the wrong assumption hides this from
parity gates.

**How to apply:** When implementing or debugging any Kimi component (quant packing,
routing, RoPE/YaRN, norm eps, head splits), confirm against Kimi's real data/config
rather than trusting DeepSeek defaults. Concrete techniques that worked: histogram
the raw int codes (a clean bell curve centered at 8 ⇒ offset-binary, not signed);
check dequantized weight **mean** against a known-good native-bf16 tensor (the shared
expert is zero-mean) — a DC bias means the convention is wrong.
