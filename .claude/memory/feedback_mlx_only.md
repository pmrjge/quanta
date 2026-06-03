---
name: mlx-only-implementation
description: "Implement everything in MLX, including the parity reference (plain mlx.core) — not a torch port"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Implement every component of quanta in MLX. This includes the parity *reference*
forward: it must be a dead-simple, obviously-correct **plain `mlx.core`**
implementation (literal ops mirroring the authoritative HF `modeling_deepseek.py`
math), NOT a torch/transformers port. The runtime forward is `mlx.nn` + `mx.fast`
primitives. The parity gate diffs plain-`mlx.core` reference vs the `mlx.nn`
runtime. Load source weights via `mx.load` (safetensors), not torch.

**Why:** user stated on 2026-05-23 "our project aims to implement everything in
MLX." CLAUDE.md's methodology offers "plain mlx.core (or a HF/transformers
reference, offline)" — the user wants the plain-mlx.core branch taken, even though
torch is technically allowed offline.

**How to apply:** never reach for torch as the reference. When a primitive needs
independent validation, check it against hand-computed values / numpy in MLX, not
torch. Reserve torch/transformers strictly for things MLX genuinely cannot do
offline (e.g. a tokenizer with no MLX equivalent). See [[artifact-output-path]].
