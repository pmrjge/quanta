---
name: artifact-output-path
description: Where quanta writes baked Kimi artifacts and the naming pattern for them
metadata:
  node_type: memory
  type: project
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

Baked quanta artifacts are written to `~/models/<source_name>-quanta_<type>`.
Current focus: `source_name = Kimi-K2.6` (so e.g. `~/models/Kimi-K2.6-quanta_<type>`).
The whole quantized model is held RAM-resident: **no offload, no streaming, no
`_offload` sibling.** Source checkpoint `~/models/Kimi-K2.6` is never deleted.

**Why:** user specified this output location/naming and the all-resident policy
on 2026-05-23 (M3 Ultra, 512 GB, ≤490 GiB working set — the target fits resident).
Keeps baked bundles beside the source under `~/models`, outside the repo.

**How to apply:** use this path when building the bake pipeline (plan step 3).
Confirm the exact `<type>` slug with the user before the first bake — it should
encode the quant split (e.g. routed gate/up int3 g128 + down int4 g128). Not
needed for the parity harness (plan step 1), which is the current focus.
