---
name: feedback-selfcontained-artifact
description: "Baked artifacts must be fully self-contained — config.json + manifest + safetensors + all metadata, relative refs only"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e88fadb0-5f2c-48da-82a1-e14175af6551
---

A baked artifact must be a **fully self-contained, relocatable bundle**: it includes
all `safetensors`, the `manifest.json`, a proper `config.json`, and all metadata,
with **only relative references within the artifact's own folder** — no absolute
paths, no source-checkpoint paths, no symlinks, no cache paths, no external refs.

**Why:** User requirement, stated explicitly. The artifact must work standalone if
copied/moved. Extends CLAUDE.md's existing rule ("manifest references are relative,
in-artifact only") to also mandate a self-contained `config.json` + complete
metadata, so the artifact is loadable without the original source tree present.

**How to apply:** At bake time, write a complete `config.json` and `manifest.json`
into the artifact dir; every internal reference is relative to the artifact root.
Runtime offload/state lives in the sibling `<artifact>_offload`, never inside the
artifact, and `manifest.json` is never mutated at runtime. See
[[project-artifact-output-path]] for where artifacts are written.
