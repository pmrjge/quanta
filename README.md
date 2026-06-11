# quanta

**Parity-first, MLX-native quantization + sparse-MoE inference runtime** for an M3 Ultra
(512 GB unified memory). Every component is gated against a numeric reference before it is
optimized or its quantization is judged. See [CLAUDE.md](CLAUDE.md) for the full thesis,
engineering rules, settled findings, and methodology — it is the authoritative context and is
loaded automatically at the start of every session.

This README is the **clean-install bootstrap**: how a fresh macOS machine (no `~/models`, no
prior agent memory) picks the project back up from the repository alone.

---

## Where to start (resume a session)

Read in this order:

1. **[CLAUDE.md](CLAUDE.md)** — master context: permanent engineering rules (1–8), settled
   findings, model facts, parity-first methodology, verification commands, the measured serving
   fleet baseline, and the **Active task** paragraph (current = Nex-N2-Pro / Qwen3.5-397B N3).
2. **[PLAN_nex_n2_pro.md](PLAN_nex_n2_pro.md)** — the **active task's** full handover.
3. The other `PLAN_*.md` — durable handovers for completed / paused tracks
   ([PLAN_nemotron_ultra.md](PLAN_nemotron_ultra.md), [PLAN_minference.md](PLAN_minference.md),
   [PLAN.md](PLAN.md) (#18), [PLAN_153.md](PLAN_153.md), [PLAN_qwen35_experts.md](PLAN_qwen35_experts.md)).
4. **[INITIAL_PROMPT.md](INITIAL_PROMPT.md)** — the original project brief.
5. **`.claude/memory/`** — the agent "brain" snapshot (settled findings, user profile, per-task
   records). Restore it on a fresh machine (below); index is `.claude/memory/MEMORY.md`.

---

## Clean-install bootstrap (fresh macOS)

```bash
# 1. Install uv (https://docs.astral.sh/uv/). Python 3.13 is pinned in .python-version.
# 2. Dependencies (pick what you need):
uv sync                       # runtime only (mlx, numpy, regex, torch, tqdm)
uv sync --extra reference     # + offline parity refs (transformers/safetensors/sentencepiece) — needed for parity gates & bakes
uv sync --extra omlx          # + oMLX serving integration

# 3. Arm the committed parity pre-commit hook (runs the fast fail-open gate guards):
git config core.hooksPath .githooks

# 4. Restore the agent "brain" into Claude Code's per-project memory dir:
scripts/restore_claude_memory.sh            # copy the snapshot (safe, default)
scripts/restore_claude_memory.sh --symlink   # OR symlink, so future memory edits land back in the repo
```

`uv.lock` is committed for reproducible installs (`uv lock --check` must pass). CI
(`.github/workflows/parity-gates.yml`, Apple-silicon `macos-14`) runs the full suite on push/PR.

---

## What runs WITHOUT the models (any machine, no GPU)

The **model-free parity sweep** verifies interface + logic on stubs — ~100 gates, no weights,
no `~/models`, peaks ~0.4 GiB. This is the bulk of the test surface:

```bash
uv run --with pytest pytest tests/ -m "not slow" -q   # fast inner loop (env + fail-open guards, ~instant)
uv run --with pytest pytest tests/ -q                 # full: subprocess-runs every model-free parity gate (~4 min)
uv run python -m parity.run_modelfree_sweep [--jobs N]  # standalone, parallel equivalent
uv run --with ruff ruff check src tests
uv run python -m compileall -q src tests
uv lock --check
```

The sweep auto-discovers `parity/*_test.py` and **excludes** real-weight (SOLO) gates by a
multi-signal detector + a 4 GiB RSS watchdog. When you add/remove/reclassify a gate, regenerate
the identity manifest: `uv run python -m parity.run_modelfree_sweep --update-manifest`.

---

## What needs the MODELS + an M3 Ultra (real-weight, SOLO)

Real-weight gates (`parity/*_real_test.py`, or any that load from `~/models`) load a **9–306 GiB**
artifact and **must run one at a time** — two resident models OOM-reboot the box. They are
excluded from the sweep and are the SOLO end-to-end arbiters (teacher-forced perplexity).

Models live **outside the repo** under `~/models` (never committed; see `.gitignore`). A clean
install has **none** — obtain the bf16 sources and re-bake the quantized artifacts. The served
fleet (resident sizes from CLAUDE.md's baseline table):

| model (architecture) | bf16 source under `~/models` | bake script | resident artifact |
|---|---|---|---|
| **Nex-N2-Pro** = Qwen3.5-397B-A17B (`qwen3_5_moe`) | `Nex-N2-Pro` (HF `nex-agi/Nex-N2-Pro`, 739 GiB) | `parity/run_bake_nex_n2_pro_int4g64.py` | `Nex-N2-Pro-quanta_int4g64` — **214 GiB (SHIPPED)** |
| **Nemotron-Ultra-550B** (`nemotron_h` hybrid) | `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | `parity/run_bake_nemotron_ultra_int4rtn_g64.py` (+ `…_mtp_…` sidecar) | `…-quanta_int4rtn_g64` — 306 GiB |
| **DSV4-Flash** (sparse MoE + compressed-KV) | `DeepSeek-V4-Flash` | `parity/run_bake_dsv4.py` | `…-quanta_int4g64` — 180 GiB |
| **Nemotron-Super-120B** (`nemotron_h`) | `NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | `parity/run_bake_nemotron_int4g64.py` | `…-quanta_int4g64` — 68 GiB |
| **Qwen3.6-35B-A3B** (`quanta.qwen35`) | (qwen35 source) | qwen35 bake path | `…-quanta_int4g64` — 19 GiB |
| **InternLM2.5-7B** (dense GQA, 1M ctx) | `internlm2_5-7b-chat-1m` | int8-g64 bake | `…-quanta_int8g64` — 9 GiB |

Exact source HF ids and the per-model bake recipes are encoded in the `parity/run_bake_*.py`
scripts and `src/quanta/bake/`. **Keep `~/models/Kimi-K2.6`** — the int4 reference teacher; never
delete (CLAUDE.md). Baked artifacts are immutable, self-contained bundles (relative in-artifact
refs only); runtime offload state lives in the sibling `<artifact>_offload`, never inside.

---

## Repo layout

- `src/quanta/<model>/` — per-model runtimes (`dsv4`, `nemotron`, `qwen35`, `internlm2`, `minimax`,
  `glm`, `qwen25`, `eagle`, …) + shared `modeling/`, `paged/`, `spec/`, `bake/`, `shim/`.
- `parity/*_test.py` — gates (model-free sweep + SOLO real-weight); `parity/run_bake_*.py` — bakes.
- `tests/` — pytest entry; `tests/test_parity_modelfree.py` drives the model-free sweep.
- `.githooks/pre-commit`, `.github/workflows/parity-gates.yml` — enforcement.
- `pth_data/` — the oMLX import-hook `.pth` shipped to site-packages (serving autopatch).

Serving: `quanta-omlx serve …` (the `[project.scripts]` entry) arms the autopatch and routes quanta
artifacts to `QuantaOmlxEngine`. The engine emits **raw** output; oMLX owns response shaping.

---

## Hardware

One **M3 Ultra, 512 GB** unified memory; usable working-set ceiling **≈ 490.4 GiB**. The whole
quantized model is held RAM-resident (no offload/streaming). **One model resident at a time.**

---

## Status (2026-06-11)

**Active:** Nex-N2-Pro (Qwen3.5-397B-A17B) — **N3 serving**, over the **shipped int4-g64 artifact
(214 GiB)**. N0–N3-2 complete (enablement → layer parity @ 397B → int4/int6 bake + ppl arbiter →
resident/batched serving re-gate → `qwen3_coder`/`qwen3` parsers). **Remaining N3:** 1M needle
gate, paged-KV + prefix caching (only the 15 full-attn layers hold KV), MInference sparse-prefill
on those 15 layers, fused/batched Gated-DeltaNet decode-step, multi-stream decode past B=32. See
[PLAN_nex_n2_pro.md](PLAN_nex_n2_pro.md).
