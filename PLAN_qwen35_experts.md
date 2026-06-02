# PLAN_qwen35_experts.md — active task handover (qwen35: routed-expert `gather_qmm` packing)

> Durable, repo-tracked handover for the NEXT task. Read `CLAUDE.md` first (permanent rules +
> quant policy), then **`PLAN_153.md`** (the just-finished Qwen3.6 **option B** packed-mixer work —
> the `nn.QuantizedLinear`/`_packed_linear`/`_load_quant_triplet` machinery this task reuses), then
> this. **Status: COMPLETE — M1 ✅ `a6b3b49`, M2 ✅ `d17882e`, M3 ✅ `f720fda`; `packed_experts`
> graduated default ON (63.4 → 20.3 GiB resident, greedy-exact, ppl +0.24% on real prose).** This completes the Qwen3.6 runtime's *documented* `gather_qmm`
> design: keep the int4 **routed experts** packed and dispatch via `mx.gather_qmm` instead of
> dequantizing them to bf16 and running `mx.gather_mm`. **The big memory lever: ~79 → ~30 GiB
> resident** (the option-B bench peaked at **79.3 GiB for a 21 GiB int4 artifact** — the experts are
> dequantized). It is **batch-invariant** (so it composes with option B without reviving the #153
> batch-M bug) and **greedy-exact** vs the bf16 path (same int4 codes). **DSV4 already does exactly
> this** (`src/quanta/dsv4/moe.py:_swiglu_stack_packed`) — the precise template.

---

## Why — the finding (option B M4 bench, `Qwen3.6-35B-A3B-quanta_int4g64`)

The graduated option-B runtime peaks at **79.3 GiB @ B=32** for a **~21 GiB** int4-g64 artifact.
`21 × 16/4.25 ≈ 79` — i.e. the resident path holds the model's weights as **dequantized bf16**, not
int4. Option B packs the *mixer* projections (`nn.QuantizedLinear`), but those are a small slice; the
**routed experts dominate** (512 experts × {fused gate+up, down} × 40 layers). They are still
dequantized to bf16.

This is a **known, documented gap.** `src/quanta/qwen35/moe.py:11-12` already states the intent:

> *"Dispatch is sparse `mx.gather_mm` over the gathered (token, slot) rows … The post-bake resident
> runtime **swaps `gather_mm` → `gather_qmm` over the packed stacks** (same `[E,out,in]` layout)."*

…and `batched_runtime.py:7-8` already *describes* the experts as "all gather_qmm-fed". The runtime
never realized it — it took the simplest-first dequant path.

## Root cause (exact lines)

- `src/quanta/qwen35/runtime.py:165-171` — `_load_block` calls `moe = art.moe(i)` then
  `blk.mlp.set_experts(moe["experts_gate_up"], moe["experts_down"])`.
- `src/quanta/qwen35/artifact.py:171-185` — `moe(i)` uses `self.read(...)`, and
  `read → _dequant → mx.dequantize → .astype(bf16)` (artifact.py:117-122, 105-115). So the int4
  expert stacks come back **bf16**.
- `src/quanta/qwen35/moe.py:62-78` — `_routed_sparse` runs `mx.gather_mm` over those **bf16** stacks.

## The fix — keep experts packed, dispatch `gather_qmm` (mirror DSV4)

Hold the routed-expert stacks as their **packed int4 triplet** (`.weight_packed` + `.weight_scale` +
`.weight_bias` + `(bits, group_size)` from the manifest — rule 6, read the width, never hardcode) and
route through `mx.gather_qmm`. **Qwen35 is simpler than DSV4** (the template): plain **affine** int4
(NO AWQ `awq_scale` division — DSV4 has it, Qwen35 does not) and a **fused** `gate_up` stack ⇒ **two**
`gather_qmm` calls (gate_up, then down), not three.

**Correctness — why this is safe and in-scope:**
1. **Batch-invariant.** `gather_qmm` is the same per-(token,slot) **M=1 matvec** structure as
   `gather_mm` (each gathered slot is `[out,in] @ [in,1]`), so it has **no batch-M GEMM tiling** —
   it does **not** revive the #153 dense-bf16 batch-M reorder that option B fixed. The settled
   "MoE / `gather_mm` is exempt — do NOT touch" finding stays true; this preserves the matvec
   structure, only swapping the dequant-then-dense kernel for the fused-quantized one. It composes
   with option B's chunked mixer packing with no interaction.
2. **Greedy-exact, not a quant change.** Packed vs bf16-dequant hold the **same int4 codes**; only
   the kernel differs (fused dequant vs separate `mx.dequantize` + dense). Expect `~1e-6` (exactly the
   packed-vs-bf16 forward equivalence option B's `qwen35_forward_test` already proved for the mixer).
   Teacher-forced ppl cannot distinguish them.
3. **Shared expert stays bf16** (CLAUDE.md quant policy: *shared expert never quantized*; it is one
   always-on expert per layer — tiny). Router `gate` stays bf16. DSV4's `_shared` is bf16 too.

**The `gather_qmm` call convention** (the one non-obvious bit — arg order differs from `gather_mm`,
which puts the *weight* first; `gather_qmm` puts the *activation* first, weight second — see
`dsv4/moe.py:114-124`). Minimal unsorted swap mirroring the current `_routed_sparse` index scheme
(`exp = idx.reshape(-1)`, `tok = repeat(arange(n), topk)`, `mc = n*topk`):

```python
x_in = xf[tok]                                            # [mc, hidden]  gather activations per slot
rows = mx.arange(mc, dtype=mx.int32)
gu = mx.gather_qmm(x_in[:, None, :].astype(mx.bfloat16),  # activation FIRST
                   gate_up["packed"], gate_up["scale"], gate_up["bias"],
                   lhs_indices=rows, rhs_indices=exp, transpose=True,
                   group_size=gate_up["group_size"], bits=gate_up["bits"])[:, 0, :]   # [mc, 2*inter]
h = _swiglu_gate_up(gu, inter)                            # existing helper -> [mc, inter]
d = mx.gather_qmm(h[:, None, :].astype(mx.bfloat16),
                  down["packed"], down["scale"], down["bias"],
                  lhs_indices=rows, rhs_indices=exp, transpose=True,
                  group_size=down["group_size"], bits=down["bits"])[:, 0, :]          # [mc, hidden]
return d.reshape(n, topk, hidden)
```
`transpose=True` matches the `[E, out, in]` stack layout (`x[.,in] @ W[.,out,in].T`): gate_up is
`[E, 2*inter, hidden]` (out=2*inter, in=hidden) ✓, down is `[E, hidden, inter]` (out=hidden, in=inter)
✓. Keep it **unsorted** (no `sorted_indices`) for M1 so the parity gate is a clean packed-vs-dequant
diff at identical dispatch order. (DSV4 sorts by expert id for bandwidth amortization — that is
greedy-exact too since it is a permutation of independent matvecs, but defer it as an optional M3 perf
lever, not the correctness path.)

**Flag (rule 4):** new `packed_experts: bool` (default **False** until its own parity gate +
real-model ppl/memory are green, then graduate True), threaded exactly like option B's `packed` —
`Qwen35ResidentModel` → `Qwen35BatchedResidentModel` (`__init__` + `from_inner`) → `shim/omlx`.
**Independent of `packed`** (the mixer flag, already graduated True): one packs the mixer projections,
the other the experts; keep the two graduations separate. Mirror DSV4's `DSV4ResidentModel(...,
packed_experts=...)` (`dsv4/runtime.py:85`).

## Milestones (model-free M1–M2; M3 = solo-GPU real-model gate)

- **M1 ✅ `a6b3b49` — packed routed path in `moe.py` + `Qwen35MoEModule`.** Add `_routed_sparse_packed(xf, idx,
  gate_up, down, inter)` (the `gather_qmm` body above), and dispatch it from `qwen35_moe` by
  **auto-detecting a dict** (`isinstance(p["experts_gate_up"], dict)`) — exactly DSV4's
  `routed_fn = _swiglu_stack_packed if isinstance(experts["w1"], dict) else _swiglu_stack`
  (`dsv4/moe.py:171`). Extend `Qwen35MoEModule` (`model.py:28`) to optionally hold the packed triplets
  (a `set_experts_packed(gate_up_dict, down_dict)` sibling of `set_experts`) and pass them through
  `_params()`/`__call__`. **Gate (model-free, tiny config):** packed-experts MoE == bf16-dequant MoE
  (`_routed_sparse` on the `mx.dequantize` of the same codes) **greedy-exact** (expect `~1e-6`), and
  the existing dense oracle (`_routed_dense`) still matches both. (The current `gather_mm`-vs-dense
  gate lives in the Qwen35 MoE/forward parity test — extend it with a packed leg, building the packed
  triplet from `mx.quantize` codes exactly as `qwen35_forward_test._packed_forward_parity` does for
  the mixer.)
- **M2 ✅ `d17882e` — wire `packed_experts` through `_load_block` + the runtimes.** Under `packed_experts=True`,
  `_load_block` reads the packed triplets (**reuse `_load_quant_triplet`** — runtime.py:66, it is
  shape-agnostic and already returns `(.weight_packed, .weight_scale, .weight_bias, bits, gs)` for the
  3-D stacks at `…mlp.experts.gate_up_proj` / `…experts.down_proj`) and calls `set_experts_packed`
  instead of `art.moe()`+`set_experts`; shared expert + `gate` stay bf16 (`art.read`). **Update
  `_block_arrays` (runtime.py:175-181)** — it currently appends `blk.mlp.experts_gate_up,
  experts_down` (bf16 arrays) for `mx.eval`; under packed these are triplet dicts, so it must eval the
  packed/scale/bias components or the resident set will not be pinned. Thread `packed_experts` through
  `Qwen35ResidentModel.__init__` (198) → `Qwen35BatchedResidentModel.__init__` + `from_inner`
  (`batched_runtime.py`) → `shim/omlx.py` construction. The batched MoE (`batched_runtime.py:286`) goes
  through `blk.mlp(...)`, so it inherits `gather_qmm` for free — no separate batched change. **Gate:**
  `qwen35_forward_test` packed-experts-vs-bf16 forward greedy-exact + full model-free regression
  (`qwen35_batched_test`, `qwen35_batched_loopkill_test`, `qwen35_forward_test`) — the loop-kill is
  unaffected (experts were already batch-invariant).
- **M3 ✅ `f720fda` — real-model gate + graduate (SOLO GPU; one model at a time).** New gate
  `uv run python -m parity.qwen35_batched_bench experts` loads the real int4-g64 bake **TWICE**
  (`packed_experts` False then True, ONE model resident at a time — `gc.collect`+`mx.clear_cache`
  between, 140 GiB wired limit) and asserts: (a) **resident 63.4 → 20.3 GiB** (−43.1, **0.32×**) — the
  bf16-dequant RESIDENT is 63.4, not the option-B 79.3 which was the PEAK@B32 incl. transient; (b)
  **greedy-exact** 48-tok cached-decode trace; (c) **ppl 2.3150 → 2.3204** (rel **0.24%**) + top-1 agree
  **0.9858** on REAL prose (`_teacher_forced` mirrors `parity/ppl.py`: repo PROSE fixture +
  `Qwen35Tokenizer.from_pretrained`; Qwen3.5 has no BOS so `add_bos` is a no-op). **Methodology note:**
  a first attempt on SYNTHETIC ids falsely failed (c) at |ΔNLL|≈2e-2 — near-uniform synthetic ids
  amplify bf16 rounding; CLAUDE.md's arbiter is ppl + top-1 on REAL prose (model confident → robust),
  and bf16-dequant is the LOSSIER path (it rounds each dequant weight to bf16 PRE-matmul; `gather_qmm`
  keeps full precision in the fused dequant), so packed isn't drifting — greedy is bit-identical. All
  green → **graduated `packed_experts=True`** default in `Qwen35ResidentModel.__init__` (runtime.py:225)
  + `Qwen35BatchedResidentModel.__init__` (batched_runtime.py:327); `from_inner` detects from layers +
  the shim rides the default ⇒ serving auto-upgraded (no shim edit). RAN SOLO.

## File anchors
- `src/quanta/qwen35/moe.py` — `_routed_sparse` (62), `_swiglu_gate_up` (55, reuse), `_routed_dense`
  (81, the oracle), `_shared` (99, stays bf16), `qwen35_moe` (108, add dict auto-detect dispatch).
  **ADD** `_routed_sparse_packed`.
- `src/quanta/qwen35/model.py` — `Qwen35MoEModule` (28): `__init__` (32), `set_experts` (46, add a
  `set_experts_packed` sibling), `_params` (49), `__call__` (60).
- `src/quanta/qwen35/runtime.py` — `_load_block` (101; lines **165-171** = the `art.moe()`+`set_experts`
  to bypass under `packed_experts`), `_load_quant_triplet` (66, **reuse for the 3-D stacks**),
  `_block_arrays` (175, **must eval the packed triplets**), `Qwen35ResidentModel.__init__` (198, add
  `packed_experts`).
- `src/quanta/qwen35/artifact.py` — `raw` (124), `get` (63), `manifest` (59), `_load_quant_triplet`'s
  inputs; `moe` (171, the dequant path to bypass — optionally add a `moe_packed(i)` accessor returning
  the triplets, mirroring `moe`, for a cleaner loader).
- `src/quanta/qwen35/batched_runtime.py` — `Qwen35BatchedResidentModel.__init__` + `from_inner`
  (thread `packed_experts`); batched MoE call (286, UNCHANGED — rides `blk.mlp`).
- `src/quanta/shim/omlx.py` — Qwen35 resident/batched construction (thread `packed_experts`).
- **Template:** `src/quanta/dsv4/moe.py` — `_swiglu_stack_packed` (93), `_qmm` convention (114-124),
  `dsv4_moe` dict auto-detect (171); `src/quanta/dsv4/runtime.py:85` (the `packed_experts` flag).

## Gates
```bash
uv run --with numpy python -m parity.qwen35_forward_test            # M1 (packed-experts==bf16, +mixer) / M2
uv run --with numpy python -m parity.qwen35_batched_test            # regression
uv run --with numpy python -m parity.qwen35_batched_loopkill_test   # regression (loop-kill unaffected)
uv run python -m parity.qwen35_batched_bench 32                     # M3 (SOLO GPU): resident GiB + greedy + ppl
# before each commit: pytest tests/ -q · ruff check src tests · python -m compileall -q src tests · uv lock --check · git diff --check
```

## Kick-off prompt for the next agent
```
quanta — qwen35 routed-expert gather_qmm packing: keep the int4 routed experts PACKED and dispatch via
mx.gather_qmm instead of dequantizing them to bf16 + mx.gather_mm. The big memory lever: ~79 -> ~30 GiB
resident (the Qwen3.6 option-B bench peaks at 79.3 GiB for a 21 GiB int4 artifact because the experts
are dequantized). This completes the runtime's own documented design (src/quanta/qwen35/moe.py:11-12:
"the resident runtime swaps gather_mm -> gather_qmm over the packed stacks").

Read first, in order: CLAUDE.md (rules + quant policy: shared expert is bf16, never quantized);
MEMORY.md + memory/project_qwen35_experts.md (cross-session memory); PLAN_qwen35_experts.md (full
design, milestones M1-M3, the gather_qmm call convention, file anchors, gates); PLAN_153.md (the just-
finished option-B packed-MIXER work — the _packed_linear / _load_quant_triplet machinery you reuse);
then src/quanta/dsv4/moe.py (_swiglu_stack_packed / dsv4_moe dict auto-detect — the EXACT template,
but Qwen35 is SIMPLER: plain affine int4, NO awq_scale, and a FUSED gate_up = 2 gather_qmm calls not 3).

Key facts: gather_qmm is a per-(token,slot) M=1 matvec -> batch-invariant -> does NOT revive the #153
batch-M reorder option B fixed; the "MoE is exempt, do NOT touch" finding STAYS true (you preserve the
matvec structure, only swap the kernel). Packed vs bf16-dequant are the SAME int4 codes -> greedy-exact
(~1e-6), NOT a quant change. New flag packed_experts (default False, rule 4), threaded like option B's
`packed`, INDEPENDENT of it. Shared expert + router gate stay bf16. Reuse _load_quant_triplet for the
3-D stacks. Update _block_arrays so the packed triplets are eval'd/pinned.

Start with M1 (packed routed path in moe.py + Qwen35MoEModule + the model-free packed==dequant gate).
Then M2 (wire packed_experts through _load_block + the runtimes + forward parity + regression), then M3
(SOLO-GPU real-model gate: resident ~30 GiB + greedy-exact + teacher-forced ppl unchanged -> graduate).

Cadence (STANDING — do not violate): single linear thread, NO subagents/workflows. Implement -> gate
green -> commit each milestone (named files only, never `git add -A`; trailer
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`; no push; no hook skip; commits
land on main) -> STOP for the user to compact before the next milestone. RUN ONLY ONE MODEL AT A TIME
(M3 only; OOM-reboot hazard). Never delete ~/models/Kimi-K2.6. No mlx-lm/torch on the runtime hot path.

Start with M1.
```

---

## Out of scope here (full project map)
- **8-bit KV** is already implemented + default-ON (`KVCache(quantized=True)`, `make_caches`,
  runtime.py:226) for the 10 GQA layers; the 30 GDN layers carry an O(1) recurrent state (no growing
  KV). It is a **long-context** lever, orthogonal to the resident-weight 79 GiB — not part of this task.
- **DSV4 #153 core (paged-batched latent) M2–M4 still remain** — see `PLAN_153.md`. Separate thread.
