---
name: project_qwen35_experts
description: keep Qwen3.6 routed experts packed int4 + gather_qmm (not dequant-bf16 + gather_mm); DONE — 63→20 GiB resident; M1✅a6b3b49 M2✅d17882e M3✅f720fda (packed_experts default ON); handover PLAN_qwen35_experts.md
metadata:
  node_type: memory
  type: project
  originSessionId: ccd01e00-d722-4889-9328-5748d8571d41
---

**Qwen3.6 (qwen35) routed-expert `gather_qmm` packing — DONE: M1 ✅ (`a6b3b49`) + M2 ✅ (`d17882e`) + M3 ✅ (`f720fda`); packed_experts default ON.** The follow-up after
[[project_paged_batched_153]] option B. The option-B bench peaks **79.3 GiB for a 21 GiB int4-g64
artifact** because the runtime **dequantizes the routed experts to bf16** (`_load_block` →
`art.moe(i)` → `read`/`_dequant`, runtime.py:165-171 / artifact.py:171) and runs `mx.gather_mm` on
bf16 (`moe.py:_routed_sparse`:62). Keeping them packed int4 + `mx.gather_qmm` → **~79 → ~30 GiB**
resident (the experts dominate the artifact; mixer is already packed by option B).

**It's the runtime's OWN documented design** (`moe.py:11-12`: "the resident runtime swaps
`gather_mm` → `gather_qmm` over the packed stacks") — never realized; took the simplest-first dequant.

**Why it's safe + in-scope (NOT a re-open of the #153 batch-M bug):** `gather_qmm` is the same
per-(token,slot) **M=1 matvec** as `gather_mm` → **batch-invariant**, so it does NOT revive the
dense-bf16 batch-M reorder option B fixed; the "MoE is exempt, don't touch" finding STAYS true (only
the kernel swaps, the matvec structure is preserved). Packed vs bf16-dequant are the **same int4
codes** → **greedy-exact** (~1e-6, like option B's packed-vs-bf16 mixer), NOT a quant change. Teacher-
forced ppl can't distinguish them.

**Template = DSV4** (`dsv4/moe.py:_swiglu_stack_packed`:93 + `dsv4_moe` dict auto-detect:171;
`dsv4/runtime.py:85` flag). **Qwen35 is SIMPLER:** plain **affine** int4 (NO `awq_scale` division —
DSV4 has it, Qwen35 doesn't) + **fused** gate_up ⇒ **2** `gather_qmm` calls not 3. `gather_qmm` puts
the **activation first, weight second** (opposite of `gather_mm`); `transpose=True` over the
`[E,out,in]` stacks. Keep it **unsorted** for the parity gate (`sorted_indices` is an optional perf
lever). **Shared expert + router gate stay bf16** (CLAUDE.md: shared expert never quantized).

**Plan:** new flag `packed_experts` (default False, rule 4, INDEPENDENT of option B's already-graduated
`packed` mixer flag), threaded like `packed`. **M1 ✅ `a6b3b49`** — `moe.py:_routed_sparse_packed`
(2 `gather_qmm` calls: fused gate_up then down; plain affine, NO awq; activation-FIRST arg order,
`transpose=True`, unsorted) + `qwen35_moe` dict auto-detect (`isinstance(p["experts_gate_up"],dict)`,
mirrors `dsv4_moe` routed_fn) + fail-loud rule-6 guard (`sparse=False`+dict rejected) +
`Qwen35MoEModule.set_experts_packed` + `parity/qwen35_forward_test._moe_packed_parity` (packed
gather_qmm == bf16-dequant gather_mm == dense oracle on SAME `mx.quantize` codes: **d_pack=1.8e-7,
d_dense=1.2e-7, d_mod=0**; f32 throughout ⇒ clean fused-vs-separate-dequant ULP, NOT a quant change;
`gather_qmm` empirically handles act≠scale dtype, output follows scale dtype, so NO activation cast —
unlike DSV4/Nemotron's hardcoded bf16). Gates green incl. batched+loopkill (bf16 path untouched).
**M2 ✅ `d17882e`** wired `packed_experts` (default False, rule 4, INDEPENDENT of option-B mixer
`packed`) through the loader + both runtimes — NO shim change (the shim sites ride the constructor
default, so M3 flipping it auto-upgrades serving): (a) **artifact.py** new `moe_packed(i)` (+ private
`_packed_triplet(base)`) — the memory-lean sibling of `moe()`: routed experts come back as affine
triplet dicts `{packed,scale,bias,group_size,bits}` held VERBATIM (no dequant), width from the
manifest (rule-6 fail-loud on non-`affine_packed`); router gate + shared expert stay bf16. (b)
**runtime.py** `_load_block(packed_experts=False)` branches `art.moe_packed`+`set_experts_packed` vs
`art.moe`+`set_experts` (shared/gate tail identical); **`_block_arrays` now dict-aware** — evals the
packed triplet COMPONENT arrays (`packed`/`scale`/`bias`), NOT the dict / int meta (rule-8 pinning);
`Qwen35ResidentModel.__init__(packed_experts=False)` threads+stores it. (c) **batched_runtime.py**
`__init__(packed_experts=False)` threads to the inner `Qwen35ResidentModel`; `from_inner` DETECTS it
from the passed layers (`isinstance(layers[0].mlp.experts_gate_up, dict)`) — the batched MoE rides
`blk.mlp(...)`→`qwen35_moe` auto-detect so gather_qmm comes free. Gate
`parity/qwen35_forward_test._packed_experts_forward_parity`: assembled `Qwen35Model` with
`set_experts_packed` == bf16-dequant reference on SAME codes, **greedy-exact |Δlogit|=1.07e-6** +
`_block_arrays` plumbing assert (every surfaced entry an `mx.array`); regressions green
(`qwen35_batched_test`, `qwen35_batched_loopkill_test` — bf16 path untouched). NOTE the runtime helper
is `_load_quant_triplet` (returns a 5-TUPLE, used by the mixer `_packed_linear`); the experts use the
new artifact `moe_packed`/`_packed_triplet` (returns the DICT) — both read the same `affine_packed`
siblings. **M3 ✅ `f720fda` (SOLO-GPU real-model gate, graduated; TASK COMPLETE).** New gate
`uv run python -m parity.qwen35_batched_bench experts` (added to the option-B bench): loads the real
int4-g64 bake TWICE (`packed_experts` False then True, ONE model resident at a time — `gc.collect`+
`clear_cache` between, 140 GiB wired limit), asserts (a) **resident 63.4→20.3 GiB (0.32×, −43 GiB)**
[the bf16-dequant RESIDENT is 63.4, NOT the option-B 79.3 which was the PEAK@B32 incl. transient],
(b) **greedy-exact** 48-tok trace (`_greedy_trace`, cached decode path), (c) **ppl unchanged**
2.3150→2.3204 (rel **0.24%**) + top-1 agree **0.9858** on REAL prose (`_teacher_forced` mirrors
parity/ppl.py: repo PROSE fixture + `Qwen35Tokenizer.from_pretrained`; Qwen3.5 has NO BOS so add_bos
is a no-op). **METHODOLOGY LESSON:** first attempt used SYNTHETIC ids (`_distinct_prompt`) → the model
is near-uniform there (NLL≈ln(vocab)), which AMPLIFIES bf16 rounding into |ΔNLL|=1.9e-2 = a FALSE fail.
CLAUDE.md's ppl arbiter is REAL prose (model confident → robust); switched to it → 0.24%. And
bf16-dequant is the LOSSIER path (it rounds each dequant weight to bf16 PRE-matmul; `gather_qmm` keeps
full precision in the fused dequant), so packed isn't drifting — greedy decode is bit-identical. →
**graduated `packed_experts=True` default** in `Qwen35ResidentModel.__init__` (runtime.py:225) +
`Qwen35BatchedResidentModel.__init__` (batched_runtime.py:327); `from_inner` detects from layers + the
shim rides the default ⇒ serving auto-upgraded (NO shim edit). Regressions green
(qwen35_forward_test/batched/loopkill — model-free, flip-unaffected).

Full handover (design, the exact `gather_qmm` snippet, file anchors, gates, kickoff prompt) in repo
**`PLAN_qwen35_experts.md`**. Cadence: single thread, NO subagents, commit each milestone → STOP to
compact; ONE model at a time (M3, OOM-reboot hazard). 8-bit KV is already default-ON (10 GQA layers;
GDN has O(1) state) and orthogonal to this — long-context lever, not the 79 GiB.
