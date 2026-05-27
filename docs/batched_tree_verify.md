# Batched verify for tree-shaped spec decoding

Status: **design only** (no code). Follow-on to #157 (W-parallel chain-verify tree drafting,
landed for DSV4 / Qwen3.5 / Nemotron in `700adb3` / `99040c6` / `ff202da`).

## TL;DR

Today's `spec_generate_tree(model, mtp, ..., width, depth, ...)` enumerates `W^D` root-to-leaf
draft paths and chain-verifies each *sequentially* — i.e., runs `W^D` separate single-stream
`model([cur, *path_drafts])` forwards per round, with the decode cache rolled back between paths.
At `W=2, D=2` that's **5 forwards per round** vs `spec_generate_k(k=2)`'s 1 forward — making the
W-parallel form ~2–3× more expensive per accepted token than chained spec, even though its
*acceptance rate* per round is higher (the tree explores more candidates).

**Batched verify** collapses those `W^D` sequential verify forwards into one **`B = W^D`-wide
batched forward**, amortizing the dominant cost — the **routed-expert weight reads** — across all
`B` candidate paths in a single MoE pass. Expected economics at `W=2, D=2`:

| Path                              | Forwards/round | Expert-stack reads/round | mean_accept (best case) | Cost / accepted token |
|-----------------------------------|----------------|--------------------------|-------------------------|-----------------------|
| `spec_generate_k(k=2)` (chained)  | 1              | depth+1 = 3              | 2.78                    | **1.08** (baseline)   |
| `spec_generate_tree` sequential   | W^D + 1 = 5    | (W^D + 1) × (depth+1) = 15 | 3.00 (verify-arbitrated) | **5.00**             |
| `spec_generate_tree` BATCHED      | (depth+1) + 1 = 4 | (depth+1) + 1 = 4     | 3.00 (same)             | **1.33**              |

So batched verify makes tree drafting roughly **on par with chained k=2 on cost**, with the same
or better mean_accept depending on MTP head quality. *Above* `W=2 D=2` (e.g. `W=4 D=2` → `B=16`,
`W=2 D=3` → `B=8`) the batched-vs-sequential gap widens; the tree shape becomes a real lever.

The MTP head + main-model arbitrate every emitted token, so the **output is bit-identical to
plain greedy** in both forms (rule 4 / losslessness preserved).

## Background — what the sequential form does today

Per spec round, the W-parallel chain-verify form is (paraphrased from `dsv4/spec.py` /
`qwen35/spec.py` / `nemotron/spec.py`):

```text
prefix state  =  cache at offset (q + 1)   # prompt + already-committed tokens
build_tree(mtp, ...)  -> [tokens, parents] of size 1 + sum(W^d for d in 1..D)
paths         =  enumerate_paths(parents, tokens)   # W^D paths of length D
for path_drafts in paths:
    vlog, vcaps = model([cur, *path_drafts], offset=q+1)  # advances cache by depth+1
    # … pick longest matching prefix, track best …
    cache.truncate(q+1)                               # roll back to round start
commit_seq = [cur, *best_path[:best_j]]
model(commit_seq, offset=q+1)                         # advance cache by 1 + best_j
```

The hot cost is the `for path_drafts in paths` loop: `W^D` independent single-stream forwards
through the full stack (43 layers for DSV4, 60 for Qwen3.5, 88 for Nemotron). Each forward reads
the routed-expert stack from RAM/MoE residency for `depth+1` tokens (`top_k` experts per token).

## The batched form

The hot loop becomes:

```text
prefix state  =  cache at offset (q + 1)   # SAME as today
paths         =  enumerate_paths(...)      # SAME as today  -> B = W^D paths
# (1) replicate the cache state B times
replicas      =  cache.replicate(B)        # B parallel single-stream caches, each == prefix state
# (2) batched verify
for t in 0 .. depth:
    per_stream_tokens = [verify_seq_b[t] for b in 0..B-1]
    # verify_seq_b is [cur, *paths[b]] of length depth+1; verify_seq_b[0] == cur for all b
    logits_t = batched_decode_step(replicas, per_stream_tokens, offset=q+1+t)
    # logits_t : [B, vocab] — per-stream argmax at position t
# (3) interpret outputs and pick best path
for b in 0..B-1:
    j_b = longest prefix where paths[b][i] == argmax(logits_{i+1}[b])
                                    for i in 0..depth-1
    bonus_b = argmax(logits_{j_b+1}[b])
best_b = argmax_b j_b   (ties: BFS-leftmost — matches sequential form)
# (4) commit: SAME as today, on the original (un-replicated) cache
commit_seq = [cur, *paths[best_b][:j_best]]
model(commit_seq, caches=cache, offset=q+1)   # single-stream advance
# replicas are dropped
```

Key invariants:
- The original `cache` is **read-only** during the verify (no truncate, no mutation). All
  per-path divergence happens in the **replicas**, which are discarded after pick.
- The commit re-feeds the accepted prefix through the original (un-replicated) single-stream
  cache, exactly as today. This is **bit-identical** to the sequential form's commit-forward.
- Output is bit-identical to the sequential form (and therefore to plain greedy): the same
  paths are tried, the same longest-matching-prefix selection runs, and the same commit-forward
  advances the cache.

## The hot piece: `cache.replicate(B)`

The single hard new piece. Each model's cache holds a mix of:
- **non-recurrent KV storage** (full-attention layers, DSV4 dense/compressed/indexed regimes,
  Qwen3.5 GQA, Nemotron GQA) — per-token `[head_dim, num_heads]` slices up to current length;
- **recurrent state** (Qwen3.5 GDN: `[1, hv, dk, dv]`; Nemotron Mamba-2: SSM `[1, n_groups,
  state_size, head_dim]` + conv `[1, k-1, conv_dim]`);
- **bounded auxiliary windows** (DSV4 compressor pooling-remainder ring, Lightning-Indexer state).

`replicate(B)` is the operation that turns a B=1 instance of any of these into a B-wide one
where row `b` initially equals the original. After the batched verify each row diverges; only
the picked row's state would have value, and the existing single-stream cache already has it
(because the commit-forward re-feeds the accepted prefix). So **the B-wide replica is throw-
away** — we never persist any row of it past the round.

Per-regime replication:

| Cache piece                          | Replication             | Notes                                          |
|--------------------------------------|-------------------------|------------------------------------------------|
| KV store `[L, H, head_dim]`          | `mx.broadcast_to`       | The B replicas share *read-only* the prefix    |
|                                      | along a new B-leading   | KV; they each ALSO own a private suffix slot   |
|                                      | axis or per-row copy    | of size `depth+1` they write into during the   |
|                                      |                         | batched verify. Reuse: prefix tile             |
|                                      |                         | broadcast + per-row append-buffer of size      |
|                                      |                         | `depth+1`.                                     |
| Compressor pooling state             | `mx.broadcast_to` →     | Per-row evolves independently as suffix tokens |
| (DSV4 ratio-4 / ratio-128 layers)    | row-private after first | are pooled in. Bounded ring (max_rollback+1    |
|                                      | write                   | tokens) → low memory cost per row.             |
| Lightning Indexer state (DSV4)       | per-row copy            | `[1, index_n_heads, index_head_dim]` × layers  |
|                                      |                         | — small.                                       |
| GDN recurrent state (Qwen3.5)        | `mx.broadcast_to` →     | `[1, hv, dk, dv]` → `[B, hv, dk, dv]`. Per-row |
|                                      | row-private after first | evolves under suffix tokens.                   |
|                                      | write                   |                                                |
| GDN conv state (Qwen3.5)             | `mx.broadcast_to`       | `[1, k-1, conv_dim]` → `[B, k-1, conv_dim]`    |
| Mamba SSM state (Nemotron)           | `mx.broadcast_to` →     | `[1, n_groups, state_size, head_dim]` →        |
|                                      | row-private after first | `[B, ...]`                                     |
|                                      | write                   |                                                |
| Mamba conv state (Nemotron)          | `mx.broadcast_to`       | `[1, k-1, conv_dim]` → `[B, k-1, conv_dim]`    |

Implementation hint: **lazy materialization**. `replicate(B)` doesn't have to deep-copy; it can
return a wrapper that `broadcast_to`s on the first write per row. MLX's lazy graph + the small
suffix length (`depth+1`) keeps the materialization cheap. A first cut can deep-copy and we
fold lazy materialization in only if benchmark says it matters.

Each model's `decode.py` gets a `replicate(b)` method on its `*Cache` class (and helpers for
the per-layer states they hold). Memory cost during the round:
- DSV4 (1M ctx, head_dim=128, H=128, 43 layers, ratio=4 avg): prefix KV ~10 MB read-only-tiled
  + `B * depth+1` write slots per layer ~ KB. **Trivial** vs the 169 GB resident model.
- Qwen3.5 (1M ctx, full-attn-15-layers, GDN-45-layers): prefix KV on 15 layers + GDN state
  `B × hv × dk × dv` per layer × 45 = small ([B, 32, 256, 256] = ~67 MB for B=16 worst case,
  one layer, bf16 = ~33 MB; × 45 = 1.5 GB). Bounded.
- Nemotron (1M ctx, 8 GQA + 80 Mamba): prefix GQA KV (8 layers, small) + SSM `[B, n_groups,
  state_size, head_dim]` per layer × 80. With `B=16` and Nemotron's state_size=128 → `[16, 8,
  128, 64]` × bf16 ≈ 4 MB × 80 = ~320 MB. Bounded.

All three are **safe under the 490 GiB working-set ceiling** even at `W=4 D=2 → B=16`.

## Batched-verify helper: `batch_verify(model, cache, cur, paths, *, depth, last_layer)`

A new helper per-model. Returns `(best_j, best_bonus, best_path, best_hidden)`:

```python
def batch_verify(model, cache, cur, paths, *, depth, last_layer):
    B = len(paths)
    verify_seqs = [[cur, *p] for p in paths]    # B × (depth+1)
    replicas = cache.replicate(B)                # B-wide cache, all rows == prefix
    # Loop the depth+1 positions through batched_decode_step.
    # batched_decode_step expects a per-stream token + an offsets list; all rows share offset
    # = q+1+t at step t (they all started from the same prefix), so offsets are uniform.
    per_pos_logits = []                          # depth+1 × [B, vocab]
    per_pos_hidden = []                          # depth+1 × [B, hc, dim] (DSV4) or [B, hidden]
    for t in range(depth + 1):
        toks = [seq[t] for seq in verify_seqs]
        # SHARED forward; returns B per-stream logits (and optional hidden capture).
        logits_t, hidden_t = model.batch_step(
            tokens=toks, caches=replicas, offset=q + 1 + t, capture_layer=last_layer,
        )
        per_pos_logits.append(logits_t)
        per_pos_hidden.append(hidden_t)
    # interpret outputs row-by-row (cheap Python loop over B ≤ 64)
    best_j, best_b = -1, 0
    for b in range(B):
        bp = [int(mx.argmax(per_pos_logits[t][b]).item()) for t in range(depth + 1)]
        j = 0
        while j < depth and verify_seqs[b][1 + j] == bp[j]:
            j += 1
        if j > best_j:
            best_j, best_b = j, b
    best_path  = paths[best_b]
    best_bonus = int(mx.argmax(per_pos_logits[best_j][best_b]).item())
    best_hidden = per_pos_hidden[best_j][best_b][None, None]   # match sequential's hidden shape
    return best_j, best_bonus, best_path, best_hidden
```

`model.batch_step(tokens, caches=, offset=, capture_layer=)` is the *single per-model* API we
add. Internally it reuses the existing `batched_decode_step` machinery from
`*_batched_runtime.py` — the same `[B,1,hidden]` → batched-MoE → split-and-attn pattern that
already exists for agentic serving. The new contract is just:
- All `B` streams share the *same* offset (in agentic serving they have ragged offsets).
- It optionally returns the captured residual at one specific layer (for MTP feature).

The implementation is a thin shim around `batched_decode_step`; no new attention / MoE code.

## Integration into `spec_generate_tree`

Each model's `spec_generate_tree` gets a new opt-in arg `batched: bool = False`. When
`batched=True`:

```python
# replaces the for-path-in-paths sequential loop:
best_j, best_bonus, best_path, best_hidden = batch_verify(
    model, caches, cur, paths, depth=depth, last_layer=last,
)
# … commit step UNCHANGED from sequential form …
```

Default is `False` so the proven sequential path is the documented default. Once parity holds
end-to-end on real models, we flip the default to `True`.

## Parity gates (model-free; safe alongside GPU jobs)

Per model, one new file in `parity/`:

- `parity/dsv4_batched_tree_verify_test.py`
- `parity/qwen35_batched_tree_verify_test.py`
- `parity/nemotron_batched_tree_verify_test.py`

Each asserts:
1. **batched == sequential** — `spec_generate_tree(model, ..., batched=True)` returns the
   bit-identical token list to `spec_generate_tree(model, ..., batched=False)` for both a
   perfect-leftmost and a wrong-all stub MTP. `mean_accept`, `rounds`, `max_accept` all match.
2. **`B=1` short-circuit** — `width=1` (i.e. a chain) bypasses batching and matches
   `spec_generate_k(k=depth)`.
3. **Cache invariance** — after batched verify completes, the original (un-replicated) cache
   is at the same offset and contains the same state as if `spec_generate_tree(batched=False)`
   had run on it.
4. **Replicate fidelity** — `cache.replicate(B)` immediately followed by reading row `b` of
   each replica gives the original cache's content (a structural sanity test of the new
   replication API).

All tests use the same stub patterns as `parity/{dsv4,qwen35,nemotron}_tree_spec_test.py`
(deterministic next-token chain, `[1, hc, dim]` capture, B-wide stubs of the
`batched_decode_step` API) — no checkpoint, no GPU, a few KB of tensors per assertion.

## End-to-end gates (deferred to GPU/memory-available session)

Per model, defer until GPU is free:
- **Real-model parity** — run `spec_generate_tree(batched=True)` vs `=False` on the resident
  baked model with the real MTP head + a real prompt; assert tokens match bit-for-bit.
- **Throughput bench** — measure tok/s for `spec_generate_k(k=2)` vs `spec_generate_tree(W=2,
  D=2, batched=False)` vs `spec_generate_tree(W=2, D=2, batched=True)`; expect batched to
  reach ≥ chained's throughput while keeping the higher acceptance rate.

These live as docstring-only entries in the parity files (per the same pattern as
`parity/dsv4_int4_ppl.py` / `parity/eagle_spec_longprompt_sweep.py`).

## Implementation plan (5 commits)

1. **Shared docs + skeleton** — this file (already drafted), plus the `batch_step` Protocol
   in `quanta.spec.batched` so all three models agree on the contract.

2. **DSV4** (pilot — pure attention, three regimes is the messy case):
   - `src/quanta/dsv4/decode.py` — `DSV4Cache.replicate(B)` + per-layer-state replication
     across KV / ckv / compressor remainder / indexer state.
   - `src/quanta/dsv4/batched_runtime.py` — `DSV4BatchedResidentModel.batch_step(...)`
     wrapper around the existing `batched_decode_step`.
   - `src/quanta/dsv4/spec.py` — `batch_verify` helper + `batched=` arg on `spec_generate_tree`.
   - `parity/dsv4_batched_tree_verify_test.py` — model-free parity gate.

3. **Qwen3.5** — same structure; hybrid recurrent state (GDN) + GQA full-attention.

4. **Nemotron** — same structure; hybrid recurrent state (Mamba-2 SSM + conv) + GQA.

5. **Flip defaults + bench** — once all three are parity-clean, set `batched=True` as the
   `spec_generate_tree` default. Drop docstring notes for the deferred GPU benchmarks.

Each commit is independently mergeable (per-model files are disjoint). The shared `batch_step`
Protocol in commit 1 keeps the per-model contracts uniform.

## Open questions (answer before commit 2)

1. **Lazy vs eager replication.** Lazy (broadcast + materialize-on-write) is cleaner and
   probably faster but harder to test. Recommend eager for the first cut; switch lazy if
   benchmark says it matters. *Decision:* eager.

2. **`B=W^D` ceiling.** At `W=8 D=2 → B=64` the replicate cost on a Mamba SSM `[B, n_groups,
   state_size, head_dim]` becomes non-trivial (~1.3 GB at `state_size=128`). Cap `B ≤ 32` and
   fall back to sequential above? *Decision:* yes, with a documented fallback at
   `W^D > batched_verify_max_b`.

3. **Mask vs replicate for pure attention?** DSV4 *could* in principle do single-forward tree-
   mask form for its attention regimes (no recurrent state), bypassing replication entirely.
   That would be true EAGLE-2, and would be faster than batched-replicate. But it requires
   per-regime mask plumbing (dense / compressed / indexed), and Qwen3.5/Nemotron need the
   replicate path anyway. *Decision:* defer the mask-form for DSV4 to a separate task —
   uniform replication across all three first.

4. **Interaction with int8 KV cache.** Both DSV4 and the others ship int8 KV by default
   (#122 / #123). Replication of an int8 KV cache is a copy of the int8 codes + scales — no
   re-quantization needed. *Decision:* `replicate(B)` returns same-mode caches; no flag.

## Risks

- **Replication bugs.** The replicate API is new code per model. A bug that mutates the
  prefix instead of the replica would silently corrupt the next round. The parity test "Cache
  invariance" assertion catches this; commit 2 (DSV4 pilot) is where we de-risk it.
- **Batched attention numerical drift.** SDPA reorders reductions; B-wide batched SDPA may
  not be bit-identical to single-stream SDPA. Recommend testing with `argmax_match >= 0.99`
  rather than bit-identity for real-model parity; the model-free stubs still produce
  bit-identical outputs (the stub doesn't go through SDPA).
- **MoE batched dispatch drift.** Same story — `gather_qmm` with sorted dispatch may reorder.
  Same tolerance.

## Non-goals

- *True* B-stream agentic serving with tree spec — this is single-stream tree spec with a
  B-wide *internal* batch for verify. Cross-stream batching with paged caches is #152.
- Cross-model abstraction beyond the `batch_step` Protocol. Each model keeps its own
  per-regime cache replication; abstracting that further is premature.
- Changing the W-parallel chain-verify contract. The output remains bit-identical to plain
  greedy regardless of `batched=True/False`.
