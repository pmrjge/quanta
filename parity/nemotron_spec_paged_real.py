"""Real-weight gate: Nemotron tree-spec over a PAGED triple == over the discrete triple (#158-160 M3).

The model-free pilot (``parity/nemotron_spec_paged_test.py``) gates the spec loop's paged-lifecycle
handling with a stub; this gates it on the int4-g64 resident Nemotron-Super (68 GiB, 8 GQA + 80 Mamba)
— the weight-level proof that routing a request's tree-spec through a PAGED ``(caches, ssm, conv)``
triple (KV in the shared block pool; ``replicate_state`` forks the sequence COW + rebinds the per-layer
views) is equivalent to the discrete per-stream path, and therefore lossless vs greedy.

  * **discrete** — ``spec_generate_tree(...)`` (default ``make_state=None`` → discrete KV caches).
  * **paged**    — ``spec_generate_tree(..., make_state=<paged factory>)`` building a real paged triple
    over a shared ``PagedKVCacheManager`` (the serving wiring of M3).

Criterion: bit-identical token streams, OR ``argmax_match >= 0.99`` — the SAME tolerance
``parity/nemotron_batched_tree_verify_real.py`` uses, because on a **bf16 Mamba hybrid** the verify
forward differs from a T=1 step by ~1 bf16 ULP (the settled M2/M3 finding) and a single near-tie can
flip then chaos-diverge two equally-valid greedy trajectories. The losslessness of *discrete* tree-spec
vs greedy is owned by that existing gate + the lossless contract; this adds the paged == discrete leg.

The artifact ships NO ``mtp.*`` keys (Super), so the MTP is random-init — FINE for parity (verify
arbitrates against the main-model greedy regardless of drafter quality; only the accept rate is the
~1/W floor). Per the settled MTP-M4 finding, tree-spec on this hybrid is a <1x B=1 lever — this gate
proves the CAPABILITY (paged tree-spec is lossless), not a speed win.

ORCHESTRATOR runs this SOLO — loads ~68 GiB resident Nemotron-Super. Run only with the memory free.

    uv run --with tokenizers python -m parity.nemotron_spec_paged_real
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.nemotron.batched_runtime import NemotronBatchedResidentModel
from quanta.nemotron.mtp import NemotronMTP
from quanta.nemotron.spec import spec_generate_tree
from quanta.nemotron.tokenizer import NemotronTokenizer
from quanta.paged import PagedKVCacheManager

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-quanta_int4g64"
PROMPT = (
    "Write a short paragraph explaining how the invention of the printing press changed the "
    "spread of scientific knowledge in early modern Europe. Keep it to two or three sentences."
)
PROMPT_T_MAX = 64
MAX_NEW = 32
WIDTH = 2
DEPTH = 2
BLOCK = 16


def _paged_make_state(model: NemotronBatchedResidentModel):
    """make_state factory backed by ONE shared PagedKVCacheManager (the serving wiring): each spec run
    forks a fresh sequence and builds a paged ``(caches, ssm, conv)`` triple via make_paged_state."""
    spec = model.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=4096,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="nemo-spec-paged")

    def make_state(*, max_rollback: int = 1):
        del max_rollback                                  # paged truncate handles rollback (no ring)
        return model.make_paged_state(mgr, mgr.new_sequence())

    return make_state, mgr


def main() -> None:
    print(f"[spec-paged-real] loading {ART}", flush=True)
    mx.set_wired_limit(int(120 * 1024**3))
    t0 = time.perf_counter()
    model = NemotronBatchedResidentModel(ART)
    print(f"[spec-paged-real] resident in {time.perf_counter() - t0:.1f}s "
          f"({model.cfg.num_hidden_layers} layers, attn_paged_layers={model.paged_kv_spec['n_layers']})",
          flush=True)

    mtp = NemotronMTP(model.cfg)                           # random-init (no mtp.* keys) — fine for parity
    embed, head = model.embed_w, model.lm_head_w
    tok = NemotronTokenizer(ART)
    ids = tok.encode(PROMPT, add_bos=False)[:PROMPT_T_MAX]
    print(f"[spec-paged-real] prompt {len(ids)} tok; mtp=random-init (parity is drafter-independent)",
          flush=True)

    # --- discrete tree-spec (default make_state=None) ---
    mx.random.seed(0)
    d0 = time.perf_counter()
    disc, disc_st = spec_generate_tree(model, mtp, embed, head, ids,
                                       width=WIDTH, depth=DEPTH, max_new=MAX_NEW)
    print(f"[spec-paged-real] discrete tree-spec: {len(disc)} tok in {time.perf_counter() - d0:.1f}s "
          f"mean_accept={disc_st['mean_accept']:.2f} rounds={disc_st['rounds']}", flush=True)

    mx.clear_cache()

    # --- paged tree-spec (make_state -> paged triple) ---
    make_state, mgr = _paged_make_state(model)
    mx.random.seed(0)
    p0 = time.perf_counter()
    paged, paged_st = spec_generate_tree(model, mtp, embed, head, ids,
                                         width=WIDTH, depth=DEPTH, max_new=MAX_NEW, make_state=make_state)
    print(f"[spec-paged-real] paged tree-spec   : {len(paged)} tok in {time.perf_counter() - p0:.1f}s "
          f"mean_accept={paged_st['mean_accept']:.2f} rounds={paged_st['rounds']} "
          f"(mgr={mgr.get_stats()})", flush=True)

    # --- compare (bit-identical preferred; argmax_match >= 0.99 tolerance for the bf16-ULP hybrid) ---
    n = min(len(disc), len(paged))
    if disc == paged:
        print(f"\nPASS — paged tree-spec BIT-IDENTICAL to discrete ({len(disc)} tokens, "
              f"mean_accept={paged_st['mean_accept']:.2f}) — #158-160 M3 real-weight gate", flush=True)
        return
    matched = sum(1 for i in range(n) if disc[i] == paged[i])
    ratio = matched / n if n else 0.0
    first = next((i for i in range(n) if disc[i] != paged[i]), -1)
    print(f"\n[spec-paged-real] not bit-identical: argmax_match={matched}/{n}={ratio:.4f}, "
          f"first diff at i={first}", flush=True)
    if first >= 0:
        lo, hi = max(0, first - 3), min(n, first + 4)
        print(f"    discrete[{lo}:{hi}] = {disc[lo:hi]}", flush=True)
        print(f"    paged   [{lo}:{hi}] = {paged[lo:hi]}", flush=True)
    if ratio < 0.99:
        raise SystemExit(f"FAIL — paged vs discrete argmax_match {ratio:.4f} < 0.99 (not a bf16-ULP near-tie)")
    print(f"\nPASS — paged tree-spec == discrete within the bf16-ULP hybrid tolerance "
          f"(argmax_match {ratio:.4f} >= 0.99, first divergence a documented near-tie) — #158-160 M3",
          flush=True)


if __name__ == "__main__":
    main()
