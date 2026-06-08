"""Real-weight gate: DSV4 tree-spec over a PAGED cache == over the discrete cache == greedy (#158-160 M1).

The model-free pilot (``parity/dsv4_spec_paged_test.py``) gates the spec loop's cache-protocol
handling with a stub model; this gates it on the int4-g64 resident DSV4 + native MTP head — the
weight-level proof that routing a request's tree-spec through a
:class:`~quanta.dsv4.decode.PagedDSV4Cache` (latent in the shared block pool, ``replicate(B)`` forks
the sequence COW) is **bit-identical** to the discrete per-stream path, and therefore lossless vs
plain greedy decode.

  * **discrete** — ``spec_generate_tree(...)`` (default ``make_state=None`` → discrete ``DSV4Cache``).
  * **paged**    — ``spec_generate_tree(..., make_state=<paged factory>)`` building a real
    ``PagedDSV4Cache`` over a shared ``PagedKVCacheManager`` (the serving wiring of M1).
  * **greedy**   — plain argmax decode through the batched runtime (the losslessness anchor).

PASS ⟺ all three token streams are identical (max_new=32). The existing
``parity/dsv4_batched_tree_verify_real.py`` already gates discrete tree-spec == greedy; this adds the
paged == discrete leg, closing tree-spec-over-paged on real weights for DSV4 (the pure-attention
keeper where tree-spec is a real B=1 latency lever).

ORCHESTRATOR runs this SOLO — loads ~180 GiB resident DSV4 + native MTP. Run only with the memory free.

    uv run --with tokenizers python -m parity.dsv4_spec_paged_real
"""

from __future__ import annotations

import time

import mlx.core as mx

from quanta.dsv4.batched_runtime import DSV4BatchedResidentModel
from quanta.dsv4.runtime import DSV4ResidentModel
from quanta.dsv4.spec import spec_generate_tree
from quanta.dsv4.tokenizer import DeepSeekV4Tokenizer
from quanta.paged import PagedKVCacheManager

ART = "/Users/pmrj/models/DeepSeek-V4-Flash-quanta_int4g64"
PROMPT = (
    "Write a short paragraph explaining how the invention of the printing press changed the "
    "spread of scientific knowledge in early modern Europe. Keep it to two or three sentences."
)
PROMPT_T_MAX = 64
MAX_NEW = 32
WIDTH = 2
DEPTH = 2
BLOCK = 16          # paged block size (latent blocks); 4 prompt blocks at PROMPT_T_MAX=64


def _argmax(row: mx.array) -> int:
    return int(mx.argmax(row).item())


def _paged_make_state(rt: DSV4BatchedResidentModel):
    """A make_state factory backed by ONE shared PagedKVCacheManager (the serving wiring): each spec
    run forks a fresh sequence and builds a paged cache sized for the tree's rollback depth."""
    spec = rt.paged_kv_spec
    mgr = PagedKVCacheManager(num_layers=spec["n_layers"], block_size=BLOCK, max_blocks=4096,
                              group_size=spec["group_size"], bits=spec["bits"],
                              quantized=spec["quantized"], model_name="dsv4-spec-paged",
                              single_stream=spec.get("single_stream", False))

    def make_state(*, max_rollback: int = 1):
        return rt.make_paged_state(mgr, mgr.new_sequence(), max_rollback=max_rollback)

    return make_state, mgr


def _greedy(rt: DSV4BatchedResidentModel, ids: list[int], n: int) -> list[int]:
    """Plain greedy decode through the batched runtime (prefill + argmax steps) — the loss anchor."""
    cache = rt.make_cache()
    row = rt.prefill(mx.array(ids), cache)[0, -1]
    mx.eval(row)
    out = [_argmax(row)]
    for _ in range(n - 1):
        row = rt.step_batch([mx.array([out[-1]])], [cache], [cache.offset])[0][0, -1]
        mx.eval(row)
        out.append(_argmax(row))
    return out


def main() -> None:
    print(f"[spec-paged-real] loading {ART}", flush=True)
    t0 = time.perf_counter()
    inner = DSV4ResidentModel(ART)
    print(f"[spec-paged-real] resident in {time.perf_counter() - t0:.1f}s "
          f"({inner.cfg.num_hidden_layers} layers, packed={inner.packed_experts}, "
          f"mtp={'on' if inner.mtp is not None else 'OFF — FATAL'})", flush=True)
    if inner.mtp is None:
        raise RuntimeError("DSV4 MTP not loaded — artifact missing mtp.0.* keys")

    model = DSV4BatchedResidentModel.from_inner(inner)
    tok = DeepSeekV4Tokenizer.from_pretrained(ART)
    ids = tok.encode(PROMPT, add_bos=True)[:PROMPT_T_MAX]
    print(f"[spec-paged-real] prompt {len(ids)} tok (BOS-first={ids[0] == tok.bos_id})", flush=True)

    mtp, embed, head = inner.mtp, inner.embed_w, inner.lm_head_w

    # --- greedy anchor (losslessness reference) ---
    g0 = time.perf_counter()
    greedy = _greedy(model, ids, MAX_NEW)
    print(f"[spec-paged-real] greedy: {len(greedy)} tok in {time.perf_counter() - g0:.1f}s", flush=True)

    # --- discrete tree-spec (default make_state=None) ---
    mx.random.seed(0)
    d0 = time.perf_counter()
    disc_tokens, disc_stats = spec_generate_tree(model, mtp, embed, head, ids,
                                                 width=WIDTH, depth=DEPTH, max_new=MAX_NEW)
    print(f"[spec-paged-real] discrete tree-spec: {len(disc_tokens)} tok in "
          f"{time.perf_counter() - d0:.1f}s mean_accept={disc_stats['mean_accept']:.2f} "
          f"rounds={disc_stats['rounds']}", flush=True)

    # --- paged tree-spec (make_state → PagedDSV4Cache) ---
    make_state, mgr = _paged_make_state(model)
    mx.random.seed(0)
    p0 = time.perf_counter()
    paged_tokens, paged_stats = spec_generate_tree(model, mtp, embed, head, ids,
                                                   width=WIDTH, depth=DEPTH, max_new=MAX_NEW,
                                                   make_state=make_state)
    print(f"[spec-paged-real] paged tree-spec: {len(paged_tokens)} tok in "
          f"{time.perf_counter() - p0:.1f}s mean_accept={paged_stats['mean_accept']:.2f} "
          f"rounds={paged_stats['rounds']} (mgr stats={mgr.get_stats()})", flush=True)

    # --- assertions ---
    n = min(len(greedy), len(disc_tokens), len(paged_tokens))
    pd = disc_tokens[:n] == paged_tokens[:n]
    dg = disc_tokens[:n] == greedy[:n]
    pg = paged_tokens[:n] == greedy[:n]
    print(f"\n[spec-paged-real] paged==discrete : {pd}", flush=True)
    print(f"[spec-paged-real] discrete==greedy: {dg}", flush=True)
    print(f"[spec-paged-real] paged==greedy   : {pg}", flush=True)
    if not (pd and pg):
        # locate first divergence for a precise failure
        for i in range(n):
            if paged_tokens[i] != disc_tokens[i] or paged_tokens[i] != greedy[i]:
                raise SystemExit(
                    f"FAIL — divergence at pos {i}: paged={paged_tokens[i]} discrete={disc_tokens[i]} "
                    f"greedy={greedy[i]}")
        raise SystemExit("FAIL — length mismatch among paged/discrete/greedy streams")
    if disc_stats["mean_accept"] != paged_stats["mean_accept"]:
        raise SystemExit(f"FAIL — mean_accept diverged: discrete={disc_stats['mean_accept']} "
                         f"paged={paged_stats['mean_accept']} (same model+MTP must give same accepts)")
    print(f"\nPASS — DSV4 tree-spec over a paged cache is bit-identical to discrete AND greedy "
          f"({n} tokens, mean_accept={paged_stats['mean_accept']:.2f}) — #158-160 M1 real-weight gate")


if __name__ == "__main__":
    main()
