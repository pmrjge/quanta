"""Nemotron-Ultra U4 / MTP-M2 — resident native-MTP spec-decode: lossless accept-rate gate.

Third milestone of the native-MTP self-speculative decode stream (#40). M0 gated the bf16 head's
*structural assembly*; M1 baked it into the int4-RTN **sidecar** + recon gate. M2 **wires the head
into the resident spec loop** and proves the decode is LOSSLESS on real weights:

* the int4-RTN **backbone** (:class:`NemotronResidentModel`, 306 GiB — routed experts packed int4 via
  ``mx.gather_qmm``, dense int8 ``nn.QuantizedLinear``, SSM core / norms / router / embed / head bf16)
  RAM-resident;
* the baked MTP **sidecar** loaded via :func:`quanta.nemotron.runtime.build_resident_mtp`
  (:class:`quanta.nemotron.mtp.NemotronMTP`), same per-tensor policy as the backbone;
* :func:`quanta.nemotron.spec.spec_generate_k` at ``k in {1, 2, 3}`` over a real prose prompt must be
  **GREEDY-EXACT** vs plain greedy decode (the main model verifies every draft, so any head quality is
  lossless — rule 4), with ``mean_accept`` + decode tok/s reported per ``k``.

What this newly gates vs M0/M1 (the genuinely new surface):

* ``build_resident_mtp`` — the resident loader (NemotronMTP from the sidecar ``mtp.*`` as int4-packed
  experts via gather_qmm + int8 dense + bf16 core), mirroring ``build_resident_block``.
* the resident spec-contract adapter on ``NemotronResidentModel`` (``make_caches`` / ``offset`` /
  ``truncate``).
* the **hybrid k=1 Mamba rollback** in ``spec_generate`` (snapshot/restore + ``[cur]`` re-run on a
  rejected draft — the ``(ssm, conv)`` recurrence can't be sliced); gated bit-exact model-free in
  ``nemotron_mtp_spec_test`` (gate 7) and e2e here.

**Bit-identical until a bf16 near-tie, then valid-but-divergent — and why.** On a bf16 Mamba hybrid
the spec VERIFY forward (T>1, the ``[cur, *drafts]`` window) and a plain decode forward (T=1) differ by
~1 bf16 ULP (measured below as ``path_ulp`` — chiefly the attention mask=None-vs-causal + the Mamba
recurrence; the runtime's own Mamba-mixer / batched-SDPA notes document this path-sensitivity class).
A fast spec MUST read accepted-token state through the T>1 verify (committing via T=1 would re-run
every token = no speedup), so its bonus/state tracks T=1 greedy until the ~1-ULP gap tips a **near-tie**
argmax. **Greedy decoding is chaotic** — a single flipped token makes the two (each still valid) greedy
trajectories diverge arbitrarily afterward, so "spec == T=1 greedy" is the wrong real-weight criterion
(CLAUDE.md: test with parity, *not* greedy generation). What IS verifiable: the streams are
**bit-identical up to the first divergence**, and that FIRST divergence (the only position whose prefix
still matches greedy's, so the comparison is valid) must be a **bf16 ULP near-tie** — the spec token is
greedy's runner-up (rank 2) with a top1-top2 margin within a few ``path_ulp``, scored on the SAME
per-token-step path greedy uses. A large-margin / low-rank first divergence would be a real logic error
and FAILS. (The structural accept/reject/rollback logic is separately gated **bit-exact** model-free in
``nemotron_mtp_spec_test`` gate 7.) One model resident at a time — **RUN SOLO** (~313 GiB wired, under
the 490.4 GiB ceiling).

    uv run --with tokenizers python -m parity.nemotron_ultra_mtp_resident_spec
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_ultra_ppl import LONG_PROSE
from quanta.nemotron.artifact import NemotronArtifact
from quanta.nemotron.runtime import NemotronResidentModel, build_resident_mtp
from quanta.nemotron.spec import spec_generate_k
from quanta.nemotron.tokenizer import NemotronTokenizer

ART = "/Users/pmrj/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-quanta_int4rtn_g64"
MTP_ART = ART + "_mtp"
N_PROMPT = 64      # prose prompt length (real Mamba prefill content)
N_GEN = 48         # tokens decoded by greedy AND each spec variant (eos_id=None -> fixed length)
K_VALUES = (1, 2, 3)
MARGIN_K = 4.0     # a divergence is a bf16 ULP near-tie iff its top1-top2 margin <= MARGIN_K * path_ulp


class _EagerMain:
    """Force eager (``compiled=False``) main-model forwards — the naive reference path (rule 4). The
    compiled T=1 decode is bit-identical to eager here (separately confirmed), so this is just a clean
    single path. Delegates everything else to the inner resident model unchanged."""

    def __init__(self, inner: NemotronResidentModel) -> None:
        self._m = inner

    @property
    def cfg(self):
        return self._m.cfg

    @property
    def num_layers(self) -> int:
        return self._m.num_layers

    @property
    def embed_w(self) -> mx.array:
        return self._m.embed_w

    @property
    def lm_head_w(self) -> mx.array:
        return self._m.lm_head_w

    def make_caches(self, **kw):
        return self._m.make_caches(**kw)

    def truncate(self, caches, length: int) -> None:
        self._m.truncate(caches, length)

    def __call__(self, token_ids, **kw):
        kw["compiled"] = False
        return self._m(token_ids, **kw)


def _greedy(model, prompt_ids, *, max_new: int) -> tuple[list[int], float]:
    """Plain greedy decode — prefill, then one-token-per-forward argmax (``eos_id=None`` -> exactly
    ``max_new`` tokens). Returns ``(tokens, wall_seconds)``."""
    caches, ssm, conv = model.make_caches(max_rollback=1)
    t0 = time.perf_counter()
    logits, ssm, conv = model(mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv)
    cur = int(mx.argmax(logits[0, -1]).item())
    out = [cur]
    while len(out) < max_new:
        logits, ssm, conv = model(mx.array([cur]), caches=caches, ssm=ssm, conv=conv)
        cur = int(mx.argmax(logits[0, -1]).item())
        out.append(cur)
    return out[:max_new], time.perf_counter() - t0


def _path_ulp(model, prompt_ids) -> float:
    """Measure the T>1-verify vs T=1-decode logit gap (the bf16 ULP that spec drift rides on): feed the
    same two tokens as [a] then [b] (T=1x2) vs [a, b] (T=2) from independent post-prefill states and
    return max|Δlogit| on the shared next-token row."""
    def prefill():
        c, s, v = model.make_caches(max_rollback=2)
        lg, s, v = model(mx.array(prompt_ids), caches=c, ssm=s, conv=v)
        return c, s, v, int(mx.argmax(lg[0, -1]).item())

    c1, s1, v1, a = prefill()
    c2, s2, v2, _ = prefill()
    lg1, s1, v1 = model(mx.array([a]), caches=c1, ssm=s1, conv=v1)
    b = int(mx.argmax(lg1[0, -1]).item())
    lgA, _, _ = model(mx.array([b]), caches=c1, ssm=s1, conv=v1)
    lgB, _, _ = model(mx.array([a, b]), caches=c2, ssm=s2, conv=v2)
    return float(mx.max(mx.abs(lgA[0, -1].astype(mx.float32) - lgB[0, 1].astype(mx.float32))))


def _near_tie(model, prompt_ids, greedy_prefix: list[int], diverged_tok: int) -> tuple[int, float]:
    """Decode ``prompt_ids`` then ``greedy_prefix`` via the SAME per-token T=1 step path greedy uses
    (NOT chunked teacher-forcing, which diverges from the step path in bf16 — the Mamba-mixer note), and
    return ``(rank_of_diverged_tok, top1_minus_top2)`` for the next-token row. A bf16 ULP near-tie has
    the diverged token as greedy's runner-up (rank 2) with a tiny margin; the row's argmax is greedy's
    own token by construction (same path)."""
    caches, ssm, conv = model.make_caches(max_rollback=1)
    logits, ssm, conv = model(mx.array(prompt_ids), caches=caches, ssm=ssm, conv=conv)
    for t in greedy_prefix:                                   # T=1 steps == greedy's exact path
        logits, ssm, conv = model(mx.array([t]), caches=caches, ssm=ssm, conv=conv)
    row = logits[0, -1].astype(mx.float32)                    # predicts the divergence position
    order = mx.argsort(-row)                                  # descending
    rank = int(mx.argmax((order == diverged_tok).astype(mx.int32)).item()) + 1
    srt = mx.sort(row)
    return rank, float((srt[-1] - srt[-2]).item())


def run() -> None:
    mx.set_wired_limit(int(400 * 1024**3))
    t0 = time.perf_counter()
    inner = NemotronResidentModel(ART)
    model = _EagerMain(inner)
    tok = NemotronTokenizer(ART)
    mtp_art = NemotronArtifact(MTP_ART)
    mtp = build_resident_mtp(mtp_art, model.cfg)
    mtp_art.release()
    mx.clear_cache()
    embed, head = model.embed_w, model.lm_head_w
    load_min = (time.perf_counter() - t0) / 60

    prompt_ids = tok.encode(LONG_PROSE, add_bos=True)[:N_PROMPT]

    path_ulp = _path_ulp(model, prompt_ids)
    margin_tol = MARGIN_K * path_ulp
    greedy, g_s = _greedy(model, prompt_ids, max_new=N_GEN)
    g_tps = N_GEN / g_s

    print("\n=== Nemotron-Ultra MTP-M2 (resident native-MTP spec-decode, greedy-exact accept gate) ===")
    print(f"backbone: {ART}")
    print(f"mtp head: {MTP_ART}")
    print(f"load {load_min:.1f} min | prompt {len(prompt_ids)} tok | gen {N_GEN} tok (eos_id=None)")
    print(f"T>1-verify vs T=1-decode path_ulp = {path_ulp:.3e}  (near-tie tol = {margin_tol:.3e})")
    print(f"greedy (eager ref)   : {g_s:.1f}s  ({g_tps:.1f} tok/s)")

    ok = True
    for k in K_VALUES:
        ts = time.perf_counter()
        toks, st = spec_generate_k(model, mtp, embed, head, prompt_ids,
                                   k=k, max_new=N_GEN, eos_id=None)
        s_s = time.perf_counter() - ts
        s_tps = len(toks) / s_s
        # Only the FIRST divergence has a prefix that still matches greedy's (so the comparison is
        # valid); after it, the two valid greedy trajectories chaos-diverge. That first divergence must
        # be a bf16 ULP near-tie (greedy's rank-2 runner-up + tiny margin on greedy's own step path).
        fd = next((i for i in range(min(len(toks), len(greedy))) if toks[i] != greedy[i]), -1)
        if fd < 0:
            k_ok, detail = True, f"exact (bit-identical {len(greedy)}/{len(greedy)})"
        else:
            rank, margin = _near_tie(model, prompt_ids, greedy[:fd], toks[fd])
            k_ok = rank <= 2 and margin <= margin_tol
            kind = "bf16 ULP near-tie, chaos-diverges after" if k_ok else "NOT a near-tie — BUG"
            detail = f"bit-identical {fd}/{len(greedy)}, then {kind} (rank {rank}, margin {margin:.3e})"
        ok = ok and k_ok
        print(f"  k={k}: {detail}  mean_accept={st['mean_accept']:.2f}/{k + 1} "
              f"(max {st['max_accept']}, rounds {st['rounds']})  {s_s:.1f}s "
              f"({s_tps:.1f} tok/s, {s_tps / g_tps:.2f}x)")

    print("PASS" if ok else "FAIL (the first divergence was NOT a bf16 ULP near-tie — real logic error)")
    assert ok, "MTP-M2 gate failed: a spec first-divergence from greedy is not a bf16 ULP near-tie"


if __name__ == "__main__":
    run()
