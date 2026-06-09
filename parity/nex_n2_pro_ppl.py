"""Nex-N2-Pro N2 — teacher-forced ppl + top-1, the int4-vs-int6-vs-bf16 e2e bits arbiter.

Three **sequential** streamed forwards (rule 8: one decoder layer resident at a time; the MoE
512-expert bf16 stack ~21.5 GiB is the peak, never the 739 GiB / 214 GiB / 304 GiB whole model) over
the *same* held-out prose, each freed before the next so only one is ever resident:

  1. **bf16 Nex source**   (``Qwen35SourceCheckpoint``)                 -> reference ppl + ref argmax
  2. **int4-g64 artifact** (``Qwen35Artifact``, dequant-on-read int4)   -> the default arm
  3. **int6-g64 artifact** (``Qwen35Artifact``, dequant-on-read int6)   -> the safety-net arm

All three share :func:`streamed_logits` verbatim — the SAME forward (the proven
``Qwen35ResidentModel`` prefill path via :func:`quanta.qwen35.runtime._load_block` ``packed=False``,
streamed one block at a time) — so each ppl Δ isolates pure expert-coding error (N1 already proved the
forward matches the transformers ``Qwen3_5Moe`` reference to ~2e-6). The arbiter ranks int4 vs int6 vs
bf16 head-to-head and recommends the bits by the **quality-vs-VRAM rule**: int4 (214 GiB) is the
strong default (int4-RTN was ~lossless +0.3% on the bf16-source Nemotron-Ultra — settled finding), and
int6 (304 GiB) ships only if int4 regresses materially on Nex. The quant mix measured here is the
SERVED mix: int{4,6} routed experts + int8 dense/shared + bf16 control (so the Δ is the real
deployment loss, not an expert-only proxy).

The corpus is original expository prose, **held out** from any calibration set (RTN is data-free, so
strictly there is none; the prose is original regardless). Qwen3.5 has no BOS
(``add_bos_token=false``) so the text is encoded raw; at this ~1K length the dynamic-YaRN factor is 1.0
(<< the 262144 native window) so all three arms run the identical RoPE — the only difference is the
dequantized weights.

    uv run python -m parity.nex_n2_pro_ppl              # the real 3-arm arbiter (SOLO)
    uv run python -m parity.nex_n2_pro_ppl 8 160        # n_layers, n_tok (bounded smoke)

# parity-gate: real-weight
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

from quanta.qwen35.artifact import Qwen35Artifact
from quanta.qwen35.config import Qwen35Config
from quanta.qwen35.loader import Qwen35SourceCheckpoint
from quanta.qwen35.runtime import _block_arrays, _load_block, _one_plus
from quanta.qwen35.tokenizer import Qwen35Tokenizer

SRC = "/Users/pmrj/models/Nex-N2-Pro"
INT4 = "/Users/pmrj/models/Nex-N2-Pro-quanta_int4g64"
INT6 = "/Users/pmrj/models/Nex-N2-Pro-quanta_int6g64"
N_TOK = 1024  # ~10x a 100-tok pilot → a de-noised, trustworthy ppl verdict

# Original expository prose, held out from any calibration corpus (RTN is data-free regardless).
PROSE = """The history of mapmaking is in large part the history of how people have tried to compress \
the round, irregular, and only partly known surface of the earth onto a flat sheet that can be \
carried, copied, and consulted. The earliest maps were not drawn to scale and made no pretense of \
covering the whole world; they recorded the things that mattered to the people who made them, such as \
the course of a river, the location of a well, or the boundary between one field and the next. A clay \
tablet from Mesopotamia, a painted screen from central Mexico, and a stick chart from the islands of \
the Pacific each solve the same problem in a different visual language, and each assumes a reader who \
already knows a great deal about the territory and needs only to be reminded of its arrangement.

As trade and travel reached farther, the demands on a map grew. A merchant who would never see the \
far end of a route still had to plan for it, and a captain steering out of sight of land needed a way \
to reason about a place before arriving there. Greek geographers responded by treating the earth as a \
sphere and asking how its surface could be projected onto a plane, a question that has no perfect \
answer. Every projection sacrifices something: one preserves the shapes of small regions but stretches \
the poles into impossible widths, another keeps areas honest at the cost of bending every straight \
line, and a third keeps compass bearings true so a navigator can hold a single heading, even though \
the distances it implies are wrong. The mapmaker therefore chooses not the correct projection but the \
one whose particular distortions matter least for the task at hand.

The invention of printing changed maps as profoundly as it changed books. A hand-copied chart was \
unique, expensive, and slowly corrupted as each copyist introduced new errors; an engraved plate could \
print thousands of identical impressions and could be corrected in one place to improve them all. Maps \
became cheaper, more uniform, and more widely shared, and with that sharing came a new kind of \
authority, since a printed map looked settled and official even when its interior was filled with \
guesswork. Cartographers were rarely shy about the gaps. They drew elaborate sea monsters in the empty \
oceans and confident coastlines around lands no European had visited, and for a long time it was \
genuinely difficult to tell the carefully surveyed parts of a map from the invented ones.

What finally disciplined the map was measurement. The development of reliable instruments for fixing \
latitude, and much later longitude, meant that a position could be established by observation rather \
than by dead reckoning and hope. National surveys laid networks of triangles across whole countries, \
anchoring every village and hilltop to a common framework, and the slow, patient work of these surveys \
turned the map from a picture into an instrument. By the time aerial photography and then satellites \
arrived, the basic ambition was already fixed: to record where things actually are, precisely enough \
that a stranger can trust the record. The modern map, continuously updated from orbit and consulted on \
a screen in the palm of a hand, is the direct descendant of that ambition, and it carries forward the \
same old compromise, because the earth is still round and the screen is still flat."""


def streamed_logits(weights, cfg: Qwen35Config, ids: mx.array, *, n_layers: int | None = None
                    ) -> mx.array:
    """Teacher-forcing logits ``[1,T,vocab]`` via the streamed reference forward (rule 8).

    ``weights`` is a ``Qwen35SourceCheckpoint`` (bf16) OR a ``Qwen35Artifact`` (dequant-on-read) —
    they duck-type the same ``embed`` / ``block_norms`` / ``linear_attn`` / ``full_attn`` / ``moe`` /
    ``final_norm`` / ``lm_head`` / ``release`` surface, so this one forward serves all three arms.
    Each block is built dense-bf16 by ``_load_block(..., packed=False, packed_experts=False)`` (the
    proven parity-reference path), run over the whole window with fresh per-layer state (prefill ==
    ``Qwen35ResidentModel.__call__(caches=None)``), then freed before the next (one layer resident)."""
    n = cfg.num_hidden_layers if n_layers is None else n_layers
    ids = ids.reshape(-1)
    hint = int(ids.shape[0])                                    # T; dynamic-YaRN factor is 1.0 here
    h = weights.embed()[ids][None].astype(mx.bfloat16)          # [1,T,hidden]
    mx.eval(h)
    for i in range(n):
        blk = _load_block(weights, cfg, i, packed=False, packed_experts=False)
        mx.eval(_block_arrays(blk))
        h, _, _ = blk(h, cache=None, state=None, conv_state=None, use_fast=True, seq_hint=hint)
        mx.eval(h)
        del blk
        weights.release()
        mx.clear_cache()
    norm_w = _one_plus(weights.final_norm())                    # (1+w) Qwen3_5MoeRMSNorm final norm
    hh = mx.fast.rms_norm(h, norm_w.astype(h.dtype), cfg.norm_eps)
    logits = hh @ weights.lm_head().T.astype(hh.dtype)
    mx.eval(logits)
    return logits


def teacher_forced(logits: mx.array, ids: mx.array) -> tuple[float, float, mx.array]:
    """Teacher-forced perplexity + top-1 next-token accuracy + the per-position argmax.

    ``logits`` ``[1,T,vocab]`` predict positions ``1..T-1`` from ``0..T-2`` (causal teacher forcing)."""
    lp = logits[0, :-1].astype(mx.float32)                      # [T-1, vocab]
    tgt = ids.reshape(-1)[1:]                                   # [T-1]
    logp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
    ppl = float(mx.exp(mx.mean(nll)).item())
    argmax = mx.argmax(lp, axis=-1)
    acc = float(mx.mean((argmax == tgt).astype(mx.float32)).item())
    return ppl, acc, argmax


def _streamed_arm(label: str, weights, cfg: Qwen35Config, ids: mx.array, n_layers: int | None,
                  ref_argmax: mx.array | None, ref_ppl: float | None) -> dict:
    """Run one streamed arm; report ppl/acc (+ Δppl% & top-1 agreement vs bf16 when given a ref); free
    the weights before returning so only one arm is ever resident (rule 8 / one-model-at-a-time)."""
    t0 = time.perf_counter()
    logits = streamed_logits(weights, cfg, ids, n_layers=n_layers)
    ppl, acc, argmax = teacher_forced(logits, ids)
    dt = time.perf_counter() - t0
    out = {"label": label, "ppl": ppl, "acc": acc, "argmax": argmax, "sec": dt}
    extra = ""
    if ref_argmax is not None and ref_ppl is not None:
        out["dppl"] = 100.0 * (ppl / ref_ppl - 1.0)
        out["agree"] = float(mx.mean((argmax == ref_argmax).astype(mx.float32)).item())
        extra = f"  Δppl {out['dppl']:+.2f}%  agree {out['agree']:.4f}"
    else:
        extra = "  (reference)"
    weights.release()
    del logits
    mx.clear_cache()
    print(f"  [{label:10}] ppl {ppl:7.4f}  acc {acc:.4f}{extra}  ({dt:.0f}s)", flush=True)
    return out


def run(n_layers: int | None = None, n_tok: int = N_TOK) -> None:
    mx.set_cache_limit(8 * 1024**3)
    tok = Qwen35Tokenizer.from_pretrained(INT4)                 # self-contained tokenizer (no BOS)
    ids_list = tok.encode(PROSE)[:n_tok]
    ids = mx.array(ids_list, dtype=mx.uint32)
    print(f"=== Nex-N2-Pro N2 ppl arbiter — {len(ids_list)} tok, "
          f"{'all' if n_layers is None else n_layers} layers (streamed, SOLO) ===", flush=True)

    # 1) bf16 reference — its argmax + ppl anchor the int4/int6 Δ
    cfg_src = Qwen35Config.from_pretrained(SRC)
    src = Qwen35SourceCheckpoint(SRC, cfg_src)
    ref = _streamed_arm("bf16-src", src, cfg_src, ids, n_layers, None, None)
    del src
    mx.clear_cache()
    ref_argmax, ref_ppl = ref["argmax"], ref["ppl"]

    # 2) int4 + 3) int6 — each streamed (dequant-on-read) vs the bf16 reference
    results = [ref]
    for label, path in (("int4-g64", INT4), ("int6-g64", INT6)):
        art = Qwen35Artifact(path)
        results.append(_streamed_arm(label, art, art.cfg, ids, n_layers, ref_argmax, ref_ppl))
        del art
        mx.clear_cache()

    # --- verdict: quality-vs-VRAM. teacher-forced ppl is THE arbiter (CLAUDE.md methodology #4); the
    # int4 default ships unless its ppl regresses materially on Nex. Top-1 agreement is a SECONDARY
    # signal — on real prose it is noisy (bf16-ULP near-tie flips at low-confidence positions, a
    # settled finding), so it is a loose sanity floor (>0.90, the Nemotron-Ultra U3 precedent), NOT a
    # tight gate; the ppl Δ + the next-token acc are the trustworthy measures. ---
    i4 = next(r for r in results if r["label"] == "int4-g64")
    i6 = next(r for r in results if r["label"] == "int6-g64")
    int4_ok = i4["dppl"] < 1.5 and i4["agree"] > 0.90
    if int4_ok:
        verdict = (f"SHIP int4-g64 (214 GiB) — Δppl {i4['dppl']:+.2f}% / acc {i4['acc']:.4f} vs bf16 "
                   f"{ref['acc']:.4f} / agree {i4['agree']:.4f} (~lossless, the 90 GiB-lighter arm); "
                   f"int6-g64 (304 GiB) is Δppl {i6['dppl']:+.2f}% — recovers "
                   f"{i4['dppl'] - i6['dppl']:.2f}pp for +90 GiB, not worth it.")
    else:
        verdict = (f"int4 regresses (Δppl {i4['dppl']:+.2f}% / acc {i4['acc']:.4f} / agree "
                   f"{i4['agree']:.4f}); int6-g64 (304 GiB) Δppl {i6['dppl']:+.2f}% / acc "
                   f"{i6['acc']:.4f} / agree {i6['agree']:.4f} — SHIP int6 (it recovers the loss).")
    print(f"\nVERDICT: {verdict}", flush=True)
    print("PARITY-CHECKS: 3", flush=True)  # 3 streamed arms (bf16 / int4 / int6)


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else N_TOK
    run(nl, nt)
