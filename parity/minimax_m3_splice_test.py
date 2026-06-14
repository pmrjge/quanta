"""Model-free M3-6c (vision V3a) gate: the multimodal prefill **splice** + the runtime
``inputs_embeds`` path.

Two deterministic pieces, on tiny synthetic dims (no real weights, no checkpoint IO — runs in the
model-free sweep):

* :func:`quanta.minimax.model_vision_m3.splice_image_embeddings` — replaces the embedding rows at the
  ``image_token_index`` placeholder positions with the merged ViT tokens, in sequence order (the
  shipped ``processing_minimax.py`` rule: each ``]<]image[>[`` → ``num_tokens = grid.prod()//merge**2``
  placeholders of id ``image_token_index``). Pinned against an obvious nested-loop numpy oracle, incl.
  the multi-image fill order (image-0's tokens first, then image-1's). Rule-6 refusals: placeholder
  count != vision-token count, hidden-dim mismatch, length mismatch. Text-only (no placeholders) is a
  bit-exact passthrough for both ``[T,h]`` and ``[1,T,h]`` shapes.

* the runtime ``inputs_embeds`` path — :meth:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel.__call__`
  / ``prefill_chunked`` / ``multimodal_prefill`` accept a pre-spliced embedding stream instead of token
  ids. On a tiny synthetic (fully-bf16) model: ``inputs_embeds == token_ids`` **bit-exact** (prefill and
  cached); ``multimodal_prefill == manual embed+splice+forward`` bit-exact; the chunked ``inputs_embeds``
  admit == the single-shot ``inputs_embeds`` prefill bit-exact (bf16 mixer); rule-6 (both/neither
  input) refused.

The decisive real-weight wiring (the image is *seen* at 397B; rope_section still PINNED-pending the V3b
arbiter) is ``parity/minimax_m3_multimodal_real.py`` (SOLO).

    uv run python -m parity.minimax_m3_splice_test
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from parity.minimax_m3_runtime_test import _build_model, _synth
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.model_vision_m3 import splice_image_embeddings

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _maxabs(a: mx.array, b: mx.array) -> float:
    an, bn = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    return float(np.max(np.abs(an - bn)))


def _splice_oracle(text_np: np.ndarray, ids_np: np.ndarray, vis_np: np.ndarray, img_id: int
                   ) -> np.ndarray:
    """Obvious nested-loop splice: fill placeholder rows with vision rows in sequence order."""
    out = text_np.copy()
    vi = 0
    for i, tid in enumerate(ids_np):
        if int(tid) == img_id:
            out[i] = vis_np[vi]
            vi += 1
    assert vi == vis_np.shape[0]
    return out


# tiny synthetic config with a SMALL image_token_index (in-vocab for the embed table) -------------- #
def _cfg_mm(img_id: int = 5) -> MiniMaxM3Config:
    tc = {
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 32,
        "dense_intermediate_size": 96, "shared_intermediate_size": 32,
        "num_hidden_layers": 5,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "rotary_dim": 8, "partial_rotary_factor": 0.5, "rope_theta": 5e6,
        "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
        "attention_output_gate": False,
        "num_local_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "scoring_func": "sigmoid", "use_routing_bias": True, "routed_scaling_factor": 2.0,
        "norm_topk_prob": True,
        "moe_layer_freq": [0, 0, 1, 1, 1],
        "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 1048576, "tie_word_embeddings": False,
        "eos_token_id": 200020,
    }
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "config.json").write_text(json.dumps(
            {"model_type": "minimax_m3_vl", "text_config": tc, "image_token_index": img_id}))
        return MiniMaxM3Config.from_pretrained(d)


def run() -> None:
    mx.random.seed(0)
    IMG = 5
    H = 64

    # --- 1) splice scatter == nested-loop oracle (scattered placeholders) -------------------------
    T = 20
    rng = np.random.default_rng(0)
    ids_np = rng.integers(0, 40, size=T).astype(np.int32)
    slots = np.array([2, 3, 9, 14, 15], dtype=np.int64)          # placeholder positions
    ids_np[slots] = IMG
    ids_np[ids_np == IMG] = IMG                                  # (no-op; explicit)
    # ensure no *other* position accidentally equals IMG
    for i in range(T):
        if i not in slots and ids_np[i] == IMG:
            ids_np[i] = IMG + 1
    M = int((ids_np == IMG).sum())
    text_np = rng.standard_normal((T, H)).astype(np.float32)
    vis_np = (rng.standard_normal((M, H)) * 3.0).astype(np.float32)  # distinct magnitude from text
    te, ids, vis = mx.array(text_np), mx.array(ids_np), mx.array(vis_np)
    got = splice_image_embeddings(te, ids, vis, IMG)
    exp = _splice_oracle(text_np, ids_np, vis_np, IMG)
    _ck(got.shape == (T, H), f"splice shape {got.shape}")
    _ck(_maxabs(got, mx.array(exp)) == 0.0, "splice != nested-loop oracle (scattered placeholders)")

    # --- 2) multi-image fill order: image-0 rows fill the first block, image-1 the second ---------
    # ids = [t,t, START, IMG x4, END, t, START, IMG x2, END, t]
    ids2 = np.array([10, 11, 200029, IMG, IMG, IMG, IMG, 200030,
                     12, 200029, IMG, IMG, 200030, 13], dtype=np.int32)
    T2 = ids2.shape[0]
    img0 = (np.ones((4, H)) * 1.0).astype(np.float32)            # image-0: rows all 1.0
    img1 = (np.ones((2, H)) * 2.0).astype(np.float32)            # image-1: rows all 2.0
    vis2 = np.concatenate([img0, img1], axis=0)                  # tower concat order
    text2 = rng.standard_normal((T2, H)).astype(np.float32)
    out2 = splice_image_embeddings(mx.array(text2), mx.array(ids2), mx.array(vis2), IMG)
    out2n = np.asarray(out2.astype(mx.float32))
    p = np.where(ids2 == IMG)[0]
    _ck(np.all(out2n[p[:4]] == 1.0), "image-0 rows not placed in first placeholder block")
    _ck(np.all(out2n[p[4:]] == 2.0), "image-1 rows not placed in second placeholder block")
    # non-placeholder rows are the original text, untouched
    keep = np.array([i for i in range(T2) if ids2[i] != IMG])
    _ck(np.max(np.abs(out2n[keep] - text2[keep])) == 0.0, "splice perturbed non-placeholder rows")

    # --- 3) rule-6 refusals: count, hidden-dim, length -------------------------------------------
    for bad_vis, why in (
        (mx.array(rng.standard_normal((M - 1, H)).astype(np.float32)), "count (too few)"),
        (mx.array(rng.standard_normal((M + 2, H)).astype(np.float32)), "count (too many)"),
        (mx.array(rng.standard_normal((M, H + 1)).astype(np.float32)), "hidden-dim"),
    ):
        try:
            splice_image_embeddings(te, ids, bad_vis, IMG)
            _ck(False, f"splice accepted a {why} mismatch (rule 6)")
        except ValueError:
            _ck(True, f"splice refused {why} (rule 6)")
    try:
        splice_image_embeddings(te, mx.array(ids_np[:-1]), vis, IMG)
        _ck(False, "splice accepted a token-id/embeds length mismatch (rule 6)")
    except ValueError:
        _ck(True, "splice refused length mismatch (rule 6)")

    # --- 4) text-only passthrough (no placeholders), [T,h] and [1,T,h] ---------------------------
    ids_txt = mx.array((rng.integers(6, 40, size=T)).astype(np.int32))   # none == IMG (IMG=5)
    empty = mx.zeros((0, H), dtype=mx.float32)
    out_txt = splice_image_embeddings(te, ids_txt, empty, IMG)
    _ck(_maxabs(out_txt, te) == 0.0, "text-only splice not a passthrough ([T,h])")
    te3 = te[None]
    out_txt3 = splice_image_embeddings(te3, ids_txt, empty, IMG)
    _ck(out_txt3.shape == (1, T, H) and _maxabs(out_txt3, te3) == 0.0,
        "text-only splice not a passthrough ([1,T,h])")

    # --- 5) runtime inputs_embeds == token_ids (bit-exact), tiny fully-bf16 model -----------------
    cfg = _cfg_mm(IMG)
    w = _synth(cfg, mx.random.key(1))
    m = _build_model(cfg, w, packed=False)                       # fully bf16 (mixer + experts) ⇒ exact
    Tt = 12
    tok = mx.array([(i * 7 + 9) % cfg.vocab_size for i in range(Tt)], dtype=mx.int32)
    l_ids = m(tok)
    l_emb = m(inputs_embeds=m.embed_tokens(tok))
    mx.eval(l_ids, l_emb)
    _ck(_maxabs(l_ids, l_emb) == 0.0, "inputs_embeds prefill != token_ids prefill (bit-exact)")
    # cached forward, both ways
    lc_ids = m(tok, caches=m.make_caches())
    lc_emb = m(inputs_embeds=m.embed_tokens(tok), caches=m.make_caches())
    mx.eval(lc_ids, lc_emb)
    _ck(_maxabs(lc_ids, lc_emb) == 0.0, "inputs_embeds cached != token_ids cached (bit-exact)")
    # [1,T,hidden] accepted == [T,hidden]
    l_emb3 = m(inputs_embeds=m.embed_tokens(tok)[None])
    _ck(_maxabs(l_emb3, l_emb) == 0.0, "inputs_embeds [1,T,h] != [T,h]")

    # --- 6) multimodal_prefill == manual embed+splice+forward; chunked-embeds == single-shot ------
    placed = [2, 3, 7, 8]                                        # 4 placeholders in the prompt
    mm = np.array([(i * 5 + 1) % cfg.vocab_size for i in range(Tt)], dtype=np.int32)
    for q in placed:
        mm[q] = IMG
    for i in range(Tt):                                          # scrub stray IMGs outside `placed`
        if i not in placed and mm[i] == IMG:
            mm[i] = IMG + 1
    mm_ids = mx.array(mm)
    vtok = (mx.random.normal((len(placed), cfg.hidden_size), key=mx.random.key(7)) * 0.3
            ).astype(mx.bfloat16)
    l_mm = m.multimodal_prefill(mm_ids, vtok)
    spliced = splice_image_embeddings(m.embed_tokens(mm_ids), mm_ids, vtok, cfg.image_token_index)
    l_manual = m(inputs_embeds=spliced)
    mx.eval(l_mm, l_manual)
    _ck(_maxabs(l_mm, l_manual) == 0.0, "multimodal_prefill != manual embed+splice+forward")
    # the image actually changed the stream: spliced logits differ from a no-splice run
    l_nosplice = m(mm_ids)                                       # placeholders keep embed_w[IMG]
    _ck(_maxabs(l_mm, l_nosplice) > 1e-3, "splice did not change the logits (image not wired in)")
    # chunked inputs_embeds == single-shot inputs_embeds (bf16 mixer ⇒ bit-exact), last position
    l_chunk = m.prefill_chunked(caches=m.make_caches(), inputs_embeds=spliced, chunk_tokens=4)
    mx.eval(l_chunk)
    _ck(_maxabs(l_chunk, l_manual[:, -1:]) == 0.0,
        "chunked inputs_embeds != single-shot inputs_embeds (last pos, bf16 bit-exact)")

    # --- 7) rule-6: both / neither input refused -------------------------------------------------
    try:
        m(tok, inputs_embeds=m.embed_tokens(tok))
        _ck(False, "__call__ accepted both token_ids and inputs_embeds (rule 6)")
    except ValueError:
        _ck(True, "__call__ refused both inputs (rule 6)")
    try:
        m()
        _ck(False, "__call__ accepted neither input (rule 6)")
    except ValueError:
        _ck(True, "__call__ refused neither input (rule 6)")

    print("\n=== MiniMax-M3-VL M3-6c (vision V3a) splice + inputs_embeds gate (model-free) ===")
    print("(1) splice == nested-loop oracle (|Δ|=0); (2) multi-image fill order correct")
    print("(3) rule-6 count/hidden/length refused; (4) text-only passthrough ([T,h]+[1,T,h])")
    print("(5) inputs_embeds == token_ids BIT-EXACT (prefill + cached)")
    print("(6) multimodal_prefill == manual; image changes logits; chunked-embeds == single-shot")
    print("(7) rule-6 both/neither input refused")
    print(f"PARITY-CHECKS: {_N}")
    print(f"PASS — M3 multimodal splice + inputs_embeds path output-equivalent + rule-6 honored "
          f"({_N} checks).")


if __name__ == "__main__":
    run()
