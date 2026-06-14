"""MiniMax-M3-VL M3-6c (vision V3a): real-weight multimodal prefill **wiring** re-gate (SOLO, ~235 GiB).

Loads BOTH the int4-g64 text decoder (the serving config — packed mixer + packed int4 experts, all 60
layers, ~233 GiB via :class:`quanta.minimax.runtime_m3.MiniMaxM3ResidentModel`) AND the dense CLIP-ViT
vision tower (the 523 vision tensors, ~1.6 GiB via
:func:`quanta.minimax.model_vision_m3.load_vision_model`), processes a real factor-aligned image
through the native processor (resize == identity ⇒ the exactly-reproducible path), runs the real ViT →
merged LLM tokens, splices them into a real prompt's embedding stream at the ``image_token_index``
(200025) placeholders, and runs the real multimodal prefill. The arbiter is **the full wiring is
correct and the image is actually SEEN**, on the actual weights:

  1. **shapes** — one 56×56 image → grid ``(1,4,4)`` → 16 ViT patches → 4 merged tokens of width
     ``hidden`` (6144); the prompt carries exactly 4 ``image_token_index`` placeholders (the splice's
     rule-6 count check passes only when these match).
  2. **image is SEEN (causal split).** The spliced prefill vs a no-splice run of the SAME token ids
     (placeholders keep ``embed_w[200025]``): logits BEFORE the first image position are **BIT-EXACT**
     (|Δ|==0 — causal isolation: those positions never attend the image rows), logits AT/AFTER it
     **differ materially** (the merged ViT tokens flow into the decoder and change every downstream
     position). A dead splice (image not wired in) would leave the whole stream unchanged.
  3. **finite + sane** — the multimodal logits are all finite, non-degenerate, not exploded.
  4. **inputs_embeds == token_ids BIT-EXACT @ scale.** On a plain text prompt the ``inputs_embeds``
     path reproduces the token-id path bit-for-bit on the real packed model (the splice rides a
     numerically-transparent embed substitution; gated tiny-synthetic in
     ``parity/minimax_m3_splice_test.py``).

This validates the splice + ViT→text wiring end-to-end at 397B. It does **NOT** settle the
[PINNED-pending-e2e] vision ``rope_section`` — every candidate section produces "an image is seen"; the
section is judged DOWNSTREAM by the LLM on real natural-image content, which needs an image decoder +
ground-truth and is the separate V3b arbiter (no decoder ships in-env, so V3b is deferred). Here the
section is the default ``(8,16,16)``.

    uv run python -m parity.minimax_m3_multimodal_real           # full re-gate (all 60 layers, SOLO)
    uv run python -m parity.minimax_m3_multimodal_real 4         # n_layers (bounded smoke)

# parity-gate: real-weight
"""

from __future__ import annotations

import sys
import time

import numpy as np

import mlx.core as mx

from quanta.minimax import model_vision_m3 as V
from quanta.minimax.image_m3 import MiniMaxM3ImageProcessor
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel

ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"
VSTART, VEND = 200029, 200030          # ]<]start of image[>[ / ]<]end of image[>[
SEEN_FLOOR = 1.0                       # at/after the image, max|Δlogit| must exceed this (image wired in)

_N = 0


def _ck(cond: bool, msg: str) -> None:
    global _N
    assert cond, msg
    _N += 1


def _set_wired() -> None:
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        rec = int(info.get("max_recommended_working_set_size", 0))
        if rec > 0:
            mx.set_wired_limit(rec)
    except Exception:  # noqa: BLE001 — wired-limit is an optimization, never fail the gate on it
        pass


def _grad_image(h: int, w: int, seed: int) -> np.ndarray:
    """A smooth structured RGB image (gradient + ripple) — real spatial structure, not noise."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (255 * (0.5 + 0.5 * np.sin(6.0 * xx + seed))).astype(np.uint8)
    g = (255 * yy).astype(np.uint8)
    b = (255 * (0.5 + 0.5 * np.cos(4.0 * yy * xx + seed))).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def run(n_layers: int | None = None) -> None:
    full = n_layers is None
    mx.set_cache_limit(8 * 1024**3)
    _set_wired()

    print(f"=== MiniMax-M3-VL V3a multimodal wiring re-gate "
          f"({'all 60' if full else n_layers} layers, SOLO) ===", flush=True)
    t0 = time.perf_counter()
    text = MiniMaxM3ResidentModel(ART, n_layers=n_layers, packed=True, packed_experts=True)
    t_text = time.perf_counter() - t0
    cfg = text.cfg
    IMG = cfg.image_token_index
    print(f"  text decoder: {text.num_layers}L resident in {t_text:.0f}s (packed mixer + int4 experts)",
          flush=True)
    t1 = time.perf_counter()
    vis = V.load_vision_model(ART)
    print(f"  vision tower: {vis.cfg.num_hidden_layers}L ViT in {time.perf_counter() - t1:.0f}s "
          f"(rope_section {vis.rope_section} — default, PINNED-pending V3b)", flush=True)

    # ---- (1) real image → ViT → merged tokens; build a prompt with matching placeholders ----------
    proc = MiniMaxM3ImageProcessor()
    out = proc.preprocess(_grad_image(56, 56, 1))
    _ck(out.grid_thw == [(1, 4, 4)], f"grid_thw {out.grid_thw}")
    pv = mx.array(out.pixel_values).astype(mx.bfloat16)
    vtok = vis(pv, out.grid_thw)
    mx.eval(vtok)
    n_img = out.grid_thw[0][0] * out.grid_thw[0][1] * out.grid_thw[0][2] // (vis.merge ** 2)
    _ck(vtok.shape == (n_img, cfg.hidden_size), f"merged token shape {vtok.shape} (want ({n_img},"
                                                f"{cfg.hidden_size}))")

    pre = [cfg.bos_token_id, 1037, 2048, 511]          # arbitrary in-vocab text context
    post = [777, 88, 9001, 123, 456]
    ids = pre + [VSTART] + [IMG] * n_img + [VEND] + post
    first_img = len(pre) + 1                            # position of the first IMG placeholder
    n_slots = sum(1 for t in ids if t == IMG)
    _ck(n_slots == n_img, f"prompt placeholders {n_slots} != merged tokens {n_img}")
    ids_arr = mx.array(ids, dtype=mx.int32)

    # ---- (2) image is SEEN: prefix bit-exact, suffix differs --------------------------------------
    l_img = text.multimodal_prefill(ids_arr, vtok)     # embed → splice → prefill ([1,T,vocab])
    l_txt = text(ids_arr)                              # no splice (placeholders keep embed_w[IMG])
    mx.eval(l_img, l_txt)
    d_prefix = _maxabs(l_img[:, :first_img], l_txt[:, :first_img])
    d_suffix = _maxabs(l_img[:, first_img:], l_txt[:, first_img:])
    print(f"  [seen] prefix |Δ| {d_prefix:.2e} (want 0) | suffix max|Δ| {d_suffix:.3f} "
          f"(want > {SEEN_FLOOR})", flush=True)
    if full:
        _ck(d_prefix == 0.0, f"prefix logits not bit-exact ({d_prefix}) — causal isolation broken")
        _ck(d_suffix > SEEN_FLOOR, f"image not wired in: suffix max|Δ| {d_suffix:.3f} <= {SEEN_FLOOR}")

    # ---- (3) finite + sane ----------------------------------------------------------------------
    ln = np.asarray(l_img[0].astype(mx.float32), dtype=np.float64)
    _ck(bool(np.all(np.isfinite(ln))), "multimodal logits have non-finite values")
    _ck(float(ln.std()) > 1e-3, f"multimodal logits degenerate (std {float(ln.std()):.2e})")
    _ck(float(np.max(np.abs(ln))) < 1e4, f"multimodal logits exploded (max|x| {np.max(np.abs(ln)):.2e})")

    # ---- (4) inputs_embeds == token_ids BIT-EXACT @ scale (plain text) ---------------------------
    txt_ids = mx.array([cfg.bos_token_id, 12, 345, 6789, 222, 4, 91, 1000], dtype=mx.int32)
    a = text(txt_ids)
    b = text(inputs_embeds=text.embed_tokens(txt_ids))
    mx.eval(a, b)
    d_emb = _maxabs(a, b)
    print(f"  [embeds] text-only inputs_embeds vs token_ids: |Δ| {d_emb:.2e}", flush=True)
    _ck(d_emb == 0.0, f"inputs_embeds != token_ids on real model: |Δ| {d_emb}")

    del text, vis
    mx.clear_cache()

    if full:
        print(f"\nVERDICT: V3a multimodal wiring VALIDATED @ 397B — real image → ViT → {n_img} merged "
              f"tokens spliced into the text stream; image is SEEN (prefix BIT-EXACT |Δ| 0, suffix "
              f"max|Δ| {d_suffix:.2f}); logits finite/sane; inputs_embeds == token_ids BIT-EXACT. The "
              f"splice + ViT→text path ships; vision rope_section is the separate V3b arbiter.",
              flush=True)
    else:
        print(f"\nSMOKE ok — multimodal path ran ({n_layers} layers); "
              f"prefix |Δ| {d_prefix:.1e}, suffix |Δ| {d_suffix:.2f}, embeds |Δ| {d_emb:.1e} "
              f"(numbers not meaningful partial).", flush=True)
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(nl)
