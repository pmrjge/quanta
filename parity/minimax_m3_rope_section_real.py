"""MiniMax-M3-VL V3b: settle the vision ``rope_section`` — the heavy ~235 GiB real-weight arbiter (SOLO).

The ``(t,h,w)`` freq-pair split of the 3-D vision RoPE is the **one** knob no on-disk artifact fixes
(CLAUDE.md: transformers CLIP uses learned pos-embeds; even Qwen3-VL's vision tower is 2-D, so there
is no reference to diff against). Every candidate section produces "an image is seen" (the V3a wiring
gate), so the section can only be judged **downstream by the LLM** on real natural-image content. This
is that arbiter: it loads the int4 text decoder (all 60 layers, packed mixer + int4 experts) AND the
dense CLIP-ViT once, then for each candidate in
:func:`quanta.minimax.model_vision_m3.candidate_rope_sections` it runs

    image → ViT(section) → merged tokens → splice into [instruction + image + caption] →
    teacher-forced perplexity of the CAPTION span,

mutating ``vis.rope_section`` in place between candidates (the weights are identical — only the rope
changes, the lever the model-free ``parity/minimax_m3_rope_section_test.py`` proves is live). The
section that minimizes caption ppl is the one the ViT was trained for: a wrong rope mis-rotates the
patch tokens, so the merged features are less informative and the caption is harder to predict. A
**text-only baseline** (the same caption with no image) bounds "does the image help at all".

The arbiter is only as good as its inputs — give it a REAL natural image and a TRUE caption of it::

    uv run python -m parity.minimax_m3_rope_section_real /path/to/photo.jpg "a true caption of it"

Reduced-layer / synthetic-image **smoke** (validates the whole code path cheaply — NO verdict; a
gradient image and a truncated decoder cannot rank rope sections)::

    uv run python -m parity.minimax_m3_rope_section_real --layers 2
    uv run python -m parity.minimax_m3_rope_section_real --layers 4 /path/to/photo.jpg "caption"

Decoding the image file needs PIL (the ``reference`` extra, offline — rule 5); the native
:class:`quanta.minimax.image_m3.MiniMaxM3ImageProcessor` then turns it into patches. An off-grid image
goes through the best-effort bicubic resize (documented unpinned); a factor-aligned (28-multiple) image
hits the identity-resize exact path.

# parity-gate: real-weight
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

import mlx.core as mx

from quanta.minimax import model_vision_m3 as V
from quanta.minimax.config_m3 import MiniMaxM3Config
from quanta.minimax.image_m3 import MiniMaxM3ImageProcessor
from quanta.minimax.runtime_m3 import MiniMaxM3ResidentModel
from quanta.minimax.tokenizer import MiniMaxTokenizer

ART = "/Users/pmrj/models/MiniMax-M3-quanta_int4g64"
VSTART, VEND = 200029, 200030          # ]<]start of image[>[ / ]<]end of image[>[
INSTRUCTION = "Describe this image.\n"
DEFAULT_CAPTION = ("a clear photograph showing the scene described in natural language with everyday "
                   "objects and their colors and positions")

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
    except Exception:  # noqa: BLE001 — wired-limit is an optimization, never fail the arbiter on it
        pass


def _grad_image(h: int, w: int, seed: int) -> np.ndarray:
    """A smooth structured RGB image (gradient + ripple) — real spatial structure, not noise. Used only
    for the synthetic SMOKE path (a gradient cannot rank rope sections — that needs a real image)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (255 * (0.5 + 0.5 * np.sin(6.0 * xx + seed))).astype(np.uint8)
    g = (255 * yy).astype(np.uint8)
    b = (255 * (0.5 + 0.5 * np.cos(4.0 * yy * xx + seed))).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _span_ppl(logits: mx.array, ids_list: list[int], start: int, length: int) -> float:
    """Teacher-forced perplexity of the token span ``ids[start:start+length]`` (each predicted from the
    preceding position): ``logits[0, start-1 : start+length-1]`` vs ``ids[start:start+length]``."""
    lp = logits[0, start - 1: start + length - 1].astype(mx.float32)        # [length, vocab]
    tgt = mx.array(ids_list[start: start + length], dtype=mx.int32)         # [length]
    logp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
    return float(mx.exp(mx.mean(nll)).item())


def run(image_path: str | None = None, caption: str | None = None,
        n_layers: int | None = None) -> None:
    full = n_layers is None and image_path is not None
    mx.set_cache_limit(8 * 1024**3)
    _set_wired()
    caption = caption or DEFAULT_CAPTION

    mode = ("FULL verdict" if full else
            ("synthetic-image SMOKE" if image_path is None else f"{n_layers}-layer SMOKE"))
    print(f"=== MiniMax-M3-VL V3b rope_section arbiter ({mode}, SOLO) ===", flush=True)
    if image_path is None:
        print("  [smoke] no image given → synthetic gradient (cannot rank sections; code-path only)",
              flush=True)
    else:
        print(f"  image: {image_path}", flush=True)
    print(f"  caption ({len(caption)} chars): {caption!r}", flush=True)

    # ---- decode + process the image -------------------------------------------------------------
    if image_path is None:
        rgb = _grad_image(56, 56, 1)                                        # factor-aligned, exact path
    else:
        from parity._image_decode import decode_image_rgb  # noqa: PLC0415 — reference extra, offline
        rgb = decode_image_rgb(image_path)
    proc = MiniMaxM3ImageProcessor()
    out = proc.preprocess(rgb)
    grid = out.grid_thw
    n_img = sum(t * h * w for (t, h, w) in grid) // (MiniMaxM3ImageProcessor().merge ** 2)
    pv = mx.array(out.pixel_values).astype(mx.bfloat16)
    print(f"  processed: grid_thw {grid} → {n_img} merged LLM tokens "
          f"(pixel_values {out.pixel_values.shape})", flush=True)

    # ---- load the decoder (packed int4, n_layers) + the dense ViT (once) ------------------------
    t0 = time.perf_counter()
    text = MiniMaxM3ResidentModel(ART, n_layers=n_layers, packed=True, packed_experts=True)
    cfg: MiniMaxM3Config = text.cfg
    IMG = cfg.image_token_index
    print(f"  text decoder: {text.num_layers}L resident in {time.perf_counter() - t0:.0f}s "
          f"(packed mixer + int4 experts)", flush=True)
    vis = V.load_vision_model(ART)
    hd = vis.cfg.hidden_size // vis.cfg.num_attention_heads
    sections = V.candidate_rope_sections(hd)
    print(f"  vision tower: {vis.cfg.num_hidden_layers}L ViT; sweeping {len(sections)} rope_sections "
          f"of head_dim {hd}", flush=True)

    tok = MiniMaxTokenizer(os.path.join(ART, "tokenizer.json"), cfg)
    instr_ids = tok.encode(INSTRUCTION)
    cap_ids = tok.encode(" " + caption)
    L = len(cap_ids)

    # prompt: [bos] + instruction + [VSTART] + [IMG]*n + [VEND] + caption ; teacher-force the caption.
    pre = [cfg.bos_token_id, *instr_ids, VSTART, *([IMG] * n_img), VEND]
    ids_list = [*pre, *cap_ids]
    cap_start = len(pre)
    _ck(sum(1 for t in ids_list if t == IMG) == n_img, "placeholder count != merged tokens")

    # ---- text-only baseline: same caption, no image (does the image help at all?) ---------------
    base_ids = [cfg.bos_token_id, *instr_ids, *cap_ids]
    base_logits = text(mx.array(base_ids, dtype=mx.int32))
    mx.eval(base_logits)
    base_ppl = _span_ppl(base_logits, base_ids, len(instr_ids) + 1, L)
    del base_logits
    mx.clear_cache()
    print(f"\n  text-only caption ppl (no image): {base_ppl:.4f}", flush=True)

    # ---- sweep: per-section caption ppl conditioned on the image --------------------------------
    ids_arr = mx.array(ids_list, dtype=mx.int32)
    results: list[tuple[tuple[int, int, int], float]] = []
    print(f"  {'section':>14}   image-cap-ppl   Δ vs text-only", flush=True)
    for sec in sections:
        vis.rope_section = sec
        vtok = vis(pv, grid)
        mx.eval(vtok)
        logits = text.multimodal_prefill(ids_arr, vtok)                    # [1,T,vocab]
        mx.eval(logits)
        ppl = _span_ppl(logits, ids_list, cap_start, L)
        results.append((sec, ppl))
        d = 100.0 * (ppl / base_ppl - 1.0)
        star = "  ←default" if sec == V.default_rope_section(hd) else ""
        print(f"  {str(sec):>14}   {ppl:11.4f}   {d:+7.2f}%{star}", flush=True)
        del logits, vtok
        mx.clear_cache()

    del text, vis
    mx.clear_cache()

    results.sort(key=lambda r: r[1])
    best_sec, best_ppl = results[0]
    worst_ppl = results[-1][1]
    spread = 100.0 * (worst_ppl / best_ppl - 1.0)
    default_ppl = next(p for s, p in results if s == V.default_rope_section(hd))

    # ---- verdict --------------------------------------------------------------------------------
    print(f"\n  text-only baseline ppl {base_ppl:.4f}", flush=True)
    print(f"  BEST section {best_sec} → image-cap-ppl {best_ppl:.4f} "
          f"({100.0 * (best_ppl / base_ppl - 1.0):+.2f}% vs text-only)", flush=True)
    print(f"  on-disk default {V.default_rope_section(hd)} → {default_ppl:.4f} "
          f"({100.0 * (default_ppl / best_ppl - 1.0):+.2f}% vs best)", flush=True)
    print(f"  ppl spread best→worst across sections: {spread:.2f}%", flush=True)

    if full:
        helps = best_ppl < base_ppl
        verdict = (f"SETTLE rope_section = {best_sec} (caption ppl {best_ppl:.4f}; "
                   f"{'beats' if helps else 'does NOT beat'} the {base_ppl:.4f} text-only baseline, so "
                   f"the image {'is informative' if helps else 'did not help — INSPECT'}; default "
                   f"{V.default_rope_section(hd)} is {100.0 * (default_ppl / best_ppl - 1.0):+.2f}% off "
                   f"best).")
        print(f"\nVERDICT: {verdict}", flush=True)
        _ck(all(math.isfinite(p) for _, p in results), "a section produced a non-finite caption ppl")
        _ck(math.isfinite(base_ppl), "text-only baseline ppl is non-finite")
        _ck(spread > 0.0, "rope_section made NO difference at scale (the knob is inert @ 397B — bug?)")
    else:
        print(f"\nSMOKE ok — arbiter ran ({mode}); ppls are NOT a verdict (synthetic image and/or a "
              f"truncated decoder). Run with no --layers and a real image+caption for the verdict.",
              flush=True)
    print(f"PARITY-CHECKS: {_N}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Settle the MiniMax-M3-VL vision rope_section (SOLO).")
    ap.add_argument("image", nargs="?", default=None, help="path to a REAL natural image (omit ⇒ smoke)")
    ap.add_argument("caption", nargs="?", default=None, help="a TRUE caption of the image")
    ap.add_argument("--layers", type=int, default=None, help="bounded decoder prefix (smoke)")
    a = ap.parse_args()
    run(a.image, a.caption, a.layers)
