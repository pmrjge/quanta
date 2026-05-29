"""Layer-by-layer parity: quanta Qwen35 forward vs the HF qwen3_5_moe oracle — localize the bug.

The bf16-dequant reference forward is garbage (0/32) and the weights are verified correct, so the bug
is a forward-MATH convention error. This diffs the per-layer residual stream against the HF
transformers reference (the external oracle the qwen35 module's gates never had). The FIRST layer whose
residual diverges beyond fp/quant tolerance localizes the broken op.

Run in TWO phases (separate processes ⇒ torch freed before mlx loads — memory-safe):

    uv run --with transformers --with torch python -u -m parity.qwen36_layer_parity hf     # → npz
    uv run python -u -m parity.qwen36_layer_parity diff                                    # quanta + diff

Both encode the SAME text with the quanta tokenizer (single id source). HF runs on CPU bf16 with hooks
on the first K decoder layers + an early abort (skip the slow full-depth CPU MoE). Layer 0 is GDN
(linear); layer 3 is the first full-attention layer — so the first divergence cleanly separates a
GatedDeltaNet bug from an attention bug from a MoE/shared bug.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from quanta.qwen35.tokenizer import Qwen35Tokenizer

SRC = "/Users/pmrj/models/Qwen3.6-35B-A3B"
ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"
NPZ = "/tmp/qwen36_hf_caps.npz"
TEXT = "The capital of France is Paris. Water is made of hydrogen and oxygen."
K = 7  # capture decoder layers 0..K-1 (GDN 0,1,2 / full 3 / GDN 4,5 / full ... ) then abort


def _ids() -> list[int]:
    tok = Qwen35Tokenizer.from_pretrained(ART)
    return tok.encode(TEXT, add_bos=False)


class _StopForward(Exception):
    pass


def phase_hf() -> None:
    import torch
    import transformers as tf

    ids = _ids()
    print(f"HF oracle: {len(ids)} tokens; loading {SRC} (CPU bf16) ...", flush=True)
    cfg = tf.AutoConfig.from_pretrained(SRC)
    cls = getattr(tf, cfg.architectures[0])
    model = cls.from_pretrained(SRC, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                attn_implementation="eager")
    model.eval()
    text = model.model.language_model                      # embed_tokens + layers[40] + norm
    layers = text.layers

    caps: dict[str, np.ndarray] = {}
    hooks = []

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[f"L{i}"] = h.detach().float().cpu().numpy()[0]   # [T, hidden]
            if i == K - 1:
                raise _StopForward
        return hook

    for i in range(K):
        hooks.append(layers[i].register_forward_hook(mk(i)))

    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        emb = text.embed_tokens(inp).float().cpu().numpy()[0]     # [T, hidden] (input to layer 0)
        try:
            text(input_ids=inp, use_cache=False)
        except _StopForward:
            pass
    for h in hooks:
        h.remove()
    np.savez(NPZ, ids=np.array(ids), embed=emb, **caps)
    print(f"saved HF embed + layers 0..{K-1} to {NPZ}", flush=True)


def phase_diff() -> None:
    from quanta.qwen35.runtime import Qwen35ResidentModel

    d = np.load(NPZ)
    ids = [int(x) for x in d["ids"]]
    mx.set_wired_limit(int(120 * 1024 ** 3))
    model = Qwen35ResidentModel(ART)
    _, caps = model(mx.array(ids), capture_layers=list(range(K)))
    mx.eval(*caps.values())

    def rel(a_np, b_mx) -> float:
        a = mx.array(np.asarray(a_np, dtype=np.float32))
        b = b_mx.astype(mx.float32)
        return float((mx.max(mx.abs(a - b)) / (mx.max(mx.abs(a)) + 1e-6)).item())

    qe = model.embed_w[mx.array(ids)].astype(mx.float32)
    print(f"\nper-layer residual rel-err vs HF oracle ({len(ids)} tokens):", flush=True)
    print(f"  embed (input to L0)   rel={rel(d['embed'], qe):.3e}  "
          f"(≈0 ⇒ aligned input)", flush=True)
    first_bad = None
    for i in range(K):
        r = rel(d[f"L{i}"], caps[i])
        kind = "F" if model.cfg.is_full_attention(i) else "L"
        flag = "  <== FIRST DIVERGENCE" if (first_bad is None and r > 0.15) else ""
        if flag:
            first_bad = i
        print(f"  L{i:02d}[{kind}] rel={r:.3e}{flag}", flush=True)
    if first_bad is None:
        print("  (no divergence in captured layers — extend K / look at MoE-vs-mixer split)", flush=True)
    else:
        k = "GatedDeltaNet (linear)" if model.cfg.is_linear_attention(first_bad) else "gated-GQA (full)"
        print(f"\n>>> bug first appears at layer {first_bad} [{k}] — bisect mixer vs MoE there.", flush=True)


def run() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "diff"
    if mode == "hf":
        phase_hf()
    elif mode == "diff":
        phase_diff()
    else:
        raise SystemExit("mode must be 'hf' or 'diff'")


if __name__ == "__main__":
    run()
