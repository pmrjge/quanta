"""Localize the Qwen3.6 forward garbage-output bug — ONE model load, several cheap probes.

The bf16-dequant reference runtime produces degenerate repetition on every prompt. This script loads
the resident model once and probes WHERE the forward breaks, without an external reference:

  (1) prompt encoding sanity: decode the chat-encoded ids back to text (rule out a tokenizer bug).
  (2) per-layer residual stream stats during prefill (capture_layers=all): mean|·| / max|·| after
      every block — a blow-up/collapse/NaN at a specific layer localizes the bug to that layer's
      mixer (linear GatedDeltaNet vs full gated-GQA) or its MoE. Layer 0 is linear; layer 3 full.
  (3) logits distribution at the first decode position: top-5 tokens + the entropy — degenerate
      (near-constant argmax) vs plausible.
  (4) raw completion (NO chat template): "The capital of France is" — if even this degenerates from
      token 1 the bug is the forward, not the chat template.

~65 GB resident — run SOLO.

    uv run python -u -m parity.qwen36_forward_triage
"""

from __future__ import annotations

import mlx.core as mx

from quanta.qwen35.runtime import Qwen35ResidentModel
from quanta.qwen35.tokenizer import Qwen35Tokenizer

ART = "/Users/pmrj/models/Qwen3.6-35B-A3B-quanta_int4g64"


def run() -> None:
    mx.set_wired_limit(int(120 * 1024 ** 3))
    print(f"loading {ART} ...", flush=True)
    model = Qwen35ResidentModel(ART)
    tok = Qwen35Tokenizer.from_pretrained(ART)
    cfg = model.cfg
    print(f"loaded {model.num_layers} layers, resident≈{mx.get_active_memory()/1024**3:.1f} GiB", flush=True)

    # (1) prompt encoding sanity
    ids = tok.encode_chat([{"role": "user", "content": "What is the capital of France?"}],
                          add_generation_prompt=True)
    print(f"\n(1) chat prompt: {len(ids)} ids; round-trip decode:\n    {tok.decode(ids)!r}", flush=True)

    # (2) per-layer residual stats during prefill (capture every block)
    cap = list(range(model.num_layers))
    logits, caps = model(mx.array(ids), capture_layers=cap)
    mx.eval(logits, *caps.values())
    emb = model.embed_w[mx.array(ids)].astype(mx.float32)
    print(f"\n(2) embedding   mean|·|={float(mx.mean(mx.abs(emb))):.4f} max|·|={float(mx.max(mx.abs(emb))):.3f}",
          flush=True)
    print("    per-layer post-block residual (L=linear, F=full):", flush=True)
    for i in cap:
        h = caps[i].astype(mx.float32)
        kind = "F" if cfg.is_full_attention(i) else "L"
        nan = bool(mx.any(mx.isnan(h)).item())
        print(f"      L{i:02d}[{kind}]  mean|·|={float(mx.mean(mx.abs(h))):8.3f}  "
              f"max|·|={float(mx.max(mx.abs(h))):10.2f}  nan={nan}", flush=True)

    # (3) logits distribution at the last prompt position
    lg = logits[0, -1].astype(mx.float32)
    probs = mx.softmax(lg)
    top = mx.argsort(-lg)[:5]
    ent = float(-mx.sum(probs * mx.log(probs + 1e-9)).item())
    print(f"\n(3) final-norm mean|·|={float(mx.mean(mx.abs(mx.fast.rms_norm(logits[0,-1:], model.norm_w.astype(mx.bfloat16), cfg.norm_eps)))):.4f}",
          flush=True)
    print(f"    logits: mean={float(mx.mean(lg)):.3f} std={float(mx.std(lg)):.3f} entropy={ent:.3f} "
          f"(uniform≈{float(mx.log(mx.array(float(cfg.vocab_size)))):.2f})", flush=True)
    for t in top.tolist():
        print(f"      tok {t:>7} p={float(probs[t]):.4f}  {tok.decode([t])!r}", flush=True)

    # (4) raw completion (no chat template)
    raw = tok.encode("The capital of France is", add_bos=False)
    from quanta.qwen35.generate import generate
    out = generate(model, raw, max_new_tokens=12, temperature=0.0, eos_id=tuple(cfg.eos_token_ids))
    print(f"\n(4) raw completion ids={len(raw)} -> {tok.decode(out)!r}", flush=True)


if __name__ == "__main__":
    run()
