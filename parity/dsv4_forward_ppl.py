"""#74 e2e gate: the full 43-layer MLX forward == the authors' code chained across all layers (CPU).

Runs the complete streamed MLX forward (:func:`quanta.dsv4.model.dsv4_logits`) and compares its logits
to the authors' real forward built by chaining ``load_block`` over all 43 layers (embed -> HC-expand
-> blocks -> HC-head -> RMSNorm -> lm_head), on a short sequence. This is the strongest correctness
gate for the whole text model. Also reports the teacher-forced perplexity of the sequence as a smoke
number (a *quality* ppl on real prose awaits the BPE tokenizer, #75).

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_forward_ppl [seqlen]

NOTE: streams every layer's experts (fp4->f32) — minutes, but memory-disciplined (≤1 layer resident).
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import mlx.core as mx

from quanta.dsv4 import model as MODEL
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from parity import dsv4_torch_ref as ref

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def _torch_forward(M, args, cfg, ids_t):
    """Authors' forward, chaining one Block at a time (on-demand experts), float32 -> logits [1,S,V]."""
    emb = ref.torch_dequant("embed.weight")
    h = emb[ids_t[0]][None]                                       # [1,S,dim]
    h = h.unsqueeze(2).expand(1, h.shape[1], cfg.hc_mult, cfg.hidden_size).contiguous()
    for L in range(cfg.num_hidden_layers):
        blk = ref.load_block(M, args, cfg, L)
        with torch.no_grad():
            h = blk(h, 0, ids_t)
        del blk
    head, norm, fn, scale, base = ref.load_final_head(M, args, cfg)
    with torch.no_grad():
        red = head.hc_head(h, fn, scale, base)
        return (norm(red).float() @ head.weight.float().T).numpy().astype(np.float64)


def run() -> None:
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = DeepSeekV4Config.from_pretrained(ART)
    M, args = ref.load_model_module(cfg, max_seq_len=max(64, s))
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(1, s)).astype(np.int64)

    print(f"[dsv4 forward] streaming MLX full forward (43 layers), S={s} ...", flush=True)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    logits_mx = MODEL.dsv4_logits(ck, mx.array(ids), cfg, dtype=mx.float32)
    mx.eval(logits_mx)
    lm = np.array(logits_mx.astype(mx.float32)).astype(np.float64)

    print("[dsv4 forward] chaining authors' Blocks (oracle) ...", flush=True)
    lr = _torch_forward(M, args, cfg, torch.from_numpy(ids))

    rel = float(np.max(np.abs(lr - lm))) / float(np.max(np.abs(lr)))
    finite = bool(np.isfinite(lm).all())

    # teacher-forced ppl of this (arbitrary) sequence — smoke number, not a quality claim
    lg = lm[0]
    tgt = ids[0, 1:]
    lse = np.log(np.exp(lg[:-1] - lg[:-1].max(-1, keepdims=True)).sum(-1)) + lg[:-1].max(-1)
    ce = (lse - lg[:-1][np.arange(s - 1), tgt]).mean()
    ppl = float(np.exp(ce))

    ok = finite and lm.shape == (1, s, cfg.vocab_size) and rel < 2e-3
    print(f"  logits {lm.shape} finite={finite}  rel(MLX vs authors' chained)={rel:.2e}")
    print(f"  teacher-forced ppl (random ids, smoke)={ppl:.1f}  argmax[0,:5]={lm[0, :5].argmax(-1).tolist()}")
    print("PASS (full forward == authors' code)" if ok else "FAIL")


if __name__ == "__main__":
    run()
