"""Oracle gate: MLX compressed-layer attention + Lightning Indexer == the authors' real code (CPU).

Two checks against the authors' ``Attention`` / ``Indexer`` (inference/model.py):
  1. Full compressed attention at moderate length (ncomp <= index_topk, so the indexer selects all
     causal compressed tokens): MLX ``attention_compressed`` vs authors' ``Attention.forward`` for
     L2 (ratio 4, indexer) and L3 (ratio 128).
  2. Indexer **top-k selection** at long length (ncomp > index_topk=512, where it truly discriminates):
     the set of compressed tokens MLX selects must equal the authors' ``Indexer`` top-k, for late
     queries (causal_count > 512).

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_indexer_test
"""

from __future__ import annotations

import numpy as np
import torch

import mlx.core as mx

from quanta.dsv4 import attention as A
from quanta.dsv4 import indexer as I
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from parity import dsv4_torch_ref as ref

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def _f32(d):
    """Recursively cast a (possibly nested) loader param dict to float32."""
    return {k: (_f32(v) if isinstance(v, dict) else v.astype(mx.float32)) for k, v in d.items()}


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    M, args = ref.load_model_module(cfg, max_seq_len=2560)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    rng = np.random.default_rng(0)
    ok = True

    # --- 1) full compressed attention vs authors' Attention (moderate length) --
    print("=== compressed attention vs authors' Attention (T=256) ===")
    for L in (2, 3):
        attn_ref = ref.load_attention(M, args, cfg, L)
        p_f32 = _f32(ck.attention(L))
        ck.release()
        x = (rng.standard_normal((1, 256, cfg.hidden_size)) * 0.5).astype(np.float32)
        with torch.no_grad():
            o_ref = attn_ref(torch.from_numpy(x), 0).numpy().astype(np.float64)
        o_mx = np.array(I.attention_compressed(mx.array(x), p_f32, cfg, L).astype(mx.float32)).astype(np.float64)
        rel = float(np.max(np.abs(o_ref - o_mx))) / float(np.max(np.abs(o_ref)))
        good = o_mx.shape == o_ref.shape and rel < 1e-3
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] L{L} ratio={cfg.compress_ratio(L):3d}  rel={rel:.2e}")

    # --- 2) indexer top-k selection vs authors' Indexer (long: ncomp>512) ------
    print("=== indexer top-k selection vs authors' Indexer (T=2304, ncomp=576) ===")
    s = 2304
    attn_ref = ref.load_attention(M, args, cfg, 2)
    p_idx = _f32(ck.attention(2))
    idx_p = p_idx["indexer"]
    ck.release()
    x = (rng.standard_normal((1, s, cfg.hidden_size)) * 0.5).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        qr_ref = attn_ref.q_norm(attn_ref.wq_a(xt))                      # authors' qr
        ref_topk = attn_ref.indexer(xt, qr_ref, 0, s).numpy()           # [1,s,k], -1 / +offset(s)
    cos, sin = A.rope_tables(cfg, 2, s, 0)
    qr_mx, _, _ = A.project_qkv(mx.array(x), p_idx, cfg, cos, sin)
    iscore, ncomp = I.indexer_index_score(mx.array(x), qr_mx, idx_p, cfg, cos, sin)
    iscore = np.array(iscore[0].astype(mx.float32))                      # [s, ncomp]

    worst = 1.0
    for i in (600, 1500, 2100, 2303):                                   # mix of <512 and >512 causal
        causal = (i + 1) // 4
        k = min(cfg.index_topk, causal)
        my = set(np.argsort(iscore[i, :causal])[::-1][:k].tolist())
        rf = {int(v) - s for v in ref_topk[0, i] if v >= 0}
        overlap = len(my & rf) / max(len(rf), 1)
        worst = min(worst, overlap)
        print(f"    query {i:4d}: causal={causal} k={k} |my∩ref|/|ref|={overlap:.3f} "
              f"(|my|={len(my)} |ref|={len(rf)})")
    good = worst > 0.995
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] selection overlap >= 0.995 (worst={worst:.3f})")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
