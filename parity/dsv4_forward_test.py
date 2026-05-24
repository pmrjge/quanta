"""Oracle gate: MLX block + final-head == the authors' real Block.forward / head (model.py on CPU).

The capstone assembly check. Each decoder block (HC-pre -> norm -> attention -> HC-post -> HC-pre ->
norm -> MoE -> HC-post) is validated against the authors' ``Block.forward`` for all three layer types
(L0 dense+hash, L2 compressed+indexer+hash, L3 window+score), and the final head (HC-head -> RMSNorm
-> lm_head) against the authors' ``ParallelHead``. Together with the per-component gates, this proves
the full forward equals the authors' code. (Running the 43-layer streamed forward end-to-end is
``parity/dsv4_forward_ppl.py``.)

    uv run --with torch --with safetensors --with numpy python -m parity.dsv4_forward_test
"""

from __future__ import annotations

import numpy as np
import torch

import mlx.core as mx

from quanta.dsv4 import model as MODEL
from quanta.dsv4.config import DeepSeekV4Config
from quanta.dsv4.hyper import hc_head
from quanta.dsv4.attention import _rms_w
from quanta.dsv4.loader import DeepSeekV4SourceCheckpoint
from parity import dsv4_torch_ref as ref

ART = "/Users/pmrj/models/DeepSeek-V4-Flash"


def _rel(a_np, b_mx):
    b = np.array(b_mx.astype(mx.float32)).astype(np.float64)
    return float(np.max(np.abs(a_np - b))) / float(np.max(np.abs(a_np)))


def run() -> None:
    cfg = DeepSeekV4Config.from_pretrained(ART)
    M, args = ref.load_model_module(cfg, max_seq_len=64)
    ck = DeepSeekV4SourceCheckpoint(ART, cfg)
    rng = np.random.default_rng(0)
    s, hc, dim = 8, cfg.hc_mult, cfg.hidden_size
    ids = rng.integers(0, cfg.vocab_size, size=(1, s)).astype(np.int64)
    ok = True

    # --- (a) per-block stitching vs authors' Block.forward ---------------------
    print("=== block stitching vs authors' Block.forward (T=8) ===")
    for L in (0, 2, 3):
        h = (rng.standard_normal((1, s, hc, dim)) * 0.3).astype(np.float32)
        blk = ref.load_block(M, args, cfg, L)
        with torch.no_grad():
            o_ref = blk(torch.from_numpy(h), 0, torch.from_numpy(ids.reshape(1, s))).numpy().astype(np.float64)
        p = MODEL.load_block_params(ck, cfg, L, dtype=mx.float32)
        o_mx = MODEL.dsv4_block(mx.array(h), p, cfg, L, mx.array(ids))
        rel = _rel(o_ref, o_mx)
        good = o_mx.shape == (1, s, hc, dim) and rel < 1e-3
        ok = ok and good
        kind = ("dense+hash" if not cfg.has_compressor(L) and cfg.is_hash(L)
                else "compressed+indexer+hash" if cfg.has_indexer(L) and cfg.is_hash(L)
                else "compressed+score" if cfg.has_compressor(L) else "dense+score")
        print(f"  [{'OK' if good else 'FAIL'}] L{L} {kind:24s} rel={rel:.2e}")
        ck.release()

    # --- (b) final head (HC-head -> RMSNorm -> lm_head) ------------------------
    print("=== final head vs authors' ParallelHead ===")
    h = (rng.standard_normal((1, s, hc, dim)) * 0.3).astype(np.float32)
    head_t, norm_t, fn_t, scale_t, base_t = ref.load_final_head(M, args, cfg)
    with torch.no_grad():
        red = head_t.hc_head(torch.from_numpy(h), fn_t, scale_t, base_t)
        logits_ref = (norm_t(red).float() @ head_t.weight.float().T).numpy().astype(np.float64)
    fhc = ck.final_hc()
    hh = hc_head(mx.array(h), fhc["fn"].astype(mx.float32), fhc["scale"].astype(mx.float32),
                 fhc["base"].astype(mx.float32), cfg.hc_mult, cfg.norm_eps, cfg.hc_eps)
    hh = _rms_w(hh, ck.final_norm().astype(mx.float32), cfg.norm_eps)
    logits_mx = hh @ ck.head().astype(mx.float32).T
    ck.release()
    rel = _rel(logits_ref, logits_mx)
    good = logits_mx.shape == (1, s, cfg.vocab_size) and rel < 1e-3
    ok = ok and good
    print(f"  [{'OK' if good else 'FAIL'}] head -> logits{logits_mx.shape}  rel={rel:.2e}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    run()
