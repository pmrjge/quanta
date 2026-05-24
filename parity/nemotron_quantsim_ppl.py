"""Nemotron-H expert-quant scheme selector — e2e teacher-forced ppl, no re-bake (#38).

Streams the bf16 source one layer resident (rule-8) and applies a candidate quant scheme to
each weight *on the fly* (quantize -> dequantize roundtrip), so several expert schemes can be
compared e2e without baking each. Dense always-on linears are held at int8 (the artifact's
floor) across all schemes, so the ppl delta isolates the **expert** scheme. Reuses the proven
``streamed_logits`` forward via a duck-typed checkpoint wrapper — same PROSE/tokens/fp32 head
as the bf16 reference (5.981), so numbers compare directly.

This is the parity-first answer to "which expert quant holds quality": AWQ misfired on the
relu^2 down-proj (+75% ppl), so here we measure plain affine int4 / int4-g64 / nvfp4-g16 /
int8 e2e and pick by ppl, not by per-projection reconstruction.

    uv run --with tokenizers python -m parity.nemotron_quantsim_ppl
"""

from __future__ import annotations

import time

import mlx.core as mx

from parity.nemotron_ppl import PROSE, streamed_logits
from quanta.nemotron.config import NemotronHConfig
from quanta.nemotron.loader import (
    ATTENTION_SUFFIXES,
    MAMBA_SUFFIXES,
    MOE_NONEXPERT_SUFFIXES,
    NemotronSourceCheckpoint,
)
from quanta.nemotron.tokenizer import NemotronTokenizer

SRC = "/Users/pmrj/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
BF16_PPL = 5.981

# Suffixes that are int8-affine dense linears in the bake (everything else in the per-kind dict
# is SSM core / norm / router / bias → bf16 passthrough).
_DENSE_INT8 = {
    "in_proj.weight", "out_proj.weight",
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "fc1_latent_proj.weight", "fc2_latent_proj.weight",
    "shared_experts.up_proj.weight", "shared_experts.down_proj.weight",
}

# Expert scheme: (mode, group_size, bits). Dense is always int8 g128.
SCHEMES: dict[str, tuple[str, int, int]] = {
    "int4_affine_g128": ("affine", 128, 4),
    "int4_affine_g64": ("affine", 64, 4),
    "nvfp4_g16": ("nvfp4", 16, 4),
    "int8_g128": ("affine", 128, 8),
}


def _roundtrip(w: mx.array, mode: str, gs: int, bits: int) -> mx.array:
    """Quantize then dequantize back to bf16 — simulate a baked weight's runtime value."""
    r = mx.quantize(w.astype(mx.bfloat16), group_size=gs, bits=bits, mode=mode)
    return mx.dequantize(*r, group_size=gs, bits=bits, mode=mode).astype(mx.bfloat16)


class _QuantSim:
    """Duck-types NemotronSourceCheckpoint: int8 on dense linears, scheme on experts, rest bf16."""

    def __init__(self, src: NemotronSourceCheckpoint, mode: str, gs: int, bits: int) -> None:
        self.src, self.mode, self.gs, self.bits = src, mode, gs, bits

    def read(self, key: str) -> mx.array:
        return self.src.read(key)  # embed / norm_f / head (bf16)

    def release(self) -> None:
        self.src.release()

    def _dense(self, d: dict[str, mx.array]) -> dict[str, mx.array]:
        return {k: (_roundtrip(v, "affine", 128, 8) if k in _DENSE_INT8 else v) for k, v in d.items()}

    def mamba_tensors(self, i: int) -> dict[str, mx.array]:
        return self._dense(self.src.mamba_tensors(i))

    def attention_tensors(self, i: int) -> dict[str, mx.array]:
        return self._dense(self.src.attention_tensors(i))

    def moe_nonexpert_tensors(self, i: int) -> dict[str, mx.array]:
        return self._dense(self.src.moe_nonexpert_tensors(i))

    def expert_stacks(self, i: int, n: int) -> dict[str, mx.array]:
        es = self.src.expert_stacks(i, n)
        out = {p: _roundtrip(es[p], self.mode, self.gs, self.bits) for p in ("up", "down")}
        mx.eval(list(out.values()))
        return out


def run() -> None:
    cfg = NemotronHConfig.from_pretrained(SRC)
    tok = NemotronTokenizer(SRC)
    ids = mx.array(tok.encode(PROSE, add_bos=False)[:192])
    targets = ids[1:]
    print(f"=== Nemotron-H expert-quant scheme sweep (dense int8; tokens={ids.shape[0]}) ===", flush=True)
    print(f"bf16 reference ppl   : {BF16_PPL:.3f}\n", flush=True)
    _ = MAMBA_SUFFIXES, ATTENTION_SUFFIXES, MOE_NONEXPERT_SUFFIXES  # (suffix sets documented above)
    for name, (mode, gs, bits) in SCHEMES.items():
        src = NemotronSourceCheckpoint(SRC)
        t0 = time.perf_counter()
        logits = streamed_logits(_QuantSim(src, mode, gs, bits), cfg, ids)
        lg = logits[:-1].astype(mx.float32)
        ce = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0]
        ppl = mx.exp(ce.mean()).item()
        acc = (mx.argmax(lg, axis=-1) == targets).astype(mx.float32).mean().item()
        print(f"{name:18s}: ppl {ppl:7.3f}  (Δ {100 * (ppl - BF16_PPL) / BF16_PPL:+5.1f}%)  "
              f"top1 {acc:.3f}  [{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)
        del src
        mx.clear_cache()


if __name__ == "__main__":
    run()
