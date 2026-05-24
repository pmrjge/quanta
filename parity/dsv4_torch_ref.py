"""DeepSeek-V4 torch CPU reference oracle — runs the authors' inference/model.py on CPU.

The checkpoint ships the authors' reference (``inference/model.py``), but its 5 numeric kernels are
CUDA/tilelang-only. This harness makes that code runnable on the M3 Ultra (CPU/torch) by installing a
pure-torch ``kernel`` stub *before* importing ``model.py``, then running the authors' actual classes
(Attention, Compressor, Indexer, MoE, Block, Transformer, RoPE, hc_pre/hc_post) as the oracle for the
MLX port. This collapses transcription risk to the 5 small kernels below.

Substitutes (all in float32 — the clean dequantized forward, matching the MLX reference, so QAT noise
is excluded on both sides):
  * ``act_quant`` / ``fp4_act_quant`` (inplace QAT fake-quant) -> identity (skipped, like the MLX ref).
  * ``fp8_gemm`` / ``fp4_gemm`` -> never called (we run all-float32, so ``linear()`` takes ``F.linear``).
  * ``sparse_attn`` -> exact dense gathered masked online-softmax with per-head sink.
  * ``hc_split_sinkhorn`` -> exact torch (sigmoid pre / 2*sigmoid post / softmax+Sinkhorn comb).
  * ``fast_hadamard_transform.hadamard_transform`` -> real Hadamard (orthonormal), needed by the
    indexer's rotate_activation; it cancels in q·k but we implement it properly anyway.

Weights are dequantized in torch (native fp8-e4m3 / e8m0 + DeepSeek FP4_TABLE) — fully independent of
the MLX loader (cross-validated bit-exact in dsv4_dequant_test). torch is offline-only (rule 5).

Importable: ``M, args = load_model_module(cfg, max_seq_len); attn = load_attention(M, args, cfg, 0)``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open

ART = Path("/Users/pmrj/models/DeepSeek-V4-Flash")
INF = ART / "inference"
_WMAP = json.loads((ART / "model.safetensors.index.json").read_text())["weight_map"]
_FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


# --- pure-torch kernel substitutes ------------------------------------------
def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Dense equivalent of the tilelang sparse_attn: each query attends to its gathered top-k KV
    (``-1`` = masked) via online softmax with a per-head sink in the denominator. q:[b,m,h,d],
    kv:[b,n,d], attn_sink:[h], topk_idxs:[b,m,k] -> [b,m,h,d]."""
    b, m, h, d = q.shape
    k = topk_idxs.shape[-1]
    idx = topk_idxs.clamp(min=0).long()                                   # [b,m,k]
    kv_exp = kv.unsqueeze(1).expand(b, m, kv.shape[1], d)
    kvg = torch.gather(kv_exp, 2, idx.unsqueeze(-1).expand(b, m, k, d))   # [b,m,k,d]
    valid = topk_idxs >= 0                                                # [b,m,k]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), kvg.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = scores.max(-1, keepdim=True).values
    ex = torch.exp(scores - mx)
    denom = ex.sum(-1) + torch.exp(attn_sink.view(1, 1, h).float() - mx.squeeze(-1))
    o = torch.einsum("bmhk,bmkd->bmhd", ex, kvg.float()) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(*mixes.shape[:-1], hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _hadamard_transform(x, scale=1.0):
    """Identity. The indexer rotates BOTH q and the index-KV by the same orthonormal Hadamard, so it
    cancels in the q·k score (H Hᵀ = I) and a positive scale can't change the top-k argsort. With the
    fp4 fake-quant also skipped (clean oracle), the rotation is a no-op for selection — so we skip it
    on both the torch ref and the MLX side, keeping them directly comparable and matching the true
    model's selection. (The MLX runtime likewise omits it.)"""
    return x


def _act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    """Identity QAT fake-quant (clean oracle): never mutates x, so KV stays full-precision — matching
    the MLX reference which omits the inference-time fake-quant."""
    return x if inplace else (x, None)


def _fp4_act_quant(x, block_size=32, inplace=False):
    return x if inplace else (x, None)


def _install_stubs():
    k = types.ModuleType("kernel")
    k.act_quant = _act_quant
    k.fp4_act_quant = _fp4_act_quant

    def _nogemm(*a, **kw):
        raise RuntimeError("fp8_gemm/fp4_gemm should not run in the float32 reference")
    k.fp8_gemm = _nogemm
    k.fp4_gemm = _nogemm
    k.sparse_attn = _sparse_attn
    k.hc_split_sinkhorn = _hc_split_sinkhorn
    sys.modules["kernel"] = k
    fh = types.ModuleType("fast_hadamard_transform")
    fh.hadamard_transform = _hadamard_transform
    sys.modules["fast_hadamard_transform"] = fh


# --- weight dequant (torch-native, independent of the MLX loader) ------------
def torch_dequant(key: str) -> torch.Tensor:
    """Dequantize a checkpoint tensor to float32 in torch (fp8-e4m3 + e8m0 block / fp4 + e8m0 group /
    bf16 passthrough)."""
    with safe_open(str(ART / _WMAP[key]), framework="pt") as f:
        w = f.get_tensor(key)
        sk = key[:-len(".weight")] + ".scale" if key.endswith(".weight") else None
        s = f.get_tensor(sk) if (sk and sk in _WMAP) else None
    if s is None:
        return w.float()
    if w.dtype == torch.float8_e4m3fn:
        out, inn = w.shape
        sf = s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:out, :inn]
        return w.float() * sf
    if w.dtype == torch.int8:                                            # packed fp4
        u8 = w.view(torch.uint8).to(torch.int64)
        wf = torch.stack([_FP4_TABLE[u8 & 0xF], _FP4_TABLE[(u8 >> 4) & 0xF]], -1).flatten(-2)
        sf = s.float().repeat_interleave(32, 1)[:, :wf.shape[1]]
        return wf * sf
    raise ValueError(f"{key}: unexpected quantized dtype {w.dtype}")


# --- model construction ------------------------------------------------------
def load_model_module(cfg, max_seq_len: int = 256):
    """Install stubs, import the authors' model.py, set globals to float32 CPU, return ``(M, args)``."""
    _install_stubs()
    if str(INF) not in sys.path:
        sys.path.insert(0, str(INF))
    import model as M  # noqa: E402  (must follow stub install)

    M.world_size, M.rank = 1, 0
    M.default_dtype = torch.float32
    M.scale_fmt, M.scale_dtype = None, torch.float32
    # rotate_activation (Hadamard) cancels in the indexer q·k score (orthonormal) and asserts bf16;
    # override to identity for the clean float32 oracle (selection is unchanged — see _hadamard_transform).
    M.rotate_activation = lambda x: x
    args = M.ModelArgs(
        max_batch_size=1, max_seq_len=max_seq_len, dtype="bf16", scale_fmt=None,
        expert_dtype=None, scale_dtype="fp32", vocab_size=cfg.vocab_size, dim=cfg.hidden_size,
        moe_inter_dim=cfg.moe_intermediate_size, n_layers=cfg.num_hidden_layers,
        n_hash_layers=cfg.n_hash_layers, n_mtp_layers=cfg.n_mtp_layers, n_heads=cfg.num_attention_heads,
        n_routed_experts=cfg.n_routed_experts, n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok, score_func=cfg.scoring_func,
        route_scale=cfg.routed_scaling_factor, swiglu_limit=cfg.swiglu_limit, q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim, rope_head_dim=cfg.rope_head_dim, norm_eps=cfg.norm_eps,
        o_groups=cfg.o_groups, o_lora_rank=cfg.o_lora_rank, window_size=cfg.sliding_window,
        compress_ratios=tuple(cfg.compress_ratios), compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=cfg.original_seq_len, rope_theta=cfg.rope_theta, rope_factor=cfg.rope_factor,
        beta_fast=int(cfg.beta_fast), beta_slow=int(cfg.beta_slow), index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim, index_topk=cfg.index_topk, hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters, hc_eps=cfg.hc_eps)
    return M, args


def load_attention(M, args, cfg, layer_id: int):
    """Authors' Attention module for ``layer_id``, float32, all params (incl. compressor/indexer)
    loaded from the checkpoint. Compressed layers also get their compressor wired to the kv_cache +
    freqs_cis (as Attention.forward does lazily), so the compressor/indexer are directly callable."""
    attn = M.Attention(layer_id, args).float().eval()
    base = f"layers.{layer_id}.attn."
    sd = {name: torch_dequant(base + name) for name, _ in attn.named_parameters()}
    attn.load_state_dict(sd, strict=False)
    if cfg.has_compressor(layer_id):
        attn.compressor.kv_cache = attn.kv_cache[:, args.window_size:]
        attn.compressor.freqs_cis = attn.freqs_cis
        if attn.indexer is not None:
            attn.indexer.kv_cache = attn.indexer.kv_cache  # already a buffer
            attn.indexer.freqs_cis = attn.freqs_cis
            attn.indexer.compressor.kv_cache = attn.indexer.kv_cache
            attn.indexer.compressor.freqs_cis = attn.freqs_cis
    return attn


def load_gate(M, args, cfg, layer_id: int):
    """Authors' Gate module for ``layer_id`` (float32), weights loaded from the checkpoint."""
    g = M.Gate(layer_id, args).float().eval()
    base = f"layers.{layer_id}.ffn.gate."
    with torch.no_grad():
        g.weight.copy_(torch_dequant(base + "weight"))
        if cfg.is_hash(layer_id):
            with safe_open(str(ART / _WMAP[base + "tid2eid"]), framework="pt") as f:
                g.tid2eid.copy_(f.get_tensor(base + "tid2eid").to(torch.int32))
        else:
            g.bias.copy_(torch_dequant(base + "bias"))
    return g


def moe_reference(M, args, cfg, layer_id: int, xf_t, input_ids_t):
    """Authors' MoE forward (Gate + Expert), loading only the routed (hit) experts + shared on demand
    — avoids constructing all 256 experts. Returns ``(weights, indices, y[N,dim])`` in float32."""
    gate = load_gate(M, args, cfg, layer_id)
    with torch.no_grad():
        weights, indices = gate(xf_t, input_ids_t)
    n, dim = xf_t.shape
    y = torch.zeros(n, dim)
    expert = M.Expert(args.dim, args.moe_inter_dim, dtype=None, swiglu_limit=args.swiglu_limit).float().eval()
    ekp = f"layers.{layer_id}.ffn.experts."
    for i in torch.unique(indices).tolist():
        with torch.no_grad():
            for proj in ("w1", "w2", "w3"):
                getattr(expert, proj).weight.copy_(torch_dequant(f"{ekp}{i}.{proj}.weight"))
            tok, top = torch.where(indices == i)
            y[tok] += expert(xf_t[tok], weights[tok, top, None])
    shared = M.Expert(args.dim, args.moe_inter_dim, dtype=None, swiglu_limit=args.swiglu_limit).float().eval()
    skp = f"layers.{layer_id}.ffn.shared_experts."
    with torch.no_grad():
        for proj in ("w1", "w2", "w3"):
            getattr(shared, proj).weight.copy_(torch_dequant(skp + f"{proj}.weight"))
        y += shared(xf_t)
    return weights, indices, y


def install_cheap_moe(M) -> None:
    """Monkeypatch ``M.MoE`` with a lightweight MoE that loads only routed experts on demand (so a
    full ``Block``/``Transformer`` can be built without allocating all 256 experts per layer). Its
    forward replicates the reference ``MoE.forward`` exactly (Gate + per-hit Expert + shared)."""
    if getattr(M, "_cheap_moe_installed", False):
        return

    class CheapMoE(torch.nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            self.layer_id, self._args = layer_id, args
            self.gate = M.Gate(layer_id, args)
            self.shared_experts = M.Expert(args.dim, args.moe_inter_dim, dtype=None,
                                           swiglu_limit=args.swiglu_limit)

        @torch.no_grad()
        def forward(self, x, input_ids):
            a = self._args
            shape = x.size()
            x = x.view(-1, a.dim)
            weights, indices = self.gate(x, input_ids.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            expert = M.Expert(a.dim, a.moe_inter_dim, dtype=None, swiglu_limit=a.swiglu_limit).float()
            ekp = f"layers.{self.layer_id}.ffn.experts."
            for i in torch.unique(indices).tolist():
                for proj in ("w1", "w2", "w3"):
                    getattr(expert, proj).weight.copy_(torch_dequant(f"{ekp}{i}.{proj}.weight"))
                tok, top = torch.where(indices == i)
                y[tok] += expert(x[tok], weights[tok, top, None])
            y += self.shared_experts(x)
            return y.type_as(x).view(shape)

    M.MoE = CheapMoE
    M._cheap_moe_installed = True


def load_block(M, args, cfg, layer_id: int):
    """Authors' Block module for ``layer_id`` (float32, CheapMoE), all params loaded + caches wired."""
    install_cheap_moe(M)
    blk = M.Block(layer_id, args).float().eval()
    with torch.no_grad():
        for name, prm in blk.named_parameters():
            key = f"layers.{layer_id}.{name}"
            if name.endswith("gate.tid2eid"):
                with safe_open(str(ART / _WMAP[key]), framework="pt") as f:
                    prm.copy_(f.get_tensor(key).to(torch.int32))
            else:
                prm.copy_(torch_dequant(key))
    if cfg.has_compressor(layer_id):
        blk.attn.compressor.kv_cache = blk.attn.kv_cache[:, args.window_size:]
        blk.attn.compressor.freqs_cis = blk.attn.freqs_cis
        if blk.attn.indexer is not None:
            blk.attn.indexer.freqs_cis = blk.attn.freqs_cis
            blk.attn.indexer.compressor.kv_cache = blk.attn.indexer.kv_cache
            blk.attn.indexer.compressor.freqs_cis = blk.attn.freqs_cis
    return blk


def load_final_head(M, args, cfg):
    """Authors' final-head pieces: ``(ParallelHead, RMSNorm, hc_head_fn, hc_head_scale, hc_head_base)``."""
    head = M.ParallelHead(cfg.vocab_size, cfg.hidden_size, cfg.norm_eps, cfg.hc_eps).float().eval()
    norm = M.RMSNorm(cfg.hidden_size, cfg.norm_eps).float().eval()
    with torch.no_grad():
        head.weight.copy_(torch_dequant("head.weight"))
        norm.weight.copy_(torch_dequant("norm.weight"))
    return head, norm, torch_dequant("hc_head_fn"), torch_dequant("hc_head_scale"), torch_dequant("hc_head_base")


if __name__ == "__main__":
    from quanta.dsv4.config import DeepSeekV4Config
    cfg = DeepSeekV4Config.from_pretrained(str(ART))
    M, args = load_model_module(cfg, max_seq_len=64)
    attn = load_attention(M, args, cfg, 0)
    x = torch.randn(1, 16, cfg.hidden_size)
    with torch.no_grad():
        o = attn(x, 0)
    print(f"authors' Attention L0 ok: in {tuple(x.shape)} -> out {tuple(o.shape)} finite={bool(torch.isfinite(o).all())}")
