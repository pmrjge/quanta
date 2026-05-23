"""Bake pipeline: quantize the source checkpoint into a self-contained resident artifact.

Policy (per CLAUDE.md + project decision): routed experts → int3 GPTQ (error-feedback,
group-128); non-experts (attention, dense L0 MLP, lm_head) → affine int8 (group-128);
shared expert, norms, router control tensors → bf16/fp32. Output is an immutable,
relocatable bundle (config.json + manifest + safetensors, relative refs only) the
resident runtime loads and decodes with mx.gather_qmm / mx.quantized_matmul.
"""
