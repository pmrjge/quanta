"""Nemotron-H (NVIDIA-Nemotron-3-Super-120B-A12B) support: hybrid Mamba-2/attention/MoE.

Scaffold so far: config dataclass, HF-tokenizer encode/chat wrapper, and the per-tensor
quantization policy (the int4/int8/bf16 mix). The Mamba-2 SSD runtime + bake orchestration
land in later steps; these pieces are pure-Python/json + the tokenizers lib and carry no
dependency on the not-yet-built runtime.
"""
