"""Qwen3.5-397B-A17B (``qwen3_5_moe``) support for quanta.

A **3:1 hybrid** text decoder — 45 Gated-DeltaNet *linear-attention* layers (O(1) recurrent
state) interleaved 3-to-1 with 15 gated-GQA *full-attention* layers — over a 512-expert top-10
**softmax** MoE (+ 1 shared expert) on every layer, plus 1 native MTP module for speculative
decode. Source is **bf16** and **multimodal**: the text decoder lives under
``model.language_model.*`` and a 27-block ViT under ``model.visual.*`` (vision is a *deferred*
stage; quanta bakes the language model, cf. the upstream ``--language-model-only`` mode).

The baked artifact targets **1,010,000-token context** via **length-adaptive (dynamic) YaRN**:
the official ``factor=4`` YaRN over the native 262,144 window is applied *only* once a sequence
exceeds the native length, so short prompts keep native quality while long jobs reach 1M (the
upstream warning that *static* YaRN degrades short context — see README — is sidestepped because
quanta owns the RoPE path).

Submodules are imported directly (``quanta.qwen35.config``, ``quanta.qwen35.loader``,
``quanta.qwen35.tokenizer``, …) to keep package import side-effect-free and avoid pulling MLX in
before it is needed.
"""
