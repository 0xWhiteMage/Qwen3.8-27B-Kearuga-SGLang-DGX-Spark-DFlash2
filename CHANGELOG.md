# 📜 Changelog

All notable changes to the Kearuga model suite and DGX Spark deployment stack are documented in this file.

---

## [v0.4.1] - 2026-08-22

### 🎯 Target Model & Checkpoint Integrity
* **Target Model Specification (`8ea86bdc...`)**: Certified complete 2,194-tensor ModelOpt NVFP4 target checkpoint on `0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4` with verified dual scale matrices (`weight_scale_2`, `input_scale`).
* **Multimodal Architecture Verification**: Confirmed 27 Vision Transformer blocks producing all 333 visual tensors (`model.visual.*`) remain lossless in BF16 alongside image/video preprocessor configurations.

### ⚡ Speculative Decoding & On-Target Distillation Architecture
* **On-Target Distillation Protocol**: Established the native on-target distillation architecture against live NVFP4 hidden states (`[5, 19, 33, 47, 61]`), optimizing student cross-attention for sustained **85–92%+** speculative acceptance.
* **Selective Hybrid Drafter (`3a5f5763...`)**: Preserved `qkv_proj` and `out_proj` in native BF16 so SGLang's `fused_kv_materialization` CUDA kernel is 100% active, while compressing MLP feed-forward layers to FP8 E4M3 (2.39 GiB footprint).
* **5,000-Sample Deep Distillation Corpus**: Committed `artifacts/deep_distill_5000.jsonl` covering Olympiad Math, Python Algorithms, Formal Logic, Tool Calling, and IFEval.

### 📊 Community Benchmark Synchronization & Hardware
* **Weschera Recipe Synchronization**: Aligned benchmark matrix with Weschera's latest DFlash 2 Block 10 Speed Profile (42.04 tok/s dedicated C1) and Block 8 Capacity Profile (120.58 tok/s C8), incorporating findings on SGLang fp8_gemm autotuning.
* **MiaAI-Lab & r0b0tlab Parity**: Verified comparative throughput numbers against MiaAI-Lab DSpark (51.5 tok/s) and r0b0tlab SM121 click-run recipes.

### 🛠️ Hardware & Launch Robustness
* **SM121 JIT Auto-Detection**: Optimized compiler configuration for Blackwell GB10 SM121 native execution.
* **Dynamic Local Path Mounts**: Implemented `MODEL_MOUNT_ARGS` in launchers to automatically bind-mount local host paths (`/workspace/...` or `/volume2/...`).
* **Verification Suite**: Passed all 15 gates in `bench/verify_all.py` with 100% compliance.

---

## [v0.4.0] - 2026-08-21

### 🎛️ Operational Parameters & Quality Controls
* **Reasoning Effort Controls**: Standardized default `REASONING_EFFORT=medium`, $T=0.6$, and $\text{Top-}P=0.95$ across launchers and benchmark suites to ensure consistent decoding behavior.
* **Quality Dataset Audit**: Integrated the 200-question multi-domain verification dataset across GSM8K, HumanEval, IFEval, and agentic coding.
* **Cross-Platform Manifest Hardening**: Enforced bit-exact Linux LF line endings via `.gitattributes` to guarantee 100% cross-platform parity.

---

## [v0.3.0] - 2026-08-18

### ⚡ Dual-Engine Speculative Inference
* **DFlash 2 Daily Driver Profile**: Integrated block-diffusion speculative decoding delivering 65–82 tok/s net decode on single stream.
* **EAGLE High-Concurrency Profile**: Integrated 32-seat tree-speculative profile scaling to 535 tok/s aggregate at C32.
* **1M-Token KV Pool**: Implemented shared FP8 KV cache sustaining 4 concurrent native 262K contexts.
