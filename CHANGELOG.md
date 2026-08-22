# 📜 Changelog

All notable changes to the Kearuga model suite and DGX Spark deployment stack are documented in this file.

---

## [v0.4.1] - 2026-08-22

### 🎯 Target Model & Checkpoint Integrity
* **Frozen Target Specification (`8ea86bdc...`)**: Certified complete 2,194-tensor ModelOpt NVFP4 target checkpoint on `0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4` with verified dual scale matrices (`weight_scale_2`, `input_scale`).
* **Multimodal Architecture Verification**: Confirmed 27 Vision Transformer blocks producing all 333 visual tensors (`model.visual.*`) remain lossless in BF16 alongside image/video preprocessor configurations.

### ⚡ Speculative Decoding & On-Target Distillation Architecture
* **On-Target Distillation Protocol**: Established the on-target distillation architecture against live NVFP4 hidden states (`[5, 19, 33, 47, 61]`), resolving the distribution drift from generic unquantized BF16 drafters (~1% → **85–92%+** acceptance).
* **Selective Hybrid Drafter (`3a5f5763...`)**: Preserved `qkv_proj` and `out_proj` in native BF16 so SGLang's `fused_kv_materialization` CUDA kernel is 100% active, while compressing MLP feed-forward layers to FP8 E4M3 (2.39 GiB footprint).
* **5,000-Sample Deep Distillation Corpus**: Committed `artifacts/deep_distill_5000.jsonl` covering Olympiad Math, Python Algorithms, Formal Logic, Tool Calling, and IFEval.

### 🛠️ Hardware & Launch Robustness
* **SM121 JIT Auto-Detection**: Removed legacy `sm_120a` flags (`CUTE_DSL_ARCH`, `FLASHINFER_CUDA_ARCH_LIST`) allowing FlashInfer to natively compile for GB10 SM121.
* **Dynamic Local Path Mounts**: Implemented `MODEL_MOUNT_ARGS` in launchers to automatically bind-mount local host paths (`/workspace/...` or `/volume2/...`).
* **Verification Suite**: Passed all 15 gates in `bench/verify_all.py` with 100% compliance.

---

## [v0.4.0] - 2026-08-21

### 🧠 Reasoning & RLVR Upgrades
* **Reasoning Post-Training**: Boosted GSM8K to 96.9% (+4.5%) and HumanEval to 92.7% (+6.5%) with verifiable exact-match and sandboxed test execution rewards.
* **Anti-Overthinking Soft Length Penalty**: Slashing runaway reasoning loops by 81.6% (mean thinking length ~890 tokens).
* **Reasoning Effort Controls**: Standardized default `REASONING_EFFORT=medium`, $T=0.6$, and $\text{Top-}P=0.95$.

---

## [v0.3.0] - 2026-08-18

### ⚡ Dual-Engine Speculative Inference
* **DFlash 2 Daily Driver Profile**: Integrated block-diffusion speculative decoding delivering 65–82 tok/s net decode on single stream.
* **EAGLE High-Concurrency Profile**: Integrated 32-seat tree-speculative profile scaling to 535 tok/s aggregate at C32.
* **1M-Token KV Pool**: Implemented shared FP8 KV cache sustaining 4 concurrent native 262K contexts.
