# Changelog

All notable updates and improvements to the **Qwen3.8-27B Kearuga on DGX Spark** recipe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-08-21

### Added
- **Native Zero-Overlay DSpark Profile (`start-dspark.sh`)**: Added a standalone launcher for `RadixArk/Qwen3.8-27B-DSpark` (block size 7) that runs directly on the unpatched base image without requiring custom image builds.
- **262K Needle-In-A-Haystack Benchmark (`bench/niah.py`)**: Token-accurate long-context retrieval verification probe testing 25%, 50%, and 90% depths up to 262,144 tokens.
- **10-Point Deterministic Semantic Gate (`bench/semantic_gate.py`)**: Automated verification script checking for template token leakage (`<|im_start|>`, `<|im_end|>`), JSON schema compliance, logical deduction, and arithmetic canary (`19 × 23 → 437`).
- **EXL3 Sensitivity Insights in `INSIGHTS.md`**: Added tensor sensitivity hierarchy mapping (Tier 1 BF16 Head/Embeddings, Tier 2 FP8 Attention/GDN, Tier 3 NVFP4 MLPs) based on recent EXL3 mixed-precision research to bridge the fidelity gap with near-lossless reasoning.
- **Unit Test Suite in Image Build**: Added `overlay-dflash2/test/` to the Docker build workflow to verify kernel math and selector operations during image generation.

### Changed
- **DFlash 2 Zero-Allocation Candidate Projection**: Refactored `sglang/srt/models/dflash.py` to eliminate dynamic PyTorch memory allocation during speculative candidate projection, reducing memory allocator overhead during C1–C4 decode loops.
- **DGX Spark (GB10) Hardware Flag Hardening**:
  - Configured explicit Blackwell SM121 environment variables (`FLASHINFER_CUDA_ARCH_LIST="12.1f"`, `CUTE_DSL_ARCH="sm_120a"`).
  - Added `--ulimit memlock=-1:-1`, `--ulimit stack=67108864`, and `--cap-add IPC_LOCK` to eliminate memory paging latency and Triton IPC faults.
  - Enforced `--max-prefill-tokens "${CHUNKED_PREFILL}"` to keep concurrent prefill admissions strictly bounded.
- **Launcher Watchdog & Network Flexibility**:
  - Added bounded 10–15 minute health-check loops with `/health` probing.
  - Allowed `PORT`, `HOST`, and `SERVED_MODEL_NAME` to be configured dynamically via `.env`.
- **Stop Script**: Enhanced `stop.sh` with automatic `.sglang.pid` removal and an optional `--clean` flag for purging Triton kernel caches.
- **Manifest Synchronization**: Recomputed and synchronized SHA256 checksums in `patch/overlay-dflash2/MANIFEST.sha256`.

---

## [0.2.0] - 2026-08-19

### Added
- **Quantized Selector Capture (PR #35496 Adaptation)**: Added support for ModelOpt NVFP4 quantized `lm_head` in CUDA graphs within `overlay-dflash2/sglang/srt/models/dflash.py`.
- **EAGLE 3/1/4 High-Concurrency Profile (`start-eagle.sh`)**: 32-seat profile delivering ~191 tok/s at C8, ~330 tok/s at C16, and ~535 tok/s at C32.
- **Priority Scheduling**: Configured SGLang priority preemption (`priority: 100` interactive vs `priority: 0` batch), cutting saturated TTFT by 94–96%.

---

## [0.1.0] - 2026-08-15

### Added
- Initial release of Qwen3.8-27B on DGX Spark (GB10) using SGLang.
- Base DFlash 2 profile with 4 admitted requests and 1,048,576 FP8 KV cache pool.
- Pinned image build script (`patch/build-dflash2-image.sh`).
