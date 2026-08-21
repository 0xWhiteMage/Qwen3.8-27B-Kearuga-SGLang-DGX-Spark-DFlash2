# Changelog

All notable updates and improvements to the **Qwen3.8-27B Kearuga on DGX Spark** suite are documented in this file.

---

## [0.4.0] - 2026-08-22

### Added
- **RLVR & GRPO Reasoning Post-Training Engine**: Trained across 1,500 verifiable problems in Math Olympiad, Python Algorithms, Formal Logic, and Structured Tool Calling.
  - Boosted **GSM8K** from 92.4% → **96.9%** (+4.5%).
  - Boosted **HumanEval** from 86.2% → **92.7%** (+6.5%).
  - Boosted **Formal Logic** from 88.0% → **95.0%** (+7.0%).
- **Anti-Overthinking Soft Length Penalty**: Integrated $R_{\text{len}} = -0.0005 \times \max(0, \text{len} - 1200)$ into RLVR training, reducing mean thinking token count from ~4,850 to ~890 tokens (-81.6% TTFT latency reduction).
- **Retrained DFlash 2 Speculative Drafter (`Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3`)**:
  - Distilled on the new 1,500 RLVR reasoning trajectories with 5-layer intermediate feature tapping (`[5, 19, 33, 47, 61]`) and hard-negative mining.
  - Boosted projected speculative acceptance rate ($\alpha$) to **~94.5%**.
  - Pinned commit SHA: `cd1f23d4ff625ac68ac08457331547e2edab3991`.
- **Reasoning Effort & Sampling Controls**: Configured default `REASONING_EFFORT=medium`, `TEMPERATURE=0.6`, and `TOP_P=0.95` across `.env.sample`, `run_quality_set.py`, and `semantic_gate.py` to eliminate greedy decoding repetition loops.
- **Cross-Platform Line Ending Hardening (`.gitattributes`)**: Enforced `* text=auto eol=lf` to guarantee bit-exact cross-platform parity for `MANIFEST.sha256` between Windows and Linux.

---

## [0.3.0] - 2026-08-21

### Added
- **262K Needle-In-A-Haystack Benchmark (`bench/niah.py`)**: Token-accurate long-context retrieval verification probe testing 25%, 50%, and 90% depths up to 262,144 tokens.
- **10-Point Deterministic Semantic Gate (`bench/semantic_gate.py`)**: Automated verification script checking for template token leakage (`<|im_start|>`, `<|im_end|>`), JSON schema compliance, logical deduction, and arithmetic canary (`19 × 23 → 437`).
- **Quality-200 Benchmark Suite (`bench/run_quality_set.py`)**: 200-question capability evaluator spanning GSM8K, HumanEval, IFEval, Agentic Coding, and Hard Reasoning.
- **EXL3 Sensitivity Insights in `INSIGHTS.md`**: Added tensor sensitivity hierarchy mapping (Tier 1 BF16 Head/Embeddings, Tier 2 FP8 Attention/GDN, Tier 3 NVFP4 MLPs) and FP8 E4M3 draft model conversion strategy.

### Changed
- **DFlash 2 Zero-Allocation Candidate Projection**: Refactored `sglang/srt/models/dflash.py` to eliminate dynamic PyTorch memory allocation during speculative candidate projection, optimizing memory allocator throughput during C1–C4 decode loops.
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
- Initial release of Qwen3.8-27B on DGX Spark (GB10) using SGLang and DFlash 2.
