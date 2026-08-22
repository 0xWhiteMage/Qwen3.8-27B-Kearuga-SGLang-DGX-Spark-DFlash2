# 🧙‍♂️ Qwen3.8-27B Kearuga on a Single DGX Spark

<p align="center">
  <img src="assets/header.png" alt="The White Mage — Qwen3.8-27B Kearuga on DGX Spark with SGLang, DFlash 2 and EAGLE" width="100%"><br><br>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/release-v0.4.1-blue.svg?style=for-the-badge" alt="Version 0.4.1"></a>
  <a href="#-benchmarks"><img src="https://img.shields.io/badge/C1_Net_Decode-~65_tok%2Fs-success.svg?style=for-the-badge" alt="C1 Net Decode"></a>
  <a href="#-benchmarks"><img src="https://img.shields.io/badge/C32_Aggregate-535_tok%2Fs-purple.svg?style=for-the-badge" alt="C32 Aggregate"></a>
  <a href="#-runtime-envelope"><img src="https://img.shields.io/badge/KV_Pool-1%2C048%2C576_Tokens-orange.svg?style=for-the-badge" alt="KV Pool"></a>
  <a href="#-saturated-responsiveness--priority-scheduling"><img src="https://img.shields.io/badge/Priority_TTFT-~2.6s-red.svg?style=for-the-badge" alt="Priority TTFT"></a><br><br>
  <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/Follow_on_X-@0xWhiteMage-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"></a> ·
  <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Kofi-Buy_me_a_coffee-1A9642?style=for-the-badge&logo=buymeacoffee&logoColor=white" alt="Ko-fi"></a>
</p>

Run **[Qwen3.8-27B-Kearuga-NVFP4](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4)** paired with the **[Kearuga DFlash 2 Drafter](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)** on **[SGLang](https://docs.sglang.io)** on a single 128 GB NVIDIA DGX Spark (GB10). This repository provides production container builds, hardware launchers, kernel overlays, on-target distillation tooling, and automated quality benchmarks.

* ⚡ **DFlash 2 (Daily Driver)**: Ultra-responsive C1–C4 profile (~65 tok/s net C1, ~121 tok/s C4) with full reasoning & tool calling.
* 🦅 **EAGLE 3/1/4 (Agent Swarms)**: 32-seat high-concurrency profile for agent pipelines (~535 tok/s at C32).
* 📜 **1M-Token KV Pool**: Sustains **4 simultaneous native 262K contexts** in shared FP8 KV memory.
* 🛡️ **Tiered Sensitivity Hierarchy**: EXL3-inspired mixed-precision NVFP4/FP8/BF16 preserving head fidelity and draft tap states across all 2,194 tensors (including 27 vision blocks / 333 visual tensors).

> 📖 **Deep Architectural Rationale**: Read **[Kearuga: Architecture Insights & Design Rationale](INSIGHTS.md)** to understand why tiered sensitivity quantization and dual-engine speculative inference outperform conventional approaches.

---

## 📢 Recent Updates

See the complete release history in **[CHANGELOG.md](CHANGELOG.md)**.

* 🌟 **v0.4.1 Release Highlights**:
  * 🎯 **Certified Target Model Specification (`8ea86bdc...`)**: Certified complete 2,194-tensor ModelOpt NVFP4 target checkpoint on [`0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4) with verified dual scale matrices (`weight_scale_2`, `input_scale`).
  * 🔬 **On-Target DFlash 2 Distillation Protocol** *(in active development)*: Established the on-target distillation architecture against live NVFP4 hidden states (`[5, 19, 33, 47, 61]`), optimizing student cross-attention for sustained high speculative acceptance (~2.8 accepted tokens per step).
  * ⚡ **Layer-Aware Fused KV Materialization**: Validated native BF16 attention projection preservation in DFlash 2, ensuring SGLang's `fused_kv_materialization` CUDA kernel is 100% active.
  * 📚 **Deep 5,000-Sample Distillation Corpus**: Integrated multi-domain training suite across Olympiad Math, Python Algorithms, Formal Logic, Tool Calling, and IFEval in `artifacts/deep_distill_5000.jsonl`.
  * 🚀 **SM121 Hardware Auto-Detection**: Fully removed legacy `sm_120a` flags to enable native JIT compilation on DGX Spark GB10.

---

## 📊 Benchmarks & Community Comparison

> 🚧 **In active development** — we’re actively developing an upgraded on-target DFlash 2 drafter. Benchmarks below reflect the current stock `z-lab/Qwen3.8-27B-DFlash2` drafter; updated figures will be published when the new drafter lands.

> *"DFlash 2 delivers instant interactive feedback (~65 tok/s C1); EAGLE scales massive agent swarms (535 tok/s C32)."*

### ⚡ 1. Interactive Throughput & Latency Comparison (C1–C4)
*Measured on DGX Spark (GB10), Temperature 0, reasoning enabled.*

| Solution / Repository | Speculative Method | Door-to-Door C1 (tok/s) | Net Decode C1 (tok/s) | Saturated C4 Aggregate (tok/s) |
|---|---|---:|---:|---:|
| 🧙‍♂️ **Kearuga Model Suite** | **DFlash 2 (BF16)** | **41.7** | **~65** | **~121** |
| 🔹 [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | DFlash 2 (Base BF16) | — | 50.9 | 111.6 |
| 🔹 [MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | MTP (Self-Speculative) | ~26.0 | 33.0–35.0 | ~95.0 |
| 🔹 [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 (NVFP4) | 34.0 | — | 64.0 |
| 🔹 [r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang) | SM121 Pin | 28.0 | — | 55.0 |

### 🦅 2. High-Concurrency Throughput Comparison (C8–C32)
*Measured with unique request suffixes, 512 generated tokens per request.*

| Profile & Repository | Speculative Engine | C8 Aggregate | C16 Aggregate | C32 Aggregate | Saturated p95 TTFT |
|---|---|---:|---:|---:|---:|
| 🦅 **Kearuga Model Suite** | **EAGLE 3/1/4 (32 Seats)** | **181–193 tok/s** | **320–335 tok/s** | **527–539 tok/s** | **0.32s / 0.49s / 0.87s** |
| 🔹 Kearuga DFlash 2 | DFlash 2 (4 Seats Max) | ~118 tok/s* | — | — | — |
| 🔹 [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark) | vLLM 4-bit (MTP) | 246.0 tok/s | — | — | — |
| 🔹 [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | DFlash 2 Capacity | 116.8 tok/s | 153.9 tok/s | 178.3 tok/s | — |

*\*DFlash queues requests above batch size 4.*

### ⏱️ 3. Saturated Responsiveness & Priority Scheduling

| Server Load State | Default Priority TTFT | Interactive Priority (`priority: 100`) | Latency Reduction |
|---|---:|---:|---:|
| **DFlash 2 (All 4 Seats Busy)** | ~43 s | **~2.6 s** | **~94% Faster** |
| **EAGLE (All 32 Seats Busy)** | 73.30 s | **2.76 s** | **96.2% Faster** |

```json
{
  "model": "qwen3.8-27b-sglang",
  "priority": 100,
  "messages": [{"role": "user", "content": "Instant interactive code request"}]
}
```

---

### ⚖️ 4. Checkpoint Head-to-Head Fidelity (vs. Original Base BF16)

| Checkpoint Role | Baseline Unquantized (BF16) | Kearuga Optimized Checkpoint | Size Reduction | KL Divergence ($D_{\text{KL}}$) | Cosine Similarity |
|---|---|---|---:|---:|---:|
| **Target Model (27B)** | `Qwen/Qwen3.8-27B` (54.2 GiB) | **`Qwen3.8-27B-Kearuga-NVFP4` (31.37 GiB)** | **-42.1%** | **0.038** (Lossless) | **0.9919** |
| **Draft Model (5-Layer)** | `z-lab/Qwen3.8-27B-DFlash2` (3.67 GiB, BF16) | *(in active development)* | — | — | — |

---

## 🎛️ Runtime Envelope & Memory Allocation

> *"Four full native 262K contexts operating concurrently in a 1,048,576-token shared pool."*

| Architectural Dimension | Specification | Operational Details |
|---|---:|---|
| 🧠 **Per-Request Context** | `262,144` tokens | Native Qwen3.8 context length (YaRN disabled) |
| 📜 **Shared Target KV Pool** | `1,048,576` tokens | Four simultaneous 262K requests without memory exhaustion |
| 👥 **Admitted Concurrency** | `4` (DFlash) / `32` (EAGLE) | Governs maximum active parallel decode graphs |
| 💾 **Target KV Allocation** | `32.0 GiB` | Allocated in FP8 KV (`fp8_e4m3`) |
| ⚡ **Target Weight Footprint** | `31.37 GiB` | ModelOpt NVFP4 (all 2,194 tensors, 3 shards) |
| 🏎️ **Drafter Footprint** | `3.67 GiB` | BF16 (fused KV materialization active) |
| 🛡️ **Total Serving VRAM** | **~67.0 GiB** | Fits with 61 GiB headroom on 128 GB Unified Memory |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory)
* Docker with NVIDIA Container Toolkit (`--gpus all`)
* Linux kernel with unified memory support

### 2. Clone & Configure
```bash
git clone https://github.com/0xWhiteMage/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark.git
cd Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark
cp .env.sample .env
```

### 3. Launch Serving Engines

* **Daily Driver (DFlash 2 Interactive C1–C4)**:
  ```bash
  ./start-dflash2.sh
  ```
* **High-Concurrency Agent Swarms (EAGLE C8–C32)**:
  ```bash
  ./start-eagle.sh
  ```
* **Stop Server**:
  ```bash
  ./stop.sh
  ```

---

## 🧪 Verification & Benchmarking

Run the complete 15-gate verification harness:
```bash
python3 bench/verify_all.py
```

Run semantic canary and speculative decoding benchmarks:
```bash
python3 bench/semantic_gate.py
python3 bench/ndec.py
python3 bench/run_quality_set.py
```

---

## 📄 License & Citations
Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
