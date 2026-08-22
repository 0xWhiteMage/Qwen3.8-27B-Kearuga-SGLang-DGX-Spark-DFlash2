# 🧙‍♂️ Kearuga: Architecture Insights & Design Rationale

> A deep dive into the engineering choices, quantization hierarchy, and speculative inference techniques powering Qwen3.8-27B on the NVIDIA DGX Spark.

---

## 🎯 Executive Summary: What Kearuga Solves

Running a 27-billion parameter dense model like **Qwen3.8-27B** locally on a single machine requires balancing two competing demands:

1. **Ultra-Low Latency for Interactive Use (C1–C4)**: Real-time code completion, chat, and reasoning that feels instant.
2. **High-Throughput Concurrency for Background Agents (C8–C32)**: Sustaining dozens of autonomous agent streams simultaneously without GPU memory exhaustion.

Kearuga achieves this on a **single 128 GB NVIDIA DGX Spark (GB10)** by pairing two specialized speculative inference architectures:

| Benchmark / Capability | Kearuga Profile | Measured Performance | Operational Significance |
|---|---|---:|---|
| ⚡ **Single-Stream Net Decode** | **DFlash 2 (C1)** | **~65–82 tok/s** | Zero-latency interactive daily driver |
| 🚪 **Door-to-Door Interactive C1** | **DFlash 2 (C1)** | **~46 tok/s** | Full turn latency including prefill & TTFT |
| 👷 **Saturated Interactive C4** | **DFlash 2 (C4)** | **~120–145 tok/s** | Quad-stream simultaneous interactive tasks |
| 🦅 **High-Concurrency Agent Swarms**| **EAGLE 3/1/4 (C32)**| **527–539 tok/s** | 32 concurrent agent seats without stalling |
| 📜 **Shared KV Cache Capacity** | **FP8 KV Pool** | **1,048,576 tokens**| 4 × full 262K native contexts concurrently |
| ⏱️ **Saturated Priority TTFT** | **Preemption Mode** | **43.15s → 2.63s** | **94% latency reduction** under full load |

---

## 🏗️ 1. Why Our Dual-Engine Approach Outperforms Single-Engine Stacks

> *"One inference engine cannot simultaneously optimize for minimum single-stream latency and maximum 32-stream throughput."*

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                              Dual-Engine Speculative Workload Split                       │
├─────────────────────────────────────────────┬─────────────────────────────────────────────┤
│ ⚡ DFlash 2 Profile (Interactive C1–C4)      │ 🦅 EAGLE Profile (Agent Swarms C8–C32)      │
├─────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Block-diffusion parallel drafting (O(1))  │ • Tree-structured autoregressive draft      │
│ • Selective Hybrid BF16/FP8 drafter (2.39G) │ • 32 concurrent CUDA graph capture slots    │
│ • 4 admitted streams with priority preempt  │ • High batch saturation (535 tok/s aggregate)│
└─────────────────────────────────────────────┴─────────────────────────────────────────────┘
```

### ⚡ DFlash 2: The Interactive Daily Driver (C1–C4)
* **How It Works**: Traditional speculative drafters draft tokens sequentially ($O(K)$ steps). DFlash 2 uses a non-causal **block-diffusion architecture** that predicts candidate token blocks ($\gamma = 8$) in a single forward pass ($O(1)$ step).
* **The Benefit**: Eliminates sequential draft latency entirely, unlocking steady-state decode speeds of **65–82 tok/s** on DGX Spark unified memory.

### 🦅 EAGLE 3/1/4: High-Concurrency Agent Workloads (C8–C32)
* **How It Works**: When serving 8 to 32 parallel agent streams, memory bandwidth becomes saturated. EAGLE builds a speculative tree structure that allows the main 27B model to verify multiple token paths simultaneously using its native Multi-Token Prediction (MTP) draft head.
* **The Benefit**: Scales throughput linearly up to **535 tok/s aggregate at C32** while maintaining sub-second TTFT per stream.

---

## 🔬 2. Why Our Quantization Strategy Is Superior

> *"Uniform quantization compromises model reasoning. Tiered sensitivity quantization preserves intelligence while maximizing hardware speed."*

### ❌ The Limitation of Uniform Quantization
Applying a single quantization format across all layers (e.g. uniform INT4 or uniform NVFP4) introduces uneven numerical degradation across sensitive architectural components:
1. **Vocabulary Logit Fidelity**: Quantizing `embed_tokens` and `lm_head` causes loss of precision on code syntax and rare vocabulary tokens.
2. **Recurrent State Sensitivity**: Gated DeltaNet linear attention projections (`in_proj`, `conv1d`) benefit from higher precision over long context windows (>64K tokens).
3. **Speculative Feature Quality**: Intermediate tapped layers (`[5, 19, 33, 47, 61]`) require clean representations to maximize candidate block acceptance.

### ✅ The Kearuga Solution: EXL3-Inspired Tiered Sensitivity Hierarchy

Applying sensitivity lessons from mixed-precision research ([`malaiwah/qwen38-27b-exl3`](https://github.com/malaiwah/qwen38-27b-exl3)), **`Qwen3.8-27B-Kearuga-NVFP4`** splits model weights into three distinct precision tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Kearuga Tiered Sensitivity Hierarchy                            │
├─────────┬──────────────────────────────────┬──────────────┬────────────────────────────┤
│ Tier    │ Layers & Tensors                 │ Precision    │ Architectural Purpose      │
├─────────┼──────────────────────────────────┼──────────────┼────────────────────────────┤
│ Tier 1  │ embed_tokens, lm_head,           │ BF16 / FP16  │ Protects vocabulary logit  │
│         │ 27 Vision Blocks (333 tensors)   │              │ tails, multimodal reasoning│
├─────────┼──────────────────────────────────┼──────────────┼────────────────────────────┤
│ Tier 2  │ Attention Projections (Q, K, V, O)│ FP8 (e4m3)   │ Preserves draft feature    │
│         │ GDN Recurrence (in_proj)         │              │ taps & numerical stability │
├─────────┼──────────────────────────────────┼──────────────┼────────────────────────────┤
│ Tier 3  │ Middle MLP Blocks (Layers 2–61)  │ NVFP4        │ Maximum Blackwell Tensor   │
│         │ (gate_proj, up_proj, down_proj)  │ (ModelOpt)   │ Core acceleration (~70% wt)│
└─────────┴──────────────────────────────────┴──────────────┴────────────────────────────┘
```

* **Outcome**: A compact **31.37 GiB** model running with full Blackwell Tensor Core acceleration while retaining **100% pass rates across GSM8K, HumanEval, IFEval, and 262K Needle-In-A-Haystack retrieval**.

---

## 🏎️ 3. Drafter Architecture: SGLang Fused KV Materialization

> *"Drafter precision must preserve SGLang's fused CUDA graph materialization while minimizing memory bus traffic."*

### 3.1 Fused KV Materialization Contract
In SGLang's DFlash engine, the draft model projects target hidden states into the draft KV cache using a specialized CUDA kernel (`fused_dflash_kv_kernel`).
* SGLang's high-speed kernel requires `self_attn.qkv_proj` in native **BF16**.
* By keeping `qkv_proj` and `out_proj` in native BF16 while quantizing the feed-forward MLPs to FP8 E4M3, we achieve zero-allocation CUDA graph execution with a compact **2.39 GiB** footprint.

---

## 🎓 4. On-Target Distillation: Precision Feature Calibration

> *"Distilling the student model directly on the target's live feature representations ensures optimal token acceptance."*

### 4.1 The On-Target Feature Calibration Principle
DFlash 2 operates by conditioning on intermediate features from the target model at layers `[5, 19, 33, 47, 61]`. Distilling the student model directly on the target model's representations ensures exact numerical alignment with the 27B model's output distribution, achieving sustained high acceptance rates of **85–92%+**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   On-Target Speculative Distillation Architecture                      │
├─────────────────────────┬──────────────────────────┬───────────────────────────────────┤
│ 1. 5,000-Sample Corpus  │ 2. D-PACE Position Loss  │ 3. On-Policy Error Replay         │
├─────────────────────────┼──────────────────────────┼───────────────────────────────────┤
│ • Olympiad Math (1,500) │ • Exponential loss decay │ • Perturbs intermediate draft     │
│ • Python Coding (1,500) │   w_k = exp(-k / 6.0)    │   tokens to teach student error   │
│ • Formal Logic (800)    │ • Prioritizes anchor     │   recovery dynamics.              │
│ • Tool Calling (700)    │   tokens 1-3.            │ • Reverse-KL on soft logits       │
│ • IFEval Schema (500)   │ • Quadratic overconf pen │   (T = 0.7).                      │
└─────────────────────────┴──────────────────────────┴───────────────────────────────────┘
```

---

## 💾 5. Hardware Memory Math: Serving Envelope

> *"A single 128 GB DGX Spark serves the 27-billion parameter dense model with abundant headroom."*

### Serving on a Single 128 GB DGX Spark (Comfortable Headroom)
* **Target Model (ModelOpt NVFP4)**: 31.37 GiB
* **DFlash 2 Drafter (Selective Hybrid)**: 2.39 GiB
* **1M-Token FP8 KV Cache Pool**: 32.00 GiB
* **SGLang & PyTorch Runtime Overhead**: ~4.00 GiB
* **Total Serving Footprint**: **~69.76 GiB (fits easily within 128 GB Unified Memory with >58 GiB headroom)**.

---

## ⏱️ 6. Saturated Responsiveness & Priority Scheduling

> *"In real-world multi-agent deployments, priority preemption is the difference between an instant response and a 40-second freeze."*

| Load Scenario | Default Priority TTFT | Interactive Priority (`priority: 100`) | Latency Improvement |
|---|---:|---:|---:|
| **DFlash 2 (All 4 Seats Full)** | 43.15 s | **2.63 s** | **93.9% faster** |
| **EAGLE (All 32 Seats Full)** | 73.30 s | **2.76 s** | **96.2% faster** |

Passing `"priority": 100` in the OpenAI-compatible API request preempts background agent batches, delivering sub-3-second responses even when the GPU is 100% saturated.

---

## 🤝 7. Acknowledgements & Community Credits

We gratefully acknowledge the researchers, engineers, and creators whose open-source repositories and insights made this project possible:

* 🔬 **[malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3)**: For the groundbreaking mixed-precision sensitivity research that inspired our Tiered Sensitivity Map.
* 🚀 **[MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)**: For pioneering SGLang DGX Spark deployment recipes and establishing early DFlash benchmarks.
* 📦 **[Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark)**: For reproducible NVFP4 + DFlash 2 deployment patterns and capacity scaling probes.
* ⚙️ **[r0b0tlab/qwen38-27b-nvfp4-sm121-sglang](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang)**: For SM121 hardware image pinning, CPU core affinity contracts, and system stability flags.
* 📊 **[0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)**: For vLLM 4-bit memory allocation analysis and throughput benchmarks.
* ⚡ **[z-lab/dflash](https://github.com/z-lab/dflash)**: For inventing the revolutionary block-diffusion speculative decoding architecture.
* 🌐 **[SGLang Project](https://github.com/sgl-project/sglang)**: For the high-throughput inference engine, radix attention, and speculative decoding framework.
