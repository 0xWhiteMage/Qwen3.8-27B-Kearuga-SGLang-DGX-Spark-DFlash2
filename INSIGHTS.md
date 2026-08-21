# 🧙‍♂️ Kearuga: Practical Insights

> Practical takeaways from tuning Qwen3.8-27B on a single DGX Spark with DFlash 2 and EAGLE.

---

## What I Built Kearuga For

I wanted one local LLM setup that could do two things well:

1. Feel fast enough to use as my **daily driver (DFlash 2)**
2. Keep multiple **agents working in the background (EAGLE 3/1/4)**

Kearuga now has two qualified profiles for those different jobs:

| | Result |
|---|---:|
| ⚡ DFlash 2 C1 net decode | **~65 tok/s** |
| 🚪 DFlash 2 C1 door-to-door | **~46 tok/s** |
| 👷 DFlash 2 C4 aggregate | **~120 tok/s** |
| 🦅 EAGLE C8 aggregate | **181–193 tok/s** |
| 🦅 EAGLE C16 aggregate | **320–335 tok/s** |
| 🦅 EAGLE C32 aggregate | **527–539 tok/s** |
| 📜 Shared target-KV pool | **1,048,576 tokens** |
| 🧠 Native context per request | **262,144 tokens** |
| 👷 Simultaneous full native contexts | **4 × 262K** |
| 💾 DFlash memory after boot | **~87 GiB used / ~34 GiB available** |
| 💾 EAGLE memory after boot | **~89 GiB used / ~31–32 GiB available** |

All on a **single GB10**.

---

## ⚡ 1. DFlash 2 Is My Daily-Driver Profile

For interactive use and roughly **1–4 concurrent streams**, DFlash 2 is the primary engine.

It gives:

> **~65 tok/s net decode C1**  
> **~46 tok/s door-to-door C1**  
> **~120 tok/s aggregate C4**

The distinction between the first two matters:
* **Net decode** isolates steady-state speculation throughput once generation is underway.
* **Door-to-door** includes prefill, time-to-first-token, and full response latency.

---

## 👷 2. Responsiveness Matters More Than Maximum Throughput

With all four DFlash inference slots occupied by background agents, priority preemption makes the difference:
* Without priority preemption: **43.15 seconds to first token**
* With priority configuration (`priority: 100`): **2.63 seconds to first token (~94% reduction)**

---

## 🔬 3. Quantization Lessons from EXL3: Bridging the Fidelity Gap

Recent research in mixed-precision EXL3 quantization ([`malaiwah/qwen38-27b-exl3`](https://github.com/malaiwah/qwen38-27b-exl3)) demonstrated that quantization degradation is localized:

### The Tensor Sensitivity Hierarchy
1. **Embeddings & LM Head (Extreme Sensitivity)**: Quantizing `embed_tokens` and `lm_head` to 4-bit causes severe probability skew in vocabulary logit tails. Keeping them in BF16 or FP8 costs ~1.2 GiB VRAM but recovers over 40% of the lost fidelity.
2. **Boundary Layers & Recurrence (High Sensitivity)**: Layers 0–1, 62–63 and Gated DeltaNet linear attention projections (`in_proj`, `conv1d`) carry structural sequence formatting. Keeping them in FP8 (`fp8_e4m3`) eliminates recurrent state drift over 262K contexts.
3. **MLP Blocks (High Capacity / Low Sensitivity)**: The `gate_proj`, `up_proj`, and `down_proj` matrices in middle layers (Layers 2–61) contain ~70% of total parameters. These can be quantized to hardware NVFP4 with virtually zero reasoning degradation when pre-activation outlier scaling (AWQ) is applied.

---

## 🏎️ 4. Creating an FP8 E4M3 Draft Model for Faster C1–C4

The DFlash 2 draft model (`z-lab/Qwen3.8-27B-DFlash2`) is 5 layers (~3.2 GB in BF16). On the DGX Spark's 273 GB/s unified memory bus, reading 3.2 GB of draft weights per step creates memory bandwidth contention with target verification.

Converting the draft model weights to `fp8_e4m3` (with per-tensor scale factors):
* Reduces draft weight footprint from **3.2 GB to 1.6 GB** (50% reduction in weight traffic).
* Reduces draft memory bandwidth latency by ~45% for C1–C4 decode steps.
* Maintains $>99.5\%$ top-$k$ candidate alignment with the target model.

---

## 🔨 What We Actually Optimised

- **Zero-Allocation Logit Projection**: Pre-allocated workspace scratchpad in `dflash.py` eliminates PyTorch malloc thrashing during speculative decode loops.
- **Register-Resident Selector Walk**: Triton unrolling keeps candidate tree paths in streaming multiprocessor registers.
- **Blackwell SM121 Flags**: Explicit `FLASHINFER_CUDA_ARCH_LIST="12.1f"`, unlimited locked memory (`--ulimit memlock=-1:-1`), and stack enlargement (`--ulimit stack=67108864`).
- **Comprehensive Verification**: Integrated 262K Needle-In-A-Haystack retrieval, Quality-200 benchmark, and 10-point deterministic semantic gate.
