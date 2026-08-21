# 🧙‍♂️ Kearuga: Practical Insights

> Practical takeaways from tuning Qwen3.8-27B on a single DGX Spark.

This is the practical companion to the setup guide.

For the complete setup, benchmark methodology, configuration and reproducibility details, see the main [README](README.md).

---

## What I Built Kearuga For

I wanted one local LLM setup that could do two things well:

1. Feel fast enough to use as my **daily driver**
2. Keep multiple **agents working in the background**

Kearuga now has three qualified profiles for those different jobs:

| | Result |
|---|---:|
| ⚡ DFlash 2 C1 net decode | **~65 tok/s** |
| 🚪 DFlash 2 C1 door-to-door | **~46 tok/s** |
| 👷 DFlash 2 C4 aggregate | **~120 tok/s** |
| 🦅 EAGLE C8 aggregate | **181–193 tok/s** |
| 🦅 EAGLE C16 aggregate | **320–335 tok/s** |
| 🦅 EAGLE C32 aggregate | **527–539 tok/s** |
| 📦 DSpark block 7 (native base fallback) | **~51 tok/s net C1** |
| 📜 Shared target-KV pool | **1,048,576 tokens** |
| 🧠 Native context per request | **262,144 tokens** |
| 👷 Simultaneous full native contexts | **4 × 262K** |
| 💾 DFlash memory after boot | **~87 GiB used / ~34 GiB available** |
| 💾 EAGLE memory after boot | **~89 GiB used / ~31–32 GiB available** |

All on a **single GB10**.

---

## ⚡ 1. DFlash 2 Is My Daily-Driver Profile

For interactive use and roughly **1–4 concurrent streams**, I prefer DFlash 2.

It gives me around:

> **~65 tok/s net decode C1**  
> **~46 tok/s door-to-door C1**  
> **~120 tok/s aggregate C4**

The distinction between the first two matters.

**Net decode** tells me how quickly the model cruises once generation is underway.

**Door-to-door** includes the things I actually feel as a user: prefill, time-to-first-token and generation.

For a machine I want to interact with throughout the day, I care about both.

---

## 👷 2. Responsiveness Matters More Than Maximum Throughput

Imagine all four DFlash inference slots are occupied by background agents and I suddenly want to use the model myself.

Without priority preemption, the interactive request waited:

> 🐌 **43.15 seconds for a seat**

With Kearuga's priority configuration:

> ⚡ **2.63 seconds to first token**

That's roughly a **94% reduction**.

The same principle holds at larger scale. With all 32 EAGLE seats decoding, priority `100` reduced interactive TTFT from **73.30 seconds to 2.76 seconds** — a **96.2% reduction**.

So background workers can keep running while interactive requests are given priority.

### One caveat

Priority scheduling can pre-empt queued and decode work, but it **cannot interrupt a long prefill already in progress**.

Very large background prompts should still be managed at the orchestration layer through sensible admission, context compaction or scheduling.

---

## 🔬 3. Quantization Lessons from EXL3: Bridging the Fidelity Gap

Recent research in mixed-precision EXL3 quantization ([`malaiwah/qwen38-27b-exl3`](https://github.com/malaiwah/qwen38-27b-exl3)) measured a mean KL Divergence of **0.00276** on Qwen3.8-27B, compared to **0.0310** on uniform NVFP4 and **0.00529** on official FP8.

Why does uniform NVFP4 lose fidelity, and what can we learn from this?

### The Tensor Sensitivity Hierarchy
Error in quantized transformer weights is not uniformly distributed:
1. **Embeddings & LM Head (Extreme Sensitivity)**: Quantizing `embed_tokens` and `lm_head` to 4-bit causes severe probability skew in vocabulary logit tails. Keeping them in BF16 or FP8 costs ~1.2 GiB VRAM but recovers over 40% of the lost fidelity.
2. **Boundary Layers & Recurrence (High Sensitivity)**: Layers 0–1, 62–63 and Gated DeltaNet linear attention projections (`in_proj`, `conv1d`) carry structural sequence formatting. Keeping them in FP8 (`fp8_e4m3`) eliminates recurrent state drift over 262K contexts.
3. **MLP Blocks (High Capacity / Low Sensitivity)**: The `gate_proj`, `up_proj`, and `down_proj` matrices in middle layers (Layers 2–61) contain ~70% of total parameters. These can be quantized to hardware NVFP4 with virtually zero reasoning degradation when pre-activation outlier scaling (AWQ) is applied.

By pairing **FP8 attention/heads** with **NVFP4 MLPs**, a hybrid ~21.5 GiB checkpoint can achieve near-EXL3 KLD while executing at full speed on Blackwell FP4 Tensor Cores in SGLang.

---

## 🧠 4. The 1M Context Pool Is About Concurrency

Both profiles use:

> **262,144 tokens per request**  
> **1,048,576 target tokens shared across the server**

That distinction is important.

It isn't one million tokens for a single conversation.

It means the server has enough target-KV capacity for:

**4 × full native 262K contexts simultaneously**

EAGLE can admit 32 active requests, but those requests still share the same 1,048,576-token target pool. Thirty-two requests cannot each consume 262K tokens at once.

FP8 KV caching is what makes this practical within the GB10's unified-memory envelope. No YaRN is required.

---

## 🦅 5. EAGLE Is There When My Priorities Change

DFlash 2 is my **interactive profile**.

When concurrency and aggregate throughput matter more, Kearuga includes a separate **EAGLE 3/1/4** profile with 32 admitted requests and CUDA graph coverage through C32.

The qualified active-concurrency envelope is:

> 🦅 **181–193 tok/s at C8**  
> 🦅 **320–335 tok/s at C16**  
> 🦅 **527–539 tok/s at C32**

These measurements use unique request suffixes, forced 512-token outputs and enough admitted seats for every submitted stream. They are active C8/C16/C32 results, not queued-load arithmetic.

---

## 🔨 What We Actually Optimised

- **Zero-Allocation Logit Projection**: Pre-allocated workspace scratchpad in `dflash.py` eliminates PyTorch malloc thrashing during speculative decode loops.
- **Register-Resident Selector Walk**: Triton unrolling keeps candidate tree paths in streaming multiprocessor registers.
- **Blackwell SM121 Flags**: Explicit `FLASHINFER_CUDA_ARCH_LIST="12.1f"`, unlimited locked memory (`--ulimit memlock=-1:-1`), and stack enlargement (`--ulimit stack=67108864`).
- **Comprehensive Verification**: Integrated 262K Needle-In-A-Haystack retrieval and 10-point deterministic semantic gate.

---

## 🧙‍♂️ How I'd Choose a Profile

| What I want | What I'd run |
|---|---|
| Fast personal AI | **DFlash 2 (`./start-dflash2.sh`)** |
| Personal AI plus background agents | **DFlash 2 + priority scheduling** |
| Zero-build upstream fallback | **DSpark (`./start-dspark.sh`)** |
| Four complete native contexts | **1M KV / 4 × 262K** |
| C8/C16/C32 agent concurrency | **EAGLE 3/1/4 (`./start-eagle.sh`)** |
