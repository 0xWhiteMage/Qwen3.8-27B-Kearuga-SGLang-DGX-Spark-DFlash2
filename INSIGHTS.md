# 🧙‍♂️ Kearuga: Practical Insights

> Practical takeaways from tuning Qwen3.8-27B on a single DGX Spark.

This is the practical companion to the setup guide.

For the complete setup, benchmark methodology, configuration and reproducibility details, see the main [README](README.md).

---

## What I Built Kearuga For

I wanted one local LLM setup that could do two things well:

1. Feel fast enough to use as my **daily driver**
2. Keep multiple **agents working in the background**

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

## 🧠 3. The 1M Context Pool Is About Concurrency

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

## 🎨 4. Context Can Also Be Traded for Memory

The canonical 1M profile isn't necessarily the profile everyone should use.

Don't need four complete 262K contexts? Shrink the shared KV pool.

For example:

> **1,048,576 → 786,432 tokens**  
> **4 × 262K → 3 × 262K contexts**  
> **~32 GiB → ~24 GiB target KV**  
> **~8 GiB target KV reclaimed**

On DFlash, the draft KV pool also scales down, bringing the measured total KV reduction to approximately **10.5 GiB**.

That extra headroom could be used for other workloads. I'm particularly interested in experimenting with:

**Qwen + agents + ComfyUI + local image/video generation**

on the same Spark.

A quantised MiniMax H3 stack is one workload I would like to explore. That combined workload is **not a qualified Kearuga profile yet**; it is an example of why configurable KV capacity is useful.

Instead of asking:

> *What's the maximum context I can fit?*

I'd rather ask:

> **What's the most useful combination of workloads I can fit into 128 GB?**

---

## 🦅 5. EAGLE Is There When My Priorities Change

DFlash 2 is my **interactive profile**.

When concurrency and aggregate throughput matter more, Kearuga includes a separate **EAGLE 3/1/4** profile with 32 admitted requests and CUDA graph coverage through C32.

The qualified active-concurrency envelope is:

> 🦅 **181–193 tok/s at C8**  
> 🦅 **320–335 tok/s at C16**  
> 🦅 **527–539 tok/s at C32**

These measurements use unique request suffixes, forced 512-token outputs and enough admitted seats for every submitted stream. They are active C8/C16/C32 results, not queued-load arithmetic.

### DFlash 2

**Best for:**

- Daily-driver use
- Interactive requests
- Coding and research
- 1–4 concurrent streams
- Background agents plus foreground interaction

### EAGLE 3/1/4

**Best for:**

- C8, C16 and C32 concurrency
- Agent fleets
- Batch workloads
- Throughput-first serving

Only one speculative head runs at a time, so changing between them requires a restart.

---

## 🔨 What I Actually Optimised

Kearuga isn't one magic flag. The performance comes from treating the serving configuration as a complete system:

- NVFP4 target weights
- DFlash 2 and EAGLE speculative decoding
- EAGLE tree geometry
- ReplaySSM
- FP8 KV cache
- Quantised selector capture
- CUDA graph sizing through each profile's admitted concurrency
- Torch compile coverage
- Continuous-decode scheduling
- DFlash draft-window sizing
- GDN and Mamba state allocation
- GB10 CPU affinity
- Bounded shared KV capacity
- Request admission
- Priority scheduling

The recipe keeps the settings that remained correct and repeatable across warm runs, cold restarts, memory checks and canary tests on the actual DGX Spark.

---

## 🧙‍♂️ How I'd Choose a Profile

| What I want | What I'd run |
|---|---|
| Fast personal AI | **DFlash 2** |
| Personal AI plus background agents | **DFlash 2 + priority scheduling** |
| Four complete native contexts | **1M KV / 4 × 262K** |
| More memory for other workloads | **Smaller KV pool** |
| C8/C16/C32 agent concurrency | **EAGLE 3/1/4** |
| Creative plus LLM experimentation | **Smaller KV + separately qualified workloads** |

There isn't one universally "best" configuration.

The interesting part of the DGX Spark is deciding **what you want its 128 GB unified-memory pool to do for you**.

For me, that means making it less like a machine that can merely *fit* a large model...

…and more like an **always-on local AI workstation**.

---

## Full Recipe

Everything required to reproduce the setup is in the main [README](README.md):

- [Installation and deployment](README.md#deployment)
- [Benchmarks](README.md#benchmark)
- [Runtime envelope](README.md#runtime-envelope)
- [Configuration](README.md#configuration)
- [Credits](README.md#credits)

The projects, models, recipes and contributors that made Kearuga possible are credited there.

**Clone. Build. Boot. Experiment. 🧙‍♂️**
