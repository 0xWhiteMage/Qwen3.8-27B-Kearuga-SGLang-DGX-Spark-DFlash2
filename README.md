# Qwen3.8-27B Kearuga on a single DGX Spark

<p align="center">
  <img src="assets/header.png" alt="The White Mage — Qwen3.8-27B Kearuga on DGX Spark with SGLang, DFlash 2 and EAGLE" width="100%"><br><br>
  <a href="#benchmark"><strong>65 tok/s net C1</strong></a> ·
  <a href="#benchmark"><strong>120 tok/s C4 aggregate</strong></a> ·
  <a href="#runtime-envelope"><strong>4 × native 262K contexts</strong></a> ·
  <a href="#ttft-idle-and-saturated-responsiveness"><strong>43.15 s → 2.63 s saturated TTFT</strong></a><br>
  <a href="#eagle-high-concurrency-throughput-c8c32"><strong>EAGLE: 191 C8 · 330 C16 · 535 C32 tok/s</strong></a><br><br>
  <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/Follow_on_X-@0xWhiteMage-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"></a><br>
  <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Kofi-Buy_me_a_coffee-1A9642?style=for-the-badge&logo=buymeacoffee&logoColor=white" alt="Ko-fi"></a>
</p>

Run **[Qwen3.8-27B](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)** with **[SGLang](https://docs.sglang.io)** on one 128 GB NVIDIA DGX Spark. The repo includes the pinned image build, launchers and benchmark harnesses.

- **DFlash 2:** fastest C1/C4 profile, with thinking and tool calling
- **EAGLE 3/1/4:** 32-seat high-concurrency profile for C8, C16 and C32
- **Native context:** four simultaneous 262K requests in a 1,048,576-token FP8 KV pool

> **Why two profiles?** DFlash makes the Spark a responsive daily driver; EAGLE turns it into a 32-seat agent engine. Read [Kearuga: Practical Insights](INSIGHTS.md) for the profile rationale, priority-TTFT results and memory trade-offs.

---

## Runtime envelope

### DFlash 2 daily-driver profile

> **Four full native 262K contexts on one GB10.**

`CONTEXT_LENGTH` limits one request. `MAX_TOTAL_TOKENS` is the shared target-KV pool.

| Component | Canonical value | Measured/operational meaning |
|---|---:|---|
| Per-request context | `262144` tokens | Native Qwen3.8 context; YaRN is off |
| Shared target-token pool | `1048576` tokens | Four simultaneous full-length 262K contexts |
| Admitted requests | `4` | Requests above four queue instead of entering the decode batch |
| Target KV allocation | `32.0 GiB` | Reported by `/v1/loads` with FP8 KV at the 1M pool |
| DFlash draft KV allocation | `10.0 GiB` | 5.0 GiB K + 5.0 GiB V in startup logs |
| Target weights | ~`23 GiB` | RadixArk NVFP4 target, pinned at `554ebba9…` |
| DFlash draft weights | ~`3.2 GiB` | z-lab DFlash 2 draft, pinned at `50307d4…` |
| GDN state pool | `20` slots | Five state slots × four admitted requests |
| Host unified memory after boot | ~`87 GiB` used / ~`34 GiB` available | GB10 RAM and VRAM are one pool; `nvidia-smi` is not a discrete-VRAM accounting source here |
| Decode graph cap | batch size `4` | Matches the admitted-request limit |
| DFlash draft window | `2048` tokens | Matches the checkpoint's native sliding-attention window |

The image uses a pinned SM121 base plus the checksum-verified overlay in `patch/overlay-dflash2/`.

### EAGLE high-concurrency profile

> **32 admitted requests sharing the same 1,048,576-token target-KV pool.**

EAGLE uses the same 1,048,576-token target pool with 32 admitted requests, 128 Mamba state slots and CUDA graph coverage through C32.

---

## Benchmark

> **DFlash: 65 tok/s net C1, 120 tok/s C4. EAGLE: 191 C8, 330 C16, 535 C32.**

The results are grouped by workload: interactive throughput, high concurrency, TTFT/responsiveness and correctness/quality.

**Jump to:** [DFlash C1–C4](#dflash-2-interactive-throughput-c1c4) · [EAGLE C8–C32](#eagle-high-concurrency-throughput-c8c32) · [TTFT / responsiveness](#ttft-idle-and-saturated-responsiveness) · [Correctness / quality](#quality-correctness-gates)

### DFlash 2: Interactive throughput (C1–C4)

> **Daily-driver profile: ~46 tok/s door-to-door C1, ~65 net C1 and ~120 aggregate C4.**

Temperature 0, thinking off, one DGX Spark. Door-to-door C1 includes prefill and TTFT; net C1 isolates steady decode; C4 is aggregate throughput across four streams.

| Setup | Door-to-door C1 (tok/s) | Net-decode C1 (tok/s) | C4 aggregate (tok/s) |
|---|---:|---:|---:|
| **This recipe (DFlash 2, 1M pool, selector captured)** | **46** | **65** | **120** |
| [MiaAI](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) DFlash 2 / DSpark | — | 50.9 / 51.5 | 111.6 |
| [r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang) | 28 | — | 55 |
| [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | 34 | — | 64 |

External rows are published figures with different harnesses and are included as context.

### EAGLE: High-concurrency throughput (C8–C32)

> **Need C16/C32? EAGLE scales to ~330/~535 tok/s.**

Unique request suffixes, forced 512-token outputs, thinking off, one DGX Spark:

| Profile | C8 | C16 | C32 | Typical saturated TTFT p95 |
|---|---:|---:|---:|---:|
| **EAGLE 3/1/4, 32 seats (this recipe)** | **181–193** | **320–335** | **527–539** | **0.32 / 0.49 / 0.87 s** |
| DFlash 2, four seats (this recipe) | ~118* | — | — | — |
| [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) capacity profile (published) | 116.8 | 153.9 | 178.3 | — |
| [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark) vLLM (published) | 246 | — | — | — |

\* DFlash C8 queues behind four admitted requests. External rows use different engines, seat limits, token pools and harnesses.

### TTFT: Idle and saturated responsiveness

> **Short-prompt idle TTFT stays near 0.3 seconds on both profiles.**

#### Idle TTFT

| Profile | ~645 prompt tokens | ~4.7K prompt tokens | ~18.7K prompt tokens |
|---|---:|---:|---:|
| **DFlash 2** | **0.35 s** | **2.17 s** | **11.10 s** |
| **EAGLE 3/1/4** | **0.30 s** | **2.20 s** | **10.50 s** |

TTFT includes prompt prefill; there is no hidden pre-generation stage.

#### Priority scheduling under saturation

| Profile and load | Normal priority TTFT | Interactive priority `100` | Reduction |
|---|---:|---:|---:|
| DFlash 2, all four seats occupied | 43.15 s | **2.63 s** | **~94%** |
| EAGLE, all 32 seats occupied | 73.30 s | **2.76 s** | **96.2%** |

The launchers enable SGLang [priority scheduling](https://github.com/sgl-project/sglang/blob/c4271c3fe1262fc2adbd162c33b25de5255251c5/python/sglang/srt/server_args.py#L841-L866). Use `priority: 100` for interactive work and `priority: 0` for batch work.

```json
{
  "model": "qwen3.8-27b-sglang",
  "priority": 100,
  "messages": [{"role": "user", "content": "Interactive request"}]
}
```

Priority preempts queued and decode work, not an active long prefill. Admit or compact oversized batch prompts before dispatch.

### Quality: Correctness gates

> **Checkpoint qualification and profile-level canaries remain separate evidence.**

Checkpoint-level thinking-off qualification measured [GSM8K](https://github.com/openai/grade-school-math) **86.5%** and [HumanEval](https://github.com/openai/human-eval) **90.9%**. Exact DFlash and EAGLE profile promotions passed arithmetic, GSM8K-flex and FizzBuzz canaries. Arithmetic canary: `19 × 23 → 437`.

---

## Deployment

> **Clone, build, boot. No external launcher required.**

```bash
git clone https://github.com/0xWhiteMage/Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2.git
cd Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2
cp .env.sample .env

bash patch/build-dflash2-image.sh
./start-dflash2.sh
curl -s http://127.0.0.1:8888/v1/models
```

First boot pulls the main weights (~16.5 GB) and the DFlash 2 draft (~3.9 GB). Endpoint: `http://127.0.0.1:8888/v1`, model name `qwen3.8-27b-sglang`.

Metrics: `http://127.0.0.1:8888/metrics`

Stop: `./stop.sh`

### Benchmark harness

```bash
./bench/bench.sh                              # essay, tool-call and 16K-TTFT smoke
python3 bench/ndec.py                         # C1 net-decode probes
python3 bench/scale.py --widths 4 --max-tokens 256  # DFlash C4 scale probe
python3 bench/scale.py                        # EAGLE C8/C16/C32 scale probe
python3 bench/priority_ttft.py --streams 32   # saturated priority probe
python3 bench/prefix_cache.py                 # cold vs reused long-prefix TTFT
```

### High-concurrency head (EAGLE 3/1/4)

For C8+, stop DFlash and run `./start-eagle.sh`. The first boot captures graphs through C32 and takes roughly 7.5 minutes. One head runs at a time.

---

## Configuration

> **Pinned models, pinned image, explicit memory.**

| Setting | Canonical value | Benefit |
|---|---|---|
| DFlash profile | draft `8`, window `2048`, four seats | Fast C1/C4 decode |
| EAGLE profile | `3/1/4`, 32 seats, continuous decode `1` | C8/C16/C32 throughput |
| Model revisions | Target `554ebba9…`, draft `50307d4…` | Reproducible weights |
| Selector | SGLang [#35496](https://github.com/sgl-project/sglang/pull/35496) adaptation | Captures the NVFP4 selector in the CUDA graph |
| Context | `262144` per request | Native context; YaRN off |
| Shared KV | `1048576`, `fp8_e4m3` | 32 GiB target KV allocation |
| Decode graphs | DFlash `4`, EAGLE `32` | Matches each profile's admitted concurrency |
| CPU | `5-9,15-19` | Uses the ten Cortex-X5 cores |
| Priority | default `0`, interactive `100` | Protects interactive decode latency |
| Chat | thinking on, `qwen3_coder` tools | Reasoning and tool calling without restart |

GB10 uses unified memory. After boot, `/v1/loads` reports 32 GiB target KV and the host retains about 31–34 GiB available.

---

## Credits

> **Open recipes made this build reproducible.**

### Recipes and serving stacks

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark):** DFlash 2 image, net-decode clock and dual-head launcher patterns
- **[r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang):** SM121 compatibility, pinned base image and overlay build pattern
- **[Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark):** pinned DFlash 2 recipe, quality suite and speed/capacity profiles
- **[hasso5703](https://github.com/hasso5703/dgx-spark-qwen38):** early DGX Spark serving and memory notes
- **[0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark):** vLLM comparison on the same hardware

### Engine, weights, model

- **[SGLang](https://github.com/sgl-project/sglang):** serving engine, DFlash 2 [#35371](https://github.com/sgl-project/sglang/pull/35371) and selector capture [#35496](https://github.com/sgl-project/sglang/pull/35496)
- **[z-lab / Inco](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2):** DFlash 2 draft model
- **[RadixArk](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4):** NVFP4 target checkpoint
- **[Qwen](https://huggingface.co/Qwen/Qwen3.8-27B):** base model family

---

## License

MIT for this recipe. The launcher, weights, and datasets keep their own licences.
