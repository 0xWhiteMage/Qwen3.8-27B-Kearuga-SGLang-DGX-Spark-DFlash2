# Qwen3.8-27B Kearuga on a single DGX Spark

<p align="center">
  <img src="assets/header.png" alt="The White Mage — Qwen3.8-27B Kearuga on DGX Spark with SGLang, DFlash 2 and EAGLE" width="100%"><br><br>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-v0.3.0-blue.svg?style=for-the-badge" alt="Version 0.3.0"></a><br><br>
  <a href="#benchmark"><strong>65 tok/s net C1</strong></a> ·
  <a href="#benchmark"><strong>120 tok/s C4 aggregate</strong></a> ·
  <a href="#runtime-envelope"><strong>4 × native 262K contexts</strong></a> ·
  <a href="#ttft-idle-and-saturated-responsiveness"><strong>43.15 s → 2.63 s saturated TTFT</strong></a><br>
  <a href="#eagle-high-concurrency-throughput-c8c32"><strong>EAGLE: 191 C8 · 330 C16 · 535 C32 tok/s</strong></a><br><br>
  <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/Follow_on_X-@0xWhiteMage-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X"></a> ·
  <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Kofi-Buy_me_a_coffee-1A9642?style=for-the-badge&logo=buymeacoffee&logoColor=white" alt="Ko-fi"></a>
</p>

Run **[Qwen3.8-27B](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)** with **[SGLang](https://docs.sglang.io)** on one 128 GB NVIDIA DGX Spark (GB10). The repo includes the pinned image build, launchers, kernel overlays and benchmark harnesses.

- **DFlash 2:** fastest C1/C4 profile, with thinking and tool calling (~65 tok/s net C1, ~120 tok/s C4)
- **EAGLE 3/1/4:** 32-seat high-concurrency profile for C8, C16 and C32 (~535 tok/s at C32)
- **Native context:** four simultaneous 262K requests in a 1,048,576-token FP8 KV pool

> **Why two profiles?** DFlash 2 makes the Spark a responsive daily driver for interactive work and coding; EAGLE turns it into a 32-seat agent engine. Read [Kearuga: Practical Insights](INSIGHTS.md) for profile rationale, priority-TTFT results, quantization sensitivity analysis and memory trade-offs.

---

## 📢 Recent Updates

See the full [Changelog (CHANGELOG.md)](CHANGELOG.md) for release notes.

* **v0.3.0**:
  * **Kernel & Overlay Upgrades**: Integrated zero-allocation candidate projection in `dflash.py` and register-resident Triton selector walk to optimize C1–C4 decode loops.
  * **Hardware Hardening**: Added `--ulimit memlock=-1:-1`, `--ulimit stack=67108864`, and explicit Blackwell flags (`FLASHINFER_CUDA_ARCH_LIST="12.1f"`, `CUTE_DSL_ARCH="sm_120a"`).
  * **Context & Quality Validation**: Added 262K Needle-In-A-Haystack (`bench/niah.py`) and 10-check deterministic Semantic Gate (`bench/semantic_gate.py`).
  * **Fidelity Sensitivity Map**: Added tensor sensitivity mapping in `INSIGHTS.md` based on EXL3 mixed-precision research to bridge NVFP4 with BF16 reasoning fidelity.

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
| [MiaAI](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) DFlash 2 | — | 50.9 | 111.6 |
| [r0b0tlab](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang) | 28 | — | 55 |
| [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | 34 | — | 64 |

### EAGLE: High-concurrency throughput (C8–C32)

> **Need C16/C32? EAGLE scales to ~330/~535 tok/s.**

Unique request suffixes, forced 512-token outputs, thinking off, one DGX Spark:

| Profile | C8 | C16 | C32 | Typical saturated TTFT p95 |
|---|---:|---:|---:|---:|
| **EAGLE 3/1/4, 32 seats (this recipe)** | **181–193** | **320–335** | **527–539** | **0.32 / 0.49 / 0.87 s** |
| DFlash 2, four seats (this recipe) | ~118* | — | — | — |
| [Weschera](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) capacity profile (published) | 116.8 | 153.9 | 178.3 | — |
| [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark) vLLM (published) | 246 | — | — | — |

\* DFlash C8 queues behind four admitted requests.

### TTFT: Idle and saturated responsiveness

#### Idle TTFT

| Profile | ~645 prompt tokens | ~4.7K prompt tokens | ~18.7K prompt tokens |
|---|---:|---:|---:|
| **DFlash 2** | **0.35 s** | **2.17 s** | **11.10 s** |
| **EAGLE 3/1/4** | **0.30 s** | **2.20 s** | **10.50 s** |

#### Priority scheduling under saturation

| Profile and load | Normal priority TTFT | Interactive priority `100` | Reduction |
|---|---:|---:|---:|
| DFlash 2, all four seats occupied | 43.15 s | **2.63 s** | **~94%** |
| EAGLE, all 32 seats occupied | 73.30 s | **2.76 s** | **96.2%** |

```json
{
  "model": "qwen3.8-27b-sglang",
  "priority": 100,
  "messages": [{"role": "user", "content": "Interactive request"}]
}
```

---

## Deployment

> **Clone, build, boot. No external launcher required.**

```bash
git clone https://github.com/0xWhiteMage/Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2.git
cd Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2
cp .env.sample .env

# 1. Build the DFlash 2 overlay image
bash patch/build-dflash2-image.sh

# 2. Launch DFlash 2 (Daily Driver)
./start-dflash2.sh

# 3. Check health and models
curl -s http://127.0.0.1:8888/v1/models
curl -s http://127.0.0.1:8888/health
```

Stop: `./stop.sh` (or `./stop.sh --clean` to clear Triton caches).

### Benchmark & Validation Suite

```bash
python3 bench/semantic_gate.py                # 10-point deterministic correctness gate & canary
python3 bench/niah.py --context-size 65536    # Needle-in-a-Haystack long-context retrieval
python3 bench/run_quality_set.py              # 200-question quality suite
./bench/bench.sh                              # essay, tool-call, and 16K-TTFT smoke
python3 bench/ndec.py                         # C1 net-decode delta probes
python3 bench/scale.py --widths 4             # DFlash C4 scale probe
python3 bench/scale.py                        # EAGLE C8/C16/C32 scale probe
python3 bench/priority_ttft.py --streams 32   # saturated priority probe
```

---

## Configuration

| Setting | Canonical value | Benefit |
|---|---|---|
| DFlash profile | draft `8`, window `2048`, four seats | Fast C1/C4 decode (~65 tok/s) |
| EAGLE profile | `3/1/4`, 32 seats, continuous decode `1` | C8/C16/C32 throughput (~535 tok/s) |
| Model revisions | Target `554ebba9…`, draft `50307d4…` | Reproducible weights |
| Selector | SGLang [#35496](https://github.com/sgl-project/sglang/pull/35496) adaptation | Captures NVFP4 selector in CUDA graph |
| Context | `262144` per request | Native context; YaRN off |
| Shared KV | `1048576`, `fp8_e4m3` | 32 GiB target KV allocation |
| Hardware Flags | `--ulimit memlock=-1:-1`, `FLASHINFER 12.1f` | Eliminates TMA/paging faults on GB10 |
| CPU Pinning | `5-9,15-19` | Cortex-X5 core affinity |
| Priority | default `0`, interactive `100` | Protects interactive decode latency |
| Chat | thinking on, `qwen3_coder` tools | Reasoning and tool calling without restart |

---

## License

MIT for this recipe. The launcher, weights, and datasets keep their own licences.
