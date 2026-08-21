# Qwen3.8-27B Kearuga: Master Architecture & Execution Summary

> Technical blueprint, kernel specifications, hardware runtime contracts, and agent orchestration guide for serving **Qwen3.8-27B** with **SGLang**, **DFlash 2**, and **EAGLE** on the **NVIDIA DGX Spark (GB10 / SM121)**.

---

## 1. System Architecture & Active Checkpoints

This recipe provides two qualified serving profiles on a single 128 GB NVIDIA DGX Spark:
1. **Interactive Daily Driver (DFlash 2)**: ~65 tok/s net C1 decode, ~120 tok/s aggregate C4, native tool-calling and reasoning.
2. **Agent Engine (EAGLE 3/1/4)**: 32 admitted requests, scaling to ~191 tok/s at C8, ~330 tok/s at C16, and ~535 tok/s at C32.
3. **Native Context**: Four concurrent 262,144-token conversations within a shared 1,048,576 FP8 KV cache pool (~32 GiB KV memory).

### Pinned Model Weights Used:
* **Target Model**: `RadixArk/Qwen3.8-27B-NVFP4` (Pinned commit `554ebba9b5f1b79dc11246341960360e6ef05ef4`)
* **DFlash 2 Draft**: `z-lab/Qwen3.8-27B-DFlash2` (Pinned commit `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`)
* **EAGLE 3/1/4**: In-checkpoint MTP draft head for C8–C32 concurrency

---

## 2. Exact Files Modified & Added in this Repository

| Component | File Path | Status | What Changed |
|---|---|---|---|
| **DFlash 2 Overlay** | `patch/overlay-dflash2/sglang/srt/models/dflash.py` | Modified | SGLang PR #35496 adaptation for ModelOpt NVFP4 `lm_head` + contiguous candidate logit projection |
| **Triton Kernels** | `patch/overlay-dflash2/sglang/kernels/ops/speculative/dflash.py` | Modified | Validated candidate selector loop and contiguous memory loads |
| **Overlay Manifest** | `patch/overlay-dflash2/MANIFEST.sha256` | Modified | Updated SHA256 checksums of all 6 overlay files for bit-exact builds |
| **Overlay Unit Tests** | `patch/overlay-dflash2/test/` | Added | Unit tests for candidate logits projection and aux hidden state mapping |
| **Image Builder** | `patch/build-dflash2-image.sh` | Modified | Updated to copy unit tests into Docker image and verify checksums |
| **DFlash Launcher** | `start-dflash2.sh` | Modified | Added `--ulimit memlock=-1:-1`, `--cap-add IPC_LOCK`, `FLASHINFER 12.1f`, `--max-prefill-tokens`, health watchdog, and `.env` config |
| **EAGLE Launcher** | `start-eagle.sh` | Modified | Added `--ulimit memlock=-1:-1`, `--cap-add IPC_LOCK`, `FLASHINFER 12.1f`, `--max-prefill-tokens`, health watchdog, and `.env` config |
| **Stop Script** | `stop.sh` | Modified | Added `.sglang.pid` removal and optional `--clean` flag for purging Triton cache |
| **Config Sample** | `.env.sample` | Modified | Documented all runtime ports, hosts, context lengths, and speculative knobs |
| **Semantic Gate** | `bench/semantic_gate.py` | Added | 10-point deterministic correctness gate (anti-leakage, JSON schemas, arithmetic canary `19 × 23 → 437`) |
| **NIAH 262K Probe** | `bench/niah.py` | Added | Token-accurate Needle-In-A-Haystack retrieval at 25%, 50%, 90% depths up to 262K context |
| **Quality Suite** | `bench/run_quality_set.py` | Added | 200-question capability evaluator (GSM8K, HumanEval, IFEval, Agentic Coding, Hard Reasoning) |
| **Flex Math Scorer** | `bench/score_flex_gsm8k.py` | Added | Math reasoning evaluation extractor |
| **Quality Artifacts** | `bench/artifacts/quality-200.jsonl` | Added | 200 held-out evaluation questions and references |
| **Benchmark Driver** | `bench/bench.sh` | Modified | Integrated semantic gate, throughput probes, and 16k TTFT benchmarks |
| **Changelog** | `CHANGELOG.md` | Added | Full release history following Keep-a-Changelog and SemVer |
| **Documentation** | `README.md` | Modified | Version `v0.3.0` badge, recent updates section, benchmark tables, and hardware flags |
| **Insights & Analysis**| `INSIGHTS.md` | Modified | Detailed operational takeaways, priority TTFT measurements, and quantization sensitivity hierarchy |
| **Master Summary** | `summary.md` | Added | Complete system architecture blueprint and agent orchestration guide |

---

## 3. DGX Spark (GB10 / SM121) Hardware Runtime Contract

In `start-dflash2.sh` and `start-eagle.sh`:

```bash
docker run -d \
  --name "qwen3.8-27b-sglang" \
  --network host --ipc host --privileged \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  --gpus all --shm-size 32g \
  --cpuset-cpus "5-9,15-19" \
  -e FLASHINFER_CUDA_ARCH_LIST="12.1f" \
  -e CUTE_DSL_ARCH="sm_120a" \
  -e PYTHONUNBUFFERED=1 \
  ...
```

* **`--ulimit memlock=-1:-1` & `--cap-add IPC_LOCK`**: Eliminates virtual memory page-locking latency and Triton IPC faults under high KV allocation pressure.
* **`FLASHINFER_CUDA_ARCH_LIST="12.1f"`**: Forces FlashInfer to compile native Blackwell TMA/MMA tensor instructions, preventing unaligned instruction fallbacks.
* **`--cpuset-cpus "5-9,15-19"`**: Pins thread execution to the Grace CPU Cortex-X5 high-performance cluster.
* **`--max-prefill-tokens 8192` & `--chunked-prefill-size 8192`**: Enforces strict batch prefill budgets during concurrent admissions.

---

## 4. Verification & Benchmark Protocol

### Stage 1: Deterministic Semantic Gate
```bash
python3 bench/semantic_gate.py --base-url http://127.0.0.1:8888/v1
```

### Stage 2: Quality-200 Capability Benchmark
```bash
python3 bench/run_quality_set.py --base-url http://127.0.0.1:8888/v1 --run-id eval-v1
```

### Stage 3: Long-Context Retrieval (262K NIAH)
```bash
python3 bench/niah.py --base-url http://127.0.0.1:8888/v1 --context-size 65536
python3 bench/niah.py --base-url http://127.0.0.1:8888/v1 --context-size 262144
```

### Stage 4: Throughput & Priority Preemption Ladder
```bash
python3 bench/ndec.py                         # Isolates net-decode tok/s
python3 bench/scale.py --widths 1,2,4,8,16,32 # Measures throughput and draft acceptance
python3 bench/priority_ttft.py --streams 32   # Verifies saturated priority TTFT reduction
```

---

## 5. Future Roadmap / Research Directions

These are theoretical quantization and model training techniques documented in `INSIGHTS.md` for future experimentation:
* **Hybrid Sensitivity Quantization**: Quantizing middle MLP layers in NVFP4 while preserving embeddings, `lm_head`, and boundary layers in BF16/FP8 to achieve EXL3-grade fidelity.
* **FP8 (`fp8_e4m3`) DFlash Draft Weights**: Converting the 3.2 GB BF16 draft model weights to `fp8_e4m3` (1.6 GB) to reduce memory bandwidth overhead on unified architectures.
