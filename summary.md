# Qwen3.8-27B Kearuga: Master Architecture & Agent Orchestration Runbook

> **Mission**: Deploy, optimize, calibrate, and verify the high-performance **Qwen3.8-27B** serving stack on the **NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory)** with **SGLang**, **DFlash 2**, and **EAGLE**.

---

## 1. System Architecture & Active Production Models

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           System Serving Architecture                           │
├──────────────────────────┬──────────────────────────┬───────────────────────────┤
│ Target Model (NVFP4)     │ Draft Model (DFlash 2)   │ Hardware (DGX Spark GB10) │
├──────────────────────────┼──────────────────────────┼───────────────────────────┤
│ • RadixArk NVFP4 target  │ • z-lab DFlash 2 draft   │ • Grace Blackwell SM121   │
│ • Hybrid FP8/FP4 weights │ • 5 layers, window 2048  │ • 128 GB Unified Memory   │
│ • SGLang RadixAttention  │ • Grouped conv + selector│ • 273 GB/s bandwidth      │
│ • Shared 1M token pool   │ • Zero-alloc logit proj  │ • Cortex-X5 core affinity │
└──────────────────────────┴──────────────────────────┴───────────────────────────┘
```

* **Target Model**: `RadixArk/Qwen3.8-27B-NVFP4` (Pinned commit `554ebba9b5f1b79dc11246341960360e6ef05ef4`)
* **DFlash 2 Draft**: `0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3` (Pinned commit `cd1f23d4ff625ac68ac08457331547e2edab3991`)
* **EAGLE 3/1/4**: In-checkpoint MTP draft head for C8–C32 high concurrency
* **Shared Memory Contract**: 1,048,576 FP8 KV cache pool (~32 GiB KV), native 262,144 tokens context per request, 4 admitted DFlash streams, 32 admitted EAGLE streams.

---

## 2. Complete Repository Map

| Component | File Path | Status | Purpose |
|---|---|---|---|
| **DFlash 2 Overlay** | `patch/overlay-dflash2/sglang/srt/models/dflash.py` | Modified | Candidate projection without dynamic memory allocation |
| **Triton Kernels** | `patch/overlay-dflash2/sglang/kernels/ops/speculative/dflash.py` | Modified | Candidate selector walk and block preparation |
| **Worker Coordinator**| `patch/overlay-dflash2/sglang/srt/speculative/dflash_worker_v2.py` | Modified | Speculative verification, Mamba state rollback, and grammar masks |
| **Overlay Manifest** | `patch/overlay-dflash2/MANIFEST.sha256` | Modified | SHA256 checksums of all 6 overlay files |
| **Overlay Unit Tests** | `patch/overlay-dflash2/test/` | Added | Unit tests for logits projection and aux hidden state mapping |
| **Image Builder** | `patch/build-dflash2-image.sh` | Modified | Bakes overlay files and unit tests into Docker image |
| **DFlash Launcher** | `start-dflash2.sh` | Modified | Interactive daily-driver launcher (C1–C4 decode, ~65 tok/s net) |
| **EAGLE Launcher** | `start-eagle.sh` | Modified | High-concurrency launcher (C8–C32 decode, ~535 tok/s) |
| **Stop Script** | `stop.sh` | Modified | Process cleanup and Triton cache purger |
| **Environment Config** | `.env.sample` | Modified | Pinned revisions, ports, memory fractions, and scheduler knobs |
| **Semantic Gate** | `bench/semantic_gate.py` | Added | 10-point deterministic correctness gate & arithmetic canary |
| **NIAH 262K Probe** | `bench/niah.py` | Added | Needle-In-A-Haystack long-context retrieval benchmark |
| **Quality-200 Suite** | `bench/run_quality_set.py` | Added | 200-question capability benchmark (GSM8K, HumanEval, IFEval) |
| **FP8 Drafter Tool** | `tools/convert_dflash_fp8.py` | Added | Converts DFlash2 draft weights to `fp8_e4m3` (1.6 GB) |
| **Hybrid NVFP4 Tool** | `tools/quantize_hybrid_nvfp4.py` | Added | Sensitivity-guided ModelOpt quantization from BF16 |
| **Documentation** | `README.md` / `INSIGHTS.md` | Modified | Architecture, benchmark logs, and EXL3 sensitivity analysis |

---

## 3. AGENT ORCHESTRATION PLAYBOOK

An autonomous agent executing work on this repository should follow these 4 operational phases:

### Phase 1: Environment & Kernel Overlay Verification
1. Verify overlay file integrity:
   ```bash
   cd patch/overlay-dflash2 && sha256sum -c MANIFEST.sha256
   ```
2. Build the qualified DFlash 2 Docker image:
   ```bash
   bash patch/build-dflash2-image.sh
   ```

### Phase 2: Serving Profile Deployment
Launch the primary interactive daily-driver profile:
```bash
./start-dflash2.sh
```
*Mandatory Hardware Flags Enforced in Launcher*:
* `--ulimit memlock=-1:-1 --cap-add IPC_LOCK`: Eliminates unified memory page locking latency.
* `FLASHINFER_CUDA_ARCH_LIST="12.1f" CUTE_DSL_ARCH="sm_120a"`: Blackwell SM121 TMA/MMA instruction targeting.
* `--cpuset-cpus "5-9,15-19"`: Grace CPU Cortex-X5 cluster affinity.
* `--max-prefill-tokens 8192 --chunked-prefill-size 8192`: Prevents prefill compute spikes from starving active decode streams.

### Phase 3: Automated Quality & Verification Protocol
Run all verification stages in sequence:
```bash
# 1. Deterministic Semantic Gate
python3 bench/semantic_gate.py --base-url http://127.0.0.1:8888/v1

# 2. 262K Long-Context Needle-In-A-Haystack
python3 bench/niah.py --base-url http://127.0.0.1:8888/v1 --context-size 65536

# 3. Quality-200 Capability Benchmark
python3 bench/run_quality_set.py --base-url http://127.0.0.1:8888/v1 --run-id eval-v1

# 4. Latency & Throughput Probes
./bench/bench.sh
python3 bench/ndec.py
python3 bench/scale.py --widths 4
```

### Phase 4: Model Weight Tuning & Optimization (Tools)

#### A. Converting DFlash 2 Draft Weights to FP8 (`fp8_e4m3`):
Reduces draft model memory traffic from 3.2 GB to 1.6 GB, cutting unified memory bus latency by ~45%:
```bash
python3 tools/convert_dflash_fp8.py \
    --input-model 0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3 \
    --output-dir ./models/Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3
```

#### B. Generating a Custom Hybrid NVFP4 Checkpoint from BF16:
Applies EXL3 sensitivity lessons (BF16 Head/Embeddings + FP8 Attention/GDN + NVFP4 MLPs) to achieve near-lossless reasoning:
```bash
python3 tools/quantize_hybrid_nvfp4.py \
    --model-id Qwen/Qwen3.8-27B \
    --output-dir ./models/Qwen3.8-27B-Kearuga-NVFP4
```

---

## 4. Operational Health & Troubleshooting Commands

* **Check HTTP Readiness**: `curl -fsS http://127.0.0.1:8888/health`
* **Inspect Active Models**: `curl -fsS http://127.0.0.1:8888/v1/models`
* **Tail Server Logs**: `tail -f .sglang.log`
* **Stop Container**: `./stop.sh`
* **Clean Shutdown & Triton Purge**: `./stop.sh --clean`
