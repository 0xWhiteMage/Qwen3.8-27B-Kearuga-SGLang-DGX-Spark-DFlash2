# Qwen3.8-27B Kearuga: Master Architecture & Execution Summary

> Technical blueprint, kernel specifications, hardware runtime contracts, and agent orchestration guide for serving **Qwen3.8-27B** with **SGLang**, **DFlash 2**, and **EAGLE** on the **NVIDIA DGX Spark (GB10 / SM121)**.

---

## 1. Executive Mission & System Architecture

This recipe turns a single 128 GB NVIDIA DGX Spark into a dual-mode high-performance serving system:
1. **Interactive Daily Driver (DFlash 2)**: ~65 tok/s net C1 decode, ~120 tok/s aggregate C4, native tool-calling and reasoning.
2. **Agent Engine (EAGLE 3/1/4)**: 32 admitted requests, scaling to ~191 tok/s at C8, ~330 tok/s at C16, and ~535 tok/s at C32.
3. **Native Context**: Four concurrent 262,144-token conversations within a shared 1,048,576 FP8 KV cache pool (~32 GiB KV memory).

---

## 2. Kernel & Runtime Overlay Modifications

Located in `patch/overlay-dflash2/`:

| Component | Target File | Optimization |
|---|---|---|
| **Zero-Alloc Projection** | `sglang/srt/models/dflash.py` | Eliminates dynamic PyTorch tensor reallocation during candidate projection via contiguous logit slice reuse. |
| **Quantized Selector** | `sglang/srt/models/dflash.py` | Adapts SGLang PR #35496 to project ModelOpt NVFP4 quantized `lm_head` in CUDA graphs without non-contiguous weight stride crashes. |
| **Triton Selector Walk** | `sglang/kernels/ops/speculative/dflash.py` | Unrolls candidate tree traversal loop in streaming multiprocessor registers for low concurrency ($C \le 4$). |
| **Aux Hidden States** | `sglang/srt/speculative/dflash_worker_v2.py` | Streamlines target layer hidden-state capture (`[1, 13, 25, 37, 49]`) into draft KV cache without CPU synchronization. |
| **Automated Build Tests** | `patch/build-dflash2-image.sh` | Bakes unit tests (`test_dflash_logits.py`, `test_spec_aux_hidden_state.py`) into the Docker image with SHA256 manifest verification. |

---

## 3. DGX Spark (GB10 / SM121) Hardware Runtime Contract

In `start-dflash2.sh` and `start-eagle.sh`:

```bash
docker run -d   --name "qwen3.8-27b-sglang"   --network host --ipc host --privileged   --cap-add IPC_LOCK   --ulimit memlock=-1:-1   --ulimit stack=67108864   --gpus all --shm-size 32g   --cpuset-cpus "5-9,15-19"   -e FLASHINFER_CUDA_ARCH_LIST="12.1f"   -e CUTE_DSL_ARCH="sm_120a"   -e PYTHONUNBUFFERED=1   ...
```

* **`--ulimit memlock=-1:-1` & `--cap-add IPC_LOCK`**: Eliminates virtual memory page-locking latency and Triton IPC faults under high KV allocation pressure.
* **`FLASHINFER_CUDA_ARCH_LIST="12.1f"`**: Forces FlashInfer to compile native Blackwell TMA/MMA tensor instructions, preventing unaligned instruction fallbacks.
* **`--cpuset-cpus "5-9,15-19"`**: Pins thread execution to the Grace CPU Cortex-X5 high-performance cluster.
* **`--max-prefill-tokens 8192` & `--chunked-prefill-size 8192`**: Enforces strict batch prefill budgets during concurrent admissions.

---

## 4. Bridging NVFP4 to EXL3: Tensor Sensitivity Mapping

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Qwen3.8-27B Tensor Sensitivity Hierarchy                   │
├───────────────────────┬────────────────────────────┬─────────────────────────┤
│ Tier 1: Extreme       │ Tier 2: High Sensitivity   │ Tier 3: High Capacity   │
│ (Keep in BF16)        │ (Keep in FP8 fp8_e4m3)     │ (Quantize to NVFP4)     │
├───────────────────────┼────────────────────────────┼─────────────────────────┤
│ • embed_tokens        │ • Attention Q, K, V, O     │ • MLP gate_proj         │
│ • lm_head             │ • GDN in_proj / conv1d     │ • MLP up_proj           │
│ • Layers 0–1, 62–63   │ • Layer norms & biases     │ • MLP down_proj         │
│   (Boundary Layers)   │ • MTP head projections     │   (Middle Layers 2–61)  │
└───────────────────────┴────────────────────────────┴─────────────────────────┘
```

---

## 5. Verification & Benchmark Protocol

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
