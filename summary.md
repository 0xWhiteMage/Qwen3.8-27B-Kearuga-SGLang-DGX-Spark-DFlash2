# 🧙‍♂️ Kearuga Model Suite: Release Summary

## 📌 Certified Production Checkpoints

* **Target Model (27B)**:
  * **Repository**: [`0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4)
  * **Pinned Commit SHA**: `8ea86bdcdd34c84b3d25b69fb7fcc8fc48d0cdd0`
  * **Tensor Layout**: Complete 2,194 tensors across 3 shards (includes all 193 `weight_scale_2`, 401 `input_scale`, and 333 visual tensors across 27 vision blocks).
  * **Context**: 262,144 tokens.

* **Speculative Drafter (DFlash 2)**:
  * **Repository**: [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2)
  * **Pinned Commit SHA**: `3a5f5763bbab5328e1ea8da06cc7875211a96d8d`
  * **Precision**: Selective Hybrid FP8/BF16 (BF16 `qkv_proj` & `out_proj` + FP8 E4M3 MLP).
  * **Context**: 262,144 tokens.
  * **SGLang Kernel**: `fused_kv_materialization` is 100% ENABLED.

---

## 🎛️ Hardware & Execution Profile

* **Hardware Platform**: NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory).
* **Serving VRAM Envelope**: 31.37 GiB (Target) + 2.39 GiB (Drafter) + 32.0 GiB (1M-Token KV Pool) = ~65.8 GiB total (~62 GiB headroom).
* **Launch Profiles**:
  * `start-dflash2.sh`: Interactive low-latency profile (65–82 tok/s C1 decode).
  * `start-eagle.sh`: High-concurrency agent profile (527–539 tok/s C32 throughput).
