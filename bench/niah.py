#!/usr/bin/env python3
"""Needle-In-A-Haystack (NIAH) long-context verification probe up to 262K tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
import sys

NEEDLE_SECRET = "SPARK-GB10-262K-RETRIEVAL-KEY-7734"
FILLER = (
    "The DGX Spark features 128 GB of unified memory and Blackwell tensor cores. "
    "A serving scheduler orchestrates KV cache, speculative drafting, and prefix caching. "
    "Distributed linear attention layers maintain recurrence across long sequences without degradation. "
)

def build_prompt(total_tokens_approx: int, depth_fraction: float = 0.5) -> tuple[str, str]:
    token_len_est = len(FILLER.split())
    num_blocks = max(1, int(total_tokens_approx / token_len_est))
    needle_block_idx = int(num_blocks * depth_fraction)

    blocks = []
    for i in range(num_blocks):
        if i == needle_block_idx:
            blocks.append(f"SPECIAL RECORD: The unique secret verification code is {NEEDLE_SECRET}. Remember this code.")
        else:
            blocks.append(FILLER)

    context = "\n".join(blocks)
    prompt = f"{context}\n\nQuestion: What is the unique secret verification code mentioned in the text? Reply only with the code."
    return prompt, NEEDLE_SECRET

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--depths", nargs="+", type=float, default=[0.25, 0.50, 0.90],
                    help="Needle placement fractions (e.g. 0.25 0.50 0.90)")
    ap.add_argument("--context-size", type=int, default=65536,
                    help="Target context length in tokens to test (e.g. 32768, 65536, 131072, 262144)")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    models_req = urllib.request.Request(f"{base_url}/models")
    with urllib.request.urlopen(models_req, timeout=30) as r:
        model = json.loads(r.read().decode())["data"][0]["id"]

    print(f"=== NIAH Context Probe ({args.context_size} tokens) on {model} ===")
    all_passed = True

    for d in args.depths:
        prompt, secret = build_prompt(args.context_size, depth_fraction=d)
        print(f"Testing depth {d*100:.0f}% (~{len(prompt.split())} words)...", end=" ", flush=True)
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode())
            text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            dt = time.perf_counter() - t0
            passed = secret in text
            if passed:
                print(f"PASS (latency {dt:.2f}s, response: {text.strip()!r})")
            else:
                print(f"FAIL (response: {text.strip()!r})")
                all_passed = False
        except Exception as e:
            print(f"ERROR: {e}")
            all_passed = False

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
