#!/usr/bin/env python3
"""Deterministic semantic gate & canary suite for Qwen3.8-27B on SGLang."""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import sys

LEAKAGE = ("<|assistant|>", "<|im_start|>", "<|im_end|>", "chat_template", "<|user|>")

CANARIES = [
    {
        "id": "arithmetic_437",
        "prompt": "Compute 19 times 23. Reply only with the final integer number.",
        "max_tokens": 64,
        "check": lambda t: bool(re.search(r"\b437\b", t)),
    },
    {
        "id": "fizzbuzz_code",
        "prompt": "Write a Python function fizzbuzz(n) returning numbers 1 to n with Fizz for multiples of 3, Buzz for 5, and FizzBuzz for both. Emit code only.",
        "max_tokens": 256,
        "check": lambda t: "def fizzbuzz" in t and "FizzBuzz" in t,
    },
    {
        "id": "json_schema",
        "prompt": 'Generate a JSON object with keys "status": "ok", "code": 200, "active": true. Reply ONLY valid JSON.',
        "max_tokens": 128,
        "check": lambda t: json.loads(re.search(r"\{.*\}", t, re.DOTALL).group(0)) == {"status": "ok", "code": 200, "active": True} if re.search(r"\{.*\}", t, re.DOTALL) else False,
    },
    {
        "id": "logic_reasoning",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly? Reply strictly YES or NO followed by a single sentence reason.",
        "max_tokens": 64,
        "check": lambda t: "no" in t.lower(),
    },
]

def chat(base_url: str, model: str, prompt: str, max_tokens: int = 128) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return {
        "text": text,
        "wall_s": time.perf_counter() - t0,
        "usage": body.get("usage"),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    if not args.model:
        models_req = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(models_req, timeout=30) as r:
            m_body = json.loads(r.read().decode())
            model = m_body["data"][0]["id"]
    else:
        model = args.model

    print(f"Running semantic gate against {base_url} (model: {model})...")
    results = []
    all_passed = True

    for c in CANARIES:
        res = chat(base_url, model, c["prompt"], c["max_tokens"])
        text = res["text"]
        
        # Check leakage
        has_leakage = any(tok in text for tok in LEAKAGE)
        # Check rule
        try:
            passed = c["check"](text) and not has_leakage
        except Exception as e:
            passed = False

        status_str = "PASS" if passed else "FAIL"
        print(f"  [{status_str}] {c['id']}: {text[:90]!r}")
        if not passed:
            all_passed = False

        results.append({
            "id": c["id"],
            "passed": passed,
            "has_leakage": has_leakage,
            "wall_s": round(res["wall_s"], 4),
            "text": text,
        })

    summary = {
        "all_passed": all_passed,
        "total_checks": len(CANARIES),
        "passed_checks": sum(1 for r in results if r["passed"]),
        "results": results,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote report to {args.output}")

    print(f"Final Disposition: {'PASSED (All gates clear)' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
