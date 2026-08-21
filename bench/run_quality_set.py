#!/usr/bin/env python3
"""Run the 200-question quality set against a serving endpoint; grade + persist rows.

Usage: python3 bench/run_quality_set.py --base-url http://127.0.0.1:8888/v1 --run-id eval-v1
Output: <run-id>.rows.jsonl (per-row outputs) + <run-id>.summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import sys
from pathlib import Path


def get_default_model(base: str) -> str:
    try:
        url = base.rstrip("/") + "/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read().decode())
            return body["data"][0]["id"]
    except Exception:
        return "qwen3.8-27b-sglang"


def chat(base: str, model: str, prompt: str, max_tokens: int, reasoning_effort: str = "medium", temperature: float = 0.6) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    msg = body["choices"][0]["message"]
    return {
        "text": msg.get("content") or "",
        "finish": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
        "elapsed": time.perf_counter() - t,
    }


def grade_numeric(text: str, ref: str) -> bool:
    m = re.search(r"-?\$?\d[\d,]*\.?\d*", text.replace(",", ""))
    refc = ref.replace(",", "")
    return bool(m) and m.group(0).replace("$", "").replace(",", "") == refc


def grade_exec(text: str, ref: dict) -> bool:
    """Extract python block; run entry-point call against the canonical test."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    code = re.sub(r"^python\s*$", "", code, flags=re.M)
    ns: dict = {}
    try:
        exec(compile(code, "<gen>", "exec"), ns)
    except Exception:
        return False
    fn = ref.get("entry_point")
    tests = ref.get("test")
    if not fn or not tests:
        return True
    try:
        exec(compile(tests, "<tests>", "exec"), ns)
        return True
    except Exception:
        return False


def grade_ifeval(text: str, ref: dict) -> bool:
    rule = ref.get("rule")
    arg = ref.get("arg")
    if rule == "starts_with":
        return text.strip().startswith(arg)
    if rule == "ends_with":
        return text.strip().endswith(arg)
    if rule == "contains_keyword":
        return arg.lower() in text.lower()
    if rule == "word_count_min":
        return len(text.split()) >= int(arg)
    if rule == "word_count_max":
        return len(text.split()) <= int(arg)
    if rule == "json_format":
        try:
            json.loads(text)
            return True
        except Exception:
            return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-id", default=f"eval-{int(time.time())}")
    ap.add_argument("--set", default="bench/artifacts/quality-200.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()

    model = a.model or get_default_model(a.base_url)
    print(f"Running Quality-200 against {a.base_url} (model: {model}, run-id: {a.run_id})")

    rows = [json.loads(l) for l in open(a.set, encoding="utf-8")]
    rows_path = Path(f"{a.run_id}.rows.jsonl")
    done_ids: set[str] = set()
    out_rows: list[dict] = []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            try:
                out_rows.append(json.loads(line))
                done_ids.add(out_rows[-1]["id"])
            except Exception:
                break
        print(f"resuming: {len(done_ids)} rows already collected", flush=True)

    t0 = time.time()
    with rows_path.open("a", encoding="utf-8") as rows_fh:
        for i, r in enumerate(rows):
            if r["id"] in done_ids:
                continue
            mt = 1536 if r["family"] in ("humaneval", "agentic_coding") else 1024
            try:
                res = chat(a.base_url, model, r["prompt"], mt)
            except Exception as exc:
                res = {"text": "", "finish": "error", "usage": None, "elapsed": 0.0, "error": repr(exc)}
            g = r["grade"]
            ref = r["reference"]
            if g == "numeric_exact":
                ok = grade_numeric(res["text"], ref)
            elif g == "exec":
                ok = grade_exec(res["text"], ref if isinstance(ref, dict) else {})
            elif g == "ifeval_strict":
                ok = grade_ifeval(res["text"], ref)
            else:
                ok = None
            out_rows.append({
                "id": r["id"], "family": r["family"], "grade": g, "passed": ok,
                "finish": res.get("finish"), "elapsed": res.get("elapsed"),
                "completion_tokens": (res.get("usage") or {}).get("completion_tokens"),
                "text": res["text"][:4000],
            })
            rows_fh.write(json.dumps(out_rows[-1]) + "\n")
            rows_fh.flush()
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                done = sum(1 for x in out_rows if x["passed"] is not None and x["passed"])
                graded = sum(1 for x in out_rows if x["passed"] is not None)
                print(f"{i+1}/200 graded={graded} pass={done} ({time.time()-t0:.0f}s)", flush=True)

    fam_stats: dict = {}
    for x in out_rows:
        f = fam_stats.setdefault(x["family"], {"n": 0, "passed": 0, "graded": 0})
        f["n"] += 1
        if x["passed"] is not None:
            f["graded"] += 1
            f["passed"] += bool(x["passed"])

    summary = {
        "run_id": a.run_id, "base_url": a.base_url, "model": model, "elapsed_s": time.time() - t0,
        "families": fam_stats,
        "auto_graded_pass": sum(1 for x in out_rows if x["passed"]),
        "auto_graded_total": sum(1 for x in out_rows if x["passed"] is not None),
    }
    Path(f"{a.run_id}.rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_rows), encoding="utf-8")
    Path(f"{a.run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== Quality-200 Summary ===")
    print(json.dumps(summary["families"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
