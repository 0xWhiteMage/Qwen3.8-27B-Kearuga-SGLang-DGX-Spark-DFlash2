#!/usr/bin/env python3
"""Controlled concurrency throughput, TTFT, correctness, and stability probe for either profile."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROMPT = (
    "Write a production-quality Python implementation of a bounded thread-safe "
    "LRU cache with TTL expiry, type hints, and docstrings. Output code only."
)
CANARIES = [
    ("mul_19x23", "What is 19 times 23? Reply with the integer only.", 16, lambda s: "437" in s.replace(" ", "")),
    ("gsm8k_flex", "Natalia sold 48 clips in April and half as many in May. How many clips altogether? Reply with the integer only.", 16, lambda s: bool(re.search(r"\b72\b", s))),
    ("code_fizz", "Write a Python function fizzbuzz(n: int) -> list[str] for 1..n. Code only.", 200, lambda s: "FizzBuzz" in s and "def " in s),
]

def request_json(url: str, body: dict, timeout: int = 600) -> tuple[dict, float]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.perf_counter() - t0

def get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)

def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = min(len(ys)-1, max(0, math.ceil(q * len(ys)) - 1))
    return ys[i]

def stream_one(base: str, model: str, idx: int, max_tokens: int) -> dict:
    suffix = f"\nRequest marker {idx}: keep behaviour deterministic."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT + suffix}],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = {}; finish = None; chars = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text and first is None:
                    first = time.perf_counter()
                chars += len(text)
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    comp = int(usage.get("completion_tokens") or max_tokens)
    return {
        "idx": idx,
        "ttft_s": round((first or end)-t0, 4),
        "e2e_s": round(end-t0, 4),
        "completion_tokens": comp,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "finish_reason": finish,
        "chars": chars,
    }

def run_width(base: str, model: str, width: int, max_tokens: int, rep: int) -> dict:
    t0 = time.perf_counter(); rows=[]
    with ThreadPoolExecutor(max_workers=width) as ex:
        fs=[ex.submit(stream_one, base, model, rep*1000+i, max_tokens) for i in range(width)]
        for f in as_completed(fs): rows.append(f.result())
    wall=time.perf_counter()-t0
    rows.sort(key=lambda x:x["idx"])
    total=sum(x["completion_tokens"] for x in rows)
    ttfts=[x["ttft_s"] for x in rows]
    return {
        "width": width, "rep": rep, "wall_s": round(wall,4),
        "total_completion_tokens": total,
        "aggregate_tok_s": round(total/wall,3),
        "ttft_mean_s": round(statistics.mean(ttfts),4),
        "ttft_p50_s": round(statistics.median(ttfts),4),
        "ttft_p95_s": round(percentile(ttfts,.95),4),
        "ttft_max_s": round(max(ttfts),4),
        "per_stream": rows,
    }

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--tag", help="Result filename stem; defaults to a UTC timestamp")
    ap.add_argument("--widths", default="8,16,32")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out-dir", default="bench/results")
    args=ap.parse_args(); base=args.base.rstrip("/"); root=base.rsplit("/v1",1)[0]
    tag=args.tag or datetime.now(timezone.utc).strftime("scale-%Y%m%d-%H%M%S")
    model=get_json(base+"/models")["data"][0]["id"]
    out={"tag":tag,"timestamp":datetime.now(timezone.utc).isoformat(),"base":base,"model":model,"max_tokens":args.max_tokens,"server_before":get_json(root+"/v1/loads"),"canaries":[],"runs":{}}
    for cid,prompt,limit,check in CANARIES:
        obj,wall=request_json(base+"/chat/completions",{"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0,"top_p":1.0,"max_tokens":limit,"chat_template_kwargs":{"enable_thinking":False}})
        msg=(obj.get("choices") or [{}])[0].get("message") or {}; text=msg.get("content") or ""; reasoning=msg.get("reasoning_content") or ""
        rec={"id":cid,"pass":bool(check(text) and not reasoning),"wall_s":round(wall,4),"preview":" ".join(text.split())[:180]}
        out["canaries"].append(rec); print(f"canary {cid}: {rec['pass']} {rec['preview']!r}",flush=True)
        if not rec["pass"]: return 2
    for width in [int(x) for x in args.widths.split(",") if x]:
        out["runs"][f"c{width}"]=[]
        for rep in range(1,args.reps+1):
            rec=run_width(base,model,width,args.max_tokens,rep)
            out["runs"][f"c{width}"].append(rec)
            print(f"c{width} rep{rep}: {rec['aggregate_tok_s']} tok/s wall={rec['wall_s']}s ttft p50/p95={rec['ttft_p50_s']}/{rec['ttft_p95_s']}s",flush=True)
    out["server_after"]=get_json(root+"/v1/loads")
    out["summary"]={}
    for key,recs in out["runs"].items():
        vals=[r["aggregate_tok_s"] for r in recs]; p95=[r["ttft_p95_s"] for r in recs]
        out["summary"][key]={"mean_tok_s":round(statistics.mean(vals),3),"min_tok_s":min(vals),"max_tok_s":max(vals),"mean_ttft_p95_s":round(statistics.mean(p95),4)}
    path=Path(args.out_dir); path.mkdir(parents=True,exist_ok=True); target=path/f"{tag}.json"; target.write_text(json.dumps(out,indent=2))
    print(json.dumps(out["summary"],indent=2)); print(f"wrote {target}")
    return 0
if __name__=="__main__": raise SystemExit(main())
