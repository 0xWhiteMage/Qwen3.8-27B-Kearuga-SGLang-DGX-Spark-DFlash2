#!/usr/bin/env python3
import argparse, json, threading, time, urllib.request, uuid

ap=argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8888/v1/chat/completions')
ap.add_argument('--long-priority', type=int, default=0)
ap.add_argument('--interactive-priority', type=int, default=100)
ap.add_argument('--long-max-tokens', type=int, default=800)
ap.add_argument('--streams', type=int, default=4)
args=ap.parse_args()

ready=[threading.Event() for _ in range(args.streams)]
stop=threading.Event()
long_rows=[None]*args.streams

def request(payload, on_first=None, stop_early=False):
    req=urllib.request.Request(args.url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    t0=time.perf_counter(); first=None; chunks=0
    with urllib.request.urlopen(req,timeout=600) as r:
        for raw in r:
            if not raw.startswith(b'data:') or raw.startswith(b'data: [DONE]'):
                continue
            try: obj=json.loads(raw[5:])
            except Exception: continue
            delta=((obj.get('choices') or [{}])[0].get('delta') or {})
            if delta.get('content') or delta.get('reasoning_content'):
                chunks += 1
                if first is None:
                    first=time.perf_counter()
                    if on_first: on_first()
                if stop_early or stop.is_set():
                    break
    return {'ttft_s':None if first is None else first-t0,'elapsed_s':time.perf_counter()-t0,'chunks':chunks}

def long_worker(i):
    para=('A scheduler coordinates requests, memory, prefix cache, prefill, and decode on one accelerator. ')*50
    payload={'model':'qwen3.8-27b-sglang','messages':[{'role':'user','content':f'Batch {i} {uuid.uuid4()}. Write a detailed technical analysis.\n{para}'}],
             'max_tokens':args.long_max_tokens,'temperature':0,'stream':True,'priority':args.long_priority,
             'chat_template_kwargs':{'enable_thinking':False}}
    try: long_rows[i]=request(payload,on_first=ready[i].set)
    except Exception as e: long_rows[i]={'error':repr(e)}; ready[i].set()

threads=[threading.Thread(target=long_worker,args=(i,),daemon=True) for i in range(args.streams)]
for t in threads: t.start()
for e in ready:
    if not e.wait(180):
        stop.set(); raise SystemExit('long request did not reach first token')

time.sleep(0.5)
prompt=('The quick brown fox jumps over the lazy dog while a river passes through the valley. ')*300
interactive={'model':'qwen3.8-27b-sglang','messages':[{'role':'user','content':f'Interactive {uuid.uuid4()}. Reply only READY.\n{prompt}'}],
             'max_tokens':2,'temperature':0,'stream':True,'priority':args.interactive_priority,
             'chat_template_kwargs':{'enable_thinking':False}}
try:
    high=request(interactive,stop_early=True)
finally:
    stop.set()
for t in threads: t.join(timeout=10)
print(json.dumps({'interactive':high,'long':long_rows,'priorities':[args.long_priority,args.interactive_priority]},indent=2))
