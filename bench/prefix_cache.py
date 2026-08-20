#!/usr/bin/env python3
import json,time,urllib.request,uuid
URL='http://127.0.0.1:8888/v1/chat/completions'
static=("Stable system knowledge, tool schema, policy, examples, and project context remain byte-identical across turns. ")*1600

def run(suffix):
 body={'model':'qwen3.8-27b-sglang','messages':[{'role':'user','content':static+'\n'+suffix}],
       'max_tokens':2,'temperature':0,'stream':True,'priority':100,
       'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 t0=time.perf_counter(); first=None; usage=None
 with urllib.request.urlopen(req,timeout=300) as r:
  for raw in r:
   if not raw.startswith(b'data:') or raw.startswith(b'data: [DONE]'): continue
   try: obj=json.loads(raw[5:])
   except Exception: continue
   delta=((obj.get('choices') or [{}])[0].get('delta') or {})
   if first is None and (delta.get('content') or delta.get('reasoning_content')): first=time.perf_counter()
   if obj.get('usage'): usage=obj['usage']
 return {'ttft_s':(first or time.perf_counter())-t0,'usage':usage}
print(json.dumps({'cold':run('Request A '+str(uuid.uuid4())+'. Reply READY.'),'warm_prefix':run('Request B '+str(uuid.uuid4())+'. Reply READY.')},indent=2))
