#!/usr/bin/env python3
"""Optional JSONL importer (developer tool; NOT part of the running C++ service)."""
import argparse,json,time,urllib.request
p=argparse.ArgumentParser(description='Insert JSONL vectors through HTTP in batches')
p.add_argument('file',help='One JSON object per line, each with id, vector, optional metadata')
p.add_argument('--url',default='http://127.0.0.1:8080')
p.add_argument('--batch',type=int,default=256)
a=p.parse_args()
if not 1<=a.batch<=1024:p.error('--batch must be in 1..1024')

def flush(items):
    if not items:return
    request=urllib.request.Request(a.url+'/v1/vectors/batch',json.dumps({'vectors':items},separators=(',',':')).encode(),{'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request,timeout=180) as resp:
        result=json.load(resp)
    return result

t=time.perf_counter();total=0;batch=[]
with open(a.file,encoding='utf-8') as f:
    for line_no,line in enumerate(f,1):
        if not line.strip():continue
        try:item=json.loads(line)
        except ValueError as e:raise ValueError(f'line {line_no}: invalid JSON') from e
        batch.append(item)
        if len(batch)>=a.batch:
            flush(batch);total+=len(batch);batch.clear()
            if total%10000<a.batch:print(f'{total:,} vectors loaded',flush=True)
    if batch:flush(batch);total+=len(batch)
secs=time.perf_counter()-t
print(f'Finished {total:,} vectors in {secs:.2f}s ({total/max(secs,1e-9):,.1f} vectors/s)')
