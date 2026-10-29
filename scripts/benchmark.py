#!/usr/bin/env python3
"""Simple HTTP throughput test. Load index first; works with Python stdlib."""
import argparse,concurrent.futures,json,time,statistics,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8080');p.add_argument('--dimensions',type=int,default=1024);p.add_argument('--workers',type=int,default=16);p.add_argument('--requests',type=int,default=1000);p.add_argument('--k',type=int,default=10)
a=p.parse_args();v=[0.0]*a.dimensions;v[0]=1.0
body=json.dumps({'vector':v,'k':a.k}).encode()
def one(_):
    t=time.perf_counter()
    req=urllib.request.Request(a.url+'/v1/search',body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as resp:json.load(resp)
    return (time.perf_counter()-t)*1000
# Warmup before measuring
for i in range(5):one(i)
t=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:dur=list(pool.map(one,range(a.requests)))
secs=time.perf_counter()-t;s=sorted(dur)
print(json.dumps({'requests':a.requests,'concurrency':a.workers,'qps':round(a.requests/secs,1),'p50_ms':round(s[int(.5*(len(s)-1))],3),'p95_ms':round(s[int(.95*(len(s)-1))],3),'p99_ms':round(s[int(.99*(len(s)-1))],3)},indent=2))
