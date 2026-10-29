#!/usr/bin/env python3
"""Run after CUDA Docker starts; validates GPU exact top-K across multiple 256-vector blocks.
Requires fresh index or may have additional records; use ID prefix and k to distinguish.
Creates 1024 varied embeddings and compares final GPU ranking/scores with CPU reference.
"""
import argparse,json,math,random,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8080');p.add_argument('--dimensions',type=int,default=1024);p.add_argument('--count',type=int,default=1024);p.add_argument('--k',type=int,default=20)
a=p.parse_args()
if a.count<512:p.error('--count >=512 required to span blocks')
rng=random.Random(71043)
def norm(xs):
    s=math.sqrt(sum(v*v for v in xs));return [v/s for v in xs]
def post(path, obj):
    req=urllib.request.Request(a.url+path,json.dumps(obj).encode(),{'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=120) as resp:return json.load(resp)

# All vectors have a distinctive first 4 components and varying remaining dimensions.
# This tests winner order across multiple tiles, masking and float-score accuracy.
items=[]
for i in range(a.count):
    v=[rng.uniform(-1,1) for _ in range(a.dimensions)]
    if i%200==0:v=[0.0]*a.dimensions;v[i%a.dimensions]=1.0
    items.append((f'gpu-check-{i:05d}',norm(v)))
for start in range(0,len(items),256):
    post('/v1/vectors/batch',{'vectors':[{'id':id,'vector':v} for id,v in items[start:start+256]]})
print('GPU correctness: indexed',len(items),'vectors')
for turn in range(8):
    query=norm([rng.uniform(-1,1) for _ in range(a.dimensions)])
    expected=sorted(((sum(x*y for x,y in zip(v,query)),id) for id,v in items),key=lambda x:(-x[0],x[1]))
    hits=post('/v1/search',{'vector':query,'k':a.k})['results']
    ids=[h['id'] for h in hits]
    # Assumes a fresh index; otherwise extra user data can legitimately rank higher.
    assert ids==[id for _,id in expected[:a.k]],(turn,ids,[id for _,id in expected[:a.k]])
    for hit,(score,id) in zip(hits,expected):
        assert hit['id']==id and abs(hit['score']-score)<2e-4,(turn,hit,score)
print('GPU exact top-K + float-score validation passed across multiple CUDA tiles')
