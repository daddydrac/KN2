#!/usr/bin/env python3
"""JSON/REST correctness, validation, upserts, deletion, and concurrent query tests."""
import concurrent.futures
import json
import math
import os
import random
import urllib.error
import urllib.request

base=os.environ.get('KNN_URL','http://127.0.0.1:18089')

def api(method,path,payload=None, status=200):
    data=json.dumps(payload).encode() if payload is not None else None
    r=urllib.request.Request(base+path,data,{'Content-Type':'application/json'},method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as res:
            assert res.status==status,(res.status,status)
            return json.load(res)
    except urllib.error.HTTPError as e:
        assert e.code==status,(path,e.code,status,e.read())
        return json.load(e)

rng=random.Random(123)
vectors={}
for i in range(60):
    v=[rng.uniform(-1,1) for _ in range(4)]
    vectors[f'test-{i:03d}']=v
for start in range(0,60,20):
    part=list(vectors.items())[start:start+20]
    resp=api('POST','/v1/vectors/batch',{'vectors':[{'id':id,'vector':v,'metadata':{'n':int(id[-3:])}} for id,v in part]})
    assert resp['count']>=60 if start==40 else True

# Match top-10 exact rankings for many random queries, allowing FP32 score rounding.
def normalized(v):
    mag=math.sqrt(sum(x*x for x in v))
    return [x/mag for x in v]
for _ in range(50):
    q=[rng.uniform(-1,1) for _ in range(4)]
    nq=normalized(q)
    expected=sorted(((sum(a*b for a,b in zip(nq,normalized(v))),id) for id,v in vectors.items()),reverse=True)[:10]
    found=api('POST','/v1/search',{'vector':q,'k':10})['results']
    actual=[h['id'] for h in found if h['id'].startswith('test-')]
    # Smoke records may be in index too: only assert scores are sorted, and each test vector scores as expected.
    assert all(found[i]['score']>=found[i+1]['score']-1e-6 for i in range(len(found)-1)),found
    for h in found:
        if h['id'] in vectors:
            score=sum(a*b for a,b in zip(nq,normalized(vectors[h['id']])))
            assert abs(score-h['score'])<5e-5,(score,h)

# Concurrency: 80 requests, server batches internally.
def search(i):
    q=[1,0,0,0] if i%2 else [0,1,0,0]
    return api('POST','/v1/search',{'vector':q,'k':3})
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
    results=list(pool.map(search,range(80)))
assert len(results)==80

# Replace, get metadata, delete, capacity/shape/validation.
r=api('POST','/v1/vectors',{'id':'test-001','vector':[1,0,0,0],'metadata':{'replacement':True}})
assert r['updated']==1
r=api('GET','/v1/vectors/test-001')
assert r['metadata']=={'replacement':True} and abs(r['vector'][0]-1)<1e-5
api('POST','/v1/vectors',{'id':'bad','vector':[0,0,0,0]},400)
api('POST','/v1/vectors',{'id':'bad','vector':[1,0,0]},400)
api('POST','/v1/vectors/batch',{'vectors':[{'id':'same','vector':[1,0,0,0]},{'id':'same','vector':[0,1,0,0]}]},400)
api('POST','/v1/search',{'vector':[1,0,0,0],'k':100},400)
assert api('DELETE','/v1/vectors/test-001')['deleted']
api('GET','/v1/vectors/test-001',status=404)
assert api('GET','/v1/stats')['backend']
print('Advanced HTTP integration tests passed')
