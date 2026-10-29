#!/usr/bin/env python3
"""Optional developer smoke test; the running service has no Python dependency."""
import json
import os
import urllib.request

base=os.environ.get('KNN_URL','http://127.0.0.1:8080')
def send(method,path,payload=None):
    body=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request(base+path,body,headers={'Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(request,timeout=20) as r:return json.load(r)

assert send('GET','/healthz')['status']=='ok'
assert send('POST','/v1/vectors',{'id':'smoke-a','vector':[1,0,0,0],'metadata':{'tag':'test'}})['count']>=1
send('POST','/v1/vectors/batch',{'vectors':[{'id':'smoke-b','vector':[.9,.1,0,0]},{'id':'smoke-c','vector':[0,0,1,0]}]})
r=send('POST','/v1/search',{'vector':[1,0,0,0],'k':3})
assert [h['id'] for h in r['results']][:2]==['smoke-a','smoke-b'],r
assert abs(r['results'][0]['score']-1)<1e-5,r
assert send('GET','/v1/vectors/smoke-a')['metadata']=={'tag':'test'}
r=send('POST','/v1/search/batch',{'queries':[{'vector':[1,0,0,0],'k':2},{'vector':[0,0,1,0],'k':1}]})
assert r['queries'][1]['results'][0]['id']=='smoke-c',r
assert send('DELETE','/v1/vectors/smoke-c')['deleted']
assert send('GET','/v1/stats')['count']>=2
print('HTTP smoke tests passed')
