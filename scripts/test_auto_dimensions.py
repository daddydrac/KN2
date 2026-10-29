#!/usr/bin/env python3
"""Integration test for first-insert dimension inference and WAL persistence.

Requires CPU reference executable or GPU service binary at --server. Runs a fresh
isolated process twice using temporary WAL; not safe against an already-in-use port.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

parser = argparse.ArgumentParser()
parser.add_argument('--server', default='./build/knn_server')
parser.add_argument('--port', type=int, default=18119)
args = parser.parse_args()
root = f'http://127.0.0.1:{args.port}'

def query(method, path, payload=None, expected=200):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(root + path, data=data, headers={'Content-Type':'application/json'} if data else {}, method=method)
    try:
        with urlopen(req, timeout=10) as res:
            assert res.status == expected, (res.status, expected)
            return json.load(res)
    except HTTPError as e:
        assert e.code == expected, (e.code, expected)
        return json.load(e)

with tempfile.TemporaryDirectory() as tmp:
    env = {**os.environ, 'KNN_DIMENSIONS':'0', 'KNN_CAPACITY':'100',
           'KNN_PORT':str(args.port), 'KNN_WAL_PATH':str(Path(tmp) / 'test.wal')}
    def start():
        process = subprocess.Popen([args.server], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if process.poll() is not None: raise RuntimeError('server exited')
            try:
                query('GET', '/healthz')
                return process
            except Exception:
                time.sleep(0.1)
        raise RuntimeError('server did not start')

    proc = start()
    try:
        assert query('GET','/v1/stats')['dimensions'] == 0
        query('POST','/v1/search', {'vector':[1,0,0,0,0,0], 'k':1}, 400)
        query('POST','/v1/vectors', {'id':'first', 'vector':[1,0,0,0,0,0]})
        assert query('GET','/v1/stats')['dimensions'] == 6
        query('POST','/v1/vectors', {'id':'wrong','vector':[1,0,0,0]}, 400)
        assert query('POST','/v1/search', {'vector':[1,0,0,0,0,0], 'k':1})['results'][0]['id'] == 'first'
    finally:
        proc.terminate(); proc.wait(timeout=5)
    proc = start()
    try:
        stats = query('GET','/v1/stats')
        assert stats['dimensions'] == 6 and stats['count'] == 1, stats
        assert query('POST','/v1/search', {'vector':[1,0,0,0,0,0], 'k':1})['results'][0]['id'] == 'first'
        print('Auto-dimension discovery, mismatch rejection and WAL replay passed')
    finally:
        proc.terminate(); proc.wait(timeout=5)
