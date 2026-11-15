#!/usr/bin/env python3
"""Read existing engine gauges into an offline benchmark artifact; no telemetry writes."""
import argparse
import json
import time
from urllib.request import urlopen


NAMES = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
         'num_preemptions_total', 'prefix_cache_queries_total', 'prefix_cache_hits_total')


def sample(url, backend='vllm'):
    with urlopen(url.rstrip('/') + '/metrics', timeout=10) as response:
        raw = response.read().decode()
    if backend == 'strata':
        data = json.loads(raw)
        # Pinned Strata Service.metrics(): engine facts and numeric request
        # history, excluding hardware time series and API conversation records.
        return {'unix_s': time.time(), 'backend': backend,
                'metrics': {key: data.get(key) for key in
                            ('engine', 'live', 'requests', 'totals', 'conversation_cache')}}
    lines = raw.splitlines()
    return {'unix_s': time.time(), 'metrics': [line for line in lines
        if any(line.startswith('vllm:' + name + '{') for name in NAMES)
        or line.startswith('vllm:spec_decode_')]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', required=True)
    p.add_argument('--backend', choices=('vllm', 'strata'), default='vllm')
    p.add_argument('--seconds', type=float, default=10800)
    p.add_argument('--interval', type=float, default=2)
    args = p.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        p.error('duration and interval must be positive')
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            start = time.monotonic()
            try:
                row = sample(args.worker, args.backend)
            except Exception as exc:
                row = {'unix_s': time.time(), 'error': f'{type(exc).__name__}: {exc}'}
            print(json.dumps(row), flush=True)
            time.sleep(max(0, args.interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
