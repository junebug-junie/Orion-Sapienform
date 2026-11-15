#!/usr/bin/env python3
"""Sidecar measurement to JSONL; no cognitive telemetry or production writes.

Run on Hecate while a leased benchmark runs. Stop with SIGINT. GPU energy is a
sampled estimate, not a wall-plug reading. Bandwidth and NVLink byte counters are
explicitly unavailable unless a separate supported counter tool is used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def sample():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    gpus = []
    if p.returncode == 0:
        for line in p.stdout.splitlines():
            values = [v.strip() for v in line.split(',')]
            gpus.append(dict(zip(('index', 'memory_used_mib', 'utilization_percent', 'power_w', 'temperature_c'), values)))
    return {'unix_s': time.time(), 'monotonic_s': time.monotonic(), 'gpus': gpus,
            'gpu_query_error': p.stderr if p.returncode else None,
            'proc_stat': Path('/proc/stat').read_text().splitlines()[0],
            'meminfo': Path('/proc/meminfo').read_text(),
            'host_memory_bandwidth_bytes_s': None, 'nvlink_bytes_s': None,
            'kv_residency': None, 'unavailable': ['host_memory_bandwidth', 'nvlink_traffic', 'kv_residency']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--interval', type=float, default=1)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error('duration and interval must be positive')
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            start = time.monotonic()
            print(json.dumps(sample()), flush=True)
            time.sleep(max(0, args.interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
