#!/usr/bin/env python3
"""Exercise private Hecate admission, disconnects, refusals and recovery via Gateway.

Requires commissioning ownership and an idle lane. Only GET /metrics bypasses
Gateway. This deliberately does not provoke a CUDA OOM on a live worker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid

import httpx

from deep_context import build_messages, judge, stream_chat
from orion.evals.leased_llm import operator_lane_sync


def metrics(url):
    response = httpx.get(url.rstrip('/') + '/metrics', timeout=10)
    response.raise_for_status()
    names = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc', 'num_preemptions_total')
    values = {}
    for line in response.text.splitlines():
        for name in names:
            if line.startswith('vllm:' + name + '{'):
                values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[1])
    if 'num_requests_running' not in values or 'num_requests_waiting' not in values:
        raise ValueError('required engine metrics unavailable')
    return values


def wait_idle(url, timeout=30):
    start = time.monotonic()
    samples = []
    while time.monotonic() - start < timeout:
        row = metrics(url)
        samples.append({'elapsed_s': time.monotonic() - start, **row})
        if row['num_requests_running'] == row['num_requests_waiting'] == 0:
            return samples
        time.sleep(0.5)
    raise TimeoutError(f'engine still active after disconnect: {samples[-1]}')


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--gateway', required=True)
    p.add_argument('--worker', required=True, help='read-only engine metrics URL')
    p.add_argument('--pool', required=True, help='read-only pool state URL')
    p.add_argument('--model', default='deep.thinker.v1')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    run = uuid.uuid4().hex[:12]
    seed = int(run, 16)
    prompts = {n: build_messages(tok, n, seed + n) for n in (8192, 32768, 132000, 245761)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        def record(row):
            out.write(json.dumps(row) + '\n'); out.flush()
            print(json.dumps({k: row[k] for k in ('test', 'passed') if k in row}), flush=True)

        def call(lane, n):
            messages, expected = prompts[n]
            result = stream_chat(lane, {'messages': messages, 'max_tokens': 16384,
                'temperature': 0, 'chat_template_kwargs': {'enable_thinking': True}}, 1800)
            result['failures'] = judge(result, expected=expected, model=args.model,
                                       prompt_tokens=n, output_budget=16384)
            return result

        try:
            # A second hold must remain queued while the first owns both cards.
            with ThreadPoolExecutor(max_workers=1) as executor:
                with operator_lane_sync('agent-deep', holder=f'operator:deep-ops-first-{run}',
                                        gateway=args.gateway, queue_wait_sec=900) as first:
                    wait_idle(args.worker)
                    attempted = threading.Event()
                    second_holder = f'operator:deep-ops-second-{run}'
                    def second_call():
                        attempted.set()
                        requested = time.time()
                        with operator_lane_sync('agent-deep', holder=second_holder,
                                                gateway=args.gateway, queue_wait_sec=900) as second:
                            granted = time.time()
                            result = call(second, 8192)
                            return {'requested_at': requested, 'granted_at': granted,
                                    'seat': second.summary(), 'result': result}
                    future = executor.submit(second_call)
                    assert attempted.wait(10), 'second caller did not start'
                    queued = []
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        response = httpx.get(args.pool.rstrip('/') + '/v1/pool', timeout=10)
                        response.raise_for_status()
                        queued = [r for r in response.json()['leases'] if r['holder'] == second_holder]
                        if any(r['status'] in ('queued', 'backlogged') for r in queued):
                            break
                        if future.done():
                            raise AssertionError('second hold granted while first still held the lane')
                        time.sleep(0.5)
                    assert any(r['status'] in ('queued', 'backlogged') for r in queued), 'queue not observed'
                    first_result = call(first, 132000)
                    release_started = time.time()
                second_result = future.result(timeout=900)
            passed = (not first_result['failures'] and not second_result['result']['failures']
                      and second_result['granted_at'] >= release_started)
            record({'test': 'serialized_queue', 'passed': passed, 'queue_observed': queued,
                    'first_result': first_result, 'release_started_at': release_started, 'second': second_result})
            assert passed, 'queue or answer validation failed'

            with operator_lane_sync('agent-deep', holder=f'operator:deep-ops-failures-{run}',
                                    gateway=args.gateway, queue_wait_sec=900) as lane:
                for test, extra, n in [('over_budget', {}, 245761), ('multiple_completions', {'n': 2}, 8192)]:
                    response = httpx.post(args.gateway.rstrip('/') + '/v1/chat/completions',
                        headers=lane.headers(), json={'model': lane.route, 'messages': prompts[n][0],
                        'max_tokens': 16384, **extra}, timeout=60)
                    record({'test': test, 'passed': response.status_code == 400,
                            'http_status': response.status_code, 'error': response.text[:1000]})
                    assert response.status_code == 400, 'unsafe request was not refused'
                # Close a real active stream after a token, while holding the seat.
                saw_token = False
                with httpx.stream('POST', args.gateway.rstrip('/') + '/v1/chat/completions',
                    headers=lane.headers(), json={'model': lane.route, 'messages': prompts[32768][0],
                    'stream': True, 'max_tokens': 16384, 'min_tokens': 16384, 'ignore_eos': True}, timeout=180) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            chunk = json.loads(line[6:])
                            if any((c.get('delta', {}).get('content') or c.get('delta', {}).get('reasoning')
                                    or c.get('delta', {}).get('reasoning_content')) for c in chunk.get('choices', [])):
                                saw_token = True
                                break
                samples = wait_idle(args.worker)
                recovery = call(lane, 8192)
                passed = saw_token and not recovery['failures']
                record({'test': 'cancel_and_recover', 'passed': passed, 'idle_samples': samples, 'recovery': recovery})
                assert passed, 'cancellation recovery failed'
                # A fresh prefix prevents the timeout case being an exact replay.
                messages, _ = build_messages(tok, 32768, seed + 999)
                def timeout_call():
                    try:
                        httpx.post(args.gateway.rstrip('/') + '/v1/chat/completions', headers=lane.headers(),
                            json={'model': lane.route, 'messages': messages, 'max_tokens': 16384,
                                  'min_tokens': 16384, 'ignore_eos': True},
                            timeout=httpx.Timeout(30, read=5))
                    except httpx.ReadTimeout:
                        return True
                    return False
                active_observed = False
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(timeout_call)
                    while not future.done():
                        active_observed |= metrics(args.worker)['num_requests_running'] > 0
                        time.sleep(0.1)
                    timed_out = future.result()
                samples = wait_idle(args.worker)
                recovery = call(lane, 8192)
                passed = timed_out and active_observed and not recovery['failures']
                record({'test': 'client_timeout_and_recover', 'passed': passed,
                        'active_observed': active_observed, 'idle_samples': samples, 'recovery': recovery})
                assert passed, 'timeout recovery failed'
        except BaseException as exc:
            record({'test': 'run', 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__':
    main()
