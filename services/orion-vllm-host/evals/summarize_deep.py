#!/usr/bin/env python3
"""Join benchmark requests to raw host samples; emit reproducible measured summaries.

GPU energy integrates sampled board power, not wall-plug energy. CPU utilization
is whole-host /proc/stat busy time and includes unrelated services. Peaks are
sampled peaks, not CUDA allocator peaks. Missing coverage yields null energy.
"""
import argparse
import json
from pathlib import Path


def numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def power(row):
    values = [numeric(g.get('power_w')) for g in row.get('gpus', [])]
    return sum(values) if values and all(v is not None for v in values) else None


def integrate(samples, start, end, max_gap=6):
    """Trapezoids clipped to the request interval; require complete sampled coverage."""
    if not samples or end <= start or samples[0]['unix_s'] > start or samples[-1]['unix_s'] < end:
        return None
    joules = 0.0
    for a, b in zip(samples, samples[1:]):
        left, right = max(start, a['unix_s']), min(end, b['unix_s'])
        if left >= right:
            continue
        dt = b['unix_s'] - a['unix_s']
        pa, pb = power(a), power(b)
        if dt <= 0 or dt > max_gap or pa is None or pb is None:
            return None
        pleft = pa + (pb - pa) * (left - a['unix_s']) / dt
        pright = pa + (pb - pa) * (right - a['unix_s']) / dt
        joules += (pleft + pright) * 0.5 * (right - left)
    return joules


def cpu_busy(a, b):
    # guest/guest_nice are already in user/nice; only the first eight counters.
    av, bv = ([int(x) for x in row['proc_stat'].split()[1:9]] for row in (a, b))
    delta = [y - x for x, y in zip(av, bv)]
    total = sum(delta)
    return 100 * (total - delta[3] - delta[4]) / total if total > 0 and min(delta) >= 0 else None


def summarize(row, samples):
    keys = ('mode', 'suite', 'repeat', 'task_id', 'prompt_tokens_requested', 'output_budget',
            'elapsed_s', 'ttft_s', 'cached_tokens', 'stream_decode_tokens_per_s', 'failures', 'passed')
    out = {k: row[k] for k in keys if k in row}
    out['usage'] = row.get('usage')
    usage = row.get('usage') or {}
    timings = usage.get('timings') or {}
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    out['uncached_prompt_tokens_per_prefill_s'] = (
        (usage['prompt_tokens'] - cached) / timings['prefill_s']
        if cached is not None and timings.get('prefill_s') and 'prompt_tokens' in usage else None)
    start, end = row.get('started_at_epoch_s'), row.get('finished_at_epoch_s')
    if start is None or end is None:
        return out
    selected = [s for s in samples if start <= s['unix_s'] <= end]
    out['host_sample_count'] = len(selected)
    out['sampled_gpu_energy_j'] = integrate(samples, start, end)
    out['gpus'] = {}
    for sample in selected:
        for gpu in sample.get('gpus', []):
            dest = out['gpus'].setdefault(gpu['index'], {})
            for metric in ('memory_used_mib', 'utilization_percent', 'temperature_c', 'power_w'):
                value = numeric(gpu.get(metric))
                if value is not None:
                    dest['sampled_peak_' + metric] = max(dest.get('sampled_peak_' + metric, value), value)
    out['whole_host_cpu_busy_percent'] = cpu_busy(selected[0], selected[-1]) if len(selected) > 1 else None
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True, type=Path)
    p.add_argument('--results', required=True, type=Path, nargs='+')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    samples = sorted((json.loads(s) for s in args.host.read_text().splitlines()), key=lambda s: s['unix_s'])
    with args.out.open('x') as output:
        for path in args.results:
            for line in path.read_text().splitlines():
                output.write(json.dumps({'source': path.name, **summarize(json.loads(line), samples)}) + '\n')
