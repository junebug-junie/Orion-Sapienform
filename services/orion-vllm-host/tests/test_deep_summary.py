from pathlib import Path
import runpy

import pytest


MOD = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'evals/summarize_deep.py'))


def sample(t, watts):
    return {'unix_s': t, 'gpus': [{'power_w': str(watts)}]}


def test_energy_clips_both_edges_and_integrates_linear_power():
    samples = [sample(0, 0), sample(2, 20), sample(4, 40)]
    assert MOD['integrate'](samples, 1, 3) == pytest.approx(40)


def test_missing_coverage_gaps_and_unreadable_power_are_not_zero_energy():
    assert MOD['integrate']([sample(1, 10), sample(3, 10)], 0, 3) is None
    assert MOD['integrate']([sample(0, 10), sample(20, 10)], 1, 2) is None
    assert MOD['integrate']([sample(0, 'N/A'), sample(2, 10)], 0, 2) is None


def test_cpu_guest_is_not_counted_twice_and_iowait_is_idle():
    a = {'proc_stat': 'cpu 0 0 0 0 0 0 0 0 0 0'}
    b = {'proc_stat': 'cpu 20 0 10 60 10 0 0 0 20 0'}
    assert MOD['cpu_busy'](a, b) == 30


def test_summary_distinguishes_cached_tokens_and_incomplete_request():
    out = MOD['summarize']({'mode': 'fresh_prefix', 'usage': {'prompt_tokens': 100,
        'prompt_tokens_details': {'cached_tokens': 20}, 'timings': {'prefill_s': 2}}}, [])
    assert out['uncached_prompt_tokens_per_prefill_s'] == 40
    assert 'sampled_gpu_energy_j' not in out
