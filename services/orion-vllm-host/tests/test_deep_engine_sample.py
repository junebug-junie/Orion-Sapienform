import io
import json
from pathlib import Path
import runpy


SAMPLE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'evals/sample_engine.py'))['sample']


def test_strata_sample_keeps_cache_facts_without_hardware_history(monkeypatch):
    payload = {'engine': {'max_context': 262144}, 'live': {'state': 'idle'},
               'requests': [{'hit_rate': 0.8, 'pcie_share': None}],
               'totals': {'drafts_accepted': 7}, 'conversation_cache': {'enabled': False},
               'history': {'gpu0': [1, 2, 3]}, 'hardware': {'gpu': 'unused'}}
    monkeypatch.setitem(SAMPLE.__globals__, 'urlopen',
                        lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    result = SAMPLE('http://worker', 'strata')
    assert result['metrics']['live']['state'] == 'idle'
    assert result['metrics']['requests'][0]['pcie_share'] is None
    assert result['metrics']['totals']['drafts_accepted'] == 7
    assert 'history' not in result['metrics']
    assert 'hardware' not in result['metrics']


def test_vllm_sample_keeps_speculation_counters_and_existing_gauges(monkeypatch):
    payload = ('# HELP vllm:spec_decode_num_drafts Drafts\n'
               'vllm:spec_decode_num_drafts_total{model="deep"} 3\n'
               'vllm:num_requests_running{model="deep"} 0\n'
               'unrelated_metric 12\n')
    monkeypatch.setitem(SAMPLE.__globals__, 'urlopen', lambda *a, **k: io.BytesIO(payload.encode()))
    assert SAMPLE('http://worker')['metrics'] == payload.splitlines()[1:3]
