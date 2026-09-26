"""Operator reads cannot consume offers or turn unavailable evidence into zero."""
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import dream_routes as routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


@pytest.mark.parametrize("enabled,due,soon,ready", [
    (True, True, False, True), (True, True, True, False),
    (False, True, False, False), (True, False, False, False),
])
def test_readiness_honors_all_gates(client, monkeypatch, enabled, due, soon, ready):
    payload = {"enabled": enabled, "should_sleep": due, "is_idle": due,
               "too_soon": soon, "candidates": 3, "pressure": {
                   "since": "2026-09-26T00:00:00Z", "computed_at": "2026-09-26T01:00:00Z",
                   "pressure": 3, "threshold": 3, "idle_minutes": 50 if due else None,
                   "idle_required_minutes": 45, "counts": {"metacog": 3}}}
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url):
            assert url.endswith('/dreams/cycle/pressure')
            return httpx.Response(200, json=payload, request=httpx.Request('GET', url))
    monkeypatch.setattr(routes.httpx, "AsyncClient", Client)
    result = client.get('/api/dream/pressure')
    assert result.status_code == 200
    assert result.json()['ready'] is ready
    assert result.headers['cache-control'] == 'no-store'


def test_upstream_failure_is_unavailable(client, monkeypatch):
    class Broken:
        def __init__(self, **kwargs): raise RuntimeError('private connection details')
    monkeypatch.setattr(routes.httpx, 'AsyncClient', Broken)
    result = client.get('/api/dream/pressure')
    assert result.status_code == 503
    assert 'private' not in result.text


def test_cursor_uses_time_and_id_and_bound_params(client, monkeypatch):
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    seen = []
    def rows(sql, params):
        seen.append((sql, params))
        return [{'cycle_id': cid, 'started_at': now} for cid in ('dc-c', 'dc-b', 'dc-a')]
    monkeypatch.setattr(routes, '_rows', rows)
    result = client.get('/api/dream/cycles?limit=2&before=2026-09-26T00:00:00Z&before_id=dc-z').json()
    assert result['has_more']
    assert result['next_cursor']['before_id'] == 'dc-b'
    assert [r['cycle_id'] for r in result['cycles']] == ['dc-c', 'dc-b']
    assert '(started_at, cycle_id) < (:before, :before_id)' in seen[0][0]
    assert seen[0][1]['limit'] == 3
    assert client.get('/api/dream/cycles?limit=1000').status_code == 422
    assert client.get('/api/dream/cycles?before_id=dc-z').status_code == 422


def test_detail_reads_current_offer_state_without_claiming(client, monkeypatch):
    queries = []
    def rows(sql, params):
        queries.append(sql)
        if 'cycle_json' in sql:
            return [{'cycle_json': {'cycle_id': 'dc-a', 'replay': []}}]
        return [{'hypothesis_id': 'dh-a', 'offered_run_id': 'run-1', 'arm': 'control'}]
    monkeypatch.setattr(routes, '_rows', rows)
    data = client.get('/api/dream/cycles/dc-a').json()
    assert data['hypotheses'][0]['offered_run_id'] == 'run-1'
    assert all(q.startswith('SELECT ') for q in queries)
    assert client.post('/api/dream/cycles/dc-a').status_code == 405
    monkeypatch.setattr(routes, '_rows', lambda *args: [])
    assert client.get('/api/dream/cycles/dc-missing').status_code == 404


def test_scorecard_reuses_scoring_and_refuses_unavailable_graph(client, monkeypatch):
    monkeypatch.setattr(routes, '_rows', lambda *args: [{'hypothesis_id': 'dh-a', 'arm': 'dream'}])
    monkeypatch.setattr(routes, '_prior_rows', lambda: [
        {'formed_from': 'dream_hypothesis:dh-a', 'status': 'supported', 'times_tested': 1}])
    data = client.get('/api/dream/scorecard').json()
    assert data['arms']['dream']['supported'] == 1
    assert data['arms']['control']['support_rate'] is None
    assert data['verdict'].startswith('too early')
    def broken(): raise RuntimeError('secret')
    monkeypatch.setattr(routes, '_prior_rows', broken)
    result = client.get('/api/dream/scorecard')
    assert result.status_code == 503
    assert 'secret' not in result.text


def test_truncated_score_is_never_presented_as_complete(client, monkeypatch):
    monkeypatch.setattr(routes, 'SCORE_LIMIT', 1)
    monkeypatch.setattr(routes, '_rows', lambda *args: [{}, {}])
    monkeypatch.setattr(routes, '_prior_rows', lambda: [])
    assert client.get('/api/dream/scorecard').json()['detail'] == 'dream_scorecard_limit_exceeded'


def test_db_failure_does_not_report_empty_history(client, monkeypatch):
    def broken(): raise RuntimeError('secret')
    monkeypatch.setattr(routes, '_engine', broken)
    result = client.get('/api/dream/cycles')
    assert result.status_code == 503
    assert 'secret' not in result.text


def test_template_asset_and_router_are_wired():
    from jinja2 import Environment, FileSystemLoader
    root = Path(__file__).resolve().parents[1]
    rendered = Environment(loader=FileSystemLoader(root / 'templates')).get_template('index.html').render(HUB_UI_ASSET_VERSION='dream-test')
    assert 'id="dreamTabButton" href="#dream"' in rendered
    assert '/static/js/dream-tab.js?v=dream-test' in rendered
    assert 'router.include_router(dream_router)' in (root / 'scripts/api_routes.py').read_text()
    assert 'window.OrionDream?.activate()' in (root / 'static/js/app.js').read_text()
