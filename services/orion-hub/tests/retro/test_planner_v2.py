"""Zelda planner v2 (spec 2026-10-22 section D): prompt, parse, reject, shed. No live GPU."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB = Path(__file__).resolve().parents[2]


def _grid():
    grid = [[False] * 32 for _ in range(22)]
    for c in range(32):
        grid[0][c] = grid[21][c] = True
    return grid


def decision(**over):
    obs = dict(status='observed', schema='zelda-1-ram-v1', mode='play', level=0, room=119, next_room=119,
               link=dict(x=120, y=141, facing=8, invincible_frames=0), hearts=dict(current=3.0, containers=3),
               items=dict(sword=0), enemies=[dict(slot=2, type=7, x=48, y=96, hp=1)],
               walls=dict(grid=_grid(), hash='abc'), frame=900)
    d = dict(decision_id='d-1', trigger='room_entry', observation=obs,
             supported_goals=['exit', 'enter', 'fight', 'retreat', 'collect', 'explore', 'wait'])
    d.update(over)
    return d


ATLAS = dict(lineage_id='lin-1', rooms={
    '0:119': dict(visits=2, deaths=1, exits_seen=dict(N=103, S=None, E=120, W=None),
                  caves=[dict(tile=dict(x=8, y=2), entered=True)], learned_blocked=[[1, 1]]),
    '0:103': dict(visits=1, deaths=0, exits_seen={}),
}, items_gained=[dict(item='sword', room='0:119')])
HISTORY = [dict(decision_id=f'd{i}', goal=dict(kind='exit', dir='E'), prediction=dict(kind='reach_room', value=120),
                outcome='prediction_met', verdict='met') for i in range(7)]
GOOD = dict(goal=dict(kind='exit', dir='N'), prediction=dict(kind='reach_room', value=103),
            rationale='North is explored; west is not. Try north to confirm the map.', budget_seconds=12)


@pytest.fixture
def planner(monkeypatch):
    from scripts import retro_planner
    from orion.gpu_pool.shed_backoff import ShedBackoff
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(retro_planner, 'SHED_BACKOFF',
                        ShedBackoff(name='test', clock=lambda: clock.t, rng=lambda: 0.5, log=lambda m: None))
    monkeypatch.setattr(retro_planner, '_in_flight', False)
    retro_planner.clock = clock
    return retro_planner


def reply(payload):
    calls = []
    async def chat(config, prompt, *, max_tokens):
        calls.append(prompt)
        return (payload(len(calls)) if callable(payload) else payload), 'corr-1'
    chat.calls = calls
    return chat


def test_prompt_has_room_map_atlas_neighbourhood_history_and_no_raw_grid(planner):
    prompt = planner.build_zelda_prompt(decision(), ATLAS, HISTORY)
    rows = prompt.split('\n')
    map_rows = [r for r in rows if r[:3].strip().isdigit() and len(r) == 35]
    assert len(map_rows) == 22 and map_rows[0][3:] == '#' * 32
    assert 'L' in map_rows[(141 - 64) // 8] and map_rows[(141 - 64) // 8][3 + 120 // 8] == 'L'
    assert map_rows[(96 - 64) // 8][3 + 48 // 8] == '2'        # enemy slot 2
    assert map_rows[2][3 + 8] == 'C'                           # known cave from the atlas
    assert '"walls"' not in prompt and 'true,false' not in prompt
    assert 'unexplored exits S,W' in prompt and 'N 0:103: visited 1x' in prompt and 'W 0:118: never visited' in prompt
    assert 'death sites: 0:119 (1)' in prompt and 'sword at 0:119' in prompt
    assert prompt.count('verdict met') == 5                    # last 5 decisions only
    assert 'Trigger for this decision: room_entry' in prompt


def test_neighbours_respect_the_16x8_grid(planner):
    assert planner.neighbours(0, 0) == {'S': 16, 'E': 1}
    assert planner.neighbours(0, 127) == {'N': 111, 'W': 126}


@pytest.mark.parametrize('raw,reason', [
    ('not json at all', 'planner_unparseable'),
    (dict(GOOD, tool='shell'), 'planner_invalid_goal'),
    (dict(GOOD, goal=dict(kind='teleport')), 'planner_invalid_goal'),
    (dict(GOOD, goal=dict(kind='exit', dir='NE')), 'planner_invalid_goal'),
    (dict(GOOD, budget_seconds=90), 'planner_invalid_goal'),
    (dict(GOOD, rationale=''), 'planner_invalid_goal'),
    (dict(GOOD, goal=dict(kind='fight', target=5)), 'planner_unknown_target'),
    (dict(GOOD, prediction=dict(kind='reach_room')), 'planner_unchecked_prediction'),
])
def test_bad_answers_are_rejected_and_never_become_a_goal(planner, raw, reason):
    chat = reply(dict(content=raw if isinstance(raw, str) else json.dumps(raw)))
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, HISTORY, chat=chat))
    assert exc.value.reason == reason


def test_unsupported_goal_for_this_decision_is_rejected(planner):
    chat = reply(dict(content=json.dumps(GOOD)))
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, decision(supported_goals=['wait', 'fight']), ATLAS, [], chat=chat))
    assert exc.value.reason == 'planner_unsupported_goal'


@pytest.mark.parametrize('d,reason', [
    (None, 'decision_pending'),
    (dict(decision_id='d', observation=dict(status='unknown', reason='fingerprint'), supported_goals=['wait']),
     'observation_unknown'),
    (dict(decision(), supported_goals=['jump_right']), 'decision_supported_goals_invalid'),
])
def test_unknown_observation_never_calls_the_model(planner, d, reason):
    chat = reply(dict(content=json.dumps(GOOD)))
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, d, ATLAS, [], chat=chat))
    assert exc.value.reason == reason and chat.calls == []


def test_good_answer_becomes_goal_v2_with_trace_id(planner):
    chat = reply(dict(content=json.dumps(GOOD)))
    goal = asyncio.run(planner.plan_zelda(None, decision(), ATLAS, HISTORY, chat=chat))
    assert goal['goal'] == {'kind': 'exit', 'dir': 'N'} and goal['prediction'] == {'kind': 'reach_room', 'value': 103}
    assert goal['decision_id'] == 'd-1' and goal['planner_id'] == 'llm-gateway:metacog_background:corr-1'
    assert goal['budget_seconds'] == 12 and len(chat.calls) == 1
    fight = dict(GOOD, goal=dict(kind='fight', target='nearest'), prediction=dict(kind='enemies_cleared'))
    goal = asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=reply(dict(content=json.dumps(fight)))))
    assert goal['goal'] == {'kind': 'fight', 'target': 'nearest'}


SHED = dict(content='', raw=dict(error='gpu_pool_unavailable', details=dict(reason='shed:cabinet_hot', retry_after_sec=45)))


def test_shed_is_surfaced_with_reason_and_retry_then_backs_off_without_calling(planner):
    chat = reply(SHED)
    with pytest.raises(planner.PlannerShed) as exc:
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=chat))
    assert exc.value.details['reason'] == 'shed:cabinet_hot' and exc.value.details['retry_after_sec'] >= 45
    assert exc.value.public()['detail'] == 'gpu_pool_unavailable'
    # Inside the window: refused locally, no RPC.
    with pytest.raises(planner.PlannerShed) as exc:
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=chat))
    assert exc.value.details.get('skipped_locally') and len(chat.calls) == 1
    # Window over: exactly one probe; a real answer ends the episode.
    planner.clock.t += 400
    ok = reply(dict(content=json.dumps(GOOD)))
    asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=ok))
    assert len(ok.calls) == 1 and planner.SHED_BACKOFF.reason('metacog_background') is None


def test_non_shed_pool_refusal_and_transport_error_are_unavailable(planner):
    refused = dict(content='', raw=dict(error='gpu_pool_unavailable', details=dict(reason='no_seat')))
    with pytest.raises(planner.PlannerUnavailable) as exc:
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=reply(refused)))
    assert exc.value.details['reason'] == 'no_seat'
    async def broken(config, prompt, *, max_tokens): raise TimeoutError()
    with pytest.raises(planner.PlannerUnavailable):
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=broken))
    with pytest.raises(planner.PlannerUnavailable) as exc:
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=reply(dict(content='', raw=dict(error='x')))))
    assert exc.value.reason == 'planner_empty_result'


def test_single_request_in_flight(planner):
    gate = asyncio.Event()
    async def slow(config, prompt, *, max_tokens):
        await gate.wait()
        return dict(content=json.dumps(GOOD)), 'c'
    async def both():
        first = asyncio.create_task(planner.plan_zelda(None, decision(), ATLAS, [], chat=slow))
        await asyncio.sleep(0)
        with pytest.raises(planner.PlannerBusy):
            await planner.plan_zelda(None, decision(), ATLAS, [], chat=slow)
        gate.set()
        return await first
    assert asyncio.run(both())['decision_id'] == 'd-1'
    assert planner._in_flight is False


# --- route: co-play v2 path --------------------------------------------------------------------

def _routes():
    spec = importlib.util.spec_from_file_location('plan_routes_v2', HUB / 'scripts/retro_routes.py')
    routes = importlib.util.module_from_spec(spec); spec.loader.exec_module(routes)
    return routes


class FakeStore:
    def __init__(self): self.human = 0; self.beats = set(); self.dropped = set(); self.status_value = {}
    async def mark_human(self): self.human += 1
    async def beat_browser(self, sid): self.beats.add(sid)
    async def drop_browser(self, sid): self.dropped.add(sid)
    async def status(self): return self.status_value
    async def journal(self, n): return [dict(event='refused', reason='cabinet_hot')]


def _app(tmp_path, monkeypatch, state, posted, atlas=ATLAS, store=None):
    routes = _routes()
    (tmp_path / 'hub.token').write_text('x' * 48)
    cfg = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_COGNITION_ENABLED=True,
        HUB_RETRO_ORIGIN='http://testserver', HUB_RETRO_AUTH_DIR=str(tmp_path), HUB_RETRO_SOCKET='/unused')
    class Response:
        def __init__(self, data, status=200): self.data = data; self.status = status
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def json(self): return self.data
    class Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        def get(self, url, **kw): return Response(atlas if url.endswith('/atlas') else state)
        def request(self, method, url, **kw): return self.get(url)
        def post(self, url, **kw):
            posted.append((url, kw['json'])); return Response(kw['json'])
    monkeypatch.setattr(routes.aiohttp, 'ClientSession', Client)
    app = FastAPI(); app.include_router(routes.create_router(cfg, presence=store or FakeStore()))
    return app


BODY = dict(connection_id='c' * 36, generation='g' * 36, port=0, expected_epoch=2)
PATH = '/api/retro/sessions/' + 'a' * 36 + '/plan'
H = {'Origin': 'http://testserver'}


def _state(**cognition):
    return dict(generation='g' * 36, scripted=[dict(port=0, epoch=2, cognition=dict(decision=decision(), goal=None,
                                                                                     history=HISTORY, **cognition))])


def test_route_shed_returns_reason_and_retry_and_posts_no_goal(tmp_path, monkeypatch, planner):
    posted = []
    async def shed(config, prompt, *, max_tokens): return SHED, 'c'
    monkeypatch.setattr(planner, '_gateway_chat', shed)
    with TestClient(_app(tmp_path, monkeypatch, _state(), posted)) as c:
        r = c.post(PATH, json=BODY, headers=H)
    assert r.status_code == 503 and r.json()['detail'] == 'gpu_pool_unavailable'
    assert r.json()['reason'] == 'shed:cabinet_hot' and r.json()['retry_after_sec'] >= 45
    assert int(r.headers['Retry-After']) >= 45 and posted == []


def test_route_posts_goal_v2_with_browser_sponsor_and_marks_human(tmp_path, monkeypatch, planner):
    posted = []; store = FakeStore()
    async def good(config, prompt, *, max_tokens): return dict(content=json.dumps(GOOD)), 'c9'
    monkeypatch.setattr(planner, '_gateway_chat', good)
    with TestClient(_app(tmp_path, monkeypatch, _state(), posted, store=store)) as c:
        r = c.post(PATH, json=BODY, headers=H)
        assert r.status_code == 200
    url, goal = posted[0]
    assert url.endswith('/goal') and goal['connection_id'] == 'c' * 36 and goal['decision_id'] == 'd-1'
    assert goal['expected_epoch'] == 2 and goal['planner_id'].endswith(':c9') and 'attempt_id' not in goal
    assert store.human >= 1


def test_route_rejection_and_unknown_observation_post_nothing(tmp_path, monkeypatch, planner):
    posted = []
    async def junk(config, prompt, *, max_tokens): return dict(content='{"goal":"walk_right"}'), 'c'
    monkeypatch.setattr(planner, '_gateway_chat', junk)
    with TestClient(_app(tmp_path, monkeypatch, _state(), posted)) as c:
        assert c.post(PATH, json=BODY, headers=H).status_code == 422
    unknown = _state(); unknown['scripted'][0]['cognition']['decision']['observation'] = dict(status='unknown')
    with TestClient(_app(tmp_path, monkeypatch, unknown, posted)) as c:
        r = c.post(PATH, json=BODY, headers=H)
        assert r.status_code == 409 and r.json()['detail'] == 'observation_unknown'
    assert posted == []


def test_route_serves_atlas_and_leisure_status(tmp_path, monkeypatch, planner):
    store = FakeStore(); store.status_value = dict(state='playing', session_id='s')
    with TestClient(_app(tmp_path, monkeypatch, _state(), [], store=store)) as c:
        assert c.get('/api/retro/sessions/' + 'a' * 36 + '/atlas').json()['lineage_id'] == 'lin-1'
        body = c.get('/api/retro/leisure').json()
        assert body['state'] == 'playing' and body['journal'][0]['reason'] == 'cabinet_hot'
        assert c.post('/api/retro/sessions/' + 'a' * 36 + '/atlas', headers=H).status_code == 404
