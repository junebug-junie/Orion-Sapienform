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
    obs = dict(status='observed', profile='zelda-1-ram-v1', mode='play', level=0, room=119, next_room=119,
               link=dict(x=120, y=141, tile=[15, 11], facing='N', invincible_frames=0, controllable=True),
               hearts=dict(current=3.0, containers=3), items=dict(sword=0),
               enemies=[dict(slot=2, type=7, x=48, y=96, tile=[6, 5], hp=1)],
               drops=[dict(slot=4, x=200, y=150)], walls=_grid(), walls_hash='abc', frame=900)
    d = dict(decision_id='3:1', port=0, expected_epoch=3, observation=obs, previous=None,
             supported_goals=['exit', 'enter', 'fight', 'retreat', 'collect', 'explore', 'wait'],
             supported_predictions=['reach_room', 'enter_cave', 'item_gained', 'enemies_cleared', 'survive', 'unknown'])
    d.update(over)
    return d


# Shape of Retro's GET /sessions/{id}/atlas (services/orion-retro/app/zelda/atlas.py neighbourhood()).
ATLAS = dict(lineage_id='lin-1', level=0, room=119,
             current=dict(visits=2, deaths=1, exits_seen=dict(N=103, S=None, E=120, W=None),
                          caves=[dict(tile=[8, 2], entered=True, contents_seen=[])], learned_blocked=[[1, 1]]),
             # Room 119 is on the bottom row: Retro leaves the off-grid S side out.
             adjacent=dict(N=dict(room=103, known=True, seen_exit=103),
                           E=dict(room=120, known=True, seen_exit=120), W=dict(room=118, known=False, seen_exit=None)),
             rooms_known=3, rooms_with_unvisited_sides=['0:119', '0:120'],
             caves=[dict(room='0:119', tile=[8, 2], entered=True)], deaths=[dict(room='0:119', deaths=1)],
             map={'0:119': dict(visits=2, exits=dict(N=103, E=120), caves=1, deaths=1)})
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
    assert map_rows[11][3 + 15] == 'L'                          # Retro's own link.tile
    assert map_rows[5][3 + 6] == '2'                            # enemy slot 2 at its tile
    assert map_rows[(150 + 11 - 64 + 4) // 8][3 + (200 + 4) // 8] == 'd'   # drop, Retro's tile convention
    assert map_rows[2][3 + 8] == 'C' and map_rows[1][3 + 1] == 'x'          # cave + learned-blocked
    assert '"walls"' not in prompt and 'true,false' not in prompt
    assert 'sides never crossed (may be walls) S,W' in prompt and 'N 0:103: known, reached from here before' in prompt
    assert 'with sides never crossed: 0:119, 0:120' in prompt and 'S 0:' not in prompt
    assert 'W 0:118: never visited' in prompt and 'death sites: 0:119 (1)' in prompt
    assert prompt.count('verdict met') == 5                    # last 5 decisions only
    assert 'Why a decision now: first decision of this session' in prompt


def test_map_without_walls_says_so_instead_of_inventing_floor(planner):
    d = decision(); d['observation'].pop('walls')
    text = planner.render_room_map(d['observation'], ATLAS)
    assert 'walls not included' in text and '#' not in text and 'L' in text
    d['observation']['mode'] = 'scroll'
    assert planner.render_room_map(d['observation'], ATLAS) == 'room map unavailable in mode scroll'


def test_history_keeps_each_finished_goal_once(planner):
    from collections import deque
    h = deque(maxlen=5)
    for row in [dict(decision_id='a'), dict(decision_id='a'), dict(decision_id='b')]:
        planner.remember_finished(h, dict(last_finished=row))
    assert [r['decision_id'] for r in h] == ['a', 'b']


@pytest.mark.parametrize('raw,reason', [
    ('not json at all', 'planner_unparseable'),
    (dict(GOOD, tool='shell'), 'planner_invalid_goal'),
    (dict(GOOD, goal=dict(kind='teleport')), 'planner_invalid_goal'),
    (dict(GOOD, goal=dict(kind='exit', dir='NE')), 'planner_invalid_goal'),
    (dict(GOOD, budget_seconds=90), 'planner_invalid_goal'),
    (dict(GOOD, rationale=''), 'planner_invalid_goal'),
    (dict(GOOD, goal=dict(kind='fight', target=5)), 'planner_unknown_target'),
    (dict(GOOD, goal=dict(kind='fight', target='nearest'), prediction=dict(kind='enemies_cleared')), None),
    (dict(GOOD, prediction=dict(kind='reach_room')), 'planner_prediction_value_mismatch'),
    (dict(GOOD, prediction=dict(kind='survive', value=3)), 'planner_prediction_value_mismatch'),
])
def test_bad_answers_are_rejected_and_never_become_a_goal(planner, raw, reason):
    chat = reply(dict(content=raw if isinstance(raw, str) else json.dumps(raw)))
    d = decision()
    if reason is None:   # fight with no enemy in the room (Retro: no_enemies_present)
        d['observation']['enemies'] = []; reason = 'planner_no_enemies'
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, d, ATLAS, HISTORY, chat=chat))
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
    (dict(decision(), observation=dict(decision()['observation'], mode='menu',
                                       menu=dict(kind='continue', cursor=0, activated=False))), 'observation_unknown'),
    (dict(decision(), observation=dict(decision()['observation'], mode='death')), 'observation_unknown'),
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
    assert goal['decision_id'] == '3:1' and goal['planner_id'] == 'llm-gateway:metacog_background:corr-1'
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
    assert asyncio.run(both())['decision_id'] == '3:1'
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
    c = dict(policy_id='zelda-navigator-v1', attempt=decision(), goal=None, last_finished=HISTORY[-1])
    c.update(cognition)
    return dict(generation='g' * 36, scripted=[dict(port=0, epoch=2, policy_id='zelda-navigator-v1', cognition=c)])


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
    assert url.endswith('/goal') and goal['connection_id'] == 'c' * 36 and goal['decision_id'] == '3:1'
    assert goal['expected_epoch'] == 2 and goal['planner_id'].endswith(':c9') and 'attempt_id' not in goal
    assert store.human >= 1


def test_route_rejection_and_unknown_observation_post_nothing(tmp_path, monkeypatch, planner):
    posted = []
    async def junk(config, prompt, *, max_tokens): return dict(content='{"goal":"walk_right"}'), 'c'
    monkeypatch.setattr(planner, '_gateway_chat', junk)
    with TestClient(_app(tmp_path, monkeypatch, _state(), posted)) as c:
        assert c.post(PATH, json=BODY, headers=H).status_code == 422
    unknown = _state(); unknown['scripted'][0]['cognition']['attempt']['observation'] = dict(status='unknown')
    with TestClient(_app(tmp_path, monkeypatch, unknown, posted)) as c:
        r = c.post(PATH, json=BODY, headers=H)
        assert r.status_code == 409 and r.json()['detail'] == 'observation_unknown'
    assert posted == []


def test_route_serves_atlas_and_leisure_status(tmp_path, monkeypatch, planner):
    import time
    store = FakeStore(); store.status_value = dict(state='playing', session_id='s', updated_at=time.time())
    with TestClient(_app(tmp_path, monkeypatch, _state(), [], store=store)) as c:
        assert c.get('/api/retro/sessions/' + 'a' * 36 + '/atlas').json()['lineage_id'] == 'lin-1'
        body = c.get('/api/retro/leisure').json()
        assert body['state'] == 'playing' and body['journal'][0]['reason'] == 'cabinet_hot'
        store.status_value['updated_at'] = time.time() - 60       # a runner that stopped reporting
        assert c.get('/api/retro/leisure').json()['state'] == 'stale'
        assert c.post('/api/retro/sessions/' + 'a' * 36 + '/atlas', headers=H).status_code == 404


def test_death_is_only_a_trigger(planner):
    d = decision(previous=dict(decision_id='3:0', goal=dict(kind='fight', target='nearest'),
                               outcome='death', verdict='refuted'))
    prompt = planner.build_zelda_prompt(d, ATLAS, [])
    assert 'fight nearest -> death' in prompt and 'Retro already chose Continue' in prompt
    assert 'Retro already chose Continue' not in planner.build_zelda_prompt(decision(), ATLAS, [])
