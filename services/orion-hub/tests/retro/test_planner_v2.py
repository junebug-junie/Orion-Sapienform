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


def test_retro_terrain_is_the_map_and_entities_overlay_it(planner):
    d = decision(); obs = d['observation']; obs.pop('walls')
    terrain = ['#' * 32] + ['#' + '.' * 30 + '#' for _ in range(20)] + ['#' * 32]
    terrain[2] = terrain[2][:8] + 'C' + terrain[2][9:]; terrain[3] = terrain[3][:4] + 'x' + terrain[3][5:]
    obs['terrain'] = terrain
    prompt = planner.build_zelda_prompt(d, None, [])
    rows = [r for r in prompt.split('\n') if r[:3].strip().isdigit() and len(r) == 35]
    assert rows[0][3:] == '#' * 32 and rows[2][3 + 8] == 'C' and rows[3][3 + 4] == 'x'
    assert rows[11][3 + 15] == 'L' and rows[5][3 + 6] == '2' and 'walls not included' not in prompt
    assert '"terrain"' not in prompt                           # sent once, as the map
    assert '# blocked, . walkable, x blocked' in prompt and 'C a cave you entered' in prompt
    obs['terrain'] = ['bad']
    assert 'walls not included' in planner.render_room_map(obs, None)   # malformed: say unknown



# --- co-play session end -> app.state.retro_session_end_sink ----------------------------------

def _coplay_app(tmp_path, monkeypatch, frames, sink=True):
    import time as _t
    routes = _routes()
    monkeypatch.setattr(routes, 'COPLAY_POLL_SEC', 0.01)
    (tmp_path / 'hub.token').write_text('x' * 48)
    cfg = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_COGNITION_ENABLED=True,
        HUB_RETRO_ORIGIN='http://testserver', HUB_RETRO_AUTH_DIR=str(tmp_path), HUB_RETRO_SOCKET='/unused')
    world = SimpleNamespace(polls=0, ended=[])
    def status():
        world.polls += 1
        policy = dict(port=0, epoch=7, policy_id='zelda-navigator-v1', cognition=dict(
            policy_id='zelda-navigator-v1', attempt=None,
            goal=dict(decision_id=f'7:{world.polls}', goal=dict(kind='exit', dir='N')),
            last_finished=dict(decision_id=f'7:{world.polls - 1}', verdict='met' if world.polls % 2 else 'refuted',
                               outcome='prediction_met')))
        # Retro's record is session-wide and grows each poll: the handoff must report the delta.
        rec = dict(rooms_new=10 + 2 * world.polls, deaths=world.polls // 3)
        if frames is not None: rec['frames_by_controller'] = frames(world.polls)
        return dict(session_id='a' * 36, state='running', game_id='the-legend-of-zelda', loaded_save=None,
                    scripted=[policy] if world.polls < 4 else [],
                    zelda=dict(atlas=dict(lineage_id='lin-c', session=rec)))
    class Response:
        def __init__(self, data, status=200): self.data = data; self.status = status
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def json(self): return self.data
    class Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        def get(self, url, **kw): return Response(status())
        def request(self, method, url, **kw):
            return Response({'scripted': [dict(port=0, epoch=7, policy_id='zelda-navigator-v1')]})
    monkeypatch.setattr(routes.aiohttp, 'ClientSession', Client)
    app = FastAPI(); app.include_router(routes.create_router(cfg, presence=FakeStore()))
    if sink:
        async def collect(ended): world.ended.append(ended)
        app.state.retro_session_end_sink = collect
    return app, world


HANDOFF = dict(connection_id='c' * 36, generation='g' * 36, port=0, expected_epoch=6, action='start',
               duration_seconds=60, cognition=True)


def _wait(cond):
    import time as _t
    for _ in range(300):
        if cond(): return True
        _t.sleep(0.01)
    return False


def test_coplay_end_reaches_the_sink_with_atlas_record(tmp_path, monkeypatch):
    # The human played 500 frames before the handoff and 40 more during it.
    app, world = _coplay_app(tmp_path, monkeypatch, frames=lambda n: dict(
        human=500 + (40 if n >= 3 else 0), script=100 * n, leisure=0, none=0))
    with TestClient(app) as c:
        assert c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=HANDOFF, headers=H).status_code == 200
        assert _wait(lambda: world.ended)
    e = world.ended[0]
    assert (e.session_id, e.leisure_id, e.controller, e.lineage_id) == ('a' * 36, None, 'co_play', 'lin-c')
    # First poll: rooms 12, deaths 0; last poll (policy gone, poll 4): rooms 18, deaths 1.
    assert e.rooms_new == 6 and e.deaths == 1
    assert e.controller_frames == {'human': 40, 'script': 300, 'leisure': 0, 'none': 0}
    assert e.human_may_be_in_control is True and e.decisions_planned == 3 and sum(e.verdicts.values()) == 3


def test_coplay_unknown_counts_are_none_and_read_as_maybe_human(tmp_path, monkeypatch):
    app, world = _coplay_app(tmp_path, monkeypatch, frames=None)
    with TestClient(app) as c:
        c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=HANDOFF, headers=H)
        assert _wait(lambda: world.ended)
    e = world.ended[0]
    assert e.controller_frames is None and e.human_may_be_in_control is True


def test_coplay_orion_only_frames_and_no_sink(tmp_path, monkeypatch):
    # 500 human frames BEFORE the handoff do not make this handoff "human".
    app, world = _coplay_app(tmp_path, monkeypatch, frames=lambda n: dict(human=500, script=100 * n, leisure=0, none=0))
    with TestClient(app) as c:
        c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=HANDOFF, headers=H)
        assert _wait(lambda: world.ended)
    assert world.ended[0].human_may_be_in_control is False
    app2, world2 = _coplay_app(tmp_path, monkeypatch, frames=None, sink=False)
    with TestClient(app2) as c:
        c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=HANDOFF, headers=H)
        assert _wait(lambda: world2.polls >= 4)   # watched to the end; no sink -> nothing written
    assert world2.ended == []
    # A non-cognition (scripted baseline) handoff is not watched at all.
    app3, world3 = _coplay_app(tmp_path, monkeypatch, frames=None)
    with TestClient(app3) as c:
        c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=dict(HANDOFF, cognition=False), headers=H)
        import time as _t; _t.sleep(0.1)
    assert world3.polls == 0



def test_coplay_handoff_opens_its_play_act_once_before_the_end(tmp_path, monkeypatch):
    # Regression (2026-10-23): co-play opened no play act, and it is the path every played session
    # took, so grammar_acts held 0 retro acts while sessions ran. The watcher now calls the start
    # sink once, at the first Zelda cognition it sees, with the handoff's start epoch.
    app, world = _coplay_app(tmp_path, monkeypatch, frames=None)
    order = []
    async def start(sid, started_at): order.append(('start', sid, started_at))
    async def end(ended): order.append(('end', ended.session_id, ended.started_at)); world.ended.append(ended)
    app.state.retro_session_start_sink = start
    app.state.retro_session_end_sink = end
    with TestClient(app) as c:
        c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=HANDOFF, headers=H)
        assert _wait(lambda: world.ended)
    assert [o[0] for o in order] == ['start', 'end']
    assert order[0][1:] == order[1][1:]   # same session and start epoch: the same act key



def test_coplay_handoff_gone_before_first_poll_opens_no_act(tmp_path, monkeypatch):
    # Review finding: the policy already gone on the first poll still exports the row, but the
    # start sink never fired, so the export must not close an act (see retro_memory_export).
    app, world = _coplay_app(tmp_path, monkeypatch, frames=None)
    world.polls = 3   # the next status() is poll 4: running, no scripted policy
    started = []
    async def start(sid, started_at): started.append(sid)
    app.state.retro_session_start_sink = start
    with TestClient(app) as c:
        c.post('/api/retro/sessions/' + 'a' * 36 + '/scripted', json=HANDOFF, headers=H)
        assert _wait(lambda: world.ended)
    assert started == []


def test_record_delta_is_none_when_retro_reset_or_did_not_report():
    from scripts.retro_session_end import record_delta
    a = dict(rooms_new=5, deaths=1, controller_frames=dict(human=10, script=0))
    b = dict(rooms_new=2, deaths=1, controller_frames=dict(human=5, script=9))   # a save load reset
    d = record_delta(a, b)
    assert d['rooms_new'] is None and d['deaths'] == 0 and d['controller_frames'] is None
    assert record_delta(dict(rooms_new=None), dict(rooms_new=3))['rooms_new'] is None


def _closed_south(**over):
    # Live 2026-10-23 room 119: south side solid, Link already at the bottom wall.
    previous = dict(decision_id='3:0', goal=dict(kind='explore', dir='S'), prediction=dict(kind='reach_room', value=120),
                    outcome='completed', verdict='refuted', applied_frames=0,
                    start_digest='ea8443189cdd1bb1', end_digest='ea8443189cdd1bb1')
    return decision(open_sides=['N', 'W', 'E'], previous=previous, **over)


def test_prompt_names_open_sides_and_moves_that_did_nothing(planner):
    d = _closed_south()
    prompt = planner.build_zelda_prompt(d, ATLAS, [d['previous']])
    assert ('an exit can only use these): N (room 103, visited), W (room 118, never visited), '
            'E (room 120, visited). No way out: S.') in prompt
    assert 'verdict refuted; nothing changed' in prompt
    assert 'Sides Link can walk off: not known here.' in planner.build_zelda_prompt(decision(), ATLAS, [])


@pytest.mark.parametrize('goal,reason', [
    (dict(kind='exit', dir='S'), 'planner_side_closed'),
    (dict(kind='explore', dir='S'), 'planner_repeats_no_effect'),
])
def test_impossible_or_repeated_goal_is_retried_once_with_the_reason(planner, goal, reason):
    bad = dict(GOOD, goal=goal, prediction=dict(kind='reach_room', value=120))
    chat = reply(lambda n: dict(content=json.dumps(bad if n == 1 else GOOD)))
    out = asyncio.run(planner.plan_zelda(None, _closed_south(), ATLAS, [], chat=chat))
    assert out['goal'] == {'kind': 'exit', 'dir': 'N'} and len(chat.calls) == 2
    assert 'Your previous answer was refused' in chat.calls[1] and 'refused' not in chat.calls[0]
    chat = reply(dict(content=json.dumps(bad)))              # refused twice: no goal, never a third call
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, _closed_south(), ATLAS, [], chat=chat))
    assert exc.value.reason == reason and len(chat.calls) == 2


def test_explore_toward_a_closed_side_is_allowed_once_and_moved_repeats_are_fine(planner):
    walk = dict(GOOD, goal=dict(kind='explore', dir='S'), prediction=dict(kind='unknown'))
    first = decision(open_sides=['N', 'W', 'E'])                     # walking to a wall once can be looking
    assert asyncio.run(planner.plan_zelda(None, first, ATLAS, [], chat=reply(dict(content=json.dumps(walk)))))
    moved = dict(_closed_south()['previous'], goal=dict(kind='exit', dir='N'), applied_frames=40,
                 end_digest='0000000000000000')
    again = dict(GOOD)                                               # same exit, but Link moved last time
    out = asyncio.run(planner.plan_zelda(None, decision(open_sides=['N'], previous=moved), ATLAS, [],
                                         chat=reply(dict(content=json.dumps(again)))))
    assert out['goal'] == {'kind': 'exit', 'dir': 'N'}


def test_not_controllable_is_never_retried(planner):
    d = decision(); d['observation']['link']['controllable'] = False
    chat = reply(dict(content=json.dumps(GOOD)))
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, d, ATLAS, [], chat=chat))
    assert exc.value.reason == 'observation_unknown' and len(chat.calls) == 1


@pytest.mark.parametrize('previous', [
    dict(goal=dict(kind='wait'), outcome='budget_expired', applied_frames=0, start_digest='a', end_digest='a'),
    dict(goal=dict(kind='exit', dir='N'), outcome='stopped:human_takeover', applied_frames=0,
         start_digest='a', end_digest='a'),
    dict(goal=dict(kind='exit', dir='N'), outcome='budget_expired', applied_frames=0,
         start_digest='unknown', end_digest='unknown'),
])
def test_waits_and_goals_that_never_ran_may_be_repeated(planner, previous):
    answer = dict(GOOD, goal=previous['goal'], prediction=dict(kind='survive') if previous['goal']['kind'] == 'wait'
                  else GOOD['prediction'])
    d = decision(open_sides=['N', 'W', 'E'], previous=dict(previous, decision_id='3:0', prediction={}, verdict='unknown'))
    chat = reply(dict(content=json.dumps(answer)))
    assert asyncio.run(planner.plan_zelda(None, d, ATLAS, [], chat=chat))['goal'] == previous['goal']
    assert len(chat.calls) == 1


def test_slow_first_answer_is_not_retried(planner, monkeypatch):
    bad = dict(GOOD, goal=dict(kind='exit', dir='S'), prediction=dict(kind='reach_room', value=120))
    ticks = iter([0.0, 20.0])
    monkeypatch.setattr(planner, '_now', lambda: next(ticks))
    chat = reply(dict(content=json.dumps(bad)))
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, _closed_south(), ATLAS, [], chat=chat))
    assert exc.value.reason == 'planner_side_closed' and len(chat.calls) == 1


def test_a_stray_value_on_a_non_room_prediction_is_dropped_not_refused(planner):
    # Live 2026-10-23 room 118: "fight nearest, enemies_cleared, value 1" was refused 13 times.
    fight = dict(GOOD, goal=dict(kind='fight', target='nearest'), prediction=dict(kind='enemies_cleared', value=1))
    chat = reply(dict(content=json.dumps(fight)))
    goal = asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=chat))
    assert goal['prediction'] == {'kind': 'enemies_cleared', 'value': None} and len(chat.calls) == 1
    room = dict(GOOD, prediction=dict(kind='reach_room'))          # reach_room without a room is still refused
    with pytest.raises(planner.PlannerRejected) as exc:
        asyncio.run(planner.plan_zelda(None, decision(), ATLAS, [], chat=reply(dict(content=json.dumps(room)))))
    assert exc.value.reason == 'planner_prediction_value_mismatch'
    assert 'reach_room needs "value"' in planner.refusal_note(exc.value)
