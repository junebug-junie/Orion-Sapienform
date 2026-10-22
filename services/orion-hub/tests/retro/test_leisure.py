"""Solo leisure (spec 2026-10-22 section E): every refusal reason, the runner, preemption.

Deterministic: an in-memory Redis, a scripted Retro and a fake clock. No live services.
"""
import asyncio
import fnmatch
import itertools
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import retro_leisure as L
from scripts import retro_planner as P
from scripts.retro_presence import (BROWSER_PREFIX, HUMAN_AT_KEY, JOURNAL_KEY, LEDGER_KEY, LOCK_KEY,
                                    STATUS_KEY, PresenceStore)


class FakeRedis:
    def __init__(self): self.kv = {}; self.lists = {}; self.fail = False
    def _check(self):
        if self.fail: raise ConnectionError('redis down')
    async def get(self, k): self._check(); return self.kv.get(k)
    async def set(self, k, v, ex=None, nx=False):
        self._check()
        if nx and k in self.kv: return None
        self.kv[k] = v; return True
    async def delete(self, k): self._check(); self.kv.pop(k, None)
    async def expire(self, k, ttl): self._check(); return k in self.kv
    async def scan_iter(self, match=None, count=None):
        self._check()
        for k in list(self.kv):
            if fnmatch.fnmatch(k, match): yield k
    async def lpush(self, k, v): self._check(); self.lists.setdefault(k, []).insert(0, v)
    async def ltrim(self, k, a, b): self._check(); self.lists[k] = self.lists.get(k, [])[a:b + 1]
    async def lrange(self, k, a, b): self._check(); return self.lists.get(k, [])[a:b + 1]
    async def eval(self, script, numkeys, key, owner, *args):
        self._check()
        if self.kv.get(key) != owner: return 0
        if "'del'" in script: self.kv.pop(key, None)
        return 1


class Clock:
    def __init__(self, t=1_000_000.0): self.t = t
    def __call__(self): return self.t
    async def sleep(self, s): self.t += s; await asyncio.sleep(0)


def zelda_decision(n):
    # Shape of Retro's scripted[].cognition.attempt (services/orion-retro/app/zelda/navigate.py offer()).
    return dict(decision_id=f'5:{n}', port=0, expected_epoch=5, supported_goals=['exit', 'wait'],
                supported_predictions=['reach_room', 'survive'],
                observation=dict(status='observed', level=0, room=119, link=dict(controllable=True)))


class FakeRetro:
    def __init__(self, clock, saves=()):
        self.clock = clock; self.session = None; self.saves = list(saves); self.calls = []; self.goals = []
        self.on_tick = None; self.n = 0; self.goal_status = None; self.fail_gets = 0; self.get_status = None
        self.offered_at = None; self.control_gate = None
    async def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self.fail_gets and method == 'GET' and path.startswith('sessions/') and path.count('/') == 1:
            self.fail_gets -= 1
            raise TimeoutError('retro slow')
        if path == 'health/ready': return 200, {'session': self.session}
        if path.startswith('games/'): return 200, self.saves
        if method == 'POST' and path == 'sessions':
            self.session = dict(session_id='s' * 36, generation='g' * 36, state='running', game_id=body['game_id'],
                                scripted=[], loaded_save=body.get('save_id'))
            return 200, dict(self.session)
        sid, _, op = path[len('sessions/'):].partition('/')
        if not self.session or sid != self.session['session_id']: return 409, {'detail': 'unknown_session'}
        if method == 'GET' and not op:
            if self.on_tick: self.on_tick(self)
            if self.get_status: return self.get_status
            # Real Retro drops a leisure policy after two unanswered 30 s offers and the game
            # pauses for no_controller (services/orion-retro/app/scripted.py leisure_sponsor_lost).
            if (self.session['scripted'] and self.offered_at is not None
                    and self.clock.t - self.offered_at >= 60):
                self.session['scripted'] = []; self.session['state'] = 'paused'
            return 200, dict(self.session)
        if op == 'atlas':   # Retro's neighbourhood() view
            return 200, {'lineage_id': 'lin-9', 'level': 0, 'room': 119, 'current': None, 'adjacent': {},
                         'rooms_known': 0, 'rooms_with_unvisited_sides': [], 'caves': [], 'deaths': [], 'map': {}}
        if op == 'resume': return 200, {'state': 'running'}
        if op == 'control':
            assert body == {'leisure_id': body['leisure_id'], 'port': 0, 'action': 'acquire'}
            if self.control_gate: await self.control_gate.wait()
            self.lease_epoch = 4
            return 200, {'port': 0, 'lease': 'l' * 40, 'epoch': 4, 'leisure': True}
        if op == 'scripted':
            assert body['leisure_id'] and body['cognition'] is True and 'connection_id' not in body
            if body['action'] == 'start':
                assert body['expected_epoch'] == self.lease_epoch
                self.session['scripted'] = [dict(port=body['port'], epoch=5, policy_id='zelda-navigator-v1',
                                                 cognition=dict(policy_id='zelda-navigator-v1', attempt=zelda_decision(0),
                                                                goal=None, last_finished=None))]
                self.offered_at = self.clock.t
                return 200, {'scripted': [dict(p) for p in self.session['scripted']]}
            assert body['expected_epoch'] == 5
            self.session['scripted'] = []
            return 200, {'scripted': []}
        if op == 'goal':
            if self.goal_status:
                return self.goal_status
            self.goals.append(body)
            policy = self.session['scripted'][0]
            policy['cognition']['last_finished'] = dict(decision_id=body['decision_id'], goal=body['goal'],
                                                        verdict='met' if self.n % 2 == 0 else 'refuted')
            self.n += 1
            policy['cognition']['attempt'] = zelda_decision(self.n)
            self.offered_at = self.clock.t
            return 200, body
        if op == 'save': return 200, {'saved': {'save_id': 'f' * 32}}
        if op == 'stop': self.session['state'] = 'stopped'; return 200, self.session
        return 404, {'detail': 'not_found'}


GOAL = dict(goal=dict(kind='exit', dir='N'), prediction=dict(kind='reach_room', value=103),
            rationale='north unexplored', budget_seconds=10, planner_id='llm-gateway:metacog_background:x')


def config(**over):
    base = dict(RETRO_LEISURE_ENABLED=True, HUB_RETRO_ENABLED=True, RETRO_LEISURE_GAME_IDS='the-legend-of-zelda',
                RETRO_LEISURE_SESSION_MAX_SEC=1800, RETRO_LEISURE_DAILY_MAX_SEC=5400,
                RETRO_LEISURE_MIN_GAP_SEC=10800)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def world():
    clock = Clock(); redis = FakeRedis(); retro = FakeRetro(clock)
    reg = SimpleNamespace(value=('idle', 'idle', None))
    plans = []
    async def plan(decision, atlas, history):
        plans.append(decision['decision_id'])
        if world_ns.plan_error: raise world_ns.plan_error
        return dict(GOAL, decision_id=decision['decision_id'])
    async def regulation(): return reg.value
    deps = L.LeisureDeps(store=PresenceStore(redis, clock=clock), retro=retro, regulation=regulation,
                         plan=plan, clock=clock, sleep=clock.sleep)
    world_ns = SimpleNamespace(clock=clock, redis=redis, retro=retro, reg=reg, plans=plans, deps=deps, plan_error=None)
    L._running_here = False
    return world_ns


def journal(w):
    import json
    return [json.loads(r) for r in w.redis.lists.get(JOURNAL_KEY, [])]


# --- the pure gate -----------------------------------------------------------------------------

BASE = L.LeisureGateInputs(now=1e6, enabled=True, retro_enabled=True, game_id='the-legend-of-zelda',
                           allowed_games=('the-legend-of-zelda',), session_max_sec=1800, daily_max_sec=5400,
                           min_gap_sec=10800, arousal_level='idle', arousal_reason='idle', cabinet_reflex=None,
                           presence_read_ok=True, seconds_since_human=None, browsers_connected=0,
                           retro_session_state=None, ledger=[])

START_REFUSALS = {
    'disabled': dict(enabled=False),
    'retro_disabled': dict(retro_enabled=False),
    'game_not_allowed': dict(game_id='metroid'),
    'arousal_unknown': dict(arousal_level='unknown'),
    'cabinet_hot': dict(cabinet_reflex='cabinet_hot', arousal_level='strained'),
    'cabinet_unknown': dict(cabinet_reflex='cabinet_unknown'),
    'arousal_strained': dict(arousal_level='strained'),
    'presence_unknown': dict(presence_read_ok=False),
    'browser_connected': dict(browsers_connected=1),
    'human_recent': dict(seconds_since_human=599.0),
    'retro_unavailable': dict(retro_session_state='unreadable'),
    'retro_session_active': dict(retro_session_state='paused'),
    'ledger_unknown': dict(ledger=None),
    'cooldown': dict(ledger=[dict(leisure_id='old', started_at=1e6 - 10799, ended_at=1e6 - 10000, seconds=60)]),
    'daily_cap': dict(ledger=[dict(leisure_id=f'o{i}', started_at=1e6 - 80000 + i, ended_at=1, seconds=1800)
                              for i in range(3)]),
}


@pytest.mark.parametrize('reason', sorted(START_REFUSALS))
def test_every_start_refusal_is_named(reason):
    assert L.leisure_block_reason(replace(BASE, **START_REFUSALS[reason])) == reason


def test_clear_gate_starts_and_old_or_short_history_does_not_block():
    assert L.leisure_block_reason(BASE) is None
    old = [dict(leisure_id='o', started_at=1e6 - 90000, ended_at=1, seconds=5000)]
    assert L.leisure_block_reason(replace(BASE, ledger=old, seconds_since_human=601.0)) is None
    # A crashed runner's open row is charged up to the session cap, not forever.
    crashed = [dict(leisure_id='o', started_at=1e6 - 20000, ended_at=None, seconds=0)]
    assert L.leisure_block_reason(replace(BASE, ledger=crashed)) is None
    assert L.leisure_block_reason(replace(BASE, ledger=crashed, daily_max_sec=1800)) == 'daily_cap'


RUNNING = replace(BASE, own_leisure_id='me', elapsed_sec=120.0, browsers_connected=None,
                  retro_session_state='unreadable')


def test_running_session_rechecks_and_a_watching_browser_does_not_end_it():
    assert L.leisure_block_reason(replace(RUNNING, browsers_connected=2)) is None
    assert L.leisure_block_reason(replace(RUNNING, seconds_since_human=30.0)) == 'human_present'
    assert L.leisure_block_reason(replace(RUNNING, seconds_since_human=500.0)) is None   # before we started
    assert L.leisure_block_reason(replace(RUNNING, elapsed_sec=1800.0)) == 'session_cap'
    used = [dict(leisure_id='o', started_at=1e6 - 3600, ended_at=1, seconds=5300)]
    assert L.leisure_block_reason(replace(RUNNING, ledger=used)) == 'daily_cap'
    assert L.leisure_block_reason(replace(RUNNING, cabinet_reflex='cabinet_hot')) == 'cabinet_hot'
    assert L.leisure_block_reason(replace(RUNNING, enabled=False)) == 'disabled'


def test_arousal_and_heat_never_loosen_a_refusal():
    """Over a grid of inputs: anything refused stays refused under every arousal/heat value."""
    others = [dict(browsers_connected=1), dict(seconds_since_human=10.0), dict(retro_session_state='running'),
              START_REFUSALS['cooldown'], START_REFUSALS['daily_cap'], dict(enabled=False), dict(ledger=None)]
    for extra, level, reflex in itertools.product(others, ('engaged', 'idle', 'strained', 'unknown'),
                                                  (None, 'cabinet_hot', 'cabinet_unknown')):
        assert L.leisure_block_reason(replace(BASE, **extra, arousal_level=level, cabinet_reflex=reflex)) is not None


# --- presence store ------------------------------------------------------------------------------

def test_presence_store_reads_fail_closed():
    clock = Clock(); redis = FakeRedis(); store = PresenceStore(redis, clock=clock)
    async def go():
        assert await store.seconds_since_human() == (True, None)
        await store.mark_human(); clock.t += 42
        assert await store.seconds_since_human() == (True, 42.0)
        await store.beat_browser('a'); assert await store.browsers_connected() == 1
        await store.drop_browser('a'); assert await store.browsers_connected() == 0
        redis.fail = True
        await store.mark_human()   # best effort: never raises on the human path
        assert await store.seconds_since_human() == (False, None)
        assert await store.browsers_connected() is None and await store.ledger() is None
    asyncio.run(go())


# --- runner refusals reach the journal ------------------------------------------------------------

@pytest.mark.parametrize('setup,reason', [
    (lambda w: setattr(w.reg, 'value', ('strained', 'strained', 'cabinet_hot')), 'cabinet_hot'),
    (lambda w: setattr(w.reg, 'value', ('strained', 'strained', None)), 'arousal_strained'),
    (lambda w: setattr(w.reg, 'value', ('unknown', 'stale', None)), 'arousal_unknown'),
    (lambda w: w.redis.kv.__setitem__(BROWSER_PREFIX + 'x', '1'), 'browser_connected'),
    (lambda w: w.redis.kv.__setitem__(HUMAN_AT_KEY, str(w.clock.t - 60)), 'human_recent'),
    (lambda w: setattr(w.retro, 'session', dict(session_id='h', state='running')), 'retro_session_active'),
    (lambda w: w.redis.kv.__setitem__(LEDGER_KEY, '[{"leisure_id":"o","started_at":%f,"ended_at":1,"seconds":60}]'
                                       % (w.clock.t - 3600)), 'cooldown'),
    (lambda w: w.redis.kv.__setitem__(LOCK_KEY, 'other-runner'), 'leisure_already_running'),
    (lambda w: setattr(w.redis, 'fail', True), 'arousal_unknown'),
])
def test_runner_refusals_are_journaled_and_touch_nothing(world, setup, reason):
    setup(world)
    if reason == 'arousal_unknown' and world.redis.fail:
        async def unreadable(): return ('unknown', 'read_failed', None)
        world.deps.regulation = unreadable
    out = asyncio.run(L.run_leisure_session(config(), act_id='act-1', deps=world.deps))
    assert out.started is False and out.refusal == reason
    assert not any(m == 'POST' for m, _, _ in world.retro.calls)
    if not world.redis.fail:
        assert journal(world)[0]['event'] == 'refused' and journal(world)[0]['reason'] == reason


def test_switch_off_refuses_before_any_io(world):
    out = asyncio.run(L.run_leisure_session(config(RETRO_LEISURE_ENABLED=False), deps=world.deps))
    assert out.refusal == 'disabled' and world.retro.calls == []
    assert asyncio.run(L.should_start(config(RETRO_LEISURE_ENABLED=False), deps=world.deps)) == 'disabled'
    assert asyncio.run(L.should_start(config(HUB_RETRO_ENABLED=False), deps=world.deps)) == 'retro_disabled'
    assert asyncio.run(L.should_start(config(), deps=world.deps)) is None


# --- runner sessions ------------------------------------------------------------------------------

def test_session_plays_loads_latest_save_and_ends_at_the_cap(world):
    world.retro.saves = [dict(save_id='b' * 32, compatible=True), dict(save_id='c' * 32, compatible=True)]
    out = asyncio.run(L.run_leisure_session(config(RETRO_LEISURE_SESSION_MAX_SEC=120), act_id='act-7', deps=world.deps))
    assert out.started and out.ended.end_reason == 'session_cap'
    started = [b for m, p, b in world.retro.calls if m == 'POST' and p == 'sessions'][0]
    assert started['save_id'] == 'b' * 32 and started['game_id'] == 'the-legend-of-zelda'
    assert world.retro.goals and all(g['leisure_id'] == out.ended.leisure_id and g['expected_epoch'] == 5
                                     and 'connection_id' not in g for g in world.retro.goals)
    assert len(set(world.plans)) == len(world.plans)        # each decision planned once
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert ops[:4] == ['sessions', 'resume', 'control', 'scripted']   # leisure acquire, then hand to script
    assert ops[-3:] == ['scripted', 'save', 'stop']          # policy stop, save, close the game
    e = out.ended
    assert (e.session_id, e.lineage_id, e.act_id, e.loaded_save_id, e.end_save_id) == \
        ('s' * 36, 'lin-9', 'act-7', 'b' * 32, 'f' * 32)
    assert e.decisions_planned == len(world.retro.goals) and e.verdicts['met'] >= 1 and e.verdicts['refuted'] >= 1
    import json
    ledger = json.loads(world.redis.kv[LEDGER_KEY])
    assert ledger[-1]['ended_at'] is not None and ledger[-1]['seconds'] >= 120
    assert json.loads(world.redis.kv[STATUS_KEY])['state'] == 'ended' and LOCK_KEY not in world.redis.kv
    assert [r['event'] for r in journal(world)][-2:] == ['started', 'starting']
    # The next start is refused by the 3 h gap.
    assert asyncio.run(L.should_start(config(), deps=world.deps)) == 'cooldown'


def test_fresh_game_when_no_save(world):
    out = asyncio.run(L.run_leisure_session(config(RETRO_LEISURE_SESSION_MAX_SEC=60), deps=world.deps))
    started = [b for m, p, b in world.retro.calls if m == 'POST' and p == 'sessions'][0]
    assert 'save_id' not in started and out.ended.loaded_save_id is None


def test_human_takeover_preempts_without_stopping_their_game(world):
    def preempt(retro):
        if world.clock.t > 1_000_000 + 20 and retro.session['scripted']:
            world.redis.kv[HUMAN_AT_KEY] = str(world.clock.t)   # the relay marked a human action
            retro.session['scripted'] = []                    # Retro's arbiter dropped the leisure grant
    world.retro.on_tick = preempt
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'human_preempted'
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert 'stop' not in ops and 'save' not in ops and ops.count('scripted') == 1   # their game: hands off
    assert out.ended.human_may_be_in_control and out.ended.end_save_error == 'skipped:human_may_be_in_control'
    assert world.clock.t - 1_000_000 < 25                   # seen on the next 1 s poll


def test_policy_gone_without_a_human_is_not_called_preemption(world):
    world.retro.on_tick = lambda r: r.session.__setitem__('scripted', []) if world.clock.t > 1_000_010 else None
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'policy_ended_by_retro'
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert 'stop' not in ops and ops[-1] == 'save'           # never stops a game it no longer drives


def test_policy_gone_with_unreadable_presence_fails_safe(world):
    # The human's mark has not landed (or Redis is down) when the policy disappears.
    def drop(retro):
        if world.clock.t > 1_000_010 and retro.session['scripted']:
            retro.session['scripted'] = []; world.redis.fail = True
    world.retro.on_tick = drop
    async def reg(): return ('idle', 'idle', None)
    world.deps.regulation = reg
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'policy_ended_presence_unknown' and out.ended.human_may_be_in_control
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert 'stop' not in ops and 'save' not in ops


def test_human_action_without_takeover_releases_the_port_but_keeps_their_game(world):
    # e.g. the human pressed Pause or Save: the relay marks a human, the policy still holds the port.
    def press(retro):
        if world.clock.t > 1_000_005: world.redis.kv[HUMAN_AT_KEY] = str(world.clock.t)
    world.retro.on_tick = press
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'human_present'
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert ops.count('scripted') == 2 and 'stop' not in ops and 'save' not in ops   # released, nothing else


def test_retro_transport_errors_retry_then_close_out(world):
    world.retro.on_tick = lambda r: setattr(r, 'fail_gets', 10) if world.clock.t > 1_000_003 and not r.fail_gets else None
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'retro_unreachable'
    assert journal(world)[0]['event'] == 'ended' and LOCK_KEY not in world.redis.kv


def test_cancellation_still_releases_the_port_and_journals(world):
    async def go():
        task = asyncio.create_task(L.run_leisure_session(config(), deps=world.deps))
        for _ in range(200):
            await asyncio.sleep(0)
            if world.retro.goals: break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(go())
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert ops[-3:] == ['scripted', 'save', 'stop'] and journal(world)[0]['reason'] == 'cancelled'
    assert LOCK_KEY not in world.redis.kv


def test_stale_decision_refusal_is_play_not_failure(world):
    world.retro.goal_status = (409, {'detail': 'stale_decision'})
    out = asyncio.run(L.run_leisure_session(config(RETRO_LEISURE_SESSION_MAX_SEC=60), deps=world.deps))
    assert out.ended.end_reason == 'session_cap' and out.ended.planner_rejections == 0
    assert any(r['event'] == 'goal_stale' for r in journal(world))


def test_daily_cap_is_enforced_every_poll(world):
    import json
    world.redis.kv[LEDGER_KEY] = json.dumps([dict(leisure_id='o', started_at=world.clock.t - 50000, ended_at=1,
                                                  seconds=5400 - 10)])
    out = asyncio.run(L.run_leisure_session(config(RETRO_LEISURE_MIN_GAP_SEC=0), deps=world.deps))
    assert out.ended.end_reason == 'daily_cap' and out.ended.duration_sec <= 12


def test_heat_appearing_mid_session_ends_it_at_the_30s_recheck(world):
    def heat(retro):
        if world.clock.t > 1_000_000 + 10: world.reg.value = ('strained', 'strained', 'cabinet_hot')
    world.retro.on_tick = heat
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'cabinet_hot' and 25 <= world.clock.t - 1_000_000 <= 35
    assert journal(world)[0]['event'] == 'ended' and journal(world)[0]['reason'] == 'cabinet_hot'


def test_three_consecutive_sheds_end_the_session_and_post_no_goal(world):
    # Short shed windows: three sheds land before Retro's 60 s no-goal drop.
    world.plan_error = P.PlannerShed('gpu_pool_unavailable', reason='shed:cabinet_hot', retry_after_sec=15.0)
    seen = []
    orig = world.deps.store.write_status
    async def spy(status): seen.append(status); await orig(status)
    world.deps.store.write_status = spy
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'shed_x3' and out.ended.sheds == 3 and world.retro.goals == []
    waiting = [s for s in seen if s['state'] == 'waiting']
    assert waiting and waiting[0]['waiting'] == {'reason': 'shed:cabinet_hot', 'retry_after_sec': 15.0}
    sheds = [r for r in journal(world) if r['event'] == 'planner_shed']
    assert len(sheds) == 3 and sheds[0]['reason'] == 'shed:cabinet_hot'
    assert len(world.plans) == 3                              # one probe per retry window, no hammering
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert ops[-3:] == ['scripted', 'save', 'stop']


def test_long_shed_lets_retro_drop_the_policy_and_the_orphan_game_is_closed(world):
    # The shared backoff waits 30 s, then 60 s: Retro drops the policy at 60 s with no goal.
    world.plan_error = P.PlannerShed('gpu_pool_unavailable', reason='shed:cabinet_hot', retry_after_sec=30.0)
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'shed_wait_policy_lost' and world.retro.goals == []
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert ops[-2:] == ['save', 'stop'] and ops.count('scripted') == 1   # paused orphan, no human: closed
    assert world.retro.session['state'] == 'stopped'


def test_non_200_status_is_a_transport_problem_not_a_finished_game(world):
    world.retro.on_tick = lambda r: setattr(r, 'get_status', (503, {'detail': 'retro_unavailable'})) \
        if world.clock.t > 1_000_003 else None
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'retro_unreachable:503'
    ops = [p.rsplit('/', 1)[-1] for m, p, _ in world.retro.calls if m == 'POST']
    assert ops[-3:] == ['scripted', 'save', 'stop']            # still releases the port


def test_cancellation_during_start_still_closes_out(world):
    async def go():
        world.retro.control_gate = asyncio.Event()
        task = asyncio.create_task(L.run_leisure_session(config(), deps=world.deps))
        for _ in range(50):
            await asyncio.sleep(0)
        task.cancel()
        world.retro.control_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(go())
    assert journal(world)[0]['reason'] == 'cancelled' and LOCK_KEY not in world.redis.kv
    assert world.retro.session['state'] == 'stopped'           # the game it started is closed


def test_invalid_planner_answers_never_become_goals(world):
    world.plan_error = P.PlannerRejected('planner_invalid_goal')
    out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    assert out.ended.end_reason == 'planner_failed_x3' and world.retro.goals == []
    assert out.ended.planner_rejections == 3


def test_unknown_observation_waits_without_counting_as_failure(world):
    world.plan_error = P.PlannerRejected('observation_unknown')
    out = asyncio.run(L.run_leisure_session(config(RETRO_LEISURE_SESSION_MAX_SEC=60), deps=world.deps))
    assert out.ended.end_reason == 'session_cap' and out.ended.planner_rejections == 0 and world.retro.goals == []


def test_one_session_per_process(world):
    L._running_here = True
    try:
        out = asyncio.run(L.run_leisure_session(config(), deps=world.deps))
    finally:
        L._running_here = False
    assert out.refusal == 'leisure_already_running'
