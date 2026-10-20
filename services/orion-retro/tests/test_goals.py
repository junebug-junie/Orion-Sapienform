import pytest
from app.goals import Goals
from app.scripted import Scripted
from app.control import Arbiter
from app.observation import decode, outcome, Observer
from app.contracts import Goal
from pydantic import ValidationError
from types import SimpleNamespace


def fixture():
    rows = []
    authority = Arbiter([0])
    controller = Scripted(authority, lambda kind, **kw: rows.append(dict(kind=kind, **kw)))
    authority.control('human', 0, 'acquire', 0)
    controller.start('human', 0, 60, 0, 0, 1, True)
    goals = controller.active[0]['goals']
    goals.offer(0, 2, 1, {'status': 'unknown'}, 0)
    data = dict(port=0, expected_epoch=2, attempt_id='2:1', goal='jump_right',
                rationale='fixture only', planner_id='test')
    return authority, controller, goals, data, rows


def test_goal_applies_only_after_ack_and_expires_to_neutral():
    a, s, g, data, rows = fixture()
    s.tick(0); a.frame(0)
    s.tick(.07)
    assert a.frame(.07)[0] == [0, 0]
    s.goal('human', data, .07)
    s.tick(.14)
    assert a.frame(.14)[0] == [384, 0]
    assert rows[-1]['kind'] == 'goal_committed'
    assert g.mask(5.08, 0) == 0


def test_retry_is_idempotent_and_cannot_extend_goal():
    a, s, g, data, rows = fixture()
    g.accept(data, 0)
    assert g.accept(data, 4) == data
    assert g.until == 5
    assert len([r for r in rows if r['kind'] == 'goal_committed']) == 1
    with pytest.raises(ValueError, match='conflict'): g.accept(data | {'goal': 'wait'}, 4)


def test_stale_attempt_and_epoch_rejected():
    a, s, g, data, rows = fixture()
    g.offer(0, 2, 300, {}, 1)
    with pytest.raises(ValueError, match='stale_attempt'): s.goal('human', data, 1)
    with pytest.raises(ValueError, match='active_controller'):
        s.goal('human', data | {'expected_epoch': 1}, 1)
    with pytest.raises(ValueError, match='expired_attempt'):
        g.accept(data | {'attempt_id': '2:2'}, 32)


def test_takeover_disconnect_and_pause_invalidate_late_planner():
    for stop in ('takeover', 'disconnect', 'pause'):
        a, s, g, data, rows = fixture()
        if stop == 'takeover': a.control('human', 0, 'takeover', .01)
        else: s.cancel(stop)
        with pytest.raises(ValueError, match='active_controller'): s.goal('human', data, .02)
        assert not any(r['kind'] == 'goal_committed' for r in rows)


def test_sponsor_and_budget_are_authoritative():
    a, s, g, data, rows = fixture()
    with pytest.raises(ValueError): s.goal('other', data, 0)
    with pytest.raises(ValueError): s.goal('human', data, 61)


def test_unknown_observation_is_not_failure_or_calm():
    assert Observer(SimpleNamespace(identity={})).read()['status'] == 'unknown'
    assert decode(bytes(2048))['status'] == 'unknown'
    assert decode(bytes([255])*2048)['status'] == 'unknown'
    assert outcome({}, {}, 'walk_right') == 'unknown'
    ram = bytearray(2048); ram[0x4b] = 4; ram[0x46] = 1; ram[0x26] = 5
    before = decode(ram)
    assert outcome(before, before, 'wait') == 'unknown'
    assert outcome(before, before, 'walk_right') == 'no_horizontal_progress'
    ram[0x25] = 10
    assert outcome(before, decode(ram), 'jump_right') == 'horizontal_position_changed'
    ram[0x4b] = 3
    assert outcome(before, decode(ram), 'jump_right') == 'health_decreased'
    ram[0xa7] = 1
    assert outcome(before, decode(ram), 'jump_right') == 'unknown'


def test_goal_contract_has_no_arbitrary_input_or_tool_surface():
    _, _, _, data, _ = fixture()
    body = data | dict(connection_id='a'*36, generation='b'*36)
    for extra in ({'goal': 'reach_exit'}, {'mask': 65535}, {'rationale': ''}, {'port': True}):
        with pytest.raises(ValidationError): Goal(**(body | extra))


def test_delayed_planner_baseline_is_first_applied_frame_not_offer():
    a, s, g, data, rows = fixture()
    g.accept(data, 20)
    assert g.start_observation is None
    observed = dict(status='observed', level=0, x_subpixels=160,
                    health_half_hearts=4)
    g.applied_start(1201, observed)
    g.applied_start(1202, observed | {'x_subpixels': 200})
    assert g.start_frame == 1201 and g.start_observation == observed
    assert outcome(g.start_observation, observed, 'jump_right') == 'no_horizontal_progress'


def test_late_accepted_goal_keeps_its_full_action_window():
    a, s, g, data, rows = fixture()
    g.accept(data, 29)
    assert not g.ready_for_observation(30.1)
    assert not g.ready_for_observation(33.9)
    assert g.ready_for_observation(34)


def test_wait_submitted_before_ack_is_not_attributed_to_new_goal():
    a, s, g, data, rows = fixture()
    s.tick(0); a.frame(0)  # takeover barrier; initial wait is pending
    s.goal('human', data, .01)
    event = a.frame(.02)[1][0]
    s.annotate(event)
    assert 'goal_attempt_id' not in event
    s.tick(.07)
    event = a.frame(.07)[1][0]
    s.annotate(event)
    assert event['goal_attempt_id'] == data['attempt_id']
