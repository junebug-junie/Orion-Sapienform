import pytest
from app.control import Arbiter
from app.scripted import Scripted, POLICY_ID
from app.contracts import ScriptControl
from pydantic import ValidationError


def setup():
    a = Arbiter([0, 1])
    rows = []
    s = Scripted(a, lambda kind, **fields: rows.append(dict(kind=kind, **fields)))
    a.control('human', 0, 'acquire', 0)
    return a, s, rows


def test_explicit_owned_handoff_only_and_neutral_barrier():
    a, s, rows = setup()
    for owner, port, now in [('other', 0, 0), ('human', 1, 0), ('human', 0, 3)]:
        with pytest.raises(ValueError, match='requires_owned'):
            s.start(owner, port, 5, now, 10, 1)
    s.start('human', 0, 5, 0, 10, 1)
    s.tick(0)
    assert a.frame(0)[0] == [0, 0]
    masks, events, _ = a.frame(.02)
    assert masks == [384, 0]
    s.annotate(events[0])
    assert events[0]['policy_id'] == POLICY_ID
    assert rows[0]['kind'] == 'policy_started'
    assert 'lease' not in str(s.public()) and 'human' not in str(s.public())


def test_takeover_never_released_or_reacquired_by_script():
    a, s, rows = setup()
    s.start('human', 0, 5, 0, 0, 1)
    s.tick(0); a.frame(0); a.frame(.01)
    human = a.control('second', 0, 'takeover', .02)
    a.submit('second', dict(human, seq=0, mask=64), .02)
    s.tick(.02)
    assert s.public() == []
    assert a.frame(.02)[0] == [0, 0]
    assert a.frame(.03)[0] == [64, 0]
    s.tick(1)
    assert a.leases[0]['connection'] == 'second'
    assert rows[-1]['reason'] == 'human_takeover'


def test_delayed_handoff_cannot_consume_a_newer_human_lease():
    a, s, rows = setup()
    newer = a.control('human', 0, 'takeover', .01)
    with pytest.raises(ValueError, match='requires_owned'):
        s.start('human', 0, 5, .02, 0, 1)
    assert a.leases[0]['epoch'] == newer['epoch']
    assert a.leases[0]['connection'] == 'human'
    assert not s.public() and not rows


def test_deadline_miss_neutralizes_without_catchup():
    a, s, rows = setup()
    s.start('human', 0, 5, 0, 0, 1)
    s.tick(0); a.frame(0); a.frame(.01)
    s.tick(.11)
    assert not s.public()
    assert a.frame(.11)[0] == [0, 0]
    assert rows[-1]['reason'] == 'policy_deadline_missed'
    s.tick(.12)
    assert not a.leases


def test_bounded_rate_budget_and_sponsor_disconnect():
    a, s, rows = setup()
    a.control('other', 1, 'acquire', 0)
    s.start('human', 0, 5, 0, 0, 1)
    s.start('other', 1, 10, 0, 0, 1)
    events = []
    for i in range(500):
        now = i / 100
        s.tick(now)
        events.extend(a.frame(now)[1])
    assert 65 <= len([e for e in events if e['port'] == 0]) <= 75
    s.tick(5)
    assert [p['port'] for p in s.public()] == [1]
    assert any(r.get('reason') == 'budget_expired' for r in rows)
    s.cancel('sponsor_disconnected', 'unrelated')
    assert s.public()
    s.cancel('sponsor_disconnected', 'other')
    assert not s.public()
    assert a.frame(5)[0] == [0, 0]


@pytest.mark.parametrize('change', [
    {'duration_seconds': 0}, {'duration_seconds': 1801}, {'duration_seconds': True},
    {'port': 2}, {'action': 'takeover'}, {'policy_id': 'untrusted'},
])
def test_invalid_script_contract(change):
    body = dict(connection_id='a'*36, generation='b'*36, port=0, action='start', expected_epoch=1)
    with pytest.raises(ValidationError):
        ScriptControl(**(body | change))
