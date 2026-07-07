from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.embodiment.arbiter import ArbiterState, decide
from orion.schemas.embodiment import EmbodimentIntentV1


def _intent(source: str, kind: str = "wander") -> EmbodimentIntentV1:
    return EmbodimentIntentV1(kind=kind, source=source, reason="r", correlation_id="c")


def test_deliberate_always_accepted_and_sets_hold():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    state = ArbiterState()
    d = decide(_intent("deliberate"), state, now=now, hold_sec=8)
    assert d.accept is True
    assert state.deliberate_hold_until == now + timedelta(seconds=8)


def test_involuntary_preempted_during_hold():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    state = ArbiterState(deliberate_hold_until=now + timedelta(seconds=5))
    d = decide(_intent("involuntary"), state, now=now, hold_sec=8)
    assert d.accept is False
    assert d.status == "preempted"
    assert "hold active" in d.reason


def test_involuntary_accepted_after_hold_expires():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    state = ArbiterState(deliberate_hold_until=now - timedelta(seconds=1))
    d = decide(_intent("involuntary"), state, now=now, hold_sec=8)
    assert d.accept is True


def test_involuntary_accepted_when_no_hold():
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    d = decide(_intent("involuntary"), ArbiterState(), now=now, hold_sec=8)
    assert d.accept is True
