from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.gpu_pool.perception_load import PerceptionLoad

T0 = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
REALERT = timedelta(hours=6)


def test_sums_only_the_last_hour():
    p = PerceptionLoad(3360)
    p.record(T0 - timedelta(minutes=90), 3000)
    p.record(T0 - timedelta(minutes=30), 500)
    assert p.held_last_hour(T0) == 500


def test_quiet_load_never_alerts():
    p = PerceptionLoad(3360)
    for m in range(0, 60, 5):
        p.record(T0 - timedelta(minutes=m), 140)   # ~p90 hour, 1,680 s
    assert p.check(T0, REALERT) is None


def test_high_alerts_once_then_waits_for_realert_then_recovers():
    p = PerceptionLoad(3360)
    p.record(T0, 3400)
    assert p.check(T0, REALERT) == "high"
    p.record(T0 + timedelta(minutes=10), 100)
    assert p.check(T0 + timedelta(minutes=10), REALERT) is None          # still high, no repeat
    assert p.check(T0 + timedelta(minutes=70), REALERT) == "recovered"   # window emptied
    assert p.check(T0 + timedelta(minutes=80), REALERT) is None


def test_realert_after_interval_while_still_high():
    p = PerceptionLoad(100)
    p.record(T0, 200)
    assert p.check(T0, timedelta(minutes=30)) == "high"
    p.record(T0 + timedelta(minutes=40), 200)
    assert p.check(T0 + timedelta(minutes=40), timedelta(minutes=30)) == "high"


def test_zero_or_negative_holds_are_ignored():
    p = PerceptionLoad(10)
    p.record(T0, 0)
    p.record(T0, -5)
    assert p.held_last_hour(T0) == 0
