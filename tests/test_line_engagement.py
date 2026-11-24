"""Juniper's engagement grade for a game line (orion/grammar/line_engagement.py), pure. The resolver path against real
Postgres is services/orion-sql-writer/tests/test_retro_line_engagement_integration.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.grammar.line_engagement import LINE_ENGAGE_HORIZON_SEC, line_grade, line_id_of

T0 = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)
H = T0 + timedelta(seconds=LINE_ENGAGE_HORIZON_SEC)
AFTER = H + timedelta(seconds=30)


def g(row, *, turn=None, now=AFTER):
    return line_grade(row, opened_at=T0, horizon_at=H, now=now, juniper_turn_at=turn)


def at(sec):
    return T0 + timedelta(seconds=sec)


def test_nothing_is_graded_before_the_horizon():
    assert g({"thumb": "up", "thumb_by": "juniper", "thumb_at": at(10)}, now=H - timedelta(seconds=1)) is None


def test_juniper_thumb_decides_inside_the_horizon():
    assert g({"thumb": "up", "thumb_by": "juniper", "thumb_at": at(250)}).value == 1.0
    down = g({"thumb": "down", "thumb_by": "juniper", "thumb_at": at(20), "replied_at": at(10), "replied_by": "juniper"})
    assert (down.value, down.source) == (0.0, "resolver") and down.observed_at == at(20)


def test_a_guest_thumb_or_reply_is_never_juniper_engagement():
    assert g({"thumb": "up", "thumb_by": "guest", "thumb_at": at(10)}).value == 0.0
    assert g({"replied_at": at(30), "replied_by": "guest"}).value == 0.0


def test_a_thumb_outside_the_horizon_is_not_counted():
    assert g({"thumb": "up", "thumb_by": "juniper", "thumb_at": at(400)}).value == 0.0
    assert g({"thumb": "up", "thumb_by": "juniper", "thumb_at": at(-5)}).value == 0.0


def test_a_reply_counts_only_within_two_minutes():
    assert g({"replied_at": at(119), "replied_by": "juniper"}).value == 1.0
    assert g({"replied_at": at(121), "replied_by": "juniper"}).value == 0.0
    assert g({}, turn=at(60)).value == 1.0 and g({}, turn=at(200)).value == 0.0


def test_a_missing_row_closes_as_expired_never_as_a_grade():
    r = g(None)
    assert (r.value, r.source) == (0.0, "expired") and r.observed_at == H


def test_naive_times_read_as_utc():
    naive = {"thumb": "up", "thumb_by": "juniper", "thumb_at": at(30).replace(tzinfo=None)}
    assert g(naive).value == 1.0


def test_the_line_id_comes_from_the_acts_test_before():
    assert line_id_of("retro_line:abc") == "abc"
    assert line_id_of("retro_line:") is None and line_id_of("outreach:x") is None and line_id_of(None) is None
