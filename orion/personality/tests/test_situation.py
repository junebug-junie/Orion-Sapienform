from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.personality.situation import SituationInputs, situation_at
from orion.temporal_self.day import day_phase_for

T = datetime(2026, 10, 16, 22, 0, tzinfo=timezone.utc)


def test_arousal_is_the_latest_transition_at_or_before_the_row():
    inputs = SituationInputs(arousal=[(T, "idle"), (T - timedelta(hours=1), "engaged"), (T + timedelta(minutes=30), "strained")])
    assert situation_at(T - timedelta(minutes=30), inputs)["arousal"] == "engaged"
    assert situation_at(T, inputs)["arousal"] == "idle"
    assert situation_at(T + timedelta(minutes=31), inputs)["arousal"] == "strained"


def test_no_transition_on_record_is_unknown_not_idle():
    assert situation_at(T, SituationInputs())["arousal"] == "unknown"
    assert situation_at(T, SituationInputs(arousal=[(T + timedelta(seconds=1), "idle")]))["arousal"] == "unknown"


def test_company_from_open_or_spanning_arcs_only():
    inputs = SituationInputs(company=[(T - timedelta(minutes=5), T - timedelta(minutes=1)), (T + timedelta(minutes=10), None)])
    assert situation_at(T - timedelta(minutes=3), inputs)["company"] is True
    assert situation_at(T, inputs)["company"] is False  # the arc ended a minute before
    assert situation_at(T + timedelta(hours=2), inputs)["company"] is True  # still open


def test_day_phase_is_orions_local_day():
    assert situation_at(T, SituationInputs())["day_phase"] == day_phase_for(T)


def test_company_is_unknown_past_the_chronicle_watermark():
    # Live 2026-10-17: an arc that ended at 00:20:40 still had ended_at NULL when 00:24 rows were
    # stamped (the chronicle closes arcs ~5 min late), so they read "with company".
    arc_open_as_stored = SituationInputs(company=[(T - timedelta(hours=2), None)],
                                         company_known_until=T - timedelta(minutes=6))
    assert situation_at(T - timedelta(minutes=10), arc_open_as_stored)["company"] is True
    assert situation_at(T, arc_open_as_stored)["company"] is None
