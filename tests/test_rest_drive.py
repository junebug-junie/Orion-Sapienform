"""The rest drive, pure (orion/regulation/rest_drive.py + orion/schemas/drive_reading.py).

Pins: state precedence, the reader's tri-state verdict (only a fresh `due` is
tired; absent/stale/unparseable/no_reading/future is unknown), that easing only
ever lengthens a cooldown, and the lifecycle across a sleep reset.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from orion.regulation.rest_drive import (
    eased_cooldown_sec,
    no_rest_reading,
    read_rest_drive,
    rest_drive_view,
)
from orion.schemas.dream_cycle import SleepPressureV1
from orion.schemas.drive_reading import DriveReadingV1, parse_drive_reading

T0 = datetime(2026, 10, 9, 6, 33, tzinfo=timezone.utc)  # a real sleep start (dream_cycle)
MAX_AGE = 1800.0


def _p(level, since=T0, at=None):
    return SleepPressureV1(since=since, computed_at=at or since, pressure=level, threshold=3.0,
                           idle_required_minutes=45, idle_minutes=120)


def _read(level, *, now, last_end=None, overdue=False, has_candidates=True, errors=()):
    return read_rest_drive(_p(level, at=now), now=now, source_ref="dp-test", last_attempt_end=last_end,
                           min_interval_hours=6.0, overdue=overdue, has_candidates=has_candidates,
                           source_errors=errors)


# --- producer: state precedence ------------------------------------------------


def test_states_from_level_alone():
    now = T0 + timedelta(hours=12)
    assert _read(0.0, now=now).state == "resting"
    assert _read(2.3, now=now).state == "building"
    due = _read(3.0, now=now)
    assert due.state == "due" and due.due_reason == "threshold"


def test_refractory_wins_over_a_high_level():
    end = T0 + timedelta(seconds=6)
    r = _read(13.0, now=T0 + timedelta(hours=1), last_end=end)
    assert r.state == "refractory" and r.refractory_until == end + timedelta(hours=6)


def test_overdue_needs_material_to_replay():
    now = T0 + timedelta(hours=49)
    r = _read(0.0, now=now, overdue=True)
    assert r.state == "due" and r.due_reason == "overdue"
    assert _read(0.0, now=now, overdue=True, has_candidates=False).state == "resting"


def test_source_errors_are_no_reading_never_an_undercount():
    r = _read(1.0, now=T0 + timedelta(hours=8), errors=["current:metacog"])
    assert r.state == "no_reading" and r.level is None and "metacog" in r.no_reading_reason


def test_schema_refuses_inconsistent_readings():
    with pytest.raises(ValidationError):
        DriveReadingV1(observed_at=T0, threshold=3.0, state="due", level=4.0, source_ref="x")  # no due_reason
    with pytest.raises(ValidationError):
        DriveReadingV1(observed_at=T0, threshold=3.0, state="building", source_ref="x")  # no level
    with pytest.raises(ValidationError):
        DriveReadingV1(observed_at=T0, threshold=3.0, state="no_reading", level=1.0,
                       no_reading_reason="x", source_ref="x")
    with pytest.raises(ValidationError):
        DriveReadingV1(observed_at=T0, threshold=3.0, state="resting", level=0.0, source_ref="x", extra=1)


# --- reader verdict --------------------------------------------------------------


def test_only_a_fresh_due_reading_is_tired():
    now = T0 + timedelta(hours=18)
    due = _read(3.3, now=now)
    assert rest_drive_view(due.model_dump_json(), now=now, max_age_sec=MAX_AGE).verdict == "tired"
    assert rest_drive_view(due.model_dump_json().encode(), now=now, max_age_sec=MAX_AGE).verdict == "tired"
    for level in (0.0, 2.0):
        assert rest_drive_view(_read(level, now=now), now=now, max_age_sec=MAX_AGE).verdict == "not_tired"


@pytest.mark.parametrize(
    "raw,now_offset,reason",
    [
        (None, 0, "absent"),
        ("{not json", 0, "unparseable"),
        ('{"schema_version": "drive.reading.v1"}', 0, "unparseable"),
        ("DUE", 1801, "stale"),
        ("DUE", -400, "future_stamped"),
        ("NO_READING", 0, "no_reading"),
    ],
)
def test_everything_but_a_fresh_reading_is_unknown(raw, now_offset, reason):
    at = T0 + timedelta(hours=18)
    if raw == "DUE":
        raw = _read(9.0, now=at).model_dump_json()
    elif raw == "NO_READING":
        raw = no_rest_reading(now=at, source_ref="dp-x", threshold=3.0, reason="pressure_read_failed:OSError")
    v = rest_drive_view(raw, now=at + timedelta(seconds=now_offset), max_age_sec=MAX_AGE)
    assert v.verdict == "unknown" and v.reason == reason


def test_easing_only_ever_lengthens_and_only_when_tired():
    now = T0 + timedelta(hours=18)
    tired = rest_drive_view(_read(3.3, now=now), now=now, max_age_sec=MAX_AGE)
    calm = rest_drive_view(_read(1.0, now=now), now=now, max_age_sec=MAX_AGE)
    unknown = rest_drive_view(None, now=now, max_age_sec=MAX_AGE)
    assert eased_cooldown_sec(2700, tired, 2.0) == 5400
    assert eased_cooldown_sec(2700, calm, 2.0) is None
    assert eased_cooldown_sec(2700, unknown, 2.0) is None
    assert eased_cooldown_sec(2700, tired, 1.0) is None
    assert eased_cooldown_sec(2700, tired, 0.5) is None  # a mood never shortens a cooldown


def test_parse_drive_reading_handles_models_dicts_and_junk():
    r = _read(1.0, now=T0 + timedelta(hours=8))
    assert parse_drive_reading(r) is r
    assert parse_drive_reading(r.model_dump(mode="json")) == r
    assert parse_drive_reading(42) is None


# --- lifecycle across a sleep reset (the live 10-09 -> 10-10 shape) ---------------


def test_lifecycle_rest_build_due_sleep_refractory_rest():
    """Live: sleep 10-09 06:33, pressure 0 then < 3 for ~18 h, crossed 3 at 00:41,
    slept 01:01, 0.0 at the first check after. The reader is tired only in the
    crossing-to-sleep stretch."""
    sleep1_end = T0 + timedelta(seconds=6)
    timeline = [
        (T0 + timedelta(hours=1), 0.0, sleep1_end, "refractory", "not_tired"),
        (T0 + timedelta(hours=7), 0.0, sleep1_end, "resting", "not_tired"),
        (T0 + timedelta(hours=12), 2.3, sleep1_end, "building", "not_tired"),
        (T0 + timedelta(hours=18, minutes=8), 3.3, sleep1_end, "due", "tired"),
    ]
    sleep2_start = T0 + timedelta(hours=18, minutes=28)
    sleep2_end = sleep2_start + timedelta(seconds=7)
    timeline += [
        (sleep2_end + timedelta(minutes=1), 0.0, sleep2_end, "refractory", "not_tired"),
        (sleep2_end + timedelta(hours=6, minutes=1), 0.0, sleep2_end, "resting", "not_tired"),
    ]
    for at, level, last_end, state, verdict in timeline:
        since = sleep2_start if at > sleep2_start else T0
        r = read_rest_drive(_p(level, since=since, at=at), now=at, source_ref="dp", last_attempt_end=last_end,
                            min_interval_hours=6.0, overdue=False, has_candidates=True)
        assert r.state == state, (at, r.state)
        assert rest_drive_view(r, now=at, max_age_sec=MAX_AGE).verdict == verdict
    # And the last `due` reading, left unrefreshed, decays to unknown -- not tired forever.
    due = read_rest_drive(_p(3.3, at=timeline[3][0]), now=timeline[3][0], source_ref="dp",
                          last_attempt_end=sleep1_end, min_interval_hours=6.0, overdue=False, has_candidates=True)
    assert rest_drive_view(due, now=timeline[3][0] + timedelta(hours=1), max_age_sec=MAX_AGE).verdict == "unknown"
