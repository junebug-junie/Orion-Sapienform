"""Offline eval of the self-finding replay over a planted two-day hourly history, through the host's
real hysteresis (``record.advance``: a change must hold 3 evaluations).

Planted: walkway grading is broken from the start (told once at the first evaluation); it is broken
for 30 hours, then works again (told once as a recovery after 3 holding hours). A prediction faculty
is reliable, wobbles to unreliable for 2 hours (never held: no message), then turns unreliable for
good (one message). A forecast flips between miscalibrated and reliable every hour (no message).
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1

T0 = datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)


def _load():
    path = Path(__file__).with_name("run_self_finding_replay.py")
    spec = importlib.util.spec_from_file_location("run_self_finding_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rec(faculty, candidate, now):
    return FacultyRecordV1(faculty=faculty, gradeability="graded", window_start=now - timedelta(days=7),
                           window_end=now, n_bets=20, n_graded=10, n_failed_to_execute=10, n_expired=0,
                           weight_graded=5.0, reliability=0.9, reliability_lo=0.6, reliability_hi=1.0,
                           claim_kind="success", candidate_status=candidate, status=candidate, candidate_held=1)


def _states():
    out = []
    for h in range(48):
        now = T0 + timedelta(hours=h)
        walkway = "broken" if h < 30 else "reliable"
        prime = "unreliable" if h in (10, 11) or h >= 40 else "reliable"
        forecast = "miscalibrated" if h % 2 else "reliable"
        out.append(SelfMonitorStateV1(computed_at=now, window_days=7.0, faculties=[
            _rec("act.actions.journal.make.walkway_grade", walkway, now),
            _rec("act.prime.predict", prime, now),
            _rec("forecast.office.temp_c", forecast, now)]))
    return out


def test_planted_history_tells_exactly_three_findings():
    mod = _load()
    per_hour = mod.replay(_states())
    told = [(h, f["kind"], f["faculty"], f["from_status"]) for h, fs in enumerate(per_hour) for f in fs]
    assert told == [
        (0, "became_broken", "act.actions.journal.make.walkway_grade", "unseen"),
        (32, "recovered", "act.actions.journal.make.walkway_grade", "broken"),
        (42, "became_unreliable", "act.prime.predict", "reliable"),
    ]


def test_first_evaluation_of_one_live_shaped_state():
    mod = _load()
    out = mod.would_tell(mod.held(_states()[0]))
    assert [f["kind"] for f in out["findings"]] == ["became_broken"]
    assert "act.actions.journal.make.walkway_grade" in out["prompt"]
    assert "10 of 20 tries never ran" in out["prompt"]
