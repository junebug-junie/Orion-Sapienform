"""Eval: the track-record cue's numbers all resolve to the record, on a state shaped like the live
2026-11-03 report (patch 1 first findings: walkway-grade broken 22/28, turn size ranges 6%/8% vs 80%,
Prime predictions 2%, cabinet forecast reliable)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1  # noqa: E402
from orion.self_monitor.cue import project_track_record_cue  # noqa: E402
from orion.self_monitor.view import self_monitor_view  # noqa: E402
from self_track_record_truth import check_cue  # noqa: E402

NOW = datetime(2026, 11, 3, 3, 40, tzinfo=timezone.utc)


def _r(faculty, status, **kw):
    base = dict(faculty=faculty, sources=["s"], claim_kind="success", gradeability="graded",
                window_start=NOW - timedelta(days=7), window_end=NOW, n_bets=kw.get("n_graded", 0),
                n_graded=0, n_failed_to_execute=0, n_expired=0, weight_graded=1.0,
                candidate_status=status, status=status, status_since=NOW, candidate_held=3)
    base.update(kw)
    return FacultyRecordV1(**base)


LIVE_SHAPED = [
    _r("act.actions.journal.make.walkway_grade", "broken", n_bets=28, n_graded=6, n_failed_to_execute=22,
       reliability=1.0, stated_mean=0.48, execution_failure=0.74, execution_failure_lo=0.55),
    _r("act.prime.predict", "unreliable", n_graded=36, reliability=0.02, reliability_hi=0.14, stated_mean=0.11),
    _r("turn.self.steps", "miscalibrated", claim_kind="interval", n_graded=1657, reliability=0.06, stated_mean=0.8,
       calibrated_share=0.06, calibrated_lo=0.04, calibrated_hi=0.07),
    _r("turn.self.reply_length", "miscalibrated", claim_kind="interval", n_graded=1657, reliability=0.08,
       stated_mean=0.8, calibrated_share=0.08, calibrated_lo=0.06, calibrated_hi=0.09),
    _r("turn.self.asks_question", "miscalibrated", claim_kind="event", n_graded=1657, reliability=0.11,
       stated_mean=0.07, calibrated_share=0.11, calibrated_lo=0.10, calibrated_hi=0.13),
    _r("forecast.office.temp_c", "miscalibrated", claim_kind="interval", n_graded=1094, reliability=0.66,
       stated_mean=0.8, calibrated_share=0.66, calibrated_lo=0.63, calibrated_hi=0.68),
    _r("forecast.athena.cabinet_temp_c", "reliable", claim_kind="interval", n_graded=10070, reliability=0.84,
       stated_mean=0.8),
    _r("reverie.expectation", "unknown", gradeability="ungradeable:never_settled"),
]


def test_live_shaped_cue_is_truthful_and_leads_with_what_matters() -> None:
    raw = SelfMonitorStateV1(computed_at=NOW, window_days=7.0, faculties=LIVE_SHAPED).model_dump_json()
    view = self_monitor_view(raw, now=NOW)
    cue = project_track_record_cue(view, now=NOW)
    assert check_cue(cue, view) == []
    assert cue.shown[:3] == ["act.actions.journal.make.walkway_grade", "act.prime.predict", "turn.self.steps"]
    assert "cabinet" not in cue.text


def test_truth_check_catches_a_number_not_in_the_record() -> None:
    raw = SelfMonitorStateV1(computed_at=NOW, window_days=7.0, faculties=LIVE_SHAPED).model_dump_json()
    view = self_monitor_view(raw, now=NOW)
    cue = project_track_record_cue(view, now=NOW)
    forged = cue.model_copy(update={"text": cue.text.replace("22 of 28", "23 of 28")})
    assert any("23" in p for p in check_cue(forged, view))
