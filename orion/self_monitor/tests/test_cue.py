"""The track-record cue (spec step 3): only notable faculties, numbers from the record, fail closed to ""."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.self_monitor.cue import MAX_CHARS, plain_name, project_track_record_cue, sentence
from orion.self_monitor.view import self_monitor_view

NOW = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)


def _rec(faculty: str, status: str, **kw) -> FacultyRecordV1:
    base = dict(
        faculty=faculty, sources=["s"], claim_kind="success", gradeability="graded",
        window_start=NOW - timedelta(days=7), window_end=NOW, n_bets=30, n_graded=30,
        n_failed_to_execute=0, n_expired=0, weight_graded=20.0, candidate_status=status, status=status,
        status_since=NOW - timedelta(hours=3), candidate_held=3,
    )
    base.update(kw)
    return FacultyRecordV1(**base)


WALKWAY = _rec("act.actions.journal.make.walkway_grade", "broken", n_bets=28, n_graded=6,
               n_failed_to_execute=22, reliability=1.0, execution_failure=0.74, execution_failure_lo=0.55,
               stated_mean=0.48)
STEPS = _rec("turn.self.steps", "miscalibrated", claim_kind="interval", n_bets=1997, n_graded=1657,
             reliability=0.06, stated_mean=0.80)
REPLY = _rec("turn.self.reply_length", "miscalibrated", claim_kind="interval", n_bets=1997, n_graded=1657,
             reliability=0.08, stated_mean=0.80)
CABINET = _rec("forecast.athena.cabinet_temp_c", "reliable", claim_kind="interval", n_graded=10070,
               reliability=0.84, stated_mean=0.80)
UNGRADED = _rec("reverie.expectation", "unknown", gradeability="ungradeable:never_settled", n_bets=0,
                n_graded=0, weight_graded=0.0)


def _raw(*records: FacultyRecordV1, computed_at: datetime = NOW) -> str:
    return SelfMonitorStateV1(computed_at=computed_at, window_days=7.0, faculties=list(records)).model_dump_json()


def _cue(*records, computed_at=NOW, now=NOW, **kw):
    return project_track_record_cue(self_monitor_view(_raw(*records, computed_at=computed_at), now=now), now=now, **kw)


def test_renders_walkway_broken_and_turn_size_miscalibrated_lines() -> None:
    cue = _cue(WALKWAY, STEPS, CABINET, UNGRADED)
    assert cue.text.startswith("My walkway-grade journal runs failed to start 22 of 28 times this week.")
    assert "My guesses about how many steps my turns take hold 6% of the time (I say 80%)." in cue.text
    assert cue.shown == [WALKWAY.faculty, STEPS.faculty]
    assert cue.text.endswith("I can't grade 1 other thing I do yet.")
    assert cue.skip_reason is None and "cabinet" not in cue.text


def test_nothing_rendered_when_everything_reliable_or_unknown() -> None:
    cue = _cue(CABINET, UNGRADED, _rec("act.x.make", "unknown"))
    assert cue.text == "" and cue.skip_reason == "nothing_notable" and cue.shown == []


def test_candidate_not_yet_held_is_not_cued() -> None:
    flapping = _rec("act.prime.predict", "reliable", candidate_status="unreliable", candidate_held=1,
                    reliability=0.02)
    assert _cue(flapping).text == ""


@pytest.mark.parametrize("raw, reason", [(None, "absent"), ("{not json", "unparseable"), (b"\xff", "unparseable")])
def test_absent_or_garbage_state_renders_nothing(raw, reason) -> None:
    cue = project_track_record_cue(self_monitor_view(raw, now=NOW), now=NOW)
    assert cue.text == "" and cue.skip_reason == reason


def test_stale_and_future_state_render_nothing() -> None:
    stale = _cue(WALKWAY, computed_at=NOW - timedelta(hours=4))
    assert stale.text == "" and stale.skip_reason == "stale" and stale.state_age_sec == 4 * 3600
    future = _cue(WALKWAY, computed_at=NOW + timedelta(hours=1))
    assert future.text == "" and future.skip_reason == "future_stamped"


def test_priority_broken_then_unreliable_then_miscalibrated_and_cap() -> None:
    prime = _rec("act.prime.predict", "unreliable", n_graded=36, reliability=0.02, stated_mean=0.11)
    many = [_rec(f"forecast.room{i}.temp_c", "miscalibrated", claim_kind="interval", n_graded=100 + i,
                 reliability=0.5, stated_mean=0.8) for i in range(10)]
    cue = _cue(*many, STEPS, prime, WALKWAY, REPLY)
    assert len(cue.text) <= MAX_CHARS
    assert cue.shown[:3] == [WALKWAY.faculty, prime.faculty, STEPS.faculty]
    assert cue.lines_dropped == len(many) + 4 - len(cue.shown)
    assert "I can't grade" not in cue.text  # never ahead of a dropped notable line
    assert "My Prime predictions work 2% of the time (36 tries this week)." in cue.text


def test_equal_evidence_breaks_tie_by_size_of_miss() -> None:
    asks = _rec("turn.self.asks_question", "miscalibrated", claim_kind="event", n_graded=1657,
                reliability=0.11, stated_mean=0.07)
    cue = _cue(asks, REPLY, STEPS, max_chars=160)
    assert cue.shown == [STEPS.faculty, REPLY.faculty]


def test_deterministic_regardless_of_input_order() -> None:
    a = _cue(WALKWAY, STEPS, REPLY, UNGRADED)
    b = _cue(UNGRADED, REPLY, STEPS, WALKWAY)
    assert (a.text, a.shown) == (b.text, b.shown)


def test_text_carries_no_identifiers() -> None:
    cue = _cue(WALKWAY, STEPS, REPLY, UNGRADED)
    for token in ("act.", "turn.self", "_", "forecast.", "faculty"):
        assert token not in cue.text


def test_every_number_comes_from_the_record() -> None:
    assert sentence(WALKWAY) == "My walkway-grade journal runs failed to start 22 of 28 times this week."
    streak = _rec("act.thought.painting.make", "broken", n_graded=2, n_failed_to_execute=5,
                  execution_failure_lo=0.3, streak_kind="failed_to_execute", streak_len=5)
    assert sentence(streak) == "My last 5 painting runs all failed to start."
    under = _rec("act.prime.infer", "miscalibrated", reliability=1.0, stated_mean=0.72)
    assert sentence(under) == "My Prime inferences work 100% of the time (I expect 72%)."
    event = _rec("turn.self.asks_question", "miscalibrated", claim_kind="event", reliability=0.11, stated_mean=0.07)
    assert sentence(event) == "My guesses that I'll ask a question come true 11% of the time (I say 7%)."
    assert sentence(CABINET) is None
    assert sentence(_rec("x.y", "miscalibrated", reliability=None)) is None


@pytest.mark.parametrize("key, name", [
    ("act.actions.journal.make.walkway_grade", "walkway-grade journal runs"),
    ("act.prime.predict", "Prime predictions"),
    ("act.hub.chat_reply.say", "chat reply messages"),
    ("act.aitown.session.play", "AI Town sessions"),
    ("forecast.office.temp_c", "office temperature forecast ranges"),
    ("body.retro:zelda.traverse", "Zelda traverse moves"),
    ("turn.self.mood", "guesses about my turns' mood"),
    ("energy.bill", "energy bill bets"),
    ("brand_new.thing", "brand new thing bets"),
])
def test_plain_names(key, name) -> None:
    assert plain_name(key) == name


def test_live_shaped_state_from_report_json_renders_within_cap() -> None:
    # The report's --json is exactly a SelfMonitorStateV1; the view accepts the dict too.
    data = json.loads(_raw(WALKWAY, STEPS, REPLY, CABINET, UNGRADED))
    cue = project_track_record_cue(self_monitor_view(data, now=NOW), now=NOW)
    assert 0 < len(cue.text) <= MAX_CHARS
