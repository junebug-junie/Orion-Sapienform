"""The track-record cue (spec step 3): only notable faculties, numbers from the record, fail closed to ""."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.self_monitor.cue import MAX_CHARS, named, plain_name, project_track_record_cue, sentence
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
             reliability=0.06, stated_mean=0.80, calibrated_share=0.06, calibrated_lo=0.04, calibrated_hi=0.07)
REPLY = _rec("turn.self.reply_length", "miscalibrated", claim_kind="interval", n_bets=1997, n_graded=1657,
             reliability=0.08, stated_mean=0.80, calibrated_share=0.08, calibrated_lo=0.06, calibrated_hi=0.09)
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
    assert cue.text.startswith("My walkway-grade journal runs (act.actions.journal.make.walkway_grade) failed to start "
                                "22 of 28 times this week.")
    assert ("My guesses about how many steps my turns take (turn.self.steps) hold 6% of the time (I say 80%)."
            in cue.text)
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
    prime = _rec("act.prime.predict", "unreliable", n_graded=36, reliability=0.02, reliability_hi=0.14,
                 stated_mean=0.11)
    many = [_rec(f"forecast.room{i}.temp_c", "miscalibrated", claim_kind="interval", n_graded=100 + i,
                 reliability=0.5, stated_mean=0.8, calibrated_share=0.5, calibrated_lo=0.4, calibrated_hi=0.6)
            for i in range(10)]
    cue = _cue(*many, STEPS, prime, WALKWAY, REPLY)
    assert len(cue.text) <= MAX_CHARS
    assert cue.shown[:3] == [WALKWAY.faculty, prime.faculty, STEPS.faculty]
    assert cue.lines_dropped == len(many) + 4 - len(cue.shown)
    assert "I can't grade" not in cue.text  # never ahead of a dropped notable line
    assert "My Prime predictions (act.prime.predict) work 2% of the time (36 tries this week)." in cue.text


def test_equal_evidence_breaks_tie_by_size_of_miss() -> None:
    asks = _rec("turn.self.asks_question", "miscalibrated", claim_kind="event", n_graded=1657,
                reliability=0.11, stated_mean=0.07, calibrated_share=0.11, calibrated_lo=0.10, calibrated_hi=0.13)
    cue = _cue(asks, REPLY, STEPS, max_chars=210)
    assert cue.shown == [STEPS.faculty, REPLY.faculty]


def test_deterministic_regardless_of_input_order() -> None:
    a = _cue(WALKWAY, STEPS, REPLY, UNGRADED)
    b = _cue(UNGRADED, REPLY, STEPS, WALKWAY)
    assert (a.text, a.shown) == (b.text, b.shown)


def test_text_names_each_faculty_plain_first_then_its_id() -> None:
    cue = _cue(WALKWAY, STEPS, REPLY, UNGRADED)
    for faculty in cue.shown:
        assert f"My {plain_name(faculty)} ({faculty}) " in cue.text
    prose = cue.text
    for faculty in cue.shown:
        prose = prose.replace(f"({faculty})", "")
    for token in ("act.", "turn.self", "_", "forecast.", "faculty"):
        assert token not in prose  # no identifier outside a shown faculty's own brackets


def test_ids_never_push_the_cue_over_its_cap() -> None:
    long_ids = [_rec(f"act.actions.journal.make.walkway_grade_variant_number_{i}", "broken", n_bets=28, n_graded=6,
                     n_failed_to_execute=22, execution_failure=0.74, execution_failure_lo=0.55) for i in range(6)]
    cue = _cue(*long_ids)
    assert 0 < len(cue.text) <= MAX_CHARS
    assert cue.lines_dropped == len(long_ids) - len(cue.shown) > 0
    assert all(cue.text.count(f"({f})") == 1 for f in cue.shown)  # whole lines only, never a cut id


def test_every_number_comes_from_the_record() -> None:
    assert sentence(WALKWAY) == ("My walkway-grade journal runs (act.actions.journal.make.walkway_grade) failed to "
                                "start 22 of 28 times this week.")
    streak = _rec("act.thought.painting.make", "broken", n_graded=2, n_failed_to_execute=5,
                  execution_failure_lo=0.3, streak_kind="failed_to_execute", streak_len=5)
    assert sentence(streak) == "My last 5 painting runs (act.thought.painting.make) all failed to start."
    under = _rec("act.prime.infer", "miscalibrated", reliability=1.0, stated_mean=0.72, calibrated_share=1.0,
                 calibrated_lo=0.98, calibrated_hi=1.0)
    assert sentence(under) == "My Prime inferences (act.prime.infer) work 100% of the time (I expect 72%)."
    event = _rec("turn.self.asks_question", "miscalibrated", claim_kind="event", reliability=0.11,
                 stated_mean=0.07, calibrated_share=0.11, calibrated_lo=0.10, calibrated_hi=0.13)
    assert sentence(event) == ("My guesses that I'll ask a question (turn.self.asks_question) come true 11% of the "
                              "time (I say 7%).")
    assert sentence(CABINET) is None
    assert sentence(_rec("x.y", "miscalibrated", reliability=None)) is None


def test_miscalibrated_uses_the_calibration_share_not_overall_reliability() -> None:
    # 30% of all graded bets upheld, but 60% of the ones that stated a confidence: the claim is about those.
    r = _rec("act.prime.infer", "miscalibrated", reliability=0.30, stated_mean=0.90, calibrated_share=0.60,
             calibrated_lo=0.50, calibrated_hi=0.70)
    assert sentence(r) == "My Prime inferences (act.prime.infer) work 60% of the time (I expect 90%)."


def test_hysteresis_lag_never_prints_numbers_that_contradict_the_status() -> None:
    healed = _rec("act.prime.predict", "unreliable", candidate_status="reliable", reliability=0.7,
                  reliability_lo=0.55, reliability_hi=0.82, stated_mean=0.6)
    assert sentence(healed) is None
    agrees = _rec("turn.self.steps", "miscalibrated", candidate_status="reliable", claim_kind="interval",
                  reliability=0.79, stated_mean=0.80, calibrated_share=0.79, calibrated_lo=0.75, calibrated_hi=0.83)
    assert sentence(agrees) is None
    still_off = agrees.model_copy(update={"calibrated_share": 0.6, "calibrated_lo": 0.55, "calibrated_hi": 0.65})
    assert sentence(still_off) == ("My guesses about how many steps my turns take (turn.self.steps) hold 60% of the "
                                  "time (I say 80%).")


def test_percent_never_rounds_to_a_false_zero_or_hundred() -> None:
    r = _rec("act.prime.predict", "unreliable", reliability=0.002, reliability_hi=0.05)
    assert "work 1% of the time" in sentence(r)
    s = _rec("act.prime.infer", "miscalibrated", reliability=0.998, stated_mean=0.7, calibrated_share=0.998,
             calibrated_lo=0.97, calibrated_hi=1.0)
    assert "work 99% of the time" in sentence(s)


def test_window_wording_follows_the_record_window() -> None:
    assert "this week" in sentence(WALKWAY)
    three = WALKWAY.model_copy(update={"window_start": NOW - timedelta(days=3)})
    assert sentence(three).endswith("22 of 28 times in the last 3 days.")


@pytest.mark.parametrize("key, name", [
    ("act.actions.journal.make.walkway_grade", "walkway-grade journal runs"),
    ("act.prime.predict", "Prime predictions"),
    ("act.hub.chat_reply.say", "chat reply messages"),
    ("act.aitown.session.play", "AI Town sessions"),
    ("forecast.office.temp_c", "office temperature forecast ranges"),
    ("body.retro:zelda.traverse", "zelda traverse moves"),
    ("turn.self.mood", "guesses about my turns' mood"),
    ("turn.self", "turn self bets"),
    ("energy.bill", "energy bill bets"),
    ("forecast.energy.bill_kwh", "power-bill kWh forecast ranges"),
    ("forecast.energy.bill_usd", "power-bill dollar forecast ranges"),
    ("brand_new.thing", "brand new thing bets"),
    ("act.make", "make bets"),
    ("act.durable.run", "durable runs"),
    ("act.hub.say", "hub messages"),
    ("forecast.desk.power_w", "desk power forecast ranges"),
    ("act.durable.journal_compose.make.world-pulse-journal", "journal compose world pulse journal runs"),
])
def test_plain_names(key, name) -> None:
    assert plain_name(key) == name
    assert named(key) == f"{name} ({key})"


def test_cue_reuses_the_one_shared_naming_function() -> None:
    from orion.self_monitor import cue, names

    assert cue.plain_name is names.plain_name and cue.named is names.named


def test_live_shaped_state_from_report_json_renders_within_cap() -> None:
    # The report's --json is exactly a SelfMonitorStateV1; the view accepts the dict too.
    data = json.loads(_raw(WALKWAY, STEPS, REPLY, CABINET, UNGRADED))
    cue = project_track_record_cue(self_monitor_view(data, now=NOW), now=NOW)
    assert 0 < len(cue.text) <= MAX_CHARS


def test_reducer_stores_the_calibration_share_the_cue_prints() -> None:
    from orion.schemas.self_monitor import SettledBetV1
    from orion.self_monitor.record import reduce_faculty

    bets = [SettledBetV1(bet_id=f"b{i}", faculty="turn.self.steps", source="t", made_at=NOW - timedelta(hours=i + 1),
                         settled_at=NOW - timedelta(hours=i), claim_kind="interval",
                         stated=0.8 if i < 20 else None, outcome="upheld" if i % 10 == 0 or i >= 20 else "refuted",
                         settled_by="world", evidence_ref=f"r{i}") for i in range(30)]
    rec = reduce_faculty("turn.self.steps", bets, now=NOW)
    assert rec.calibrated_share is not None and rec.calibrated_lo <= rec.calibrated_share <= rec.calibrated_hi
    assert rec.calibrated_share < 0.2 < rec.reliability  # stated subset: 2/20 upheld; all graded: 12/30
    assert rec.status == "miscalibrated"
    assert "hold 10% of the time (I say 80%)" in sentence(rec)


def test_a_few_more_bets_do_not_outrank_a_far_bigger_miss() -> None:
    fetch = _rec("turn.self.fetches_outside", "miscalibrated", claim_kind="event", n_graded=1663,
                 reliability=0.51, stated_mean=0.38, calibrated_share=0.51, calibrated_lo=0.49, calibrated_hi=0.54)
    steps = STEPS.model_copy(update={"n_graded": 1659})
    small = _rec("act.hub.world_pulse_read.look", "miscalibrated", n_graded=424, reliability=0.99,
                 stated_mean=0.10, calibrated_share=0.99, calibrated_lo=0.97, calibrated_hi=1.0)
    assert _cue(fetch, steps, small).shown == [steps.faculty, fetch.faculty, small.faculty]
