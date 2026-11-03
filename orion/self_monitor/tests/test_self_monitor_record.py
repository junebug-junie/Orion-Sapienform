"""Self-monitor patch 1: adapters (live row shapes, 2026-11-03) and the faculty reducer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.self_monitor import SettledBetV1
from orion.self_monitor import adapters
from orion.self_monitor.catalog import UNGRADEABLE
from orion.self_monitor.record import BROKEN_STREAK, HOLD, MIN_GRADED, advance, reduce_bets, reduce_faculty
from orion.self_monitor.report import render
from orion.schemas.self_monitor import SelfMonitorStateV1

NOW = datetime(2026, 11, 3, 3, 0, tzinfo=timezone.utc)


def _bet(i: int, outcome: str, *, faculty: str = "act.make.thing", kind: str = "success", stated=0.5,
         age_h: float = 1.0, settled_by: str = "world") -> SettledBetV1:
    at = NOW - timedelta(hours=age_h)
    return SettledBetV1(bet_id=f"t:{faculty}:{i}", faculty=faculty, source="test", made_at=at - timedelta(minutes=5),
                        settled_at=at, claim_kind=kind, stated=stated, outcome=outcome, settled_by=settled_by,
                        evidence_ref=f"t:{i}")


# ---- adapters, fed rows shaped like the live tables -------------------------------------------

def _act_row(**kw):
    row = {"act_id": "a1", "verb": "make", "object_ref": "journal_trigger:walkway_grade", "predicted": 0.39,
           "opened_at": NOW - timedelta(hours=2), "closed_at": NOW - timedelta(hours=1), "outcome_value": 0,
           "outcome_source": "producer", "outcome_note": ""}
    row.update(kw)
    return row


def test_act_faculty_keeps_a_named_trigger_and_drops_instance_ids() -> None:
    assert adapters.act_faculty("make", "journal_trigger:walkway_grade") == "act.make.journal_trigger.walkway_grade"
    assert adapters.act_faculty("play", "retro_session:6c359e66-96c3-4eb2-b1a5-b857ec81059c") == "act.play.retro_session"
    assert adapters.act_faculty("maintain", "topic_foundry_model:orion-hub-aitown-v1-0272029b:v1") == "act.maintain.topic_foundry_model"
    assert adapters.act_faculty("look", "https://www.deseret.com/politics/x/") == "act.look.url"
    assert adapters.act_faculty("say", "juniper") == "act.say.juniper"


def test_an_act_that_never_ran_is_failed_to_execute_not_refuted() -> None:
    for note in ("failed: RPC timeout waiting on orion:mind:llm:reply:x", "raised:RuntimeError:journal_compose_fail",
                 "abandoned:run_abandoned", "no_read_evidence:http_error"):
        assert adapters.grammar_act(_act_row(outcome_note=note)).outcome == "failed_to_execute", note
    assert adapters.grammar_act(_act_row(outcome_source="expired", outcome_note="no producer close")).outcome == "failed_to_execute"
    assert adapters.grammar_act(_act_row(outcome_note="no_read_evidence")).outcome == "refuted"
    assert adapters.grammar_act(_act_row(outcome_value=1, outcome_note="stage1_completed")).outcome == "upheld"
    assert adapters.grammar_act(_act_row(outcome_value=None)).outcome == "no_change"


def test_act_stated_is_the_prediction_made_at_open_and_juniper_settles_resolver_acts() -> None:
    bet = adapters.grammar_act(_act_row(outcome_source="resolver", outcome_value=1, verb="say", object_ref="juniper"))
    assert bet.stated == 0.39 and bet.settled_by == "juniper" and bet.claim_kind == "success"
    assert adapters.grammar_act(_act_row(closed_at=None)) is None


def test_turn_self_bets_map_status_and_void_reasons() -> None:
    base = {"bet_id": "c:turn.self.steps", "faculty": "turn.self.steps", "claim_kind": "interval",
            "placed_at": NOW - timedelta(hours=2), "settled_at": NOW, "stated_p": None, "coverage": 0.8,
            "void_reason": None}
    assert adapters.turn_self_bet({**base, "status": "broken"}).outcome == "refuted"
    assert adapters.turn_self_bet({**base, "status": "upheld"}).stated == 0.8
    assert adapters.turn_self_bet({**base, "status": "void", "void_reason": "failed_to_execute"}).outcome == "failed_to_execute"
    assert adapters.turn_self_bet({**base, "status": "void", "void_reason": "ungradeable_no_stated"}) is None
    p = {**base, "faculty": "turn.self.asks_question", "claim_kind": "probability", "stated_p": 0.07, "coverage": None}
    bet = adapters.turn_self_bet({**p, "status": "not_happened"})
    assert bet.claim_kind == "event" and bet.stated == 0.07 and bet.outcome == "refuted"


def test_forecast_uses_the_champion_80_percent_range_only() -> None:
    row = {"forecast_id": 7, "series_id": "athena.cabinet_temp_c", "role": "champion", "issued_at": NOW,
           "scored_at": NOW, "in_80": False}
    bet = adapters.forecast(row)
    assert bet.faculty == "forecast.athena.cabinet_temp_c" and bet.stated == 0.8 and bet.outcome == "refuted"
    assert adapters.forecast({**row, "role": "baseline"}) is None
    assert adapters.forecast({**row, "in_80": None}) is None


def test_action_bet_void_and_percept_open_rows_are_not_bets() -> None:
    ab = {"bet_id": "b", "domain": "retro:zelda", "skill": "traverse", "p": 0.7, "placed_at": NOW, "settled_at": NOW}
    assert adapters.action_bet({**ab, "result": "success"}).faculty == "body.retro:zelda.traverse"
    assert adapters.action_bet({**ab, "result": "void"}) is None
    pe = {"expectation_id": "e", "emitted_at": NOW, "scored_at": NOW, "confidence": 0.83}
    assert adapters.percept_expectation({**pe, "status": "met"}).outcome == "upheld"
    assert adapters.percept_expectation({**pe, "status": "open"}) is None


# ---- reducer --------------------------------------------------------------------------------

def test_walkway_grade_case_is_broken_not_wrong() -> None:
    # Live 10-28..11-03: 22 of 28 walkway_grade runs never executed; the 6 that ran all worked.
    bets = [_bet(i, "failed_to_execute", age_h=i * 5) for i in range(22)] + \
           [_bet(100 + i, "upheld", age_h=i * 20 + 2) for i in range(6)]
    rec = reduce_faculty("act.make.thing", bets, now=NOW)
    assert rec.candidate_status == "broken"
    assert rec.n_failed_to_execute == 22 and rec.n_graded == 6
    assert rec.reliability == 1.0  # what ran, worked: the failure is execution, not judgment


def test_a_short_failure_streak_alone_marks_broken() -> None:
    bets = [_bet(i, "upheld", age_h=48 + i) for i in range(30)] + \
           [_bet(100 + i, "failed_to_execute", age_h=i * 0.1) for i in range(BROKEN_STREAK)]
    rec = reduce_faculty("act.make.thing", bets, now=NOW)
    assert rec.streak_kind == "failed_to_execute" and rec.streak_len == BROKEN_STREAK
    assert rec.candidate_status == "broken"


def test_unreliable_miscalibrated_and_reliable() -> None:
    fail = [_bet(i, "refuted", stated=0.1) for i in range(20)] + [_bet(99, "upheld", stated=0.1)]
    assert reduce_faculty("f", fail, now=NOW).candidate_status == "unreliable"

    narrow = [_bet(i, "upheld" if i < 2 else "refuted", kind="interval", stated=0.8) for i in range(40)]
    rec = reduce_faculty("f", narrow, now=NOW)
    assert rec.candidate_status == "miscalibrated" and "coverage" in rec.reasons[0]

    good = [_bet(i, "upheld" if i % 5 else "refuted", kind="interval", stated=0.8) for i in range(100)]
    assert reduce_faculty("f", good, now=NOW).candidate_status == "reliable"


def test_too_few_graded_bets_is_unknown_with_a_reason() -> None:
    rec = reduce_faculty("f", [_bet(i, "upheld") for i in range(MIN_GRADED - 1)], now=NOW)
    assert rec.candidate_status == "unknown" and rec.gradeability == "ungradeable:too_few"


def test_self_settled_bets_never_count_as_calibration_evidence() -> None:
    rec = reduce_faculty("f", [_bet(i, "upheld", settled_by="self") for i in range(50)], now=NOW)
    assert rec.n_graded == 0 and rec.gradeability == "ungradeable:self_graded"


def test_no_stated_confidence_is_named() -> None:
    rec = reduce_faculty("f", [_bet(i, "upheld", stated=None) for i in range(50)], now=NOW)
    assert rec.gradeability == "ungradeable:no_stated_confidence" and rec.candidate_status == "unknown"


def test_old_failures_fade_so_a_healed_faculty_recovers() -> None:
    old = [_bet(i, "refuted", age_h=6 * 24 + i * 0.1) for i in range(30)]
    fresh = [_bet(100 + i, "upheld", age_h=i * 0.5) for i in range(30)]
    rec = reduce_faculty("f", old + fresh, now=NOW)
    assert rec.reliability > 0.75  # raw share is 0.5; the decay favours the recent half
    gone = reduce_faculty("f", [_bet(i, "refuted", age_h=8 * 24) for i in range(30)], now=NOW)
    assert gone.n_bets == 0  # outside the window entirely: no permanent zero


def test_each_bet_counts_once_and_catalog_entries_yield_to_real_bets() -> None:
    bet = _bet(1, "upheld", faculty="reverie.expectation")
    records = reduce_bets([bet, bet], now=NOW, ungradeable=UNGRADEABLE)
    rev = next(r for r in records if r.faculty == "reverie.expectation")
    assert rev.n_bets == 1 and rev.sources == ["test"]
    dream = next(r for r in records if r.faculty == "dream.hypothesis")
    assert dream.gradeability == "ungradeable:self_graded" and dream.unlock


def test_status_changes_only_after_the_candidate_holds() -> None:
    good = reduce_faculty("f", [_bet(i, "upheld", stated=0.97) for i in range(40)], now=NOW)
    held = advance(None, good, now=NOW)
    assert held.status == "reliable"
    bad = reduce_faculty("f", [_bet(i, "refuted") for i in range(40)], now=NOW)
    for k in range(1, HOLD):
        held = advance(held, bad, now=NOW + timedelta(hours=k))
        assert held.status == "reliable", k
    held = advance(held, bad, now=NOW + timedelta(hours=HOLD))
    assert held.status == "unreliable" and held.status_since == NOW + timedelta(hours=HOLD)
    flip = advance(held, good, now=NOW + timedelta(hours=HOLD + 1))  # one good hour does not flip back
    assert flip.status == "unreliable" and flip.candidate_held == 1


def test_render_lists_broken_first_and_names_source_errors() -> None:
    bets = [_bet(i, "failed_to_execute", faculty="act.make.x") for i in range(12)] + \
           [_bet(100 + i, "upheld", faculty="act.look.url") for i in range(40)]
    state = SelfMonitorStateV1(computed_at=NOW, window_days=7, faculties=reduce_bets(bets, now=NOW),
                               source_errors=["vision_percept_expectation: OperationalError: x"])
    text = render(state).splitlines()
    assert text[2].startswith("broken") and "act.make.x" in text[2]
    assert text[-1].startswith("SOURCE ERROR vision_percept_expectation")


# ---- reader contract ------------------------------------------------------------------------

def test_view_is_unknown_when_absent_stale_future_or_garbage() -> None:
    from orion.self_monitor.view import self_monitor_view

    state = SelfMonitorStateV1(computed_at=NOW, window_days=7,
                               faculties=reduce_bets([_bet(i, "upheld", stated=0.97) for i in range(40)], now=NOW))
    raw = state.model_dump_json()
    fresh = self_monitor_view(raw, now=NOW + timedelta(minutes=30))
    assert fresh.known and fresh.status("act.make.thing") == "reliable"
    assert fresh.status("act.never.seen") == "unknown"
    assert self_monitor_view(None, now=NOW).reason == "absent"
    assert self_monitor_view("{not json", now=NOW).reason == "unparseable"
    stale = self_monitor_view(raw, now=NOW + timedelta(hours=4))
    assert not stale.known and stale.faculty("act.make.thing") is None  # stale changes nothing
    assert self_monitor_view(raw, now=NOW - timedelta(hours=1)).reason == "future_stamped"
    assert self_monitor_view(raw.encode(), now=NOW).known
