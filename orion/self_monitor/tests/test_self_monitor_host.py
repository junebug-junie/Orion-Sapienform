"""Patch 2 pure seams: the shared read-and-adapt path, the compact hysteresis carry, and one evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.self_monitor import report
from orion.self_monitor.collect import SOURCES, build_state, erred_sources, psycopg_sql
from orion.self_monitor.host import FACULTY_KEEPALIVE_SEC, evaluate, event_row, mark_recorded
from orion.self_monitor.record import HOLD, advance, advance_carry, carry_of

NOW = datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def rec(faculty: str, candidate: str, *, now: datetime = NOW, sources=("grammar_acts",)) -> FacultyRecordV1:
    return FacultyRecordV1(
        faculty=faculty, sources=list(sources), gradeability="graded", window_start=now - timedelta(days=7),
        window_end=now, n_bets=12, n_graded=12, n_failed_to_execute=0, n_expired=0, weight_graded=8.0,
        candidate_status=candidate, status=candidate, candidate_held=1, reasons=[candidate],
    )


def state(*records: FacultyRecordV1, errors=(), now: datetime = NOW) -> SelfMonitorStateV1:
    return SelfMonitorStateV1(computed_at=now, window_days=7.0, faculties=list(records), source_errors=list(errors))


# --- one query/adapt path -------------------------------------------------------------------------


def test_report_and_host_share_one_source_list_and_psycopg_text_differs_only_in_the_parameter():
    assert report.SOURCES is SOURCES
    for _, sql, _ in SOURCES:
        assert sql.count(":since") == 1
        assert psycopg_sql(sql) == sql.replace(":since", "%(since)s") and ":since" not in psycopg_sql(sql)


def test_build_state_names_failed_missing_and_bad_sources():
    rows = [{"forecast_id": f"f{i}", "series_id": "athena.cabinet_temp_c", "role": "champion",
             "issued_at": NOW - timedelta(hours=i + 1), "scored_at": NOW - timedelta(hours=i), "in_80": True}
            for i in range(3)]
    bad = [{"bet_id": "x", "settled_at": NOW, "result": "success"}]   # no placed_at: the adapter raises
    out = build_state({"prediction_forecast": rows, "grammar_acts": RuntimeError("boom"),
                       "prediction_action_bet": bad, "prediction_turn_self_bet": [],
                       "prediction_turn_run": []},
                      now=NOW, window_days=7.0)
    errs = "\n".join(out.source_errors)
    assert "grammar_acts: RuntimeError: boom" in errs
    assert "vision_percept_expectation: not read" in errs
    assert "prediction_action_bet: adapter failed on 1/1 rows" in errs
    assert {f.faculty: f.n_bets for f in out.faculties}["forecast.athena.cabinet_temp_c"] == 3
    assert erred_sources(out) == {"grammar_acts", "vision_percept_expectation", "prediction_action_bet"}
    assert erred_sources(state(errors=["engine: OperationalError: refused"])) == {n for n, _, _ in SOURCES}


# --- compact carry ---------------------------------------------------------------------------------


def test_advance_carry_matches_advance_on_whole_records():
    held = advance(None, rec("a", "reliable"), now=NOW)
    via_record, via_carry = held, held
    for k, cand in enumerate(["broken", "broken", "reliable", "broken", "broken", "broken"], start=1):
        t = NOW + k * HOUR
        via_record = advance(via_record, rec("a", cand, now=t), now=t)
        via_carry = advance_carry(json.loads(json.dumps(carry_of(via_carry))), rec("a", cand, now=t), now=t)
        assert (via_record.status, via_record.status_since, via_record.candidate_held) == (
            via_carry.status, via_carry.status_since, via_carry.candidate_held)
    assert via_carry.status == "broken" and via_carry.status_since == NOW + 6 * HOUR


def test_advance_carry_treats_an_unparseable_carry_as_a_first_evaluation():
    out = advance_carry({"status": "nonsense", "status_since": "not a date"}, rec("a", "broken"), now=NOW)
    assert out.status == "broken" and out.status_since == NOW and out.candidate_held == HOLD


# --- one evaluation --------------------------------------------------------------------------------


def run(candidate: SelfMonitorStateV1, carry, now: datetime, *, ok: bool = True):
    ev = evaluate(candidate, carry, now=now)
    return ev, (mark_recorded(ev.carry, ev.changes, now=now) if ok else ev.carry)


def test_first_sight_records_quiet_statuses_silently_and_writes_notable_ones():
    ev, carry = run(state(rec("a", "reliable"), rec("b", "broken"), rec("c", "unknown")), None, NOW)
    assert [(c.faculty, c.from_status, c.to_status) for c in ev.changes] == [("b", None, "broken")]
    assert {k: v["recorded"] for k, v in carry.items()} == {"a": "reliable", "b": "broken", "c": "unknown"}
    assert set(carry["a"]) == {"status", "status_since", "candidate_status", "candidate_held", "sources",
                               "recorded", "row_at"}


def test_held_change_writes_once_then_nothing_until_the_keepalive():
    _, carry = run(state(rec("a", "reliable")), None, NOW)
    rows = 0
    for k in range(1, HOLD + 4):
        t = NOW + k * HOUR
        ev, carry = run(state(rec("a", "unreliable", now=t), now=t), carry, t)
        rows += len(ev.changes)
        if k == HOLD:
            assert [(c.from_status, c.to_status, c.keepalive) for c in ev.changes] == [("reliable", "unreliable", False)]
    assert rows == 1
    t = NOW + HOLD * HOUR + timedelta(seconds=FACULTY_KEEPALIVE_SEC)
    ev, _ = run(state(rec("a", "unreliable", now=t), now=t), carry, t)
    assert [(c.to_status, c.keepalive) for c in ev.changes] == [("unreliable", True)]


def test_a_quiet_change_still_writes_a_row():
    _, carry = run(state(rec("a", "broken")), None, NOW)
    for k in range(1, HOLD + 1):
        t = NOW + k * HOUR
        ev, carry = run(state(rec("a", "reliable", now=t), now=t), carry, t)
    assert [(c.from_status, c.to_status) for c in ev.changes] == [("broken", "reliable")]


def test_unwritten_rows_leave_the_carry_unrecorded():
    ev, carry = run(state(rec("b", "broken")), None, NOW, ok=False)
    assert carry["b"]["recorded"] is None and len(ev.changes) == 1
    ev2, _ = run(state(rec("b", "broken", now=NOW + HOUR), now=NOW + HOUR), carry, NOW + HOUR)
    assert [(c.from_status, c.to_status) for c in ev2.changes] == [(None, "broken")]


def test_dropped_faculty_kept_while_its_source_erred_then_closed():
    _, carry = run(state(rec("b", "broken")), None, NOW)
    ev, carry2 = run(state(errors=["grammar_acts: timeout"], now=NOW + HOUR), carry, NOW + HOUR)
    assert ev.changes == [] and carry2["b"] == carry["b"]
    ev, carry3 = run(state(now=NOW + 2 * HOUR), carry2, NOW + 2 * HOUR)
    assert [(c.from_status, c.to_status, c.dropped) for c in ev.changes] == [("broken", "unknown", True)]
    assert "b" not in carry3
    _, quiet = run(state(rec("a", "reliable")), None, NOW)
    ev, after = run(state(now=NOW + HOUR), quiet, NOW + HOUR)
    assert ev.changes == [] and after == {}            # a quiet faculty leaves without a row


def test_event_row_ids_are_deterministic_and_typed():
    _, carry = run(state(rec("a", "reliable")), None, NOW)
    t = NOW + HOLD * HOUR
    ev = evaluate(state(rec("a", "broken", now=t), now=t),
                  {"a": {**carry["a"], "candidate_status": "broken", "candidate_held": HOLD - 1}}, now=t)
    (change,) = ev.changes
    row = event_row(change, day_id="2026-11-03", now=t)
    assert row["event_id"] == f"faculty_status_changed:a:{t.isoformat()}:broken"
    assert row["occurred_at"] == t and row["label"] == "broken" and row["subject_ref"] == "faculty:a"
    assert event_row(change, day_id="2026-11-03", now=t) == row
    payload = json.loads(row["payload"])
    assert payload["from"] == "reliable" and payload["n_graded"] == 12 and payload["keepalive"] is False


def test_a_retried_change_row_is_stamped_at_its_write_time_so_the_chronicle_still_reads_it():
    """Review finding: a row stamped at ``since`` and retried an hour later sat behind the chronicle's
    watermark (now - 300 s) and never folded. Every row is stamped when written."""
    ev, carry = run(state(rec("b", "broken")), None, NOW, ok=False)
    t = NOW + HOUR
    ev2 = evaluate(state(rec("b", "broken", now=t), now=t), carry, now=t)
    (change,) = ev2.changes
    row = event_row(change, day_id="2026-11-03", now=t)
    assert row["occurred_at"] == t and json.loads(row["payload"])["since"] == NOW.isoformat()
    assert row["event_id"] == event_row(ev.changes[0], day_id="2026-11-03", now=NOW)["event_id"]


def test_a_faculty_whose_source_erred_keeps_its_held_status_and_counts_nothing():
    _, carry = run(state(rec("a", "reliable")), None, NOW)
    for k in range(1, HOLD + 2):
        t = NOW + k * HOUR
        ev, carry = run(state(rec("a", "broken", now=t), errors=["grammar_acts: timeout"], now=t), carry, t)
        assert ev.state.faculties[0].status == "reliable" and ev.changes == []
    assert carry["a"]["candidate_held"] == HOLD and carry["a"]["candidate_status"] == "reliable"


def test_a_retried_drop_row_keeps_its_id():
    _, carry = run(state(rec("b", "broken")), None, NOW)
    ids = []
    for k in (1, 2):
        t = NOW + k * HOUR
        ev = evaluate(state(now=t), carry, now=t)
        ids.append(event_row(ev.changes[0], day_id="2026-11-03", now=t)["event_id"])
    assert ids[0] == ids[1]
