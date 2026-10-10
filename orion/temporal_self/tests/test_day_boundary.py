"""Day boundary (acceptance check 4): local midnight, carries, timestamp casts."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from orion.temporal_self import ReducerConfig, advance_clock, build_frame, close_day, drain_closed_days, fold, initial_state
from orion.temporal_self.day import as_utc, day_id_for, day_phase_for, day_window
from orion.temporal_self.sources import chat_turn, metacog_observation
from orion.temporal_self.tests.fixtures import ev, ticks

CFG = ReducerConfig()
A = "node:substrate.bus_synaptic"
DENVER = ZoneInfo("America/Denver")
# 23:58 local on 10-09 is 05:58Z on 10-10 (MDT, UTC-6).
LATE = datetime(2026, 10, 10, 5, 58, tzinfo=timezone.utc)
MIDNIGHT = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)


def _context_day_phase_label():
    """Pull the original helper out of orion/situational/context.py without importing it."""
    path = Path(__file__).resolve().parents[2] / "situational" / "context.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_day_phase_label")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["_day_phase_label"]


def test_day_phase_matches_time_context_minute_for_minute():
    original = _context_day_phase_label()
    base = datetime(2026, 10, 9, 0, 0, tzinfo=DENVER)
    for m in range(24 * 60):
        local = base + timedelta(minutes=m)
        assert day_phase_for(local) == original(local.hour, local.minute)


def test_day_id_is_local_date_and_agrees_with_time_context():
    assert day_id_for(LATE) == "2026-10-09"
    assert day_id_for(MIDNIGHT) == "2026-10-10"
    assert day_id_for(LATE) == LATE.astimezone(DENVER).strftime("%Y-%m-%d")


def test_day_window_is_half_open_and_dst_safe():
    start, end = day_window("2026-10-09")
    assert start == datetime(2026, 10, 9, 6, 0, tzinfo=timezone.utc) and end == MIDNIGHT
    s, e = day_window("2026-11-01")  # DST ends: a 25 hour day
    assert (e - s) == timedelta(hours=25)


def test_naive_and_text_columns_land_in_the_right_local_day():
    # Naive chat_history_log.created_at and metacog_trigger.timestamp are UTC; TEXT ISO too.
    naive_before = datetime(2026, 10, 10, 5, 59, 59)
    naive_after = datetime(2026, 10, 10, 6, 0, 1)
    assert chat_turn({"id": "1", "session_id": "s", "created_at": naive_before}).day_id == "2026-10-09"
    assert chat_turn({"id": "2", "session_id": "s", "created_at": naive_after}).day_id == "2026-10-10"
    m = metacog_observation({"id": "m", "severity": "critical", "trigger_timestamp": naive_after,
                             "timestamp": "2026-10-10T05:00:00+00:00"})
    assert m.day_id == "2026-10-10"  # the trigger's own time wins over the metacog text column
    text_only = metacog_observation({"id": "m2", "severity": "critical", "timestamp": "2026-10-10T05:59:00"})
    assert text_only.day_id == "2026-10-09"
    assert as_utc("2026-10-10 05:59:00+00").day == 10  # orion_biometrics_summary.timestamp shape


def test_attention_arc_open_at_2358_closes_at_midnight_and_continuation_is_carried():
    before = ticks([A] * 4, start=LATE)  # 05:58:00 .. 05:59:30Z
    after = ticks([A] * 3, start=MIDNIGHT + timedelta(seconds=30), prefix="N")
    s = fold(initial_state(), before + after, cfg=CFG)
    s, (day,) = drain_closed_days(s)
    (old,) = [a for a in day.arcs if a.kind == "attention"]
    assert old.closed_reason == "day_boundary" and old.ended_at == old.last_seen_at
    (new,) = [a for a in s.arcs.values() if a.kind == "attention"]
    assert new.day_id == "2026-10-10"
    assert new.carried_from_previous_day and new.carried_from_arc_id == old.arc_id
    assert old.arc_id in s.entered_day_with
    assert day.frame.day_id == "2026-10-09" and day.closed_at == MIDNIGHT


def test_concern_spans_days_as_a_carried_continuation():
    raised = ev("attention_loop_raised", "t1", LATE, subject="loop-1", table="attention_salience_trace", label="x")
    s = advance_clock(fold(initial_state(), events=[raised], cfg=CFG), MIDNIGHT + timedelta(minutes=1), CFG)
    s, (day,) = drain_closed_days(s)
    assert day.arcs[0].closed_reason == "day_boundary"
    (cont,) = s.arcs.values()
    assert cont.kind == "concern" and cont.status == "open" and cont.carried_from_previous_day
    (thread,) = build_frame(s, MIDNIGHT + timedelta(minutes=2), CFG).open_threads
    assert thread.carried_from_previous_day


def test_process_crossing_midnight_belongs_to_the_day_it_completed():
    cycle = ev("dream_cycle", "cy", LATE, ended=MIDNIGHT + timedelta(minutes=3), subject="cy", table="dream_cycle")
    s = fold(initial_state(), ticks([None], start=LATE - timedelta(minutes=1)) , [cycle], cfg=CFG)
    s, days = drain_closed_days(s)
    (arc,) = s.arcs.values()
    assert arc.day_id == "2026-10-10" and arc.began_at == LATE


def test_close_day_without_new_rows_and_restart_at_midnight():
    s = fold(initial_state(), ticks([A] * 4, start=LATE), cfg=CFG)
    s, day = close_day(s, "2026-10-09", CFG)
    assert day is not None and day.day_id == "2026-10-09"
    assert all(a.status == "closed" for a in day.arcs)
    # A restart right after midnight (state rebuilt from its checkpoint) folds on cleanly.
    s2 = type(s).model_validate_json(s.model_dump_json())
    s2 = fold(s2, ticks([A] * 3, start=MIDNIGHT + timedelta(seconds=40), prefix="P"), cfg=CFG)
    (arc,) = s2.arcs.values()
    assert arc.carried_from_previous_day


def test_concern_return_after_midnight_is_measured_from_the_real_last_raise():
    r1 = ev("attention_loop_raised", "t1", LATE, subject="loop-1", table="attention_salience_trace")
    r2 = ev("attention_loop_raised", "t2", MIDNIGHT + timedelta(minutes=10), subject="loop-1", table="attention_salience_trace")
    s = fold(initial_state(), events=[r1, r2], cfg=CFG)
    (cont,) = [a for a in s.arcs.values() if a.day_id == "2026-10-10"]
    assert cont.attention_returns == 0  # 12 minutes after the last raise is not a return


def test_closed_days_final_frame_shows_what_was_active_at_midnight():
    s = fold(initial_state(), ticks([A] * 4, start=LATE), cfg=CFG)
    s, day = close_day(s, "2026-10-09", CFG)
    assert day.frame.active_arc is not None and day.frame.active_arc.subject_ref == A
    assert day.arcs[0].closed_reason == "day_boundary"
