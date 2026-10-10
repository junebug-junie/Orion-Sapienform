"""Rest drive eval (scripts/analysis/measure_rest_drive_easing.py): fixtures, no live services."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("rest_drive_easing", ROOT / "scripts/analysis/measure_rest_drive_easing.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

T = datetime(2026, 10, 8, tzinfo=timezone.utc)


def at(seconds):
    return (T + timedelta(seconds=seconds)).isoformat()


def check(seconds, pressure, *, last_end_s=-7 * 3600, errors=None, counts=None):
    return dict(kind="pressure_check", observation=dict(
        check_id=f"dp-{seconds}", observed_at=at(seconds), source_errors=errors or [], formula="novelty.v1",
        forced=False, trigger="pressure", last_window_start=at(last_end_s - 10), last_attempt_end=at(last_end_s),
        min_interval_hours=6, check_interval_sec=600, lookback_hours=48,
        reading=dict(since=at(last_end_s - 10), computed_at=at(seconds), pressure=pressure, threshold=3,
                     counts=counts if counts is not None else {"metacog": 1}, new_counts={},
                     idle_minutes=60, idle_required_minutes=45)))


def consumer(name, seconds):
    return dict(kind="consumer", consumer=name, at=at(seconds))


def test_saved_lane_classifies_like_the_producer():
    rows = [check(0, 0.0), check(600, 2.0), check(1200, 3.3), check(1800, 3.3, errors=["current:metacog"]),
            check(2400, 9.0, last_end_s=2300)]
    states = [r.state for _, r in m.saved_timeline(rows)]
    assert states == ["resting", "building", "due", "no_reading", "refractory"]


def test_tired_minutes_hold_until_next_check_and_never_past_staleness():
    rows = [check(0, 3.3), check(600, 1.0), check(3600, 3.3)]  # last one: no next check
    minutes, states = m.tired_minutes_per_day(m.saved_timeline(rows))
    assert minutes == {"2026-10-08": 20.0}  # 10 + the trailing 10-minute check slot
    # A due reading followed by a 2 h gap counts 30 minutes, not 2 h.
    minutes, _ = m.tired_minutes_per_day(m.saved_timeline([check(0, 3.3), check(7200, 0.0)]))
    assert minutes == {"2026-10-08": 30.0}


def test_held_events_need_tired_and_the_stretched_window():
    timeline = m.saved_timeline([check(0, 0.0), check(10000, 3.3), check(20000, 3.3), check(30000, 0.0)])
    events = [T + timedelta(seconds=s) for s in (5000, 10500, 21000, 21500, 22000)]
    out = m.held_events(timeline, events, base_sec=2700, multiplier=2.0)
    # 5000: rested. 10500: tired, gap 5500 >= 5400 -> not held. 21000: tired, gap 10500 -> not held.
    # 21500: tired, gap 500 < base -> the plain cooldown, not the drive.
    # 22000: 2000 s after the last reading -> stale -> unknown, not tired.
    assert out["while_tired"] == 3 and out["held"] == 0
    out = m.held_events(timeline, [T + timedelta(seconds=s) for s in (7000, 10500)], base_sec=2700, multiplier=2.0)
    assert out["held"] == 1 and out["examples"][0]["gap_sec"] == 3500


def test_unknown_states_never_count_as_tired():
    timeline = m.saved_timeline([check(0, 3.3, errors=["prior:resonance"])])
    assert m.tired_minutes_per_day(timeline)[0] == {"2026-10-08": 0.0}
    assert m.held_events(timeline, [T + timedelta(seconds=10), T + timedelta(seconds=3000)],
                         base_sec=2700, multiplier=2.0)["while_tired"] == 0


def test_export_stays_inside_one_read_only_transaction():
    sql = m.export_sql(T, T + timedelta(days=1))
    assert sql.startswith("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;")
    assert sql.rstrip().endswith("ROLLBACK;") and sql.count("ROLLBACK;") == 1
    assert "curiosity_offer_decisions" in sql and "turn_started_at" in sql and "endogenous_outreach_decisions" in sql
    assert "NOT forced" in sql and "curiosity_outreach" in sql


def test_a_sleep_end_cuts_the_tired_hold_like_the_post_sleep_publish():
    timeline = m.saved_timeline([check(0, 3.3), check(600, 0.0, last_end_s=130)])
    sleep_ends = [T + timedelta(seconds=130)]
    minutes, _ = m.tired_minutes_per_day(timeline, sleep_ends=sleep_ends)
    assert minutes["2026-10-08"] == 2.2  # 130 s, cut at the sleep end, not the 600 s slot
    # A send after the sleep ended is not "while tired", even within the 600 s slot.
    out = m.held_events(timeline, [T - timedelta(seconds=3000), T + timedelta(seconds=300)],
                        base_sec=2700, multiplier=2.0, sleep_ends=sleep_ends)
    assert out["while_tired"] == 0 and out["held"] == 0


def test_door_a_sends_reset_the_clock_but_are_never_counted():
    timeline = m.saved_timeline([check(0, 3.3)])
    events = [(T - timedelta(seconds=4000), False), (T - timedelta(seconds=3500), True), (T + timedelta(seconds=10), False)]
    out = m.held_events(timeline, events, base_sec=2700, multiplier=2.0)
    # Door-A at -3500 resets the clock: gap 3510 -> held; the Door-A row itself is not counted.
    assert out["events"] == 2 and out["held"] == 1 and out["examples"][0]["gap_sec"] == 3510
