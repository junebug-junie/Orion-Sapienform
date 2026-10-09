"""Boundary fixtures for three offline reports; no live services required."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


focus = load("focus_report", "services/orion-attention-runtime/evals/replay_focus_runs.py")
dream = load("dream_report", "scripts/analysis/measure_dream_pressure_crossings.py")
arousal = load("arousal_report", "scripts/analysis/measure_arousal_replay.py")
T = datetime(2026, 10, 8, tzinfo=timezone.utc)


def run(n, target, start, end, ticks=10, censored=False):
    return dict(run_id=str(n), target_id=target, started_at=(T+timedelta(seconds=start)).isoformat(),
                ended_at=(T+timedelta(seconds=end)).isoformat(), tick_count=ticks, left_censored=censored)


def test_focus_midnight_censoring_gap_and_tick_denominators():
    rows = [run(1, "node:substrate.biometrics", -60, 60, censored=True),
            run(2, "node:substrate.execution", 120, 240, ticks=60),
            run(3, "node:substrate.execution", 300, 600, ticks=150)]
    result = focus.exposure(rows, T-timedelta(seconds=30), T+timedelta(seconds=400))
    assert result["unaccounted_seconds"] == 120
    assert result["other_winner_runs"] == 2
    assert result["full_window_winner_ticks"] == 60  # never prorate clipped ticks
    assert result["other_tick_share"] == 1
    assert result["targets"]["node:substrate.execution"]["median_complete_seconds"] == 120
    assert result["targets"]["node:substrate.biometrics"]["median_complete_seconds"] is None
    assert result["days"][0]["wall_span_seconds"]["node:substrate.biometrics"] == 30
    assert result["days"][1]["share_of_utc_day"]["node:substrate.biometrics"] == 60/86400


def test_focus_empty_and_downtime_are_not_idle_or_activity():
    empty = focus.exposure([], T, T+timedelta(days=1))
    assert empty["other_tick_share"] is None
    assert empty["unaccounted_seconds"] == 86400
    result = focus.exposure([run(1, "other", 0, 3600, ticks=2)], T, T+timedelta(hours=1))
    assert result["targets"]["other"]["wall_span_seconds"] == 3600
    assert result["full_window_winner_ticks"] == 2
    assert result["rolling_30_minutes"][0]["share_of_window"]["other"] == 1
    assert "downtime" in result["caveats"][0]


@pytest.mark.parametrize("rows", [[run(1, "a", 0, 60)]*2,
                                  [run(1, "a", 0, 60), run(2, "b", 30, 90)],
                                  [run(1, "a", 60, 0)]])
def test_focus_rejects_duplicate_overlap_and_negative_span(rows):
    with pytest.raises(ValueError):
        focus.exposure(rows, T, T+timedelta(hours=1))


def cycle(n, hours, pressure=100, trigger="pressure", status="completed", novelty=False):
    reading = dict(pressure=pressure, threshold=3, counts={"metacog": 50}, idle_minutes=60)
    if novelty:
        reading["new_counts"] = {"metacog": 1}
    return dict(cycle_id=str(n), started_at=(T+timedelta(hours=hours)).isoformat(),
                ended_at=(T+timedelta(hours=hours, seconds=10)).isoformat(),
                trigger=trigger, status=status, reading=reading)


def test_dream_attempt_end_clock_manual_failed_and_formula_split():
    rows = [cycle(0, -6, status="failed"), cycle(1, 10/3600),
            cycle(2, 1, trigger="manual"), cycle(3, 7+10/3600, pressure=1, novelty=True)]
    report = dream.report(rows, T, T+timedelta(days=1))
    assert report["by_formula"]["legacy"]["comparable_automatic_cycles"] == 1
    assert report["cycles"][0]["timer_aligned"]
    assert report["cycles"][2]["timer_aligned"]  # manual attempt resets refractory too
    assert report["cycles"][2]["legacy_ceiling"] is None
    assert report["by_formula"]["novelty"]["below_threshold_at_cycle"] == 1
    assert report["independent_check_samples"] == 0
    assert report["verdict"].startswith("UNVERIFIED")


def test_dream_cycle_variation_is_not_evidence_of_discharge():
    rows = [cycle(1, 0, pressure=150), cycle(2, 6.01, pressure=20), cycle(3, 12.02, pressure=140)]
    result = dream.report(rows, T, T+timedelta(days=1))
    assert result["legacy_timing_consistent_with_timer"]
    assert result["cycles"][0]["fraction_of_legacy_ceiling"] == 150/175
    assert result["verdict"].startswith("UNVERIFIED")
    assert dream.report([], T, T+timedelta(days=1))["by_formula"]["legacy"]["min_pressure"] is None


def test_dream_rejects_nonfinite_pressure():
    with pytest.raises(ValueError):
        dream.report([cycle(1, 0, pressure=float("nan"))], T, T+timedelta(days=1))


def test_manual_low_pressure_does_not_disprove_automatic_timer_pattern():
    rows = [cycle(1, 0, pressure=1, trigger="manual"), cycle(2, 6+10/3600)]
    result = dream.report(rows, T, T+timedelta(days=1))
    assert result["by_formula"]["legacy"]["below_threshold_at_cycle"] == 1
    assert result["by_formula"]["legacy"]["comparable_automatic_below_threshold"] == 0
    assert result["legacy_timing_consistent_with_timer"]


def check(seconds, pressure, *, errors=None, last_start=T, check_id=None):
    return dict(kind="pressure_check", observation=dict(
        check_id=check_id or str(seconds), observed_at=(T+timedelta(seconds=seconds)).isoformat(),
        source_errors=errors or [], formula="novelty.v1", forced=False,
        last_window_start=last_start.isoformat(), last_attempt_end=(T+timedelta(seconds=10)).isoformat(),
        min_interval_hours=6, check_interval_sec=600,
        reading=dict(pressure=pressure, threshold=3, idle_minutes=60, idle_required_minutes=45)))


def test_real_check_curve_needs_matched_sleep_and_unbroken_fall_then_rise():
    rows = [cycle(1, 0, pressure=4, novelty=True),
            check(0, 4, last_start=T-timedelta(hours=7)), check(600, 0), check(1200, 1)]
    result = dream.report(rows, T, T+timedelta(days=1))
    assert result["check_history"]["observed_fall_and_rise"]
    assert result["check_history"]["zero_samples"] == 1
    assert result["independent_check_samples"] == 3
    rows.insert(3, check(900, 0, errors=["current:metacog"]))
    broken = dream.report(rows, T, T+timedelta(days=1))
    assert not broken["check_history"]["observed_fall_and_rise"]
    assert broken["check_history"]["source_failed_samples"] == 1


def test_failed_dream_or_cadence_gap_cannot_prove_discharge_and_recovery():
    rows = [cycle(1, 0, status="failed", novelty=True),
            check(0, 4, last_start=T-timedelta(hours=7)), check(600, 0), check(1200, 1)]
    assert not dream.report(rows, T, T+timedelta(days=1))["check_history"]["observed_fall_and_rise"]
    rows[0]["status"] = "completed"
    rows[-1] = check(4000, 2)
    curve = dream.report(rows, T, T+timedelta(days=1))["check_history"]
    assert not curve["observed_fall_and_rise"] and curve["cadence_gaps"] == 1


def test_low_pressure_can_hold_an_otherwise_eligible_check():
    rows = [check(7*3600, 1)]
    result = dream.report(rows, T, T+timedelta(days=1))
    assert result["check_history"]["idle_timer_clear_below_threshold"] == 1
    assert "dream_pressure_observation" in dream.export_sql(T, T+timedelta(days=1), with_checks=True)


def test_arousal_saved_gpu_snapshot_export_is_host_scoped_and_stale_stays_unknown():
    sql = arousal.export_sql(T, T+timedelta(days=1), gpu_host="athena")
    assert "gpu_pool_state_history WHERE host = 'athena'" in sql
    with pytest.raises(ValueError):
        arousal.export_sql(T, T+timedelta(days=1), gpu_host="athena'; DELETE")
    rows = [dict(kind="chat", at=T.isoformat(), juniper=True),
            dict(kind="heat", at=T.isoformat(), temp_c=27),
            dict(kind="gpu_state", at=T.isoformat(), host="athena", backlog_depth={})]
    result = arousal.replay(rows, T, T+timedelta(seconds=30))
    assert result["totals"]["strict_seconds"] == dict(engaged=20, idle=0, strained=0, unknown=10)
    rows.append(dict(kind="gpu_state", at=T.isoformat(), host="circe", backlog_depth={}))
    with pytest.raises(ValueError):
        arousal.replay(rows, T, T+timedelta(seconds=30))


def test_arousal_backlog_sustain_clear_hysteresis_and_chat_boundary():
    c = arousal.Classifier()
    def step(seconds, hot=False, backlog=True, last_turn=T):
        return c.step(T+timedelta(seconds=seconds), hot=hot, backlog=backlog, last_turn=last_turn)
    assert step(0) == "engaged"
    assert step(299) == "engaged"
    assert step(300) == "strained"
    assert step(305, backlog=False) == "strained"
    assert step(904, backlog=False) == "strained"
    assert step(905, backlog=False) == "engaged"
    assert step(2700, backlog=False) == "idle"


def test_arousal_stale_interrupts_sustain_and_never_earns_clear_time():
    c = arousal.Classifier()
    def step(s, hot=False, backlog=False):
        return c.step(T+timedelta(seconds=s), hot=hot, backlog=backlog, last_turn=T)
    assert step(0, hot=True, backlog=None) == "strained"
    assert step(5, hot=None, backlog=None) == "unknown"
    assert step(700) == "strained"  # cannot claim 10 clear minutes during an outage
    assert step(1300) == "engaged"
    assert step(1310, backlog=True) == "engaged"
    assert step(1600, backlog=None) == "unknown"
    assert step(1610, backlog=True) == "engaged"


def test_arousal_hourly_missing_gpu_outreach_exclusion_and_focus_context():
    rows = [dict(kind="chat", at=(T-timedelta(hours=2)).isoformat(), juniper=True),
            dict(kind="chat", at=T.isoformat(), juniper=False),
            dict(kind="outreach", at=T.isoformat()),
            dict(kind="focus", at=T.isoformat(), ended_at=(T+timedelta(minutes=30)).isoformat(), target_id="bio")]
    rows += [dict(kind="heat", at=(T+timedelta(seconds=s)).isoformat(), temp_c=27)
             for s in range(0, 3600, 30)]
    result = arousal.replay(rows, T, T+timedelta(hours=1))
    hour = result["hours"][0]
    assert hour["strict_seconds"]["unknown"] == 3600
    assert hour["provisional_seconds"]["idle"] == 3600
    assert hour["all_chat_provisional_seconds"]["engaged"] == 2700
    assert hour["outreach_sent"] == 1
    assert hour["focus_wall_span_seconds"] == {"bio": 1800}


def test_arousal_fresh_heat_wins_even_when_other_inputs_missing():
    result = arousal.replay([dict(kind="heat", at=T.isoformat(), temp_c=35)], T, T+timedelta(minutes=2))
    assert result["totals"]["strict_seconds"] == dict(engaged=0, idle=0, strained=95, unknown=25)


def test_arousal_partial_hour_boundaries_and_future_turn_no_lookahead():
    start = T+timedelta(minutes=59, seconds=58)
    end = T+timedelta(hours=1, seconds=7)
    rows = [dict(kind="chat", at=(T+timedelta(hours=1)).isoformat(), juniper=True),
            dict(kind="heat", at=start.isoformat(), temp_c=27),
            dict(kind="gpu_state", at=start.isoformat(), backlog_depth={})]
    result = arousal.replay(rows, start, end)
    assert [h["seconds"] for h in result["hours"]] == [2, 7]
    assert result["hours"][0]["strict_seconds"]["unknown"] == 2
    assert result["hours"][1]["strict_seconds"]["engaged"] == 7


@pytest.mark.parametrize("module", [focus, dream, arousal])
def test_export_queries_enforce_readonly_and_bounded_windows(module):
    sql = module.export_sql(T, T+timedelta(days=1))
    assert sql.startswith("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;")
    assert "statement_timeout" in sql
    assert sql.endswith("ROLLBACK;")
    assert "2026-10-09" in sql
    assert not any(word in sql.upper().split() for word in ["INSERT", "UPDATE", "DELETE", "CREATE"])
    with pytest.raises(ValueError):
        module.export_sql(T, T)
    with pytest.raises(ValueError):
        module.export_sql("2026-10-08", T)
