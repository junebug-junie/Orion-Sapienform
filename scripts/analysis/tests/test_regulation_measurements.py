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


# --- R1a: focus hogging (PR #2369 rev 4) ------------------------------------------------

def hours(h):
    return h * 3600


def test_r1a_empty_table_is_no_data_not_calm():
    result = focus.r1a_report([], T, T+timedelta(days=1))
    assert result["verdict"].startswith("NO_DATA")
    assert result["hog_stretches"] == [] and result["share_runs_reaching_min_streak"] is None
    assert result["unaccounted_hours"] == 24


def test_r1a_three_hour_hold_is_a_hog_but_only_baseline_before_the_instrument_week():
    rows = [run(1, "A", 0, hours(3), ticks=5400), run(2, "B", hours(3), hours(3.5), ticks=900)]
    result = focus.r1a_report(rows, T, T+timedelta(hours=4))
    assert result["hog_stretches"][0]["target_id"] == "A"
    assert result["hog_stretches"][0]["seconds"] > hours(2)
    assert result["verdict"].startswith("BASELINE_ONLY")
    week = focus.r1a_report(rows, T, T+timedelta(days=8), instrument_since=T)
    assert week["verdict"].startswith("BUILD_R1")


def test_r1a_even_alternation_and_downtime_never_make_a_hog():
    rows = [run(i, "A" if i % 2 else "B", i*60, (i+1)*60, ticks=30) for i in range(240)]
    result = focus.r1a_report(rows, T, T+timedelta(days=8), instrument_since=T)
    assert result["hog_stretches"] == []
    assert result["verdict"].startswith("DO_NOT_BUILD_R1")
    # A wins every recorded second, but the recorder is off half of each window.
    gappy = [run(i, "A", i*1800, i*1800+840, ticks=420) for i in range(12)]
    assert focus.r1a_report(gappy, T, T+timedelta(hours=6))["hog_stretches"] == []


def test_r1a_left_censored_run_counts_as_span_not_complete_duration_and_stuck_runs_listed():
    rows = [run(1, "A", 0, hours(5), ticks=9000, censored=True)]
    result = focus.r1a_report(rows, T, T+timedelta(hours=6))
    assert result["complete_run_seconds"]["n"] == 0
    assert result["stuck_runs"][0]["target_id"] == "A"
    # Share series is above 50% from the window ending 0:30 to the one ending 5:10.
    assert result["longest_majority_stretch_seconds_by_target"]["A"] == hours(4) + 45*60


def test_r1a_return_window_merges_same_target_runs_only_within_r():
    rows = [run(1, "A", 0, 60), run(2, "B", 60, 100), run(3, "A", 100, 160), run(4, "A", 1900, 1960)]
    probe = {p["return_minutes"]: p for p in focus.r1a_report(rows, T, T+timedelta(hours=1))["return_window_probe"]}
    assert probe[1]["arcs_by_target"]["A"] == 2 and probe[1]["returns_by_target"]["A"] == 1
    assert probe[30]["arcs_by_target"]["A"] == 1 and probe[30]["returns_by_target"]["A"] == 2


# --- R2a: dream pressure replay with the service's own builders ---------------------------

def src(kind, seconds, key, weight_field=None, *, has_text=True, ident=None):
    return dict(kind="source", source_kind=kind, at=(T+timedelta(seconds=seconds)).isoformat(),
                id=ident or f"{kind}-{seconds}", severity=weight_field, trigger_kind=None,
                dedupe_key=key, has_text=has_text)


def test_r2a_pressure_matches_service_keyed_candidates_and_compute_pressure():
    svc = dream._service_replay()
    rows = [src("metacog", 10, "k1", "degraded"), src("metacog", 20, "k1", "critical"),
            src("compaction_request", 30, "theme"), src("resonance", 40, "loop", "3"),
            src("crystallization", 50, "c1", "0.7"), src("metacog", 60, "k2", "degraded", has_text=False)]
    items = dream.prepare_items(rows)
    value, counts, new = dream.pressure_at(items, [x[0] for x in items], T+timedelta(minutes=5), T)
    raw = {"metacog": [dict(id="m", severity="critical", summary="x", dedupe_key="k1")],
           "compaction_request": [dict(request_id="r", theme="x", dedupe_key="theme")],
           "resonance": [dict(alert_id="a", theme_key="x", violation_count=3, dedupe_key="loop")],
           "crystallization": [dict(crystallization_id="c1", subject="x", summary="x", salience=0.7, dedupe_key="c1")]}
    assert (value, counts, new) == svc.compute_pressure(svc.keyed_candidates(raw), ())
    assert value == 1.0 + 0.5 + 0.6 + 0.7


def test_r2a_prior_keys_hold_pressure_and_rejected_rows_still_count_as_seen():
    rows = [src("metacog", -hours(10), "old", "critical", has_text=False),
            src("metacog", 60, "old", "critical"), src("metacog", 120, "new", "critical")]
    items = dream.prepare_items(rows)
    value, counts, new = dream.pressure_at(items, [x[0] for x in items], T+timedelta(minutes=5), T)
    assert value == 1.0 and counts == {"metacog": 2} and new == {"metacog": 1}


def r2a_rows(sources, chat_seconds=()):
    sleep = dict(cycle_id="c0", started_at=(T-timedelta(hours=7)).isoformat(),
                 ended_at=(T-timedelta(hours=7)+timedelta(minutes=5)).isoformat(),
                 trigger="pressure", status="completed", reading=dict(pressure=4, threshold=3))
    chat = [dict(kind="chat", at=(T+timedelta(seconds=s)).isoformat(), juniper=True) for s in chat_seconds]
    return [sleep] + sources + chat


def test_r2a_empty_sources_read_zero_and_unknown_idle_never_sleeps():
    result = dream.replay_novelty(r2a_rows([]), T, T+timedelta(hours=12), thresholds=(3,))
    window = result["sleep_windows"][0]
    assert window["max_pressure"] == 0 and window["zero_checks"] == window["checks"]
    assert window["hours_to_first_crossing"] == {"3": None}
    # One new thing an hour, but no chat row ever: idle is unknown, so no sleep.
    sources = [src("metacog", hours(h), f"k{h}", "critical") for h in range(12)]
    sim = dream.replay_novelty(r2a_rows(sources), T, T+timedelta(hours=12), thresholds=(3,))
    assert sim["simulated_schedules"]["all_chat_idle"][0]["sleeps"] == 0


def test_r2a_slow_accrual_makes_a_sleep_wait_past_the_clock():
    sources = [src("metacog", hours(h), f"k{h}", "critical") for h in range(0, 30, 2)]
    result = dream.replay_novelty(r2a_rows(sources, chat_seconds=[-hours(8)]), T, T+timedelta(hours=30),
                                  thresholds=(3,), formula_boundary=T-timedelta(days=1))
    sim = result["simulated_schedules"]["all_chat_idle"][0]
    assert sim["sleeps"] >= 2 and sim["sleeps_later_than_clock"] >= 1
    assert sim["timer_clear_checks_held_below_threshold"] > 0
    assert result["sleep_windows"][0]["era"] == "after_2557"
    assert result["sleep_windows"][0]["hours_to_first_crossing"]["3"] > 7


def test_r2a_validates_against_saved_checks():
    sources = [src("metacog", 60, "k1", "critical")]
    rows = r2a_rows(sources) + [check(600, 1.0, last_start=T-timedelta(hours=7)),
                                check(1200, 0.4, last_start=T-timedelta(hours=7))]
    result = dream.replay_novelty(rows, T, T+timedelta(hours=1))
    assert result["validation"] == dict(saved_checks_compared=2, exact=1, max_abs_diff=0.6)


def test_r2a_export_mirrors_the_service_source_queries_and_carries_no_text():
    store = (ROOT / "services/orion-dream/app/cycle_store.py").read_text()
    assert f'METACOG_KEY_RE = r"{dream.METACOG_KEY_RE}"' in store
    for fragment in ["severity IN ('degraded', 'critical')", "h.op IN ('auto_activate', 'approve')",
                     "c.status = 'active'", "DISTINCT ON (lower(theme))", "DISTINCT ON (lower(theme_key))"]:
        assert fragment in store
    sql = dream.export_sql(T, T+timedelta(days=1), with_sources=True)
    assert "h.op IN ('auto_activate', 'approve') AND c.status = 'active'" in sql
    assert "2026-10-04" in sql  # two 48 h lookbacks before the start
    # UNION ALL takes column names from the first branch; every branch projects
    # ids, numbers, md5 keys and booleans only, never theme/summary/subject text.
    for projection in [
            "SELECT 'source' AS kind, 'metacog' AS source_kind, ts AS at, id::text AS id, severity, trigger_kind,\n"
            "        md5(lower(dedupe_key)) AS dedupe_key, has_text FROM (",
            "SELECT 'source', 'compaction_request', created_at, request_id::text, NULL, NULL,\n"
            "        md5(lower(theme)), btrim(coalesce(theme, '')) <> ''",
            "SELECT 'source', 'resonance', created_at, alert_id::text, violation_count::text, NULL,\n"
            "        md5(lower(theme_key)), btrim(coalesce(theme_key, '')) <> ''",
            "SELECT 'source', 'crystallization', h.created_at, c.crystallization_id::text, c.salience::text, NULL,\n"
            "        md5(c.crystallization_id::text),"]:
        assert projection in sql
    assert sql.count("SELECT 'source'") == 4


def test_saved_checks_record_an_upward_threshold_crossing():
    rows = [check(0, 2), check(600, 3.5)]
    curve = dream.report(rows, T, T+timedelta(days=1))["check_history"]
    assert curve["upward_threshold_crossing_count"] == 1


# --- R3a: arousal summary --------------------------------------------------------------------

def test_arousal_summary_ties_and_all_unknown_never_read_as_idle(tmp_path):
    assert arousal.hour_label(dict(engaged=0, idle=1800, strained=1800, unknown=0)) == "unknown"
    assert arousal.hour_label(dict(engaged=0, idle=0, strained=0, unknown=0)) == "unknown"
    result = arousal.replay([], T, T+timedelta(hours=2))
    bar = result["summary"]["acceptance_bar"]["strict_seconds"]
    assert bar["result"] == "NO_KNOWN_TIME" and bar["unknown_hours"] == 2
    assert result["summary"]["hour_label_counts"]["strict_seconds"] == {"unknown": 2}
    path = tmp_path / "hourly.csv"
    arousal.write_hourly_csv(result, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and lines[1].split(",")[2] == "unknown"


def test_arousal_counts_degenerate_backlog_and_strict_strain_exits():
    rows = [dict(kind="chat", at=T.isoformat(), juniper=True)]
    rows += [dict(kind="heat", at=(T+timedelta(seconds=s)).isoformat(), temp_c=35 if s < 60 else 27)
             for s in range(0, 1800, 30)]
    rows += [dict(kind="gpu_state", at=(T+timedelta(seconds=s)).isoformat(), host="circe", backlog_depth={})
             for s in range(0, 1800, 5)]
    result = arousal.replay(rows, T, T+timedelta(minutes=30))
    assert result["gpu_state_nonzero_backlog_snapshots"] == 0
    assert result["strict_strain_exits_per_day"] == {"2026-10-08": 1}
    bar = result["summary"]["acceptance_bar"]["strict_seconds"]
    assert bar["result"].startswith("FAIL")  # idle never reached in 30 minutes
