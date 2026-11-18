"""orion/hardware_watch/rules.py: every opening arm, every resolve arm, and the boundaries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.autonomy.thermal_gate import DEFAULT_ELEVATED_C
from orion.hardware_watch.rules import (
    Baseline, CoolingPoint, CoolingRuleConfig, HeatRuleConfig, TempPoint, cooling_verdict, heat_verdict,
    cabinet_rise_c, held_for, percentile, shed_verdict,
)

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


def series(start: float, end: float, step: float = 5.0, **kw) -> list[CoolingPoint]:
    out, t = [], start
    while t <= end:
        watts = kw["watts"](t) if callable(kw.get("watts")) else kw.get("watts", 850.0)
        out.append(CoolingPoint(at(t), watts, kw.get("stale", False), kw.get("online", True), kw.get("ready", True)))
        t += step
    return out


def varying(t: float) -> float:
    return 850.0 + (int(t) // 60) % 7   # a new value every minute, like the real meter


# --- held_for ---------------------------------------------------------------------------------

def test_held_for_counts_from_the_oldest_point_of_the_newest_run():
    pts = [TempPoint(at(0), 1), TempPoint(at(10), 5), TempPoint(at(20), 5)]
    assert held_for(pts, lambda p: p.value > 3, at(25), ts=lambda p: p.ts, max_gap_sec=60) == 15


def test_held_for_is_zero_when_the_newest_point_is_old():
    pts = [TempPoint(at(0), 5)]
    assert held_for(pts, lambda p: p.value > 3, at(100), ts=lambda p: p.ts, max_gap_sec=60) == 0


# --- cooling ----------------------------------------------------------------------------------

def test_healthy_ac_neither_opens_nor_blocks_resolve():
    v = cooling_verdict(series(0, 3900, watts=varying), at(3900))
    assert v.open_reason is None and v.resolve


def test_low_power_opens_after_three_minutes_not_before():
    pts = series(0, 600, watts=varying) + series(605, 780, watts=33.1)
    assert cooling_verdict(pts, at(780)).open_reason is None           # 175 s low
    pts += series(785, 790, watts=33.1)
    assert cooling_verdict(pts, at(790)).open_reason == "low_power"     # 185 s low


def test_stale_rows_for_five_minutes_open_no_fresh_sample():
    pts = series(0, 600, watts=varying) + series(605, 895, watts=None, stale=True)
    assert cooling_verdict(pts, at(895)).open_reason is None            # 295 s without a live reading
    v = cooling_verdict(pts + series(900, 900, watts=None, stale=True), at(900))   # 300 s
    assert v.open_reason == "no_fresh_sample" and not v.resolve


def test_offline_device_names_itself():
    pts = series(0, 600, watts=varying) + series(605, 1000, watts=850.0, online=False)
    assert cooling_verdict(pts, at(1000)).open_reason == "device_offline"


def test_controller_not_ready_names_itself():
    pts = series(0, 600, watts=varying) + series(605, 1000, watts=850.0, ready=False)
    assert cooling_verdict(pts, at(1000)).open_reason == "controller_not_ready"


def test_no_rows_at_all_opens_no_samples():
    assert cooling_verdict([], at(0)).open_reason == "no_samples"
    pts = series(0, 600, watts=varying)
    assert cooling_verdict(pts, at(1000)).open_reason == "no_samples"   # rows stopped 400 s ago


def test_pre_2382_rows_with_unknown_staleness_are_judged_live():
    pts = series(0, 3900, watts=varying, stale=None)
    assert cooling_verdict(pts, at(3900)).open_reason is None


def test_frozen_reading_opens_at_sixty_minutes():
    pts = series(0, 3595, watts=850.0)
    assert cooling_verdict(pts, at(3595)).open_reason is None
    pts = series(0, 3605, watts=850.0)
    assert cooling_verdict(pts, at(3605)).open_reason == "frozen"


def test_frozen_high_reading_does_not_resolve():
    """850 W identical for an hour satisfies >= 500 W for 10 min; resolve must still refuse."""
    v = cooling_verdict(series(0, 4000, watts=850.0), at(4000))
    assert v.open_reason == "frozen" and not v.resolve


def test_resolve_needs_ten_minutes_of_good_live_watts():
    pts = series(0, 600, watts=33.1) + series(605, 1200, watts=varying)
    assert not cooling_verdict(pts, at(1200)).resolve                   # 595 s good
    pts += series(1205, 1210, watts=varying)
    assert cooling_verdict(pts, at(1210)).resolve


def cycling(t: float) -> float:
    """Thermostat cycle seen live 2026-10-03: ~750 W compressor for 2 min, ~105 W idle for 2 min."""
    return 750.0 if int(t) % 240 < 120 else 105.0


def test_thermostat_cycling_ac_resolves():
    """The incident that shed every lane for 4 h: the AC cycled, never holding 500 W for 10 min."""
    pts = series(0, 900, watts=cycling)
    v = cooling_verdict(pts, at(900))
    assert v.open_reason is None and v.resolve


def test_dip_past_low_sec_stops_counting_as_working():
    """300 W is neither 'low' (<150) nor 'good' (>=500): it cannot open low_power, so cooling_ok_sec
    alone shows whether the dip still counts. The 180 s cutoff after the last >=500 W reading."""
    base = series(0, 100, watts=850.0)
    inside = cooling_verdict(base + series(105, 100 + 175, watts=300.0), at(100 + 175))
    outside = cooling_verdict(base + series(105, 100 + 190, watts=300.0), at(100 + 190))
    assert inside.open_reason is None and inside.detail["cooling_ok_sec"] > 0
    assert outside.open_reason is None and outside.detail["cooling_ok_sec"] == 0 and not outside.resolve


def test_resolve_refused_while_newest_live_is_too_old():
    pts = series(0, 1200, watts=varying)
    assert not cooling_verdict(pts, at(1200 + 130)).resolve


def test_custom_thresholds_are_used():
    cfg = CoolingRuleConfig(low_watts=900.0, low_sec=10)
    assert cooling_verdict(series(0, 60, watts=850.0), at(60), cfg).open_reason == "low_power"


# --- heat -------------------------------------------------------------------------------------

BASE = Baseline(p75=62.0, p95=73.0, n=20000, history_sec=7 * 86400)


def temps(start, end, value, step=30.0):
    out, t = [], start
    while t <= end:
        out.append(TempPoint(at(t), value(t) if callable(value) else value))
        t += step
    return out


def test_cpu_opens_above_p95_after_ten_minutes():
    assert heat_verdict(temps(0, 570, 75.0), at(570), BASE).open_reason is None
    assert heat_verdict(temps(0, 600, 75.0), at(600), BASE).open_reason == "above_p95"


def test_equal_to_p95_is_not_above():
    assert heat_verdict(temps(0, 1200, 73.0), at(1200), BASE).open_reason is None


def test_cpu_resolves_below_p75_only():
    assert not heat_verdict(temps(0, 60, 65.0), at(60), BASE).resolve
    assert heat_verdict(temps(0, 60, 61.0), at(60), BASE).resolve


def test_p95_arm_waits_for_minimum_history():
    young = Baseline(62.0, 73.0, 100, history_sec=3600)
    v = heat_verdict(temps(0, 1200, 80.0), at(1200), young)
    assert v.open_reason is None and v.detail["armed"] is False


def test_gpu_ceiling_is_always_armed():
    cfg = HeatRuleConfig(min_history_sec=3 * 86400, ceiling_c=85.0, ceiling_sustain_sec=120, ceiling_rearm_c=80.0)
    none = Baseline(None, None, 0, 0.0)
    assert heat_verdict(temps(0, 90, 86.0), at(90), none, cfg).open_reason is None
    assert heat_verdict(temps(0, 120, 86.0), at(120), none, cfg).open_reason == "above_ceiling"
    assert not heat_verdict(temps(0, 60, 81.0), at(60), none, cfg).resolve
    assert heat_verdict(temps(0, 60, 79.0), at(60), none, cfg).resolve


def test_silent_sensor_neither_opens_nor_resolves():
    v = heat_verdict(temps(0, 60, 50.0), at(1000), BASE)
    assert v.open_reason is None and not v.resolve


def test_percentile_matches_postgres_percentile_cont():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10
    assert percentile([], 0.5) is None
    assert percentile(list(range(101)), 0.95) == 95


# --- shed -------------------------------------------------------------------------------------

def test_shed_on_rise_of_one_degree_in_fifteen_minutes():
    pts = temps(0, 900, lambda t: 26.0 + t / 900.0)   # 26 -> 27 over 15 min
    v = shed_verdict(pts, at(900))
    assert v.requested and v.reason == "cabinet_rising" and v.rise_c == 1.0


def test_no_shed_when_cool_and_flat():
    v = shed_verdict(temps(0, 900, 26.0), at(900))
    assert not v.requested and v.reason is None


def test_shed_at_thermal_gate_elevated_without_a_rise():
    v = shed_verdict(temps(0, 900, DEFAULT_ELEVATED_C), at(900))
    assert v.requested and v.reason == "cabinet_elevated"


def test_unreadable_cabinet_sheds():
    assert shed_verdict([], at(0)).reason == "cabinet_unreadable"
    assert shed_verdict(temps(0, 60, 25.0), at(60 + 301)).reason == "cabinet_unreadable"


# --- cabinet_rise_c: the shared rise function (reflex 1.0 C, learned action 0.5 C) ------------

def test_cabinet_rise_is_newest_minus_window_minimum():
    pts = temps(0, 900, lambda t: 26.0 + t / 1800.0)   # 26 -> 26.5 over 15 min
    assert cabinet_rise_c(pts, at(900), 900) == 0.5
    # the same series under the reflex's 1.0 C threshold does not shed; the caller owns thresholds
    assert shed_verdict(pts, at(900)).reason is None


def test_cabinet_rise_ignores_readings_outside_the_window_and_is_never_negative():
    pts = temps(0, 1800, lambda t: 30.0 - t / 1800.0)   # falling 30 -> 29
    assert cabinet_rise_c(pts, at(1800), 900) == 0.0
    early_low = [TempPoint(at(0), 20.0)] + temps(1000, 1800, 26.0)
    assert cabinet_rise_c(early_low, at(1800), 900) == 0.0


def test_cabinet_rise_none_without_readings_in_window():
    assert cabinet_rise_c([], at(0)) is None
    assert cabinet_rise_c(temps(0, 60, 25.0), at(2000), 900) is None


def test_shed_verdict_reports_the_shared_rise():
    pts = temps(0, 900, lambda t: 26.0 + t / 900.0)
    assert shed_verdict(pts, at(900)).rise_c == cabinet_rise_c(pts, at(900), 900)
