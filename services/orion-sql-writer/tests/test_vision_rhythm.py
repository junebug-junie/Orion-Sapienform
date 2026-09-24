"""Pure-function tests for the rhythm learner."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.vision_rhythm import (
    arrivals_from_windows,
    coverage_fraction,
    circular_distance,
    circular_kde,
    expect_key_ttl,
    expectation_id,
    fit_subject,
    minute_of_day,
    outcome_narrative,
    overlaps,
    plan_expectations,
    score_window,
)

TZ = ZoneInfo("America/Denver")


def _local(d: date, h: int, m: int) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, tzinfo=TZ).astimezone(timezone.utc)


def _days(n, start=date(2026, 9, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def test_no_expectation_under_five_days_of_support() -> None:
    occ = [_local(d, 7, 40) for d in _days(4)] * 3  # 12 occurrences, 4 days
    assert fit_subject(occ, tz=TZ, day_kind="any") == []
    occ5 = [_local(d, 7, 40) for d in _days(5)]
    fits = fit_subject(occ5, tz=TZ, day_kind="any")
    assert len(fits) == 1 and abs(fits[0].peak_minute - (7 * 60 + 40)) <= 1
    assert fits[0].support_days == 5 and fits[0].confidence == 1.0


def test_confidence_uses_days_the_camera_watched() -> None:
    occ = [_local(d, 7, 40) for d in _days(5)]
    watched = set(_days(10))
    (f,) = fit_subject(occ, tz=TZ, day_kind="any", observed_days=watched, min_confidence=0.4)
    assert f.confidence == 0.5
    assert fit_subject(occ, tz=TZ, day_kind="any", observed_days=watched, min_confidence=0.6) == []


def test_window_needs_its_own_five_days_of_support() -> None:
    # Six visits, three of them near 18:00 by coincidence: not a rhythm.
    days = _days(6)
    occ = [_local(days[0], 15, 21), _local(days[1], 18, 0), _local(days[2], 13, 52),
           _local(days[3], 18, 1), _local(days[4], 15, 33), _local(days[5], 18, 22)]
    assert fit_subject(occ, tz=TZ, day_kind="any", min_confidence=0.0) == []


def test_day_kind_filters_occurrences() -> None:
    days = _days(14)  # 2026-09-01 is a Tuesday
    occ = [_local(d, 7, 40) for d in days if d.weekday() < 5]
    assert fit_subject(occ, tz=TZ, day_kind="weekday")
    assert fit_subject(occ, tz=TZ, day_kind="weekend") == []


def test_circular_wrap_across_midnight() -> None:
    assert circular_distance(1430, 10) == 20
    days = _days(6)
    occ = [_local(d, 23, 55) for d in days[::2]] + [_local(d, 0, 5) for d in days[1::2]]
    (f,) = fit_subject(occ, tz=TZ, day_kind="any", min_confidence=0.5)
    # One peak at midnight, not two peaks 23h50m apart.
    assert circular_distance(f.peak_minute, 0) <= 5
    assert f.start_offset_min < 0 < f.end_offset_min or f.end_offset_min > 1440 or f.start_offset_min < 1440 < f.end_offset_min


def test_kde_is_symmetric_around_the_wrap() -> None:
    d = circular_kde([0], 20)
    assert d[10] == d[1430] and d[0] > d[10]


def test_plan_emits_only_future_windows_within_24h_and_wraps_midnight() -> None:
    occ = [_local(d, 0, 5) for d in _days(6)]
    fits = {"any": fit_subject(occ, tz=TZ, day_kind="any")}
    now = _local(date(2026, 9, 10), 12, 0)
    plan = plan_expectations(fits, now=now, tz=TZ)
    assert len(plan) == 1
    p = plan[0]
    assert now < p.window_start <= now + timedelta(hours=24)
    assert p.window_start.astimezone(TZ).date() in (date(2026, 9, 10), date(2026, 9, 11))
    assert p.window_start < _local(date(2026, 9, 11), 0, 5) < p.window_end


def test_plan_prefers_specific_day_kind_model() -> None:
    days = _days(21)
    weekday = fit_subject([_local(d, 7, 0) for d in days if d.weekday() < 5], tz=TZ, day_kind="weekday")
    anyf = fit_subject([_local(d, 12, 0) for d in days], tz=TZ, day_kind="any")
    now = _local(date(2026, 9, 21), 5, 0)  # Monday 05:00
    plan = plan_expectations({"weekday": weekday, "weekend": [], "any": anyf}, now=now, tz=TZ)
    assert plan and all(p.fit.day_kind == "weekday" for p in plan)


def test_any_model_is_not_applied_to_a_kind_of_day_that_breaks_it() -> None:
    days = _days(10)  # Tue 9/1 .. Thu 9/10: 8 weekdays, 2 weekend days
    occ = [_local(d, 7, 40) for d in days if d.weekday() < 5]
    watched = set(days)
    fits = {dk: fit_subject(occ, tz=TZ, day_kind=dk, observed_days=watched) for dk in ("weekday", "weekend", "any")}
    assert fits["any"] and fits["any"][0].weekend_rate == 0.0
    saturday_eve = _local(date(2026, 9, 11), 12, 0)  # Friday noon -> next 24h is Saturday morning
    assert plan_expectations(fits, now=saturday_eve, tz=TZ) == []
    # Without any watched weekend day, the "any" model is still allowed.
    only_weekdays = {d for d in days if d.weekday() < 5}
    fits2 = {"any": fit_subject(occ, tz=TZ, day_kind="any", observed_days=only_weekdays)}
    assert fits2["any"][0].weekend_rate is None
    assert plan_expectations(fits2, now=saturday_eve, tz=TZ)


def test_scoring_met_missed_unscorable() -> None:
    assert score_window(occurred=True, coverage=0.0) == "met"
    assert score_window(occurred=False, coverage=0.95) == "missed"
    assert score_window(occurred=False, coverage=0.0) == "unscorable"
    # One frame in a long window is not "the camera was watching".
    assert score_window(occurred=False, coverage=0.3) == "unscorable"


def test_coverage_bridges_the_live_census_cadence_but_not_outages() -> None:
    start = datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)
    t0 = start.timestamp()
    # Live shape: 5 s windows every 10 s.
    live = [(t0 + i * 10, t0 + i * 10 + 5) for i in range(180)]
    assert coverage_fraction(live, start, end) == 1.0
    # Same cadence but only the first 10 minutes (camera died).
    assert 0.3 < coverage_fraction(live[:60], start, end) < 0.4
    assert coverage_fraction([], start, end) == 0.0
    # A single census row.
    assert coverage_fraction([(t0 + 60, t0 + 65)], start, end) < 0.05


def test_arrivals_are_debounced() -> None:
    t = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    windows = [(t, 1), (t + timedelta(seconds=10), 0), (t + timedelta(seconds=20), 1),  # flicker
               (t + timedelta(seconds=30), 0), (t + timedelta(seconds=1000), 2)]
    assert arrivals_from_windows(windows, gap_sec=300) == [t, t + timedelta(seconds=1000)]


def test_expectation_id_is_idempotent() -> None:
    ws = datetime(2026, 9, 1, 13, 30, tzinfo=timezone.utc)
    assert expectation_id("walkway", "label:vehicle", ws) == expectation_id("walkway", "label:vehicle", ws)
    assert expectation_id("walkway", "label:vehicle", ws) != expectation_id("walkway", "label:package", ws)


def test_overlap_dedupes_refit_windows() -> None:
    a = datetime(2026, 9, 1, 13, 30, tzinfo=timezone.utc)
    assert overlaps(a + timedelta(minutes=5), a + timedelta(minutes=40), [(a, a + timedelta(minutes=30))])
    assert not overlaps(a + timedelta(minutes=30), a + timedelta(minutes=40), [(a, a + timedelta(minutes=30))])


def test_expect_key_ttl_only_for_open_unmet_windows() -> None:
    now = datetime(2026, 9, 1, 13, 40, tzinfo=timezone.utc)
    ws, we = now - timedelta(minutes=10), now + timedelta(minutes=20)
    assert expect_key_ttl([(ws, we, False)], now) == 1200
    assert expect_key_ttl([(ws, we, True)], now) is None
    assert expect_key_ttl([(now + timedelta(minutes=5), we, False)], now) is None


def test_narratives_are_plain() -> None:
    ws = _local(date(2026, 9, 1), 7, 30)
    we = _local(date(2026, 9, 1), 7, 55)
    miss = outcome_narrative(status="missed", subject_label="dog #a1b2c3", stream_id="walkway", window_start=ws,
                             window_end=we, confidence=0.8, support_days=9, tz=TZ, coverage=0.93)
    assert "07:30-07:55" in miss and "did not come" in miss and "93% of that window" in miss
    met = outcome_narrative(status="met", subject_label="mail truck", stream_id="walkway", window_start=ws,
                            window_end=we, confidence=0.8, support_days=9, tz=TZ,
                            arrived_at=_local(date(2026, 9, 1), 7, 41))
    assert "came at 07:41" in met


def test_minute_of_day_is_local() -> None:
    assert minute_of_day(datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc), TZ) == 8 * 60
