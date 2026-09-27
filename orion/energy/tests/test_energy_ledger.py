from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.energy.testing import hourly, make_test_ledger
from orion.schemas.energy import EnergyUsageIntervalV1

ROOT = Path(__file__).resolve().parents[3]
TARIFF = load_tariff(ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
DENVER = ZoneInfo("America/Denver")
M = 1.071 * 1.0518
B1S, B2S = 0.098332 * M, 0.125263 * M
NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _iv(start: datetime, kwh: float, *, retrieved: datetime = NOW, point: str = "UP123") -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop",
        usage_point_id=point,
        interval_start=start,
        interval_end=start + timedelta(hours=1),
        energy_kwh=kwh,
        retrieved_at=retrieved,
    )


def _span(
    start: datetime,
    end: datetime,
    kwh: float,
    *,
    retrieved: datetime = NOW,
    point: str = "UP123",
) -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop",
        usage_point_id=point,
        interval_start=start,
        interval_end=end,
        energy_kwh=kwh,
        retrieved_at=retrieved,
    )


def _ledger(day: int = 1) -> UsageLedger:
    return UsageLedger(TARIFF, tz=DENVER, cycle_start_day=day)


def _backfill(led: UsageLedger, ts: datetime, point: str = "UP123") -> None:
    """Contiguous zero-kWh span from the cycle start containing ts up to ts."""
    cycle_start, _ = led.cycle_bounds(ts)
    led.upsert(_span(cycle_start, ts, 0.0, point=point))


def test_newer_retrieval_replaces_older_is_ignored() -> None:
    led = _ledger()
    t = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    assert led.upsert(_iv(t, 1.0, retrieved=NOW)) is True
    assert led.upsert(_iv(t, 2.0, retrieved=NOW + timedelta(days=1))) is True
    assert led.upsert(_iv(t, 9.0, retrieved=NOW - timedelta(days=1))) is False
    assert led.intervals_overlapping("UP123", t, t + timedelta(hours=1))[0].energy_kwh == 2.0


def test_equal_retrieved_at_replaces() -> None:
    led = _ledger()
    t = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    assert led.upsert(_iv(t, 1.0, retrieved=NOW)) is True
    assert led.upsert(_iv(t, 5.0, retrieved=NOW)) is True
    assert led.intervals_overlapping("UP123", t, t + timedelta(hours=1))[0].energy_kwh == 5.0


def test_cycle_bounds_mid_month_start_day() -> None:
    led = _ledger(day=15)
    start, end = led.cycle_bounds(datetime(2026, 9, 10, 12, tzinfo=DENVER))
    assert (start, end) == (datetime(2026, 8, 15, tzinfo=DENVER), datetime(2026, 9, 15, tzinfo=DENVER))
    start, _ = led.cycle_bounds(datetime(2026, 9, 15, 0, 30, tzinfo=DENVER))
    assert start == datetime(2026, 9, 15, tzinfo=DENVER)


def test_cycle_bounds_dst_end_november() -> None:
    led = _ledger(day=1)
    ts = datetime(2026, 11, 15, 12, tzinfo=DENVER)
    start, end = led.cycle_bounds(ts)
    assert start == datetime(2026, 11, 1, tzinfo=DENVER)
    assert end == datetime(2026, 12, 1, tzinfo=DENVER)

    # Nov 1 00:00–04:00 local spans the DST fall-back (5 UTC hours: 06:00Z..11:00Z).
    cycle_start_utc = datetime(2026, 11, 1, 6, tzinfo=timezone.utc)
    for i in range(5):
        led.upsert(_iv(cycle_start_utc + timedelta(hours=i), 1.0))
    query_at = datetime(2026, 11, 1, 11, tzinfo=timezone.utc)
    kwh, as_of = led.cycle_kwh_before("UP123", query_at)
    assert kwh == pytest.approx(5.0)
    assert as_of == query_at


def test_cycle_start_day_clamps_to_short_month() -> None:
    led = _ledger(day=31)
    start, end = led.cycle_bounds(datetime(2026, 3, 5, tzinfo=DENVER))
    assert start == datetime(2026, 2, 28, tzinfo=DENVER)
    assert end == datetime(2026, 3, 31, tzinfo=DENVER)


def test_accrual_crosses_block_boundary_in_order() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    _backfill(led, base)
    for i in range(3):
        led.upsert(_iv(base + timedelta(hours=i), 200.0))
    start, _ = led.cycle_bounds(base)
    rows = led.accrue_cycle("UP123", start, computed_at=NOW)
    priced = [r for r in rows if r.energy_kwh > 0]
    assert [r.interval_cost_usd for r in priced] == pytest.approx([200 * B1S, 200 * B1S, 200 * B2S])
    assert priced[-1].cycle_accumulated_kwh == pytest.approx(600.0)
    assert priced[-1].cycle_to_date_total_usd == pytest.approx(400 * B1S + 200 * B2S + 12.16)
    assert priced[0].marginal_usd_per_kwh == pytest.approx(B1S)
    assert priced[1].marginal_usd_per_kwh == pytest.approx(B2S)  # 400 used: next kWh is block 2
    assert priced[0].cycle_start == date(2026, 7, 1)
    assert led.interval_cost_usd("UP123", base) == pytest.approx(200 * B1S)


def test_late_interval_reprices_later_ones() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    later_at = base + timedelta(hours=1)
    _backfill(led, base)
    led.upsert(_iv(base, 100.0))
    led.upsert(_iv(later_at, 200.0))
    start, _ = led.cycle_bounds(base)
    led.accrue_cycle("UP123", start, computed_at=NOW)
    assert led.interval_cost_usd("UP123", later_at) == pytest.approx(200 * B1S)
    led.upsert(_iv(base, 300.0, retrieved=NOW + timedelta(days=1)))
    led.accrue_cycle("UP123", start, computed_at=NOW)
    assert led.interval_cost_usd("UP123", later_at) == pytest.approx(100 * B1S + 100 * B2S)


def test_cycle_kwh_before_counts_only_finished_intervals_in_cycle() -> None:
    led = _ledger()
    t = datetime(2026, 9, 10, 16, tzinfo=timezone.utc)
    assert led.cycle_kwh_before("UP123", t) is None
    led.upsert(_iv(datetime(2026, 8, 31, 5, tzinfo=timezone.utc), 50.0))  # previous cycle (Denver Aug 30)
    assert led.cycle_kwh_before("UP123", t) is None
    _backfill(led, t - timedelta(hours=2))
    led.upsert(_iv(t - timedelta(hours=2), 100.0))
    led.upsert(_iv(t - timedelta(hours=1), 50.0))
    led.upsert(_iv(t, 7.0))  # starts at t: not finished before t
    kwh, as_of = led.cycle_kwh_before("UP123", t)
    assert kwh == pytest.approx(150.0)
    assert as_of == t


def test_cycle_kwh_before_none_when_data_starts_after_cycle_start() -> None:
    led = _ledger()
    t = datetime(2026, 9, 10, 16, tzinfo=timezone.utc)
    led.upsert(_iv(t - timedelta(hours=2), 100.0))
    assert led.cycle_kwh_before("UP123", t) is None


def test_cycle_kwh_before_none_on_mid_cycle_hole() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    first = cycle_start
    second = cycle_start + timedelta(hours=2)
    t = second + timedelta(hours=1)
    led.upsert(_span(cycle_start, first + timedelta(hours=1), 10.0))
    led.upsert(_iv(second, 20.0))
    assert led.cycle_kwh_before("UP123", t) is None


def test_cycle_kwh_before_returns_after_hole_filled() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    hole_start = cycle_start + timedelta(hours=1)
    second = cycle_start + timedelta(hours=2)
    t = second + timedelta(hours=1)
    led.upsert(_span(cycle_start, hole_start, 10.0))
    led.upsert(_iv(second, 20.0))
    assert led.cycle_kwh_before("UP123", t) is None
    led.upsert(_iv(hole_start, 5.0))
    kwh, as_of = led.cycle_kwh_before("UP123", t)
    assert kwh == pytest.approx(35.0)
    assert as_of == second + timedelta(hours=1)


def test_accrue_cycle_stops_at_hole() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    hole_start = cycle_start + timedelta(hours=1)
    second = cycle_start + timedelta(hours=2)
    led.upsert(_span(cycle_start, hole_start, 10.0))
    led.upsert(_iv(second, 20.0))
    rows = led.accrue_cycle("UP123", cycle_start, computed_at=NOW)
    assert len(rows) == 1
    assert rows[0].energy_kwh == pytest.approx(10.0)
    assert led.interval_cost_usd("UP123", second) is None


def test_accrue_cycle_includes_rows_after_hole_filled() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    hole_start = cycle_start + timedelta(hours=1)
    second = cycle_start + timedelta(hours=2)
    led.upsert(_span(cycle_start, hole_start, 10.0))
    led.upsert(_iv(second, 20.0))
    led.accrue_cycle("UP123", cycle_start, computed_at=NOW)
    led.upsert(_iv(hole_start, 5.0))
    rows = led.accrue_cycle("UP123", cycle_start, computed_at=NOW)
    assert len(rows) == 3
    assert led.interval_cost_usd("UP123", second) == pytest.approx(rows[-1].interval_cost_usd)


def test_upsert_clears_stale_interval_cost_before_reaccrue() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    _backfill(led, base)
    led.upsert(_iv(base, 100.0))
    start, _ = led.cycle_bounds(base)
    led.accrue_cycle("UP123", start, computed_at=NOW)
    assert led.interval_cost_usd("UP123", base) is not None
    led.upsert(_iv(base, 150.0))
    assert led.interval_cost_usd("UP123", base) is None


def test_accrue_cycle_ignores_other_usage_point() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    led.upsert(_span(cycle_start, cycle_start + timedelta(hours=1), 10.0, point="UP9"))
    led.upsert(_span(cycle_start, cycle_start + timedelta(hours=1), 50.0))
    rows = led.accrue_cycle("UP123", cycle_start, computed_at=NOW)
    assert len(rows) == 1
    assert rows[0].energy_kwh == pytest.approx(50.0)


def test_interval_before_cycle_start_excluded_from_accrual() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    before = cycle_start - timedelta(hours=1)
    led.upsert(_iv(before, 99.0))
    led.upsert(_span(cycle_start, cycle_start + timedelta(hours=1), 10.0))
    rows = led.accrue_cycle("UP123", cycle_start, computed_at=NOW)
    assert len(rows) == 1
    assert rows[0].energy_kwh == pytest.approx(10.0)
    assert led.interval_cost_usd("UP123", before) is None


def test_all_cycles_and_usage_points() -> None:
    led = _ledger()
    led.upsert(_iv(datetime(2026, 7, 2, 18, tzinfo=timezone.utc), 1.0))
    led.upsert(_iv(datetime(2026, 8, 2, 18, tzinfo=timezone.utc), 1.0, point="UP9"))
    assert led.usage_points() == {"UP123", "UP9"}
    assert led.all_cycles() == {
        ("UP123", datetime(2026, 7, 1, tzinfo=DENVER)),
        ("UP9", datetime(2026, 8, 1, tzinfo=DENVER)),
    }


def test_cycle_coverage_contiguous_prefix() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    led.upsert(_span(cycle_start, cycle_start + timedelta(hours=2), 10.0))
    led.upsert(_iv(cycle_start + timedelta(hours=2), 20.0))
    priced, total, covered = led.cycle_coverage("UP123", cycle_start)
    assert (priced, total) == (2, 2)
    assert covered == cycle_start + timedelta(hours=3)


def test_cycle_coverage_stops_at_hole() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    hole_start = cycle_start + timedelta(hours=1)
    second = cycle_start + timedelta(hours=2)
    led.upsert(_span(cycle_start, hole_start, 10.0))
    led.upsert(_iv(second, 20.0))
    priced, total, covered = led.cycle_coverage("UP123", cycle_start)
    assert (priced, total) == (1, 2)
    assert covered == hole_start


def test_cycle_coverage_empty_when_starts_late() -> None:
    led = _ledger()
    cycle_start, _ = led.cycle_bounds(datetime(2026, 9, 10, tzinfo=DENVER))
    late = cycle_start + timedelta(hours=5)
    led.upsert(_iv(late, 1.0))
    priced, total, covered = led.cycle_coverage("UP123", cycle_start)
    assert (priced, total) == (0, 1)
    assert covered is None


def test_reupsert_reaccrue_two_year_ledger_under_five_seconds() -> None:
    led = _ledger()
    cycle_start = datetime(2024, 9, 1, tzinfo=DENVER)
    start = cycle_start.astimezone(timezone.utc)
    retrieved = datetime(2026, 9, 1, tzinfo=timezone.utc)
    newer = retrieved + timedelta(days=1)
    for i in range(17_520):
        t = start + timedelta(hours=i)
        led.upsert(_iv(t, 0.1, retrieved=retrieved))
    led.accrue_cycle("UP123", cycle_start, computed_at=NOW)
    t0 = time.perf_counter()
    for i in range(17_520):
        t = start + timedelta(hours=i)
        led.upsert(_iv(t, 0.2, retrieved=newer))
    elapsed = time.perf_counter() - t0
    for point, cs in sorted(led.all_cycles()):
        led.accrue_cycle(point, cs, computed_at=NOW)
    assert elapsed < 5.0
    assert led.interval_cost_usd("UP123", start) is not None


_S = datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_window_prefix_stops_at_the_first_hole() -> None:
    led = make_test_ledger()
    for iv in hourly(_S, 10, skip=frozenset({4})):
        led.upsert(iv)
    prefix, covered = led.window_prefix("UP1", _S, _S + timedelta(hours=10))
    assert len(prefix) == 4
    assert covered == _S + timedelta(hours=4)


def test_window_prefix_empty_when_start_missing() -> None:
    led = make_test_ledger()
    for iv in hourly(_S + timedelta(hours=1), 3):
        led.upsert(iv)
    assert led.window_prefix("UP1", _S, _S + timedelta(hours=5)) == ([], None)


def test_latest_interval_end() -> None:
    led = make_test_ledger()
    assert led.latest_interval_end("UP1") is None
    for iv in hourly(_S, 5, skip=frozenset({2})):
        led.upsert(iv)
    assert led.latest_interval_end("UP1") == _S + timedelta(hours=5)
