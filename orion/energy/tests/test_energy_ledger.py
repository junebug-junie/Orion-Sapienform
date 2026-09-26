from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
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


def _ledger(day: int = 1) -> UsageLedger:
    return UsageLedger(TARIFF, tz=DENVER, cycle_start_day=day)


def test_newer_retrieval_replaces_older_is_ignored() -> None:
    led = _ledger()
    t = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    assert led.upsert(_iv(t, 1.0, retrieved=NOW)) is True
    assert led.upsert(_iv(t, 2.0, retrieved=NOW + timedelta(days=1))) is True
    assert led.upsert(_iv(t, 9.0, retrieved=NOW - timedelta(days=1))) is False
    assert led.intervals_overlapping("UP123", t, t + timedelta(hours=1))[0].energy_kwh == 2.0


def test_cycle_bounds_mid_month_start_day() -> None:
    led = _ledger(day=15)
    start, end = led.cycle_bounds(datetime(2026, 9, 10, 12, tzinfo=DENVER))
    assert (start, end) == (datetime(2026, 8, 15, tzinfo=DENVER), datetime(2026, 9, 15, tzinfo=DENVER))
    start, _ = led.cycle_bounds(datetime(2026, 9, 15, 0, 30, tzinfo=DENVER))
    assert start == datetime(2026, 9, 15, tzinfo=DENVER)


def test_cycle_start_day_clamps_to_short_month() -> None:
    led = _ledger(day=31)
    start, end = led.cycle_bounds(datetime(2026, 3, 5, tzinfo=DENVER))
    assert start == datetime(2026, 2, 28, tzinfo=DENVER)
    assert end == datetime(2026, 3, 31, tzinfo=DENVER)


def test_accrual_crosses_block_boundary_in_order() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    for i in range(3):
        led.upsert(_iv(base + timedelta(hours=i), 200.0))
    start, _ = led.cycle_bounds(base)
    rows = led.accrue_cycle("UP123", start, computed_at=NOW)
    assert [r.interval_cost_usd for r in rows] == pytest.approx([200 * B1S, 200 * B1S, 200 * B2S])
    assert rows[-1].cycle_accumulated_kwh == pytest.approx(600.0)
    assert rows[-1].cycle_to_date_total_usd == pytest.approx(400 * B1S + 200 * B2S + 12.16)
    assert rows[0].marginal_usd_per_kwh == pytest.approx(B1S)
    assert rows[1].marginal_usd_per_kwh == pytest.approx(B2S)  # 400 used: next kWh is block 2
    assert rows[0].cycle_start == date(2026, 7, 1)
    assert led.interval_cost_usd("UP123", base) == pytest.approx(200 * B1S)


def test_late_interval_reprices_later_ones() -> None:
    led = _ledger()
    base = datetime(2026, 7, 2, 18, tzinfo=timezone.utc)
    led.upsert(_iv(base + timedelta(hours=1), 200.0))
    start, _ = led.cycle_bounds(base)
    led.accrue_cycle("UP123", start, computed_at=NOW)
    led.upsert(_iv(base, 300.0))  # arrives late, earlier in the cycle
    rows = led.accrue_cycle("UP123", start, computed_at=NOW)
    assert rows[1].interval_cost_usd == pytest.approx(100 * B1S + 100 * B2S)


def test_cycle_kwh_before_counts_only_finished_intervals_in_cycle() -> None:
    led = _ledger()
    t = datetime(2026, 9, 10, 16, tzinfo=timezone.utc)
    assert led.cycle_kwh_before("UP123", t) is None
    led.upsert(_iv(datetime(2026, 8, 31, 5, tzinfo=timezone.utc), 50.0))  # previous cycle (Denver Aug 30)
    assert led.cycle_kwh_before("UP123", t) is None
    led.upsert(_iv(t - timedelta(hours=2), 100.0))
    led.upsert(_iv(t, 7.0))  # starts at t: not finished before t
    kwh, as_of = led.cycle_kwh_before("UP123", t)
    assert kwh == pytest.approx(100.0)
    assert as_of == t - timedelta(hours=1)


def test_all_cycles_and_usage_points() -> None:
    led = _ledger()
    led.upsert(_iv(datetime(2026, 7, 2, 18, tzinfo=timezone.utc), 1.0))
    led.upsert(_iv(datetime(2026, 8, 2, 18, tzinfo=timezone.utc), 1.0, point="UP9"))
    assert led.usage_points() == {"UP123", "UP9"}
    assert led.all_cycles() == {
        ("UP123", datetime(2026, 7, 1, tzinfo=DENVER)),
        ("UP9", datetime(2026, 8, 1, tzinfo=DENVER)),
    }
