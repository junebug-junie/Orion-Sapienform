from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import reconcile_actual, reconcile_forecast
from orion.energy.testing import flat_test_tariff, hourly, make_test_ledger
from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1, EnergyUsageIntervalV1

DENVER = ZoneInfo("America/Denver")

S = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def _bill(**over) -> EnergyBillActualV1:
    base = dict(
        source="file_drop",
        billing_period_start=date(2026, 9, 1),
        billing_period_end=date(2026, 9, 3),
        kwh_billed=470.0,
        energy_charge=48.0,
        customer_charge=10.0,
        taxes=3.0,
        current_charges=64.6,
        retrieved_at=NOW,
    )
    base.update(over)
    return EnergyBillActualV1(**base)


def _ledger_with(intervals):
    led = make_test_ledger()
    for iv in intervals:
        led.upsert(iv)
    return led


def test_actual_full_period_hand_oracle() -> None:
    # 48 h x 10 kWh = 480 kWh: 400 @ 0.10 + 80 @ 0.12 = 49.60, + 10 fixed = 59.60.
    rec = reconcile_actual(_bill(), ledger=_ledger_with(hourly(S, 48, kwh=10.0)), usage_point_id="UP1", computed_at=NOW)
    assert rec.reconcile_gap is None
    assert rec.orion_kwh == pytest.approx(480.0)
    assert rec.orion_energy_usd == pytest.approx(49.6)
    assert rec.orion_total_usd == pytest.approx(59.6)
    assert rec.utility_basis == "pre_tax"
    assert rec.utility_total_usd == pytest.approx(61.6)  # 64.60 - 3.00 tax
    assert rec.delta_usd == pytest.approx(-2.0)
    assert rec.delta_kwh == pytest.approx(10.0)
    assert rec.delta_pct == pytest.approx(-2.0 / 61.6)
    assert rec.bucket_deltas == pytest.approx({"energy_charge": 1.6, "customer_charge": 0.0})
    assert rec.orion_method == "metered_period"
    assert rec.tariff_version == "test-flat-v1"


def test_actual_bucket_with_adjustments() -> None:
    rec = reconcile_actual(
        _bill(adjustments=1.5), ledger=_ledger_with(hourly(S, 48, kwh=10.0)), usage_point_id="UP1", computed_at=NOW
    )
    assert rec.bucket_deltas["energy_charge_plus_adjustments"] == pytest.approx(49.6 - 49.5)


def test_actual_without_tax_line_is_labeled_tax_unknown() -> None:
    rec = reconcile_actual(
        _bill(taxes=None), ledger=_ledger_with(hourly(S, 48, kwh=10.0)), usage_point_id="UP1", computed_at=NOW
    )
    assert rec.utility_basis == "tax_unknown"
    assert rec.utility_total_usd == pytest.approx(64.6)


def test_actual_with_a_hole_is_a_gap_not_a_number() -> None:
    rec = reconcile_actual(
        _bill(), ledger=_ledger_with(hourly(S, 48, kwh=10.0, skip=frozenset({30}))), usage_point_id="UP1", computed_at=NOW
    )
    assert rec.reconcile_gap == "usage_incomplete"
    assert rec.orion_total_usd is None and rec.delta_usd is None
    assert rec.orion_covered_through == S + timedelta(hours=30)


def test_actual_with_no_usage() -> None:
    rec = reconcile_actual(_bill(), ledger=make_test_ledger(), usage_point_id="UP1", computed_at=NOW)
    assert rec.reconcile_gap == "no_usage"
    assert rec.orion_covered_through is None


def test_actual_first_hour_missing_is_usage_incomplete() -> None:
    rec = reconcile_actual(
        _bill(), ledger=_ledger_with(hourly(S, 48, skip=frozenset({0}))), usage_point_id="UP1", computed_at=NOW
    )
    assert rec.reconcile_gap == "usage_incomplete"
    assert rec.orion_covered_through is None
    assert rec.orion_total_usd is None


def _forecast(**over) -> EnergyBillForecastV1:
    base = dict(
        source="file_drop",
        billing_period_start=date(2026, 9, 1),
        billing_period_end=date(2026, 10, 1),
        as_of=S + timedelta(hours=72),
        projected_kwh=700.0,
        projected_total_usd=80.0,
        retrieved_at=S + timedelta(hours=72),
    )
    base.update(over)
    return EnergyBillForecastV1(**base)


def test_forecast_linear_run_rate_hand_oracle() -> None:
    # 72 kWh in 72 of 720 h -> 720 kWh: 400 @ .10 + 320 @ .12 = 78.40, + 10 = 88.40.
    rec = reconcile_forecast(_forecast(), ledger=_ledger_with(hourly(S, 72)), usage_point_id="UP1", computed_at=NOW)
    assert rec.orion_method == "linear_run_rate"
    assert rec.orion_kwh == pytest.approx(720.0)
    assert rec.orion_total_usd == pytest.approx(88.4)
    assert rec.delta_kwh == pytest.approx(20.0)
    assert rec.delta_usd == pytest.approx(8.4)
    assert rec.delta_pct == pytest.approx(8.4 / 80.0)
    assert rec.utility_basis == "tax_unknown"


def test_forecast_needs_a_day_of_usage() -> None:
    rec = reconcile_forecast(_forecast(), ledger=_ledger_with(hourly(S, 12)), usage_point_id="UP1", computed_at=NOW)
    assert rec.reconcile_gap == "usage_incomplete"
    assert rec.orion_covered_through == S + timedelta(hours=12)


def _denver_dst_forecast(*, hours: int, as_of_hours: int) -> tuple[UsageLedger, EnergyBillForecastV1]:
    start = datetime(2026, 10, 15, tzinfo=DENVER)
    led = UsageLedger(flat_test_tariff(), tz=DENVER, cycle_start_day=1)
    for iv in hourly(start, hours):
        led.upsert(iv)
    as_of = start.astimezone(timezone.utc) + timedelta(hours=as_of_hours)
    forecast = EnergyBillForecastV1(
        source="file_drop",
        billing_period_start=date(2026, 10, 15),
        billing_period_end=date(2026, 11, 15),
        as_of=as_of,
        projected_kwh=700.0,
        projected_total_usd=80.0,
        retrieved_at=as_of,
    )
    return led, forecast


def test_forecast_dst_partial_coverage_projects_over_real_seconds() -> None:
    # Oct 15–Nov 15 Denver is 745 real hours; 744 UTC-hourly intervals end 1 h before period end.
    led, forecast = _denver_dst_forecast(hours=744, as_of_hours=744)
    rec = reconcile_forecast(forecast, ledger=led, usage_point_id="UP1", computed_at=NOW)
    assert rec.orion_kwh == pytest.approx(745.0)


def test_forecast_dst_full_coverage_returns_metered_kwh() -> None:
    led, forecast = _denver_dst_forecast(hours=745, as_of_hours=745)
    rec = reconcile_forecast(forecast, ledger=led, usage_point_id="UP1", computed_at=NOW)
    assert rec.orion_kwh == pytest.approx(745.0)


def test_forecast_quarter_hour_intervals_use_real_covered_time() -> None:
    # 72 h of 15-min intervals at 0.25 kWh = 72 kWh; same 720 h oracle as hourly.
    got = S + timedelta(days=60)
    start_utc = S.astimezone(timezone.utc)
    intervals = [
        EnergyUsageIntervalV1(
            source="file_drop",
            usage_point_id="UP1",
            interval_start=start_utc + timedelta(minutes=15 * i),
            interval_end=start_utc + timedelta(minutes=15 * (i + 1)),
            energy_kwh=0.25,
            retrieved_at=got,
        )
        for i in range(72 * 4)
    ]
    rec = reconcile_forecast(_forecast(), ledger=_ledger_with(intervals), usage_point_id="UP1", computed_at=NOW)
    assert rec.orion_kwh == pytest.approx(720.0)
    assert rec.orion_total_usd == pytest.approx(88.4)


def test_forecast_without_end_uses_ledger_cycle() -> None:
    rec = reconcile_forecast(
        _forecast(billing_period_end=None), ledger=_ledger_with(hourly(S, 72)), usage_point_id="UP1", computed_at=NOW
    )
    assert rec.billing_period_end == date(2026, 10, 1)
    assert rec.orion_total_usd == pytest.approx(88.4)


def test_forecast_kwh_only_leaves_usd_delta_unknown() -> None:
    rec = reconcile_forecast(
        _forecast(projected_total_usd=None), ledger=_ledger_with(hourly(S, 72)), usage_point_id="UP1", computed_at=NOW
    )
    assert rec.delta_usd is None and rec.delta_pct is None
    assert rec.delta_kwh == pytest.approx(20.0)
