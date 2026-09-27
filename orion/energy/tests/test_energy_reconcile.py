from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from orion.energy.reconcile import reconcile_actual, reconcile_forecast
from orion.energy.testing import hourly, make_test_ledger
from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1

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
