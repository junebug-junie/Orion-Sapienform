from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyCostAccruedV1,
    EnergyImporterStatusV1,
    EnergyReconcileV1,
    EnergyRunCostEstimatedV1,
    EnergyStakesSnapshotV1,
    EnergyUsageIntervalV1,
)

T0 = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc)
_T = datetime(2026, 9, 12, 6, tzinfo=timezone.utc)


def _usage(**kw) -> EnergyUsageIntervalV1:
    base = dict(
        source="file_drop",
        usage_point_id="UP123",
        interval_start=T0,
        interval_end=T1,
        energy_kwh=1.234,
        retrieved_at=T1,
    )
    base.update(kw)
    return EnergyUsageIntervalV1(**base)


def test_usage_interval_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        _usage(interval_end=T0)


def test_usage_interval_naive_datetimes_become_utc() -> None:
    iv = _usage(interval_start=datetime(2026, 9, 10, 18), interval_end=datetime(2026, 9, 10, 19))
    assert iv.interval_start.tzinfo is not None
    assert iv.interval_seconds == 3600


def test_usage_interval_rejects_negative_kwh() -> None:
    with pytest.raises(ValidationError):
        _usage(energy_kwh=-0.1)


def test_usage_interval_rejects_non_finite_kwh() -> None:
    with pytest.raises(ValidationError):
        _usage(energy_kwh=float("inf"))
    with pytest.raises(ValidationError):
        _usage(energy_kwh=float("nan"))


def test_accrued_round_trips_json() -> None:
    acc = EnergyCostAccruedV1(
        usage_point_id="UP123",
        interval_start=T0,
        interval_end=T1,
        energy_kwh=1.0,
        interval_cost_usd=0.11,
        marginal_usd_per_kwh=0.11,
        cycle_start=date(2026, 9, 1),
        cycle_accumulated_kwh=101.0,
        cycle_energy_cost_usd=11.2,
        cycle_to_date_total_usd=23.36,
        tariff_version="rmp-ut-sch1-2026-08-10",
        computed_at=T1,
    )
    again = EnergyCostAccruedV1.model_validate(acc.model_dump(mode="json"))
    assert again == acc
    assert again.cost_basis == "pre_tax"


def _run(**kw) -> EnergyRunCostEstimatedV1:
    base = dict(
        intent_id="i-1",
        workload_kind="reverie.diffusion",
        node="circe",
        window_start=T0,
        window_end=T1,
        settlement_outcome="settled",
        computed_at=T1,
    )
    base.update(kw)
    return EnergyRunCostEstimatedV1(**base)


def test_run_cost_null_requires_gap_reason() -> None:
    with pytest.raises(ValidationError):
        _run(house_share_gap="house_interval_missing")  # run cost null, no run_cost_gap


def test_run_cost_value_forbids_gap_reason() -> None:
    with pytest.raises(ValidationError):
        _run(
            estimated_run_cost_usd=0.02,
            run_cost_gap="no_cycle_usage",
            house_share_gap="house_interval_missing",
        )


def test_run_cost_both_unknown_is_valid_with_reasons() -> None:
    est = _run(run_cost_gap="settlement_not_measured", house_share_gap="settlement_not_measured")
    assert est.estimated_run_cost_usd is None
    assert est.house_share_cost_usd is None


def _bill(**over) -> dict:
    base = dict(
        source="file_drop",
        billing_period_start=date(2026, 8, 12),
        billing_period_end=date(2026, 9, 11),
        kwh_billed=712.0,
        current_charges=101.23,
        retrieved_at=_T,
    )
    base.update(over)
    return base


def test_bill_actual_missing_lines_are_unknown_not_zero() -> None:
    bill = EnergyBillActualV1(**_bill())
    assert bill.taxes is None and bill.energy_charge is None and bill.credits is None


def test_bill_actual_rejects_backwards_period() -> None:
    with pytest.raises(ValueError):
        EnergyBillActualV1(**_bill(billing_period_end=date(2026, 8, 12)))


def test_forecast_with_no_projection_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnergyBillForecastV1(
            source="file_drop",
            billing_period_start=date(2026, 9, 11),
            as_of=_T,
            retrieved_at=_T,
        )


def _rec(**over) -> dict:
    base = dict(
        reconcile_kind="actual",
        usage_point_id="UP1",
        billing_period_start=date(2026, 8, 12),
        billing_period_end=date(2026, 9, 11),
        utility_as_of=_T,
        utility_kwh=712.0,
        utility_total_usd=97.13,
        utility_basis="pre_tax",
        orion_method="metered_period",
        tariff_version="t1",
        computed_at=_T,
    )
    base.update(over)
    return base


def test_reconcile_needs_exactly_one_of_total_or_gap() -> None:
    with pytest.raises(ValueError):
        EnergyReconcileV1(**_rec())
    with pytest.raises(ValueError):
        EnergyReconcileV1(**_rec(orion_total_usd=99.0, reconcile_gap="no_usage"))
    assert EnergyReconcileV1(**_rec(reconcile_gap="no_usage")).orion_total_usd is None


def test_reconcile_gap_forbids_deltas() -> None:
    with pytest.raises(ValueError):
        EnergyReconcileV1(**_rec(reconcile_gap="usage_incomplete", delta_usd=1.0))


def test_stakes_comparison_pressures_need_a_ratio() -> None:
    base = dict(as_of=_T, importer_state="healthy", pressure_reason="ratio=1.2")
    with pytest.raises(ValueError):
        EnergyStakesSnapshotV1(**base, pressure="over_forecast")
    ok = EnergyStakesSnapshotV1(**base, pressure="over_forecast", projected_to_forecast_ratio=1.2)
    assert ok.forecast_total_usd is None
    assert EnergyStakesSnapshotV1(**base, pressure="unknown").projected_to_forecast_ratio is None


def test_stakes_compared_pressure_requires_healthy_importer() -> None:
    with pytest.raises(ValueError):
        EnergyStakesSnapshotV1(
            as_of=_T,
            importer_state="stale",
            pressure="over_forecast",
            pressure_reason="ratio=1.2",
            projected_to_forecast_ratio=1.2,
        )


def test_stakes_rejects_non_finite_ratio() -> None:
    base = dict(
        as_of=_T,
        importer_state="healthy",
        pressure="over_forecast",
        pressure_reason="ratio=nan",
    )
    with pytest.raises(ValidationError):
        EnergyStakesSnapshotV1(**base, projected_to_forecast_ratio=float("nan"))


def test_reconcile_rejects_non_finite_bucket_delta() -> None:
    with pytest.raises(ValidationError):
        EnergyReconcileV1(
            **_rec(
                orion_total_usd=99.0,
                bucket_deltas={"energy": float("nan")},
            )
        )


def test_importer_status_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        EnergyImporterStatusV1(state="stale", reason="", source="file_drop", as_of=_T)
