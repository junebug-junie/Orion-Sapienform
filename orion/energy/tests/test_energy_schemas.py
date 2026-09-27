from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.energy import (
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)

T0 = datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 10, 19, 0, tzinfo=timezone.utc)


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
