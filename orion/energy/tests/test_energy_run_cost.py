from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.run_cost import estimate_run_cost
from orion.energy.tariff import load_tariff
from orion.schemas.energy import EnergyUsageIntervalV1
from orion.schemas.power import PowerIntentSettledV1

ROOT = Path(__file__).resolve().parents[3]
TARIFF = load_tariff(ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
M = 1.071 * 1.0518
B1S = 0.098332 * M
T = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
_TZ = ZoneInfo("America/Denver")


def _settled(**kw) -> PowerIntentSettledV1:
    base = dict(
        intent_id="i-1",
        workload_kind="reverie.diffusion",
        node="circe",
        gpu_index=2,
        outcome="settled",
        window_start=T,
        window_end=T + timedelta(hours=1),
        sample_count=3600,
        actual_mean_watts=250.0,
        actual_peak_watts=300.0,
        energy_joules=250.0 * 3600,
        baseline_watts=50.0,
    )
    base.update(kw)
    return PowerIntentSettledV1(**base)


def _iv(start: datetime, kwh: float) -> EnergyUsageIntervalV1:
    return EnergyUsageIntervalV1(
        source="file_drop",
        usage_point_id="UP123",
        interval_start=start,
        interval_end=start + timedelta(hours=1),
        energy_kwh=kwh,
        retrieved_at=NOW,
    )


def _ledger(*intervals: EnergyUsageIntervalV1) -> UsageLedger:
    """Build a ledger with contiguous cycle prefix for fixture intervals."""
    led = UsageLedger(TARIFF, tz=_TZ, cycle_start_day=1)
    if not intervals:
        return led

    by_point: dict[str, list[EnergyUsageIntervalV1]] = {}
    for iv in intervals:
        by_point.setdefault(iv.usage_point_id, []).append(iv)

    prepared: list[EnergyUsageIntervalV1] = []
    for point, ivs in by_point.items():
        ivs_sorted = sorted(ivs, key=lambda iv: iv.interval_start)
        cycle_start, _ = led.cycle_bounds(ivs_sorted[0].interval_start)
        first = ivs_sorted[0].interval_start
        if first.astimezone(timezone.utc) > cycle_start.astimezone(timezone.utc):
            prepared.append(
                EnergyUsageIntervalV1(
                    source="file_drop",
                    usage_point_id=point,
                    interval_start=cycle_start,
                    interval_end=first,
                    energy_kwh=0.0,
                    retrieved_at=NOW,
                )
            )

        starts = {iv.interval_start.astimezone(timezone.utc) for iv in ivs_sorted}
        for iv in ivs_sorted:
            t_minus_1 = (T - timedelta(hours=1)).astimezone(timezone.utc)
            if (
                iv.interval_start.astimezone(timezone.utc) == T.astimezone(timezone.utc)
                and t_minus_1 not in starts
            ):
                prepared.append(_iv(T - timedelta(hours=1), 0.0))
                starts.add(t_minus_1)
            prepared.append(iv)

    for iv in prepared:
        led.upsert(iv)
    for point, start in led.all_cycles():
        led.accrue_cycle(point, start, computed_at=NOW)
    return led


def test_blind_settlement_is_unknown_not_free() -> None:
    blind = _settled(
        outcome="no_samples",
        sample_count=0,
        actual_mean_watts=None,
        actual_peak_watts=None,
        energy_joules=None,
    )
    est = estimate_run_cost(
        blind,
        ledger=_ledger(_iv(T - timedelta(hours=2), 100.0)),
        usage_point_id="UP123",
        computed_at=NOW,
    )
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "settlement_not_measured"
    assert est.house_share_cost_usd is None and est.house_share_gap == "settlement_not_measured"
    assert est.energy_kwh is None


def test_incremental_cost_at_cycle_position() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.energy_basis == "incremental_over_baseline"
    assert est.energy_kwh == pytest.approx(0.2)
    assert est.estimated_run_cost_usd == pytest.approx(0.2 * B1S)
    assert est.marginal_usd_per_kwh == pytest.approx(B1S)
    assert est.cycle_kwh_basis == pytest.approx(100.0)
    assert est.cycle_kwh_basis_as_of == T - timedelta(hours=1)
    assert est.house_share_gap == "house_interval_missing"
    assert est.tariff_version == "rmp-ut-sch1-2026-08-10"


def test_gross_basis_without_baseline() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(
        _settled(baseline_watts=None), ledger=led, usage_point_id="UP123", computed_at=NOW
    )
    assert est.energy_basis == "gross"
    assert est.energy_kwh == pytest.approx(0.25)


def test_no_cycle_usage_is_a_gap() -> None:
    est = estimate_run_cost(_settled(), ledger=_ledger(), usage_point_id="UP123", computed_at=NOW)
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "no_cycle_usage"


def test_mid_cycle_hole_before_run_is_no_cycle_usage() -> None:
    led = _ledger(
        _iv(T - timedelta(hours=4), 50.0),
        _iv(T - timedelta(hours=2), 100.0),
    )
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "no_cycle_usage"


def test_unknown_usage_point_is_a_gap() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id=None, computed_at=NOW)
    assert est.run_cost_gap == "no_cycle_usage"
    assert est.house_share_gap == "house_interval_missing"


def test_house_share_when_interval_covers_window() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0), _iv(T, 1.0))
    est = estimate_run_cost(_settled(), ledger=led, usage_point_id="UP123", computed_at=NOW)
    interval_cost = 1.0 * B1S  # house position 100 kWh -> first block
    assert est.house_kwh_overlap == pytest.approx(1.0)
    assert est.house_share_cost_usd == pytest.approx(0.2 * interval_cost)
    assert est.house_share_gap is None
    assert est.estimated_run_cost_usd == pytest.approx(0.2 * B1S)


def test_house_share_needs_full_coverage() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0), _iv(T, 1.0))
    long_run = _settled(window_end=T + timedelta(hours=2), energy_joules=250.0 * 7200)
    est = estimate_run_cost(long_run, ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.house_share_cost_usd is None and est.house_share_gap == "house_interval_missing"


def test_nan_baseline_is_settlement_not_measured() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(
        _settled(baseline_watts=math.nan),
        ledger=led,
        usage_point_id="UP123",
        computed_at=NOW,
    )
    assert est.energy_kwh is None
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "settlement_not_measured"
    assert est.house_share_cost_usd is None and est.house_share_gap == "settlement_not_measured"


def test_nan_mean_is_settlement_not_measured() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(
        _settled(actual_mean_watts=math.nan),
        ledger=led,
        usage_point_id="UP123",
        computed_at=NOW,
    )
    assert est.energy_kwh is None
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "settlement_not_measured"
    assert est.house_share_cost_usd is None and est.house_share_gap == "settlement_not_measured"


def test_negative_energy_joules_gross_is_settlement_not_measured() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(
        _settled(baseline_watts=None, actual_mean_watts=None, energy_joules=-100.0),
        ledger=led,
        usage_point_id="UP123",
        computed_at=NOW,
    )
    assert est.energy_kwh is None
    assert est.estimated_run_cost_usd is None and est.run_cost_gap == "settlement_not_measured"
    assert est.house_share_cost_usd is None and est.house_share_gap == "settlement_not_measured"


def test_below_baseline_mean_is_zero_cost_not_gap() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0))
    est = estimate_run_cost(
        _settled(actual_mean_watts=30.0, baseline_watts=50.0, energy_joules=30.0 * 3600),
        ledger=led,
        usage_point_id="UP123",
        computed_at=NOW,
    )
    assert est.energy_kwh == pytest.approx(0.0)
    assert est.estimated_run_cost_usd == pytest.approx(0.0)
    assert est.run_cost_gap is None
    assert est.energy_basis == "incremental_over_baseline"


def test_partial_interval_overlap_is_house_share_gap() -> None:
    led = _ledger(_iv(T - timedelta(hours=2), 100.0), _iv(T, 1.0))
    partial = _settled(
        window_start=T - timedelta(minutes=30),
        window_end=T + timedelta(hours=1, minutes=30),
        energy_joules=250.0 * 7200,
    )
    est = estimate_run_cost(partial, ledger=led, usage_point_id="UP123", computed_at=NOW)
    assert est.house_share_cost_usd is None and est.house_share_gap == "house_interval_missing"
