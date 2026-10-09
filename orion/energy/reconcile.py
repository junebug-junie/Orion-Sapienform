"""Reconcile Orion's tariff estimate against what Rocky Mountain Power billed or projects.

Bill periods come from RMP (meter-read dates), not from ENERGY_BILLING_CYCLE_START_DAY,
and blocks reset at the bill's own start. A period Orion cannot fully see is a gap,
never a partial number dressed up as a total.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import Tariff
from orion.schemas.energy import (
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyReconcileV1,
    EnergyUsageIntervalV1,
    ReconcileGap,
    UtilityBasis,
)

# A run rate from less than a day of data is mostly time-of-day noise.
MIN_RUN_RATE_HOURS = 24.0


def _duration_seconds(start: datetime, end: datetime) -> float:
    """Elapsed seconds between instants — always UTC, never wall-clock DST arithmetic."""
    return (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()


def _any_usage_in_period(ledger: UsageLedger, usage_point_id: str, start: datetime, end: datetime) -> bool:
    return any(start <= iv.interval_start < end for iv in ledger.intervals_overlapping(usage_point_id, start, end))


def _empty_prefix_gap(ledger: UsageLedger, usage_point_id: str, start: datetime, end: datetime) -> ReconcileGap:
    return "usage_incomplete" if _any_usage_in_period(ledger, usage_point_id, start, end) else "no_usage"


def local_midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=tz)


def period_bounds(start_day: date, end_day: Optional[date], ledger: UsageLedger) -> tuple[datetime, datetime]:
    start = local_midnight(start_day, ledger.tz)
    end = local_midnight(end_day, ledger.tz) if end_day is not None else ledger.cycle_bounds(start)[1]
    return start, end


def price_intervals(
    tariff: Tariff,
    intervals: Iterable[EnergyUsageIntervalV1],
    *,
    tz: ZoneInfo,
    period_days: Optional[float] = None,
) -> tuple[float, float]:
    kwh = 0.0
    cost = 0.0
    for iv in intervals:
        month = iv.interval_start.astimezone(tz).month
        cost += tariff.energy_cost_usd(iv.energy_kwh, cycle_kwh_before=kwh, month=month, period_days=period_days)
        kwh += iv.energy_kwh
    return kwh, cost


@dataclass(frozen=True)
class PeriodProjection:
    observed_kwh: float
    observed_energy_usd: float
    projected_kwh: float
    projected_energy_usd: float
    fixed_usd: float
    covered_through: datetime

    @property
    def projected_total_usd(self) -> float:
        return self.projected_energy_usd + self.fixed_usd


def project_period(
    ledger: UsageLedger, usage_point_id: str, start: datetime, end: datetime
) -> tuple[Optional[PeriodProjection], Optional[ReconcileGap], Optional[datetime]]:
    prefix, covered = ledger.window_prefix(usage_point_id, start, end)
    if not prefix or covered is None:
        return None, _empty_prefix_gap(ledger, usage_point_id, start, end), None
    covered_sec = _duration_seconds(start, covered)
    if covered_sec < MIN_RUN_RATE_HOURS * 3600.0:
        return None, "usage_incomplete", covered
    tariff, tz = ledger.tariff, ledger.tz
    days = ledger.period_days(start, end)
    kwh, energy = price_intervals(tariff, prefix, tz=tz, period_days=days)
    period_sec = _duration_seconds(start, end)
    projected_kwh = kwh if covered >= end else kwh * period_sec / covered_sec
    last_month = (end - timedelta(seconds=1)).astimezone(tz).month
    remaining = max(0.0, projected_kwh - kwh)
    projected_energy = energy + tariff.energy_cost_usd(
        remaining, cycle_kwh_before=kwh, month=last_month, period_days=days
    )
    return (
        PeriodProjection(
            observed_kwh=kwh,
            observed_energy_usd=energy,
            projected_kwh=projected_kwh,
            projected_energy_usd=projected_energy,
            fixed_usd=tariff.fixed_usd(days),
            covered_through=covered,
        ),
        None,
        covered,
    )


def _pct(delta: Optional[float], base: Optional[float]) -> Optional[float]:
    if delta is None or base is None or base == 0.0:
        return None
    return delta / base


def _utility_pretax(bill: EnergyBillActualV1) -> tuple[float, UtilityBasis]:
    if bill.taxes is None:
        return bill.current_charges, "tax_unknown"
    return bill.current_charges - bill.taxes, "pre_tax"


def reconcile_actual(
    bill: EnergyBillActualV1, *, ledger: UsageLedger, usage_point_id: str, computed_at: datetime
) -> EnergyReconcileV1:
    tariff, tz = ledger.tariff, ledger.tz
    start, end = period_bounds(bill.billing_period_start, bill.billing_period_end, ledger)
    prefix, covered = ledger.window_prefix(usage_point_id, start, end)
    utility_total, basis = _utility_pretax(bill)
    common = dict(
        reconcile_kind="actual",
        usage_point_id=usage_point_id,
        billing_period_start=bill.billing_period_start,
        billing_period_end=bill.billing_period_end,
        utility_as_of=bill.retrieved_at,
        utility_kwh=bill.kwh_billed,
        utility_total_usd=utility_total,
        utility_basis=basis,
        orion_method="metered_period",
        orion_covered_through=covered,
        tariff_version=tariff.version,
        cost_basis=tariff.cost_basis,
        computed_at=computed_at,
    )
    if not prefix or covered is None:
        return EnergyReconcileV1(**common, reconcile_gap=_empty_prefix_gap(ledger, usage_point_id, start, end))
    if covered < end:
        return EnergyReconcileV1(**common, reconcile_gap="usage_incomplete")
    days = (bill.billing_period_end - bill.billing_period_start).days
    kwh, energy = price_intervals(tariff, prefix, tz=tz, period_days=days)
    fixed = tariff.fixed_usd(days)
    total = energy + fixed
    base_energy = energy / tariff.energy_multiplier
    customer = tariff.customer_charge_usd(days)
    # Riders on the customer charge print among the bill's adjustments, not in its customer line.
    customer_riders = customer * (tariff.customer_multiplier - 1.0)
    buckets: dict[str, float] = {}
    if bill.energy_charge is not None:
        buckets["energy_charge"] = base_energy - bill.energy_charge
        if bill.adjustments is not None:
            buckets["energy_charge_plus_adjustments"] = (energy + customer_riders) - (
                bill.energy_charge + bill.adjustments
            )
    if bill.customer_charge is not None:
        buckets["customer_charge"] = (customer if tariff.has_customer_charge else fixed) - bill.customer_charge
    delta_usd = total - utility_total
    return EnergyReconcileV1(
        **common,
        orion_kwh=kwh,
        orion_energy_usd=energy,
        orion_fixed_usd=fixed,
        orion_total_usd=total,
        delta_kwh=kwh - bill.kwh_billed,
        delta_usd=delta_usd,
        delta_pct=_pct(delta_usd, utility_total),
        bucket_deltas=buckets,
    )


def reconcile_forecast(
    forecast: EnergyBillForecastV1, *, ledger: UsageLedger, usage_point_id: str, computed_at: datetime
) -> EnergyReconcileV1:
    tariff = ledger.tariff
    start, end = period_bounds(forecast.billing_period_start, forecast.billing_period_end, ledger)
    projection, gap, covered = project_period(ledger, usage_point_id, start, end)
    common = dict(
        reconcile_kind="forecast",
        usage_point_id=usage_point_id,
        billing_period_start=forecast.billing_period_start,
        billing_period_end=forecast.billing_period_end or end.astimezone(ledger.tz).date(),
        utility_as_of=forecast.as_of,
        utility_kwh=forecast.projected_kwh,
        utility_total_usd=forecast.projected_total_usd,
        utility_basis="tax_unknown",
        orion_method="linear_run_rate",
        orion_covered_through=covered,
        tariff_version=tariff.version,
        cost_basis=tariff.cost_basis,
        computed_at=computed_at,
    )
    if projection is None:
        return EnergyReconcileV1(**common, reconcile_gap=gap)
    total = projection.projected_total_usd
    delta_kwh = None if forecast.projected_kwh is None else projection.projected_kwh - forecast.projected_kwh
    delta_usd = None if forecast.projected_total_usd is None else total - forecast.projected_total_usd
    return EnergyReconcileV1(
        **common,
        orion_kwh=projection.projected_kwh,
        orion_energy_usd=projection.projected_energy_usd,
        orion_fixed_usd=projection.fixed_usd,
        orion_total_usd=total,
        delta_kwh=delta_kwh,
        delta_usd=delta_usd,
        delta_pct=_pct(delta_usd, forecast.projected_total_usd),
    )
