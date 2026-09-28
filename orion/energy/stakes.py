"""Stakes snapshot: what the house bill looks like right now, for spend gates.

Compares Orion's pre-tax run-rate projection with RMP's forecast (which may include
tax), so the ratio leans low -- a gate reading it holds less often, not more.
Any missing or unhealthy input makes pressure `unknown`, and unknown never holds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import period_bounds, price_intervals, project_period
from orion.schemas.energy import EnergyBillForecastV1, EnergyImporterStatusV1, EnergyStakesSnapshotV1


def _forecast_is_current(forecast: EnergyBillForecastV1, *, now: datetime, tz: ZoneInfo) -> bool:
    today = now.astimezone(tz).date()
    if forecast.billing_period_start > today:
        return False
    if forecast.billing_period_end is not None and forecast.billing_period_end <= today:
        return False
    return True


def current_forecast(
    forecasts: Iterable[EnergyBillForecastV1], *, now: datetime, tz: ZoneInfo
) -> Optional[EnergyBillForecastV1]:
    live = [f for f in forecasts if _forecast_is_current(f, now=now, tz=tz)]
    return max(live, key=lambda f: (f.billing_period_start, f.as_of), default=None)


def build_stakes_snapshot(
    *,
    ledger: UsageLedger,
    usage_point_id: Optional[str],
    forecast: Optional[EnergyBillForecastV1],
    importer: EnergyImporterStatusV1,
    now: datetime,
    near_ratio: float,
    over_ratio: float,
    stale_after_hours: float,
) -> EnergyStakesSnapshotV1:
    tz, tariff = ledger.tz, ledger.tariff
    snap: dict[str, Any] = dict(
        as_of=now, usage_point_id=usage_point_id, importer_state=importer.state, tariff_version=tariff.version,
    )
    forecast_current = forecast is not None and _forecast_is_current(forecast, now=now, tz=tz)
    if forecast is not None and forecast_current:
        snap.update(forecast_total_usd=forecast.projected_total_usd, forecast_as_of=forecast.as_of)
    if usage_point_id is None:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="no_usage_point")

    if forecast_current:
        start, end = period_bounds(forecast.billing_period_start, forecast.billing_period_end, ledger)
    else:
        start, end = ledger.cycle_bounds(now)
    snap.update(cycle_start=start.astimezone(tz).date(), cycle_end=end.astimezone(tz).date())

    prefix, covered = ledger.window_prefix(usage_point_id, start, end)
    if prefix and covered is not None:
        days = ledger.period_days(start, end)
        kwh, energy = price_intervals(tariff, prefix, tz=tz, period_days=days)
        month = (covered - timedelta(seconds=1)).astimezone(tz).month
        snap.update(
            covered_through=covered, cycle_accumulated_kwh=kwh,
            cycle_to_date_total_usd=energy + tariff.fixed_usd(days),
            marginal_usd_per_kwh=tariff.marginal_usd_per_kwh(cycle_kwh=kwh, month=month, period_days=days),
        )
    projection, gap, _ = project_period(ledger, usage_point_id, start, end)
    if projection is not None:
        snap["orion_projected_total_usd"] = projection.projected_total_usd

    if importer.state != "healthy":
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason=f"importer_{importer.state}")
    if forecast is not None and not forecast_current:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="forecast_not_current")
    if projection is None:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason=f"projection_{gap}")
    coverage_lag = max(0.0, (now - projection.covered_through).total_seconds() / 3600.0)
    if coverage_lag > stale_after_hours:
        return EnergyStakesSnapshotV1(
            **snap, pressure="unknown", pressure_reason=f"coverage_lag_hours={coverage_lag:.1f}",
        )
    if forecast is None or forecast.projected_total_usd is None:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="no_forecast_total")
    if forecast.projected_total_usd <= 0.0:
        return EnergyStakesSnapshotV1(**snap, pressure="unknown", pressure_reason="forecast_nonpositive")

    ratio = projection.projected_total_usd / forecast.projected_total_usd
    pressure = "over_forecast" if ratio >= over_ratio else "near_forecast" if ratio >= near_ratio else "normal"
    return EnergyStakesSnapshotV1(
        **snap, projected_to_forecast_ratio=ratio, pressure=pressure, pressure_reason=f"ratio={ratio:.3f}",
    )
