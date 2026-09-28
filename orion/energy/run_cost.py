"""Price one settled power intent against the house tariff.

The meter-side estimate uses the INCREMENTAL draw (mean minus the card's baseline
just before the window) when a baseline exists: an idle card draws its baseline
whether or not the workload runs, so only the delta is the workload's cost.

The house share apportions each overlapping whole-house interval by the run's share
of that interval's kWh. It needs the utility data, which lags about a day, so it is
usually a gap at settlement time and filled in when the pipeline re-prices.

Cycle position and season are taken from ``window_start`` only. A run that crosses a
billing-cycle or season boundary is priced at its start position; ``marginal_usd_per_kwh``
is the blended rate when incremental kWh straddle a block threshold.

``no_cycle_usage`` means the billing cycle's usage before the run is missing or
incomplete (no intervals yet, a mid-cycle hole, or no usage point).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from orion.energy.ledger import UsageLedger
from orion.schemas.energy import EnergyRunCostEstimatedV1
from orion.schemas.power import PowerIntentSettledV1

JOULES_PER_KWH = 3_600_000.0
_COVERAGE_SLACK_SEC = 1.0


def _run_kwh(settled: PowerIntentSettledV1) -> tuple[Optional[float], Optional[str]]:
    if settled.outcome != "settled":
        return None, None
    elapsed = (settled.window_end - settled.window_start).total_seconds()
    if elapsed <= 0:
        return None, None
    if settled.baseline_watts is not None and settled.actual_mean_watts is not None:
        mean = settled.actual_mean_watts
        baseline = settled.baseline_watts
        if not (math.isfinite(mean) and math.isfinite(baseline)) or mean < 0.0:
            return None, None
        delta_w = max(0.0, mean - baseline)
        return delta_w * elapsed / JOULES_PER_KWH, "incremental_over_baseline"
    if settled.energy_joules is None:
        return None, None
    joules = settled.energy_joules
    if not math.isfinite(joules) or joules < 0.0:
        return None, None
    return joules / JOULES_PER_KWH, "gross"


def _run_cost_fields(
    kwh: float, settled: PowerIntentSettledV1, ledger: UsageLedger, point: Optional[str]
) -> dict[str, Any]:
    before = ledger.cycle_kwh_before(point, settled.window_start) if point else None
    if before is None:
        return {"run_cost_gap": "no_cycle_usage"}
    cycle_kwh, as_of = before
    month = settled.window_start.astimezone(ledger.tz).month
    days = ledger.period_days(*ledger.cycle_bounds(settled.window_start))
    cost = ledger.tariff.energy_cost_usd(kwh, cycle_kwh_before=cycle_kwh, month=month, period_days=days)
    marginal = (
        cost / kwh if kwh > 0
        else ledger.tariff.marginal_usd_per_kwh(cycle_kwh=cycle_kwh, month=month, period_days=days)
    )
    return {
        "estimated_run_cost_usd": cost,
        "marginal_usd_per_kwh": marginal,
        "cycle_kwh_basis": cycle_kwh,
        "cycle_kwh_basis_as_of": as_of,
    }


def _house_share_fields(
    kwh: float, settled: PowerIntentSettledV1, ledger: UsageLedger, point: Optional[str]
) -> dict[str, Any]:
    gap = {"house_share_gap": "house_interval_missing"}
    if not point:
        return gap
    start, end = settled.window_start, settled.window_end
    window_sec = (end - start).total_seconds()
    covered = share_cost = house_kwh = 0.0
    for iv in ledger.intervals_overlapping(point, start, end):
        interval_cost = ledger.interval_cost_usd(point, iv.interval_start)
        if interval_cost is None:
            return gap
        overlap = (min(end, iv.interval_end) - max(start, iv.interval_start)).total_seconds()
        interval_sec = float(iv.interval_seconds)
        run_in = kwh * overlap / window_sec
        house_in = iv.energy_kwh * overlap / interval_sec
        share = 1.0 if house_in <= 0 else min(1.0, run_in / house_in)
        share_cost += share * interval_cost * overlap / interval_sec
        house_kwh += house_in
        covered += overlap
    if covered + _COVERAGE_SLACK_SEC < window_sec:
        return gap
    return {"house_share_cost_usd": share_cost, "house_kwh_overlap": house_kwh}


def estimate_run_cost(
    settled: PowerIntentSettledV1,
    *,
    ledger: UsageLedger,
    usage_point_id: Optional[str],
    computed_at: datetime,
) -> EnergyRunCostEstimatedV1:
    base: dict[str, Any] = dict(
        intent_id=settled.intent_id,
        workload_kind=settled.workload_kind,
        node=settled.node,
        gpu_index=settled.gpu_index,
        window_start=settled.window_start,
        window_end=settled.window_end,
        settlement_outcome=settled.outcome,
        tariff_version=ledger.tariff.version,
        cost_basis=ledger.tariff.cost_basis,
        computed_at=computed_at,
    )
    kwh, basis = _run_kwh(settled)
    if kwh is None:
        return EnergyRunCostEstimatedV1(
            **base, run_cost_gap="settlement_not_measured", house_share_gap="settlement_not_measured"
        )
    return EnergyRunCostEstimatedV1(
        **base,
        energy_kwh=kwh,
        energy_basis=basis,
        **_run_cost_fields(kwh, settled, ledger, usage_point_id),
        **_house_share_fields(kwh, settled, ledger, usage_point_id),
    )
