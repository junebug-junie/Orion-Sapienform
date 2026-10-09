"""Pure orchestration: intervals and settlements in, bus messages out.

No I/O here, so every publish decision is testable. main.py only loops and publishes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional, Union

from pydantic import BaseModel

from orion.energy.importer_status import PortalStatus, compute_importer_status
from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import period_bounds, reconcile_actual, reconcile_forecast
from orion.energy.run_cost import estimate_run_cost
from orion.energy.stakes import build_stakes_snapshot, current_forecast
from orion.schemas.energy import (
    ENERGY_ACCRUED_KIND,
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    ENERGY_IMPORTER_STATUS_KIND,
    ENERGY_RECONCILE_KIND,
    ENERGY_RUN_COST_KIND,
    ENERGY_STAKES_KIND,
    ENERGY_USAGE_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
from orion.schemas.power import PowerIntentSettledV1

logger = logging.getLogger("orion-energy.pipeline")


@dataclass(frozen=True)
class Outbound:
    channel: str
    kind: str
    payload: BaseModel


@dataclass(frozen=True)
class EnergyChannels:
    usage: str
    accrued: str
    run_cost: str
    bill_actual: str = "orion:energy:bill:actual"
    bill_forecast: str = "orion:energy:bill:forecast"
    reconcile: str = "orion:energy:reconcile"
    stakes: str = "orion:energy:stakes:snapshot"
    importer_status: str = "orion:energy:importer:status"


@dataclass(frozen=True)
class StakesConfig:
    near_ratio: float = 1.0
    over_ratio: float = 1.10
    stale_after_hours: float = 48.0
    portal_enabled: bool = False
    portal_interval_hours: float = 24.0


Bill = Union[EnergyBillActualV1, EnergyBillForecastV1]


class EnergyPipeline:
    def __init__(
        self,
        *,
        ledger: UsageLedger,
        channels: EnergyChannels,
        usage_point_id: Optional[str] = None,
        pending_hours: float = 96.0,
        stakes: StakesConfig = StakesConfig(),
    ) -> None:
        self._ledger = ledger
        self._channels = channels
        self._configured_point = usage_point_id or None
        self._pending_hours = float(pending_hours)
        # Settlements whose cost is still partly unknown because utility data lags.
        self._pending: dict[str, PowerIntentSettledV1] = {}
        self._stakes = stakes
        # Bills keyed by natural key; a newer retrieval of the same bill wins.
        self._actuals: dict[tuple[date, date], EnergyBillActualV1] = {}
        self._forecasts: dict[tuple[date, datetime], EnergyBillForecastV1] = {}

    def usage_point(self) -> Optional[str]:
        if self._configured_point:
            return self._configured_point
        points = self._ledger.usage_points()
        return next(iter(points)) if len(points) == 1 else None

    def pending_count(self) -> int:
        return len(self._pending)

    def replay(self, intervals: Iterable[EnergyUsageIntervalV1]) -> None:
        for iv in intervals:
            self._ledger.upsert(iv)
        for point, cycle_start in sorted(self._ledger.all_cycles()):
            self._accrue_cycle(point, cycle_start, computed_at=datetime.now(timezone.utc))

    def ingest_intervals(self, intervals: Iterable[EnergyUsageIntervalV1], *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        affected: set[tuple[str, datetime]] = set()
        lo: Optional[datetime] = None
        hi: Optional[datetime] = None
        for iv in intervals:
            if self._ledger.upsert(iv):
                out.append(Outbound(self._channels.usage, ENERGY_USAGE_KIND, iv))
                affected.add((iv.usage_point_id, self._ledger.cycle_bounds(iv.interval_start)[0]))
                lo = iv.interval_start if lo is None else min(lo, iv.interval_start)
                hi = iv.interval_end if hi is None else max(hi, iv.interval_end)
        for point, cycle_start in sorted(affected):
            for accrued in self._accrue_cycle(point, cycle_start, computed_at=now):
                out.append(Outbound(self._channels.accrued, ENERGY_ACCRUED_KIND, accrued))
        if affected and lo is not None and hi is not None:
            out.extend(self._reprice_pending(now=now))
            out.extend(self._rereconcile(lo, hi, now=now))
        return out

    def _store_bill(self, bill: Bill) -> bool:
        if isinstance(bill, EnergyBillActualV1):
            key = (bill.billing_period_start, bill.billing_period_end)
            old = self._actuals.get(key)
            if old is not None and old.retrieved_at > bill.retrieved_at:
                return False
            self._actuals[key] = bill
            return True
        fkey = (bill.billing_period_start, bill.as_of)
        old_fc = self._forecasts.get(fkey)
        if old_fc is not None and old_fc.retrieved_at > bill.retrieved_at:
            return False
        self._forecasts[fkey] = bill
        return True

    def replay_bills(self, bills: Iterable[Bill]) -> None:
        for bill in bills:
            self._store_bill(bill)

    def ingest_bills(self, bills: Iterable[Bill], *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        for bill in bills:
            if not self._store_bill(bill):
                continue
            if isinstance(bill, EnergyBillActualV1):
                out.append(Outbound(self._channels.bill_actual, ENERGY_BILL_ACTUAL_KIND, bill))
            else:
                out.append(Outbound(self._channels.bill_forecast, ENERGY_BILL_FORECAST_KIND, bill))
            rec = self._reconcile(bill, now=now)
            if rec is not None:
                out.append(rec)
        return out

    def _reconcile(self, bill: Bill, *, now: datetime) -> Optional[Outbound]:
        point = bill.usage_point_id or self.usage_point()
        if point is None:
            logger.warning("energy_reconcile_skipped reason=no_usage_point period_start=%s", bill.billing_period_start)
            return None
        fn = reconcile_actual if isinstance(bill, EnergyBillActualV1) else reconcile_forecast
        rec = fn(bill, ledger=self._ledger, usage_point_id=point, computed_at=now)
        return Outbound(self._channels.reconcile, ENERGY_RECONCILE_KIND, rec)

    def _latest_forecasts(self) -> list[EnergyBillForecastV1]:
        latest: dict[date, EnergyBillForecastV1] = {}
        for fc in self._forecasts.values():
            cur = latest.get(fc.billing_period_start)
            if cur is None or fc.as_of > cur.as_of:
                latest[fc.billing_period_start] = fc
        return list(latest.values())

    def _rereconcile(self, lo: datetime, hi: datetime, *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        for bill in [*self._actuals.values(), *self._latest_forecasts()]:
            start, end = period_bounds(bill.billing_period_start, bill.billing_period_end, self._ledger)
            if start < hi and lo < end:
                rec = self._reconcile(bill, now=now)
                if rec is not None:
                    out.append(rec)
        return out

    def status_tick(
        self, *, now: datetime, portal: Optional[PortalStatus], last_file_at: Optional[datetime]
    ) -> list[Outbound]:
        point = self.usage_point()
        cfg = self._stakes
        importer = compute_importer_status(
            portal_enabled=cfg.portal_enabled, portal=portal, portal_interval_hours=cfg.portal_interval_hours,
            latest_interval_end=None if point is None else self._ledger.latest_interval_end(point),
            last_file_at=last_file_at, now=now, stale_after_hours=cfg.stale_after_hours,
        )
        snapshot = build_stakes_snapshot(
            ledger=self._ledger, usage_point_id=point,
            forecast=current_forecast(self._forecasts.values(), now=now, tz=self._ledger.tz),
            importer=importer, now=now, near_ratio=cfg.near_ratio, over_ratio=cfg.over_ratio,
            stale_after_hours=cfg.stale_after_hours,
        )
        return [
            Outbound(self._channels.importer_status, ENERGY_IMPORTER_STATUS_KIND, importer),
            Outbound(self._channels.stakes, ENERGY_STAKES_KIND, snapshot),
        ]

    def on_settlement(self, settled: PowerIntentSettledV1, *, now: datetime) -> list[Outbound]:
        est = self._estimate(settled, now=now)
        self._track(settled, est, now=now)
        return [Outbound(self._channels.run_cost, ENERGY_RUN_COST_KIND, est)]

    def _accrue_cycle(
        self, point: str, cycle_start: datetime, *, computed_at: datetime
    ) -> list[EnergyCostAccruedV1]:
        accrued_rows = self._ledger.accrue_cycle(point, cycle_start, computed_at=computed_at)
        priced, total, covered_through = self._ledger.cycle_coverage(point, cycle_start)
        if priced < total:
            gap_at = covered_through if covered_through is not None else cycle_start
            logger.warning(
                "energy_cycle_incomplete usage_point=%s cycle_start=%s gap_at=%s priced=%d held=%d",
                point,
                cycle_start.date(),
                gap_at.isoformat(),
                priced,
                total - priced,
            )
        return accrued_rows

    def _estimate(self, settled: PowerIntentSettledV1, *, now: datetime) -> EnergyRunCostEstimatedV1:
        return estimate_run_cost(settled, ledger=self._ledger, usage_point_id=self.usage_point(), computed_at=now)

    def _track(self, settled: PowerIntentSettledV1, est: EnergyRunCostEstimatedV1, *, now: datetime) -> None:
        incomplete = est.run_cost_gap == "no_cycle_usage" or est.house_share_gap == "house_interval_missing"
        if incomplete and est.run_cost_gap != "settlement_not_measured":
            self._pending[settled.intent_id] = settled
        else:
            self._pending.pop(settled.intent_id, None)
        cutoff = now - timedelta(hours=self._pending_hours)
        for intent_id, pending in list(self._pending.items()):
            if pending.window_end < cutoff:
                del self._pending[intent_id]

    def _reprice_pending(self, *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        for settled in list(self._pending.values()):
            est = self._estimate(settled, now=now)
            self._track(settled, est, now=now)
            if settled.intent_id in self._pending or est.house_share_cost_usd is not None:
                out.append(Outbound(self._channels.run_cost, ENERGY_RUN_COST_KIND, est))
        return out
