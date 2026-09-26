"""Pure orchestration: intervals and settlements in, bus messages out.

No I/O here, so every publish decision is testable. main.py only loops and publishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from pydantic import BaseModel

from orion.energy.ledger import UsageLedger
from orion.energy.run_cost import estimate_run_cost
from orion.schemas.energy import (
    ENERGY_ACCRUED_KIND,
    ENERGY_RUN_COST_KIND,
    ENERGY_USAGE_KIND,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
from orion.schemas.power import PowerIntentSettledV1


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


class EnergyPipeline:
    def __init__(
        self,
        *,
        ledger: UsageLedger,
        channels: EnergyChannels,
        usage_point_id: Optional[str] = None,
        pending_hours: float = 96.0,
    ) -> None:
        self._ledger = ledger
        self._channels = channels
        self._configured_point = usage_point_id or None
        self._pending_hours = float(pending_hours)
        # Settlements whose cost is still partly unknown because utility data lags.
        self._pending: dict[str, PowerIntentSettledV1] = {}

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
            self._ledger.accrue_cycle(point, cycle_start, computed_at=datetime.now(timezone.utc))

    def ingest_intervals(self, intervals: Iterable[EnergyUsageIntervalV1], *, now: datetime) -> list[Outbound]:
        out: list[Outbound] = []
        affected: set[tuple[str, datetime]] = set()
        for iv in intervals:
            if self._ledger.upsert(iv):
                out.append(Outbound(self._channels.usage, ENERGY_USAGE_KIND, iv))
                affected.add((iv.usage_point_id, self._ledger.cycle_bounds(iv.interval_start)[0]))
        for point, cycle_start in sorted(affected):
            for accrued in self._ledger.accrue_cycle(point, cycle_start, computed_at=now):
                out.append(Outbound(self._channels.accrued, ENERGY_ACCRUED_KIND, accrued))
        if affected:
            out.extend(self._reprice_pending(now=now))
        return out

    def on_settlement(self, settled: PowerIntentSettledV1, *, now: datetime) -> list[Outbound]:
        est = self._estimate(settled, now=now)
        self._track(settled, est, now=now)
        return [Outbound(self._channels.run_cost, ENERGY_RUN_COST_KIND, est)]

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
