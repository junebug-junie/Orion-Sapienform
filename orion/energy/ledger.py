"""In-memory whole-house usage ledger.

Rebuilt from the processed drop directory at boot, so it needs no database of its
own. Accrual always recomputes a whole billing cycle in time order: a late interval
earlier in the cycle moves every later interval's block position.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from orion.energy.tariff import Tariff
from orion.schemas.energy import EnergyCostAccruedV1, EnergyUsageIntervalV1


class UsageLedger:
    def __init__(self, tariff: Tariff, *, tz: ZoneInfo, cycle_start_day: int) -> None:
        if not 1 <= int(cycle_start_day) <= 31:
            raise ValueError("cycle_start_day must be 1-31")
        self._tariff = tariff
        self._tz = tz
        self._cycle_start_day = int(cycle_start_day)
        self._intervals: dict[str, dict[datetime, EnergyUsageIntervalV1]] = {}
        self._accrued: dict[tuple[str, datetime], EnergyCostAccruedV1] = {}

    @property
    def tariff(self) -> Tariff:
        return self._tariff

    @property
    def tz(self) -> ZoneInfo:
        return self._tz

    def upsert(self, interval: EnergyUsageIntervalV1) -> bool:
        key = interval.interval_start.astimezone(timezone.utc)
        bucket = self._intervals.setdefault(interval.usage_point_id, {})
        existing = bucket.get(key)
        if existing is not None and existing.retrieved_at > interval.retrieved_at:
            return False
        bucket[key] = interval
        return True

    def usage_points(self) -> set[str]:
        return set(self._intervals)

    def _start_on(self, year: int, month: int) -> datetime:
        day = min(self._cycle_start_day, calendar.monthrange(year, month)[1])
        return datetime(year, month, day, tzinfo=self._tz)

    def cycle_bounds(self, ts: datetime) -> tuple[datetime, datetime]:
        local = ts.astimezone(self._tz)
        this = self._start_on(local.year, local.month)
        if local >= this:
            ny, nm = (local.year + 1, 1) if local.month == 12 else (local.year, local.month + 1)
            return this, self._start_on(ny, nm)
        py, pm = (local.year - 1, 12) if local.month == 1 else (local.year, local.month - 1)
        return self._start_on(py, pm), this

    def all_cycles(self) -> set[tuple[str, datetime]]:
        return {
            (point, self.cycle_bounds(start)[0])
            for point, bucket in self._intervals.items()
            for start in bucket
        }

    def _in_cycle(self, usage_point_id: str, start: datetime, end: datetime) -> list[EnergyUsageIntervalV1]:
        bucket = self._intervals.get(usage_point_id, {})
        return sorted(
            (iv for iv in bucket.values() if start <= iv.interval_start < end),
            key=lambda iv: iv.interval_start,
        )

    def accrue_cycle(
        self, usage_point_id: str, cycle_start: datetime, *, computed_at: datetime
    ) -> list[EnergyCostAccruedV1]:
        start, end = self.cycle_bounds(cycle_start)
        cycle_kwh = 0.0
        cycle_cost = 0.0
        out: list[EnergyCostAccruedV1] = []
        for iv in self._in_cycle(usage_point_id, start, end):
            month = iv.interval_start.astimezone(self._tz).month
            cost = self._tariff.energy_cost_usd(iv.energy_kwh, cycle_kwh_before=cycle_kwh, month=month)
            cycle_kwh += iv.energy_kwh
            cycle_cost += cost
            accrued = EnergyCostAccruedV1(
                usage_point_id=usage_point_id,
                interval_start=iv.interval_start,
                interval_end=iv.interval_end,
                energy_kwh=iv.energy_kwh,
                interval_cost_usd=cost,
                marginal_usd_per_kwh=self._tariff.marginal_usd_per_kwh(cycle_kwh=cycle_kwh, month=month),
                cycle_start=start.date(),
                cycle_accumulated_kwh=cycle_kwh,
                cycle_energy_cost_usd=cycle_cost,
                cycle_to_date_total_usd=cycle_cost + self._tariff.fixed_monthly_usd,
                tariff_version=self._tariff.version,
                cost_basis=self._tariff.cost_basis,
                computed_at=computed_at,
            )
            self._accrued[(usage_point_id, iv.interval_start.astimezone(timezone.utc))] = accrued
            out.append(accrued)
        return out

    def cycle_kwh_before(self, usage_point_id: str, ts: datetime) -> Optional[tuple[float, datetime]]:
        start, _ = self.cycle_bounds(ts)
        done = [iv for iv in self._in_cycle(usage_point_id, start, ts) if iv.interval_end <= ts]
        if not done:
            return None
        return sum(iv.energy_kwh for iv in done), max(iv.interval_end for iv in done)

    def intervals_overlapping(
        self, usage_point_id: str, start: datetime, end: datetime
    ) -> list[EnergyUsageIntervalV1]:
        bucket = self._intervals.get(usage_point_id, {})
        return sorted(
            (iv for iv in bucket.values() if iv.interval_start < end and iv.interval_end > start),
            key=lambda iv: iv.interval_start,
        )

    def interval_cost_usd(self, usage_point_id: str, interval_start: datetime) -> Optional[float]:
        accrued = self._accrued.get((usage_point_id, interval_start.astimezone(timezone.utc)))
        return None if accrued is None else accrued.interval_cost_usd
