"""Hand-checkable fixtures shared by orion/energy and services/orion-energy tests.

Not used at runtime. The flat tariff makes every oracle doable on paper:
400 kWh at $0.10, then $0.12, $10 fixed, no riders, one season.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from orion.energy.ledger import UsageLedger
from orion.energy.tariff import Block, Season, Tariff
from orion.schemas.energy import EnergyUsageIntervalV1

UTC = ZoneInfo("UTC")


def flat_test_tariff() -> Tariff:
    return Tariff(
        version="test-flat-v1",
        cost_basis="pre_tax",
        seasons=(
            Season(
                name="all",
                months=frozenset(range(1, 13)),
                blocks=(Block(up_to_kwh=400.0, usd_per_kwh=0.10), Block(up_to_kwh=None, usd_per_kwh=0.12)),
            ),
        ),
        energy_multiplier=1.0,
        fixed_monthly_usd=10.0,
    )


def make_test_ledger(cycle_start_day: int = 1) -> UsageLedger:
    return UsageLedger(flat_test_tariff(), tz=UTC, cycle_start_day=cycle_start_day)


def hourly(
    start: datetime,
    hours: int,
    *,
    kwh: float = 1.0,
    point: str = "UP1",
    retrieved_at: Optional[datetime] = None,
    skip: frozenset[int] = frozenset(),
) -> list[EnergyUsageIntervalV1]:
    got = retrieved_at or (start + timedelta(days=60))
    start_utc = start.astimezone(timezone.utc)
    return [
        EnergyUsageIntervalV1(
            source="file_drop",
            usage_point_id=point,
            interval_start=start_utc + timedelta(hours=h),
            interval_end=start_utc + timedelta(hours=h + 1),
            energy_kwh=kwh,
            retrieved_at=got,
        )
        for h in range(hours)
        if h not in skip
    ]
