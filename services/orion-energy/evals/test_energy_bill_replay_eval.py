"""Replay a full synthetic September cycle and check against a hand-computed bill.

The oracle below is written out from the published Schedule 1 numbers, not by
calling the tariff code, so a bug in the tariff/ledger cannot pass by agreeing
with itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.pipeline import EnergyChannels, EnergyPipeline
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import ENERGY_ACCRUED_KIND, ENERGY_RUN_COST_KIND, EnergyUsageIntervalV1
from orion.schemas.power import PowerIntentSettledV1

REPO = Path(__file__).resolve().parents[3]
DENVER = ZoneInfo("America/Denver")
CYCLE_START = datetime(2026, 9, 1, tzinfo=DENVER)
HOURS = 30 * 24  # Sept 1 - Oct 1 local
RETRIEVED = datetime(2026, 10, 2, tzinfo=timezone.utc)

# Hand oracle: 720 kWh in a summer cycle.
MULT = (1 + (7.63 - 0.53) / 100) * (1 + (1.17 + 3.84 + 0.17) / 100)
ORACLE_ENERGY = (400 * 0.098332 + 320 * 0.125263) * MULT
ORACLE_TOTAL = ORACLE_ENERGY + 12.00 + 0.16


def _pipeline() -> EnergyPipeline:
    led = UsageLedger(
        load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"),
        tz=DENVER, cycle_start_day=1,
    )
    return EnergyPipeline(ledger=led, channels=EnergyChannels("u", "a", "r"))


def _month() -> list[EnergyUsageIntervalV1]:
    start = CYCLE_START.astimezone(timezone.utc)
    return [
        EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="UP123",
            interval_start=start + timedelta(hours=h), interval_end=start + timedelta(hours=h + 1),
            energy_kwh=1.0, retrieved_at=RETRIEVED,
        )
        for h in range(HOURS)
    ]


def test_cycle_total_matches_hand_bill() -> None:
    out = _pipeline().ingest_intervals(_month(), now=RETRIEVED)
    accrued = [o.payload for o in out if o.kind == ENERGY_ACCRUED_KIND]
    assert len(accrued) == HOURS
    last = accrued[-1]
    assert last.cycle_accumulated_kwh == pytest.approx(720.0)
    assert last.cycle_energy_cost_usd == pytest.approx(ORACLE_ENERGY, rel=1e-9)
    assert last.cycle_to_date_total_usd == pytest.approx(ORACLE_TOTAL, rel=1e-9)
    assert ORACLE_TOTAL == pytest.approx(101.6214, abs=1e-3)


def test_same_gpu_hour_costs_more_after_block_400() -> None:
    p = _pipeline()
    p.ingest_intervals(_month(), now=RETRIEVED)
    start = CYCLE_START.astimezone(timezone.utc)

    def run_at(hour: int) -> float:
        t = start + timedelta(hours=hour)
        settled = PowerIntentSettledV1(
            intent_id=f"run-{hour}", workload_kind="reverie.diffusion", node="circe", gpu_index=2,
            outcome="settled", window_start=t, window_end=t + timedelta(hours=1), sample_count=3600,
            actual_mean_watts=300.0, actual_peak_watts=320.0, energy_joules=300.0 * 3600,
            baseline_watts=50.0,
        )
        est = [o.payload for o in p.on_settlement(settled, now=RETRIEVED) if o.kind == ENERGY_RUN_COST_KIND][0]
        assert 0.0 <= est.house_share_cost_usd <= est.estimated_run_cost_usd + 1e-12
        return est.estimated_run_cost_usd

    early, late = run_at(100), run_at(600)  # 100 kWh vs 600 kWh into the cycle
    assert early == pytest.approx(0.25 * 0.098332 * MULT)
    assert late == pytest.approx(0.25 * 0.125263 * MULT)
    assert late > early
