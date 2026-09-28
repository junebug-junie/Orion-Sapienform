"""Reconcile against the REAL tariff with Plan 1's hand oracle, plus the file-drop path end to end.

The oracle is written from published Schedule 1 numbers (same as the bill-replay eval),
so reconcile cannot pass by agreeing with the tariff code.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.bills import scan_bills
from app.pipeline import EnergyChannels, EnergyPipeline
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.energy import (
    ENERGY_RECONCILE_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
    EnergyUsageIntervalV1,
)

REPO = Path(__file__).resolve().parents[3]
DENVER = ZoneInfo("America/Denver")
START = datetime(2026, 9, 1, tzinfo=DENVER).astimezone(timezone.utc)
RETRIEVED = datetime(2026, 10, 2, tzinfo=timezone.utc)

MULT = (1 + (7.63 - 0.53) / 100) * (1 + (1.17 + 3.84 + 0.17) / 100)
ORACLE_BASE_ENERGY = 400 * 0.098332 + 320 * 0.125263
ORACLE_ENERGY = ORACLE_BASE_ENERGY * MULT
ORACLE_TOTAL = ORACLE_ENERGY + 12.00 + 0.16


def _pipeline() -> EnergyPipeline:
    led = UsageLedger(
        load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"), tz=DENVER, cycle_start_day=1,
    )
    return EnergyPipeline(ledger=led, channels=EnergyChannels("u", "a", "r"), usage_point_id="UP123")


def _hours(n: int) -> list[EnergyUsageIntervalV1]:
    return [
        EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="UP123", interval_start=START + timedelta(hours=h),
            interval_end=START + timedelta(hours=h + 1), energy_kwh=1.0, retrieved_at=RETRIEVED,
        )
        for h in range(n)
    ]


def test_matching_bill_reconciles_to_zero() -> None:
    p = _pipeline()
    p.ingest_intervals(_hours(720), now=RETRIEVED)
    bill = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        kwh_billed=720.0, energy_charge=ORACLE_BASE_ENERGY, adjustments=ORACLE_ENERGY - ORACLE_BASE_ENERGY,
        taxes=5.0, current_charges=ORACLE_TOTAL + 5.0, retrieved_at=RETRIEVED,
    )  # the real bill prints block charges and riders on separate lines
    rec = [o.payload for o in p.ingest_bills([bill], now=RETRIEVED) if o.kind == ENERGY_RECONCILE_KIND][0]
    assert rec.utility_basis == "pre_tax"
    assert rec.orion_total_usd == pytest.approx(ORACLE_TOTAL, rel=1e-9)
    assert rec.delta_usd == pytest.approx(0.0, abs=1e-6)
    assert rec.delta_kwh == pytest.approx(0.0)
    assert rec.bucket_deltas["energy_charge"] == pytest.approx(0.0, abs=1e-6)
    assert rec.bucket_deltas["energy_charge_plus_adjustments"] == pytest.approx(0.0, abs=1e-6)


def test_ten_flat_days_project_to_the_full_month_oracle() -> None:
    p = _pipeline()
    p.ingest_intervals(_hours(240), now=RETRIEVED)
    fc = EnergyBillForecastV1(
        source="file_drop", billing_period_start=date(2026, 9, 1), billing_period_end=date(2026, 10, 1),
        as_of=START + timedelta(days=10), projected_total_usd=ORACLE_TOTAL, retrieved_at=START + timedelta(days=10),
    )
    rec = [o.payload for o in p.ingest_bills([fc], now=RETRIEVED) if o.kind == ENERGY_RECONCILE_KIND][0]
    assert rec.orion_kwh == pytest.approx(720.0)
    assert rec.orion_total_usd == pytest.approx(ORACLE_TOTAL, rel=1e-9)
    assert rec.delta_usd == pytest.approx(0.0, abs=1e-6)


def test_hand_entered_bill_file_reaches_reconcile(tmp_path) -> None:
    inbox, processed = tmp_path / "bills/inbox", tmp_path / "bills/processed"
    inbox.mkdir(parents=True)
    (inbox / "sept.json").write_text(json.dumps({
        "kind": "energy.bill.actual.v1", "billing_period_start": "2026-09-01",
        "billing_period_end": "2026-10-01", "kwh_billed": 720, "current_charges": round(ORACLE_TOTAL, 2),
    }))
    p = _pipeline()
    p.ingest_intervals(_hours(720), now=RETRIEVED)
    out = p.ingest_bills(scan_bills(inbox, processed, now=RETRIEVED), now=RETRIEVED)
    rec = [o.payload for o in out if o.kind == ENERGY_RECONCILE_KIND][0]
    assert rec.utility_basis == "tax_unknown"
    assert abs(rec.delta_usd) < 0.01
