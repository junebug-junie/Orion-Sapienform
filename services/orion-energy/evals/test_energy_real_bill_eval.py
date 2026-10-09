"""The first real RMP bill, replayed hour by hour through the ledger and reconcile.

Bill (billing date 2026-09-21): service Aug 12 - Sep 18 2026, 37 days, 3,772 kWh,
$554.99 of which $23.38 tax. Only the amounts are here -- no account or address.
The ledger prices each hour unrounded, so it lands within cents of the bill's
per-line-rounded total; the line-exact check is Tariff.itemize (test_energy_tariff.py).
The live ledger cannot reconcile this bill: RMP's portal throttling left Aug 7 - Sep 17
un-downloaded, so this replay is the evidence until the next bill (read ~Oct 19).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import reconcile_actual
from orion.energy.tariff import load_tariff
from orion.schemas.energy import EnergyBillActualV1, EnergyUsageIntervalV1

REPO = Path(__file__).resolve().parents[3]
DENVER = ZoneInfo("America/Denver")
RETRIEVED = datetime(2026, 9, 28, tzinfo=timezone.utc)
BILL = EnergyBillActualV1(
    source="file_drop", billing_period_start=date(2026, 8, 12), billing_period_end=date(2026, 9, 18),
    kwh_billed=3772, energy_charge=459.22, customer_charge=14.80, adjustments=57.89, fees=0.20,
    taxes=23.38, credits=-0.50, current_charges=554.99, retrieved_at=RETRIEVED,
)
HOURS = 37 * 24  # no DST change inside Aug 12 - Sep 18


def _ledger() -> UsageLedger:
    tariff = load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.r2.yaml")
    led = UsageLedger(tariff, tz=DENVER, cycle_start_day=12)
    start = datetime(2026, 8, 12, tzinfo=DENVER).astimezone(timezone.utc)
    # Uneven hours (a daily swing) so block crossing lands mid-day, like a real house.
    weights = [1.0 + 0.6 * ((h % 24) in range(14, 22)) for h in range(HOURS)]
    scale = BILL.kwh_billed / sum(weights)
    for h, w in enumerate(weights):
        led.upsert(EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="01", interval_start=start + timedelta(hours=h),
            interval_end=start + timedelta(hours=h + 1), energy_kwh=w * scale, retrieved_at=RETRIEVED,
        ))
    return led


def test_real_bill_reconciles_within_cents() -> None:
    rec = reconcile_actual(BILL, ledger=_ledger(), usage_point_id="01", computed_at=RETRIEVED)
    assert rec.reconcile_gap is None
    assert rec.tariff_version == "rmp-ut-sch1-2026-08-10-r2"
    assert (rec.utility_basis, rec.utility_total_usd) == ("pre_tax", pytest.approx(531.61))
    assert rec.delta_kwh == pytest.approx(0.0, abs=1e-6)
    assert abs(rec.delta_usd) < 0.05, rec.delta_usd
    for bucket in ("energy_charge", "energy_charge_plus_adjustments", "customer_charge"):
        assert abs(rec.bucket_deltas[bucket]) < 0.05, (bucket, rec.bucket_deltas[bucket])


def test_the_old_tariff_got_the_parts_wrong() -> None:
    """Why r2 exists. The old file's total is only ~$0.67 off, but only because two errors cancel:
    a flat 400 kWh block over-prices energy by ~$2.50, and unprorated fixed charges ($12.16
    against the bill's $14.80 customer charge) under-price it by $2.64."""
    tariff = load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml")
    old = UsageLedger(tariff, tz=DENVER, cycle_start_day=12)
    for iv in _ledger().intervals_overlapping(
        "01", datetime(2026, 8, 1, tzinfo=DENVER), datetime(2026, 10, 1, tzinfo=DENVER)
    ):
        old.upsert(iv)
    rec = reconcile_actual(BILL, ledger=old, usage_point_id="01", computed_at=RETRIEVED)
    assert rec.bucket_deltas["energy_charge"] > 2.0
    assert rec.bucket_deltas["customer_charge"] < -2.0


def test_ledger_block_size_follows_its_own_cycle_length() -> None:
    """Ledger cycles run on ENERGY_BILLING_CYCLE_START_DAY: Aug 12 - Sep 12 is 31 days -> 413 kWh block."""
    led = _ledger()
    cycle_start = datetime(2026, 8, 12, tzinfo=DENVER)
    rows = led.accrue_cycle("01", cycle_start, computed_at=RETRIEVED)
    assert len(rows) == 31 * 24
    first_rate = led.tariff.marginal_usd_per_kwh(cycle_kwh=0.0, month=8, period_days=31)
    second_rate = led.tariff.marginal_usd_per_kwh(cycle_kwh=10_000.0, month=8, period_days=31)
    for row in rows:
        expected = first_rate if row.cycle_accumulated_kwh < 413 else second_rate
        assert row.marginal_usd_per_kwh == pytest.approx(expected)
    assert rows[-1].cycle_to_date_total_usd == pytest.approx(
        rows[-1].cycle_energy_cost_usd + led.tariff.fixed_usd(31)
    )
