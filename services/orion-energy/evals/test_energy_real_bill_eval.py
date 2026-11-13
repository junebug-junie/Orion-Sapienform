"""The real RMP bills, replayed hour by hour through the ledger and reconcile.

Bill 1 (billing date 2026-09-21): service Aug 12 - Sep 18 2026, 37 days, 3,772 kWh,
$554.99 of which $23.38 tax. Bill 2 (2026-10-20): Sep 18 - Oct 19, 31 days spanning the
Oct 1 season change, 3,106 kWh, $427.39 of which $18.01 tax. Only the amounts are here --
no account or address. Reconcile prices the period like the bill (Tariff.itemize), so a
replay whose metered kWh equals the billed kWh reconciles to the cent; the line-by-line
check is test_energy_tariff.py.
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
BILL_2 = EnergyBillActualV1(
    source="file_drop", billing_period_start=date(2026, 9, 18), billing_period_end=date(2026, 10, 19),
    kwh_billed=3106, energy_charge=351.61, customer_charge=12.00, adjustments=46.11, fees=0.16,
    taxes=18.01, credits=-0.50, current_charges=427.39, retrieved_at=RETRIEVED,
)


def _ledger(bill: EnergyBillActualV1 = BILL) -> UsageLedger:
    """The bill's kWh spread over its hours (no DST change inside either bill)."""
    tariff = load_tariff(REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.r3.yaml")
    led = UsageLedger(tariff, tz=DENVER, cycle_start_day=bill.billing_period_start.day)
    start = datetime.combine(bill.billing_period_start, datetime.min.time(), DENVER).astimezone(timezone.utc)
    hours = (bill.billing_period_end - bill.billing_period_start).days * 24
    # Uneven hours (a daily swing) so block crossing lands mid-day, like a real house.
    weights = [1.0 + 0.6 * ((h % 24) in range(14, 22)) for h in range(hours)]
    scale = bill.kwh_billed / sum(weights)
    for h, w in enumerate(weights):
        led.upsert(EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="01", interval_start=start + timedelta(hours=h),
            interval_end=start + timedelta(hours=h + 1), energy_kwh=w * scale, retrieved_at=RETRIEVED,
        ))
    return led


def test_real_bill_reconciles_within_cents() -> None:
    rec = reconcile_actual(BILL, ledger=_ledger(), usage_point_id="01", computed_at=RETRIEVED)
    assert rec.reconcile_gap is None
    assert rec.tariff_version == "rmp-ut-sch1-2026-08-10-r3"
    assert (rec.utility_basis, rec.utility_total_usd) == ("pre_tax", pytest.approx(531.61))
    assert rec.delta_kwh == pytest.approx(0.0, abs=1e-6)
    assert abs(rec.delta_usd) < 0.05, rec.delta_usd
    for bucket in ("energy_charge", "energy_charge_plus_adjustments", "customer_charge"):
        assert abs(rec.bucket_deltas[bucket]) < 0.05, (bucket, rec.bucket_deltas[bucket])


def test_season_spanning_bill_reconciles_to_the_cent() -> None:
    """Bill 2 crosses Oct 1: r2 priced it all as winter and missed by ~$20; r3 splits by days."""
    rec = reconcile_actual(BILL_2, ledger=_ledger(BILL_2), usage_point_id="01", computed_at=RETRIEVED)
    assert rec.reconcile_gap is None
    assert (rec.utility_basis, rec.utility_total_usd) == ("pre_tax", pytest.approx(409.38))
    assert abs(rec.delta_usd) < 0.01, rec.delta_usd
    for bucket in ("energy_charge", "energy_charge_plus_adjustments", "customer_charge"):
        assert abs(rec.bucket_deltas[bucket]) < 0.01, (bucket, rec.bucket_deltas[bucket])


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
    """Ledger cycles run on ENERGY_BILLING_CYCLE_START_DAY: Aug 12 - Sep 12 is 31 days, a normal month
    under r3's normal_days [27, 33] -> the plain 400 kWh block (r2 stretched it to 413)."""
    led = _ledger()
    cycle_start = datetime(2026, 8, 12, tzinfo=DENVER)
    rows = led.accrue_cycle("01", cycle_start, computed_at=RETRIEVED)
    assert len(rows) == 31 * 24
    first_rate = led.tariff.marginal_usd_per_kwh(cycle_kwh=0.0, month=8, period_days=31)
    second_rate = led.tariff.marginal_usd_per_kwh(cycle_kwh=10_000.0, month=8, period_days=31)
    for row in rows:
        expected = first_rate if row.cycle_accumulated_kwh < 400 else second_rate
        assert row.marginal_usd_per_kwh == pytest.approx(expected)
    assert rows[-1].cycle_to_date_total_usd == pytest.approx(
        rows[-1].cycle_energy_cost_usd + led.tariff.fixed_usd(31)
    )
