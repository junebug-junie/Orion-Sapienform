"""Every pricing path uses its own period's length with the bill-verified (r2) tariff.

Aug 1 - Sep 1 is a 31-day cycle, so the first block is round(400 x 31/30) = 413 kWh.
At 405 kWh a 31-day period is still in block 1; a 30-day one would be in block 2.
Dropping `period_days` from any caller prices these at the block-2 rate and fails here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from orion.energy.ledger import UsageLedger
from orion.energy.reconcile import project_period, reconcile_actual
from orion.energy.run_cost import estimate_run_cost
from orion.energy.stakes import build_stakes_snapshot
from orion.energy.tariff import TariffError, load_tariff
from orion.schemas.energy import EnergyBillActualV1, EnergyImporterStatusV1, EnergyUsageIntervalV1
from orion.schemas.power import PowerIntentSettledV1

ROOT = Path(__file__).resolve().parents[3]
R2_PATH = ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.r2.yaml"
R2 = load_tariff(R2_PATH)
DENVER = ZoneInfo("America/Denver")
CYCLE_START = datetime(2026, 8, 1, tzinfo=DENVER)
CYCLE_END = datetime(2026, 9, 1, tzinfo=DENVER)
HOURS = 81  # 81 x 5 kWh = 405 kWh
COVERED = CYCLE_START + timedelta(hours=HOURS)
RATE1 = 0.098332 * R2.energy_multiplier
RATE2 = 0.125263 * R2.energy_multiplier
GOT = datetime(2026, 9, 5, tzinfo=timezone.utc)


def _ledger(hours: int = HOURS, kwh: float = 5.0, start: datetime = CYCLE_START) -> UsageLedger:
    led = UsageLedger(R2, tz=DENVER, cycle_start_day=1)
    first = start.astimezone(timezone.utc)
    for h in range(hours):
        led.upsert(EnergyUsageIntervalV1(
            source="file_drop", usage_point_id="UP1", interval_start=first + timedelta(hours=h),
            interval_end=first + timedelta(hours=h + 1), energy_kwh=kwh, retrieved_at=GOT,
        ))
    return led


def test_block_size_really_differs_between_30_and_31_days() -> None:
    assert R2.marginal_usd_per_kwh(cycle_kwh=405, month=8, period_days=31) == pytest.approx(RATE1)
    assert R2.marginal_usd_per_kwh(cycle_kwh=405, month=8, period_days=30) == pytest.approx(RATE2)


def test_stakes_prices_with_the_cycle_length() -> None:
    importer = EnergyImporterStatusV1(state="healthy", reason="x", source="file_drop", as_of=COVERED)
    snap = build_stakes_snapshot(
        ledger=_ledger(), usage_point_id="UP1", forecast=None, importer=importer,
        now=COVERED, near_ratio=1.0, over_ratio=1.1, stale_after_hours=48,
    )
    assert snap.cycle_accumulated_kwh == pytest.approx(405)
    assert snap.marginal_usd_per_kwh == pytest.approx(RATE1)
    assert snap.cycle_to_date_total_usd == pytest.approx(405 * RATE1 + R2.fixed_usd(31))


def test_projection_prices_with_the_period_length() -> None:
    led = _ledger()
    projection, gap, _ = project_period(led, "UP1", CYCLE_START, CYCLE_END)
    assert gap is None and projection is not None
    assert projection.fixed_usd == pytest.approx(R2.fixed_usd(31))
    remaining = projection.projected_kwh - 405
    expected = 405 * RATE1 + R2.energy_cost_usd(remaining, cycle_kwh_before=405, month=8, period_days=31)
    assert projection.projected_energy_usd == pytest.approx(expected)
    assert projection.projected_energy_usd == pytest.approx(405 * RATE1 + 8 * RATE1 + (remaining - 8) * RATE2)


def test_run_cost_prices_at_the_cycle_length_block_position() -> None:
    settled = PowerIntentSettledV1(
        intent_id="i-1", workload_kind="reverie.diffusion", node="circe", gpu_index=0, outcome="settled",
        window_start=COVERED.astimezone(timezone.utc), window_end=(COVERED + timedelta(hours=1)).astimezone(timezone.utc),
        sample_count=3600, actual_mean_watts=250.0, actual_peak_watts=300.0, energy_joules=250.0 * 3600,
        baseline_watts=50.0,
    )
    est = estimate_run_cost(settled, ledger=_ledger(), usage_point_id="UP1", computed_at=GOT)
    assert est.cycle_kwh_basis == pytest.approx(405)
    assert est.estimated_run_cost_usd == pytest.approx(0.2 * RATE1)  # (250 - 50) W x 1 h


def test_period_days_count_calendar_days_across_dst() -> None:
    led = _ledger(hours=0)
    start, end = datetime(2026, 10, 18, tzinfo=DENVER), datetime(2026, 11, 18, tzinfo=DENVER)
    elapsed = end.astimezone(timezone.utc) - start.astimezone(timezone.utc)
    assert elapsed.total_seconds() / 86400 == pytest.approx(31 + 1 / 24)  # fall back adds an hour
    assert led.period_days(start, end) == 31
    assert led.period_days(datetime(2026, 3, 1, tzinfo=DENVER), datetime(2026, 4, 1, tzinfo=DENVER)) == 31


def test_one_day_bill_with_a_credit_reconciles_instead_of_crashing() -> None:
    """Prorated fixed charges for 1 day ($0.40 + riders + $0.005) are less than the $0.50 credit."""
    led = _ledger(hours=24, kwh=1.0)
    bill = EnergyBillActualV1(
        source="file_drop", billing_period_start=date(2026, 8, 1), billing_period_end=date(2026, 8, 2),
        kwh_billed=24, current_charges=2.35, taxes=0.10, retrieved_at=GOT,
    )
    rec = reconcile_actual(bill, ledger=led, usage_point_id="UP1", computed_at=GOT)
    assert R2.fixed_usd(1) < 0
    assert rec.orion_fixed_usd == pytest.approx(R2.fixed_usd(1))
    # 1-day block is round(400 / 30) = 13 kWh.
    assert rec.orion_total_usd == pytest.approx(13 * RATE1 + 11 * RATE2 + R2.fixed_usd(1))


def _variant(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(R2_PATH.read_text())
    mutate(raw)
    out = tmp_path / "variant.yaml"
    out.write_text(yaml.safe_dump(raw))
    return out


def test_itemize_rounds_an_exact_half_cent_up(tmp_path) -> None:
    """7.63% of $50.00 is exactly $3.815; float math gives 3.81499999 and would print $3.81."""
    assert 7.63 / 100.0 * 50.0 < 3.815

    def flat(raw):
        for season in raw["seasons"].values():
            season["blocks"] = [{"up_to_kwh": 400, "cents_per_kwh": 10}, {"up_to_kwh": None, "cents_per_kwh": 10}]
        raw["riders"] = [{"name": "eba", "pct": 7.63, "applies_to": ["base_energy"]}]
        raw["fixed_monthly"], raw["per_bill"], raw["sales_tax"] = [], [], None

    est = load_tariff(_variant(tmp_path, flat)).itemize(500, month=7, period_days=30)
    assert dict(est.lines)["eba"] == 3.82
    assert est.total_usd == 53.82


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["riders"][0].update(applies_to=["base_energy", "base_energy"]), "counted twice"),
        (lambda r: r["riders"][0].update(pct=-120), "non-positive multiplier"),
        (lambda r: r["per_bill"][0].update(component="customer_charge"), "component"),
        (lambda r: r["fixed_monthly"][0].pop("component"), "no fixed_monthly item"),
        (lambda r: r["fixed_monthly"][1].update(name="sales_tax"), "reserved"),
        (lambda r: r["per_bill"][0].update(name="energy_block_1"), "reserved"),
        (lambda r: r["fixed_monthly"][1].update(name="schedule_92_deferral"), "unique"),
        (lambda r: r["fixed_monthly"][1].update(tax_exempt="false"), "true or false"),
        (lambda r: r["proration"].update(round_block_kwh="yes"), "true or false"),
        (lambda r: r["riders"][0].update(name=None), "name"),
    ],
)
def test_loader_rejects_configs_that_would_misprice_silently(tmp_path, mutate, match) -> None:
    with pytest.raises(TariffError, match=match):
        load_tariff(_variant(tmp_path, mutate))
