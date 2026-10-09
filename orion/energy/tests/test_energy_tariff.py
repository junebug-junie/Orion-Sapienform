from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orion.energy.tariff import TariffError, load_tariff

ROOT = Path(__file__).resolve().parents[3]
TARIFF = ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"
# (1 + 7.63% - 0.53%) * (1 + 1.17% + 3.84% + 0.17%), computed by hand.
M = 1.071 * 1.0518


def test_loads_published_numbers() -> None:
    t = load_tariff(TARIFF)
    assert t.version == "rmp-ut-sch1-2026-08-10"
    assert t.cost_basis == "pre_tax"
    assert t.energy_multiplier == pytest.approx(M)
    assert t.fixed_monthly_usd == pytest.approx(12.16)


def test_marginal_summer_first_block() -> None:
    t = load_tariff(TARIFF)
    assert t.marginal_usd_per_kwh(cycle_kwh=0.0, month=7) == pytest.approx(0.098332 * M)


def test_marginal_at_block_boundary_is_second_block() -> None:
    t = load_tariff(TARIFF)
    assert t.marginal_usd_per_kwh(cycle_kwh=400.0, month=9) == pytest.approx(0.125263 * M)


def test_marginal_winter() -> None:
    t = load_tariff(TARIFF)
    assert t.marginal_usd_per_kwh(cycle_kwh=10.0, month=1) == pytest.approx(0.087020 * M)


def test_energy_cost_splits_across_block_boundary() -> None:
    t = load_tariff(TARIFF)
    cost = t.energy_cost_usd(10.0, cycle_kwh_before=395.0, month=7)
    assert cost == pytest.approx((5 * 0.098332 + 5 * 0.125263) * M)


def test_zero_kwh_costs_zero() -> None:
    t = load_tariff(TARIFF)
    assert t.energy_cost_usd(0.0, cycle_kwh_before=100.0, month=7) == 0.0


def test_rejects_month_coverage_gap(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    raw["seasons"]["winter"]["months"] = [10, 11, 12, 1, 2, 3, 4]  # May missing
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="month"):
        load_tariff(bad)


def test_rejects_block_without_open_top(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    raw["seasons"]["summer"]["blocks"][-1]["up_to_kwh"] = 1000
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="open"):
        load_tariff(bad)


@pytest.fixture
def tariff():
    return load_tariff(TARIFF)


def test_energy_cost_rejects_nan_kwh(tariff) -> None:
    with pytest.raises(TariffError, match="kwh"):
        tariff.energy_cost_usd(float("nan"), cycle_kwh_before=0.0, month=7)


def test_energy_cost_rejects_negative_kwh(tariff) -> None:
    with pytest.raises(TariffError, match="kwh"):
        tariff.energy_cost_usd(-1.0, cycle_kwh_before=0.0, month=7)


def test_energy_cost_rejects_nan_cycle_kwh_before(tariff) -> None:
    with pytest.raises(TariffError, match="cycle_kwh_before"):
        tariff.energy_cost_usd(1.0, cycle_kwh_before=float("nan"), month=7)


def test_energy_cost_rejects_inf_kwh(tariff) -> None:
    with pytest.raises(TariffError, match="kwh"):
        tariff.energy_cost_usd(float("inf"), cycle_kwh_before=0.0, month=7)


def test_marginal_rejects_nan_cycle_kwh(tariff) -> None:
    with pytest.raises(TariffError, match="cycle_kwh"):
        tariff.marginal_usd_per_kwh(cycle_kwh=float("nan"), month=7)


def test_marginal_rejects_inf_cycle_kwh(tariff) -> None:
    with pytest.raises(TariffError, match="cycle_kwh"):
        tariff.marginal_usd_per_kwh(cycle_kwh=float("inf"), month=7)


def test_rejects_duplicate_block_bounds(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    raw["seasons"]["summer"]["blocks"] = [
        {"up_to_kwh": 400, "cents_per_kwh": 9.8332},
        {"up_to_kwh": 400, "cents_per_kwh": 12.5263},
        {"up_to_kwh": None, "cents_per_kwh": 12.5263},
    ]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="ascend"):
        load_tariff(bad)


def test_rejects_missing_tariff_version(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    del raw["tariff_version"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="tariff_version"):
        load_tariff(bad)


def test_rejects_empty_file(tmp_path: Path) -> None:
    bad = tmp_path / "empty.yaml"
    bad.write_text("")
    with pytest.raises(TariffError):
        load_tariff(bad)


def test_rejects_non_pre_tax_cost_basis(tmp_path: Path) -> None:
    raw = yaml.safe_load(TARIFF.read_text())
    raw["cost_basis"] = "post_tax"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(TariffError, match="cost_basis"):
        load_tariff(bad)


R2 = ROOT / "config/energy/tariff.rmp_ut_sch1.2026-08-10.r2.yaml"
# Energy: 1 - 0.53% + 7.63% + 1.17% + (3.84% + 0.17%) x (1 + 7.63% - 0.53%); customer charge: 1 + 1.17%.
M2 = 1 - 0.0053 + 0.0763 + 0.0117 + (0.0384 + 0.0017) * 1.071


def test_r2_rider_bases_give_hand_computed_multipliers() -> None:
    t = load_tariff(R2)
    assert t.version == "rmp-ut-sch1-2026-08-10-r2"
    assert t.energy_multiplier == pytest.approx(M2)
    assert t.customer_multiplier == pytest.approx(1.0117)


def test_r2_reproduces_the_2026_09_21_bill_to_the_cent() -> None:
    """Every printed line of the real bill: Aug 12 - Sep 18, 37 days, 3,772 kWh, $554.99."""
    est = load_tariff(R2).itemize(3772, month=9, period_days=37)
    assert dict(est.lines) == {
        "energy_block_1": 48.48,  # 493 kWh x 0.0983320
        "energy_block_2": 410.74,  # 3,279 kWh x 0.1252630
        "customer_charge_single_phase": 14.80,
        "schedule_91_home_electric_lifeline_program": 0.20,
        "schedule_98_renewable_energy_adjustment": -2.43,
        "schedule_94_energy_balancing_account": 35.04,
        "schedule_92_deferral": 5.55,
        "customer_efficiency_services": 18.89,
        "electric_vehicle_infrastructure": 0.84,
        "paperless_bill_credit": -0.50,
        "sales_tax": 23.38,
    }
    # The bill's own summary fields (as stored from the bill JSON).
    assert (est.energy_charge, est.customer_charge, est.adjustments, est.fees, est.credits) == (
        459.22, 14.80, 57.89, 0.20, -0.50,
    )
    assert (est.pre_tax_usd, est.taxes, est.total_usd) == (531.61, 23.38, 554.99)


def test_r2_block_size_prorates_with_period_length() -> None:
    t = load_tariff(R2)
    rate1, rate2 = 0.098332 * M2, 0.125263 * M2
    assert t.marginal_usd_per_kwh(cycle_kwh=450.0, month=8, period_days=37) == pytest.approx(rate1)  # 493 kWh block
    assert t.marginal_usd_per_kwh(cycle_kwh=450.0, month=8, period_days=30) == pytest.approx(rate2)  # 400 kWh block
    assert t.marginal_usd_per_kwh(cycle_kwh=492.9, month=8, period_days=37) == pytest.approx(rate1)
    assert t.marginal_usd_per_kwh(cycle_kwh=493.0, month=8, period_days=37) == pytest.approx(rate2)
    assert t.energy_cost_usd(10.0, cycle_kwh_before=488.0, month=8, period_days=37) == pytest.approx(
        5 * rate1 + 5 * rate2
    )


def test_r2_fixed_charges_prorate_and_carry_their_riders() -> None:
    t = load_tariff(R2)
    assert t.customer_charge_usd(37) == pytest.approx(14.80)
    assert t.fixed_usd(37) == pytest.approx(14.80 * 1.0117 + 0.16 * 37 / 30 - 0.50)
    assert t.fixed_usd(30) == pytest.approx(12.00 * 1.0117 + 0.16 - 0.50)


def test_legacy_tariff_ignores_period_length() -> None:
    t = load_tariff(TARIFF)
    assert t.proration_base_days is None and t.sales_tax_pct is None
    same = t.energy_cost_usd(600.0, cycle_kwh_before=0.0, month=7, period_days=37)
    assert same == pytest.approx(t.energy_cost_usd(600.0, cycle_kwh_before=0.0, month=7))
    assert t.fixed_usd(37) == pytest.approx(12.16)


def test_period_days_must_be_positive() -> None:
    with pytest.raises(TariffError, match="period_days"):
        load_tariff(R2).energy_cost_usd(1.0, cycle_kwh_before=0.0, month=7, period_days=0)


def _r2_variant(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(R2.read_text())
    mutate(raw)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    return bad


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["riders"][0].update(applies_to=["nope"]), "unknown"),
        (lambda r: r["riders"][0].update(applies_to=["schedule_94_energy_balancing_account"]), "later"),
        (lambda r: r["riders"].append(dict(r["riders"][0])), "duplicated"),
        (lambda r: r["riders"][0].update(applies_to=[]), "applies_to"),
        (lambda r: r.update(energy_adjustments={"on_base_pct": []}), "not both"),
        (lambda r: r["fixed_monthly"][0].update(component="energy"), "component"),
        (lambda r: r["proration"].update(base_days=0), "base_days"),
    ],
)
def test_r2_rejects_malformed_riders_and_charges(tmp_path, mutate, match) -> None:
    with pytest.raises(TariffError, match=match):
        load_tariff(_r2_variant(tmp_path, mutate))
