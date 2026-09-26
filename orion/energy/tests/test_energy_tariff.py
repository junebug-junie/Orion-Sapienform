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
