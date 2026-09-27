from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orion.schemas.energy import (
    EnergyCostAccruedV1,
    EnergyRunCostEstimatedV1,
    EnergyUsageIntervalV1,
)
from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("orion:energy:usage:observed", "EnergyUsageIntervalV1", "energy.usage.observed.v1", EnergyUsageIntervalV1),
    ("orion:energy:cost:accrued", "EnergyCostAccruedV1", "energy.cost.accrued.v1", EnergyCostAccruedV1),
    ("orion:energy:run_cost:estimated", "EnergyRunCostEstimatedV1", "energy.run_cost.estimated.v1", EnergyRunCostEstimatedV1),
]


def _channels() -> dict:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    return {c["name"]: c for c in raw["channels"]}


@pytest.mark.parametrize("channel,schema_id,kind,model", CASES)
def test_energy_channel_cataloged(channel, schema_id, kind, model) -> None:
    entry = _channels()[channel]
    assert entry["schema_id"] == schema_id
    assert entry["message_kind"] == kind
    assert "orion-energy" in entry["producer_services"]
    assert "orion-sql-writer" in entry["consumer_services"]


@pytest.mark.parametrize("channel,schema_id,kind,model", CASES)
def test_energy_schema_registered(channel, schema_id, kind, model) -> None:
    assert SCHEMA_REGISTRY[schema_id].kind == kind
    assert resolve(schema_id) is model


def test_energy_consumes_power_settlements() -> None:
    entry = _channels()["orion:power:intent:settled"]
    assert "orion-energy" in entry["consumer_services"]
