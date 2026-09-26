from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.telemetry.home_cooling import HomeCoolingSampleV1

CHANNEL = "orion:home:cooling:sample"
ROOT = Path(__file__).resolve().parents[1]


def _channels() -> dict:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    return {c["name"]: c for c in raw["channels"]}


def test_home_cooling_channel_cataloged() -> None:
    entry = _channels()[CHANNEL]
    assert entry["schema_id"] == "HomeCoolingSampleV1"
    assert entry["message_kind"] == "home.cooling.sample.v1"
    assert "orion-zwave" in entry["producer_services"]
    assert "orion-sql-writer" in entry["consumer_services"]


def test_home_cooling_schema_registry_aligns_with_resolve() -> None:
    reg = SCHEMA_REGISTRY["HomeCoolingSampleV1"]
    assert reg.kind == "home.cooling.sample.v1"
    assert resolve("HomeCoolingSampleV1") is HomeCoolingSampleV1
