from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.telemetry.office_env import OfficeEnvSampleV1

CHANNEL = "orion:office:env:sample"
ROOT = Path(__file__).resolve().parents[1]


def _channels() -> dict:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    return {c["name"]: c for c in raw["channels"]}


def test_office_env_channel_cataloged() -> None:
    entry = _channels()[CHANNEL]
    assert entry["schema_id"] == "OfficeEnvSampleV1"
    assert entry["message_kind"] == "office.env.sample.v1"
    assert entry["producer_services"] == ["orion-office-env"]
    assert "orion-sql-writer" in entry["consumer_services"]


def test_office_env_schema_registry_aligns_with_resolve() -> None:
    reg = SCHEMA_REGISTRY["OfficeEnvSampleV1"]
    assert reg.kind == "office.env.sample.v1"
    assert resolve("OfficeEnvSampleV1") is OfficeEnvSampleV1
