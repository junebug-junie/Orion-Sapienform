from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = "orion:cabinet:ambient:spike"


def _channels() -> dict[str, dict]:
    doc = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text(encoding="utf-8")) or {}
    return {e["name"]: e for e in doc.get("channels") or [] if isinstance(e, dict) and "name" in e}


def test_cabinet_ambient_spike_channel_cataloged() -> None:
    entry = _channels()[CHANNEL]
    assert entry["schema_id"] == "CabinetAmbientSpikeV1"
    assert entry["message_kind"] == "cabinet.ambient.spike.v1"
    assert "orion-biometrics" in entry["producer_services"]
    assert "orion-sql-writer" in entry["consumer_services"]


def test_cabinet_ambient_spike_schema_registry_aligns_with_resolve() -> None:
    reg = SCHEMA_REGISTRY["CabinetAmbientSpikeV1"]
    assert reg.kind == "cabinet.ambient.spike.v1"
    assert resolve("CabinetAmbientSpikeV1") is CabinetAmbientSpikeV1
    assert reg.model is CabinetAmbientSpikeV1
