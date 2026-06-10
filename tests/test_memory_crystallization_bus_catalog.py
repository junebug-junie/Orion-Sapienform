"""Bus catalog and schema registry coverage for memory crystallization artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.memory_crystallization import ActiveMemoryPacketV1, MemoryCrystallizationV1
from orion.schemas.registry import _REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

LIFECYCLE_CHANNELS = {
    "orion:memory:crystallization:proposed": "memory.crystallization.proposed.v1",
    "orion:memory:crystallization:validated": "memory.crystallization.validated.v1",
    "orion:memory:crystallization:approved": "memory.crystallization.approved.v1",
    "orion:memory:crystallization:rejected": "memory.crystallization.rejected.v1",
    "orion:memory:crystallization:quarantined": "memory.crystallization.quarantined.v1",
    "orion:memory:crystallization:project": "memory.crystallization.project.v1",
}


def _channels() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8")) or {}
    return {
        entry["name"]: entry
        for entry in (doc.get("channels") or [])
        if isinstance(entry, dict) and "name" in entry
    }


def test_crystallization_schemas_in_registry() -> None:
    assert "MemoryCrystallizationV1" in _REGISTRY
    assert resolve("MemoryCrystallizationV1") is MemoryCrystallizationV1
    assert "ActiveMemoryPacketV1" in _REGISTRY
    assert resolve("ActiveMemoryPacketV1") is ActiveMemoryPacketV1


def test_crystallization_lifecycle_channels_cataloged() -> None:
    channels = _channels()
    for name, message_kind in LIFECYCLE_CHANNELS.items():
        entry = channels.get(name)
        assert entry is not None, f"missing channel {name}"
        assert entry["schema_id"] == "MemoryCrystallizationV1"
        assert entry["message_kind"] == message_kind


def test_retrieved_channel_carries_active_packet() -> None:
    channels = _channels()
    entry = channels.get("orion:memory:crystallization:retrieved")
    assert entry is not None
    assert entry["schema_id"] == "ActiveMemoryPacketV1"
    assert entry["message_kind"] == "memory.crystallization.retrieved.v1"


def test_existing_vector_channel_unchanged() -> None:
    channels = _channels()
    entry = channels.get("orion:memory:vector:upsert")
    assert entry is not None
    assert entry["schema_id"] == "VectorDocumentUpsertV1"
