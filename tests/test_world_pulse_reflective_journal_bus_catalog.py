"""Bus catalog + schema registry alignment for World Pulse reflective journal."""

from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import _REGISTRY, resolve
from orion.schemas.world_pulse import WorldPulseRunResultV1
from orion.signals.registry import ORGAN_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

WORLD_PULSE_SIGNAL_CHANNELS = (
    "orion:world_pulse:run:result",
    "orion:world_pulse:digest:created",
    "orion:world_pulse:situation:brief:upsert",
    "orion:world_context:daily_capsule",
    "orion:world_pulse:graph:upsert",
)

WORLD_PULSE_JOURNAL_DOWNSTREAM = (
    "orion:journal:write",
    "orion:journal:created",
)


def _channel_index() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    return {c["name"]: c for c in (doc.get("channels") or []) if c.get("name")}


def test_world_pulse_run_result_channel_matches_handler_and_registry() -> None:
    channels = _channel_index()
    entry = channels["orion:world_pulse:run:result"]
    assert entry["message_kind"] == "world.pulse.run.result.v1"
    assert entry["schema_id"] == "WorldPulseRunResultV1"
    assert "orion-actions" in entry["consumer_services"]
    assert resolve(entry["schema_id"]) is WorldPulseRunResultV1
    assert "WorldPulseRunResultV1" in _REGISTRY


def test_world_pulse_organ_bus_channels_exist_in_channels_yaml() -> None:
    channels = _channel_index()
    organ_channels = ORGAN_REGISTRY["world_pulse"].bus_channels
    for name in organ_channels:
        assert name in channels, f"missing channel catalog entry for {name!r}"


def test_world_pulse_signal_adapter_channels_have_schema_ids() -> None:
    channels = _channel_index()
    for name in WORLD_PULSE_SIGNAL_CHANNELS:
        entry = channels[name]
        schema_id = entry["schema_id"]
        assert schema_id in _REGISTRY, f"{name} schema_id {schema_id!r} not in registry"
        assert resolve(schema_id) is _REGISTRY[schema_id]


def test_journal_downstream_channels_catalogued() -> None:
    channels = _channel_index()
    write = channels["orion:journal:write"]
    assert write["schema_id"] == "JournalEntryWriteV1"
    assert "orion-actions" in write["producer_services"]

    created = channels["orion:journal:created"]
    assert created["schema_id"] == "JournalEntryWriteV1"
    assert created["message_kind"] == "journal.entry.created.v1"
    assert "*" in created["consumer_services"] or "orion-actions" in created["consumer_services"]
