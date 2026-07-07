from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve

CHANNELS_YAML = Path(__file__).resolve().parents[1] / "orion" / "bus" / "channels.yaml"

CHANNEL_NAME = "orion:substrate:brain_frame"
SCHEMA_ID = "SubstrateBrainFrameV1"


def _channel_index() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    return {c["name"]: c for c in (doc.get("channels") or []) if c.get("name")}


def test_brain_frame_channel_exists_with_registry_schema():
    channels = _channel_index()
    assert CHANNEL_NAME in channels, f"missing channel catalog entry for {CHANNEL_NAME!r}"
    entry = channels[CHANNEL_NAME]
    assert entry["schema_id"] == SCHEMA_ID
    assert entry["message_kind"] == "substrate.brain_frame.v1"
    assert "orion-substrate-runtime" in entry["producer_services"]
    assert "orion-hub" in entry["consumer_services"]
    assert SCHEMA_ID in SCHEMA_REGISTRY
    assert resolve(SCHEMA_ID) is SCHEMA_REGISTRY[SCHEMA_ID].model
