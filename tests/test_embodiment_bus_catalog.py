from __future__ import annotations

from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_YAML = ROOT / "orion" / "bus" / "channels.yaml"

EMBODIMENT_CHANNELS: dict[str, str] = {
    "orion:embodiment:intent": "EmbodimentIntentV1",
    "orion:embodiment:outcome": "EmbodimentOutcomeV1",
    "orion:embodiment:perception": "WorldPerceptionV1",
}


def _channel_index() -> dict[str, dict]:
    doc = yaml.safe_load(CHANNELS_YAML.read_text(encoding="utf-8"))
    return {c["name"]: c for c in (doc.get("channels") or []) if c.get("name")}


def test_embodiment_channels_exist_with_registry_schema_ids() -> None:
    channels = _channel_index()
    for name, schema_id in EMBODIMENT_CHANNELS.items():
        assert name in channels, f"missing channel catalog entry for {name!r}"
        assert channels[name]["schema_id"] == schema_id
        assert schema_id in SCHEMA_REGISTRY, f"{schema_id!r} not in SCHEMA_REGISTRY"
        assert resolve(schema_id) is SCHEMA_REGISTRY[schema_id].model


def test_persona_schema_registered() -> None:
    assert "OrionTownPersonaV1" in SCHEMA_REGISTRY
    assert resolve("OrionTownPersonaV1") is SCHEMA_REGISTRY["OrionTownPersonaV1"].model


def test_intent_producers_and_consumer() -> None:
    entry = _channel_index()["orion:embodiment:intent"]
    for producer in ("orion-substrate-runtime", "orion-harness-governor", "orion-cortex-exec"):
        assert producer in entry["producer_services"]
    assert "orion-embodiment" in entry["consumer_services"]


def test_perception_consumers() -> None:
    entry = _channel_index()["orion:embodiment:perception"]
    for consumer in ("orion-substrate-runtime", "orion-self-state-runtime", "orion-cortex-exec"):
        assert consumer in entry["consumer_services"]
