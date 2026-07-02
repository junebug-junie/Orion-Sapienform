from pathlib import Path

import yaml

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.vision import VisionEdgeActivityPayload

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = "orion:vision:edge:activity"


def _channels() -> dict[str, dict]:
    doc = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text(encoding="utf-8")) or {}
    return {e["name"]: e for e in doc.get("channels") or [] if isinstance(e, dict) and "name" in e}


def test_vision_edge_activity_channel_cataloged() -> None:
    entry = _channels()[CHANNEL]
    assert entry["schema_id"] == "VisionEdgeActivityPayload"
    assert entry["message_kind"] == "vision.edge.activity.v1"
    assert "orion-vision-edge" in entry["producer_services"]
    assert "orion-vision-frame-router" in entry["consumer_services"]


def test_vision_edge_activity_schema_registry_aligns_with_resolve() -> None:
    reg = SCHEMA_REGISTRY["VisionEdgeActivityPayload"]
    assert reg.kind == "vision.edge.activity.v1"
    assert resolve("VisionEdgeActivityPayload") is VisionEdgeActivityPayload
    assert reg.model is VisionEdgeActivityPayload
