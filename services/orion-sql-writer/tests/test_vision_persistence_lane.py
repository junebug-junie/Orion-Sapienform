"""Regression tests for the vision SQL persistence lane in sql-writer."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
for mod_name in list(sys.modules):
    if mod_name == "app" or mod_name.startswith("app."):
        sys.modules.pop(mod_name, None)
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.settings import DEFAULT_ROUTE_MAP, settings  # noqa: E402


VISION_SQL_WRITE_CHANNEL = "orion:vision:events:sql-write"
VISION_EVENT_KIND = "vision.event.v1"


def test_vision_event_kind_maps_to_vision_event_sql():
    assert DEFAULT_ROUTE_MAP.get(VISION_EVENT_KIND) == "VisionEventSQL"
    assert VISION_EVENT_KIND in settings.route_map


def test_sql_writer_subscribes_to_vision_events_sql_write_channel():
    channels = settings.effective_subscribe_channels
    assert VISION_SQL_WRITE_CHANNEL in channels


def test_vision_artifacts_catalog_lists_vision_window_consumer():
    channels_path = REPO_ROOT / "orion" / "bus" / "channels.yaml"
    data = yaml.safe_load(channels_path.read_text(encoding="utf-8")) or {}
    entry = next(
        e for e in data.get("channels", []) if e.get("name") == "orion:vision:artifacts"
    )
    consumers = entry.get("consumer_services") or []
    assert "orion-vision-window" in consumers
    assert "orion-security-watcher" in consumers
    assert "orion-vision-council" in consumers
