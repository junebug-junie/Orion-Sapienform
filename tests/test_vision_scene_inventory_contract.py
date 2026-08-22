"""Contract tests for the per-window scene census.

The census exists because the event stream cannot carry object permanence: the
council only re-interprets when the observed LABEL SET changes (it logs
`reason=stable_scene` otherwise), so a pure count change emits nothing, and a
departure is a non-event by nature. So this record is written on every window,
unconditionally, for a timer-driven reducer to read later.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from orion.schemas.registry import _REGISTRY, SCHEMA_REGISTRY, resolve
from orion.schemas.vision import VisionSceneInventoryV1

_REPO = pathlib.Path(__file__).resolve().parents[1]
_KIND = "vision.scene.inventory.v1"
_CHANNEL = "orion:vision:inventory:sql-write"


def test_schema_resolves_through_the_function_the_bus_actually_calls() -> None:
    """`resolve()` reads `_REGISTRY`, not `SCHEMA_REGISTRY`.

    Registering in only one of the two is a silent runtime failure: the live
    publish raises `ValueError: Unknown schema_id` while any check against the
    other registry passes. That is exactly how this shipped broken the first
    time -- caught by a deploy log, not by a test.
    """
    assert resolve("VisionSceneInventoryV1") is VisionSceneInventoryV1
    assert "VisionSceneInventoryV1" in _REGISTRY
    assert SCHEMA_REGISTRY["VisionSceneInventoryV1"].kind == _KIND


def test_channel_is_catalogued_with_the_right_producer_and_consumer() -> None:
    cat = yaml.safe_load((_REPO / "orion" / "bus" / "channels.yaml").read_text())
    chans = cat["channels"] if isinstance(cat, dict) and "channels" in cat else cat
    entry = next((c for c in chans if c.get("name") == _CHANNEL), None)
    assert entry, f"{_CHANNEL} missing from channels.yaml"
    assert entry["schema_id"] == "VisionSceneInventoryV1"
    assert entry["producer_services"] == ["orion-vision-window"]
    assert "orion-sql-writer" in entry["consumer_services"]


def test_sql_writer_actually_subscribes_to_the_channel() -> None:
    """The env list is authoritative and is NOT merged with the code default.

    `Settings.effective_subscribe_channels` returns
    `list(self.sql_writer_subscribe_channels)` -- so adding the channel to the
    code default alone leaves the live service deaf to it. (`route_map` DOES
    merge; the two behave differently, which is the trap.)
    """
    example = (_REPO / "services" / "orion-sql-writer" / ".env_example").read_text()
    line = next(
        (l for l in example.splitlines() if l.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")),
        None,
    )
    assert line, "SQL_WRITER_SUBSCRIBE_CHANNELS missing from .env_example"
    assert _CHANNEL in json.loads(line.split("=", 1)[1]), (
        f"{_CHANNEL} not in .env_example's subscribe list; the code default is "
        "not consulted when the env key is set"
    )


def test_route_map_sends_the_kind_to_the_inventory_table() -> None:
    from importlib.machinery import SourceFileLoader

    mod = SourceFileLoader(
        "sqlw_settings",
        str(_REPO / "services" / "orion-sql-writer" / "app" / "settings.py"),
    ).load_module()
    assert mod.DEFAULT_ROUTE_MAP[_KIND] == "VisionSceneInventorySQL"


def test_payload_keeps_counts_and_detections_apart() -> None:
    """These must never be conflated again.

    `counts` is the per-frame max (what is in the room). `detections` is the
    raw tally and scales with `frame_count`. Merging them is the bug that made
    Orion watch the furniture double and halve every few minutes.
    """
    m = VisionSceneInventoryV1(
        window_id="w1", stream_id="cam0", frame_count=2,
        counts={"chair": 2}, detections={"chair": 4}, believed_labels=["chair"],
    )
    assert m.counts["chair"] == 2 and m.detections["chair"] == 4
    assert m.schema_version == _KIND


def test_payload_rejects_unknown_fields_and_negative_frames() -> None:
    with pytest.raises(Exception):
        VisionSceneInventoryV1(window_id="w", image_path="/frames/x.jpg")  # privacy
    with pytest.raises(Exception):
        VisionSceneInventoryV1(window_id="w", frame_count=-1)


def test_zero_frames_is_representable() -> None:
    """Absence of evidence must be distinguishable from evidence of absence.

    A census built from 0 frames means "I did not look", not "the room is
    empty". The reducer has to be able to tell those apart.
    """
    m = VisionSceneInventoryV1(window_id="w", frame_count=0)
    assert m.frame_count == 0 and m.counts == {}
