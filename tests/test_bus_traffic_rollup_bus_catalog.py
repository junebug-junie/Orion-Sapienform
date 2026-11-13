from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.schemas.bus_traffic import (
    BUS_TRAFFIC_ROLLUP_CHANNEL,
    BUS_TRAFFIC_ROLLUP_KIND,
    BusTrafficRollupV1,
)
from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]


def test_rollup_channel_cataloged() -> None:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    entry = {c["name"]: c for c in raw["channels"]}[BUS_TRAFFIC_ROLLUP_CHANNEL]
    assert entry["schema_id"] == "BusTrafficRollupV1"
    assert entry["message_kind"] == BUS_TRAFFIC_ROLLUP_KIND
    assert entry["producer_services"] == ["orion-bus-mirror"]
    assert entry["consumer_services"] == ["orion-sql-writer"]


def test_rollup_schema_registered_and_resolves() -> None:
    assert SCHEMA_REGISTRY["BusTrafficRollupV1"].kind == BUS_TRAFFIC_ROLLUP_KIND
    assert resolve("BusTrafficRollupV1") is BusTrafficRollupV1


def test_rollup_rejects_naive_timestamps_and_extra_fields() -> None:
    base = dict(rollup_id="x", node="athena", message_count=0,
                bucket_start=datetime(2026, 11, 13), bucket_end=datetime(2026, 11, 13, 0, 5))
    with pytest.raises(ValidationError):
        BusTrafficRollupV1(**base)
    with pytest.raises(ValidationError):
        BusTrafficRollupV1(**{**base, "bucket_start": "2026-11-13T00:00:00Z",
                              "bucket_end": "2026-11-13T00:05:00Z", "surprise": 1})
