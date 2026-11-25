from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.schemas.heartbeat_seat_rollup import (
    HEARTBEAT_SEAT_ROLLUP_CHANNEL,
    HEARTBEAT_SEAT_ROLLUP_KIND,
    HeartbeatSeatRollupV1,
)
from orion.schemas.registry import SCHEMA_REGISTRY, resolve

ROOT = Path(__file__).resolve().parents[1]


def test_rollup_channel_cataloged() -> None:
    raw = yaml.safe_load((ROOT / "orion/bus/channels.yaml").read_text())
    entry = {c["name"]: c for c in raw["channels"]}[HEARTBEAT_SEAT_ROLLUP_CHANNEL]
    assert entry["schema_id"] == "HeartbeatSeatRollupV1"
    assert entry["message_kind"] == HEARTBEAT_SEAT_ROLLUP_KIND
    assert entry["producer_services"] == ["orion-heartbeat"]
    assert entry["consumer_services"] == ["orion-sql-writer"]


def test_rollup_channel_is_exempt_from_the_integration_gate_with_a_reason() -> None:
    raw = yaml.safe_load((ROOT / "config/integration/organs.yaml").read_text())
    assert "telemetry" in raw["exempt_channels"][HEARTBEAT_SEAT_ROLLUP_CHANNEL]


def test_rollup_schema_registered_and_resolves() -> None:
    assert SCHEMA_REGISTRY["HeartbeatSeatRollupV1"].kind == HEARTBEAT_SEAT_ROLLUP_KIND
    assert resolve("HeartbeatSeatRollupV1") is HeartbeatSeatRollupV1


def test_rollup_rejects_naive_timestamps_negative_sums_and_extra_fields() -> None:
    base = dict(rollup_id="x", node="athena", seat="orion-hub", site_index=0, level_sum=0.0, atom_count=0,
                bucket_start="2026-11-25T00:00:00Z", bucket_end="2026-11-25T00:05:00Z")
    HeartbeatSeatRollupV1(**base)
    with pytest.raises(ValidationError):
        HeartbeatSeatRollupV1(**{**base, "bucket_start": datetime(2026, 11, 25)})
    with pytest.raises(ValidationError):
        HeartbeatSeatRollupV1(**{**base, "level_sum": -0.1})
    with pytest.raises(ValidationError):
        HeartbeatSeatRollupV1(**{**base, "surprise": 1})
