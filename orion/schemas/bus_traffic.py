"""Bus traffic rollup: one 5-minute bucket of bus traffic, as seen by orion-bus-mirror.

Patch 1 of docs/superpowers/specs/2026-11-11-causal-geometry-redesign.md. The mirror's raw
SQLite log keeps only 24 h and its FalkorDB graph keeps only running averages, so no bus
history survived past a day. This event carries one closed 5-minute bucket; orion-sql-writer
stores it as one row in `bus_traffic_rollup_5m`. The reader is the Patch 2 replay eval.

Channel `orion:bus:traffic:rollup`, kind `bus.traffic.rollup.v1`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BUS_TRAFFIC_ROLLUP_CHANNEL = "orion:bus:traffic:rollup"
BUS_TRAFFIC_ROLLUP_KIND = "bus.traffic.rollup.v1"
BUS_TRAFFIC_ROLLUP_BUCKET_SEC = 300


def _require_tz(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("bucket timestamps must be timezone-aware")
    return value


class BusTrafficHopV1(BaseModel):
    """Cross-organ hops seen in one bucket: a message from `to_organ` that shares a
    correlation_id with an earlier message from `from_organ` (orion-bus-mirror's ChainTracker)."""

    model_config = ConfigDict(extra="forbid")

    from_organ: str
    to_organ: str
    count: int = Field(ge=1)
    # Mean seconds between the prior organ's message and this one, as observed by the mirror.
    latency_mean_sec: float = Field(ge=0.0)


class BusTrafficRollupV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["bus.traffic.rollup.v1"] = "bus.traffic.rollup.v1"
    # Deterministic: f"{node}:{bucket_start.isoformat()}" -- a re-published bucket hits the
    # sql-writer's insert-only duplicate-key path instead of making a second row.
    rollup_id: str
    node: str
    bucket_start: datetime
    bucket_end: datetime
    # Every message the mirror received in the bucket, including undecodable ones.
    message_count: int = Field(ge=0)
    # Keyed by catalog-normalized channel name (orion.bus.census.normalize_channel_name), so
    # per-request reply channels collapse onto their wildcard catalog entry.
    channel_counts: dict[str, int] = Field(default_factory=dict)
    hops: list[BusTrafficHopV1] = Field(default_factory=list)
    decode_failures: int = Field(default=0, ge=0)

    @field_validator("bucket_start", "bucket_end")
    @classmethod
    def _bucket_times_are_tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)


# Column list of the `bus_traffic_rollup_5m` table, shared by the writer
# (services/orion-sql-writer/app/models/bus_traffic_rollup.py) and its shape test.
# Excludes `schema_version`: a fixed literal, never a column.
BUS_TRAFFIC_ROLLUP_SQL_COLUMNS: tuple[str, ...] = (
    "rollup_id",
    "node",
    "bucket_start",
    "bucket_end",
    "message_count",
    "channel_counts",
    "hops",
    "decode_failures",
)
