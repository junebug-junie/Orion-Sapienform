"""Heartbeat seat rollup: one heartbeat seat in one 5-minute bucket, as seen by orion-heartbeat.

Issue #3886. `grammar_events` keeps about 3 days, so the field-demon network replay
(scripts/demon_network_replay.py, spec docs/superpowers/specs/2026-11-24-heartbeat-seats-field-demon-design.md
rev 3) could not read the 14-28 days of seat history it needs. orion-heartbeat already routes every
atom to a seat (`route_atom`), so it sums the gate levels per seat per bucket and publishes one event
per seat per closed bucket. orion-sql-writer stores each as one row in `heartbeat_seat_rollup_5m`.

The value is the same one the replay computes from `grammar_events`: the summed `route_atom` level of
the atoms a seat absorbs in a bucket keyed by the atom's `emitted_at`. A missing row means the
heartbeat was not listening (unknown), not zero. A row with `partial=True` is an undercount: readers
treat it as missing.

Channel `orion:heartbeat:seat:rollup`, kind `heartbeat.seat.rollup.v1`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

HEARTBEAT_SEAT_ROLLUP_CHANNEL = "orion:heartbeat:seat:rollup"
HEARTBEAT_SEAT_ROLLUP_KIND = "heartbeat.seat.rollup.v1"
HEARTBEAT_SEAT_ROLLUP_BUCKET_SEC = 300


def _require_tz(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("bucket timestamps must be timezone-aware")
    return value


class HeartbeatSeatRollupV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["heartbeat.seat.rollup.v1"] = "heartbeat.seat.rollup.v1"
    # Deterministic: f"{node}:{seat}:{bucket_start.isoformat()}" -- a re-published bucket hits the
    # sql-writer's insert-only duplicate-key path instead of making a second row.
    rollup_id: str
    node: str
    # Seat name from services/orion-heartbeat/app/substrate/routing.py SEAT_ORDER, and its site index.
    seat: str
    site_index: int = Field(ge=0)
    bucket_start: datetime
    bucket_end: datetime
    # Sum of route_atom(...).level over the seat's absorbed atoms in the bucket (math.fsum, so the
    # value does not depend on arrival order). 0.0 with atom_count 0 is a real quiet seat: the
    # heartbeat was listening for the whole bucket and nothing arrived.
    level_sum: float = Field(ge=0.0)
    # Atoms that route_atom placed on this seat (measured + presence).
    atom_count: int = Field(ge=0)
    # Atoms from this seat's services that the input rule skipped (clockwork, duplicate, floor, ...).
    skipped_count: int = Field(default=0, ge=0)
    # True when the heartbeat was not listening for part of the bucket (process start, or a
    # silence on orion:grammar:event longer than its gap threshold). The sums are an undercount,
    # not a quiet seat: readers treat the bucket as missing.
    partial: bool = False

    @field_validator("bucket_start", "bucket_end")
    @classmethod
    def _bucket_times_are_tz_aware(cls, value: datetime) -> datetime:
        return _require_tz(value)


# Column list of the `heartbeat_seat_rollup_5m` table, shared by the writer
# (services/orion-sql-writer/app/models/heartbeat_seat_rollup.py) and its shape test.
# Excludes `schema_version`: a fixed literal, never a column.
HEARTBEAT_SEAT_ROLLUP_SQL_COLUMNS: tuple[str, ...] = (
    "rollup_id",
    "node",
    "seat",
    "site_index",
    "bucket_start",
    "bucket_end",
    "level_sum",
    "atom_count",
    "skipped_count",
    "partial",
)
