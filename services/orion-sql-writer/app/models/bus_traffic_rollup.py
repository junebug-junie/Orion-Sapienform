import json

from sqlalchemy import Column, DateTime, Integer, JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

from app.db import Base

# Generic JSON on SQLite (tests), JSONB on Postgres (prod).
_JSONB = JSON().with_variant(JSONB(), "postgresql")


class _JSONSafe(TypeDecorator):
    """Round-trips through json so nothing non-JSON-native reaches the driver
    (same reason as causal_geometry_snapshot._JSONSafeList: `_write_row()` dumps
    the payload without mode="json")."""

    impl = _JSONB
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.loads(json.dumps(value, default=str))


class BusTrafficRollupSQL(Base):
    """One closed 5-minute bucket of bus traffic from orion-bus-mirror.

    Channel `orion:bus:traffic:rollup`, kind `bus.traffic.rollup.v1`
    (`orion.schemas.bus_traffic.BusTrafficRollupV1`). Column names match the
    pydantic field names, so the generic `_write_row()` path needs no
    special-casing. `rollup_id` = "<node>:<bucket_start iso>" is deterministic, so
    the model is insert-only: a re-published bucket hits the duplicate-key catch.
    Reader: the Patch 2 replay eval of
    docs/superpowers/specs/2026-11-11-causal-geometry-redesign.md.
    """

    __tablename__ = "bus_traffic_rollup_5m"

    rollup_id = Column(String, primary_key=True)
    node = Column(String, nullable=False)
    bucket_start = Column(DateTime(timezone=True), nullable=False, index=True)
    bucket_end = Column(DateTime(timezone=True), nullable=False)
    message_count = Column(Integer, nullable=False)
    channel_counts = Column(_JSONSafe, nullable=False)
    hops = Column(_JSONSafe, nullable=False)
    decode_failures = Column(Integer, nullable=False)
