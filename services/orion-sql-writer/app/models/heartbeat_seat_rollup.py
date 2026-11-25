from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from app.db import Base


class HeartbeatSeatRollupSQL(Base):
    """One heartbeat seat in one closed 5-minute bucket, from orion-heartbeat (issue #3886).

    Channel `orion:heartbeat:seat:rollup`, kind `heartbeat.seat.rollup.v1`
    (`orion.schemas.heartbeat_seat_rollup.HeartbeatSeatRollupV1`). Column names match the
    pydantic field names, so the generic `_write_row()` path needs no special-casing.
    `rollup_id` = "<node>:<seat>:<bucket_start iso>" is deterministic, so the model is
    insert-only: a re-published row hits the duplicate-key catch.
    Reader: scripts/demon_network_replay.py (rollup first, grammar_events as fallback).
    """

    __tablename__ = "heartbeat_seat_rollup_5m"

    rollup_id = Column(String, primary_key=True)
    node = Column(String, nullable=False)
    seat = Column(String, nullable=False)
    site_index = Column(Integer, nullable=False)
    bucket_start = Column(DateTime(timezone=True), nullable=False, index=True)
    bucket_end = Column(DateTime(timezone=True), nullable=False)
    level_sum = Column(Float, nullable=False)
    atom_count = Column(Integer, nullable=False)
    skipped_count = Column(Integer, nullable=False)
    # Of atom_count: atoms that arrived after bucket_end, inside the flush grace (#3901).
    # NULL for rows from a heartbeat that did not measure it.
    late_count = Column(Integer, nullable=True)
    # Heartbeat was not listening for part of the bucket: an undercount, read as missing.
    partial = Column(Boolean, nullable=False, default=False)
    # Hash of the heartbeat routing rules that produced level_sum (routing.routing_version()).
    routing_version = Column(String, nullable=False)
