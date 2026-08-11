from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.db import Base


class DominanceStreakTickSQL(Base):
    """Durable, per-tick record of orion-attention-runtime's node-target DominanceStreak.

    Produced via `debug.attention.streak_tick.v1` on `orion:debug:attention:streak_tick`,
    one row per real field tick the goal-provenance producer runs -- not just the qualifying
    ticks that reach `orion:memory:goals:proposed`/`action_outcomes`-shaped durability. Exists
    to answer whether `ORION_GOAL_PROVENANCE_MIN_STREAK`'s placeholder value of 3 is right
    (see `scripts/analysis/measure_goal_provenance_streak_distribution.py`); the qualifying-
    only channel is a censored sample and cannot answer that alone. `tick_telemetry_id` is the
    primary key so re-delivered events upsert idempotently, matching `ActionOutcomeSQL`'s
    pattern.

    Meant to be temporary, high-volume debug telemetry (~1 row per real field tick, matching
    `substrate_attention_frames`' cadence) -- bounded by
    `settings.goal_provenance_streak_ticks_retention_days` (default 14, applied at boot in
    `main.py`, matching `drive_audits_retention_days`' existing pattern).
    """

    __tablename__ = "goal_provenance_streak_ticks"

    tick_telemetry_id = Column(String, primary_key=True)
    target_id = Column(String, nullable=True)
    streak_count = Column(Integer, nullable=False)
    min_streak_at_tick = Column(Integer, nullable=False)
    qualified = Column(Boolean, nullable=False)
    source_field_tick_id = Column(String, nullable=False)
    source_attention_frame_id = Column(String, nullable=False)
    observed_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
