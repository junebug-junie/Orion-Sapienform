"""Shape checks for the goal-provenance streak-tick SQL write path (no Postgres required).

Mirrors test_action_outcome_sql_shape.py's pattern: asserts the real wiring for
`orion:debug:attention:streak_tick` -> `DominanceStreakTickSQL` is registered in `MODEL_MAP`
under the `DominanceStreakTickSQL` route key, keyed off kind `debug.attention.streak_tick.v1`,
and every field on the producer schema `DominanceStreakTickV1` maps onto a real column.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.field_goal import DominanceStreakTickV1  # noqa: E402

from app.models.dominance_streak_tick import DominanceStreakTickSQL  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP  # noqa: E402


def _make_tick(**overrides) -> DominanceStreakTickV1:
    defaults = dict(
        tick_telemetry_id="streak-tick-abc-123",
        target_id="node:substrate.biometrics",
        streak_count=2,
        min_streak_at_tick=3,
        qualified=False,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        observed_at=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return DominanceStreakTickV1(**defaults)


def test_default_route_map_points_streak_tick_at_dominance_streak_tick_sql() -> None:
    assert DEFAULT_ROUTE_MAP.get("debug.attention.streak_tick.v1") == "DominanceStreakTickSQL"


def test_model_map_registers_dominance_streak_tick_sql_with_tick_schema() -> None:
    assert MODEL_MAP["DominanceStreakTickSQL"] == (DominanceStreakTickSQL, DominanceStreakTickV1)


def test_channel_is_subscribed() -> None:
    from app.settings import settings

    assert "orion:debug:attention:streak_tick" in settings.effective_subscribe_channels


def test_tick_fields_map_onto_real_columns() -> None:
    mapper = inspect(DominanceStreakTickSQL)
    valid_keys = {attr.key for attr in mapper.attrs}
    data = _make_tick().model_dump()
    missing = [field for field in data if field not in valid_keys]
    assert not missing, f"DominanceStreakTickV1 fields missing from DominanceStreakTickSQL columns: {missing}"


def test_tick_data_constructs_dominance_streak_tick_sql_without_raising() -> None:
    tick = _make_tick()
    row = DominanceStreakTickSQL(**tick.model_dump())
    assert row.tick_telemetry_id == tick.tick_telemetry_id
    assert row.target_id == tick.target_id
    assert row.streak_count == 2
    assert row.qualified is False


def test_tick_data_handles_null_target_id() -> None:
    """No node:substrate.* winner this tick -- target_id=None is a real, valid state
    (update_dominance_streak's own reset case), not an error."""
    tick = _make_tick(target_id=None, streak_count=0)
    row = DominanceStreakTickSQL(**tick.model_dump())
    assert row.target_id is None
    assert row.streak_count == 0


def test_merge_redelivery_upserts_one_row_and_preserves_created_at() -> None:
    """Re-delivery of the same tick_telemetry_id must upsert (one row), not duplicate --
    mirrors action_outcomes' merge-idempotency test against in-memory SQLite."""
    engine = create_engine("sqlite://")
    DominanceStreakTickSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _merge(tick: DominanceStreakTickV1) -> None:
        sess = Session()
        try:
            sess.merge(DominanceStreakTickSQL(**tick.model_dump()))
            sess.commit()
        finally:
            sess.close()

    _merge(_make_tick(streak_count=2))

    sess = Session()
    try:
        first = sess.get(DominanceStreakTickSQL, "streak-tick-abc-123")
        original_created_at = first.created_at
    finally:
        sess.close()
    assert original_created_at is not None

    _merge(_make_tick(streak_count=2, qualified=True))

    sess = Session()
    try:
        rows = sess.query(DominanceStreakTickSQL).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.qualified is True
        assert row.created_at == original_created_at
    finally:
        sess.close()
