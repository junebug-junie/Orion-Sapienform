"""Shape checks for the heartbeat-seat-rollup SQL write path (no Postgres required, issue #3886).

Mirrors test_bus_traffic_rollup_sql_shape.py: `HeartbeatSeatRollupSQL` is in `MODEL_MAP`, routed
from kind `heartbeat.seat.rollup.v1` in both `DEFAULT_ROUTE_MAP` and the operator `.env_example`
(the env alias REPLACES the default in live deployments), the channel is subscribed in both
places, columns match the pydantic field names, the model is insert-only, and the migration
declares the same columns with a REQUIRED-BY header and additive DDL only.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.heartbeat_seat_rollup import (  # noqa: E402
    HEARTBEAT_SEAT_ROLLUP_CHANNEL,
    HEARTBEAT_SEAT_ROLLUP_KIND,
    HEARTBEAT_SEAT_ROLLUP_SQL_COLUMNS,
    HeartbeatSeatRollupV1,
)

from app.models.heartbeat_seat_rollup import HeartbeatSeatRollupSQL  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v1.sql"
START = datetime(2026, 11, 25, 2, 15, tzinfo=timezone.utc)
END = datetime(2026, 11, 25, 2, 20, tzinfo=timezone.utc)


def _rollup(**overrides) -> HeartbeatSeatRollupV1:
    defaults = dict(
        rollup_id=f"athena:orion-bus:{START.isoformat()}",
        node="athena",
        seat="orion-bus",
        site_index=3,
        bucket_start=START,
        bucket_end=END,
        level_sum=1.234567890123,
        atom_count=27,
        skipped_count=135,
    )
    defaults.update(overrides)
    return HeartbeatSeatRollupV1(**defaults)


def _env_line(prefix: str) -> str:
    for line in (SERVICE_ROOT / ".env_example").read_text().splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"{prefix} not found in .env_example")


def test_default_route_map() -> None:
    assert DEFAULT_ROUTE_MAP.get(HEARTBEAT_SEAT_ROLLUP_KIND) == "HeartbeatSeatRollupSQL"


def test_model_map() -> None:
    assert MODEL_MAP["HeartbeatSeatRollupSQL"] == (HeartbeatSeatRollupSQL, HeartbeatSeatRollupV1)


def test_channel_in_settings_default_subscribe_list() -> None:
    assert HEARTBEAT_SEAT_ROLLUP_CHANNEL in Settings.model_fields["sql_writer_subscribe_channels"].default


def test_channel_in_env_example_subscribe_channels() -> None:
    assert f'"{HEARTBEAT_SEAT_ROLLUP_CHANNEL}"' in _env_line("SQL_WRITER_SUBSCRIBE_CHANNELS=")


def test_env_example_route_map_json() -> None:
    assert '"heartbeat.seat.rollup.v1":"HeartbeatSeatRollupSQL"' in _env_line("SQL_WRITER_ROUTE_MAP_JSON=")


def test_columns_match_pydantic_field_names() -> None:
    assert {a.key for a in inspect(HeartbeatSeatRollupSQL).attrs} == set(HEARTBEAT_SEAT_ROLLUP_SQL_COLUMNS)
    assert set(HEARTBEAT_SEAT_ROLLUP_SQL_COLUMNS) == set(HeartbeatSeatRollupV1.model_fields) - {"schema_version"}


def test_migration_declares_the_same_columns_header_and_additive_ddl_only() -> None:
    text = MIGRATION.read_text()
    assert text.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer"
    body = re.search(r"CREATE TABLE IF NOT EXISTS heartbeat_seat_rollup_5m \((.*?)\);", text, re.S).group(1)
    cols = {ln.split()[0] for ln in body.strip().splitlines() if ln.strip()}
    assert cols == set(HEARTBEAT_SEAT_ROLLUP_SQL_COLUMNS)
    statements = [s.strip() for s in re.sub(r"--[^\n]*", "", text).split(";") if s.strip()]
    assert all(s.upper().startswith(("CREATE TABLE IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS")) for s in statements)
    assert "DO $$" not in text and "GRANT" not in text.upper()


def test_real_write_row_path_from_a_bus_json_payload(monkeypatch) -> None:
    """Through the worker's own `_write_row` (field filter, schema_version drop, ISO string ->
    tz-aware datetime, INSERT_ONLY duplicate catch), fed the JSON payload the heartbeat puts on
    the bus. Re-delivery keeps one row; the float survives exactly."""
    import app.worker as worker

    assert HeartbeatSeatRollupSQL in INSERT_ONLY_MODELS
    engine = create_engine("sqlite://")
    HeartbeatSeatRollupSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(worker, "get_session", lambda: Session())
    monkeypatch.setattr(worker, "remove_session", lambda: None, raising=False)

    wire = _rollup(partial=True).model_dump(mode="json")
    worker._write_row(HeartbeatSeatRollupSQL, HeartbeatSeatRollupV1.model_validate(wire).model_dump())
    worker._write_row(
        HeartbeatSeatRollupSQL, HeartbeatSeatRollupV1.model_validate({**wire, "atom_count": 1}).model_dump()
    )

    sess = Session()
    try:
        [row] = sess.query(HeartbeatSeatRollupSQL).all()
        assert row.atom_count == 27  # first write wins
        assert row.level_sum == 1.234567890123
        assert row.partial is True
        assert row.seat == "orion-bus" and row.site_index == 3
        assert row.bucket_start.replace(tzinfo=timezone.utc) == START
    finally:
        sess.close()
