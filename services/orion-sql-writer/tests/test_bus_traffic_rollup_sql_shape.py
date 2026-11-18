"""Shape checks for the bus-traffic-rollup SQL write path (no Postgres required).

Mirrors test_causal_geometry_snapshot_sql_shape.py: `BusTrafficRollupSQL` is in
`MODEL_MAP`, routed from kind `bus.traffic.rollup.v1` in both `DEFAULT_ROUTE_MAP` and
the operator `.env_example` (the env alias REPLACES the default in live deployments),
the channel is subscribed in both places, columns match the pydantic field names, the
model is insert-only, and the migration declares the same columns.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.bus_traffic import (  # noqa: E402
    BUS_TRAFFIC_ROLLUP_CHANNEL,
    BUS_TRAFFIC_ROLLUP_KIND,
    BUS_TRAFFIC_ROLLUP_SQL_COLUMNS,
    BusTrafficHopV1,
    BusTrafficRollupV1,
)

from app.models.bus_traffic_rollup import BusTrafficRollupSQL  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_bus_traffic_rollup_5m_v1.sql"
START = datetime(2026, 11, 13, 2, 15, tzinfo=timezone.utc)
END = datetime(2026, 11, 13, 2, 20, tzinfo=timezone.utc)


def _rollup(**overrides) -> BusTrafficRollupV1:
    defaults = dict(
        rollup_id=f"athena:{START.isoformat()}",
        node="athena",
        bucket_start=START,
        bucket_end=END,
        message_count=42,
        channel_counts={"orion:exec:result:*": 40, "orion:bus:traffic:rollup": 2},
        hops=[BusTrafficHopV1(from_organ="cortex-exec", to_organ="llm-gateway", count=3, latency_mean_sec=1.5)],
        decode_failures=1,
    )
    defaults.update(overrides)
    return BusTrafficRollupV1(**defaults)


def _row_dict(event: BusTrafficRollupV1) -> dict:
    """Mirror `_write_row`'s generic filter."""
    data = event.model_dump()
    valid = {attr.key for attr in inspect(BusTrafficRollupSQL).attrs}
    return {k: v for k, v in data.items() if k in valid}


def _env_line(prefix: str) -> str:
    for line in (SERVICE_ROOT / ".env_example").read_text().splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"{prefix} not found in .env_example")


def test_default_route_map() -> None:
    assert DEFAULT_ROUTE_MAP.get(BUS_TRAFFIC_ROLLUP_KIND) == "BusTrafficRollupSQL"


def test_model_map() -> None:
    assert MODEL_MAP["BusTrafficRollupSQL"] == (BusTrafficRollupSQL, BusTrafficRollupV1)


def test_channel_in_settings_default_subscribe_list() -> None:
    assert BUS_TRAFFIC_ROLLUP_CHANNEL in Settings.model_fields["sql_writer_subscribe_channels"].default


def test_channel_in_env_example_subscribe_channels() -> None:
    assert f'"{BUS_TRAFFIC_ROLLUP_CHANNEL}"' in _env_line("SQL_WRITER_SUBSCRIBE_CHANNELS=")


def test_env_example_route_map_json() -> None:
    assert '"bus.traffic.rollup.v1":"BusTrafficRollupSQL"' in _env_line("SQL_WRITER_ROUTE_MAP_JSON=")


def test_columns_match_pydantic_field_names() -> None:
    assert {a.key for a in inspect(BusTrafficRollupSQL).attrs} == set(BUS_TRAFFIC_ROLLUP_SQL_COLUMNS)
    assert set(BUS_TRAFFIC_ROLLUP_SQL_COLUMNS) == set(BusTrafficRollupV1.model_fields) - {"schema_version"}


def test_migration_declares_the_same_columns_and_header() -> None:
    text = MIGRATION.read_text()
    assert text.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer"
    body = re.search(r"CREATE TABLE IF NOT EXISTS bus_traffic_rollup_5m \((.*?)\);", text, re.S).group(1)
    cols = {ln.split()[0] for ln in body.strip().splitlines() if ln.strip()}
    assert cols == set(BUS_TRAFFIC_ROLLUP_SQL_COLUMNS)
    assert "DO $$" not in text and "GRANT" not in text.upper()


def test_row_dict_passes_through() -> None:
    row = BusTrafficRollupSQL(**_row_dict(_rollup()))
    assert row.rollup_id == f"athena:{START.isoformat()}"
    assert row.message_count == 42
    assert row.channel_counts["orion:exec:result:*"] == 40
    assert row.hops[0]["from_organ"] == "cortex-exec"


def test_insert_only_membership() -> None:
    assert BusTrafficRollupSQL in INSERT_ONLY_MODELS


def test_insert_only_redelivery_keeps_one_row_and_jsonb_round_trips() -> None:
    engine = create_engine("sqlite://")
    BusTrafficRollupSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)

    def _insert(event: BusTrafficRollupV1) -> bool:
        sess = Session()
        try:
            sess.add(BusTrafficRollupSQL(**_row_dict(event)))
            sess.commit()
            return True
        except IntegrityError:
            sess.rollback()
            return False
        finally:
            sess.close()

    assert _insert(_rollup()) is True
    assert _insert(_rollup(message_count=1)) is False  # same rollup_id: first write wins

    sess = Session()
    try:
        [row] = sess.query(BusTrafficRollupSQL).all()
        assert row.message_count == 42
        assert row.hops == [{"from_organ": "cortex-exec", "to_organ": "llm-gateway", "count": 3, "latency_mean_sec": 1.5}]
    finally:
        sess.close()


def test_real_write_row_path_from_a_bus_json_payload(monkeypatch) -> None:
    """Through the worker's own `_write_row` (field filter, schema_version drop, ISO string
    -> tz-aware datetime, jsonb, INSERT_ONLY duplicate catch), fed the same JSON payload the
    mirror puts on the bus. Re-delivery keeps one row."""
    import app.worker as worker

    engine = create_engine("sqlite://")
    BusTrafficRollupSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(worker, "get_session", lambda: Session())
    monkeypatch.setattr(worker, "remove_session", lambda: None, raising=False)

    wire = _rollup(partial=True).model_dump(mode="json")  # what the bus envelope carries
    data = BusTrafficRollupV1.model_validate(wire).model_dump()
    worker._write_row(BusTrafficRollupSQL, data)
    worker._write_row(BusTrafficRollupSQL, BusTrafficRollupV1.model_validate({**wire, "message_count": 1}).model_dump())

    sess = Session()
    try:
        [row] = sess.query(BusTrafficRollupSQL).all()
        assert row.message_count == 42  # first write wins
        assert row.partial is True
        assert row.bucket_start.replace(tzinfo=timezone.utc) == START
        assert row.channel_counts == {"orion:exec:result:*": 40, "orion:bus:traffic:rollup": 2}
    finally:
        sess.close()
