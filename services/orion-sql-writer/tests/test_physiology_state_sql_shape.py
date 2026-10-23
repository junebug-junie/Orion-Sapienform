"""Write path for Orion's Body organ (body economy Patch A3, no Postgres):
``physiology.state.v1`` on ``orion:physiology:state`` -> ``physiology_state``.

Subscribed even under a stale operator channel list (the past silent-no-op incident), routed,
catalogued, the migration matches the model, and the full ``_write`` path (validate -> row ->
insert) lands one row per tick with no_reading readings kept as missing, not 0. Redelivery is a
no-op."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.physiology import (  # noqa: E402
    PHYSIOLOGY_STATE_CHANNEL,
    PHYSIOLOGY_STATE_KIND,
    PhysiologyConstraintReadingV1,
    PhysiologyStateV1,
)
from orion.schemas.registry import resolve  # noqa: E402

import app.worker as worker  # noqa: E402
from app.models.physiology_state import PhysiologyStateSQL, physiology_state_row  # noqa: E402

T0 = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_physiology_state_v1.sql"


def _state(**kw) -> PhysiologyStateV1:
    return PhysiologyStateV1(
        tick_id=f"physiology:{T0.isoformat()}", generated_at=T0,
        constraints=[
            PhysiologyConstraintReadingV1(constraint="heat", unit="c", level=29.5, budget=31.0,
                                          budget_kind="soft_line", source_at=T0),
            PhysiologyConstraintReadingV1(constraint="claude", unit="limit_events", level=None,
                                          state="unknown", no_reading="no_observation"),
        ],
        warnings=["lane_capacity:pool_state_unread"], **kw)


def test_subscribed_routed_and_catalogued() -> None:
    from app.settings import DEFAULT_ROUTE_MAP, Settings

    env = dict(line.split("=", 1) for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line and not line.startswith("#") and "=" in line)
    # A stale operator channel list still receives it (SQL_WRITER_SUBSCRIBE_CHANNELS replaces).
    assert PHYSIOLOGY_STATE_CHANNEL in Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]).effective_subscribe_channels
    assert PHYSIOLOGY_STATE_CHANNEL in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP[PHYSIOLOGY_STATE_KIND] == "PhysiologyStateSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])[PHYSIOLOGY_STATE_KIND] == "PhysiologyStateSQL"
    assert worker.MODEL_MAP["PhysiologyStateSQL"] == (PhysiologyStateSQL, PhysiologyStateV1)
    assert PhysiologyStateSQL in worker.INSERT_ONLY_MODELS
    assert resolve("PhysiologyStateV1") is PhysiologyStateV1
    import yaml

    catalog = yaml.safe_load((REPO_ROOT / "orion/bus/channels.yaml").read_text())
    entries = catalog if isinstance(catalog, list) else catalog["channels"]
    ch = next(c for c in entries if c["name"] == PHYSIOLOGY_STATE_CHANNEL)
    assert ch["schema_id"] == "PhysiologyStateV1" and ch["message_kind"] == PHYSIOLOGY_STATE_KIND
    assert ch["producer_services"] == ["orion-durable-runs"] and "orion-sql-writer" in ch["consumer_services"]


def test_migration_columns_match_the_model() -> None:
    text = MIGRATION.read_text()
    block = re.search(r"CREATE TABLE IF NOT EXISTS physiology_state \((.*?)\n\);", text, re.S).group(1)
    migration_cols = {line.split()[0] for line in block.strip().splitlines() if line.strip()}
    model_cols = {c.name for c in inspect(PhysiologyStateSQL).columns}
    assert model_cols == migration_cols
    assert text.startswith("-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer")


def test_row_is_json_safe() -> None:
    row = physiology_state_row(_state())
    json.dumps({k: v for k, v in row.items() if k != "generated_at"})
    assert row["constraints"][0]["source_at"] == "2026-10-23T18:00:00Z"


def _session(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    PhysiologyStateSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    return session


def test_write_path_persists_one_row_per_tick_and_redelivery_is_a_no_op(monkeypatch) -> None:
    session = _session(monkeypatch)
    payload = _state().model_dump(mode="json")
    assert asyncio.run(worker._write(PhysiologyStateSQL, PhysiologyStateV1, payload, kind=PHYSIOLOGY_STATE_KIND))
    assert not asyncio.run(worker._write(PhysiologyStateSQL, PhysiologyStateV1, payload, kind=PHYSIOLOGY_STATE_KIND))
    rows = session.query(PhysiologyStateSQL).all()
    assert len(rows) == 1
    by = {c["constraint"]: c for c in rows[0].constraints}
    assert by["heat"]["level"] == 29.5
    # A missing reading is stored as missing, never 0.
    assert by["claude"]["level"] is None and by["claude"]["no_reading"] == "no_observation"
    assert rows[0].warnings == ["lane_capacity:pool_state_unread"]
