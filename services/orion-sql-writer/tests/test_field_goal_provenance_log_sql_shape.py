"""Goal proposal log write path (no Postgres, issue #3924).

``memory.field_goals.proposed.v1`` on ``orion:memory:goals:proposed`` -> ``field_goal_provenance_log``.
Subscribed and routed even under a stale operator env, migration matches the model and fits the
post-pull auto-apply allowlist, one row per goal through the real envelope handler, a re-delivered
goal does not double-insert, and a malformed payload is rejected, logged, and sent to the fallback
log without raising.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.field_goal import FieldGoalProvenanceV1  # noqa: E402

import app.worker as worker  # noqa: E402
from app.models.field_goal_provenance_log import FieldGoalProvenanceLogSQL  # noqa: E402

CHANNEL = "orion:memory:goals:proposed"
KIND = "memory.field_goals.proposed.v1"
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_field_goal_provenance_log_v1.sql"
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _goal(**kw) -> FieldGoalProvenanceV1:
    # Same construction as services/orion-attention-runtime/app/worker.py.
    base = dict(
        subject="attention",
        model_layer="field_attention",
        entity_id="world:reading",
        kind=KIND,
        field_target_id="world:reading",
        target_kind="node",
        salience_score=0.734,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=0.734,
        ts=T0,
        provenance={"intake_channel": "internal.attention_runtime"},
    )
    base.update(kw)
    return FieldGoalProvenanceV1(**base)


def _envelope(payload: dict) -> BaseEnvelope:
    return BaseEnvelope(
        kind=KIND,
        source=ServiceRef(name="orion-attention-runtime", node="athena"),
        correlation_id=uuid4(),
        payload=payload,
    )


def _session(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    FieldGoalProvenanceLogSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None, raising=False)
    return session


def _env_example() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    )


def test_subscribed_and_routed() -> None:
    from app.settings import DEFAULT_ROUTE_MAP, Settings

    env = _env_example()
    assert CHANNEL in Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]).effective_subscribe_channels
    assert CHANNEL in Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP[KIND] == "FieldGoalProvenanceLogSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])[KIND] == "FieldGoalProvenanceLogSQL"
    assert worker.MODEL_MAP["FieldGoalProvenanceLogSQL"] == (FieldGoalProvenanceLogSQL, FieldGoalProvenanceV1)
    assert FieldGoalProvenanceLogSQL in worker.INSERT_ONLY_MODELS


def test_registry_names_sql_writer_as_consumer() -> None:
    import yaml

    channels = yaml.safe_load((REPO_ROOT / "orion/bus/channels.yaml").read_text())["channels"]
    entry = next(c for c in channels if c["name"] == CHANNEL)
    assert entry["message_kind"] == KIND
    assert "orion-sql-writer" in entry["consumer_services"]


def test_migration_matches_model_and_fits_allowlist() -> None:
    import orion.sql_migration_drift as drift

    text = MIGRATION.read_text()
    assert text.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer"
    block = re.search(r"CREATE TABLE IF NOT EXISTS field_goal_provenance_log \((.*?)\n\);", text, re.S).group(1)
    migration_cols = {ln.split()[0] for ln in block.strip().splitlines() if ln.strip()}
    assert {c.name for c in inspect(FieldGoalProvenanceLogSQL).columns} == migration_cols
    # Index names match the model, so create_all at boot and this file make the same indexes.
    for ix in FieldGoalProvenanceLogSQL.__table__.indexes:
        cols = ", ".join(c.name for c in ix.columns)
        assert f"CREATE INDEX IF NOT EXISTS {ix.name} ON field_goal_provenance_log ({cols});" in text
    assert drift.auto_apply_refusals(text, set()) == []
    rb = drift.parse_required_by(text)
    assert rb is not None and not rb.errors


def test_one_row_per_goal_with_columns_and_full_payload(monkeypatch) -> None:
    session = _session(monkeypatch)
    g1, g2 = _goal(), _goal(field_target_id="capability:memory", entity_id="capability:memory",
                            target_kind="capability", salience_score=0.5, priority=0.5)
    e1 = _envelope(g1.model_dump(mode="json"))
    asyncio.run(worker.handle_envelope(e1))
    asyncio.run(worker.handle_envelope(_envelope(g2.model_dump(mode="json"))))

    rows = {r.artifact_id: r for r in session.query(FieldGoalProvenanceLogSQL).all()}
    assert set(rows) == {g1.artifact_id, g2.artifact_id}
    r = rows[g1.artifact_id]
    assert (r.field_target_id, r.target_kind, r.salience_score, r.priority) == ("world:reading", "node", 0.734, 0.734)
    assert (r.source_field_tick_id, r.source_attention_frame_id, r.proposal_status) == ("tick-1", "frame-1", "proposed")
    assert r.ts.replace(tzinfo=timezone.utc) == T0
    assert r.received_at is not None
    assert r.envelope_id == str(e1.id)
    # The goal had no correlation_id; the envelope's fresh uuid must not be written over it.
    assert r.correlation_id is None
    # Full payload, schema fields only (no envelope extras), JSON-native.
    assert r.payload_json == g1.model_dump(mode="json")
    assert rows[g2.artifact_id].target_kind == "capability"


def test_redelivered_goal_does_not_double_insert(monkeypatch) -> None:
    session = _session(monkeypatch)
    payload = _goal().model_dump(mode="json")
    asyncio.run(worker.handle_envelope(_envelope(payload)))
    asyncio.run(worker.handle_envelope(_envelope(payload)))  # same artifact_id, new envelope
    assert session.query(FieldGoalProvenanceLogSQL).count() == 1


def test_malformed_payload_is_rejected_logged_and_does_not_raise(monkeypatch, caplog) -> None:
    session = _session(monkeypatch)
    fallbacks: list[tuple] = []
    monkeypatch.setattr(worker, "_write_fallback", lambda *a, **k: fallbacks.append(a))
    bad = _goal().model_dump(mode="json")
    bad["salience_score"] = 7.0  # out of [0, 1]
    del bad["field_target_id"]
    with caplog.at_level(logging.ERROR):
        asyncio.run(worker.handle_envelope(_envelope(bad)))  # must not raise
    assert session.query(FieldGoalProvenanceLogSQL).count() == 0
    assert any("sql_writer_validation_failed" in rec.getMessage() for rec in caplog.records)
    assert len(fallbacks) == 1 and fallbacks[0][0] == KIND
