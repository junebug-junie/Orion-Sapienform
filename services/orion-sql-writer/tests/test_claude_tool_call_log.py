"""Write path for the audit of Claude's tool calls (Patch 1b, no Postgres):
``claude.tool.call.v1`` on ``orion:claude:tool:audit`` -> ``claude_tool_call_log``.

Subscribed even under a stale operator channel list, routed, catalogued; the migration matches the model,
uses the same index names (so create_all at boot and the migration build the same thing) and fits the
post-pull auto-apply allowlist; the full ``_write`` path lands one row per phase and redelivery is a no-op.
"""

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
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(REPO_ROOT / "scripts")]

from orion.claude_tools.transport import TOOL_AUDIT_CHANNEL, TOOL_AUDIT_KIND  # noqa: E402
from orion.schemas.claude_tools import ClaudeToolCallV1  # noqa: E402
from orion.schemas.registry import resolve  # noqa: E402

import app.worker as worker  # noqa: E402
from app.models.claude_tool_call_log import ClaudeToolCallLogSQL  # noqa: E402

T0 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_claude_tool_call_log_v1.sql"


def _event(phase="result", **kw) -> ClaudeToolCallV1:
    base = dict(event_id=f"call-00000001:{phase}", phase=phase, call_id="call-00000001",
                turn_request_id="room-claude-req-1", room_id="hub-room", claude_session_id="sess-a",
                tool="body_economy", category="body", arguments={"section": "limits"}, reason="checking a claim",
                setting="ask", decision_path="asked", shared=True, orion_answer="yes", orion_scope="once",
                orion_reason="fine", consent_latency_ms=1800, result_chars=900, created_at=T0)
    base.update(kw)
    return ClaudeToolCallV1(**base)


def test_subscribed_routed_and_catalogued() -> None:
    from app.settings import DEFAULT_ROUTE_MAP, Settings

    env = dict(line.split("=", 1) for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line and not line.startswith("#") and "=" in line)
    assert TOOL_AUDIT_CHANNEL in Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]).effective_subscribe_channels
    assert TOOL_AUDIT_CHANNEL in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP[TOOL_AUDIT_KIND] == "ClaudeToolCallLogSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])[TOOL_AUDIT_KIND] == "ClaudeToolCallLogSQL"
    assert Settings(SQL_WRITER_ROUTE_MAP_JSON="{}").route_map[TOOL_AUDIT_KIND] == "ClaudeToolCallLogSQL"
    assert worker.MODEL_MAP["ClaudeToolCallLogSQL"] == (ClaudeToolCallLogSQL, ClaudeToolCallV1)
    assert ClaudeToolCallLogSQL in worker.INSERT_ONLY_MODELS
    assert resolve("ClaudeToolCallV1") is ClaudeToolCallV1
    import yaml

    catalog = yaml.safe_load((REPO_ROOT / "orion/bus/channels.yaml").read_text())
    entries = catalog if isinstance(catalog, list) else catalog["channels"]
    ch = next(c for c in entries if c["name"] == TOOL_AUDIT_CHANNEL)
    assert ch["schema_id"] == "ClaudeToolCallV1" and ch["message_kind"] == TOOL_AUDIT_KIND
    assert ch["producer_services"] == ["orion-harness-governor"] and "orion-sql-writer" in ch["consumer_services"]


def test_migration_matches_the_model_and_fits_the_auto_apply_allowlist() -> None:
    text = MIGRATION.read_text()
    block = re.search(r"CREATE TABLE IF NOT EXISTS claude_tool_call_log \((.*?)\n\);", text, re.S).group(1)
    migration_cols = {line.split()[0] for line in block.strip().splitlines() if line.strip()}
    assert {c.name for c in inspect(ClaudeToolCallLogSQL).columns} == migration_cols
    model_indexes = {ix.name for ix in ClaudeToolCallLogSQL.__table__.indexes}
    assert model_indexes == set(re.findall(r"CREATE INDEX IF NOT EXISTS (\w+)", text))
    assert text.startswith("-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub")
    import orion.sql_migration_drift as drift

    assert drift.auto_apply_refusals(text, set()) == [], "migration must fit the post-pull additive allowlist"
    rb = drift.parse_required_by(text)
    assert rb is not None and not rb.errors


def _session(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    ClaudeToolCallLogSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    return session


def test_write_path_one_row_per_phase_and_redelivery_is_a_no_op(monkeypatch) -> None:
    session = _session(monkeypatch)
    for phase in ("ask", "answer", "result"):
        payload = json.loads(_event(phase).model_dump_json())
        assert asyncio.run(worker._write(ClaudeToolCallLogSQL, ClaudeToolCallV1, payload, kind=TOOL_AUDIT_KIND))
    again = json.loads(_event("result").model_dump_json())
    assert not asyncio.run(worker._write(ClaudeToolCallLogSQL, ClaudeToolCallV1, again, kind=TOOL_AUDIT_KIND))
    rows = session.query(ClaudeToolCallLogSQL).order_by(ClaudeToolCallLogSQL.event_id).all()
    assert [r.phase for r in rows] == ["answer", "ask", "result"]
    r = rows[-1]
    assert (r.decision_path, r.orion_answer, r.orion_reason, r.shared, r.result_chars) == ("asked", "yes", "fine", True, 900)
    assert r.arguments == {"section": "limits"} and r.turn_request_id == "room-claude-req-1"
