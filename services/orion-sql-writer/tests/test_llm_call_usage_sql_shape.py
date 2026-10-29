"""Write path for the per-call token ledger (no Postgres): ``llm.call.usage.v1`` on
``orion:llm:call:usage`` -> ``llm_call_usage``.

Subscribed even under a stale operator channel list, routed, the migration matches the model, has
the correlation/time index the physiology report joins on, and fits the post-pull auto-apply
allowlist. The full ``_write`` path lands one row per call; redelivery is a no-op.
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

from orion.schemas.llm_call_usage import (  # noqa: E402
    LLM_CALL_USAGE_CHANNEL,
    LLM_CALL_USAGE_KIND,
    LlmCallUsageV1,
)

import app.worker as worker  # noqa: E402
from app.models.llm_call_usage import LlmCallUsageSQL  # noqa: E402

T0 = datetime(2026, 10, 29, 18, 0, tzinfo=timezone.utc)
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_llm_call_usage_v1.sql"


def _event(**kw) -> LlmCallUsageV1:
    base = dict(call_id="abc123", recorded_at=T0, recorder="gateway", path="http_anthropic",
                correlation_id="turn-1", route="agent", model="Qwen3.6-35B", served_by="circe-worker-chat",
                pool_role="chat", holder="http:anthropic", node="athena", outcome="served", stream=True,
                prompt_tokens=14, cached_prompt_tokens=10, completion_tokens=8, token_source="usage",
                duration_ms=230, wait_ms=4, decode_tps=73.2)
    base.update(kw)
    return LlmCallUsageV1(**base)


def test_subscribed_and_routed() -> None:
    from app.settings import DEFAULT_ROUTE_MAP, Settings

    env = dict(line.split("=", 1) for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line and not line.startswith("#") and "=" in line)
    assert LLM_CALL_USAGE_CHANNEL in Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]).effective_subscribe_channels
    assert LLM_CALL_USAGE_CHANNEL in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP[LLM_CALL_USAGE_KIND] == "LlmCallUsageSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])[LLM_CALL_USAGE_KIND] == "LlmCallUsageSQL"
    assert Settings(SQL_WRITER_ROUTE_MAP_JSON="{}").route_map[LLM_CALL_USAGE_KIND] == "LlmCallUsageSQL"
    assert worker.MODEL_MAP["LlmCallUsageSQL"] == (LlmCallUsageSQL, LlmCallUsageV1)
    assert LlmCallUsageSQL in worker.INSERT_ONLY_MODELS


def test_migration_matches_model_has_join_index_and_fits_allowlist() -> None:
    text = MIGRATION.read_text()
    block = re.search(r"CREATE TABLE IF NOT EXISTS llm_call_usage \((.*?)\n\);", text, re.S).group(1)
    migration_cols = {line.split()[0] for line in block.strip().splitlines() if line.strip()}
    assert {c.name for c in inspect(LlmCallUsageSQL).columns} == migration_cols
    # every schema field has a column (nothing silently dropped by the writer's column filter)
    assert set(LlmCallUsageV1.model_fields) - {"schema_version"} <= migration_cols
    assert "ON llm_call_usage (correlation_id, recorded_at)" in text
    assert "ON llm_call_usage (recorded_at)" in text
    assert text.startswith("-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer")
    import orion.sql_migration_drift as drift

    assert drift.auto_apply_refusals(text, set()) == [], "migration must fit the post-pull additive allowlist"
    rb = drift.parse_required_by(text)
    assert rb is not None and not rb.errors


def _session(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    LlmCallUsageSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None)
    return session


def test_write_path_one_row_per_call_and_redelivery_is_a_no_op(monkeypatch) -> None:
    session = _session(monkeypatch)
    payload = json.loads(_event().model_dump_json())
    assert asyncio.run(worker._write(LlmCallUsageSQL, LlmCallUsageV1, payload, kind=LLM_CALL_USAGE_KIND))
    assert not asyncio.run(worker._write(LlmCallUsageSQL, LlmCallUsageV1, payload, kind=LLM_CALL_USAGE_KIND))
    row = session.query(LlmCallUsageSQL).one()
    assert (row.correlation_id, row.prompt_tokens, row.cached_prompt_tokens, row.completion_tokens) == (
        "turn-1", 14, 10, 8)
    assert (row.recorder, row.path, row.token_source, row.stream, row.duration_ms) == (
        "gateway", "http_anthropic", "usage", True, 230)


def test_a_call_with_no_tokens_still_lands(monkeypatch) -> None:
    session = _session(monkeypatch)
    payload = json.loads(_event(call_id="x2", prompt_tokens=None, cached_prompt_tokens=None,
                                completion_tokens=None, token_source="none", decode_tps=None,
                                outcome="upstream_http_5xx", correlation_id=None).model_dump_json())
    assert asyncio.run(worker._write(LlmCallUsageSQL, LlmCallUsageV1, payload, kind=LLM_CALL_USAGE_KIND))
    row = session.query(LlmCallUsageSQL).one()
    assert row.prompt_tokens is None and row.outcome == "upstream_http_5xx"
