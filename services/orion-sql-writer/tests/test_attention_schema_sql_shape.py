"""Compile-time shape checks for the substrate_attention_schema write path
(no Postgres required) -- attention schema surface, 2026-09-06.

Real wiring: `AttentionSchemaSQL` is registered in `MODEL_MAP` under its own
route key, keyed off kind `attention.schema.v1`, the channel is actually
subscribed (both the .env_example list and a code-default-only Settings),
every field on `AttentionSchemaV1` maps onto a real column, and the table has
a bounded retention entry like every other table in this service.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.attention_schema import ATTENTION_SCHEMA_CHANNEL, AttentionSchemaV1  # noqa: E402

from app.grammar_retention_loop import retention_days_for  # noqa: E402
from app.grammar_truth import GRAMMAR_RETENTION_TABLES  # noqa: E402
from app.models.attention_schema import AttentionSchemaSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402


def test_the_channel_is_actually_subscribed() -> None:
    """Registering a channel/route/model is not the same as subscribing to
    it -- this repo shipped that exact gap twice (self_study:items:write,
    chat_stance:belief:write). SQL_WRITER_SUBSCRIBE_CHANNELS replaces the
    Python default wholesale, so both the .env_example list and a stale
    operator list must carry the channel."""
    example = SERVICE_ROOT / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert ATTENTION_SCHEMA_CHANNEL in json.loads(raw)

    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert ATTENTION_SCHEMA_CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP["attention.schema.v1"] == "AttentionSchemaSQL"
    assert MODEL_MAP["AttentionSchemaSQL"] == (AttentionSchemaSQL, AttentionSchemaV1)
    assert AttentionSchemaSQL in INSERT_ONLY_MODELS


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(AttentionSchemaSQL).columns}
    for name in AttentionSchemaV1.model_fields:
        if name == "schema_version":
            continue
        assert name in columns, name
    assert AttentionSchemaSQL.__tablename__ == "substrate_attention_schema"
    index_columns = {tuple(c.name for c in ix.columns) for ix in AttentionSchemaSQL.__table__.indexes}
    assert ("process", "created_at") in index_columns


def test_a_real_row_dumps_onto_the_columns_without_loss() -> None:
    row = AttentionSchemaV1(
        entry_id="substrate-abc", generated_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        process="substrate_attention", attended_id="open-loop-1", attended_label="x",
        attention_reason="bottom_up_salience:no_open_loops", reason_narrative="n",
        confidence=0.4, confidence_basis="b", predicted_next="stable",
    )
    data = row.model_dump()
    columns = {c.key for c in inspect(AttentionSchemaSQL).columns}
    dropped = {k for k in data if k not in columns}
    assert dropped == {"schema_version"}


def test_the_table_has_bounded_retention_by_default() -> None:
    assert "substrate_attention_schema" in dict(GRAMMAR_RETENTION_TABLES)
    days = retention_days_for(Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]))
    assert days["substrate_attention_schema"] == 90
    example = (SERVICE_ROOT / ".env_example").read_text()
    assert "SUBSTRATE_ATTENTION_SCHEMA_RETENTION_DAYS=90" in example
