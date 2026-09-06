"""Compile-time shape checks for the substrate_durable_run_state write path
(no Postgres required) -- durable cognition runs, 2026-09-06. Same four
guarantees as test_attention_schema_sql_shape.py: subscribed, routed,
every schema field lands on a column, bounded retention."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.durable_run import DURABLE_RUN_STATE_CHANNEL, DurableRunStateV1  # noqa: E402

from app.grammar_retention_loop import retention_days_for  # noqa: E402
from app.grammar_truth import GRAMMAR_RETENTION_TABLES  # noqa: E402
from app.models.durable_run_state import DurableRunStateSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402


def test_the_channel_is_actually_subscribed() -> None:
    raw = next(
        line.split("=", 1)[1].strip()
        for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert DURABLE_RUN_STATE_CHANNEL in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert DURABLE_RUN_STATE_CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP["durable.run.state.v1"] == "DurableRunStateSQL"
    assert MODEL_MAP["DurableRunStateSQL"] == (DurableRunStateSQL, DurableRunStateV1)


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(DurableRunStateSQL).columns}
    for name in DurableRunStateV1.model_fields:
        if name == "schema_version":
            continue
        assert name in columns, name
    assert DurableRunStateSQL.__tablename__ == "substrate_durable_run_state"


def test_the_table_has_bounded_retention_by_default() -> None:
    assert "substrate_durable_run_state" in dict(GRAMMAR_RETENTION_TABLES)
    assert retention_days_for(Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]))["substrate_durable_run_state"] == 90
