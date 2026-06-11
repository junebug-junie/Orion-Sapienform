"""Regression tests for sql-writer bus consumer stall (grammar INSERT hang blocking all tables)."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_resilience_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

DB_PATH = SQL_WRITER_ROOT / "app" / "db.py"
DB_SPEC = importlib.util.spec_from_file_location("sql_writer_db_resilience_tests", DB_PATH)
assert DB_SPEC and DB_SPEC.loader
db_module = importlib.util.module_from_spec(DB_SPEC)
DB_SPEC.loader.exec_module(db_module)


def _grammar_payload() -> dict:
    now = datetime.now(timezone.utc)
    event = GrammarEventV1(
        event_id="gev_test_resilience",
        event_kind="trace_started",
        trace_id="trace-resilience",
        emitted_at=now,
        provenance=GrammarProvenanceV1(source_service="test"),
    )
    return event.model_dump(mode="json")


def _metacog_payload() -> dict:
    return {
        "tick_id": "tick-resilience",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_service": "test-equilibrium",
        "distress_score": 0.0,
        "zen_score": 1.0,
        "services_tracked": 0,
        "snapshot": {},
    }


@pytest.mark.asyncio
async def test_grammar_persist_timeout_records_fallback_and_allows_next_envelope(monkeypatch) -> None:
    def _slow_persist(_event) -> bool:
        time.sleep(2.0)
        return True

    monkeypatch.setattr(
        "app.grammar_ledger_handler.persist_grammar_event",
        _slow_persist,
    )
    monkeypatch.setattr(worker.settings, "sql_writer_grammar_persist_timeout_sec", 0.05)

    fallback_calls: list[tuple[str, str | None]] = []

    def _fake_fallback(kind: str, correlation_id: str, payload, error: str | None = None) -> None:
        fallback_calls.append((kind, error))

    monkeypatch.setattr(worker, "_write_fallback", _fake_fallback)

    write_calls: list[str] = []

    async def _fake_write(sql_model_cls, schema_cls, payload, extra_fields=None, *, kind: str | None = None) -> bool:
        write_calls.append(kind or getattr(sql_model_cls, "__name__", "unknown"))
        return True

    monkeypatch.setattr(worker, "_write", _fake_write)

    grammar_corr = uuid4()
    grammar_env = BaseEnvelope(
        kind="grammar.event.v1",
        correlation_id=grammar_corr,
        source=ServiceRef(name="test", version="0.0.1", node="local"),
        payload=_grammar_payload(),
    )
    await worker.handle_envelope(grammar_env, bus=None)

    assert fallback_calls
    assert fallback_calls[0][0] == "grammar.event.v1"
    assert fallback_calls[0][1] is not None
    assert "timeout" in fallback_calls[0][1].lower()

    tick_env = BaseEnvelope(
        kind="metacognition.tick.v1",
        correlation_id=uuid4(),
        source=ServiceRef(name="test", version="0.0.1", node="local"),
        payload=_metacog_payload(),
    )
    await worker.handle_envelope(tick_env, bus=None)

    assert "metacognition.tick.v1" in write_calls


def test_db_connect_args_include_statement_timeout_when_configured() -> None:
    args = db_module.build_engine_connect_args(30_000)
    assert "options" in args
    assert "statement_timeout=30000" in args["options"]


def test_db_connect_args_omit_statement_timeout_when_disabled() -> None:
    assert db_module.build_engine_connect_args(0) == {}
