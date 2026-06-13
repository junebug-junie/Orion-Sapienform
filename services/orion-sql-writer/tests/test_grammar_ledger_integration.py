"""Postgres integration tests for grammar ledger persistence (real INSERT path)."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from orion.grammar.ledger import apply_grammar_trace_batch

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from grammar_integration_helpers import (
    assert_grammar_event_indexes_valid,
    bus_transport_trace_batch,
    delete_trace,
    ensure_grammar_schema,
    grammar_session_factory,
    postgres_uri,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def grammar_engine() -> Engine:
    engine = create_engine(postgres_uri(), pool_pre_ping=True)
    ensure_grammar_schema(engine)
    assert_grammar_event_indexes_valid(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def grammar_session(grammar_engine: Engine):
    session = grammar_session_factory(grammar_engine)()
    suffix = uuid.uuid4().hex[:12]
    events = bus_transport_trace_batch(trace_suffix=suffix)
    trace_id = events[0].trace_id
    try:
        yield session, events
    finally:
        session.rollback()
        session.close()
        delete_trace(grammar_engine, trace_id)


def test_bus_trace_batch_persists_13_events_quickly(grammar_session) -> None:
    session, events = grammar_session
    assert len(events) == 13

    started = time.monotonic()
    applied = apply_grammar_trace_batch(session, events)
    session.commit()
    elapsed = time.monotonic() - started

    assert applied == 13
    assert elapsed < 5.0, f"13-event batch took {elapsed:.2f}s (expected <5s on empty/small table)"

    row = session.execute(
        text("SELECT COUNT(*) FROM grammar_events WHERE trace_id = :trace_id"),
        {"trace_id": events[0].trace_id},
    ).scalar_one()
    assert row == 13


def test_bus_trace_batch_dedupes_on_replay(grammar_session) -> None:
    session, events = grammar_session

    assert apply_grammar_trace_batch(session, events) == 13
    session.commit()
    assert apply_grammar_trace_batch(session, events) == 0
    session.commit()

    row = session.execute(
        text("SELECT COUNT(*) FROM grammar_events WHERE trace_id = :trace_id"),
        {"trace_id": events[0].trace_id},
    ).scalar_one()
    assert row == 13


def test_batch_uses_set_based_insert_without_savepoints(grammar_engine: Engine) -> None:
    suffix = uuid.uuid4().hex[:12]
    events = bus_transport_trace_batch(trace_suffix=suffix)
    trace_id = events[0].trace_id
    statements: list[str] = []

    @event.listens_for(grammar_engine, "before_cursor_execute")
    def _capture_sql(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(str(statement))

    session = grammar_session_factory(grammar_engine)()
    try:
        apply_grammar_trace_batch(session, events)
        session.commit()
    finally:
        event.remove(grammar_engine, "before_cursor_execute", _capture_sql)
        session.close()
        delete_trace(grammar_engine, trace_id)

    lowered = "\n".join(statements).lower()
    event_inserts = [stmt for stmt in statements if "insert into grammar_events" in stmt.lower()]
    assert event_inserts, "expected grammar_events INSERT"
    assert len(event_inserts) == 1, "expected one set-based grammar_events INSERT per batch"
    assert "on conflict" in event_inserts[0].lower()
    assert "savepoint" not in lowered
