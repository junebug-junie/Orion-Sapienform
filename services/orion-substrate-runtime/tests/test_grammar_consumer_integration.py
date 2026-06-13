"""Postgres integration tests for substrate grammar event consumption cursors."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_TESTS = REPO_ROOT / "services/orion-sql-writer/tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_TESTS) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_TESTS))

from grammar_integration_helpers import (
    bus_transport_trace_batch,
    delete_trace,
    ensure_grammar_schema,
    load_apply_grammar_trace_batch,
    load_biometrics_substrate_store_class,
    postgres_uri,
)
from orion.substrate.transport_loop.constants import TRANSPORT_GRAMMAR_CURSOR_NAME

apply_grammar_trace_batch = load_apply_grammar_trace_batch()
BiometricsSubstrateStore = load_biometrics_substrate_store_class()

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def grammar_engine() -> Engine:
    engine = create_engine(postgres_uri(), pool_pre_ping=True)
    ensure_grammar_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def reset_transport_cursor(grammar_engine: Engine) -> None:
    with grammar_engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM substrate_reduction_cursor
                WHERE cursor_name = :cursor_name
                """
            ),
            {"cursor_name": TRANSPORT_GRAMMAR_CURSOR_NAME},
        )


@pytest.fixture
def substrate_store(grammar_engine: Engine):
    _ = grammar_engine
    return BiometricsSubstrateStore(postgres_uri())


def test_transport_cold_start_seeds_cursor_without_replaying_history(
    grammar_engine: Engine,
    substrate_store,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    old_trace = bus_transport_trace_batch(trace_suffix=f"old_{suffix}")
    old_trace_id = old_trace[0].trace_id
    old_created_at = datetime.now(timezone.utc) - timedelta(days=30)

    with grammar_engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM substrate_reduction_cursor
                WHERE cursor_name = :cursor_name
                """
            ),
            {"cursor_name": TRANSPORT_GRAMMAR_CURSOR_NAME},
        )

    session = sessionmaker(bind=grammar_engine)()
    try:
        for event in old_trace:
            event.emitted_at = old_created_at
            event.observed_at = old_created_at
        apply_grammar_trace_batch(session, old_trace)
        session.commit()
    finally:
        session.close()

    events = substrate_store.fetch_transport_grammar_events(limit=50)
    assert events == []

    with grammar_engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT last_event_created_at, last_event_id
                FROM substrate_reduction_cursor
                WHERE cursor_name = :cursor_name
                """
            ),
            {"cursor_name": TRANSPORT_GRAMMAR_CURSOR_NAME},
        ).mappings().one()
    assert row["last_event_created_at"] is not None

    delete_trace(grammar_engine, old_trace_id)


def test_transport_cursor_fetches_only_events_after_tail(
    grammar_engine: Engine,
    substrate_store,
) -> None:
    suffix = uuid.uuid4().hex[:12]
    seed_trace = bus_transport_trace_batch(trace_suffix=f"seed_{suffix}")
    seed_trace_id = seed_trace[0].trace_id

    session = sessionmaker(bind=grammar_engine)()
    try:
        apply_grammar_trace_batch(session, seed_trace)
        session.commit()
    finally:
        session.close()

    with grammar_engine.connect() as conn:
        tail = conn.execute(
            text(
                """
                SELECT created_at, event_id
                FROM grammar_events
                WHERE trace_id = :trace_id
                ORDER BY created_at DESC, event_id DESC
                LIMIT 1
                """
            ),
            {"trace_id": seed_trace_id},
        ).mappings().one()
    substrate_store.advance_transport_cursor(
        event_id=tail["event_id"],
        created_at=tail["created_at"],
    )

    new_trace = bus_transport_trace_batch(trace_suffix=f"new_{suffix}")
    new_trace_id = new_trace[0].trace_id
    newer_than_seed = datetime.now(timezone.utc) + timedelta(minutes=1)
    for event in new_trace:
        event.emitted_at = newer_than_seed
        event.observed_at = newer_than_seed
    session = sessionmaker(bind=grammar_engine)()
    try:
        apply_grammar_trace_batch(session, new_trace)
        session.commit()
    finally:
        session.close()

    events = substrate_store.fetch_transport_grammar_events(limit=50)
    assert len(events) == 13
    assert events[0].trace_id == new_trace_id

    delete_trace(grammar_engine, seed_trace_id)
    delete_trace(grammar_engine, new_trace_id)
