"""Compile-time checks for set-based grammar ledger SQL (no Postgres required)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.grammar_trace import GrammarEventSQL
from orion.schemas.grammar import GrammarEventV1, GrammarProvenanceV1


def _sample_event(*, event_id: str = "gev_shape") -> GrammarEventV1:
    now = datetime.now(timezone.utc)
    return GrammarEventV1(
        event_id=event_id,
        event_kind="trace_started",
        trace_id="bus.transport:shape:test",
        emitted_at=now,
        layer="transport",
        dimensions=["bus"],
        provenance=GrammarProvenanceV1(source_service="orion-bus"),
    )


def test_compiled_event_insert_uses_on_conflict_do_nothing() -> None:
    sample = _sample_event()
    row_a = {
        "event_id": "gev_shape_a",
        "trace_id": "bus.transport:shape:test",
        "parent_event_id": None,
        "root_event_id": None,
        "event_kind": "trace_started",
        "session_id": None,
        "turn_id": None,
        "correlation_id": None,
        "layer": "transport",
        "dimensions": ["bus"],
        "emitted_at": sample.emitted_at,
        "observed_at": None,
        "source_service": "orion-bus",
        "source_component": None,
        "source_event_id": None,
        "payload_ref": None,
        "event_json": sample.model_dump(mode="json"),
        "created_at": sample.emitted_at,
    }
    row_b = {**row_a, "event_id": "gev_shape_b"}
    stmt = (
        pg_insert(GrammarEventSQL.__table__)
        .values([row_a, row_b])
        .on_conflict_do_nothing(index_elements=["event_id"])
    )
    compiled = str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    ).upper()
    assert "ON CONFLICT" in compiled
    assert "DO NOTHING" in compiled
    assert "SAVEPOINT" not in compiled
