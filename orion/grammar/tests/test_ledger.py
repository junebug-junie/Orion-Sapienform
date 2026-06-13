from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from orion.grammar.ledger import apply_grammar_event, apply_grammar_trace_batch
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1


def _atom_event(*, event_id: str = "evt:1") -> GrammarEventV1:
    now = datetime.now(timezone.utc)
    atom = GrammarAtomV1(
        atom_id="atom:a1",
        trace_id="trace:t1",
        atom_type="observation",
        semantic_role="motion",
        layer="sensor_raw",
        summary="Motion detected",
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id="trace:t1",
        emitted_at=now,
        atom=atom,
        provenance=GrammarProvenanceV1(source_service="test"),
    )


def test_atom_emitted_uses_set_based_insert() -> None:
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [("evt:1",)]

    assert apply_grammar_event(session, _atom_event()) is True
    assert session.execute.call_count >= 2
    session.flush.assert_called_once()


def test_trace_batch_applies_multiple_events() -> None:
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = [
        ("evt:0",),
        ("evt:1",),
        ("evt:2",),
    ]
    events = [_atom_event(event_id=f"evt:{idx}") for idx in range(3)]

    applied = apply_grammar_trace_batch(session, events)

    assert applied == 3
    session.flush.assert_called_once()


def test_trace_batch_dedupes_existing_event_ids() -> None:
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []
    event = _atom_event()

    assert apply_grammar_event(session, event) is False

    session.execute.return_value.fetchall.return_value = [("evt:1",)]
    assert apply_grammar_event(session, event) is True
