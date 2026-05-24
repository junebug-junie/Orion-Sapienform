from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from orion.grammar.ledger import apply_grammar_event
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


def test_atom_emitted_calls_session_add() -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None

    assert apply_grammar_event(session, _atom_event()) is True
    assert session.add.called


def test_dedupe_returns_false_on_second_call() -> None:
    session = MagicMock()
    event = _atom_event()
    with patch(
        "orion.grammar.ledger._event_exists",
        side_effect=[False, True],
    ) as event_exists:
        assert apply_grammar_event(session, event) is True
        assert apply_grammar_event(session, event) is False
        assert event_exists.call_count == 2
