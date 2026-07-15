from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from orion.grammar.ledger import _upsert_trace, apply_grammar_event, apply_grammar_trace_batch
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.ids import cortex_exec_trace_id


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


def test_trace_started_prefers_observed_at_over_emitted_at() -> None:
    """started_at/ended_at should reflect real occurrence time (observed_at,
    via _created_at()'s existing fallback chain) not bus-publish time
    (emitted_at, uniform across one flush batch by design -- never a
    meaningful trace-duration signal). Found by review 2026-07-14: this was
    previously hardcoded to emitted_at, which is why grammar_traces.started_at/
    ended_at collapsed to zero duration even for producers with real
    per-atom observed_at."""
    session = MagicMock()
    emitted = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    observed = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    event = GrammarEventV1(
        event_id="evt:1",
        event_kind="trace_started",
        trace_id="trace:t1",
        emitted_at=emitted,
        observed_at=observed,
        provenance=GrammarProvenanceV1(source_service="test"),
    )

    _upsert_trace(session, event)

    stmt = session.execute.call_args[0][0]
    params = stmt.compile().params
    assert params["started_at"] == observed
    assert params["started_at"] != emitted


def test_harness_and_unlaned_cortex_exec_trace_ids_no_longer_share_ledger_key() -> None:
    """Regression for the trace_id collision confirmed live 2026-07-15 (corr
    99487e95-709c-4340-8bf5-c4c9840a247b): before this fix,
    HarnessGrammarCollector.trace_id and an unrelated, unlaned (root,
    trace_lane=None) CortexExecGrammarCollector call sharing the same
    node_name/correlation_id both computed the exact same bare
    `cortex.exec:{node}:{correlation_id}` string. _upsert_trace's
    on_conflict_do_update(index_elements=["trace_id"]) keys grammar_traces
    on trace_id alone (no source_service in the key), so whichever
    producer's trace_started/trace_ended landed last in the ledger silently
    overwrote the other's started_at/ended_at -- live evidence: a real ~30s
    harness motor run was erased and replaced by an unrelated ~0.7s
    cortex-exec blip. HarnessGrammarCollector now passes lane="harness_motor"
    (see orion/harness/grammar_emit.py), so the two producers upsert into
    two distinct grammar_traces rows instead of one shared row."""
    session = MagicMock()
    node, corr = "athena", "99487e95-709c-4340-8bf5-c4c9840a247b"

    harness_started = GrammarEventV1(
        event_id="evt:harness:started",
        event_kind="trace_started",
        # Mirrors HarnessGrammarCollector.trace_id's fixed lane.
        trace_id=cortex_exec_trace_id(node, corr, lane="harness_motor"),
        emitted_at=datetime(2026, 7, 15, 15, 58, 20, tzinfo=timezone.utc),
        observed_at=datetime(2026, 7, 15, 15, 58, 20, tzinfo=timezone.utc),
        provenance=GrammarProvenanceV1(source_service="orion-harness-governor"),
    )
    cortex_exec_root_started = GrammarEventV1(
        event_id="evt:cortex_exec:started",
        event_kind="trace_started",
        # Mirrors CortexExecGrammarCollector.trace_id for a root call
        # (trace_lane_for_verb() returns None for any verb outside
        # CORTEX_EXEC_ISOLATED_TRACE_LANES) -- the exact slot that used to
        # collide with the harness governor's own trace_id.
        trace_id=cortex_exec_trace_id(node, corr, lane=None),
        emitted_at=datetime(2026, 7, 15, 15, 59, 27, tzinfo=timezone.utc),
        observed_at=datetime(2026, 7, 15, 15, 59, 27, tzinfo=timezone.utc),
        provenance=GrammarProvenanceV1(source_service="orion-cortex-exec"),
    )

    _upsert_trace(session, harness_started)
    harness_stmt = session.execute.call_args[0][0]
    harness_trace_id = harness_stmt.compile().params["trace_id"]

    _upsert_trace(session, cortex_exec_root_started)
    cortex_exec_stmt = session.execute.call_args[0][0]
    cortex_exec_root_trace_id = cortex_exec_stmt.compile().params["trace_id"]

    assert harness_trace_id != cortex_exec_root_trace_id
    # Both real-world producers still compute a valid cortex.exec-shaped
    # trace_id -- only the lane slot changed, not the overall shape.
    assert harness_trace_id.startswith(f"cortex.exec:{node}:{corr}")
    assert cortex_exec_root_trace_id == f"cortex.exec:{node}:{corr}"


def test_trace_started_falls_back_to_emitted_at_when_observed_at_missing() -> None:
    """Producers that don't populate observed_at (or haven't been fixed to
    populate it accurately yet) must not regress -- _created_at()'s existing
    fallback chain (observed_at or emitted_at or now()) already handles
    this; this test locks that behavior in for _upsert_trace specifically."""
    session = MagicMock()
    emitted = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    event = GrammarEventV1(
        event_id="evt:1",
        event_kind="trace_started",
        trace_id="trace:t1",
        emitted_at=emitted,
        observed_at=None,
        provenance=GrammarProvenanceV1(source_service="test"),
    )

    _upsert_trace(session, event)

    stmt = session.execute.call_args[0][0]
    params = stmt.compile().params
    assert params["started_at"] == emitted
