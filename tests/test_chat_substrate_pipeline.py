"""Tests for chat substrate pipeline."""
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.chat_loop.pipeline import process_chat_grammar_events


def _provenance(service: str = "orion-hub") -> GrammarProvenanceV1:
    return GrammarProvenanceV1(source_service=service, source_component="test")


def _atom_event(
    trace_id: str,
    role: str,
    summary: str,
    event_id: str | None = None,
    service: str = "orion-hub",
) -> GrammarEventV1:
    eid = event_id or f"ev_{trace_id}_{role}"
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:{role}",
        trace_id=trace_id,
        atom_type="raw_span",
        semantic_role=role,
        layer="raw_input",
        summary=summary,
        source_event_id="evt-src",
    )
    return GrammarEventV1(
        event_id=eid,
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=datetime.now(timezone.utc),
        provenance=_provenance(service),
        atom=atom,
    )


def _fresh_projection() -> ChatSessionProjectionV1:
    return ChatSessionProjectionV1(
        projection_id="active_chat_session",
        generated_at=datetime.now(timezone.utc),
    )


def test_pipeline_groups_by_trace():
    """4 events across 2 traces → 2 receipts, 2 traces counted."""
    trace1 = "hub.chat:athena:turn-1"
    trace2 = "hub.chat:athena:turn-2"
    events = [
        _atom_event(trace1, "session_context", "ctx", event_id="ev1a"),
        _atom_event(trace1, "user_utterance", "User message in session s1 (10 words)", event_id="ev1b"),
        _atom_event(trace2, "session_context", "ctx", event_id="ev2a"),
        _atom_event(trace2, "user_utterance", "User message in session s2 (20 words)", event_id="ev2b"),
    ]

    saved_receipts = []
    saved_projections = []
    proj = _fresh_projection()

    stats = process_chat_grammar_events(
        events=events,
        load_projection=lambda: proj,
        save_projection=lambda p: saved_projections.append(p),
        save_receipt=lambda r: saved_receipts.append(r),
    )

    assert stats["traces"] == 2
    assert stats["receipts"] == 2
    assert stats["events"] == 4
    assert len(saved_receipts) == 2


def test_pipeline_skips_non_chat_prefix():
    """Events with non-hub.chat: trace_id are skipped entirely (not counted as traces)."""
    events = [
        _atom_event("biometrics.node:x:y", "user_utterance", "something", event_id="ev-bio"),
        _atom_event("cortex.exec:z:w", "user_utterance", "something", event_id="ev-exec"),
        _atom_event("hub.chat:athena:turn-3", "user_utterance", "User message in session s3 (5 words)", event_id="ev-chat"),
    ]

    saved_receipts = []
    proj = _fresh_projection()

    stats = process_chat_grammar_events(
        events=events,
        load_projection=lambda: proj,
        save_projection=lambda p: None,
        save_receipt=lambda r: saved_receipts.append(r),
    )

    # Only the hub.chat: trace is processed
    assert stats["traces"] == 1
    assert stats["receipts"] == 1
    assert stats["events"] == 3


def test_pipeline_returns_stats():
    """Stats dict always contains events/receipts/traces keys."""
    proj = _fresh_projection()
    stats = process_chat_grammar_events(
        events=[],
        load_projection=lambda: proj,
        save_projection=lambda p: None,
        save_receipt=lambda r: None,
    )
    assert "events" in stats
    assert "receipts" in stats
    assert "traces" in stats
    assert stats["events"] == 0
    assert stats["traces"] == 0
    assert stats["receipts"] == 0
