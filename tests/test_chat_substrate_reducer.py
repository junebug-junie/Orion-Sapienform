"""Tests for chat substrate reducer: grammar_extract and reducer."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.chat_loop.grammar_extract import (
    compute_chat_pressure_hints,
    extract_chat_turn_state,
)
from orion.substrate.chat_loop.reducer import reduce_chat_trace_events


def _provenance(service: str = "orion-hub") -> GrammarProvenanceV1:
    return GrammarProvenanceV1(source_service=service, source_component="test")


def _atom_event(
    trace_id: str,
    role: str,
    atom_type: str,
    summary: str,
    text_value: str | None = None,
    salience: float | None = None,
    confidence: float | None = None,
    layer: str = "raw_input",
    service: str = "orion-hub",
    event_id: str | None = None,
) -> GrammarEventV1:
    eid = event_id or f"ev_{role}"
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:{role}",
        trace_id=trace_id,
        atom_type=atom_type,
        semantic_role=role,
        layer=layer,
        summary=summary,
        text_value=text_value,
        salience=salience,
        confidence=confidence,
        source_event_id="evt1",
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


# ---------------------------------------------------------------------------
# extract_chat_turn_state tests
# ---------------------------------------------------------------------------

def test_extract_basic_turn_state():
    trace_id = "hub.chat:athena:turn-42"
    events = [
        _atom_event(
            trace_id,
            "session_context",
            "scene_state",
            "session context",
            text_value="sess-001",
        ),
        _atom_event(
            trace_id,
            "user_utterance",
            "raw_span",
            "User message in session sess-001 (30 words)",
        ),
    ]
    turn = extract_chat_turn_state(events)
    assert turn.turn_id == "turn-42"
    assert turn.node_id == "athena"
    assert turn.session_id == "sess-001"
    assert turn.word_count == 30
    assert not turn.has_repair_signal


def test_extract_repair_signal_present():
    trace_id = "hub.chat:athena:turn-99"
    events = [
        _atom_event(
            trace_id,
            "repair_signal",
            "signal",
            "repair detected",
            salience=0.8,
            confidence=0.9,
        ),
    ]
    turn = extract_chat_turn_state(events)
    assert turn.has_repair_signal is True
    assert turn.repair_pressure_level == pytest.approx(0.8)
    assert turn.repair_pressure_confidence == pytest.approx(0.9)


def test_extract_stance_disposition_present():
    trace_id = "hub.chat:athena:turn-201"
    events = [
        _atom_event(
            trace_id,
            "stance_disposition",
            "signal",
            "Stance disposition: defer (stale_broadcast_no_evidence; low_confidence) [boundary_register]",
            text_value="defer",
        ),
    ]
    turn = extract_chat_turn_state(events)
    assert turn.stance_disposition == "defer"
    assert turn.stance_disposition_reasons == ["stale_broadcast_no_evidence", "low_confidence"]
    assert turn.stance_boundary_register is True


def test_extract_stance_disposition_defaults_to_unknown_when_absent():
    trace_id = "hub.chat:athena:turn-202"
    events = [
        _atom_event(
            trace_id,
            "user_utterance",
            "raw_span",
            "User message in session sess-001 (5 words)",
        ),
    ]
    turn = extract_chat_turn_state(events)
    assert turn.stance_disposition == "unknown"
    assert turn.stance_disposition_reasons == []
    assert turn.stance_boundary_register is False


def test_extract_stance_disposition_no_reasons_no_parens_in_summary():
    trace_id = "hub.chat:athena:turn-203"
    events = [
        _atom_event(
            trace_id,
            "stance_disposition",
            "signal",
            "Stance disposition: proceed",
            text_value="proceed",
        ),
    ]
    turn = extract_chat_turn_state(events)
    assert turn.stance_disposition == "proceed"
    assert turn.stance_disposition_reasons == []
    assert turn.stance_boundary_register is False


def test_extract_noop_on_wrong_source():
    trace_id = "hub.chat:athena:turn-77"
    events = [
        _atom_event(
            trace_id,
            "user_utterance",
            "raw_span",
            "User message in session sess-x (10 words)",
            service="other-service",
        ),
    ]
    turn = extract_chat_turn_state(events)
    # Events from wrong service are skipped; word_count stays 0
    assert turn.word_count == 0
    assert turn.session_id == ""


def test_extract_raises_on_wrong_prefix():
    events = [
        _atom_event(
            "biometrics.node:x:y",
            "user_utterance",
            "raw_span",
            "whatever",
        ),
    ]
    with pytest.raises(ValueError, match="hub.chat:"):
        extract_chat_turn_state(events)


# ---------------------------------------------------------------------------
# compute_chat_pressure_hints tests
# ---------------------------------------------------------------------------

def test_compute_pressure_hints_load():
    trace_id = "hub.chat:athena:turn-1"
    events = [
        _atom_event(
            trace_id,
            "user_utterance",
            "raw_span",
            "User message in session s1 (150 words)",
            text_value=None,
        ),
    ]
    turn = extract_chat_turn_state(events)
    hints = compute_chat_pressure_hints(turn)
    assert hints["conversation_load"] == pytest.approx(1.0)


def test_compute_pressure_hints_repair():
    trace_id = "hub.chat:athena:turn-2"
    events = [
        _atom_event(
            trace_id,
            "repair_signal",
            "signal",
            "repair",
            salience=0.7,
            confidence=0.5,
        ),
    ]
    turn = extract_chat_turn_state(events)
    hints = compute_chat_pressure_hints(turn)
    assert hints["repair_pressure"] == pytest.approx(0.7)
    assert hints["topic_coherence"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# reducer tests
# ---------------------------------------------------------------------------

def test_reducer_empty_events_returns_noop():
    proj = _fresh_projection()
    updated, receipt = reduce_chat_trace_events(events=[], projection=proj)
    assert updated is proj
    assert receipt.accepted_event_ids == []
    assert receipt.state_deltas == []


def test_reducer_wrong_prefix_returns_noop():
    proj = _fresh_projection()
    events = [
        _atom_event(
            "biometrics.node:x:y",
            "user_utterance",
            "raw_span",
            "some summary",
        ),
    ]
    updated, receipt = reduce_chat_trace_events(events=events, projection=proj)
    assert updated is proj
    assert receipt.noop_event_ids == [e.event_id for e in events]
    assert receipt.accepted_event_ids == []


def test_reducer_produces_stable_delta_id():
    proj = _fresh_projection()
    trace_id = "hub.chat:athena:turn-5"
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    events = [
        _atom_event(
            trace_id,
            "user_utterance",
            "raw_span",
            "User message in session sess-abc (20 words)",
            event_id="ev-stable-1",
        ),
    ]

    _, receipt1 = reduce_chat_trace_events(events=events, projection=proj, now=now)
    proj2 = _fresh_projection()
    _, receipt2 = reduce_chat_trace_events(events=events, projection=proj2, now=now)

    assert receipt1.state_deltas[0].delta_id == receipt2.state_deltas[0].delta_id


def test_reducer_updates_projection_turns():
    proj = _fresh_projection()
    trace_id = "hub.chat:athena:turn-10"
    events = [
        _atom_event(
            trace_id,
            "session_context",
            "scene_state",
            "ctx",
            text_value="sess-xyz",
        ),
        _atom_event(
            trace_id,
            "user_utterance",
            "raw_span",
            "User message in session sess-xyz (5 words)",
            event_id="ev-utt",
        ),
    ]
    updated, receipt = reduce_chat_trace_events(events=events, projection=proj)
    assert "turn-10" in updated.turns
    assert updated.turns["turn-10"].session_id == "sess-xyz"
    assert receipt.accepted_event_ids  # non-empty
    assert receipt.state_deltas[0].target_kind == "chat_turn"
    assert receipt.state_deltas[0].operation == "create"
