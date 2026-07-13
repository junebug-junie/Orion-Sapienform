"""Tests for hub chat turn grammar event emitter (TDD — written before implementation)."""
from __future__ import annotations

import pytest

from orion.schemas.grammar import AtomType, GrammarEventV1, RelationType


def _build(**kwargs):
    from scripts.grammar_emit import build_chat_turn_grammar_events

    defaults = dict(
        turn_id="turn123",
        session_id="sess456",
        node_id="athena",
        word_count=10,
    )
    defaults.update(kwargs)
    return build_chat_turn_grammar_events(**defaults)


def test_build_returns_grammar_event_v1_instances():
    events = _build()
    assert len(events) > 0
    for ev in events:
        assert isinstance(ev, GrammarEventV1)


def test_trace_sequence_has_correct_event_kinds():
    events = _build()
    assert events[0].event_kind == "trace_started"
    assert events[-1].event_kind == "trace_ended"
    middle = events[1:-1]
    assert len(middle) > 0
    for ev in middle:
        assert ev.event_kind in ("atom_emitted", "edge_emitted")


def test_user_utterance_has_no_raw_text():
    events = _build(word_count=42)
    utterance_atoms = [
        ev.atom
        for ev in events
        if ev.atom is not None and ev.atom.semantic_role == "user_utterance"
    ]
    assert len(utterance_atoms) == 1
    assert utterance_atoms[0].text_value is None


def test_stable_event_ids_same_inputs_same_ids():
    from datetime import datetime, timezone

    fixed_ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    events_a = _build(turn_id="t1", session_id="s1", observed_at=fixed_ts)
    events_b = _build(turn_id="t1", session_id="s1", observed_at=fixed_ts)
    ids_a = [ev.event_id for ev in events_a]
    ids_b = [ev.event_id for ev in events_b]
    assert ids_a == ids_b


def test_repair_signal_absent_when_flag_false():
    events = _build(has_repair_signal=False)
    roles = [
        ev.atom.semantic_role
        for ev in events
        if ev.atom is not None
    ]
    assert "repair_signal" not in roles


def test_repair_signal_present_when_flag_true():
    events = _build(
        has_repair_signal=True,
        repair_pressure_level=0.75,
        repair_pressure_confidence=0.9,
    )
    repair_atoms = [
        ev.atom
        for ev in events
        if ev.atom is not None and ev.atom.semantic_role == "repair_signal"
    ]
    assert len(repair_atoms) == 1
    assert repair_atoms[0].confidence == pytest.approx(0.9)


def test_stance_disposition_absent_when_not_provided():
    events = _build()
    roles = [ev.atom.semantic_role for ev in events if ev.atom is not None]
    assert "stance_disposition" not in roles


def test_stance_disposition_present_when_provided():
    events = _build(
        stance_disposition="defer",
        stance_disposition_reasons=["stale_broadcast_no_evidence"],
        stance_boundary_register=True,
    )
    stance_atoms = [
        ev.atom
        for ev in events
        if ev.atom is not None and ev.atom.semantic_role == "stance_disposition"
    ]
    assert len(stance_atoms) == 1
    atom = stance_atoms[0]
    assert atom.text_value == "defer"
    assert "stale_broadcast_no_evidence" in atom.summary
    assert "[boundary_register]" in atom.summary


def test_stance_disposition_edge_links_to_user_utterance():
    events = _build(stance_disposition="proceed")
    edges = [ev.edge for ev in events if ev.edge is not None]
    stance_edges = [
        e for e in edges if e is not None and "stance_disposition" in e.from_atom_id
    ]
    assert len(stance_edges) == 1
    assert "user_utterance" in stance_edges[0].to_atom_id


def test_atom_types_are_valid():
    import typing

    valid_types = set(typing.get_args(AtomType))
    events = _build(has_repair_signal=True)
    for ev in events:
        if ev.atom is not None:
            assert ev.atom.atom_type in valid_types, (
                f"Invalid atom_type: {ev.atom.atom_type!r}"
            )


def test_relation_types_are_valid():
    import typing

    valid_types = set(typing.get_args(RelationType))
    events = _build(has_repair_signal=True)
    for ev in events:
        if ev.edge is not None:
            assert ev.edge.relation_type in valid_types, (
                f"Invalid relation_type: {ev.edge.relation_type!r}"
            )
