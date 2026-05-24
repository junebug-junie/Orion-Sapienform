from __future__ import annotations

from orion.grammar.seed_demo import TRACE_ID, build_vision_demo_events
from orion.schemas.grammar import GrammarEventV1


def test_build_vision_demo_events_count_and_validation() -> None:
    events = build_vision_demo_events()
    assert len(events) >= 10
    assert events[0].event_kind == "trace_started"
    assert events[0].trace_id == TRACE_ID
    assert events[-1].event_kind == "trace_ended"
    for event in events:
        GrammarEventV1.model_validate(event.model_dump(mode="json"))


def test_vision_demo_includes_required_atoms_and_edges() -> None:
    events = build_vision_demo_events()
    roles = {
        e.atom.semantic_role
        for e in events
        if e.event_kind == "atom_emitted" and e.atom is not None
    }
    assert roles >= {
        "frame_observed",
        "motion_detected",
        "object_candidate_person",
        "uncertainty_marker",
        "spatial_region",
        "scene_state",
    }
    relations = {
        e.edge.relation_type
        for e in events
        if e.event_kind == "edge_emitted" and e.edge is not None
    }
    assert "derived_from" in relations
    assert "supports" in relations
    assert "compacted_into" in relations
    assert any(e.event_kind == "compaction_emitted" for e in events)
    assert any(e.event_kind == "projection_emitted" for e in events)
