from orion.memory.consolidation_gate import ConsolidationGateResult
from orion.memory.crystallization.intake_consolidation_window import (
    _planning_and_retrieval_for_kind,
    build_crystallization_from_window,
)
from orion.memory.crystallization.validator import validate_proposal


def test_planning_and_retrieval_for_kind_procedure():
    planning_effects, retrieval_affordances = _planning_and_retrieval_for_kind(
        "procedure", "some summary"
    )
    assert planning_effects
    assert retrieval_affordances == ["retrieve_when:procedural"]


def test_planning_and_retrieval_for_kind_decision():
    planning_effects, retrieval_affordances = _planning_and_retrieval_for_kind(
        "decision", "some summary"
    )
    assert planning_effects
    assert retrieval_affordances == ["retrieve_when:semantic"]


def test_planning_and_retrieval_for_kind_semantic_is_empty():
    assert _planning_and_retrieval_for_kind("semantic", "x") == ([], [])


def test_builds_proposed_semantic_crystallization():
    turns = [
        {
            "correlation_id": "corr-1",
            "prompt": "move logistics alone",
            "response": "that sounds heavy",
            "spark_meta": {},
        }
    ]
    gate = ConsolidationGateResult(action="propose", dominant_shift="TOPIC", grammar_event_ids=["evt-1"])
    crys = build_crystallization_from_window(
        memory_window_id="win-1",
        turns=turns,
        gate=gate,
    )
    assert crys.status == "proposed"
    assert crys.kind == "semantic"
    assert crys.governance.proposed_by == "memory_consolidation_intake"
    assert "corr-1" in [e.source_id for e in crys.evidence if e.source_kind == "chat_turn"]
    assert crys.source_grammar_event_ids == ["evt-1"]


def test_stance_window_builds_approvable_proposal():
    turns = [
        {
            "correlation_id": "corr-stance-1",
            "prompt": "I want you to push back on me more, not just agree.",
            "response": "understood, I'll hold my ground when I disagree",
            "spark_meta": {},
        }
    ]
    gate = ConsolidationGateResult(
        action="propose", dominant_shift="STANCE", grammar_event_ids=["evt-stance-1"]
    )
    crys = build_crystallization_from_window(
        memory_window_id="win-stance",
        turns=turns,
        gate=gate,
    )
    assert crys.kind == "stance"
    assert crys.planning_effects, "stance must have non-empty planning_effects"
    assert crys.retrieval_affordances, "stance must have non-empty retrieval_affordances"
    assert "retrieve_when:relational" in crys.retrieval_affordances
    assert any("push back" in eff for eff in crys.planning_effects)
    result = validate_proposal(crys)
    assert result.valid is True, result.errors


def test_semantic_window_has_no_planning_or_retrieval_enrichment():
    turns = [
        {
            "correlation_id": "corr-topic-1",
            "prompt": "let's talk about the new deployment pipeline",
            "response": "sure, here is how it works",
            "spark_meta": {},
        }
    ]
    gate = ConsolidationGateResult(
        action="propose", dominant_shift="TOPIC", grammar_event_ids=["evt-topic-1"]
    )
    crys = build_crystallization_from_window(
        memory_window_id="win-topic",
        turns=turns,
        gate=gate,
    )
    assert crys.kind == "semantic"
    assert crys.planning_effects == []
    assert crys.retrieval_affordances == []


def test_window_provenance_persists_gate_scores():
    turns = [
        {
            "correlation_id": "corr-a",
            "prompt": "I'm down today",
            "response": "I hear you",
            "memory_significance_score": 0.97,
            "conversation_boundary_score": 0.88,
            "spark_meta": {"turn_change_appraisal": {"novelty_score": 0.99, "shift_kind": "STANCE"}},
        }
    ]
    gate = ConsolidationGateResult(
        action="propose",
        reasons=["substantive_shift"],
        dominant_shift="STANCE",
        window_novelty_max=0.99,
        window_significance_max=0.97,
    )
    crys = build_crystallization_from_window(memory_window_id="win-prov", turns=turns, gate=gate)
    assert crys.provenance["gate_reasons"] == ["substantive_shift"]
    assert crys.provenance["dominant_shift"] == "STANCE"
    assert crys.evidence[0].note and "memory_sig=0.97" in crys.evidence[0].note
    assert "I'm down today" in (crys.evidence[0].excerpt or "")


def test_duplicate_turn_entry_collapses_to_one_evidence_row_keeping_latest_note():
    """Regression for the live duplicate-evidence bug (2026-08-20): a window
    whose turn list already carries two entries for the same reclassified
    correlation_id (the WindowStore.append_turn bug, fixed separately) must
    still only ever mint ONE evidence row for it here -- this is the second,
    defense-in-depth layer for windows that were already open when that fix
    shipped. The later entry's note (the deeper reclassification) wins."""
    turns = [
        {
            "correlation_id": "corr-dup",
            "prompt": "same turn, first pass",
            "response": "ack",
            "memory_significance_score": 0.0,
            "spark_meta": {"turn_change_appraisal": {"shift_kind": "TOPIC"}},
        },
        {
            "correlation_id": "corr-dup",
            "prompt": "same turn, first pass",
            "response": "ack",
            "memory_significance_score": 0.15,
            "spark_meta": {"turn_change_appraisal": {"shift_kind": "STANCE"}},
        },
    ]
    gate = ConsolidationGateResult(action="propose", dominant_shift="STANCE")
    crys = build_crystallization_from_window(memory_window_id="win-dup", turns=turns, gate=gate)

    chat_turn_evidence = [e for e in crys.evidence if e.source_kind == "chat_turn"]
    assert len(chat_turn_evidence) == 1
    assert chat_turn_evidence[0].note and "shift=STANCE" in chat_turn_evidence[0].note


def test_duplicate_grammar_event_id_collapses_to_one_evidence_row():
    """The window's turn list can also duplicate the per-turn grammar-event
    lookup (fetch_grammar_evidence_for_window queries once per turn entry, so a
    duplicated turn entry doubled its results too) -- gate.grammar_event_ids
    arriving with a repeat must not mint two evidence rows for the same id."""
    turns = [{"correlation_id": "corr-1", "prompt": "hi", "response": "ok", "spark_meta": {}}]
    gate = ConsolidationGateResult(
        action="propose", dominant_shift="TOPIC", grammar_event_ids=["evt-1", "evt-1"]
    )
    crys = build_crystallization_from_window(memory_window_id="win-gram-dup", turns=turns, gate=gate)

    grammar_evidence = [e for e in crys.evidence if e.source_kind == "grammar_event"]
    assert len(grammar_evidence) == 1
    assert crys.source_grammar_event_ids == ["evt-1"]
