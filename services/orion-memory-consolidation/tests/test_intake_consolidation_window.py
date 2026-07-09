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
