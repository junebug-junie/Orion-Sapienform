from orion.memory.consolidation_gate import ConsolidationGateResult
from orion.memory.crystallization.intake_consolidation_window import build_crystallization_from_window


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
