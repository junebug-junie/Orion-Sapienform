from __future__ import annotations

from types import SimpleNamespace

from scripts.autonomy_payloads import extract_autonomy_payload


def test_extract_autonomy_payload_includes_turn_effect_fields() -> None:
    cortex_result = SimpleNamespace(
        metadata={
            "autonomy_summary": {"stance_hint": "x"},
            "turn_effect": {"turn": {"coherence": -0.1}},
            "turn_effect_evidence": {"phi_before": {"coherence": 0.4}},
            "turn_effect_status": "present",
            "turn_effect_missing_reason": None,
        }
    )
    out = extract_autonomy_payload(cortex_result)
    assert out["turn_effect"]["turn"]["coherence"] == -0.1
    assert out["turn_effect_evidence"]["phi_before"]["coherence"] == 0.4
    assert out["turn_effect_status"] == "present"


def test_extract_autonomy_payload_includes_execution_mode_and_goal_lineage() -> None:
    cortex_result = SimpleNamespace(
        metadata={
            "autonomy_execution_mode": "proposal_only",
            "autonomy_goal_lineage": {
                "goal_artifact_id": "goal-abc",
                "proposal_signature": "deadbeef01",
            },
        }
    )
    out = extract_autonomy_payload(cortex_result)
    assert out["autonomy_execution_mode"] == "proposal_only"
    assert out["autonomy_goal_lineage"]["goal_artifact_id"] == "goal-abc"
