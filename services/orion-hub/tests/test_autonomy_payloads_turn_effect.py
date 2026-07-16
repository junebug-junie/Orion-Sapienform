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


def test_extract_autonomy_payload_forwards_v2_keys() -> None:
    cortex_result = SimpleNamespace(
        metadata={
            "autonomy_summary": {"stance_hint": "x"},
            "autonomy_state_v2_preview": {"dominant_drive": "coherence"},
            "autonomy_state_delta": {"subject": "orion"},
        }
    )
    payload = extract_autonomy_payload(cortex_result)
    assert payload["autonomy_state_v2_preview"]["dominant_drive"] == "coherence"
    assert payload["autonomy_state_delta"]["subject"] == "orion"


def test_extract_autonomy_payload_forwards_drive_state_preview_alongside_v2_preview() -> None:
    # Additive, side-by-side signal -- must not replace autonomy_state_v2_preview.
    cortex_result = SimpleNamespace(
        metadata={
            "autonomy_state_v2_preview": {"dominant_drive": "coherence"},
            "drive_state_preview": {"dominant_drive": "curiosity", "pressures": {"curiosity": 0.6}},
        }
    )
    payload = extract_autonomy_payload(cortex_result)
    assert payload["autonomy_state_v2_preview"]["dominant_drive"] == "coherence"
    assert payload["drive_state_preview"]["dominant_drive"] == "curiosity"
    assert payload["drive_state_preview"]["pressures"]["curiosity"] == 0.6
