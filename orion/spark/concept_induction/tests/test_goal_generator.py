from __future__ import annotations

from orion.spark.concept_induction.goal_generator import generate_goal_statement


def test_template_mode_returns_base_template():
    text = generate_goal_statement(
        drive_origin="coherence",
        pressures={"coherence": 0.8},
        tensions=[],
        window_summary=None,
        mode="template",
    )
    assert "coherence" in text.lower() or "Stabilize internal coherence" in text


def test_evidence_rules_adds_tension_clause():
    from orion.core.schemas.drives import TensionEventV1

    tension = TensionEventV1.model_validate({
        "artifact_id": "t-1",
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "tension.coherence_gap.v1",
        "magnitude": 0.7,
        "drive_impacts": {"coherence": 0.5},
        "provenance": {
            "intake_channel": "x",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
    })
    text = generate_goal_statement(
        drive_origin="coherence",
        pressures={"coherence": 0.8},
        tensions=[tension],
        window_summary="recent deployment friction",
        mode="evidence_rules",
    )
    assert "Primary tension:" in text
    assert "deployment" in text.lower() or "friction" in text.lower()


def test_llm_mode_falls_back_to_evidence_rules(monkeypatch):
    monkeypatch.setattr(
        "orion.spark.concept_induction.goal_generator._llm_goal_text",
        lambda **kw: None,
    )
    text = generate_goal_statement(
        drive_origin="autonomy",
        pressures={"autonomy": 0.5},
        tensions=[],
        window_summary=None,
        mode="llm",
    )
    assert text
    assert len(text) <= 120
