from datetime import datetime

from orion.autonomy.models import AutonomyStateV2, InhibitedImpulseV1
from orion.autonomy.summary import summarize_autonomy_state


def test_v2_low_confidence_hazard() -> None:
    s = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="graph",
        generated_at=datetime.utcnow(),
        drive_pressures={"coherence": 0.1},
        tension_kinds=[],
        goal_headlines=[],
        confidence=0.35,
        unknowns=[],
        inhibited_impulses=[],
        attention_items=[],
    )
    out = summarize_autonomy_state(s)
    assert "avoid overconfident inner-state claims" in out.response_hazards


def test_v2_proxy_inhibition_hazard() -> None:
    s = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="graph",
        generated_at=datetime.utcnow(),
        drive_pressures={},
        tension_kinds=[],
        goal_headlines=[],
        confidence=0.9,
        unknowns=[],
        inhibited_impulses=[
            InhibitedImpulseV1(
                impulse_id="i1",
                kind="k",
                summary="s",
                inhibition_reason="proxy_signal_not_canonical_state",
            )
        ],
        attention_items=[],
    )
    out = summarize_autonomy_state(s)
    assert "do not treat proxy telemetry as canonical state" in out.response_hazards
