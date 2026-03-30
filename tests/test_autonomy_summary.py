from __future__ import annotations

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1
from orion.autonomy.summary import summarize_autonomy_state


def test_autonomy_summary_deterministic_and_bounded() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        dominant_drive="continuity",
        active_drives=["continuity", "coherence", "relational_stability", "predictive"],
        drive_pressures={"continuity": 0.95, "coherence": 0.81, "relational_stability": 0.7, "predictive": 0.2},
        tension_kinds=["identity_drift", "scope_sprawl", "timing_risk", "extra"],
        goal_headlines=[
            AutonomyGoalHeadlineV1(
                artifact_id="goal-1",
                goal_statement="Converge runtime loop without auto execution.",
                drive_origin="continuity",
                priority=0.8,
                cooldown_until=None,
                proposal_signature="sig-1",
            )
        ],
        source="graph",
    )

    first = summarize_autonomy_state(state)
    second = summarize_autonomy_state(state)

    assert first.model_dump() == second.model_dump()
    assert first.stance_hint == "preserve continuity and thread integrity"
    assert len(first.top_drives) <= 3
    assert len(first.active_tensions) <= 3
    assert len(first.proposal_headlines) <= 3
    assert "do not present proposals as commitments" in first.response_hazards


def test_autonomy_summary_empty_state_safe_defaults() -> None:
    summary = summarize_autonomy_state(None)
    assert summary.raw_state_present is False
    assert summary.proposal_headlines == []
    assert summary.response_hazards == []
