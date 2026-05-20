from __future__ import annotations

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1
from orion.autonomy.repository import AutonomyLookupV1
from orion.autonomy.summary import summarize_autonomy_lookup, summarize_autonomy_state


def test_drives_timeout_partial_state_marks_degraded() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        goal_headlines=[
            AutonomyGoalHeadlineV1(
                artifact_id="goal-1",
                goal_statement="Clarify autonomy boundaries without executing any new action.",
                drive_origin="autonomy",
                priority=0.8,
                cooldown_until=None,
                proposal_signature="sig-1",
            )
        ],
        source="graph",
    )
    summary = summarize_autonomy_lookup(
        state,
        selected_subject="orion",
        availability="available",
        subquery_diagnostics={
            "identity": {"status": "ok", "row_count": 1, "elapsed_ms": 3720.0},
            "drives": {"status": "timeout", "row_count": 0, "elapsed_ms": 20000.0, "error_type": "timeout"},
            "goals": {"status": "ok", "row_count": 3, "elapsed_ms": 27465.0},
        },
    )
    assert summary.state_quality == "degraded_drives_timeout"
    assert summary.stance_mode == "proposal_only"
    assert summary.degraded_reason == "Orion drives facet timed out"
    assert summary.dominant_drive is None
    assert summary.proposal_headlines == ["Clarify autonomy boundaries without executing any new action."]
    assert summary.facet_health["drives"] == "timeout"


def test_healthy_state_keeps_drive_fields() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        dominant_drive="coherence",
        drive_pressures={"coherence": 0.9},
        active_drives=["coherence"],
        source="graph",
    )
    summary = summarize_autonomy_lookup(
        state,
        selected_subject="orion",
        availability="available",
        subquery_diagnostics={
            "identity": {"status": "ok", "row_count": 1},
            "drives": {"status": "ok", "row_count": 12},
            "goals": {"status": "empty", "row_count": 0},
        },
    )
    assert summary.state_quality == "healthy"
    assert summary.dominant_drive == "coherence"
    assert summary.stance_mode == "normal"


def test_relationship_context_note_without_substitution() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        source="graph",
    )
    by_subject = {
        "orion": AutonomyLookupV1(
            subject="orion",
            state=state,
            availability="available",
            unavailable_reason="timeout",
            subquery_diagnostics={"drives": {"status": "timeout", "row_count": 0}},
        ),
        "relationship": AutonomyLookupV1(
            subject="relationship",
            state=None,
            availability="available",
            subquery_diagnostics={"drives": {"status": "ok", "row_count": 80}},
        ),
    }
    summary = summarize_autonomy_lookup(
        state,
        selected_subject="orion",
        availability="available",
        subquery_diagnostics=by_subject["orion"].subquery_diagnostics,
        by_subject=by_subject,
    )
    assert summary.context_note == "relationship drives are available, but were not substituted for Orion drives"
    assert summary.dominant_drive is None


def test_drives_timeout_without_proposals_is_unavailable_stance() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="orion",
        identity_summary="holds course",
        source="graph",
    )
    summary = summarize_autonomy_lookup(
        state,
        selected_subject="orion",
        availability="available",
        subquery_diagnostics={
            "identity": {"status": "ok", "row_count": 1},
            "drives": {"status": "timeout", "row_count": 0},
            "goals": {"status": "empty", "row_count": 0},
        },
    )
    assert summary.state_quality == "degraded_drives_timeout"
    assert summary.stance_mode == "unavailable"
    assert not summary.proposal_headlines

    summary = summarize_autonomy_state(None)
    assert summary.state_quality == "empty"
    assert summary.stance_mode == "unavailable"
    assert summary.raw_state_present is False
