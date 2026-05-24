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
        availability="degraded",
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
            availability="degraded",
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
        availability="degraded",
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
        availability="degraded",
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


def test_deferred_drives_degraded_reason_says_deferred_not_failed() -> None:
    """'deferred' drives status should read 'drives facet deferred', not 'drives facet failed (deferred)'."""
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
            "identity": {"status": "ok", "row_count": 1, "elapsed_ms": 120.0},
            "drives": {"status": "deferred", "row_count": 0, "elapsed_ms": 0.0},
            "goals": {"status": "ok", "row_count": 1, "elapsed_ms": 95.0},
        },
    )
    assert summary.state_quality == "degraded_drives_error"
    assert summary.stance_mode == "proposal_only"
    assert summary.degraded_reason == "Orion drives facet deferred"
    assert "failed" not in (summary.degraded_reason or "")


def test_healthy_state_with_proposals_uses_normal_stance_mode() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="autonomy",
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
            "identity": {"status": "ok", "row_count": 1},
            "drives": {"status": "ok", "row_count": 12},
            "goals": {"status": "ok", "row_count": 1},
        },
    )
    assert summary.state_quality == "healthy"
    assert summary.stance_mode == "normal"
    assert summary.goals_present is True
    assert len(summary.active_goals) == 1
    assert summary.active_goals[0].headline.startswith("Clarify autonomy")
