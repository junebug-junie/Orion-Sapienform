from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.drives import GoalProposalV1
from orion.spark.concept_induction.goals import GoalProposalEngine


def _minimal_goal(**overrides):
    base = dict(
        artifact_id="goal-abc",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.goals.proposed.v1",
        ts=datetime(2026, 5, 21, tzinfo=timezone.utc),
        confidence=0.7,
        provenance={
            "intake_channel": "orion:metacognition:tick",
            "correlation_id": "c1",
            "trace_id": "trace-long-id-12345",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
        goal_statement="Clarify autonomy boundaries without executing any new action.",
        proposal_signature="sig",
        drive_origin="autonomy",
        priority=0.5,
    )
    base.update(overrides)
    return GoalProposalV1.model_validate(base)


def test_goal_proposal_v1_accepts_new_optional_fields():
    goal = _minimal_goal(
        goal_statement_base="Clarify autonomy boundaries without executing any new action.",
        proposal_status="proposed",
        semantic_source="template",
    )
    assert goal.goal_statement_base.startswith("Clarify autonomy")
    assert goal.proposal_status == "proposed"
    assert goal.semantic_source == "template"
