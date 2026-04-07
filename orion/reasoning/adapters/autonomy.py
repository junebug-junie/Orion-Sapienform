from __future__ import annotations

from orion.autonomy.models import AutonomyStateV1
from orion.core.schemas.reasoning import ClaimV1, ReasoningProvenanceV1


def map_autonomy_state_to_reasoning(state: AutonomyStateV1, *, producer: str = "autonomy_adapter") -> list[ClaimV1]:
    """Map autonomy state into conservative reasoning claims (proposal/provisional only)."""

    artifacts: list[ClaimV1] = []

    artifacts.append(
        ClaimV1(
            anchor_scope=state.subject,
            subject_ref=state.entity_id,
            status="provisional",
            authority="local_inferred",
            confidence=0.7,
            salience=0.7,
            novelty=0.2,
            risk_tier="medium",
            observed_at=state.generated_at,
            provenance=ReasoningProvenanceV1(
                evidence_refs=[state.latest_drive_audit_id or "", state.latest_identity_snapshot_id or ""],
                source_channel="orion:autonomy",
                source_kind="AutonomyStateV1",
                producer=producer,
            ),
            claim_text=(state.identity_summary or "Autonomy state observed without identity summary."),
            claim_kind="autonomy_state_summary",
            qualifiers={
                "model_layer": state.model_layer,
                "dominant_drive": state.dominant_drive,
                "active_drives": state.active_drives,
                "tension_kinds": state.tension_kinds,
                "drive_pressures": state.drive_pressures,
            },
        )
    )

    for goal in state.goal_headlines:
        artifacts.append(
            ClaimV1(
                anchor_scope=state.subject,
                subject_ref=f"goal:{goal.proposal_signature}",
                status="proposed",
                authority="local_inferred",
                confidence=max(0.2, min(goal.priority, 1.0)),
                salience=goal.priority,
                novelty=0.4,
                risk_tier="medium",
                observed_at=state.generated_at,
                provenance=ReasoningProvenanceV1(
                    evidence_refs=[goal.artifact_id],
                    source_channel="orion:autonomy",
                    source_kind="AutonomyGoalHeadlineV1",
                    producer=producer,
                ),
                claim_text=goal.goal_statement,
                claim_kind="goal_proposal_headline",
                qualifiers={
                    "drive_origin": goal.drive_origin,
                    "priority": goal.priority,
                    "cooldown_until": goal.cooldown_until.isoformat() if goal.cooldown_until else None,
                },
            )
        )

    return artifacts
