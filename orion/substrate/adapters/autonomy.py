from __future__ import annotations

from typing import Iterable

from orion.core.schemas.cognitive_substrate import (
    DriveNodeV1,
    GoalNodeV1,
    NodeRefV1,
    StateSnapshotNodeV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    TensionNodeV1,
)
from orion.core.schemas.drives import DriveAuditV1, DriveStateV1, GoalProposalV1, IdentitySnapshotV1, TensionEventV1

from ._common import make_provenance, make_temporal


def map_autonomy_artifacts_to_substrate(
    *,
    drive_audit: DriveAuditV1 | None = None,
    drive_state: DriveStateV1 | None = None,
    goals: Iterable[GoalProposalV1] = (),
    tensions: Iterable[TensionEventV1] = (),
    identity_snapshot: IdentitySnapshotV1 | None = None,
    anchor_scope: str = "orion",
    subject_ref: str | None = None,
) -> SubstrateGraphRecordV1:
    nodes = []
    edges = []
    goal_items = list(goals)
    tension_items = list(tensions)

    resolved_subject_ref = subject_ref or (drive_audit.subject if drive_audit else None) or (drive_state.subject if drive_state else None)
    observed_at = (drive_state.updated_at if drive_state else None) or (drive_audit.ts if drive_audit else None)

    if drive_state:
        snapshot_id = f"sub-state-autonomy-{drive_state.artifact_id}"
        nodes.append(
            StateSnapshotNodeV1(
                node_id=snapshot_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=drive_state.updated_at),
                provenance=make_provenance(
                    source_kind="autonomy.drive_state",
                    source_channel=drive_state.provenance.intake_channel,
                    producer="autonomy_adapter",
                    correlation_id=drive_state.correlation_id,
                    trace_id=drive_state.trace_id,
                ),
                dimensions={k: float(v) for k, v in drive_state.pressures.items()},
                snapshot_source="drive_state",
                signals={"confidence": drive_state.confidence, "salience": max(drive_state.pressures.values(), default=0.0)},
                metadata={"activations": dict(drive_state.activations), "artifact_id": drive_state.artifact_id},
            )
        )

    drive_names = set((drive_audit.active_drives if drive_audit else []))
    if drive_audit:
        drive_names.update(drive_audit.drive_pressures.keys())
    if drive_state:
        drive_names.update(drive_state.pressures.keys())

    for drive_name in sorted(name for name in drive_names if name):
        drive_id = f"sub-drive-{drive_name}"
        nodes.append(
            DriveNodeV1(
                node_id=drive_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=observed_at),
                provenance=make_provenance(
                    source_kind="autonomy.drive",
                    source_channel=(drive_audit.provenance.intake_channel if drive_audit else "orion:autonomy"),
                    producer="autonomy_adapter",
                    correlation_id=(drive_audit.correlation_id if drive_audit else None),
                    trace_id=(drive_audit.trace_id if drive_audit else None),
                ),
                drive_kind=drive_name,
                signals={
                    "confidence": (drive_audit.confidence if drive_audit else 0.6),
                    "salience": float((drive_audit.drive_pressures.get(drive_name, 0.0) if drive_audit else drive_state.pressures.get(drive_name, 0.0))),
                },
            )
        )

    for goal in goal_items:
        goal_id = f"sub-goal-{goal.proposal_signature}"
        nodes.append(
            GoalNodeV1(
                node_id=goal_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=goal.ts, valid_to=goal.cooldown_until),
                provenance=make_provenance(
                    source_kind="autonomy.goal_proposal",
                    source_channel=goal.provenance.intake_channel,
                    producer="autonomy_adapter",
                    correlation_id=goal.correlation_id,
                    trace_id=goal.trace_id,
                ),
                goal_text=goal.goal_statement,
                priority=goal.priority,
                signals={"confidence": goal.confidence, "salience": goal.priority},
                metadata={"proposal_signature": goal.proposal_signature, "artifact_id": goal.artifact_id},
            )
        )
        edges.append(
            SubstrateEdgeV1(
                source=NodeRefV1(node_id=f"sub-drive-{goal.drive_origin}", node_kind="drive"),
                target=NodeRefV1(node_id=goal_id, node_kind="goal"),
                predicate="seeks",
                temporal=make_temporal(observed_at=goal.ts),
                provenance=make_provenance(
                    source_kind="autonomy.goal_proposal",
                    source_channel=goal.provenance.intake_channel,
                    producer="autonomy_adapter",
                ),
                confidence=goal.confidence,
                salience=goal.priority,
            )
        )

    for tension in tension_items:
        tension_id = f"sub-tension-{tension.artifact_id}"
        nodes.append(
            TensionNodeV1(
                node_id=tension_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=tension.ts),
                provenance=make_provenance(
                    source_kind="autonomy.tension_event",
                    source_channel=tension.provenance.intake_channel,
                    producer="autonomy_adapter",
                    correlation_id=tension.correlation_id,
                    trace_id=tension.trace_id,
                ),
                tension_kind=tension.kind,
                intensity=tension.magnitude,
                signals={"confidence": tension.confidence, "salience": tension.magnitude},
                metadata={"drive_impacts": dict(tension.drive_impacts), "artifact_id": tension.artifact_id},
            )
        )
        for drive_name, impact in sorted(tension.drive_impacts.items()):
            edges.append(
                SubstrateEdgeV1(
                    source=NodeRefV1(node_id=f"sub-drive-{drive_name}", node_kind="drive"),
                    target=NodeRefV1(node_id=tension_id, node_kind="tension"),
                    predicate="activates" if impact >= 0 else "suppresses",
                    temporal=make_temporal(observed_at=tension.ts),
                    provenance=make_provenance(
                        source_kind="autonomy.tension_event",
                        source_channel=tension.provenance.intake_channel,
                        producer="autonomy_adapter",
                    ),
                    confidence=tension.confidence,
                    salience=min(1.0, abs(float(impact))),
                )
            )
        for goal in goal_items:
            if tension.kind in (goal.tension_kinds or []):
                edges.append(
                    SubstrateEdgeV1(
                        source=NodeRefV1(node_id=tension_id, node_kind="tension"),
                        target=NodeRefV1(node_id=f"sub-goal-{goal.proposal_signature}", node_kind="goal"),
                        predicate="blocks",
                        temporal=make_temporal(observed_at=tension.ts),
                        provenance=make_provenance(
                            source_kind="autonomy.tension_goal_link",
                            source_channel=tension.provenance.intake_channel,
                            producer="autonomy_adapter",
                        ),
                        confidence=tension.confidence,
                        salience=tension.magnitude,
                    )
                )

    if identity_snapshot:
        identity_state_id = f"sub-state-identity-{identity_snapshot.artifact_id}"
        nodes.append(
            StateSnapshotNodeV1(
                node_id=identity_state_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=identity_snapshot.ts),
                provenance=make_provenance(
                    source_kind="autonomy.identity_snapshot",
                    source_channel=identity_snapshot.provenance.intake_channel,
                    producer="autonomy_adapter",
                    correlation_id=identity_snapshot.correlation_id,
                    trace_id=identity_snapshot.trace_id,
                ),
                dimensions={k: float(v) for k, v in identity_snapshot.drive_pressures.items()},
                snapshot_source="identity_snapshot",
                signals={"confidence": identity_snapshot.confidence, "salience": max(identity_snapshot.drive_pressures.values(), default=0.0)},
                metadata={"anchor_strategy": identity_snapshot.anchor_strategy, "summary": identity_snapshot.summary},
            )
        )

    return SubstrateGraphRecordV1(
        graph_id=f"sub-graph-autonomy-{(drive_audit.artifact_id if drive_audit else 'snapshot')}",
        anchor_scope=anchor_scope,
        subject_ref=resolved_subject_ref,
        nodes=nodes,
        edges=edges,
        created_at=make_temporal(observed_at=observed_at).observed_at,
    )
