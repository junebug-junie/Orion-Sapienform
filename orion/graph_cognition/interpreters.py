from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1
from orion.graph_cognition.evidence import EvidenceSpanV1, SignalEvidenceBundleV1
from orion.graph_cognition.features import GraphFeatureSetV1
from orion.graph_cognition.views import GraphViewBundleV1
from orion.substrate.store import MaterializedSubstrateGraphState


@dataclass(frozen=True)
class CoherenceAssessmentV1:
    score: float
    confidence: float
    evidence: SignalEvidenceBundleV1
    notes: tuple[str, ...]


@dataclass(frozen=True)
class IdentityConflictSignalV1:
    conflict_score: float
    active: bool
    confidence: float
    evidence: SignalEvidenceBundleV1


@dataclass(frozen=True)
class GoalPressureStateV1:
    pressure_score: float
    stalled_goal_count: int
    competing_goal_density: float
    confidence: float
    evidence: SignalEvidenceBundleV1


@dataclass(frozen=True)
class SocialContinuityAssessmentV1:
    continuity_score: float
    confidence: float
    degraded: bool
    evidence: SignalEvidenceBundleV1


@dataclass(frozen=True)
class ConceptDriftSignalV1:
    drift_score: float
    active: bool
    confidence: float
    evidence: SignalEvidenceBundleV1


@dataclass(frozen=True)
class ContradictionCandidateV1:
    node_id: str
    severity: float
    pressure: float


@dataclass(frozen=True)
class ContradictionCandidateSetV1:
    candidates: tuple[ContradictionCandidateV1, ...]
    confidence: float
    evidence: SignalEvidenceBundleV1


@dataclass(frozen=True)
class GraphCognitionReportV1:
    coherence: CoherenceAssessmentV1
    identity_conflict: IdentityConflictSignalV1
    goal_pressure: GoalPressureStateV1
    social_continuity: SocialContinuityAssessmentV1
    concept_drift: ConceptDriftSignalV1
    contradiction_candidates: ContradictionCandidateSetV1


def _bundle(*, node_ids: list[str], reason: str, degraded: bool = False, truncated: bool = False) -> SignalEvidenceBundleV1:
    return SignalEvidenceBundleV1(
        spans=(EvidenceSpanV1(node_ids=tuple(node_ids[:12]), edge_ids=(), reason=reason, weight=1.0),),
        truncated=truncated or len(node_ids) > 12,
        degraded=degraded,
        notes=(() if not degraded else ("degraded_input",)),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def interpret_graph_cognition(
    *,
    state: MaterializedSubstrateGraphState,
    views: GraphViewBundleV1,
    features: GraphFeatureSetV1,
) -> GraphCognitionReportV1:
    coherence_score = _clamp(
        (features.dynamic.get("coherence_trend", 0.0) * 0.4)
        + ((1.0 - min(1.0, features.structural.get("fragmentation", 0.0) / 5.0)) * 0.3)
        + ((1.0 - min(1.0, features.semantic.get("contradiction_density", 0.0))) * 0.3)
    )
    coherence = CoherenceAssessmentV1(
        score=coherence_score,
        confidence=_clamp(0.6 + (0.3 * (1.0 - float(features.degraded)))),
        evidence=_bundle(node_ids=list(views.semantic.node_ids), reason="coherence_from_structure_semantics", degraded=features.degraded, truncated=views.semantic.truncated),
        notes=features.notes,
    )

    identity_conflict_score = _clamp(
        (features.semantic.get("contradiction_density", 0.0) * 0.5)
        + (features.dynamic.get("tension_accumulation", 0.0) * 0.2)
        + (features.dynamic.get("mean_pressure", 0.0) * 0.3)
    )
    identity_conflict = IdentityConflictSignalV1(
        conflict_score=identity_conflict_score,
        active=identity_conflict_score >= 0.35,
        confidence=_clamp(0.55 + (0.35 * (1.0 - float(features.degraded)))),
        evidence=_bundle(node_ids=list(views.self_view.node_ids), reason="self_conflict_from_contradictions", degraded=features.degraded, truncated=views.self_view.truncated),
    )

    goal_pressure_score = _clamp(
        (features.dynamic.get("mean_pressure", 0.0) * 0.5)
        + (min(1.0, features.social_executive.get("stalled_goals", 0.0) / 3.0) * 0.3)
        + (features.social_executive.get("goal_competition_density", 0.0) * 0.2)
    )
    goal_pressure = GoalPressureStateV1(
        pressure_score=goal_pressure_score,
        stalled_goal_count=int(features.social_executive.get("stalled_goals", 0.0)),
        competing_goal_density=features.social_executive.get("goal_competition_density", 0.0),
        confidence=_clamp(0.6 + (0.25 * (1.0 - float(features.degraded)))),
        evidence=_bundle(node_ids=list(views.executive.node_ids), reason="goal_pressure_from_dynamic_and_goal_topology", degraded=features.degraded, truncated=views.executive.truncated),
    )

    continuity_score = _clamp(
        (features.social_executive.get("reciprocity_balance", 0.0) * 0.5)
        + ((1.0 - min(1.0, features.temporal.get("inactivity_duration_seconds", 0.0) / 172800.0)) * 0.5)
    )
    social_continuity = SocialContinuityAssessmentV1(
        continuity_score=continuity_score,
        confidence=_clamp(0.5 + (0.35 * (1.0 - float(features.degraded)))),
        degraded=features.degraded,
        evidence=_bundle(node_ids=list(views.social.node_ids), reason="social_continuity_from_reciprocity_and_freshness", degraded=features.degraded, truncated=views.social.truncated),
    )

    drift_score = _clamp(
        (features.temporal.get("recent_change_density", 0.0) * 0.35)
        + (features.semantic.get("contradiction_density", 0.0) * 0.4)
        + (min(1.0, features.temporal.get("node_churn", 0.0) / 20.0) * 0.25)
    )
    concept_drift = ConceptDriftSignalV1(
        drift_score=drift_score,
        active=drift_score >= 0.4,
        confidence=_clamp(0.6 + (0.25 * (1.0 - float(features.degraded)))),
        evidence=_bundle(node_ids=list(views.concept.node_ids), reason="concept_drift_from_temporal_and_semantic_motion", degraded=features.degraded, truncated=views.concept.truncated),
    )

    contradiction_nodes: list[BaseSubstrateNodeV1] = [
        state.nodes[node_id]
        for node_id in views.contradiction.node_ids
        if node_id in state.nodes and state.nodes[node_id].node_kind == "contradiction" and not bool(state.nodes[node_id].metadata.get("resolved", False))
    ]
    ranked = sorted(
        contradiction_nodes,
        key=lambda n: (float(n.metadata.get("severity") or 0.5), float(n.metadata.get("dynamic_pressure") or 0.0)),
        reverse=True,
    )
    candidates = tuple(
        ContradictionCandidateV1(
            node_id=node.node_id,
            severity=float(node.metadata.get("severity") or 0.5),
            pressure=float(node.metadata.get("dynamic_pressure") or 0.0),
        )
        for node in ranked[:8]
    )
    contradiction_candidates = ContradictionCandidateSetV1(
        candidates=candidates,
        confidence=_clamp(0.55 + (0.35 * (1.0 - float(features.degraded)))),
        evidence=_bundle(node_ids=[c.node_id for c in candidates], reason="explicit_contradiction_nodes", degraded=features.degraded, truncated=len(ranked) > 8),
    )

    return GraphCognitionReportV1(
        coherence=coherence,
        identity_conflict=identity_conflict,
        goal_pressure=goal_pressure,
        social_continuity=social_continuity,
        concept_drift=concept_drift,
        contradiction_candidates=contradiction_candidates,
    )
