from __future__ import annotations

from dataclasses import dataclass

from orion.graph_cognition.evidence import SignalEvidenceBundleV1
from orion.graph_cognition.interpreters import GraphCognitionReportV1


@dataclass(frozen=True)
class MetacogPerceptionBriefV1:
    top_tensions: tuple[str, ...]
    top_stabilizers: tuple[str, ...]
    overall_priority: str
    recommended_verbs: tuple[str, ...]
    confidence: float
    degraded: bool
    supporting_evidence: SignalEvidenceBundleV1
    notes_for_router: tuple[str, ...]


def _priority(report: GraphCognitionReportV1) -> str:
    if report.goal_pressure.pressure_score >= 0.7 or report.identity_conflict.active:
        return "stabilize"
    if report.concept_drift.active:
        return "reframe"
    return "advance"


def build_metacog_perception_brief(report: GraphCognitionReportV1) -> MetacogPerceptionBriefV1:
    tensions: list[str] = []
    if report.identity_conflict.active:
        tensions.append("identity_conflict")
    if report.goal_pressure.stalled_goal_count > 0:
        tensions.append("stalled_goals")
    if report.contradiction_candidates.candidates:
        tensions.append("contradiction_cluster")

    stabilizers: list[str] = []
    if report.coherence.score >= 0.55:
        stabilizers.append("coherence")
    if report.social_continuity.continuity_score >= 0.5:
        stabilizers.append("social_continuity")
    if report.goal_pressure.pressure_score < 0.5:
        stabilizers.append("manageable_goal_pressure")

    priority = _priority(report)
    verb_map = {
        "stabilize": ("deconflict", "prioritize", "repair"),
        "reframe": ("clarify", "re-anchor", "probe"),
        "advance": ("execute", "consolidate", "commit"),
    }

    confidence = max(
        0.0,
        min(
            1.0,
            (
                report.coherence.confidence
                + report.identity_conflict.confidence
                + report.goal_pressure.confidence
                + report.social_continuity.confidence
                + report.concept_drift.confidence
                + report.contradiction_candidates.confidence
            )
            / 6.0,
        ),
    )
    degraded = report.social_continuity.degraded
    notes = []
    if degraded:
        notes.append("social_view_degraded")
    if priority == "stabilize":
        notes.append("route_to_stabilization_first")

    return MetacogPerceptionBriefV1(
        top_tensions=tuple(tensions[:3]),
        top_stabilizers=tuple(stabilizers[:3]),
        overall_priority=priority,
        recommended_verbs=verb_map[priority],
        confidence=confidence,
        degraded=degraded,
        supporting_evidence=report.coherence.evidence,
        notes_for_router=tuple(notes),
    )
