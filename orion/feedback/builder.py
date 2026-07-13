from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from orion.feedback.extractors import (
    classify_pressure_deltas,
    extract_self_state_pressure_snapshot,
    normalize_cortex_result_evidence,
    pressure_delta,
)
from orion.feedback.policy import FeedbackPolicyV1
from orion.feedback.scoring import aggregate_confidence, score_for_outcome_status
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1

SourceKind = Literal[
    "dispatch_candidate",
    "policy_decision",
    "proposal_candidate",
    "cortex_result",
    "field_delta",
    "attention_delta",
    "self_state_delta",
    "absence",
    "operator_feedback",
]
OutcomeKind = Literal[
    "not_attempted",
    "dry_run",
    "prepared",
    "prepared_for_dispatch",
    "dispatched",
    "completed",
    "failed",
    "blocked",
    "deferred",
    "absent",
    "stale",
    "improved",
    "worsened",
    "unchanged",
    "unknown",
]


def stable_feedback_frame_id(*, dispatch_frame_id: str, policy_id: str) -> str:
    return f"feedback.frame:{dispatch_frame_id}:{policy_id}"


def _observation(
    *,
    observation_id: str,
    source_kind: SourceKind,
    source_id: str,
    outcome_kind: OutcomeKind,
    score: float,
    confidence: float,
    observed_at: datetime,
    reasons: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> OutcomeObservationV1:
    return OutcomeObservationV1(
        observation_id=observation_id,
        source_kind=source_kind,
        source_id=source_id,
        outcome_kind=outcome_kind,
        score=score,
        confidence=confidence,
        reasons=reasons or [],
        evidence_refs=evidence_refs or [],
        observed_at=observed_at,
    )


def _candidate_outcome_kind(candidate: ExecutionDispatchCandidateV1) -> OutcomeKind:
    if candidate.dispatch_status == "blocked":
        return "blocked"
    if candidate.dispatch_status == "prepared":
        return "prepared"
    if candidate.dispatch_status == "prepared_for_dispatch":
        return "prepared_for_dispatch"
    if candidate.dispatch_status == "dry_run":
        return "dry_run"
    if candidate.dispatch_status == "dispatched":
        return "dispatched"
    return "unknown"


def _policy_decision_outcome(decision: PolicyDecisionV1) -> OutcomeKind:
    if decision.decision == "deferred":
        return "deferred"
    if decision.decision == "rejected":
        return "blocked"
    return "not_attempted"


def _cortex_status_to_outcome(status: str) -> OutcomeKind:
    s = status.lower()
    if s in ("success", "ok", "completed"):
        return "completed"
    if s in ("failed", "error"):
        return "failed"
    return "unknown"


def _score_for_outcome_kind(outcome_kind: str, scoring) -> float:
    mapping = {
        "dry_run": scoring.dry_run_score,
        "prepared": scoring.prepared_score,
        "prepared_for_dispatch": scoring.prepared_score,
        "completed": scoring.completed_score,
        "blocked": scoring.blocked_score,
        "deferred": scoring.deferred_score,
        "failed": scoring.failed_score,
        "absent": scoring.absent_score,
        "dispatched": scoring.prepared_score,
        "not_attempted": scoring.unknown_score,
        "unknown": scoring.unknown_score,
    }
    return float(mapping.get(outcome_kind, scoring.unknown_score))


def _aggregate_outcome_status(
    observations: list[OutcomeObservationV1],
    dispatch: ExecutionDispatchFrameV1,
) -> str:
    kinds = {o.outcome_kind for o in observations}
    if dispatch.dispatch_mode == "dry_run" and not dispatch.dispatch_attempted:
        return "dry_run_only"
    if dispatch.dispatch_mode == "prepare_only" and not dispatch.dispatch_attempted:
        return "prepared_only"
    if "absent" in kinds:
        if "completed" in kinds or "failed" in kinds:
            return "mixed"
        return "absent"
    if "completed" in kinds and "failed" not in kinds:
        return "completed"
    if "failed" in kinds and "completed" not in kinds:
        return "failed"
    if kinds <= {"blocked", "deferred"} or ("blocked" in kinds and not dispatch.dispatch_attempted):
        if "deferred" in kinds:
            return "deferred"
        return "blocked"
    if len(kinds) > 1:
        return "mixed"
    return "unknown"


def build_feedback_frame(
    *,
    dispatch_frame: ExecutionDispatchFrameV1,
    policy_frame: PolicyDecisionFrameV1 | None,
    proposal_frame: ProposalFrameV1 | None,
    self_state_before: SelfStateV1 | None,
    self_state_after: SelfStateV1 | None,
    cortex_results: list[dict[str, object]] | None,
    policy: FeedbackPolicyV1,
    now: datetime | None = None,
) -> FeedbackFrameV1:
    del proposal_frame  # reserved for future proposal-level observations
    generated_at = now or datetime.now(timezone.utc)
    scoring = policy.scoring
    observations: list[OutcomeObservationV1] = []
    positive_evidence: list[str] = []
    negative_evidence: list[str] = []
    absence_evidence: list[str] = []
    warnings: list[str] = list(dispatch_frame.warnings)

    channels = list(policy.pressure_channels)
    if "coherence" not in channels:
        channels.append("coherence")

    pressure_before = extract_self_state_pressure_snapshot(self_state_before, channels)
    pressure_after = extract_self_state_pressure_snapshot(self_state_after, channels)
    delta = pressure_delta(pressure_before, pressure_after)
    pos_delta, neg_delta = classify_pressure_deltas(delta, policy.positive_delta_channels)
    positive_evidence.extend(pos_delta)
    negative_evidence.extend(neg_delta)

    for candidate in (
        list(dispatch_frame.candidates)
        + list(dispatch_frame.blocked_candidates)
        + list(dispatch_frame.dispatched_candidates)
    ):
        kind = _candidate_outcome_kind(candidate)
        observations.append(
            _observation(
                observation_id=f"obs:dispatch:{candidate.dispatch_id}:{kind}",
                source_kind="dispatch_candidate",
                source_id=candidate.dispatch_id,
                outcome_kind=kind,
                score=_score_for_outcome_kind(kind, scoring),
                confidence=candidate.confidence_score,
                observed_at=generated_at,
                reasons=list(candidate.reasons),
                evidence_refs=list(candidate.evidence_refs),
            )
        )

    if policy_frame is not None:
        for decision in policy_frame.decisions:
            outcome = _policy_decision_outcome(decision)
            observations.append(
                _observation(
                    observation_id=f"obs:policy:{decision.decision_id}:{outcome}",
                    source_kind="policy_decision",
                    source_id=decision.decision_id,
                    outcome_kind=outcome,
                    score=_score_for_outcome_kind(outcome, scoring),
                    confidence=decision.confidence_score,
                    observed_at=generated_at,
                    reasons=list(decision.reasons),
                    evidence_refs=list(decision.evidence_refs),
                )
            )

    normalized_results = [normalize_cortex_result_evidence(r) for r in (cortex_results or [])]
    dispatched_ids = {c.dispatch_id for c in dispatch_frame.dispatched_candidates}
    matched: set[str] = set()

    for raw in normalized_results:
        status = str(raw.get("status", "unknown"))
        outcome = _cortex_status_to_outcome(status)
        dispatch_id = str(raw.get("dispatch_id") or "")
        if dispatch_id:
            matched.add(dispatch_id)
        observations.append(
            _observation(
                observation_id=f"obs:cortex:{raw.get('result_id')}:{outcome}",
                source_kind="cortex_result",
                source_id=str(raw.get("result_id")),
                outcome_kind=outcome,
                score=_score_for_outcome_kind(outcome, scoring),
                confidence=0.85,
                observed_at=generated_at,
                evidence_refs=list(raw.get("evidence_refs") or []),
                reasons=[f"cortex_status:{status}"],
            )
        )

    needs_result = (
        dispatch_frame.dispatch_attempted
        and dispatch_frame.dispatch_mode == "dispatch_read_only"
        and bool(policy.absence_rules.get("dispatch_read_only_requires_result"))
    )
    if needs_result:
        for dispatch_id in dispatched_ids - matched:
            absence_evidence.append(f"missing_cortex_result:{dispatch_id}")
            observations.append(
                _observation(
                    observation_id=f"obs:absence:cortex:{dispatch_id}",
                    source_kind="absence",
                    source_id=dispatch_id,
                    outcome_kind="absent",
                    score=scoring.absent_score,
                    confidence=0.8,
                    observed_at=generated_at,
                    reasons=["expected_cortex_result_not_observed"],
                )
            )

    if self_state_before is not None and self_state_after is not None:
        readiness_delta = pressure_after.get("agency_readiness", 0.0) - pressure_before.get(
            "agency_readiness", 0.0
        )
        if readiness_delta > 0.05:
            observations.append(
                _observation(
                    observation_id=f"obs:self_state:{self_state_after.self_state_id}:improved",
                    source_kind="self_state_delta",
                    source_id=self_state_after.self_state_id,
                    outcome_kind="improved",
                    score=scoring.completed_score,
                    confidence=self_state_after.overall_confidence,
                    observed_at=generated_at,
                    reasons=["agency_readiness_increased"],
                )
            )
        elif readiness_delta < -0.05:
            observations.append(
                _observation(
                    observation_id=f"obs:self_state:{self_state_after.self_state_id}:worsened",
                    source_kind="self_state_delta",
                    source_id=self_state_after.self_state_id,
                    outcome_kind="worsened",
                    score=scoring.failed_score,
                    confidence=self_state_after.overall_confidence,
                    observed_at=generated_at,
                    reasons=["agency_readiness_decreased"],
                )
            )
        else:
            observations.append(
                _observation(
                    observation_id=f"obs:self_state:{self_state_after.self_state_id}:unchanged",
                    source_kind="self_state_delta",
                    source_id=self_state_after.self_state_id,
                    outcome_kind="unchanged",
                    score=scoring.unknown_score,
                    confidence=self_state_after.overall_confidence,
                    observed_at=generated_at,
                )
            )

    outcome_status = _aggregate_outcome_status(observations, dispatch_frame)
    outcome_score = score_for_outcome_status(outcome_status, scoring)
    if observations:
        confidence_score = aggregate_confidence(observations)
    elif dispatch_frame.candidates:
        confidence_score = dispatch_frame.candidates[0].confidence_score
    else:
        confidence_score = 0.5

    return FeedbackFrameV1(
        frame_id=stable_feedback_frame_id(
            dispatch_frame_id=dispatch_frame.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_execution_dispatch_frame_id=dispatch_frame.frame_id,
        source_policy_frame_id=dispatch_frame.source_policy_frame_id,
        source_proposal_frame_id=dispatch_frame.source_proposal_frame_id,
        source_self_state_id=dispatch_frame.source_self_state_id,
        feedback_policy_id=policy.policy_id,
        outcome_status=outcome_status,
        outcome_score=outcome_score,
        confidence_score=confidence_score,
        observations=observations,
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
        absence_evidence=absence_evidence,
        pressure_before=pressure_before,
        pressure_after=pressure_after,
        pressure_delta=delta,
        warnings=warnings,
    )
