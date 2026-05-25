from __future__ import annotations

from datetime import datetime, timezone

from orion.consolidation.policy import ConsolidationPolicyV1, MotifRuleV1
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import MotifObservationV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.self_state import SelfStateV1


def _dim_score(state: SelfStateV1, dimension_id: str) -> float:
    dim = state.dimensions.get(dimension_id)
    return float(dim.score) if dim is not None else 0.0


def _motif_id(label: str, policy_id: str) -> str:
    return f"motif:{label}:{policy_id}"


def _score_motif(*, match_count: int, total: int, policy: ConsolidationPolicyV1) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    support = match_count / float(total)
    confidence = min(1.0, match_count / float(max(1, policy.window.min_support_count)))
    return support, confidence


def detect_motifs(
    *,
    window: ConsolidationWindowData,
    policy: ConsolidationPolicyV1,
) -> list[MotifObservationV1]:
    motifs: list[MotifObservationV1] = []
    for _key, rule in policy.motif_rules.items():
        detector = _DETECTORS.get(rule.label)
        if detector is None:
            continue
        motif = detector(window=window, rule=rule, policy=policy)
        if motif is None:
            continue
        if motif.support_score < policy.motif_thresholds.min_support_score:
            continue
        if motif.confidence_score < policy.motif_thresholds.min_confidence_score:
            continue
        motifs.append(motif)
    return motifs


def _detect_loaded_but_reliable(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    cond = rule.conditions
    matches = [
        s
        for s in window.self_states
        if s.overall_condition == cond.get("overall_condition", "loaded")
        and _dim_score(s, "execution_pressure") >= float(cond.get("execution_pressure_min", 0.7))
        and _dim_score(s, "reliability_pressure") <= float(cond.get("reliability_pressure_max", 0.3))
    ]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.self_states) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.self_state_id for m in matches],
        dominant_dimensions={
            "execution_pressure": sum(_dim_score(m, "execution_pressure") for m in matches) / len(matches),
            "reliability_pressure": sum(_dim_score(m, "reliability_pressure") for m in matches) / len(matches),
        },
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
        reasons=["loaded_with_high_execution_low_reliability_pressure"],
    )


def _target_ids(frame: FieldAttentionFrameV1) -> set[str]:
    ids: set[str] = set()
    for bucket in (
        frame.dominant_targets,
        frame.node_targets,
        frame.capability_targets,
        frame.system_targets,
    ):
        for t in bucket:
            ids.add(f"{t.target_kind}:{t.target_id}" if ":" not in t.target_id else t.target_id)
            if t.target_kind == "node":
                ids.add(f"node:{t.target_id}")
            if t.target_kind == "capability":
                ids.add(f"capability:{t.target_id}")
    return ids


def _detect_attention_saturated_execution(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    cond = rule.conditions
    targets_any = set(cond.get("attention_target_any", []))
    min_salience = float(cond.get("min_overall_salience", 0.7))
    matches = [
        f
        for f in window.attention_frames
        if f.overall_salience >= min_salience and _target_ids(f).intersection(targets_any)
    ]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.attention_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=["attention_salience_saturated_on_execution_targets"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


def _detect_read_only_policy_loop(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    min_count = int(rule.conditions.get("approved_read_only_min", 1))
    matches = [
        p
        for p in window.policy_frames
        if not p.execution_allowed
        and sum(1 for d in p.decisions if d.decision == "approved_read_only") >= min_count
    ]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.policy_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=["read_only_policy_decisions_without_execution"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


def _detect_dry_run_feedback_loop(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    expected = rule.conditions.get("outcome_status", "dry_run_only")
    matches = [f for f in window.feedback_frames if f.outcome_status == expected]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.feedback_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=[f"feedback_outcome_status:{expected}"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


def _detect_blocked_review_loop(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    matches: list[str] = []
    for p in window.policy_frames:
        if p.operator_review_required or p.review_required_decisions:
            matches.append(p.frame_id)
    for d in window.dispatch_frames:
        if d.blocked_candidates:
            matches.append(d.frame_id)
    for f in window.feedback_frames:
        if f.outcome_status == "blocked":
            matches.append(f.frame_id)
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(match_count=len(matches), total=max(len(matches), 1), policy=policy)
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=matches,
        reasons=["blocked_or_operator_review_signals"],
    )


def _detect_stable_after_dry_run(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    allowed = set(rule.conditions.get("self_state_delta", {}).get("allowed", ["unchanged"]))
    matches = []
    for f in window.feedback_frames:
        if f.outcome_status != rule.conditions.get("outcome_status", "dry_run_only"):
            continue
        if f.absence_evidence or f.negative_evidence:
            continue
        has_unchanged = any(
            o.source_kind == "self_state_delta" and o.outcome_kind in allowed for o in f.observations
        )
        if has_unchanged:
            matches.append(f)
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.feedback_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=["dry_run_with_unchanged_self_state"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


_DETECTORS = {
    "loaded_but_reliable": _detect_loaded_but_reliable,
    "attention_saturated_execution": _detect_attention_saturated_execution,
    "read_only_policy_loop": _detect_read_only_policy_loop,
    "dry_run_feedback_loop": _detect_dry_run_feedback_loop,
    "blocked_review_loop": _detect_blocked_review_loop,
    "stable_after_dry_run": _detect_stable_after_dry_run,
}
