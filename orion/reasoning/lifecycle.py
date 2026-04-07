from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.reasoning_io import ReasoningArtifactV1
from orion.core.schemas.reasoning_policy import (
    EntityLifecycleEvaluationRequestV1,
    EntityLifecycleEvaluationResultV1,
)


ANCHOR_SCOPES = {"orion", "juniper", "relationship", "world", "session"}


def _ensure_tz(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def evaluate_entity_lifecycle(
    request: EntityLifecycleEvaluationRequestV1,
    *,
    artifacts: list[ReasoningArtifactV1],
) -> EntityLifecycleEvaluationResultV1:
    """Deterministic lifecycle governance for dynamic subject_ref entities/domains."""

    if not request.subject_ref:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state=request.current_state,
            lifecycle_action="none",
            reasons=["no_subject_ref"],
            confidence=0.9,
        )

    now = _ensure_tz(request.now)
    recent_cutoff = now - timedelta(days=7)
    decay_cutoff = now - timedelta(days=14)
    retire_cutoff = now - timedelta(days=30)

    matching = [a for a in artifacts if a.subject_ref == request.subject_ref]
    if not matching:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state="emerging",
            lifecycle_action="emerge",
            reasons=["first_seen_subject_ref"],
            confidence=0.7,
        )

    recent = [a for a in matching if _ensure_tz(a.observed_at) >= recent_cutoff]
    avg_salience = sum(a.salience for a in matching) / len(matching)
    latest = max(matching, key=lambda a: _ensure_tz(a.observed_at))
    latest_ts = _ensure_tz(latest.observed_at)

    if request.current_state in {"dormant", "retired"} and recent:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state="active",
            lifecycle_action="revive",
            reasons=["recent_activity_detected"],
            confidence=0.85,
        )

    if latest_ts < retire_cutoff and avg_salience < 0.2:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state="retired",
            lifecycle_action="retire",
            reasons=["stale_and_low_salience"],
            confidence=0.9,
        )

    if latest_ts < decay_cutoff and avg_salience < 0.3:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state="decaying",
            lifecycle_action="decay",
            reasons=["aging_signal"],
            confidence=0.75,
        )

    if latest_ts < recent_cutoff:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state="dormant",
            lifecycle_action="dormant",
            reasons=["no_recent_activity"],
            confidence=0.7,
        )

    if len(recent) >= 3 and avg_salience >= 0.6:
        return EntityLifecycleEvaluationResultV1(
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            prior_state=request.current_state,
            next_state="active",
            lifecycle_action="strengthen",
            reasons=["high_recency_and_salience"],
            confidence=0.8,
        )

    return EntityLifecycleEvaluationResultV1(
        anchor_scope=request.anchor_scope,
        subject_ref=request.subject_ref,
        prior_state=request.current_state,
        next_state=request.current_state or "active",
        lifecycle_action="none",
        reasons=["steady_state"],
        confidence=0.6,
    )
