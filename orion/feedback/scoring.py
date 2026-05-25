from __future__ import annotations

from orion.feedback.policy import FeedbackScoringV1
from orion.schemas.feedback_frame import OutcomeObservationV1


def score_for_outcome_status(status: str, scoring: FeedbackScoringV1) -> float:
    mapping = {
        "dry_run_only": scoring.dry_run_score,
        "prepared_only": scoring.prepared_score,
        "completed": scoring.completed_score,
        "blocked": scoring.blocked_score,
        "deferred": scoring.deferred_score,
        "failed": scoring.failed_score,
        "absent": scoring.absent_score,
        "mixed": scoring.unknown_score,
        "unknown": scoring.unknown_score,
    }
    return float(mapping.get(status, scoring.unknown_score))


def aggregate_confidence(observations: list[OutcomeObservationV1]) -> float:
    if not observations:
        return 0.0
    return sum(o.confidence for o in observations) / len(observations)
