from orion.feedback.policy import FeedbackScoringV1
from orion.feedback.scoring import aggregate_confidence, score_for_outcome_status


def test_score_for_dry_run() -> None:
    scoring = FeedbackScoringV1()
    assert score_for_outcome_status("dry_run_only", scoring) == 0.50


def test_score_for_completed() -> None:
    scoring = FeedbackScoringV1()
    assert score_for_outcome_status("completed", scoring) == 0.85


def test_aggregate_confidence_empty() -> None:
    assert aggregate_confidence([]) == 0.0


def test_aggregate_confidence_mean() -> None:
    from datetime import datetime, timezone

    from orion.schemas.feedback_frame import OutcomeObservationV1

    now = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    obs = [
        OutcomeObservationV1(
            observation_id="o1",
            source_kind="dispatch_candidate",
            source_id="s1",
            outcome_kind="dry_run",
            score=0.5,
            confidence=0.8,
            observed_at=now,
        ),
        OutcomeObservationV1(
            observation_id="o2",
            source_kind="dispatch_candidate",
            source_id="s2",
            outcome_kind="dry_run",
            score=0.5,
            confidence=0.6,
            observed_at=now,
        ),
    ]
    assert abs(aggregate_confidence(obs) - 0.7) < 1e-6
