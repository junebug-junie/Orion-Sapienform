from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.self_state import SelfStateV1
from orion.schemas.self_state_prediction import SelfStatePredictionV1


def build_next_cycle_prediction(
    self_state: SelfStateV1,
    *,
    now: datetime | None = None,
) -> SelfStatePredictionV1:
    """Linear one-step extrapolation from current scores + trajectory deltas."""
    generated_at = now or datetime.now(timezone.utc)
    predicted: dict[str, float] = {}
    for dim_id, dim in self_state.dimensions.items():
        delta = self_state.dimension_trajectory.get(dim_id, 0.0)
        predicted[dim_id] = max(0.0, min(1.0, dim.score + delta))
    return SelfStatePredictionV1(
        prediction_id=f"pred:{self_state.self_state_id}",
        generated_at=generated_at,
        source_self_state_id=self_state.self_state_id,
        predicted_dimension_scores=predicted,
    )


def compute_prediction_errors(
    actual: SelfStateV1,
    prediction: SelfStatePredictionV1,
) -> dict[str, float]:
    """Per-dimension absolute error between predicted and actual scores."""
    errors: dict[str, float] = {}
    for dim_id, predicted_score in prediction.predicted_dimension_scores.items():
        actual_dim = actual.dimensions.get(dim_id)
        if actual_dim is None:
            continue
        err = abs(actual_dim.score - predicted_score)
        if err >= 0.01:
            errors[dim_id] = round(err, 4)
    return errors
