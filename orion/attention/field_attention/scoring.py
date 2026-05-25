from __future__ import annotations

from orion.attention.field_attention.policy import AttentionWeightsV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1

URGENCY_CHANNELS = frozenset({
    "failure_pressure",
    "reliability_pressure",
    "thermal_pressure",
    "staleness",
    "execution_friction",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def weighted_pressure(
    vector: dict[str, float],
    channel_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    raw = 0.0
    dominant: dict[str, float] = {}
    for channel, value in vector.items():
        weight = float(channel_weights.get(channel, 0.0))
        if weight == 0.0:
            continue
        contrib = float(value) * weight
        raw += contrib
        if contrib > 0.0:
            dominant[channel] = contrib
    dominant = dict(sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)[:5])
    return clamp01(raw), dominant


def urgency_score(
    vector: dict[str, float],
    channel_weights: dict[str, float],
) -> float:
    urgent = 0.0
    for channel in URGENCY_CHANNELS:
        if channel not in vector:
            continue
        weight = float(channel_weights.get(channel, 0.0))
        if weight <= 0.0:
            continue
        urgent = max(urgent, float(vector[channel]) * weight)
    return clamp01(urgent)


def confidence_from_vector(
    vector: dict[str, float],
    channel_weights: dict[str, float],
) -> float:
    healthy = 0.0
    for channel, value in vector.items():
        weight = float(channel_weights.get(channel, 0.0))
        if weight < 0.0:
            healthy += float(value) * abs(weight)
    return clamp01(healthy)


def prior_salience_for_target(
    target_id: str,
    previous_frame: FieldAttentionFrameV1 | None,
) -> float:
    if previous_frame is None:
        return 0.0
    for bucket in (
        previous_frame.dominant_targets,
        previous_frame.node_targets,
        previous_frame.capability_targets,
        previous_frame.system_targets,
        previous_frame.suppressed_targets,
    ):
        for t in bucket:
            if t.target_id == target_id:
                return t.salience_score
    return 0.0


def novelty_for_target(
    target_id: str,
    current_salience: float,
    previous_frame: FieldAttentionFrameV1 | None,
) -> float:
    if previous_frame is None:
        return 0.0
    prior = prior_salience_for_target(target_id, previous_frame)
    return clamp01(abs(current_salience - prior))


def compute_salience(
    *,
    pressure_score: float,
    novelty_score: float,
    urgency_score: float,
    confidence_score: float,
    weights: AttentionWeightsV1,
) -> float:
    raw = (
        weights.pressure * pressure_score
        + weights.novelty * novelty_score
        + weights.urgency * urgency_score
        + weights.confidence * confidence_score
    )
    return clamp01(raw)
