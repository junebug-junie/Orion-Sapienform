from __future__ import annotations

from orion.schemas.field_attention_frame import FieldAttentionFrameV1


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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
