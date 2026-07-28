from __future__ import annotations

from typing import Literal

from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import (
    compute_salience,
    confidence_from_vector,
    novelty_for_target,
    urgency_score,
    weighted_pressure,
)
from orion.field.pressure import (
    RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    RECENT_PERTURBATION_ZSCORE_SATURATION,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1

ObservationMode = Literal["watch", "inspect", "summarize", "ignore"]


def observation_mode_for(salience: float, policy: FieldAttentionPolicyV1) -> ObservationMode:
    modes = policy.observation_modes
    if salience >= modes.inspect_threshold:
        return "inspect"
    if salience >= modes.summarize_threshold:
        return "summarize"
    if salience >= modes.watch_threshold:
        return "watch"
    return "ignore"


def _reasons_from_dominant(dominant: dict[str, float], prefix: str) -> list[str]:
    reasons: list[str] = []
    for channel, contrib in sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        if contrib > 0.05:
            reasons.append(f"{prefix} {channel} is elevated")
    return reasons or [f"{prefix} pressure present"]


def _build_target(
    *,
    target_id: str,
    target_kind: Literal["node", "capability"],
    vector: dict[str, float],
    channel_weights: dict[str, float],
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
    field: FieldStateV1,
    reason_prefix: str,
) -> FieldAttentionTargetV1 | None:
    pressure, dominant = weighted_pressure(vector, channel_weights)
    if pressure <= 0.0 and not any(float(v) > 0 for v in vector.values()):
        return None
    urg = urgency_score(vector, channel_weights)
    conf = confidence_from_vector(vector, channel_weights)
    pre_novelty_salience = compute_salience(
        pressure_score=pressure,
        novelty_score=0.0,
        urgency_score=urg,
        confidence_score=conf,
        weights=policy.weights,
    )
    novelty = novelty_for_target(target_id, pre_novelty_salience, previous_frame)
    salience = compute_salience(
        pressure_score=pressure,
        novelty_score=novelty,
        urgency_score=urg,
        confidence_score=conf,
        weights=policy.weights,
    )
    return FieldAttentionTargetV1(
        target_id=target_id,
        target_kind=target_kind,
        salience_score=salience,
        pressure_score=pressure,
        novelty_score=novelty,
        urgency_score=urg,
        confidence_score=conf,
        dominant_channels=dominant,
        reasons=_reasons_from_dominant(dominant, reason_prefix),
        evidence_refs=[f"field:{field.tick_id}"],
        suggested_observation_mode=observation_mode_for(salience, policy),
    )


def select_node_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
) -> list[FieldAttentionTargetV1]:
    targets: list[FieldAttentionTargetV1] = []
    for node_id, vector in field.node_vectors.items():
        t = _build_target(
            target_id=node_id,
            target_kind="node",
            vector=vector,
            channel_weights=policy.node_channel_weights,
            policy=policy,
            previous_frame=previous_frame,
            field=field,
            reason_prefix="node",
        )
        if t is not None:
            targets.append(t)
    return targets


def select_capability_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
) -> list[FieldAttentionTargetV1]:
    targets: list[FieldAttentionTargetV1] = []
    for cap_id, vector in field.capability_vectors.items():
        t = _build_target(
            target_id=cap_id,
            target_kind="capability",
            vector=vector,
            channel_weights=policy.capability_channel_weights,
            policy=policy,
            previous_frame=previous_frame,
            field=field,
            reason_prefix="capability",
        )
        if t is not None:
            targets.append(t)
    return targets


def select_system_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
) -> list[FieldAttentionTargetV1]:
    """Surfaces ``field:recent_perturbations`` as a system attention target
    when recent perturbation activity is elevated relative to this mesh's
    own recent baseline.

    2026-07-28: was ``min(1.0, count / 10.0)`` -- live-confirmed permanently
    saturated at 1.0 (real steady-state count is ~100-118 in the 60s window,
    5-10x past the old cap), so this target used to report maximum salience
    on essentially every tick regardless of whether anything was actually
    unusual. Now scores ``field.recent_perturbation_zscore`` (EWMA baseline
    maintained by apply_perturbations() -- see field_state.py's
    recent_perturbation_ewma* docstring), same fix shape as
    bus_synaptic_prediction_error's calm-floor correction: reused, not
    reinvented. A below-baseline dip (negative zscore) is clamped to 0 --
    "quieter than usual" isn't a reason to attend here, only "busier than
    usual" is.
    """
    count = len(field.recent_perturbations)
    if count == 0:
        return []
    zscore = field.recent_perturbation_zscore
    if zscore is None or field.recent_perturbation_ewma_n < RECENT_PERTURBATION_EWMA_MIN_SAMPLES:
        return []
    salience = min(1.0, max(0.0, zscore) / RECENT_PERTURBATION_ZSCORE_SATURATION)
    if salience < policy.thresholds.min_salience:
        return []
    return [
        FieldAttentionTargetV1(
            target_id="field:recent_perturbations",
            target_kind="system",
            salience_score=salience,
            pressure_score=salience,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
            dominant_channels={},
            reasons=[
                f"recent field perturbation count is {count} (z={zscore:.2f} vs baseline)"
            ],
            evidence_refs=[f"field:{field.tick_id}"],
            suggested_observation_mode=observation_mode_for(salience, policy),
        )
    ]
