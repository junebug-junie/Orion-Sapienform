from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, cast

from orion.self_state.policy import SelfStatePolicyV1
from orion.self_state.transport import (
    transport_channel_hints,
    transport_integrity_score,
    transport_summary_labels,
)
from orion.self_state.scoring import (
    agency_readiness_score,
    clamp01,
    collect_attention_channel_pressures,
    collect_field_channel_pressures,
    condition_from_intensity,
    coherence_score,
    field_intensity_score,
    map_channels_to_dimensions,
    uncertainty_score,
    weighted_overall_intensity,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

DimensionId = Literal[
    "field_intensity",
    "coherence",
    "uncertainty",
    "agency_readiness",
    "resource_pressure",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "introspection_pressure",
    "social_pressure",
    "policy_pressure",
]

ALL_DIMENSION_IDS: tuple[DimensionId, ...] = (
    "field_intensity",
    "coherence",
    "uncertainty",
    "agency_readiness",
    "resource_pressure",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "introspection_pressure",
    "social_pressure",
    "policy_pressure",
)


def stable_self_state_id(
    *,
    source_field_tick_id: str,
    source_attention_frame_id: str,
    policy_id: str,
) -> str:
    return f"self.state:{source_field_tick_id}:{source_attention_frame_id}:{policy_id}"


def _emit_summary_labels(
    *,
    dimension_scores: dict[str, float],
    overall_condition: str,
    overall_salience: float,
) -> list[str]:
    labels: list[str] = []
    if dimension_scores.get("execution_pressure", 0.0) >= 0.5:
        labels.append("execution_loaded")
    if dimension_scores.get("resource_pressure", 0.0) >= 0.5:
        labels.append("resource_pressurized")
    if dimension_scores.get("reliability_pressure", 0.0) < 0.3:
        labels.append("reliability_clear")
    if dimension_scores.get("field_intensity", 0.0) >= 0.5:
        labels.append("field_active")
    if overall_salience >= 0.7:
        labels.append("attention_saturated")
    if overall_condition in ("loaded", "strained", "unstable"):
        labels.append("orchestration_pressurized")
    if (
        dimension_scores.get("coherence", 0.0) >= 0.8
        and dimension_scores.get("execution_pressure", 0.0) >= 0.7
    ):
        labels.append("stabilized_but_loaded")
    return sorted(set(labels))


def evidence_for_dimension(
    *,
    dim_id: str,
    merged_channels: dict[str, float],
    policy: SelfStatePolicyV1,
    limit: int = 3,
) -> list[str]:
    pairs: list[tuple[str, float]] = []
    for channel, value in merged_channels.items():
        if policy.channel_dimension_map.get(channel) == dim_id:
            pairs.append((channel, value))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return [f"{ch}={v:.2f}" for ch, v in pairs[:limit]]


def build_self_state(
    *,
    field: FieldStateV1,
    attention: FieldAttentionFrameV1,
    policy: SelfStatePolicyV1,
    previous_self_state: SelfStateV1 | None = None,
    now: datetime | None = None,
    enable_transport_influence: bool = False,
) -> SelfStateV1:
    del previous_self_state  # reserved for continuity deltas in a later revision
    generated_at = now or datetime.now(timezone.utc)

    warnings: list[str] = []
    if attention.source_field_tick_id != field.tick_id:
        warnings.append(
            f"attention_source_tick_mismatch:{attention.source_field_tick_id}!={field.tick_id}"
        )

    field_channels = collect_field_channel_pressures(field)
    attn_channels = collect_attention_channel_pressures(attention, policy)
    merged_channels: dict[str, float] = dict(field_channels)
    for k, v in attn_channels.items():
        merged_channels[k] = max(merged_channels.get(k, 0.0), v)

    mapped = map_channels_to_dimensions(channel_pressures=merged_channels, policy=policy)

    coherence = coherence_score(channel_pressures=merged_channels, policy=policy)
    uncertainty = uncertainty_score(
        overall_salience=attention.overall_salience,
        coherence=coherence,
    )
    field_intensity = field_intensity_score(
        overall_salience=attention.overall_salience,
        recent_perturbation_saturation=merged_channels.get("recent_perturbation_count", 0.0),
    )

    execution_p = mapped.get("execution_pressure", 0.0)
    reliability_p = mapped.get("reliability_pressure", 0.0)
    resource_p = mapped.get("resource_pressure", 0.0)
    reasoning_p = mapped.get("reasoning_pressure", 0.0)
    continuity_p = mapped.get("continuity_pressure", 0.0)

    agency = agency_readiness_score(
        coherence=coherence,
        execution_pressure=execution_p,
        reliability_pressure=reliability_p,
        uncertainty=uncertainty,
        resource_pressure=resource_p,
    )

    dimension_scores: dict[str, float] = {
        "field_intensity": field_intensity,
        "coherence": coherence,
        "uncertainty": uncertainty,
        "agency_readiness": agency,
        "resource_pressure": resource_p,
        "execution_pressure": execution_p,
        "reasoning_pressure": reasoning_p,
        "reliability_pressure": reliability_p,
        "continuity_pressure": continuity_p,
        "introspection_pressure": 0.0,
        "social_pressure": 0.0,
        "policy_pressure": 0.0,
    }

    overall_intensity = weighted_overall_intensity(dimension_scores, policy)
    overall_condition = cast(
        Literal["quiet", "steady", "loaded", "strained", "unstable", "unknown"],
        condition_from_intensity(overall_intensity, policy.condition_thresholds),
    )

    dominant_targets = [t.target_id for t in attention.dominant_targets[:5]]
    evidence_density = clamp01(len(dominant_targets) / 5.0)
    overall_confidence = clamp01(0.5 + 0.5 * evidence_density)

    ranked_channels = [
        (ch, v)
        for ch, v in sorted(field_channels.items(), key=lambda kv: kv[1], reverse=True)
        if v >= policy.dominant_channel_threshold
    ]
    dominant_field_channels = {ch: v for ch, v in ranked_channels[:8]}

    pressure_channel_set = set(policy.pressure_channels)

    unresolved: list[str] = []
    stabilizing: list[str] = []
    for ch, v in merged_channels.items():
        if ch in policy.stabilizing_channels and v >= 0.3:
            stabilizing.append(f"{ch}={v:.2f}")
        dim = policy.channel_dimension_map.get(ch)
        if (
            dim
            and ch in pressure_channel_set
            and ch not in policy.stabilizing_channels
            and v >= policy.unresolved_pressure_threshold
        ):
            unresolved.append(f"{ch}→{dim}")

    unresolved = sorted(set(unresolved))
    stabilizing = sorted(set(stabilizing))

    dimensions: dict[str, SelfStateDimensionV1] = {}
    for dim_id in ALL_DIMENSION_IDS:
        score = dimension_scores.get(dim_id, 0.0)
        dimensions[dim_id] = SelfStateDimensionV1(
            dimension_id=dim_id,
            score=clamp01(score),
            confidence=overall_confidence,
            dominant_evidence=evidence_for_dimension(
                dim_id=dim_id,
                merged_channels=merged_channels,
                policy=policy,
            ),
            reasons=[f"{dim_id} from field+attention channel synthesis"],
        )

    summary_labels = _emit_summary_labels(
        dimension_scores=dimension_scores,
        overall_condition=overall_condition,
        overall_salience=attention.overall_salience,
    )

    if enable_transport_influence:
        hints = transport_channel_hints(field)
        integrity = transport_integrity_score(hints)
        dimension_scores["transport_integrity"] = integrity
        dimensions["transport_integrity"] = SelfStateDimensionV1(
            dimension_id="transport_integrity",
            score=integrity,
            confidence=overall_confidence,
            dominant_evidence=[
                f"bus_health={hints.get('bus_health', 0):.2f}",
                f"contract_pressure={hints.get('contract_pressure', 0):.2f}",
            ],
            reasons=["transport substrate synthesis from field capability:transport"],
        )
        summary_labels = sorted(
            set(summary_labels) | set(transport_summary_labels(hints, integrity))
        )

    return SelfStateV1(
        self_state_id=stable_self_state_id(
            source_field_tick_id=field.tick_id,
            source_attention_frame_id=attention.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_field_tick_id=field.tick_id,
        source_field_generated_at=field.generated_at,
        source_attention_frame_id=attention.frame_id,
        source_attention_generated_at=attention.generated_at,
        self_state_policy_id=policy.policy_id,
        overall_condition=overall_condition,
        overall_intensity=overall_intensity,
        overall_confidence=overall_confidence,
        dimensions=dimensions,
        dominant_attention_targets=dominant_targets,
        dominant_field_channels=dominant_field_channels,
        unresolved_pressures=unresolved,
        stabilizing_factors=stabilizing,
        warnings=warnings,
        summary_labels=summary_labels,
    )
