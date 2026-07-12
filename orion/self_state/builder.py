from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, cast

from orion.self_state.policy import SelfStatePolicyV1
from orion.self_state.transport import (
    transport_channel_hints,
    transport_integrity_score,
    transport_summary_labels,
)
from orion.self_state.scoring import (
    agency_readiness_score,
    channel_dimension_confidence,
    channels_mapped_to_dimension,
    clamp,
    clamp01,
    collect_attention_channel_pressures,
    collect_field_channel_pressures,
    condition_from_intensity,
    coherence_score,
    COMPOSITE_DIMENSION_IDS,
    field_intensity_score,
    map_channels_to_dimensions,
    uncertainty_score,
    weighted_overall_intensity,
)
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import (
    AttentionTargetSummaryV1,
    SelfStateDimensionV1,
    SelfStateV1,
)

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
)


AttentionSchemaType = Literal[
    "focused_single",
    "distributed",
    "open_loop",
    "none",
    "unknown",
]


def derive_attention_schema(
    projection: AttentionBroadcastProjectionV1 | None,
) -> tuple[AttentionSchemaType | None, int, int]:
    """Derive (attention_schema_type, attention_dwell_ticks, attention_node_count).

    Pure and total: an absent projection yields the schema defaults (None, 0, 0);
    the function never raises on well-typed input.
    """
    if projection is None:
        return None, 0, 0
    node_count = len(projection.attended_node_ids)
    if node_count == 1:
        schema_type: AttentionSchemaType = "focused_single"
    elif node_count >= 2:
        schema_type = "distributed"
    elif projection.frame.open_loops:
        schema_type = "open_loop"
    else:
        schema_type = "none"
    dwell_ticks = max(0, int(projection.dwell_ticks))
    return schema_type, dwell_ticks, node_count


def normalize_hub_presence(hub_presence: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pass a non-empty presence dict through as-is; anything else becomes None."""
    if isinstance(hub_presence, dict) and hub_presence:
        return hub_presence
    return None


def stable_self_state_id(
    *,
    source_field_tick_id: str,
    source_attention_frame_id: str,
    policy_id: str,
) -> str:
    return f"self.state:{source_field_tick_id}:{source_attention_frame_id}:{policy_id}"


def _attention_target_summary(target: FieldAttentionTargetV1) -> AttentionTargetSummaryV1:
    """Structured summary of one FieldAttentionTargetV1 (2026-07-12, inner-state
    unification Phase 1). dominant_channel/reason take the top-1 entry --
    dominant_channels is already sorted descending by contribution
    (orion/attention/field_attention/scoring.py's weighted_pressure), and
    reasons is already ordered the same way (_reasons_from_dominant), so
    the first entry of each is the single most significant one this tick.
    """
    dominant_channel = next(iter(target.dominant_channels), None)
    reason = target.reasons[0] if target.reasons else None
    return AttentionTargetSummaryV1(
        target_id=target.target_id,
        target_kind=target.target_kind,
        pressure_score=target.pressure_score,
        dominant_channel=dominant_channel,
        reason=reason,
    )


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
    if dimension_scores.get("social_pressure", 0.0) >= 0.5:
        labels.append("social_pressurized")
    if dimension_scores.get("introspection_pressure", 0.0) >= 0.5:
        labels.append("introspection_loaded")
    return sorted(set(labels))


def _provenance_label(source_id: str) -> str:
    """Friendlier evidence label: strip the "node:" prefix (e.g. "node:atlas"
    -> "atlas"); leave capability-sourced provenance (e.g.
    "capability:transport") as-is since it's still informative and isn't a
    node name to shorten."""
    if source_id.startswith("node:"):
        return source_id[len("node:") :]
    return source_id


def _reason_for_evidence(ev: str, channel_provenance: dict[str, str]) -> str:
    """Phase 3 (2026-07-12): fold node/capability provenance into the
    free-text `reasons` field, NOT `dominant_evidence` -- `dominant_evidence`
    strings are parsed by downstream consumers (e.g. orion-spark-
    introspector's `_parse_dominant_evidence_channels`, which does
    `entry.partition("=")` expecting an exact `channel_name=value` shape) and
    prefixing the channel name there would silently break that parsing.
    `reasons` has no such contract.
    """
    channel_name = ev.partition("=")[0]
    node = channel_provenance.get(channel_name)
    if node:
        return f"driven by {ev} (node: {_provenance_label(node)})"
    return f"driven by {ev}"


def evidence_for_dimension(
    *,
    dim_id: str,
    merged_channels: dict[str, float],
    policy: SelfStatePolicyV1,
    limit: int = 3,
) -> list[str]:
    # Union of score-contributing channels (channel_dimension_map) and
    # evidence-only channels (evidence_channel_map, 2026-07-12): a raw
    # hardware/load channel that's also diffused into a capability channel
    # under a different name no longer feeds the score directly (fixed
    # double-counting bug), but downstream consumers (e.g. orion-spark-
    # introspector's stuck-generic-channel bypass) still need to see it in
    # dominant_evidence.
    pairs = channels_mapped_to_dimension(dim_id, merged_channels, policy.channel_dimension_map)
    pairs = pairs + channels_mapped_to_dimension(dim_id, merged_channels, policy.evidence_channel_map)
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
    attention_broadcast: AttentionBroadcastProjectionV1 | None = None,
    hub_presence: dict[str, Any] | None = None,
) -> SelfStateV1:
    generated_at = now or datetime.now(timezone.utc)

    warnings: list[str] = []
    if attention.source_field_tick_id != field.tick_id:
        warnings.append(
            f"attention_source_tick_mismatch:{attention.source_field_tick_id}!={field.tick_id}"
        )

    field_channels, channel_provenance = collect_field_channel_pressures(field)
    attn_channels = collect_attention_channel_pressures(attention, policy)
    merged_channels: dict[str, float] = dict(field_channels)
    for k, v in attn_channels.items():
        if v > merged_channels.get(k, 0.0):
            # Attention channels have no known node/capability provenance --
            # if an attention value overrides the field value for this
            # channel, drop the (now-stale) field-sourced provenance rather
            # than mislabel an attention-derived value with a node it didn't
            # come from.
            channel_provenance.pop(k, None)
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
        "introspection_pressure": mapped.get("introspection_pressure", 0.0),
        "social_pressure": mapped.get("social_pressure", 0.0),
    }

    # Compute transport_integrity BEFORE overall_intensity so that, when the
    # flag is on, its policy weight (config/self_state/self_state_policy.v1.yaml
    # dimension_weights.transport_integrity) actually feeds the weighted
    # average instead of being folded into dimension_scores too late to
    # matter. The SelfStateDimensionV1 object itself is still built later,
    # after `dimensions`/`summary_labels` exist, reusing `transport_hints`/
    # `transport_integrity_value` computed here.
    transport_hints: dict[str, float] = {}
    transport_integrity_value: float = 0.0
    if enable_transport_influence:
        transport_hints = transport_channel_hints(field)
        transport_integrity_value = transport_integrity_score(transport_hints)
        dimension_scores["transport_integrity"] = transport_integrity_value

    overall_intensity = weighted_overall_intensity(dimension_scores, policy)
    overall_condition = cast(
        Literal["quiet", "steady", "loaded", "strained", "unstable", "unknown"],
        condition_from_intensity(overall_intensity, policy.condition_thresholds),
    )

    top_attention_targets = attention.dominant_targets[:5]
    dominant_targets = [t.target_id for t in top_attention_targets]
    dominant_attention_target_details = [
        _attention_target_summary(t) for t in top_attention_targets
    ]
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
        dominant_evidence = evidence_for_dimension(
            dim_id=dim_id,
            merged_channels=merged_channels,
            policy=policy,
        )
        reasons = (
            [_reason_for_evidence(ev, channel_provenance) for ev in dominant_evidence]
            if dominant_evidence
            else ["no contributing channel evidence this tick"]
        )
        confidence = (
            overall_confidence
            if dim_id in COMPOSITE_DIMENSION_IDS
            else channel_dimension_confidence(
                dim_id=dim_id,
                merged_channels=merged_channels,
                policy=policy,
            )
        )
        dimensions[dim_id] = SelfStateDimensionV1(
            dimension_id=dim_id,
            score=clamp01(score),
            confidence=confidence,
            dominant_evidence=dominant_evidence,
            reasons=reasons,
        )

    summary_labels = _emit_summary_labels(
        dimension_scores=dimension_scores,
        overall_condition=overall_condition,
        overall_salience=attention.overall_salience,
    )

    if enable_transport_influence:
        # transport_hints/transport_integrity_value were computed earlier
        # (before overall_intensity) so the dimension's score already
        # contributed to the weighted average via dimension_scores; here we
        # only build the display/evidence object. confidence is
        # overall_confidence, same as COMPOSITE_DIMENSION_IDS above:
        # transport_integrity is synthesized from transport hints, not a
        # direct channel_dimension_map entry, so channel_dimension_confidence
        # would always report 0.0 for it too.
        dimensions["transport_integrity"] = SelfStateDimensionV1(
            dimension_id="transport_integrity",
            score=transport_integrity_value,
            confidence=overall_confidence,
            dominant_evidence=[
                f"bus_health={transport_hints.get('bus_health', 0):.2f}",
                f"contract_pressure={transport_hints.get('contract_pressure', 0):.2f}",
            ],
            reasons=["transport substrate synthesis from field capability:transport"],
        )
        summary_labels = sorted(
            set(summary_labels)
            | set(transport_summary_labels(transport_hints, transport_integrity_value))
        )

    dimension_trajectory: dict[str, float] = {}
    trajectory_condition: Literal["improving", "degrading", "stable", "unknown"] = "unknown"
    if previous_self_state is not None:
        weighted_delta = 0.0
        total_w = 0.0
        for dim_id, score in dimension_scores.items():
            prev_dim = previous_self_state.dimensions.get(dim_id)
            if prev_dim is None:
                continue
            delta = clamp(-1.0, 1.0, score - prev_dim.score)
            if abs(delta) >= 0.02:
                dimension_trajectory[dim_id] = round(delta, 4)
            w = float(policy.dimension_weights.get(dim_id, 0.0))
            weighted_delta += delta * w
            total_w += w
        if total_w > 0:
            net = weighted_delta / total_w
            if net > policy.trajectory_threshold:
                trajectory_condition = "improving"
            elif net < -policy.trajectory_threshold:
                trajectory_condition = "degrading"
            else:
                trajectory_condition = "stable"

    (
        attention_schema_type,
        attention_dwell_ticks,
        attention_node_count,
    ) = derive_attention_schema(attention_broadcast)

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
        dominant_attention_target_details=dominant_attention_target_details,
        dominant_field_channels=dominant_field_channels,
        unresolved_pressures=unresolved,
        stabilizing_factors=stabilizing,
        warnings=warnings,
        summary_labels=summary_labels,
        dimension_trajectory=dimension_trajectory,
        trajectory_condition=trajectory_condition,
        attention_schema_type=attention_schema_type,
        attention_dwell_ticks=attention_dwell_ticks,
        attention_node_count=attention_node_count,
        hub_presence=normalize_hub_presence(hub_presence),
    )
