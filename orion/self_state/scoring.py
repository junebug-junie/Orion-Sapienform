from __future__ import annotations

from orion.self_state.policy import SelfStateConditionThresholdsV1, SelfStatePolicyV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1

PRESSURE_CHANNELS = frozenset({
    "execution_load",
    "execution_friction",
    "execution_pressure",
    "failure_pressure",
    "reasoning_load",
    "reasoning_pressure",
    "reliability_pressure",
    "cpu_pressure",
    "gpu_pressure",
    "memory_pressure",
    "disk_pressure",
    "thermal_pressure",
    "staleness",
    "pressure",
    "repair_pressure",
    "conversation_load",
    "egress_confidence_deficit",
    "prediction_error",
    "field_coherence_warning",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clamp(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, float(x)))


def condition_from_intensity(
    intensity: float,
    thresholds: SelfStateConditionThresholdsV1,
) -> str:
    x = clamp01(intensity)
    if x <= thresholds.quiet_max:
        return "quiet"
    if x <= thresholds.steady_max:
        return "steady"
    if x <= thresholds.loaded_max:
        return "loaded"
    if x <= thresholds.strained_max:
        return "strained"
    return "unstable"


def collect_field_channel_pressures(field: FieldStateV1) -> dict[str, float]:
    out: dict[str, float] = {}
    for vector in list(field.node_vectors.values()) + list(field.capability_vectors.values()):
        for channel, value in vector.items():
            if channel in PRESSURE_CHANNELS or float(value) > 0:
                out[channel] = max(out.get(channel, 0.0), clamp01(float(value)))
    n = len(field.recent_perturbations)
    if n > 0:
        out["recent_perturbation_count"] = clamp01(min(1.0, n / 20.0))
    return out


def collect_attention_channel_pressures(
    attention: FieldAttentionFrameV1,
    policy: SelfStatePolicyV1,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in attention.dominant_targets:
        kind_weight = float(policy.attention_target_weights.get(target.target_kind, 0.0))
        salience = clamp01(target.salience_score)
        for channel, contrib in target.dominant_channels.items():
            weighted = clamp01(float(contrib) * salience * kind_weight)
            out[channel] = max(out.get(channel, 0.0), weighted)
    out["overall_salience"] = clamp01(attention.overall_salience)
    return out


def map_channels_to_dimensions(
    *,
    channel_pressures: dict[str, float],
    policy: SelfStatePolicyV1,
) -> dict[str, float]:
    dims: dict[str, float] = {}
    for channel, pressure in channel_pressures.items():
        dim_id = policy.channel_dimension_map.get(channel)
        if not dim_id:
            continue
        dims[dim_id] = max(dims.get(dim_id, 0.0), clamp01(pressure))
    return dims


def coherence_score(
    *,
    channel_pressures: dict[str, float],
    policy: SelfStatePolicyV1,
) -> float:
    stabilizing = 0.0
    for channel, weight in policy.stabilizing_channels.items():
        stabilizing += clamp01(channel_pressures.get(channel, 0.0)) * float(weight)
    penalty = 0.0
    for ch in ("failure_pressure", "execution_friction", "staleness", "pressure"):
        penalty += clamp01(channel_pressures.get(ch, 0.0)) * 0.25
    return clamp01(stabilizing - penalty)


def uncertainty_score(*, overall_salience: float, coherence: float) -> float:
    return clamp01(clamp01(overall_salience) * (1.0 - clamp01(coherence)))


def agency_readiness_score(
    *,
    coherence: float,
    execution_pressure: float,
    reliability_pressure: float,
    uncertainty: float,
    resource_pressure: float,
) -> float:
    base = clamp01(coherence)
    base -= execution_pressure * 0.25
    base -= reliability_pressure * 0.35
    base -= uncertainty * 0.25
    base -= resource_pressure * 0.15
    return clamp01(base)


def field_intensity_score(
    *,
    overall_salience: float,
    recent_perturbation_saturation: float,
) -> float:
    return clamp01(0.6 * overall_salience + 0.4 * recent_perturbation_saturation)


def weighted_overall_intensity(
    dimension_scores: dict[str, float],
    policy: SelfStatePolicyV1,
) -> float:
    total_w = 0.0
    acc = 0.0
    for dim_id, weight in policy.dimension_weights.items():
        # A dimension absent from dimension_scores this tick (e.g.
        # transport_integrity when ENABLE_TRANSPORT_SELF_STATE_INFLUENCE is
        # off) is skipped entirely, not defaulted to 0.0 pressure — a phantom
        # zero would dilute the average with a fabricated "everything's
        # fine" signal for a dimension that was never actually computed.
        if dim_id not in dimension_scores:
            continue
        w = float(weight)
        if w <= 0:
            continue
        total_w += w
        acc += w * clamp01(dimension_scores[dim_id])
    if total_w <= 0:
        return 0.0
    return clamp01(acc / total_w)


def channels_mapped_to_dimension(
    dim_id: str,
    merged_channels: dict[str, float],
    channel_map: dict[str, str],
) -> list[tuple[str, float]]:
    """Channels whose channel_map entry routes to dim_id, as (channel, value) pairs.

    Shared by dimension score-confidence (channel_dimension_map only) and
    builder.py's evidence formatting (channel_dimension_map + the
    evidence-only channel_map) so both read the same filter instead of
    re-implementing it.
    """
    return [(ch, v) for ch, v in merged_channels.items() if channel_map.get(ch) == dim_id]


# Dimensions synthesized from other dimensions/attention salience rather than
# a direct channel_dimension_map routing (field_intensity: overall_salience +
# perturbation saturation; agency_readiness: a formula over other dimension
# scores). channel_dimension_confidence() can never find a contributing
# channel for these -- not a transient "no evidence this tick", a permanent
# structural fact -- so they fall back to the coarser overall_confidence
# proxy instead of a hardcoded, permanently-misleading 0.0.
COMPOSITE_DIMENSION_IDS = frozenset({"field_intensity", "agency_readiness"})


def channel_dimension_confidence(
    *,
    dim_id: str,
    merged_channels: dict[str, float],
    policy: SelfStatePolicyV1,
) -> float:
    """Real per-dimension confidence from this tick's contributing channels.

    There is no per-channel freshness data available in FieldStateV1 (only a
    whole-field generated_at) so confidence is based only on two honest
    signals: how many channels actually mapped to this dimension this tick
    (contributing-channel count) and how much they agree with each other
    (inverse of their spread). A dimension with zero contributing channels
    this tick reports 0.0 confidence rather than borrowing a global proxy --
    except COMPOSITE_DIMENSION_IDS, which never have a direct channel and are
    handled by the caller instead (see builder.py).
    """
    values = [
        v
        for _, v in channels_mapped_to_dimension(dim_id, merged_channels, policy.channel_dimension_map)
    ]
    n = len(values)
    if n == 0:
        return 0.0
    count_component = min(1.0, n / 3.0)
    # A single contributing channel has no other value to disagree with --
    # treat it as trivially self-agreeing (spread 0) rather than a second,
    # undocumented magic constant; count_component already discounts it
    # (n=1 -> 1/3) relative to multiple corroborating channels.
    agreement = 1.0 - clamp01(max(values) - min(values)) if n >= 2 else 1.0
    return clamp01(0.3 * count_component + 0.7 * agreement)
