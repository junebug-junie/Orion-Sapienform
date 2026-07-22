"""Real (non-hand-tuned) FieldStateV1 channel merge + category routing.

2026-07-22, SelfStateV1 burn (docs/superpowers/specs/2026-07-22-self-state-phi-
endo-origination-burn-spec.md). `orion/self_state/scoring.py` mixed two different
things: a legitimate raw-signal merge (this file, moved verbatim except for
imports) and a set of hand-picked, uncalibrated weighted-combination formulas
(agency_readiness/coherence/uncertainty/field_intensity -- NOT moved here, no
principled non-hand-tuned replacement exists, not rebuilt as part of this burn).

`collect_field_channel_pressures()` is real: confirmed via the 2026-07-16
merge-polarity fix (HIGHER_IS_BETTER_CHANNELS needing min() instead of max() to
stop a healthy-but-irrelevant source from permanently masking a genuinely
degraded one, live-verified against a 69h corpus). `CHANNEL_DIMENSION_MAP` here
covers only the 7 categories that were ever actually read from self_state's old
`mapped` dict (execution_pressure, resource_pressure, reasoning_pressure,
reliability_pressure, continuity_pressure, introspection_pressure,
social_pressure) -- the old policy yaml's routes to "coherence"/"uncertainty"
are deliberately NOT reproduced here: traced and confirmed those were already
dead in the old code (`orion/self_state/builder.py` computed `coherence`/
`uncertainty` from separate hand-tuned formulas, never from the channel-mapped
value, so those routing entries produced values nothing ever read).
"""
from __future__ import annotations

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

# Channels where higher = better (capacity/quality-of-service semantics), the
# opposite of PRESSURE_CHANNELS. Preserved from the 2026-07-16 fix even though
# none of these currently route anywhere in CHANNEL_DIMENSION_MAP below (their
# old routes all went to the dead "coherence" category) -- the merge-polarity
# distinction itself is a real, independent fact about these channels'
# semantics, not something tied to self_state's now-removed compression.
HIGHER_IS_BETTER_CHANNELS = frozenset({
    "availability",
    "confidence",
    "available_capacity",
    "delivery_confidence",
    "bus_health",
})

# Only the 7 categories self_state's builder.py ever actually read from the
# channel-mapped dict (execution_pressure, resource_pressure, reasoning_pressure,
# reliability_pressure, continuity_pressure, introspection_pressure,
# social_pressure). No entries here route to "coherence" or "uncertainty" --
# those composite dimensions never read from this map even before the burn.
CHANNEL_DIMENSION_MAP: dict[str, str] = {
    "thermal_pressure": "resource_pressure",
    "staleness": "continuity_pressure",
    "pressure": "resource_pressure",
    "execution_pressure": "execution_pressure",
    "reasoning_pressure": "reasoning_pressure",
    "reliability_pressure": "reliability_pressure",
    "repair_pressure": "social_pressure",
    "conversation_load": "social_pressure",
    "egress_confidence_deficit": "introspection_pressure",
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def collect_field_channel_pressures(
    field: FieldStateV1,
) -> tuple[dict[str, float], dict[str, str]]:
    """Merge node_vectors + capability_vectors into one channel-name-keyed
    pressure dict, plus a parallel provenance dict recording which source_id
    "won" the merge for each channel this tick.

    Moved verbatim (imports aside) from orion/self_state/scoring.py's function
    of the same name -- this logic was never part of the hand-tuned-coefficient
    problem, it's a real merge mechanism with live-verified correctness.
    """
    out: dict[str, float] = {}
    provenance: dict[str, str] = {}
    for source_id, vector in field.node_vectors.items():
        for channel, value in vector.items():
            v = clamp01(float(value))
            if channel in HIGHER_IS_BETTER_CHANNELS:
                if v <= out.get(channel, 1.0):
                    out[channel] = v
                    provenance[channel] = source_id
            elif (channel in PRESSURE_CHANNELS or v > 0) and v >= out.get(channel, 0.0):
                out[channel] = v
                provenance[channel] = source_id
    for capability_id, vector in field.capability_vectors.items():
        for channel, value in vector.items():
            v = clamp01(float(value))
            resolved_provenance = field.capability_provenance.get(capability_id, {}).get(
                channel, capability_id
            )
            if channel in HIGHER_IS_BETTER_CHANNELS:
                if v <= out.get(channel, 1.0):
                    out[channel] = v
                    provenance[channel] = resolved_provenance
            elif (channel in PRESSURE_CHANNELS or v > 0) and v >= out.get(channel, 0.0):
                out[channel] = v
                provenance[channel] = resolved_provenance
    # recent_perturbation_count (context_channel, not scored into any
    # dimension): 2026-07-22 correction -- this block was accidentally
    # dropped from the "moved verbatim" copy of this function (confirmed via
    # a direct diff against orion/self_state/scoring.py's original), a real
    # regression since orion-field-digester's field_channel_corpus.v1 rows
    # and orion/mood_arc/fit_encoder.py's explicit by-name exclusion both
    # depend on this key being present. Restored verbatim, including the
    # original 2026-07-16 saturating-counter fix history: field.recent_
    # perturbations used to be capped to the last 20 distinct labels EVER
    # seen (saturating this to 1.0 within a few ticks and pinning it there
    # forever); orion-field-digester's apply_perturbations now prunes it to a
    # rolling 60s wall-clock window instead, so this count reflects genuinely
    # recent activity and decays back down once a burst passes.
    n = len(field.recent_perturbations)
    if n > 0:
        out["recent_perturbation_count"] = clamp01(min(1.0, n / 20.0))
    return out, provenance


def map_channels_to_dimensions(channel_pressures: dict[str, float]) -> dict[str, float]:
    """max()-merge raw channel pressures into the 7 real named categories."""
    dims: dict[str, float] = {}
    for channel, pressure in channel_pressures.items():
        dim_id = CHANNEL_DIMENSION_MAP.get(channel)
        if not dim_id:
            continue
        dims[dim_id] = max(dims.get(dim_id, 0.0), clamp01(pressure))
    return dims


def field_pressures(field: FieldStateV1) -> dict[str, float]:
    """One-shot: merge + map. The direct FieldStateV1-native replacement for
    what SelfStateV1.dimensions used to provide for these 7 categories.
    coherence/uncertainty/agency_readiness/field_intensity are NOT included --
    no principled non-hand-tuned formula for them exists; callers that need
    those must treat them as absent (0.0 via .get default), same graceful-
    degradation behavior orion/proposals/scoring.py already had for dimension
    IDs with no scorer (e.g. "contract_pressure", which was already always
    0.0 before this burn)."""
    channel_pressures, _provenance = collect_field_channel_pressures(field)
    return map_channels_to_dimensions(channel_pressures)
