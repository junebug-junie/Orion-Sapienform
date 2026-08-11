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
    "cortex_exec_step_load",
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
    "stream_backlog_health",
})

# Only the 7 categories self_state's builder.py ever actually read from the
# channel-mapped dict (execution_pressure, resource_pressure, reasoning_pressure,
# reliability_pressure, continuity_pressure, introspection_pressure,
# social_pressure). No entries here route to "coherence" or "uncertainty" --
# those composite dimensions never read from this map even before the burn.
# 2026-08-11: `"thermal_pressure": "resource_pressure"` removed. See
# docs/superpowers/specs/2026-08-11-proposal-arena-rate-coupling-design.md
# (Patch A).
#
# `resource_pressure` is max(thermal_pressure, pressure) via the merge below.
# Measured over 28,735 real ticks (24h, substrate_field_state), thermal WON
# that max on 91.76% of ticks -- a 39-distinct-value quantized reading of one
# CPU's hottest core ((T-50)/(85-50), orion/telemetry/biometrics_pipeline.py
# ::normalize_thermal) overwriting a 1,895-distinct-value composite of five
# independent live capability channels, nine ticks out of ten. Removing it
# takes resource_pressure from 128 distinct values to 1,895: a 15x resolution
# recovery, same units, same direction, same downstream interpretation.
#
# Not merely low-resolution -- structurally wrong as a gate input.
# thermal_pressure is in services/orion-field-digester/app/digestion/decay.py's
# NODE_DECAY_CHANNELS, so a low reading is produced by the biometrics producer
# going quiet exactly as readily as by the hardware being cool. A dimension
# whose calm state is indistinguishable from its producer dying cannot gate
# anything (CLAUDE.md 0A names this failure by example: node:substrate.route's
# decayed-to-zero prediction_error).
#
# Killed here, not zeroed or left behind a flag, per CLAUDE.md 0A's "kill
# means kill, no fallback to the thing being killed". The raw
# `thermal_pressure` CHANNEL is deliberately untouched and still live
# everywhere it is read directly -- the `strain` composite
# (orion/telemetry/biometrics_pipeline.py:180), grammar emit/extract,
# services/orion-field-digester's state_deltas/decay/tensor channels, and the
# field channel corpus. This removes one routing entry, not a signal.
CHANNEL_DIMENSION_MAP: dict[str, str] = {
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


# Shared by this module's recent_perturbation_count channel and
# orion/attention/field_attention/selectors.py's select_system_targets --
# both score the same field.recent_perturbation_zscore (see field_state.py's
# recent_perturbation_ewma* docstring for why this replaced a fixed count
# cap). Reuses the z>=3.0 "anomalous" convention already live in
# services/orion-hub/scripts/bus_synaptic_graph_routes.py's anomalies()
# route and orion/substrate/prediction_error.py's
# _BUS_SYNAPTIC_ZSCORE_SATURATION, rather than inventing a new calibration.
RECENT_PERTURBATION_ZSCORE_SATURATION = 3.0

# Cold-start reliability floor for field.recent_perturbation_zscore, same
# concept as services/orion-bus-mirror/README.md's documented "count < ~5 is
# unreliable" floor for gap_zscore (orion/substrate/prediction_error.py's
# bus_synaptic_prediction_error() puts that same filtering responsibility on
# its caller, not the aggregation function -- same split here). Confirmed by
# hand: orion.bus.ewma.compute_ewma_update's variance estimate is built from
# a single prior sample after the first update, so a tiny early delta over a
# near-zero variance floor produces a wildly inflated z (simulated
# 2026-07-28: z=1000 on the *second* observation for a steady 1-unit/tick
# ramp). Below this many samples, treat the zscore as not yet trustworthy --
# absent/no-signal, not a fake reading.
RECENT_PERTURBATION_EWMA_MIN_SAMPLES = 5


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
    # depend on this key being present.
    #
    # 2026-07-28 correction to that restoration: the "n / 20.0" cap it
    # restored was itself live-confirmed saturated. field.recent_perturbations
    # is correctly pruned to a rolling 60s window (2026-07-16 fix, still
    # correct -- verified live 2026-07-28 that nothing older than ~59s
    # survives), but real steady-state mesh traffic fills that window to
    # ~100-118 distinct labels, 5-10x past the old /20.0 cap -- so this
    # channel read pinned at 1.0 under completely normal operation, not just
    # during bursts. Now scores deviation from this mesh's own recent
    # baseline instead of an absolute count (field.recent_perturbation_zscore,
    # maintained by apply_perturbations() -- see field_state.py's
    # recent_perturbation_ewma* docstring). zscore is None on the first tick
    # after a cold start (no baseline yet) or after upgrade from a
    # persisted-but-older FieldStateV1 -- leaving the key absent that tick is
    # correct ("no empty-shell cognition": absent beats a fake 0.0), it
    # populates from the next tick onward. Also gated on
    # RECENT_PERTURBATION_EWMA_MIN_SAMPLES -- see that constant's docstring
    # for why an early zscore isn't trustworthy yet either.
    n = len(field.recent_perturbations)
    if (
        n > 0
        and field.recent_perturbation_zscore is not None
        and field.recent_perturbation_ewma_n >= RECENT_PERTURBATION_EWMA_MIN_SAMPLES
    ):
        out["recent_perturbation_count"] = clamp01(
            max(0.0, field.recent_perturbation_zscore) / RECENT_PERTURBATION_ZSCORE_SATURATION
        )
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
