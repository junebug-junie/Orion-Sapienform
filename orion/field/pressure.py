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

from dataclasses import dataclass

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
# 2026-08-11: `"thermal_pressure": "resource_pressure"` removed. Patch A of
# the arena rate-coupling design. NOTE: that spec lives in PR #1554
# (branch docs/proposal-base-priority-deletion) and is NOT merged as of this
# commit, so the path
# docs/superpowers/specs/2026-08-11-proposal-arena-rate-coupling-design.md
# does not resolve on main yet. Cited by PR number deliberately -- a code
# comment pointing at a file that does not exist is how a "tracked follow-up"
# becomes tracked nowhere. Every number this patch depends on is restated
# inline here and in orion/proposals/scoring.py, so neither comment requires
# the doc to be readable.
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
# RETRACTED JUSTIFICATION, kept visible on purpose. The first version of this
# comment also argued thermal_pressure was disqualified because it sits in
# services/orion-field-digester/app/digestion/decay.py's NODE_DECAY_CHANNELS,
# making a calm reading indistinguishable from the biometrics producer dying.
# Code review falsified that: it does not distinguish the removed input from
# the kept one. Capability `pressure` is produced by apply_diffusion from
# cpu_pressure/memory_pressure/gpu_pressure/disk_pressure/stream_backlog_
# pressure/prediction_error (config/field/orion_field_topology.v1.yaml), and
# every one of those is ALSO in NODE_DECAY_CHANNELS -- and `pressure` itself is
# in CAPABILITY_DECAY_CHANNELS. Removing thermal does not remove the ambiguity.
#
# So this patch is justified by the measured resolution recovery ALONE, which
# is real and is the only claim above this line. The decay ambiguity is a
# genuine, still-open defect for the whole dimension:
# services/orion-biometrics going quiet decays every remaining input toward 0,
# resource_pressure reads calm, and because config/feedback/feedback_policy
# .v1.yaml lists `resource_pressure: decrease` under positive_delta_channels,
# the in-flight action is credited with a positive outcome for a producer
# outage. That needs a liveness/staleness guard on the dimension, not a
# different choice of input channel. Tracked as follow-up in PR #1554's design
# doc; NOT fixed here.
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


@dataclass(frozen=True)
class DimensionContributor:
    """One channel that fed a dimension's max()-merge this tick."""

    channel: str
    value: float
    # The source_id that won this CHANNEL's own merge inside
    # collect_field_channel_pressures() -- a node_id, capability_id, or the
    # resolved capability_provenance entry. None when the caller had no
    # channel-provenance dict to thread through (map_channels_to_dimensions'
    # back-compat path), which is not the same as "unknown producer".
    source_id: str | None


@dataclass(frozen=True)
class DimensionProvenance:
    """Why a dimension reads what it reads this tick.

    R1 of docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md. The
    merge that produces a dimension is a max() across channels, and the
    channel merge feeding it is itself a max() across sources -- so a
    dimension's value is one source's number, and until now which source that
    was got discarded before any consumer saw it.

    That discard is not hypothetical harm. Measured 2026-08-13 over 20,000
    live ticks: `node:substrate.codebase` won the merged `prediction_error`
    channel on 54.2% of ticks while contributing 5 distinct values in 20,000
    (an effective constant of 0.3357, so the channel has a hard floor it can
    never read calm below), while `node:substrate.biometrics` (1,188 distinct)
    and `node:substrate.bus_synaptic` (341 distinct) won 0% and contributed
    nothing at all. Same defect class as the `thermal_pressure` (18 distinct)
    vs capability `pressure` (1,325 distinct) domination fixed on 2026-08-11,
    one layer down. Neither was findable without knowing who won.

    `contributors` is every channel that mapped into this dimension, not just
    the winner -- R3's commensurability detector needs the losers to tell a
    real max() from a walkover.
    """

    dimension_id: str
    value: float
    winning_channel: str
    winning_source_id: str | None
    contributors: tuple[DimensionContributor, ...]


def map_channels_to_dimensions_with_provenance(
    channel_pressures: dict[str, float],
    channel_provenance: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, DimensionProvenance]]:
    """max()-merge raw channel pressures into the 7 real named categories,
    keeping a record of which channel (and which source behind that channel)
    won each dimension.

    The single implementation. `map_channels_to_dimensions()` is a thin
    wrapper that drops the second element, so the two can never drift into
    disagreeing about a value -- the failure mode that a parallel
    "provenance-aware copy" of this function would have invited.

    Ties: `>=` on the running max, matching
    `collect_field_channel_pressures()`'s own `>=` convention, so the
    last-iterated channel at an equal value wins. Deterministic given
    Python's insertion-ordered dicts, and consistent between the two layers.
    """
    provenance = channel_provenance or {}
    dims: dict[str, float] = {}
    winners: dict[str, tuple[str, float]] = {}
    contributors: dict[str, list[DimensionContributor]] = {}

    for channel, pressure in channel_pressures.items():
        dim_id = CHANNEL_DIMENSION_MAP.get(channel)
        if not dim_id:
            continue
        value = clamp01(pressure)
        contributors.setdefault(dim_id, []).append(
            DimensionContributor(
                channel=channel,
                value=value,
                source_id=provenance.get(channel),
            )
        )
        if dim_id not in dims or value >= dims[dim_id]:
            winners[dim_id] = (channel, value)
        dims[dim_id] = max(dims.get(dim_id, 0.0), value)

    detail: dict[str, DimensionProvenance] = {}
    for dim_id, value in dims.items():
        winning_channel, _winning_value = winners[dim_id]
        detail[dim_id] = DimensionProvenance(
            dimension_id=dim_id,
            value=value,
            winning_channel=winning_channel,
            winning_source_id=provenance.get(winning_channel),
            contributors=tuple(contributors.get(dim_id, ())),
        )
    return dims, detail


def map_channels_to_dimensions(channel_pressures: dict[str, float]) -> dict[str, float]:
    """max()-merge raw channel pressures into the 7 real named categories."""
    dims, _detail = map_channels_to_dimensions_with_provenance(channel_pressures)
    return dims


def field_pressures_with_provenance(
    field: FieldStateV1,
) -> tuple[dict[str, float], dict[str, DimensionProvenance]]:
    """`field_pressures()`, plus the record of who produced each dimension.

    Same values as `field_pressures()` by construction -- that function calls
    this one. Additive: no existing caller changes behavior, and nothing here
    touches the merge itself. Changing how the merge picks a winner is a
    cognition-loop change (it moves proposal ranking and feedback credit) and
    goes through proposal mode, not through this patch.
    """
    channel_pressures, channel_provenance = collect_field_channel_pressures(field)
    return map_channels_to_dimensions_with_provenance(
        channel_pressures, channel_provenance
    )


def field_pressures(field: FieldStateV1) -> dict[str, float]:
    """One-shot: merge + map. The direct FieldStateV1-native replacement for
    what SelfStateV1.dimensions used to provide for these 7 categories.
    coherence/uncertainty/agency_readiness/field_intensity are NOT included --
    no principled non-hand-tuned formula for them exists; callers that need
    those must treat them as absent (0.0 via .get default), same graceful-
    degradation behavior orion/proposals/scoring.py already had for dimension
    IDs with no scorer (e.g. "contract_pressure", which was already always
    0.0 before this burn)."""
    dims, _detail = field_pressures_with_provenance(field)
    return dims
