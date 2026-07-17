from datetime import datetime, timezone
from pathlib import Path

from orion.self_state.policy import SelfStatePolicyV1, load_self_state_policy
from orion.self_state.scoring import (
    agency_readiness_score,
    collect_field_channel_pressures,
    condition_from_intensity,
    coherence_score,
    map_channels_to_dimensions,
    uncertainty_score,
    weighted_overall_intensity,
)
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _field_high_execution() -> FieldStateV1:
    # Phase 1 (2026-07-12): raw node-level execution_load/execution_friction no
    # longer have a direct channel_dimension_map entry (double-counting fix,
    # see config/self_state/self_state_policy.v1.yaml). Their signal reaches
    # execution_pressure/reliability_pressure via the diffused capability
    # channel name instead (what orion-field-digester's diffusion.py actually
    # produces), so these fixtures set the post-diffusion capability channel
    # directly rather than the raw node channel.
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_scoring",
        node_vectors={"node:athena": {"execution_load": 1.0, "execution_friction": 0.0}},
        capability_vectors={"capability:orchestration": {"execution_pressure": 0.9}},
    )


def test_high_execution_load_raises_execution_pressure() -> None:
    channels, _ = collect_field_channel_pressures(_field_high_execution())
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("execution_pressure", 0.0) > 0.5


def test_raw_execution_load_alone_no_longer_double_counts() -> None:
    # Regression guard for the Phase 1 double-counting fix: raw node-level
    # execution_load with NO diffused capability channel present must not
    # map to execution_pressure at all anymore.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_scoring_raw_only",
        node_vectors={"node:athena": {"execution_load": 1.0}},
    )
    channels, _ = collect_field_channel_pressures(field)
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("execution_pressure", 0.0) == 0.0


def test_high_execution_friction_raises_reliability_pressure() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_friction",
        node_vectors={"node:athena": {"execution_friction": 1.0}},
        capability_vectors={"capability:orchestration": {"reliability_pressure": 0.9}},
    )
    channels, _ = collect_field_channel_pressures(field)
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("reliability_pressure", 0.0) > 0.5


def test_high_failure_pressure_raises_reliability() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_fail",
        node_vectors={"node:athena": {"failure_pressure": 1.0}},
        capability_vectors={"capability:orchestration": {"reliability_pressure": 0.9}},
    )
    channels, _ = collect_field_channel_pressures(field)
    dims = map_channels_to_dimensions(
        channel_pressures=channels,
        policy=POLICY,
    )
    assert dims.get("reliability_pressure", 0.0) > 0.5


def test_available_capacity_improves_coherence() -> None:
    low = coherence_score(
        channel_pressures={"cpu_pressure": 0.9},
        policy=POLICY,
    )
    high = coherence_score(
        channel_pressures={"available_capacity": 1.0, "confidence": 1.0},
        policy=POLICY,
    )
    assert high > low


def test_high_salience_low_coherence_raises_uncertainty() -> None:
    u = uncertainty_score(overall_salience=1.0, coherence=0.1)
    assert u > 0.5


def test_agency_readiness_falls_with_reliability_pressure() -> None:
    high_rel = agency_readiness_score(
        coherence=0.9,
        execution_pressure=0.2,
        reliability_pressure=0.9,
        uncertainty=0.1,
        resource_pressure=0.1,
    )
    low_rel = agency_readiness_score(
        coherence=0.9,
        execution_pressure=0.2,
        reliability_pressure=0.1,
        uncertainty=0.1,
        resource_pressure=0.1,
    )
    assert low_rel > high_rel


def test_double_counting_fix_weighted_diffusion_wins_not_raw_spike() -> None:
    # Phase 1 (2026-07-12) double-counting fix, exercised end-to-end through
    # collect_field_channel_pressures + map_channels_to_dimensions.
    #
    # node:circe --(weight=0.50)--> capability:llm_inference,
    # channel_map: gpu_pressure -> pressure
    # (config/field/orion_field_topology.v1.yaml).
    #
    # circe reports a raw gpu_pressure spike of 1.0. The correctly
    # edge-weighted diffusion of that spike into capability:llm_inference's
    # "pressure" channel is 1.0 * 0.50 = 0.50. Before the fix, raw
    # gpu_pressure=1.0 had its own direct channel_dimension_map entry to
    # resource_pressure and always won max(1.0, 0.50) == 1.0 — the topology
    # edge weight was functionally inert. After the fix, gpu_pressure has no
    # direct entry, so resource_pressure must reflect only the weighted
    # diffusion (0.50), not the raw unweighted spike (1.0).
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_double_counting",
        node_vectors={"node:circe": {"gpu_pressure": 1.0}},
        capability_vectors={"capability:llm_inference": {"pressure": 0.50}},
    )
    channels, provenance = collect_field_channel_pressures(field)
    assert channels["gpu_pressure"] == 1.0  # raw value still present in merged channels...
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    # ...but it must not compete with (and beat) its own weighted diffusion.
    assert dims.get("resource_pressure", 0.0) == 0.50
    # Phase 3 (2026-07-12): the raw node channel's provenance is the node
    # itself; the diffused capability channel falls back to the capability_id
    # since this synthetic fixture never ran real diffusion (no
    # field.capability_provenance populated) -- exercising the fallback path.
    assert provenance["gpu_pressure"] == "node:circe"
    assert provenance["pressure"] == "capability:llm_inference"


def test_node_only_channel_staleness_still_reaches_continuity_pressure() -> None:
    # Regression guard for the 6 protected node-only channels (no topology
    # diffusion edge exists for them at all): staleness must still reach
    # continuity_pressure directly via node_vectors, unaffected by the
    # double-counting fix.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_staleness",
        node_vectors={"node:athena": {"staleness": 0.8}},
    )
    channels, _ = collect_field_channel_pressures(field)
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("continuity_pressure", 0.0) == 0.8


def test_weighted_overall_intensity_skips_missing_dimension_not_zero() -> None:
    # A dimension with a policy weight but absent from dimension_scores this
    # tick (e.g. transport_integrity when the feature flag is off) must be
    # skipped entirely, not counted as a phantom 0.0 pressure that dilutes
    # the average toward "everything's fine".
    policy = SelfStatePolicyV1(dimension_weights={"a": 0.5, "b": 0.5})

    only_a = weighted_overall_intensity({"a": 1.0}, policy)
    # If "b" were phantom-zero-defaulted: (0.5*1.0 + 0.5*0.0) / 1.0 == 0.5.
    # Skipped correctly: average is over "a" alone == 1.0.
    assert only_a == 1.0

    both_present = weighted_overall_intensity({"a": 1.0, "b": 1.0}, policy)
    assert both_present == 1.0

    both_present_mixed = weighted_overall_intensity({"a": 1.0, "b": 0.0}, policy)
    assert both_present_mixed == 0.5


def test_confidence_merge_takes_worst_contributor_not_best() -> None:
    # Merge-polarity fix (2026-07-16): confidence is a HIGHER_IS_BETTER
    # channel, so the merge must take min() (worst contributor wins) across
    # capabilities, not max(). Regression guard for the live bug where
    # capability:transport's near-1.0 confidence permanently masked
    # capability:orchestration's real, varying confidence.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_confidence_polarity",
        capability_vectors={
            "capability:transport": {"confidence": 0.95},
            "capability:orchestration": {"confidence": 0.3},
        },
    )
    channels, provenance = collect_field_channel_pressures(field)
    assert channels["confidence"] == 0.3
    assert provenance["confidence"] == "capability:orchestration"


def test_available_capacity_merge_takes_worst_contributor_not_best() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_capacity_polarity",
        capability_vectors={
            "capability:transport": {"available_capacity": 0.95},
            "capability:orchestration": {"available_capacity": 0.3},
        },
    )
    channels, _ = collect_field_channel_pressures(field)
    assert channels["available_capacity"] == 0.3


def test_pressure_channel_merge_still_takes_worst_i_e_highest_contributor() -> None:
    # Regression guard: a genuine pressure-type channel (higher = worse) must
    # still merge via max() across sources -- unchanged behavior, so the
    # polarity branch above didn't accidentally flip the wrong set.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_pressure_polarity",
        node_vectors={
            "node:athena": {"cpu_pressure": 0.2},
            "node:atlas": {"cpu_pressure": 0.9},
        },
    )
    channels, provenance = collect_field_channel_pressures(field)
    assert channels["cpu_pressure"] == 0.9
    assert provenance["cpu_pressure"] == "node:atlas"


def test_expected_offline_suppression_still_merges_via_max_not_min() -> None:
    # expected_offline_suppression is deliberately NOT in
    # HIGHER_IS_BETTER_CHANNELS despite living in self_state_policy.v1.yaml's
    # stabilizing_channels block -- see HIGHER_IS_BETTER_CHANNELS docstring.
    # Whichever source reports an active suppression event (1.0) must win,
    # not be masked by another source's 0.0.
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_suppression_polarity",
        node_vectors={
            "node:athena": {"expected_offline_suppression": 0.0},
            "node:circe": {"expected_offline_suppression": 1.0},
        },
    )
    channels, _ = collect_field_channel_pressures(field)
    assert channels["expected_offline_suppression"] == 1.0


def test_condition_thresholds() -> None:
    t = POLICY.condition_thresholds
    assert condition_from_intensity(0.10, t) == "quiet"
    assert condition_from_intensity(0.30, t) == "steady"
    assert condition_from_intensity(0.55, t) == "loaded"
    assert condition_from_intensity(0.85, t) == "strained"
    assert condition_from_intensity(0.95, t) == "unstable"
