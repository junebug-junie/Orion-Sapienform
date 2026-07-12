from pathlib import Path

from orion.self_state.policy import SelfStatePolicyV1, load_self_state_policy

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config" / "self_state" / "self_state_policy.v1.yaml"


def test_load_self_state_policy_v1() -> None:
    policy = load_self_state_policy(POLICY)
    assert isinstance(policy, SelfStatePolicyV1)
    assert policy.policy_id == "self_state_policy.v1"
    assert policy.channel_dimension_map["execution_pressure"] == "execution_pressure"
    assert policy.condition_thresholds.quiet_max == 0.15


def test_double_counted_channels_removed_from_channel_dimension_map() -> None:
    # Phase 1 (2026-07-12) double-counting fix regression guard: these 11
    # channels are diffused into a capability_vectors channel under a
    # different name by a topology edge, so a direct channel_dimension_map
    # entry would always win max() against its own correctly-weighted
    # diffusion. See config/self_state/self_state_policy.v1.yaml.
    policy = load_self_state_policy(POLICY)
    removed = {
        "transport_pressure",
        "bus_health",
        "delivery_confidence",
        "execution_load",
        "execution_friction",
        "failure_pressure",
        "reasoning_load",
        "cpu_pressure",
        "gpu_pressure",
        "memory_pressure",
        "disk_pressure",
    }
    for ch in removed:
        assert ch not in policy.channel_dimension_map, ch


def test_node_only_channels_still_reach_dimensions() -> None:
    # These 6 channels have no topology diffusion edge at all; node_vectors
    # is their only source, so they must keep their direct
    # channel_dimension_map entry (removing it would zero out real signal).
    policy = load_self_state_policy(POLICY)
    protected = {
        "thermal_pressure": "resource_pressure",
        "staleness": "continuity_pressure",
        "availability": "coherence",
        "expected_offline_suppression": "coherence",
        "repair_pressure": "social_pressure",
        "conversation_load": "social_pressure",
    }
    for ch, dim in protected.items():
        assert policy.channel_dimension_map.get(ch) == dim, ch


def test_policy_channel_role_lists() -> None:
    policy = load_self_state_policy(POLICY)
    assert "execution_load" not in policy.pressure_channels
    assert "repair_pressure" in policy.pressure_channels
    assert "availability" not in policy.pressure_channels
    assert "recent_perturbation_count" in policy.context_channels
    assert "availability" in policy.stabilizing_channels
