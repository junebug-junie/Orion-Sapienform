from pathlib import Path

from orion.self_state.policy import SelfStatePolicyV1, load_self_state_policy

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config" / "self_state" / "self_state_policy.v1.yaml"


def test_load_self_state_policy_v1() -> None:
    policy = load_self_state_policy(POLICY)
    assert isinstance(policy, SelfStatePolicyV1)
    assert policy.policy_id == "self_state_policy.v1"
    assert policy.channel_dimension_map["execution_load"] == "execution_pressure"
    assert policy.condition_thresholds.quiet_max == 0.15
