from pathlib import Path

from orion.attention.field_attention.policy import FieldAttentionPolicyV1, load_attention_policy

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config" / "attention" / "field_attention_policy.v1.yaml"


def test_load_attention_policy_v1() -> None:
    policy = load_attention_policy(POLICY)
    assert isinstance(policy, FieldAttentionPolicyV1)
    assert policy.policy_id == "field_attention_policy.v1"
    assert policy.node_channel_weights["execution_load"] == 0.70
    assert policy.limits.max_node_targets == 5
